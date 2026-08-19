"""Cross-fitted confidence-conditioned abstention for v1-R² P3.

The P2 anchor is fixed at prefix six with one possible cold insertion.  A gate
uses only inference-visible resolver/route/rank signals to decide whether to
insert or abstain.  Five-fold out-of-fold predictions provide exploratory
validation evidence without reusing the old calibration/audit split as if it
were a pristine holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from anchored_interleaving import anchored_interleave
from route_admission import read_prediction_records
from route_resolve import (
    ResidualUserProjector,
    atomic_json,
    average_metrics,
    build_validation_examples,
    ranking_metrics,
    read_key_value_lines,
    read_sequences,
    read_set,
    semantic_route,
    sha256_file,
    unique_in_order,
)


FEATURE_NAMES = (
    "proposed_cold_cosine",
    "cold_top1_top2_margin",
    "cold_vs_overall_top_gap",
    "proposed_cold_resolver_rr",
    "proposed_cold_depth1_route_rr",
    "proposed_cold_gram_rr",
)
COVERAGE_GRID = tuple(x / 10 for x in range(2, 11))
PROTECTED_PREFIX = 6
COLD_QUOTA = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--item-id-file", required=True)
    p.add_argument("--item-embeddings", required=True)
    p.add_argument("--resolver-checkpoint", required=True)
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--gate-epochs", type=int, default=300)
    p.add_argument("--gate-lr", type=float, default=0.05)
    p.add_argument("--gate-l2", type=float, default=1e-2)
    p.add_argument("--train-warm-retention", type=float, default=0.985)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--recency-decay", type=float, default=0.85)
    return p.parse_args()


def stable_fold(user_id: str, folds: int = 5) -> int:
    value = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest()[:8], "big")
    return value % folds


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def auc_roc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties."""
    labels = labels.to(torch.int64).cpu()
    scores = scores.to(torch.float64).cpu()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # The frozen gram-repro PyTorch predates the ``stable`` keyword.  Tie
    # groups receive average ranks below, so their internal order is irrelevant.
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(len(scores), dtype=torch.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[labels.bool()].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def extract_inference_features(
    row: dict,
    projected_user: torch.Tensor,
    item_embeddings: torch.Tensor,
    item_to_idx: dict[str, int],
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
) -> tuple[str, list[float]]:
    """Return proposed candidate and target-free, inference-visible features."""
    gram = unique_in_order(row["v0_top50"])
    resolver = unique_in_order(row["resolver_top50"])
    protected = set(gram[:PROTECTED_PREFIX])
    eligible_cold = [x for x in resolver if x in cold_items and x not in protected]
    if len(eligible_cold) < 2:
        raise ValueError(f"Need two eligible cold candidates for {row['user_id']}")
    candidate, second = eligible_cold[:2]
    _ranking, inserted = anchored_interleave(
        gram, resolver, cold_items, PROTECTED_PREFIX, COLD_QUOTA
    )
    if inserted != [candidate]:
        raise RuntimeError(f"P2/P3 proposed-candidate mismatch for {row['user_id']}")

    indices = torch.tensor(
        [item_to_idx[candidate], item_to_idx[second], item_to_idx[resolver[0]]],
        device=projected_user.device,
    )
    candidate_score, second_score, overall_score = (
        projected_user @ item_embeddings[indices].T
    ).tolist()
    resolver_rank = resolver.index(candidate) + 1
    gram_rank = gram.index(candidate) + 1 if candidate in gram else None
    route_order = unique_in_order(item_routes[item] for item in gram)
    route = item_routes[candidate]
    route_rank = route_order.index(route) + 1 if route in route_order else None
    return candidate, [
        candidate_score,
        candidate_score - second_score,
        candidate_score - overall_score,
        reciprocal_rank(resolver_rank),
        reciprocal_rank(route_rank),
        reciprocal_rank(gram_rank),
    ]


def fit_logistic_gate(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> dict:
    mean = features.mean(0)
    std = features.std(0, unbiased=False).clamp_min(1e-6)
    x = (features - mean) / std
    y = labels.float()
    n_pos = float(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("Gate training fold must contain both correctness classes")
    torch.manual_seed(seed)
    weight = torch.nn.Parameter(torch.zeros(features.shape[1]))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    pos_weight = torch.tensor((len(y) - n_pos) / n_pos)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight + bias
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        penalty = l2 * weight.square().sum()
        loss = bce + penalty
        loss.backward()
        optimizer.step()
        if epoch in (1, epochs) or epoch % 50 == 0:
            history.append({"epoch": epoch, "loss": float(loss.detach())})
    return {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "mean": mean,
        "std": std,
        "history": history,
        "n_positive": int(n_pos),
        "n_total": len(labels),
    }


def predict_gate(model: dict, features: torch.Tensor) -> torch.Tensor:
    x = (features - model["mean"]) / model["std"]
    return torch.sigmoid(x @ model["weight"] + model["bias"])


def evaluate_rows(rows: list[dict], admit: list[bool], cold_items: set[str]) -> dict:
    metrics = {name: [] for name in ("all", "warm", "cold")}
    for row, use_slot in zip(rows, admit):
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        gram = unique_in_order(row["v0_top50"])
        resolver = unique_in_order(row["resolver_top50"])
        ranking = (
            anchored_interleave(gram, resolver, cold_items, PROTECTED_PREFIX, COLD_QUOTA)[0]
            if use_slot else gram
        )
        result = ranking_metrics(ranking, target)
        metrics["all"].append(result)
        metrics[slice_name].append(result)
    return {name: average_metrics(values) for name, values in metrics.items()}


def select_threshold(
    rows: list[dict],
    probabilities: torch.Tensor,
    cold_items: set[str],
    warm_retention: float,
) -> tuple[dict | None, list[dict]]:
    baseline = evaluate_rows(rows, [False] * len(rows), cold_items)
    candidates: list[dict] = []
    for target_coverage in COVERAGE_GRID:
        k = max(1, math.ceil(target_coverage * len(rows)))
        threshold = float(torch.topk(probabilities, k=k).values[-1])
        admit = [bool(x >= threshold) for x in probabilities.tolist()]
        metrics = evaluate_rows(rows, admit, cold_items)
        actual_coverage = sum(admit) / len(admit)
        feasible = (
            metrics["warm"]["ndcg@10"] >= warm_retention * baseline["warm"]["ndcg@10"]
            and metrics["cold"]["ndcg@10"] > baseline["cold"]["ndcg@10"]
            and metrics["all"]["ndcg@10"] > baseline["all"]["ndcg@10"]
        )
        candidates.append({
            "target_coverage": target_coverage,
            "actual_coverage": actual_coverage,
            "threshold": threshold,
            "feasible": feasible,
            "metrics": metrics,
        })
    feasible_rows = [row for row in candidates if row["feasible"]]
    if not feasible_rows:
        return None, candidates
    selected = max(
        feasible_rows,
        key=lambda row: (
            row["metrics"]["cold"]["ndcg@10"],
            row["metrics"]["all"]["ndcg@10"],
            row["metrics"]["warm"]["ndcg@10"],
            -row["actual_coverage"],
        ),
    )
    return selected, candidates


def summarize_models(metric_rows: dict[str, dict[str, list[dict[str, float]]]]) -> dict:
    return {
        model: {name: average_metrics(values) for name, values in slices.items()}
        for model, slices in metric_rows.items()
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    p0_path = Path(args.p0_predictions).resolve()
    item_id_path = Path(args.item_id_file).resolve()
    embedding_path = Path(args.item_embeddings).resolve()
    resolver_path = Path(args.resolver_checkpoint).resolve()
    cold_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [p.name for p in output_dir.iterdir() if p.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing P3 scientific artifacts: {unexpected}")
    inputs = [
        dataset_dir / "user_sequence.txt", p0_path, item_id_path,
        embedding_path, resolver_path, cold_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing a test input in P3: {path}")

    records = read_prediction_records(p0_path)
    item_to_lexical = read_key_value_lines(item_id_path)
    catalog = set(item_to_lexical)
    cold_items = read_set(cold_path)
    item_routes = {item: semantic_route(lexical, 1) for item, lexical in item_to_lexical.items()}

    embedding_payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    if set(item_ids) != catalog or not cold_items <= catalog:
        raise ValueError("Catalog, embedding, or cold-state mismatch")
    checkpoint = torch.load(resolver_path, map_location="cpu")
    model = ResidualUserProjector(
        checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    embeddings_device = embeddings_cpu.to(device)

    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    validation = build_validation_examples(
        sequences, item_to_idx, embeddings_cpu, args.max_history, args.recency_decay
    )
    record_by_uid = {str(row["user_id"]): row for row in records}
    ordered_uids = [uid for uid, _items in sequences if uid in record_by_uid]
    if len(ordered_uids) != len(records):
        raise ValueError("P0 records do not match validation users")

    config_payload = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "p0_predictions": str(p0_path),
        "item_id_file": str(item_id_path),
        "item_embeddings": str(embedding_path),
        "resolver_checkpoint": str(resolver_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P3_CONFIDENCE_ABSTENTION",
        "feature_names": FEATURE_NAMES,
        "coverage_grid": COVERAGE_GRID,
        "protected_prefix": PROTECTED_PREFIX,
        "cold_quota": COLD_QUOTA,
        "fold_rule": "uint64_be(sha256(user_id)[:8]) mod 5",
        "evaluation_status": "exploratory_cross_fitted_not_pristine_holdout",
        "test_predictions_opened": False,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "n_records": len(records),
    }
    atomic_json(output_dir / "config.json", config_payload)

    feature_rows: list[dict] = []
    all_features: list[list[float]] = []
    labels: list[int] = []
    with torch.no_grad():
        for start in range(0, len(ordered_uids), 256):
            batch_uids = ordered_uids[start:start + 256]
            histories = torch.stack([validation[uid][0] for uid in batch_uids]).to(device)
            projected = model(histories)
            for index, uid in enumerate(batch_uids):
                row = record_by_uid[uid]
                target = str(row["target"])
                if target not in catalog or bool(row["is_cold"]) != (target in cold_items):
                    raise ValueError(f"Target/cold-state mismatch for {uid}")
                candidate, features = extract_inference_features(
                    row, projected[index], embeddings_device,
                    item_to_idx, item_routes, cold_items,
                )
                label = int(candidate == target)
                all_features.append(features)
                labels.append(label)
                feature_rows.append({
                    "user_id": uid,
                    "fold": stable_fold(uid, args.folds),
                    "proposed_cold_item": candidate,
                    "candidate_is_target": bool(label),
                    "features": dict(zip(FEATURE_NAMES, features)),
                })
            print(f"[features] {min(start + 256, len(ordered_uids))}/{len(ordered_uids)}", flush=True)
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    folds_tensor = torch.tensor(
        [stable_fold(str(row["user_id"]), args.folds) for row in records], dtype=torch.int64
    )
    # Feature extraction followed sequence order; align every downstream object to it.
    ordered_records = [record_by_uid[uid] for uid in ordered_uids]
    folds_tensor = torch.tensor([stable_fold(uid, args.folds) for uid in ordered_uids])

    oof_probabilities = torch.empty(len(ordered_records))
    oof_admit = [False] * len(ordered_records)
    fold_reports: list[dict] = []
    saved_fold_models: list[dict] = []
    all_folds_feasible = True
    for fold in range(args.folds):
        held_mask = folds_tensor == fold
        train_mask = ~held_mask
        train_indices = torch.where(train_mask)[0]
        held_indices = torch.where(held_mask)[0]
        gate = fit_logistic_gate(
            features_tensor[train_indices], labels_tensor[train_indices],
            args.gate_epochs, args.gate_lr, args.gate_l2, args.seed + fold,
        )
        train_prob = predict_gate(gate, features_tensor[train_indices])
        held_prob = predict_gate(gate, features_tensor[held_indices])
        train_rows = [ordered_records[i] for i in train_indices.tolist()]
        selected, grid = select_threshold(
            train_rows, train_prob, cold_items, args.train_warm_retention
        )
        feasible = selected is not None
        all_folds_feasible &= feasible
        threshold = float("inf") if selected is None else selected["threshold"]
        held_choices = held_prob >= threshold
        for idx, prob, choice in zip(held_indices.tolist(), held_prob.tolist(), held_choices.tolist()):
            oof_probabilities[idx] = prob
            oof_admit[idx] = bool(choice)
        fold_reports.append({
            "fold": fold,
            "n_train": len(train_indices),
            "n_held": len(held_indices),
            "n_positive_train": gate["n_positive"],
            "selected": selected,
            "threshold_grid": grid,
            "held_admission_coverage": float(held_choices.float().mean()),
        })
        saved_fold_models.append({key: value for key, value in gate.items() if key != "history"})
        print(
            f"[fold] {fold}/{args.folds - 1} feasible={feasible} "
            f"held_coverage={float(held_choices.float().mean()):.4f}",
            flush=True,
        )

    model_names = ("v0_gram", "resolver_only", "p3_abstention", "label_aware_oracle")
    metric_rows = {
        model_name: {name: [] for name in ("all", "warm", "cold")}
        for model_name in model_names
    }
    prediction_records: list[dict] = []
    for idx, row in enumerate(ordered_records):
        uid = str(row["user_id"])
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        gram = unique_in_order(row["v0_top50"])
        resolver = unique_in_order(row["resolver_top50"])
        p3 = (
            anchored_interleave(gram, resolver, cold_items, PROTECTED_PREFIX, COLD_QUOTA)[0]
            if oof_admit[idx] else gram
        )
        oracle = resolver if slice_name == "cold" else gram
        rankings = {
            "v0_gram": gram,
            "resolver_only": resolver,
            "p3_abstention": p3,
            "label_aware_oracle": oracle,
        }
        for name, ranking in rankings.items():
            if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
                raise RuntimeError(f"Invalid ranking for {uid}/{name}")
            result = ranking_metrics(ranking, target)
            metric_rows[name]["all"].append(result)
            metric_rows[name][slice_name].append(result)
        prediction_records.append({
            "user_id": uid,
            "fold": int(folds_tensor[idx]),
            "target": target,
            "is_cold": slice_name == "cold",
            "proposed_cold_item": feature_rows[idx]["proposed_cold_item"],
            "candidate_is_target": bool(labels[idx]),
            "oof_probability": float(oof_probabilities[idx]),
            "admitted": oof_admit[idx],
            "p3_top50": p3[:50],
        })
    metrics = summarize_models(metric_rows)

    full_gate = fit_logistic_gate(
        features_tensor, labels_tensor, args.gate_epochs, args.gate_lr,
        args.gate_l2, args.seed + args.folds,
    )
    full_prob = predict_gate(full_gate, features_tensor)
    full_selected, full_grid = select_threshold(
        ordered_records, full_prob, cold_items, args.train_warm_retention
    )
    torch.save({
        "feature_names": FEATURE_NAMES,
        "fold_models": saved_fold_models,
        "full_model": {key: value for key, value in full_gate.items() if key != "history"},
        "full_selected_threshold": full_selected,
        "protected_prefix": PROTECTED_PREFIX,
        "cold_quota": COLD_QUOTA,
    }, output_dir / "confidence_gates.pt")

    auc = auc_roc(labels_tensor, oof_probabilities)
    admission_coverage = sum(oof_admit) / len(oof_admit)
    admitted_correct = sum(label and admit for label, admit in zip(labels, oof_admit))
    admitted_count = sum(oof_admit)
    p3 = metrics["p3_abstention"]
    v0 = metrics["v0_gram"]
    gates = {
        "all_folds_have_feasible_threshold": all_folds_feasible,
        "oof_warm_ndcg10_ge_0_97x_v0": p3["warm"]["ndcg@10"] >= 0.97 * v0["warm"]["ndcg@10"],
        "oof_cold_ndcg10_ge_2x_v0": p3["cold"]["ndcg@10"] >= 2.0 * v0["cold"]["ndcg@10"],
        "oof_cold_hit10_ge_2x_v0": p3["cold"]["hit@10"] >= 2.0 * v0["cold"]["hit@10"],
        "oof_all_ndcg10_gt_v0": p3["all"]["ndcg@10"] > v0["all"]["ndcg@10"],
        "oof_candidate_correctness_auroc_ge_0_55": auc is not None and auc >= 0.55,
        "actual_admission_coverage_lt_0_95": admission_coverage < 0.95,
        "catalog_outputs_unique": True,
        "held_fold_not_used_by_own_gate": True,
        "validation_only": True,
    }
    verdict = (
        "PASS_TO_R2_FRESH_MEDIUM_SMOKE_DISCUSSION"
        if all(gates.values()) else "FAIL_STOP_R2_P3"
    )
    summary = {
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P3_CONFIDENCE_ABSTENTION",
        "status": "completed",
        "verdict": verdict,
        "evaluation_status": "exploratory_cross_fitted_not_pristine_holdout",
        "metrics_oof": metrics,
        "gates": gates,
        "candidate_correctness": {
            "n_positive": int(labels_tensor.sum()),
            "base_rate": float(labels_tensor.float().mean()),
            "oof_auroc": auc,
            "admission_coverage": admission_coverage,
            "admitted_precision": admitted_correct / max(admitted_count, 1),
        },
        "fold_reports": fold_reports,
        "full_deployable_gate": {
            "selected": full_selected,
            "threshold_grid": full_grid,
            "training_history": full_gate["history"],
        },
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "confidence_features.jsonl").open("w") as f:
        for row in feature_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "predictions_oof.jsonl").open("w") as f:
        for row in prediction_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} coverage={admission_coverage:.4f} "
        f"auc={auc if auc is not None else 'NA'} warm={p3['warm']['ndcg@10']:.6f} "
        f"cold={p3['cold']['ndcg@10']:.6f} all={p3['all']['ndcg@10']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
