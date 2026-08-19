"""Cross-fitted counterfactual expected-utility slot router for v1-R² P4.

The frozen GRAM list remains the safety anchor and the frozen resolver proposes
one cold candidate.  A shared item-relevance model scores that candidate and
the GRAM items at ranks 7--10.  It then estimates the NDCG@10 delta of inserting
the candidate at rank 7 or rank 10, explicitly accounting for displaced GRAM
items.  All reported validation predictions are outer-fold held predictions.
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

from confidence_abstention import auc_roc, fit_logistic_gate, predict_gate
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
    "item_cosine",
    "item_vs_resolver_top_gap",
    "resolver_reciprocal_rank",
    "depth1_route_reciprocal_rank",
    "gram_reciprocal_rank",
    "catalog_is_cold",
    "is_proposed_cold",
    "gram_rank_normalized",
)
ACTION_POSITIONS = (7, 10)
COVERAGE_GRID = tuple(x / 10 for x in range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument("--item-id-file", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--resolver-checkpoint", required=True)
    parser.add_argument("--cold-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gate-epochs", type=int, default=250)
    parser.add_argument("--gate-lr", type=float, default=0.05)
    parser.add_argument("--gate-l2", type=float, default=1e-2)
    parser.add_argument("--train-warm-retention", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=22345)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    return parser.parse_args()


def stable_fold(user_id: str, folds: int) -> int:
    value = int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:8], "big")
    return value % folds


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def ndcg_discount(rank: int | None) -> float:
    return 0.0 if rank is None or rank > 10 else 1.0 / math.log2(rank + 1)


def insert_candidate(
    gram_items: list[str], resolver_items: list[str], candidate: str, position: int
) -> list[str]:
    if position not in ACTION_POSITIONS:
        raise ValueError(f"Unsupported slot position: {position}")
    gram = unique_in_order(gram_items)
    resolver = unique_in_order(resolver_items)
    prefix = gram[: position - 1]
    return unique_in_order([*prefix, candidate, *gram[position - 1 :], *resolver])


def affected_items(gram_items: list[str], candidate: str) -> list[str]:
    return unique_in_order([candidate, *unique_in_order(gram_items)[6:10]])


def extract_item_feature_vector(
    item: str,
    candidate: str,
    projected_user: torch.Tensor,
    embeddings: torch.Tensor,
    item_to_idx: dict[str, int],
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
    gram_items: list[str],
    resolver_items: list[str],
) -> list[float]:
    """Build the frozen P4 target-free shared-relevance feature vector."""
    gram = unique_in_order(gram_items)
    resolver = unique_in_order(resolver_items)
    gram_rank = {value: rank for rank, value in enumerate(gram, 1)}
    resolver_rank = {value: rank for rank, value in enumerate(resolver, 1)}
    route_order = unique_in_order(item_routes[value] for value in gram)
    route_rank = {route: rank for rank, route in enumerate(route_order, 1)}
    resolver_top_score = float(projected_user @ embeddings[item_to_idx[resolver[0]]])
    cosine = float(projected_user @ embeddings[item_to_idx[item]])
    g_rank = gram_rank.get(item)
    return [
        cosine,
        cosine - resolver_top_score,
        reciprocal_rank(resolver_rank.get(item)),
        reciprocal_rank(route_rank.get(item_routes[item])),
        reciprocal_rank(g_rank),
        float(item in cold_items),
        float(item == candidate),
        (g_rank / 50.0) if g_rank is not None else 1.02,
    ]


def expected_action_utility(
    gram_items: list[str],
    resolver_items: list[str],
    candidate: str,
    probabilities: dict[str, float],
    position: int,
) -> float:
    gram = unique_in_order(gram_items)
    changed = insert_candidate(gram, resolver_items, candidate, position)
    old_rank = {item: rank for rank, item in enumerate(gram, 1)}
    new_rank = {item: rank for rank, item in enumerate(changed, 1)}
    return sum(
        probabilities[item]
        * (ndcg_discount(new_rank.get(item)) - ndcg_discount(old_rank.get(item)))
        for item in probabilities
    )


def choose_best_action(
    row: dict, sample_probabilities: list[float]
) -> tuple[str, float, dict[str, float]]:
    probability_by_item = dict(zip(row["modeled_items"], sample_probabilities))
    utilities = {
        f"insert@{position}": expected_action_utility(
            row["v0_top50"], row["resolver_top50"], row["proposed_cold_item"],
            probability_by_item, position,
        )
        for position in ACTION_POSITIONS
    }
    action, utility = max(utilities.items(), key=lambda pair: (pair[1], -int(pair[0].split("@")[1])))
    return action, utility, utilities


def ranking_for_action(row: dict, action: str) -> list[str]:
    if action == "abstain":
        return unique_in_order(row["v0_top50"])
    position = int(action.split("@")[1])
    return insert_candidate(
        row["v0_top50"], row["resolver_top50"], row["proposed_cold_item"], position
    )


def evaluate_actions(rows: list[dict], actions: list[str], cold_items: set[str]) -> dict:
    buckets = {name: [] for name in ("all", "warm", "cold")}
    for row, action in zip(rows, actions):
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        metrics = ranking_metrics(ranking_for_action(row, action), target)
        buckets["all"].append(metrics)
        buckets[slice_name].append(metrics)
    return {name: average_metrics(values) for name, values in buckets.items()}


def policy_scores(rows: list[dict], flat_probabilities: torch.Tensor) -> list[dict]:
    scores: list[dict] = []
    for row in rows:
        probabilities = [float(flat_probabilities[index]) for index in row["sample_indices"]]
        action, utility, utilities = choose_best_action(row, probabilities)
        scores.append({"best_action": action, "best_utility": utility, "utilities": utilities})
    return scores


def apply_threshold(scores: list[dict], threshold: float) -> list[str]:
    return [
        score["best_action"]
        if score["best_utility"] > 0.0 and score["best_utility"] >= threshold
        else "abstain"
        for score in scores
    ]


def select_policy(
    rows: list[dict], scores: list[dict], cold_items: set[str], warm_retention: float
) -> tuple[dict | None, list[dict]]:
    baseline = evaluate_actions(rows, ["abstain"] * len(rows), cold_items)
    utilities = torch.tensor([score["best_utility"] for score in scores])
    grid: list[dict] = []
    for target_coverage in COVERAGE_GRID:
        k = max(1, math.ceil(target_coverage * len(rows)))
        threshold = float(torch.topk(utilities, k=k).values[-1])
        actions = apply_threshold(scores, threshold)
        metrics = evaluate_actions(rows, actions, cold_items)
        coverage = sum(action != "abstain" for action in actions) / len(actions)
        feasible = (
            metrics["warm"]["ndcg@10"] >= warm_retention * baseline["warm"]["ndcg@10"]
            and metrics["cold"]["ndcg@10"] > baseline["cold"]["ndcg@10"]
            and metrics["all"]["ndcg@10"] > baseline["all"]["ndcg@10"]
        )
        grid.append({
            "target_coverage": target_coverage,
            "actual_coverage": coverage,
            "threshold": threshold,
            "feasible": feasible,
            "metrics": metrics,
        })
    feasible_rows = [row for row in grid if row["feasible"]]
    if not feasible_rows:
        return None, grid
    selected = max(
        feasible_rows,
        key=lambda row: (
            row["metrics"]["cold"]["ndcg@10"],
            row["metrics"]["all"]["ndcg@10"],
            row["metrics"]["warm"]["ndcg@10"],
            -row["actual_coverage"],
        ),
    )
    return selected, grid


def build_feature_rows(
    records: list[dict],
    ordered_uids: list[str],
    validation: dict,
    model: ResidualUserProjector,
    embeddings_device: torch.Tensor,
    item_to_idx: dict[str, int],
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
    device: torch.device,
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    record_by_uid = {str(row["user_id"]): row for row in records}
    feature_rows: list[dict] = []
    flat_features: list[list[float]] = []
    flat_labels: list[int] = []
    with torch.no_grad():
        for offset in range(0, len(ordered_uids), 256):
            batch_uids = ordered_uids[offset : offset + 256]
            histories = torch.stack([validation[uid][0] for uid in batch_uids]).to(device)
            projected = model(histories)
            for local_index, uid in enumerate(batch_uids):
                source = record_by_uid[uid]
                gram = unique_in_order(source["v0_top50"])
                resolver = unique_in_order(source["resolver_top50"])
                protected = set(gram[:6])
                eligible = [item for item in resolver if item in cold_items and item not in protected]
                if not eligible:
                    raise ValueError(f"No eligible cold proposal for {uid}")
                candidate = eligible[0]
                modeled = affected_items(gram, candidate)
                target = str(source["target"])
                sample_indices: list[int] = []
                feature_payload: list[dict] = []
                for item in modeled:
                    features = extract_item_feature_vector(
                        item, candidate, projected[local_index], embeddings_device,
                        item_to_idx, item_routes, cold_items, gram, resolver,
                    )
                    sample_indices.append(len(flat_features))
                    flat_features.append(features)
                    flat_labels.append(int(item == target))
                    feature_payload.append(dict(zip(FEATURE_NAMES, features)))
                feature_rows.append({
                    "user_id": uid,
                    "target": target,
                    "is_cold": target in cold_items,
                    "fold": stable_fold(uid, 5),
                    "v0_top50": gram,
                    "resolver_top50": resolver,
                    "proposed_cold_item": candidate,
                    "modeled_items": modeled,
                    "sample_indices": sample_indices,
                    "item_features": feature_payload,
                })
            print(f"[features] {min(offset + 256, len(ordered_uids))}/{len(ordered_uids)}", flush=True)
    return (
        feature_rows,
        torch.tensor(flat_features, dtype=torch.float32),
        torch.tensor(flat_labels, dtype=torch.int64),
    )


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
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing P4 scientific artifacts: {unexpected}")
    inputs = [
        dataset_dir / "user_sequence.txt", p0_path, item_id_path,
        embedding_path, resolver_path, cold_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing test input in P4: {path}")

    records = read_prediction_records(p0_path)
    item_to_lexical = read_key_value_lines(item_id_path)
    catalog = set(item_to_lexical)
    cold_items = read_set(cold_path)
    item_routes = {item: semantic_route(lexical, 1) for item, lexical in item_to_lexical.items()}
    embedding_payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    if set(item_ids) != catalog or not cold_items <= catalog:
        raise ValueError("Catalog, embedding, or cold-state mismatch")

    checkpoint = torch.load(resolver_path, map_location="cpu")
    model = ResidualUserProjector(checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
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

    config = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "p0_predictions": str(p0_path),
        "item_id_file": str(item_id_path),
        "item_embeddings": str(embedding_path),
        "resolver_checkpoint": str(resolver_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P4_COUNTERFACTUAL_SLOT_ROUTER",
        "feature_names": FEATURE_NAMES,
        "action_positions": ACTION_POSITIONS,
        "coverage_grid": COVERAGE_GRID,
        "fold_rule": "uint64_be(sha256(user_id)[:8]) mod 5",
        "evaluation_status": "validation_outer_5fold_oof_exploratory",
        "test_predictions_opened": False,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
    }
    atomic_json(output_dir / "config.json", config)

    rows, features, labels = build_feature_rows(
        records, ordered_uids, validation, model, embeddings_device,
        item_to_idx, item_routes, cold_items, device,
    )
    user_folds = torch.tensor([stable_fold(uid, args.folds) for uid in ordered_uids])
    oof_sample_probabilities = torch.empty(len(features))
    oof_actions = ["abstain"] * len(rows)
    oof_policy_scores: list[dict | None] = [None] * len(rows)
    fold_reports: list[dict] = []
    saved_models: list[dict] = []
    all_folds_feasible = True
    for fold in range(args.folds):
        train_user_indices = torch.where(user_folds != fold)[0].tolist()
        held_user_indices = torch.where(user_folds == fold)[0].tolist()
        train_sample_indices = [index for row_index in train_user_indices for index in rows[row_index]["sample_indices"]]
        gate = fit_logistic_gate(
            features[train_sample_indices], labels[train_sample_indices],
            args.gate_epochs, args.gate_lr, args.gate_l2, args.seed + fold,
        )
        all_probabilities = predict_gate(gate, features)
        train_rows = [rows[index] for index in train_user_indices]
        held_rows = [rows[index] for index in held_user_indices]
        train_scores = policy_scores(train_rows, all_probabilities)
        held_scores = policy_scores(held_rows, all_probabilities)
        selected, grid = select_policy(
            train_rows, train_scores, cold_items, args.train_warm_retention
        )
        feasible = selected is not None
        all_folds_feasible &= feasible
        threshold = float("inf") if selected is None else selected["threshold"]
        held_actions = apply_threshold(held_scores, threshold)
        held_metrics = evaluate_actions(held_rows, held_actions, cold_items)
        held_baseline = evaluate_actions(held_rows, ["abstain"] * len(held_rows), cold_items)
        for local, row_index in enumerate(held_user_indices):
            oof_actions[row_index] = held_actions[local]
            oof_policy_scores[row_index] = held_scores[local]
            for sample_index in rows[row_index]["sample_indices"]:
                oof_sample_probabilities[sample_index] = all_probabilities[sample_index]
        action_counts = {action: held_actions.count(action) for action in ("abstain", "insert@7", "insert@10")}
        fold_reports.append({
            "fold": fold,
            "n_train_users": len(train_rows),
            "n_held_users": len(held_rows),
            "n_train_item_samples": len(train_sample_indices),
            "selected": selected,
            "threshold_grid": grid,
            "held_metrics": held_metrics,
            "held_baseline": held_baseline,
            "held_action_counts": action_counts,
            "held_warm_retention": held_metrics["warm"]["ndcg@10"] / held_baseline["warm"]["ndcg@10"],
            "held_overall_improved": held_metrics["all"]["ndcg@10"] > held_baseline["all"]["ndcg@10"],
        })
        saved_models.append({key: value for key, value in gate.items() if key != "history"})
        print(f"[fold] {fold} feasible={feasible} actions={action_counts}", flush=True)

    if any(score is None for score in oof_policy_scores):
        raise RuntimeError("Missing held-fold policy score")
    baseline_actions = ["abstain"] * len(rows)
    baseline = evaluate_actions(rows, baseline_actions, cold_items)
    p4 = evaluate_actions(rows, oof_actions, cold_items)
    resolver_metrics = {
        name: [] for name in ("all", "warm", "cold")
    }
    prediction_records: list[dict] = []
    candidate_indices: list[int] = []
    for index, row in enumerate(rows):
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        metrics = ranking_metrics(row["resolver_top50"], target)
        resolver_metrics["all"].append(metrics)
        resolver_metrics[slice_name].append(metrics)
        candidate_sample_index = row["sample_indices"][row["modeled_items"].index(row["proposed_cold_item"])]
        candidate_indices.append(candidate_sample_index)
        score = oof_policy_scores[index]
        prediction_records.append({
            "user_id": row["user_id"],
            "fold": int(user_folds[index]),
            "target": target,
            "is_cold": target in cold_items,
            "proposed_cold_item": row["proposed_cold_item"],
            "candidate_is_target": row["proposed_cold_item"] == target,
            "candidate_oof_relevance_probability": float(oof_sample_probabilities[candidate_sample_index]),
            "predicted_action_utilities": score["utilities"],
            "selected_action": oof_actions[index],
            "p4_top50": ranking_for_action(row, oof_actions[index])[:50],
        })
    metrics = {
        "v0_gram": baseline,
        "resolver_only": {name: average_metrics(values) for name, values in resolver_metrics.items()},
        "p4_counterfactual_slot_router": p4,
    }

    full_gate = fit_logistic_gate(
        features, labels, args.gate_epochs, args.gate_lr, args.gate_l2,
        args.seed + args.folds,
    )
    full_probabilities = predict_gate(full_gate, features)
    full_scores = policy_scores(rows, full_probabilities)
    full_selected, full_grid = select_policy(
        rows, full_scores, cold_items, args.train_warm_retention
    )
    torch.save({
        "feature_names": FEATURE_NAMES,
        "action_positions": ACTION_POSITIONS,
        "fold_models": saved_models,
        "full_model": {key: value for key, value in full_gate.items() if key != "history"},
        "full_selected_policy": full_selected,
    }, output_dir / "counterfactual_slot_router.pt")

    candidate_labels = labels[candidate_indices]
    candidate_probabilities = oof_sample_probabilities[candidate_indices]
    candidate_auc = auc_roc(candidate_labels, candidate_probabilities)
    relevance_auc = auc_roc(labels, oof_sample_probabilities)
    coverage = sum(action != "abstain" for action in oof_actions) / len(oof_actions)
    action_counts = {action: oof_actions.count(action) for action in ("abstain", "insert@7", "insert@10")}
    min_fold_warm = min(report["held_warm_retention"] for report in fold_reports)
    improving_folds = sum(report["held_overall_improved"] for report in fold_reports)
    gates = {
        "all_folds_have_feasible_policy": all_folds_feasible,
        "oof_warm_ndcg10_ge_0_98x_v0": p4["warm"]["ndcg@10"] >= 0.98 * baseline["warm"]["ndcg@10"],
        "oof_cold_ndcg10_ge_1_8x_v0": p4["cold"]["ndcg@10"] >= 1.8 * baseline["cold"]["ndcg@10"],
        "oof_cold_hit10_ge_2x_v0": p4["cold"]["hit@10"] >= 2.0 * baseline["cold"]["hit@10"],
        "oof_all_ndcg10_gt_v0": p4["all"]["ndcg@10"] > baseline["all"]["ndcg@10"],
        "minimum_fold_warm_retention_ge_0_97": min_fold_warm >= 0.97,
        "at_least_four_folds_overall_improve": improving_folds >= 4,
        "intervention_coverage_lt_0_80": coverage < 0.80,
        "catalog_outputs_unique": all(
            len(record["p4_top50"]) == len(set(record["p4_top50"]))
            and set(record["p4_top50"]) <= catalog
            for record in prediction_records
        ),
        "validation_only": True,
    }
    verdict = (
        "PASS_TO_R2_P4_FRESH_DISJOINT_CONFIRMATION_DISCUSSION"
        if all(gates.values()) else "FAIL_STOP_R2_P4"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "evaluation_status": config["evaluation_status"],
        "metrics_oof": metrics,
        "gates": gates,
        "diagnostics": {
            "intervention_coverage": coverage,
            "action_counts": action_counts,
            "minimum_fold_warm_retention": min_fold_warm,
            "improving_fold_count": improving_folds,
            "candidate_correctness_oof_auroc": candidate_auc,
            "shared_relevance_oof_auroc": relevance_auc,
            "n_candidate_positive": int(candidate_labels.sum()),
            "n_relevance_positive": int(labels.sum()),
        },
        "fold_reports": fold_reports,
        "full_deployable_policy": {
            "selected": full_selected,
            "threshold_grid": full_grid,
            "training_history": full_gate["history"],
        },
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_oof.jsonl").open("w") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} coverage={coverage:.4f} actions={action_counts} "
        f"warm={p4['warm']['ndcg@10']:.6f} cold={p4['cold']['ndcg@10']:.6f} "
        f"all={p4['all']['ndcg@10']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
