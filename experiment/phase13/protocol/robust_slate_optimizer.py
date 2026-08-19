"""Cross-fitted bootstrap-LCB candidate portfolio for v1-R² Beauty P7."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from candidate_portfolio import (
    PORTFOLIO_SIZES,
    action_size,
    build_feature_rows,
    evaluate_actions,
    expected_portfolio_utility,
    ranking_for_action,
    summarize_unconditional,
)
from confidence_abstention import fit_logistic_gate, predict_gate
from counterfactual_slot_router import stable_fold
from route_admission import read_prediction_records
from route_resolve import (
    ResidualUserProjector,
    atomic_json,
    build_validation_examples,
    read_key_value_lines,
    read_sequences,
    read_set,
    semantic_route,
    sha256_file,
)


BETA_GRID = (0.0, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--item-id-file", required=True)
    p.add_argument("--item-embeddings", required=True)
    p.add_argument("--resolver-checkpoint", required=True)
    p.add_argument("--p4-summary", required=True)
    p.add_argument("--p6-summary", required=True)
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap-models", type=int, default=3)
    p.add_argument("--gate-epochs", type=int, default=250)
    p.add_argument("--gate-lr", type=float, default=0.05)
    p.add_argument("--gate-l2", type=float, default=1e-2)
    p.add_argument("--train-warm-retention", type=float, default=0.99)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--recency-decay", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=52345)
    return p.parse_args()


def bootstrap_user_sample(train_indices: list[int], seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randint(len(train_indices), (len(train_indices),), generator=generator)
    return [train_indices[index] for index in picks.tolist()]


def labels_for_rows(rows: list[dict], n_samples: int) -> torch.Tensor:
    labels = torch.zeros(n_samples, dtype=torch.int64)
    for row in rows:
        for item, sample_index in zip(row["modeled_items"], row["sample_indices"]):
            labels[sample_index] = int(item == row["target"])
    return labels


def robust_policy_scores(
    rows: list[dict], probability_members: list[torch.Tensor], beta: float
) -> list[dict]:
    scores: list[dict] = []
    for row in rows:
        action_stats: dict[str, dict[str, float]] = {}
        for size in PORTFOLIO_SIZES:
            member_utilities = []
            for probabilities in probability_members:
                by_item = {
                    item: float(probabilities[index])
                    for item, index in zip(row["modeled_items"], row["sample_indices"])
                }
                member_utilities.append(expected_portfolio_utility(
                    row["v0_top50"], row["resolver_top50"],
                    row["portfolio_candidates"], by_item, size,
                ))
            values = torch.tensor(member_utilities)
            mean = float(values.mean())
            std = float(values.std(unbiased=False))
            action_stats[f"portfolio@{size}"] = {
                "mean_utility": mean,
                "std_utility": std,
                "robust_utility": mean - beta * std,
            }
        action, stats = max(
            action_stats.items(),
            key=lambda pair: (pair[1]["robust_utility"], -action_size(pair[0])),
        )
        scores.append({
            "best_action": action,
            "best_robust_utility": stats["robust_utility"],
            "action_statistics": action_stats,
        })
    return scores


def actions_from_scores(scores: list[dict]) -> list[str]:
    return [
        row["best_action"] if row["best_robust_utility"] > 0 else "abstain"
        for row in scores
    ]


def select_beta(
    rows: list[dict], probability_members: list[torch.Tensor], cold_items: set[str],
    warm_retention: float,
) -> tuple[dict | None, list[dict]]:
    baseline = evaluate_actions(rows, ["abstain"] * len(rows), cold_items)
    grid = []
    for beta in BETA_GRID:
        scores = robust_policy_scores(rows, probability_members, beta)
        actions = actions_from_scores(scores)
        metrics = evaluate_actions(rows, actions, cold_items)
        coverage = sum(action != "abstain" for action in actions) / len(actions)
        feasible = (
            metrics["warm"]["ndcg@10"] >= warm_retention * baseline["warm"]["ndcg@10"]
            and metrics["cold"]["ndcg@10"] > baseline["cold"]["ndcg@10"]
            and metrics["all"]["ndcg@10"] > baseline["all"]["ndcg@10"]
        )
        grid.append({"beta": beta, "coverage": coverage, "feasible": feasible, "metrics": metrics})
    feasible = [row for row in grid if row["feasible"]]
    if not feasible:
        return None, grid
    return max(feasible, key=lambda row: (
        row["metrics"]["cold"]["ndcg@10"], row["metrics"]["all"]["ndcg@10"],
        row["metrics"]["warm"]["ndcg@10"], row["beta"],
    )), grid


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    paths = {
        "p0": Path(args.p0_predictions).resolve(),
        "ids": Path(args.item_id_file).resolve(),
        "emb": Path(args.item_embeddings).resolve(),
        "resolver": Path(args.resolver_checkpoint).resolve(),
        "p4_summary": Path(args.p4_summary).resolve(),
        "p6_summary": Path(args.p6_summary).resolve(),
        "cold": Path(args.cold_items).resolve(),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [
        p.name for p in output_dir.iterdir()
        if p.name not in allowed and not p.name.startswith("status.prework_failed_")
    ]
    if unexpected:
        raise FileExistsError(f"Refusing existing P7 artifacts: {unexpected}")
    inputs = [dataset_dir / "user_sequence.txt", *paths.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing test input in P7: {path}")

    records = read_prediction_records(paths["p0"])
    item_to_lexical = read_key_value_lines(paths["ids"])
    catalog = set(item_to_lexical)
    cold_items = read_set(paths["cold"])
    routes = {item: semantic_route(value, 1) for item, value in item_to_lexical.items()}
    payload = torch.load(paths["emb"], map_location="cpu")
    item_ids = list(payload["item_ids"])
    embeddings_cpu = F.normalize(payload["embeddings"].float(), dim=1)
    item_to_idx = {item: i for i, item in enumerate(item_ids)}
    if set(item_ids) != catalog:
        raise ValueError("Embedding/catalog mismatch")
    checkpoint = torch.load(paths["resolver"], map_location="cpu")
    resolver = ResidualUserProjector(checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    resolver.load_state_dict(checkpoint["state_dict"])
    resolver.eval()
    device = torch.device(args.device)
    resolver.to(device)
    embeddings_device = embeddings_cpu.to(device)
    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    validation = build_validation_examples(sequences, item_to_idx, embeddings_cpu, args.max_history, args.recency_decay)
    record_by_uid = {str(row["user_id"]): row for row in records}
    ordered_uids = [uid for uid, _ in sequences if uid in record_by_uid]
    rows, features = build_feature_rows(
        records, ordered_uids, validation, resolver, embeddings_device,
        item_to_idx, routes, cold_items, device,
    )
    labels = labels_for_rows(rows, len(features))
    p4 = json.loads(paths["p4_summary"].read_text())["metrics_oof"]["p4_counterfactual_slot_router"]
    p6 = json.loads(paths["p6_summary"].read_text())["metrics"]["p6_candidate_portfolio"]
    config = {
        **vars(args), "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P7_ROBUST_SLATE",
        "beta_grid": BETA_GRID, "portfolio_sizes": PORTFOLIO_SIZES,
        "evaluation_status": "beauty_validation_outer_5fold_oof",
        "test_predictions_opened": False,
        "input_sha256": {str(p): sha256_file(p) for p in inputs},
    }
    atomic_json(output_dir / "config.json", config)

    folds = torch.tensor([stable_fold(uid, args.folds) for uid in ordered_uids])
    oof_actions = ["abstain"] * len(rows)
    oof_scores: list[dict | None] = [None] * len(rows)
    fold_reports, saved_models = [], []
    all_feasible = True
    for fold in range(args.folds):
        train = torch.where(folds != fold)[0].tolist()
        held = torch.where(folds == fold)[0].tolist()
        models, probability_members = [], []
        for member in range(args.bootstrap_models):
            sampled_users = bootstrap_user_sample(train, args.seed + fold * 100 + member)
            sample_indices = [index for user in sampled_users for index in rows[user]["sample_indices"]]
            model = fit_logistic_gate(
                features[sample_indices], labels[sample_indices], args.gate_epochs,
                args.gate_lr, args.gate_l2, args.seed + fold * 100 + member,
            )
            models.append(model)
            probability_members.append(predict_gate(model, features))
        train_rows = [rows[i] for i in train]
        train_probs = [member for member in probability_members]
        selected, grid = select_beta(train_rows, train_probs, cold_items, args.train_warm_retention)
        all_feasible &= selected is not None
        beta = 2.0 if selected is None else selected["beta"]
        all_scores = robust_policy_scores(rows, probability_members, beta)
        held_scores = [all_scores[i] for i in held]
        held_actions = ["abstain"] * len(held) if selected is None else actions_from_scores(held_scores)
        for local, index in enumerate(held):
            oof_actions[index] = held_actions[local]
            oof_scores[index] = held_scores[local]
        fold_reports.append({
            "fold": fold, "selected": selected, "beta_grid": grid,
            "held_action_counts": {a: held_actions.count(a) for a in ("abstain", "portfolio@2", "portfolio@3")},
            "held_metrics": evaluate_actions([rows[i] for i in held], held_actions, cold_items),
        })
        saved_models.append(models)

    if any(score is None for score in oof_scores):
        raise RuntimeError("Missing OOF score")
    metrics = summarize_unconditional(rows, cold_items)
    metrics["p7_robust_slate"] = evaluate_actions(rows, oof_actions, cold_items)
    v0, p7 = metrics["v0_gram"], metrics["p7_robust_slate"]
    action_counts = {a: oof_actions.count(a) for a in ("abstain", "portfolio@2", "portfolio@3")}
    coverage = 1 - action_counts["abstain"] / len(rows)
    cold_rows = [row for row in rows if row["is_cold"]]
    hits = {str(k): sum(row["target"] in row["portfolio_candidates"][:k] for row in cold_rows) for k in (1, 2, 3)}
    predictions = []
    for row, action, score in zip(rows, oof_actions, oof_scores):
        ranking = ranking_for_action(row, action)
        predictions.append({
            "user_id": row["user_id"], "fold": row["fold"], "target": row["target"],
            "is_cold": row["is_cold"], "portfolio_candidates": row["portfolio_candidates"],
            "selected_action": action, "action_statistics": score["action_statistics"],
            "p7_top50": ranking[:50],
        })
    gates = {
        "candidate_top3_recall_ge_2_5x_top1": hits["3"] / max(hits["1"], 1) >= 2.5,
        "all_outer_folds_train_feasible": all_feasible,
        "warm_ndcg10_ge_0_97x_v0": p7["warm"]["ndcg@10"] >= .97 * v0["warm"]["ndcg@10"],
        "cold_ndcg10_ge_2x_v0": p7["cold"]["ndcg@10"] >= 2 * v0["cold"]["ndcg@10"],
        "cold_hit10_ge_2x_v0": p7["cold"]["hit@10"] >= 2 * v0["cold"]["hit@10"],
        "all_ndcg10_gt_v0": p7["all"]["ndcg@10"] > v0["all"]["ndcg@10"],
        "cold_ndcg10_gt_p4": p7["cold"]["ndcg@10"] > p4["cold"]["ndcg@10"],
        "all_ndcg10_gt_p4": p7["all"]["ndcg@10"] > p4["all"]["ndcg@10"],
        "cold_ndcg10_gt_p6": p7["cold"]["ndcg@10"] > p6["cold"]["ndcg@10"],
        "all_ndcg10_gt_p6": p7["all"]["ndcg@10"] > p6["all"]["ndcg@10"],
        "coverage_ge_0_10_lt_0_80": .10 <= coverage < .80,
        "all_interventions_multi_candidate": all(a == "abstain" or action_size(a) >= 2 for a in oof_actions),
        "catalog_outputs_unique": all(len(x["p7_top50"]) == len(set(x["p7_top50"])) and set(x["p7_top50"]) <= catalog for x in predictions),
        "validation_only": True,
    }
    verdict = "PASS_TO_R2_P7_BEAUTY_DISCUSSION" if all(gates.values()) else "FAIL_STOP_R2_P7"
    summary = {
        "experiment_id": config["experiment_id"], "status": "completed", "verdict": verdict,
        "metrics": metrics, "p4_oof_reference": p4, "p6_oof_reference": p6, "gates": gates,
        "candidate_report": {"n_cold_users": len(cold_rows), "cumulative_hits": hits, "top3_vs_top1_ratio": hits["3"] / max(hits["1"], 1)},
        "diagnostics": {"coverage": coverage, "action_counts": action_counts},
        "fold_reports": fold_reports, "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    torch.save({"fold_bootstrap_models": saved_models, "beta_grid": BETA_GRID}, output_dir / "robust_slate.pt")
    with (output_dir / "predictions_validation.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[result] verdict={verdict} coverage={coverage:.3f} warm={p7['warm']['ndcg@10']:.6f} cold={p7['cold']['ndcg@10']:.6f} all={p7['all']['ndcg@10']:.6f}", flush=True)


if __name__ == "__main__":
    main()
