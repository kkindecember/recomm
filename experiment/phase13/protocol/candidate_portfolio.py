"""Outer-fold risk-limited multi-candidate portfolio for v1-R² P6."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from confidence_abstention import predict_gate
from counterfactual_slot_router import (
    extract_item_feature_vector,
    ndcg_discount,
    stable_fold,
)
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


PORTFOLIO_SIZES = (2, 3)
COVERAGE_GRID = tuple(value / 10 for value in range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument("--item-id-file", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--resolver-checkpoint", required=True)
    parser.add_argument("--p4-checkpoint", required=True)
    parser.add_argument("--p4-summary", required=True)
    parser.add_argument("--cold-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-warm-retention", type=float, default=0.99)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42345)
    return parser.parse_args()


def action_size(action: str) -> int:
    if action == "abstain":
        return 0
    prefix, value = action.split("@", 1)
    if prefix != "portfolio" or int(value) not in PORTFOLIO_SIZES:
        raise ValueError(f"Unsupported portfolio action: {action}")
    return int(value)


def portfolio_ranking(
    gram_items: list[str], resolver_items: list[str], candidates: list[str], size: int
) -> list[str]:
    """Protect GRAM top-(10-size), then place 2/3 cold candidates at the tail."""
    if size not in PORTFOLIO_SIZES:
        raise ValueError(f"Unsupported portfolio size: {size}")
    gram = unique_in_order(gram_items)
    resolver = unique_in_order(resolver_items)
    portfolio = unique_in_order(candidates)[:size]
    if len(portfolio) != size:
        raise ValueError(f"Portfolio has only {len(portfolio)} unique candidates")
    anchor_count = 10 - size
    return unique_in_order([*gram[:anchor_count], *portfolio, *gram[anchor_count:], *resolver])


def ranking_for_action(row: dict, action: str) -> list[str]:
    if action == "abstain":
        return unique_in_order(row["v0_top50"])
    return portfolio_ranking(
        row["v0_top50"], row["resolver_top50"], row["portfolio_candidates"],
        action_size(action),
    )


def expected_portfolio_utility(
    gram_items: list[str],
    resolver_items: list[str],
    candidates: list[str],
    probabilities: dict[str, float],
    size: int,
) -> float:
    gram = unique_in_order(gram_items)
    changed = portfolio_ranking(gram, resolver_items, candidates, size)
    old_rank = {item: rank for rank, item in enumerate(gram, 1)}
    new_rank = {item: rank for rank, item in enumerate(changed, 1)}
    return sum(
        probability
        * (ndcg_discount(new_rank.get(item)) - ndcg_discount(old_rank.get(item)))
        for item, probability in probabilities.items()
    )


def choose_best_portfolio_action(
    row: dict, sample_probabilities: list[float]
) -> tuple[str, float, dict[str, float]]:
    probabilities = dict(zip(row["modeled_items"], sample_probabilities))
    utilities = {
        f"portfolio@{size}": expected_portfolio_utility(
            row["v0_top50"], row["resolver_top50"], row["portfolio_candidates"],
            probabilities, size,
        )
        for size in PORTFOLIO_SIZES
    }
    action, utility = max(
        utilities.items(), key=lambda pair: (pair[1], -action_size(pair[0]))
    )
    return action, utility, utilities


def evaluate_actions(rows: list[dict], actions: list[str], cold_items: set[str]) -> dict:
    buckets = {name: [] for name in ("all", "warm", "cold")}
    for row, action in zip(rows, actions):
        target = str(row["target"])
        split = "cold" if target in cold_items else "warm"
        metrics = ranking_metrics(ranking_for_action(row, action), target)
        buckets["all"].append(metrics)
        buckets[split].append(metrics)
    return {name: average_metrics(values) for name, values in buckets.items()}


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
    return max(
        feasible_rows,
        key=lambda row: (
            row["metrics"]["cold"]["ndcg@10"],
            row["metrics"]["all"]["ndcg@10"],
            row["metrics"]["warm"]["ndcg@10"],
            -row["actual_coverage"],
        ),
    ), grid


def build_feature_rows(
    records: list[dict],
    ordered_uids: list[str],
    validation: dict,
    resolver: ResidualUserProjector,
    embeddings_device: torch.Tensor,
    item_to_idx: dict[str, int],
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
    device: torch.device,
) -> tuple[list[dict], torch.Tensor]:
    record_by_uid = {str(row["user_id"]): row for row in records}
    rows: list[dict] = []
    flat_features: list[list[float]] = []
    with torch.no_grad():
        for offset in range(0, len(ordered_uids), 256):
            batch_uids = ordered_uids[offset:offset + 256]
            histories = torch.stack([validation[uid][0] for uid in batch_uids]).to(device)
            projected = resolver(histories)
            for local_index, uid in enumerate(batch_uids):
                source = record_by_uid[uid]
                gram = unique_in_order(source["v0_top50"])
                resolver_items = unique_in_order(source["resolver_top50"])
                protected = set(gram[:7])
                candidates = [
                    item for item in resolver_items
                    if item in cold_items and item not in protected
                ][:3]
                if len(candidates) != 3:
                    raise ValueError(f"Fewer than three portfolio candidates for {uid}")
                modeled = unique_in_order([*candidates, *gram[7:10]])
                sample_indices: list[int] = []
                feature_payload: list[dict] = []
                for item in modeled:
                    candidate_context = item if item in candidates else candidates[0]
                    features = extract_item_feature_vector(
                        item, candidate_context, projected[local_index], embeddings_device,
                        item_to_idx, item_routes, cold_items, gram, resolver_items,
                    )
                    sample_indices.append(len(flat_features))
                    flat_features.append(features)
                    feature_payload.append({"item": item, "features": features})
                rows.append({
                    "user_id": uid,
                    "target": str(source["target"]),
                    "is_cold": str(source["target"]) in cold_items,
                    "fold": stable_fold(uid, 5),
                    "v0_top50": gram,
                    "resolver_top50": resolver_items,
                    "portfolio_candidates": candidates,
                    "modeled_items": modeled,
                    "sample_indices": sample_indices,
                    "item_features": feature_payload,
                })
            print(f"[features] {min(offset + 256, len(ordered_uids))}/{len(ordered_uids)}", flush=True)
    return rows, torch.tensor(flat_features, dtype=torch.float32)


def policy_scores(rows: list[dict], flat_probabilities: torch.Tensor) -> list[dict]:
    result: list[dict] = []
    for row in rows:
        probabilities = [float(flat_probabilities[index]) for index in row["sample_indices"]]
        action, utility, utilities = choose_best_portfolio_action(row, probabilities)
        result.append({
            "best_action": action,
            "best_utility": utility,
            "utilities": utilities,
        })
    return result


def summarize_unconditional(rows: list[dict], cold_items: set[str]) -> dict:
    return {
        "v0_gram": evaluate_actions(rows, ["abstain"] * len(rows), cold_items),
        "unconditional_portfolio2": evaluate_actions(
            rows, ["portfolio@2"] * len(rows), cold_items
        ),
        "unconditional_portfolio3": evaluate_actions(
            rows, ["portfolio@3"] * len(rows), cold_items
        ),
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
    p4_path = Path(args.p4_checkpoint).resolve()
    p4_summary_path = Path(args.p4_summary).resolve()
    cold_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [
        path.name for path in output_dir.iterdir()
        if path.name not in allowed and not path.name.startswith("status.launch_failed_")
    ]
    if unexpected:
        raise FileExistsError(f"Refusing existing P6 scientific artifacts: {unexpected}")
    inputs = [
        dataset_dir / "user_sequence.txt", p0_path, item_id_path, embedding_path,
        resolver_path, p4_path, p4_summary_path, cold_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing test input in P6: {path}")

    records = read_prediction_records(p0_path)
    item_to_lexical = read_key_value_lines(item_id_path)
    catalog = set(item_to_lexical)
    cold_items = read_set(cold_path)
    item_routes = {
        item: semantic_route(lexical, 1) for item, lexical in item_to_lexical.items()
    }
    embedding_payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    if set(item_ids) != catalog or not cold_items <= catalog:
        raise ValueError("Catalog, embedding, or cold-state mismatch")

    resolver_checkpoint = torch.load(resolver_path, map_location="cpu")
    resolver = ResidualUserProjector(
        resolver_checkpoint["dim"], resolver_checkpoint["hidden_dim"],
        resolver_checkpoint["dropout"],
    )
    resolver.load_state_dict(resolver_checkpoint["state_dict"])
    resolver.eval()
    device = torch.device(args.device)
    resolver.to(device)
    embeddings_device = embeddings_cpu.to(device)
    p4_checkpoint = torch.load(p4_path, map_location="cpu")
    p4_summary = json.loads(p4_summary_path.read_text())
    fold_models = p4_checkpoint["fold_models"]
    if len(fold_models) != args.folds:
        raise ValueError("P4 fold model count does not match P6 folds")

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
        "p4_checkpoint": str(p4_path),
        "p4_summary": str(p4_summary_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P6_CANDIDATE_PORTFOLIO",
        "portfolio_sizes": PORTFOLIO_SIZES,
        "coverage_grid": COVERAGE_GRID,
        "anchor_rule": "protect_v0_gram_top7; portfolio2@9-10; portfolio3@8-10",
        "fold_rule": "uint64_be(sha256(user_id)[:8]) mod 5",
        "evaluation_status": "validation_outer_5fold_oof_exploratory",
        "test_predictions_opened": False,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
    }
    atomic_json(output_dir / "config.json", config)

    rows, features = build_feature_rows(
        records, ordered_uids, validation, resolver, embeddings_device,
        item_to_idx, item_routes, cold_items, device,
    )
    user_folds = torch.tensor([stable_fold(uid, args.folds) for uid in ordered_uids])
    oof_actions = ["abstain"] * len(rows)
    oof_scores: list[dict | None] = [None] * len(rows)
    fold_reports: list[dict] = []
    all_folds_feasible = True
    for fold in range(args.folds):
        train_indices = torch.where(user_folds != fold)[0].tolist()
        held_indices = torch.where(user_folds == fold)[0].tolist()
        flat_probabilities = predict_gate(fold_models[fold], features)
        all_scores = policy_scores(rows, flat_probabilities)
        train_rows = [rows[index] for index in train_indices]
        held_rows = [rows[index] for index in held_indices]
        train_scores = [all_scores[index] for index in train_indices]
        held_scores = [all_scores[index] for index in held_indices]
        selected, grid = select_policy(
            train_rows, train_scores, cold_items, args.train_warm_retention
        )
        feasible = selected is not None
        all_folds_feasible &= feasible
        threshold = float("inf") if selected is None else float(selected["threshold"])
        held_actions = apply_threshold(held_scores, threshold)
        held_metrics = evaluate_actions(held_rows, held_actions, cold_items)
        held_baseline = evaluate_actions(
            held_rows, ["abstain"] * len(held_rows), cold_items
        )
        for local, row_index in enumerate(held_indices):
            oof_actions[row_index] = held_actions[local]
            oof_scores[row_index] = held_scores[local]
        fold_reports.append({
            "fold": fold,
            "n_train_users": len(train_indices),
            "n_held_users": len(held_indices),
            "selected": selected,
            "threshold_grid": grid,
            "held_baseline": held_baseline,
            "held_metrics": held_metrics,
            "held_action_counts": {
                action: held_actions.count(action)
                for action in ("abstain", "portfolio@2", "portfolio@3")
            },
        })

    if any(score is None for score in oof_scores):
        raise RuntimeError("Missing OOF portfolio score")
    metrics = summarize_unconditional(rows, cold_items)
    metrics["p6_candidate_portfolio"] = evaluate_actions(rows, oof_actions, cold_items)
    v0 = metrics["v0_gram"]
    p6 = metrics["p6_candidate_portfolio"]
    p4 = p4_summary["metrics_oof"]["p4_counterfactual_slot_router"]
    action_counts = {
        action: oof_actions.count(action)
        for action in ("abstain", "portfolio@2", "portfolio@3")
    }
    coverage = 1.0 - action_counts["abstain"] / len(rows)
    cold_rows = [row for row in rows if row["is_cold"]]
    cumulative_hits = {
        str(size): sum(row["target"] in row["portfolio_candidates"][:size] for row in cold_rows)
        for size in (1, 2, 3)
    }
    admitted_cold_hits = sum(
        row["is_cold"] and action != "abstain"
        and row["target"] in row["portfolio_candidates"][:action_size(action)]
        for row, action in zip(rows, oof_actions)
    )
    candidate_report = {
        "n_cold_users": len(cold_rows),
        "cumulative_target_hits": cumulative_hits,
        "cumulative_recall": {
            size: count / len(cold_rows) for size, count in cumulative_hits.items()
        },
        "top3_vs_top1_ratio": cumulative_hits["3"] / max(cumulative_hits["1"], 1),
        "admitted_cold_target_hits": admitted_cold_hits,
        "admitted_hits_vs_top1_ratio": admitted_cold_hits / max(cumulative_hits["1"], 1),
    }
    predictions: list[dict] = []
    for row, action, score in zip(rows, oof_actions, oof_scores):
        ranking = ranking_for_action(row, action)
        if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
            raise RuntimeError(f"Invalid P6 ranking for {row['user_id']}")
        predictions.append({
            "user_id": row["user_id"],
            "fold": row["fold"],
            "target": row["target"],
            "is_cold": row["is_cold"],
            "portfolio_candidates": row["portfolio_candidates"],
            "selected_action": action,
            "predicted_utilities": score["utilities"],
            "selected_portfolio_contains_target": (
                action != "abstain"
                and row["target"] in row["portfolio_candidates"][:action_size(action)]
            ),
            "p6_top50": ranking[:50],
        })

    gates = {
        "candidate_top3_recall_ge_2_5x_top1": candidate_report["top3_vs_top1_ratio"] >= 2.5,
        "all_outer_folds_train_feasible": all_folds_feasible,
        "warm_ndcg10_ge_0_97x_v0": p6["warm"]["ndcg@10"] >= 0.97 * v0["warm"]["ndcg@10"],
        "cold_ndcg10_ge_2x_v0": p6["cold"]["ndcg@10"] >= 2.0 * v0["cold"]["ndcg@10"],
        "cold_hit10_ge_2x_v0": p6["cold"]["hit@10"] >= 2.0 * v0["cold"]["hit@10"],
        "cold_ndcg10_gt_p4_oof": p6["cold"]["ndcg@10"] > p4["cold"]["ndcg@10"],
        "cold_hit10_gt_p4_oof": p6["cold"]["hit@10"] > p4["cold"]["hit@10"],
        "all_ndcg10_gt_p4_oof": p6["all"]["ndcg@10"] > p4["all"]["ndcg@10"],
        "coverage_ge_0_20_lt_0_80": 0.20 <= coverage < 0.80,
        "all_interventions_are_multi_candidate": all(
            action == "abstain" or action_size(action) in PORTFOLIO_SIZES
            for action in oof_actions
        ),
        "catalog_outputs_unique": all(
            len(row["p6_top50"]) == len(set(row["p6_top50"]))
            and set(row["p6_top50"]) <= catalog for row in predictions
        ),
        "outer_fold_assignment_audited": all(
            row["fold"] == stable_fold(row["user_id"], args.folds) for row in rows
        ),
        "validation_only": True,
    }
    verdict = (
        "PASS_TO_R2_P6_NEW_DOMAIN_CONFIRMATION_DISCUSSION"
        if all(gates.values()) else "FAIL_STOP_R2_P6"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "evaluation_status": config["evaluation_status"],
        "metrics": metrics,
        "p4_oof_reference": p4,
        "gates": gates,
        "candidate_report": candidate_report,
        "diagnostics": {
            "intervention_coverage": coverage,
            "action_counts": action_counts,
        },
        "fold_reports": fold_reports,
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "policy.json", {
        "fold_policies": [
            {"fold": row["fold"], "selected": row["selected"]} for row in fold_reports
        ],
        "portfolio_sizes": PORTFOLIO_SIZES,
        "coverage_grid": COVERAGE_GRID,
        "test_predictions_opened": False,
    })
    with (output_dir / "predictions_validation.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} coverage={coverage:.3f} "
        f"warm={p6['warm']['ndcg@10']:.6f} cold={p6['cold']['ndcg@10']:.6f} "
        f"all={p6['all']['ndcg@10']:.6f}", flush=True,
    )


if __name__ == "__main__":
    main()
