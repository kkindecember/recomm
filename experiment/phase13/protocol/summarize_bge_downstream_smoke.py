#!/usr/bin/env python3
"""Compare matched MiniLM/BGE collision-safe Toys diagnostic smokes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_cold_warm import (
    METRIC_NAMES,
    load_cold_items,
    load_user_target_map,
    parse_predictions_tsv,
)
from make_collision_safe_ids import read_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--candidate-safe-id", type=Path, required=True)
    parser.add_argument("--baseline-safe-id", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def id_map(path: Path) -> dict[str, list[str]]:
    return {item: list(tokens) for item, tokens in read_rows(path)}


def paired_summary(candidate: dict[str, list[float]],
                   baseline: dict[str, list[float]], users: set[str]) -> dict:
    metric = {name: index for index, name in enumerate(METRIC_NAMES)}
    result: dict[str, object] = {"n": len(users)}
    for name in ("hit@10", "ndcg@10"):
        idx = metric[name]
        candidate_values = [candidate[user][idx] for user in users]
        baseline_values = [baseline[user][idx] for user in users]
        differences = [c - b for c, b in zip(candidate_values, baseline_values)]
        result[name] = {
            "baseline_sum": sum(baseline_values),
            "candidate_sum": sum(candidate_values),
            "mean_delta": sum(differences) / len(users) if users else None,
            "candidate_better_users": sum(delta > 0 for delta in differences),
            "tied_users": sum(delta == 0 for delta in differences),
            "candidate_worse_users": sum(delta < 0 for delta in differences),
        }
    return result


def directional_verdict(candidate: dict, baseline: dict) -> tuple[str, str]:
    candidate_cold = candidate["cold"]
    baseline_cold = baseline["cold"]
    warm_guard = (
        baseline["warm"]["ndcg@10"] == 0
        or candidate["warm"]["ndcg@10"] >= 0.5 * baseline["warm"]["ndcg@10"]
    )
    candidate_key = (candidate_cold["hit@10"], candidate_cold["ndcg@10"])
    baseline_key = (baseline_cold["hit@10"], baseline_cold["ndcg@10"])
    if candidate_key > baseline_key and warm_guard:
        return (
            "DIRECTIONAL_SUPPORT_FOR_FULL_GATE_DISCUSSION",
            "BGE improved the lexicographic cold hit@10/NDCG@10 smoke outcome without catastrophic warm regression.",
        )
    if candidate_key < baseline_key or not warm_guard:
        return (
            "DIRECTIONAL_HARM_STOP",
            "BGE worsened the cold smoke outcome or triggered the warm catastrophic-regression guard.",
        )
    return (
        "INCONCLUSIVE_TIE",
        "The 45-cold-user smoke tied on hit@10 and NDCG@10.",
    )


def main() -> None:
    args = parse_args()
    candidate_metrics = load_json(args.candidate_metrics)
    baseline_metrics = load_json(args.baseline_metrics)
    candidate_rows = dict(parse_predictions_tsv(Path(candidate_metrics["predictions_tsv"])))
    baseline_rows = dict(parse_predictions_tsv(Path(baseline_metrics["predictions_tsv"])))
    if set(candidate_rows) != set(baseline_rows):
        raise ValueError("Candidate and baseline smoke user sets do not match")
    if len(candidate_rows) != 100:
        raise ValueError(f"Expected 100 matched smoke users, got {len(candidate_rows)}")

    dataset_dir = args.dataset_dir.resolve()
    targets = load_user_target_map(dataset_dir / "user_sequence.txt", "test")
    cold_items = load_cold_items(dataset_dir / "cold_split_meta" / "cold_items.txt")
    baseline_ids = id_map(args.baseline_safe_id)
    candidate_ids = id_map(args.candidate_safe_id)
    cold_users = {user for user in candidate_rows if targets[user] in cold_items}
    warm_users = set(candidate_rows) - cold_users
    if len(cold_users) != 45 or len(warm_users) != 55:
        raise ValueError(
            f"Expected matched 45 cold/55 warm users, got {len(cold_users)}/{len(warm_users)}"
        )

    transitions = {
        "neither_suffixed": 0,
        "baseline_only_suffixed": 0,
        "candidate_only_suffixed": 0,
        "both_suffixed": 0,
    }
    for user in cold_users:
        item = targets[user]
        baseline_suffix = len(baseline_ids[item]) > 5
        candidate_suffix = len(candidate_ids[item]) > 5
        key = {
            (False, False): "neither_suffixed",
            (True, False): "baseline_only_suffixed",
            (False, True): "candidate_only_suffixed",
            (True, True): "both_suffixed",
        }[(baseline_suffix, candidate_suffix)]
        transitions[key] += 1

    verdict, reason = directional_verdict(candidate_metrics, baseline_metrics)
    result = {
        "experiment": "Phase-13 BGE collision-safe downstream diagnostic smoke",
        "protocol": {
            "matched_seed": 2023,
            "train_epochs": 1,
            "debug_train_users": 100,
            "debug_test_users": 100,
            "cold_test_users": 45,
            "formal_experiment": False,
            "efficacy_gate_consumed": False,
            "interpretation_limit": "Directional pipeline evidence only; not a powered efficacy comparison.",
        },
        "baseline_minilm": baseline_metrics,
        "candidate_bge": candidate_metrics,
        "paired": {
            "all": paired_summary(candidate_rows, baseline_rows, set(candidate_rows)),
            "warm": paired_summary(candidate_rows, baseline_rows, warm_users),
            "cold": paired_summary(candidate_rows, baseline_rows, cold_users),
        },
        "cold_target_suffix_transition": transitions,
        "decision_rule": {
            "primary": "lexicographic (cold hit@10, cold ndcg@10)",
            "warm_guard": "candidate warm ndcg@10 >= 0.5 * baseline warm ndcg@10",
            "no_automatic_formal_run": True,
        },
        "verdict": verdict,
        "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[bge-smoke] verdict={verdict} "
        f"cold_hit10={candidate_metrics['cold']['hit@10']:.9f} "
        f"baseline={baseline_metrics['cold']['hit@10']:.9f}"
    )


if __name__ == "__main__":
    main()
