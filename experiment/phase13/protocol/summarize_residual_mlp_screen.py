#!/usr/bin/env python3
"""Apply frozen gates to the MiniLM residual-bridge pre-GRAM screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hierarchical_id_utils import read_id_file, read_item_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--candidate-assigned-id", type=Path, required=True)
    parser.add_argument("--candidate-id-report", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--baseline-assigned-id", type=Path, required=True)
    parser.add_argument("--baseline-id-report", type=Path, required=True)
    parser.add_argument("--source-id", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-relative-gain", type=float, default=0.02)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def cold_metrics(source_path: Path, assigned_path: Path, cold_path: Path) -> dict:
    source = read_id_file(source_path)
    assigned = read_id_file(assigned_path)
    cold_items = read_item_set(cold_path)
    n_items = len(cold_items)
    n_levels = len(next(iter(source.values())))
    missing = cold_items - source.keys() | cold_items - assigned.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} cold items from ID maps")

    position_accuracy = [
        sum(assigned[item][level] == source[item][level] for item in cold_items)
        / n_items
        for level in range(n_levels)
    ]
    prefix_accuracy = [
        sum(
            assigned[item][:prefix_length] == source[item][:prefix_length]
            for item in cold_items
        )
        / n_items
        for prefix_length in range(1, n_levels + 1)
    ]
    return {
        "n_cold": n_items,
        "n_levels": n_levels,
        "position_accuracy": position_accuracy,
        "macro_position_accuracy": sum(position_accuracy) / n_levels,
        "prefix_accuracy": prefix_accuracy,
        "exact_path_accuracy": prefix_accuracy[-1],
    }


def main() -> None:
    args = parse_args()
    candidate_history = load_json(args.candidate_history)
    baseline_history = load_json(args.baseline_history)
    candidate_id_report = load_json(args.candidate_id_report)
    baseline_id_report = load_json(args.baseline_id_report)
    candidate_cold = cold_metrics(
        args.source_id, args.candidate_assigned_id, args.cold_items
    )
    baseline_cold = cold_metrics(
        args.source_id, args.baseline_assigned_id, args.cold_items
    )

    candidate_val = float(candidate_history["best_val_avg_acc"])
    baseline_val = float(baseline_history["best_val_avg_acc"])
    validation_gate = baseline_val * (1.0 + args.validation_relative_gain)
    candidate_duplicates = int(
        candidate_id_report["input_collision"]["duplicate_excess"]
    )
    baseline_duplicates = int(
        baseline_id_report["input_collision"]["duplicate_excess"]
    )

    checks = {
        "validation_gain_at_least_2pct": candidate_val >= validation_gate,
        "cold_macro_strictly_better": candidate_cold["macro_position_accuracy"]
        > baseline_cold["macro_position_accuracy"],
        "cold_prefix2_strictly_better": candidate_cold["prefix_accuracy"][1]
        > baseline_cold["prefix_accuracy"][1],
        "cold_prefix3_strictly_better": candidate_cold["prefix_accuracy"][2]
        > baseline_cold["prefix_accuracy"][2],
        "cold_exact_path_strictly_better": candidate_cold["exact_path_accuracy"]
        > baseline_cold["exact_path_accuracy"],
        "raw_duplicate_excess_not_worse": candidate_duplicates <= baseline_duplicates,
        "collision_safe_output_unique": int(
            candidate_id_report["output_collision"]["duplicate_excess"]
        )
        == 0,
    }
    if all(checks.values()):
        verdict = "PASS_TO_SMOKE"
        reason = "Residual bridge passed every frozen validation, cold-ID, and collision gate."
    elif candidate_val > baseline_val:
        verdict = "REVIEW"
        reason = "Validation improved, but at least one frozen cold-ID/collision gate failed."
    else:
        verdict = "FAIL_STOP_RESIDUAL"
        reason = "Residual bridge did not improve the MiniLM validation reference."

    summary = {
        "experiment": "Phase-13 Toys MiniLM 2-layer residual MLP pre-GRAM screen",
        "protocol": {
            "architecture": "384->768->384 GELU residual + LayerNorm + independent heads",
            "from_scratch": True,
            "gram_training_run": False,
            "validation_relative_gain_gate": args.validation_relative_gain,
        },
        "baseline": {
            "best_val_avg_acc": baseline_val,
            "cold_id_metrics": baseline_cold,
            "raw_id_duplicate_excess": baseline_duplicates,
        },
        "candidate": {
            "best_epoch": int(candidate_history["best_epoch"]),
            "best_val_avg_acc": candidate_val,
            "best_at_epoch_limit": int(candidate_history["best_epoch"])
            == int(candidate_history["args"]["epochs"]),
            "cold_id_metrics": candidate_cold,
            "raw_id_duplicate_excess": candidate_duplicates,
            "collision_safe_modified_cold_items": int(
                candidate_id_report["n_cold_modified"]
            ),
            "collision_safe_modified_cold_rate": float(
                candidate_id_report["cold_modified_rate"]
            ),
        },
        "gate": {"minimum_val_avg_acc": validation_gate, "checks": checks},
        "verdict": verdict,
        "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[residual-screen] verdict={verdict} best_val={candidate_val:.9f} "
        f"gate={validation_gate:.9f} cold_exact={candidate_cold['exact_path_accuracy']:.9f}"
    )


if __name__ == "__main__":
    main()
