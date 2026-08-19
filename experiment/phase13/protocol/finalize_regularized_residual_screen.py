#!/usr/bin/env python3
"""Finalize winner-only cold gates for the regularized residual screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_residual_mlp_screen import cold_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-selection", type=Path, required=True)
    parser.add_argument("--winner-assigned-id", type=Path, required=True)
    parser.add_argument("--winner-id-report", type=Path, required=True)
    parser.add_argument("--baseline-assigned-id", type=Path, required=True)
    parser.add_argument("--baseline-id-report", type=Path, required=True)
    parser.add_argument("--source-id", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    selection = load_json(args.warm_selection)
    if not selection["advance_to_cold"]:
        raise ValueError("Warm selection did not authorize cold evaluation")
    winner_cold = cold_metrics(
        args.source_id, args.winner_assigned_id, args.cold_items
    )
    baseline_cold = cold_metrics(
        args.source_id, args.baseline_assigned_id, args.cold_items
    )
    winner_ids = load_json(args.winner_id_report)
    baseline_ids = load_json(args.baseline_id_report)

    checks = {
        "prefix3_gain_at_least_3pct": winner_cold["prefix_accuracy"][2]
        >= baseline_cold["prefix_accuracy"][2] * 1.03,
        "exact_path_gain_at_least_5pct": winner_cold["exact_path_accuracy"]
        >= baseline_cold["exact_path_accuracy"] * 1.05,
        "macro_drop_at_most_0_5pct": winner_cold["macro_position_accuracy"]
        >= baseline_cold["macro_position_accuracy"] * 0.995,
        "prefix2_drop_at_most_0_5pct": winner_cold["prefix_accuracy"][1]
        >= baseline_cold["prefix_accuracy"][1] * 0.995,
        "raw_duplicate_excess_not_worse": int(
            winner_ids["input_collision"]["duplicate_excess"]
        )
        <= int(baseline_ids["input_collision"]["duplicate_excess"]),
        "collision_safe_output_unique": int(
            winner_ids["output_collision"]["duplicate_excess"]
        )
        == 0,
    }
    passed = all(checks.values())
    result = {
        "experiment": "Phase-13 Toys MiniLM regularized residual pre-GRAM screen",
        "protocol": {
            "gram_training_run": False,
            "cold_evaluated_for_winner_only": True,
        },
        "warm_selection": selection,
        "baseline_cold_id_metrics": baseline_cold,
        "winner_cold_id_metrics": winner_cold,
        "winner_collision": {
            "raw_duplicate_excess": int(
                winner_ids["input_collision"]["duplicate_excess"]
            ),
            "modified_cold_items": int(winner_ids["n_cold_modified"]),
            "modified_cold_rate": float(winner_ids["cold_modified_rate"]),
        },
        "cold_gate": {"checks": checks},
        "verdict": "PASS_TO_SMOKE" if passed else "FAIL_STOP_REGULARIZED_RESIDUAL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[regularized-final] verdict={result['verdict']} "
        f"winner={selection['winner']['name']} "
        f"prefix3={winner_cold['prefix_accuracy'][2]:.9f} "
        f"exact={winner_cold['exact_path_accuracy']:.9f}"
    )


if __name__ == "__main__":
    main()
