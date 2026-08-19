#!/usr/bin/env python3
"""Apply the frozen P0 gates to BGE capacity-aware cold IDs."""
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
    parser.add_argument("--assignment-report", type=Path, required=True)
    parser.add_argument("--candidate-id", type=Path, required=True)
    parser.add_argument("--raw-bge-id", type=Path, required=True)
    parser.add_argument("--baseline-minilm-id", type=Path, required=True)
    parser.add_argument("--source-id", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.assignment_report.read_text())
    candidate = cold_metrics(args.source_id, args.candidate_id, args.cold_items)
    raw_bge = cold_metrics(args.source_id, args.raw_bge_id, args.cold_items)
    baseline = cold_metrics(args.source_id, args.baseline_minilm_id, args.cold_items)
    checks = {
        "global_unique": int(report["output_collision"]["duplicate_excess"]) == 0,
        "warm_ids_unchanged": bool(report["warm_ids_unchanged"]),
        "row_order_unchanged": bool(report["row_order_unchanged"]),
        "all_cold_fixed_length_5": int(report["cold_fixed_length"]) == 5,
        "no_appended_cold_suffix": int(report["cold_appended_suffix_count"]) == 0,
        "bge_prefix3_fully_preserved": int(
            report["cold_prefix_levels_preserved"]
        ) == 3
        and candidate["prefix_accuracy"][2] == raw_bge["prefix_accuracy"][2],
        "prefix4_not_below_minilm": candidate["prefix_accuracy"][3]
        >= baseline["prefix_accuracy"][3],
        "exact_not_below_minilm": candidate["exact_path_accuracy"]
        >= baseline["exact_path_accuracy"],
        "macro_drop_vs_minilm_at_most_0_5pct": candidate[
            "macro_position_accuracy"
        ]
        >= baseline["macro_position_accuracy"] * 0.995,
        "at_least_95pct_within_top8_per_tail_level": float(
            report["assignment_rank"]["fraction_max_rank_le_8"]
        )
        >= 0.95,
        "all_assignments_within_frozen_top16": float(
            report["assignment_rank"]["fraction_max_rank_le_16"]
        )
        == 1.0,
        "no_infeasible_group": int(report["groups"]["infeasible_groups"]) == 0,
    }
    passed = all(checks.values())
    result = {
        "experiment": "Phase-13 v1 iter2 BGE prefix-preserving capacity-aware assignment P0",
        "protocol": {
            "formal_experiment": False,
            "gram_training_run": False,
            "downstream_smoke_run": False,
            "automatic_next_stage": False,
            "selection_warning": (
                "Cold source-ID diagnostics are exploratory and cannot establish "
                "downstream recommendation efficacy."
            ),
        },
        "reference_minilm": {"cold_id_metrics": baseline},
        "raw_bge": {"cold_id_metrics": raw_bge},
        "capacity_candidate": {
            "cold_id_metrics": candidate,
            "assignment": report,
        },
        "gate": {"checks": checks},
        "verdict": (
            "PASS_TO_MEDIUM_SMOKE_DISCUSSION"
            if passed
            else "FAIL_STOP_CAPACITY_ASSIGNMENT"
        ),
        "reason": (
            "Fixed-length unique assignment passed every frozen feasibility, "
            "semantic-retention, and rank-cost check."
            if passed
            else "At least one frozen feasibility, semantic-retention, or rank-cost check failed."
        ),
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(
        f"[capacity-p0] verdict={result['verdict']} "
        f"prefix3={candidate['prefix_accuracy'][2]:.9f} "
        f"prefix4={candidate['prefix_accuracy'][3]:.9f} "
        f"exact={candidate['exact_path_accuracy']:.9f}"
    )


if __name__ == "__main__":
    main()
