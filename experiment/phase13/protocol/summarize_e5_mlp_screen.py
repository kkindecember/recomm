#!/usr/bin/env python3
"""Summarize the Phase-13 E5 MLP convergence screen with frozen gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e5-history", type=Path, required=True)
    parser.add_argument("--e5-id-report", type=Path, required=True)
    parser.add_argument("--minilm-history", type=Path, required=True)
    parser.add_argument("--minilm-id-report", type=Path, required=True)
    parser.add_argument("--previous-e5-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-floor", type=float, default=0.95)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def epoch_record(history: dict, epoch: int) -> dict:
    matches = [row for row in history["history"] if row["epoch"] == epoch]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one epoch={epoch} record")
    return matches[0]


def main() -> None:
    args = parse_args()
    e5_history = load_json(args.e5_history)
    e5_ids = load_json(args.e5_id_report)
    minilm_history = load_json(args.minilm_history)
    minilm_ids = load_json(args.minilm_id_report)
    previous_e5_history = load_json(args.previous_e5_history)

    e5_best = float(e5_history["best_val_avg_acc"])
    minilm_best = float(minilm_history["best_val_avg_acc"])
    threshold = args.relative_floor * minilm_best
    e5_dup = int(e5_ids["input_collision"]["duplicate_excess"])
    minilm_dup = int(minilm_ids["input_collision"]["duplicate_excess"])
    e5_epoch_200 = epoch_record(e5_history, 200)
    previous_epoch_200 = epoch_record(previous_e5_history, 200)

    if e5_best >= threshold:
        verdict = "PASS_TO_SMOKE"
        reason = "E5 validation accuracy reached the frozen 95%-of-MiniLM gate."
    elif e5_dup > minilm_dup:
        verdict = "FAIL_STOP_E5"
        reason = (
            "E5 validation accuracy missed the frozen gate and its raw ID collision "
            "count remained worse than MiniLM."
        )
    else:
        verdict = "REVIEW"
        reason = (
            "E5 validation accuracy missed the frozen gate, but collision behavior did "
            "not satisfy the predeclared stop condition."
        )

    n_epochs = int(e5_history["args"]["epochs"])
    best_epoch = int(e5_history["best_epoch"])
    summary = {
        "experiment": "Phase-13 Toys E5 MLP convergence screen",
        "protocol": {
            "from_scratch": True,
            "epochs": n_epochs,
            "lr": float(e5_history["args"]["lr"]),
            "batch_size": int(e5_history["args"]["batch_size"]),
            "seed": int(e5_history["args"]["seed"]),
            "relative_floor": args.relative_floor,
            "gram_training_run": False,
        },
        "reference_minilm": {
            "best_epoch": int(minilm_history["best_epoch"]),
            "best_val_avg_acc": minilm_best,
            "raw_id_duplicate_excess": minilm_dup,
        },
        "gate": {
            "minimum_val_avg_acc": threshold,
            "collision_stop_reference": minilm_dup,
        },
        "e5_result": {
            "best_epoch": best_epoch,
            "best_val_avg_acc": e5_best,
            "best_at_epoch_limit": best_epoch == n_epochs,
            "epoch_200_val_avg_acc": float(e5_epoch_200["val_avg_acc"]),
            "previous_run_epoch_200_val_avg_acc": float(
                previous_epoch_200["val_avg_acc"]
            ),
            "epoch_200_reproduced": abs(
                float(e5_epoch_200["val_avg_acc"])
                - float(previous_epoch_200["val_avg_acc"])
            )
            < 1e-12,
            "raw_id_duplicate_excess": e5_dup,
            "collision_safe_modified_cold_items": int(e5_ids["n_cold_modified"]),
            "collision_safe_modified_cold_rate": float(e5_ids["cold_modified_rate"]),
            "output_duplicate_excess": int(
                e5_ids["output_collision"]["duplicate_excess"]
            ),
        },
        "verdict": verdict,
        "reason": reason,
    }

    if summary["e5_result"]["output_duplicate_excess"] != 0:
        raise AssertionError("Collision-safe output is not globally unique")
    if not summary["e5_result"]["epoch_200_reproduced"]:
        raise AssertionError("The deterministic epoch-200 checkpoint did not reproduce")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[screen] verdict={verdict} best_epoch={best_epoch} "
        f"best_val={e5_best:.9f} gate={threshold:.9f} raw_duplicates={e5_dup}"
    )


if __name__ == "__main__":
    main()
