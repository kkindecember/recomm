#!/usr/bin/env python3
"""Apply the frozen warm-only gate for the Toys BGE encoder screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-floor", type=float, default=0.995)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def select(candidate: dict, baseline: dict, relative_floor: float) -> dict:
    candidate_val = float(candidate["best_val_avg_acc"])
    baseline_val = float(baseline["best_val_avg_acc"])
    threshold = baseline_val * relative_floor
    advance = candidate_val >= threshold
    return {
        "experiment": "Phase-13 Toys BGE-large-en-v1.5 warm-only encoder screen",
        "cold_diagnostic_computed": False,
        "protocol": {
            "encoder": "BAAI/bge-large-en-v1.5",
            "pooling": "cls",
            "l2_normalized": True,
            "text_prefix": "",
            "bridge": "one-layer independent heads",
            "epochs": int(candidate["args"]["epochs"]),
            "seed": int(candidate["args"]["seed"]),
            "gram_training_run": False,
        },
        "reference_minilm": {
            "best_epoch": int(baseline["best_epoch"]),
            "best_val_avg_acc": baseline_val,
        },
        "candidate": {
            "best_epoch": int(candidate["best_epoch"]),
            "best_val_avg_acc": candidate_val,
            "best_at_epoch_limit": int(candidate["best_epoch"])
            == int(candidate["args"]["epochs"]),
        },
        "warm_gate": {
            "relative_floor": relative_floor,
            "minimum_val_avg_acc": threshold,
            "checks": {"validation_drop_at_most_0_5pct": advance},
        },
        "advance_to_cold": advance,
        "verdict": "PASS_WARM_TO_COLD" if advance else "FAIL_WARM_GATE",
    }


def main() -> None:
    args = parse_args()
    result = select(
        load_json(args.candidate_history),
        load_json(args.baseline_history),
        args.relative_floor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[bge-warm] verdict={result['verdict']} "
        f"val={result['candidate']['best_val_avg_acc']:.9f} "
        f"gate={result['warm_gate']['minimum_val_avg_acc']:.9f}"
    )


if __name__ == "__main__":
    main()
