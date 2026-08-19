#!/usr/bin/env python3
"""Select one residual arm using warm validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-history", type=Path, required=True)
    parser.add_argument("--dropout-history", type=Path, required=True)
    parser.add_argument("--weight-decay-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_arm(name: str, path: Path) -> dict:
    with path.open() as handle:
        history = json.load(handle)
    return {
        "name": name,
        "history_path": str(path.resolve()),
        "checkpoint_path": str((path.parent / "best.pt").resolve()),
        "best_epoch": int(history["best_epoch"]),
        "best_val_hscore": float(history["best_val_hscore"]),
        "best_val_avg_acc": float(history["best_val_avg_acc"]),
        "best_val_prefix_accuracy": history["best_val_prefix_accuracy"],
        "dropout": float(history["args"]["dropout"]),
        "weight_decay": float(history["args"]["weight_decay"]),
    }


def main() -> None:
    args = parse_args()
    arms = [
        load_arm("a0_control", args.control_history),
        load_arm("a1_dropout02", args.dropout_history),
        load_arm("a2_weight_decay1e3", args.weight_decay_history),
    ]
    control = arms[0]
    winner = max(
        arms,
        key=lambda arm: (arm["best_val_hscore"], arm["best_val_avg_acc"]),
    )
    hscore_gate = control["best_val_hscore"] * 1.02
    val_floor = control["best_val_avg_acc"] * 0.995
    checks = {
        "winner_hscore_gain_at_least_2pct": winner["best_val_hscore"]
        >= hscore_gate,
        "winner_val_avg_drop_at_most_0_5pct": winner["best_val_avg_acc"]
        >= val_floor,
    }
    advance = all(checks.values())
    result = {
        "experiment": "Phase-13 MiniLM regularized residual warm-only arm selection",
        "cold_diagnostic_computed": False,
        "selection_metric": "0.5*prefix@2+0.3*prefix@3+0.2*exact",
        "arms": arms,
        "winner": winner,
        "warm_gate": {
            "minimum_hscore": hscore_gate,
            "minimum_val_avg_acc": val_floor,
            "checks": checks,
        },
        "advance_to_cold": advance,
        "verdict": "PASS_WARM_TO_COLD" if advance else "FAIL_WARM_GATE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[warm-select] verdict={result['verdict']} winner={winner['name']} "
        f"hscore={winner['best_val_hscore']:.9f} gate={hscore_gate:.9f} "
        f"val={winner['best_val_avg_acc']:.9f}"
    )


if __name__ == "__main__":
    main()
