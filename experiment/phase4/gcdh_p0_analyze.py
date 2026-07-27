#!/usr/bin/env python3
"""Apply the locked GCDH P0 paired effect and resource gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_rows(path: Path) -> dict[str, dict]:
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            user = row["user_id"]
            if user in result:
                raise ValueError(f"duplicate validation user {user}")
            result[user] = row
    return result


def paired_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, relative: bool, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    values = np.empty(5000, dtype=np.float64)
    for start in range(0, 5000, 100):
        size = min(100, 5000 - start)
        indices = rng.integers(0, n, size=(size, n))
        old = baseline[indices].mean(axis=1)
        new = candidate[indices].mean(axis=1)
        values[start : start + size] = new / old - 1.0 if relative else new - old
    return [float(value) for value in np.percentile(values, (2.5, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    results = {}
    gate_rows = []
    for offset, dataset in enumerate(config["datasets"]):
        root = args.input_root / dataset
        c0_train = load_json(root / "C0/training_summary.json")
        c1_train = load_json(root / "C1/training_summary.json")
        c0_summary = load_json(root / "C0/validation_summary.json")
        c1_summary = load_json(root / "C1/validation_summary.json")
        c0_rows = load_rows(root / "C0/validation_per_user.csv")
        c1_rows = load_rows(root / "C1/validation_per_user.csv")
        if set(c0_rows) != set(c1_rows):
            raise ValueError(f"{dataset}: validation user mismatch")
        if (
            c0_train["train_user_sha256"] != c1_train["train_user_sha256"]
            or c0_train["samples"] != c1_train["samples"]
            or c0_train["optimizer_updates"] != c1_train["optimizer_updates"]
        ):
            raise ValueError(f"{dataset}: matched training integrity failure")
        users = sorted(c0_rows)
        b_ndcg = np.asarray(
            [float(c0_rows[user]["gram_NDCG@10"]) for user in users]
        )
        c_ndcg = np.asarray(
            [float(c1_rows[user]["final_NDCG@10"]) for user in users]
        )
        b_recall = np.asarray(
            [float(c0_rows[user]["gram_Recall@10"]) for user in users]
        )
        c_recall = np.asarray(
            [float(c1_rows[user]["final_Recall@10"]) for user in users]
        )
        tail_indices = np.asarray(
            [index for index, user in enumerate(users) if c0_rows[user]["target_group"] == "tail"]
        )
        ndcg_gain = float(c_ndcg.mean() / b_ndcg.mean() - 1.0)
        recall_gain = float(c_recall.mean() - b_recall.mean())
        tail_gain = float(
            c_ndcg[tail_indices].mean() / b_ndcg[tail_indices].mean() - 1.0
        )
        c1_overall = c1_summary["groups"]["overall"]
        union_gain = float(
            c1_overall["union_hit50"] - c1_overall["gram_Recall@50"]
        )
        peak_increase = float(
            c1_train["peak_reserved_mib"] / c0_train["peak_reserved_mib"] - 1.0
        )
        checks = {
            "overall_ndcg10_relative_gain": ndcg_gain
            >= config["gates"]["overall_ndcg10_relative_gain_min"],
            "overall_recall10_absolute_gain": recall_gain
            >= config["gates"]["overall_recall10_absolute_gain_min"],
            "tail_ndcg10_relative_gain": tail_gain
            >= config["gates"]["tail_ndcg10_relative_gain_min"],
            "union_recall50_absolute_gain": union_gain
            >= config["gates"]["union_recall50_absolute_gain_min"],
            "peak_reserved_relative_increase": peak_increase
            <= config["gates"]["peak_reserved_relative_increase_max"],
        }
        results[dataset] = {
            "users": len(users),
            "tail_users": int(len(tail_indices)),
            "C0_gram_ndcg@10": float(b_ndcg.mean()),
            "C1_final_ndcg@10": float(c_ndcg.mean()),
            "overall_ndcg10_relative_gain": ndcg_gain,
            "overall_recall10_absolute_gain": recall_gain,
            "tail_ndcg10_relative_gain": tail_gain,
            "union_recall50_absolute_gain": union_gain,
            "peak_reserved_relative_increase": peak_increase,
            "bootstrap": {
                "overall_ndcg10_relative_gain_ci95": paired_bootstrap(
                    b_ndcg, c_ndcg, True, int(config["seed"]) + offset
                ),
                "overall_recall10_absolute_gain_ci95": paired_bootstrap(
                    b_recall, c_recall, False, int(config["seed"]) + 10 + offset
                ),
                "tail_ndcg10_relative_gain_ci95": paired_bootstrap(
                    b_ndcg[tail_indices],
                    c_ndcg[tail_indices],
                    True,
                    int(config["seed"]) + 20 + offset,
                ),
            },
        }
        gate_rows.append(
            {"dataset": dataset, "checks": checks, "pass": all(checks.values())}
        )
    passed = all(row["pass"] for row in gate_rows)
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": (
            "GCDH_P1_FULL_TRAINING_ALLOWED"
            if passed
            else "STOP_GCDH_NO_DUAL_HEAD_EFFECT"
        ),
        "results": results,
        "gate_rows": gate_rows,
        "integrity": {
            "matched_training": True,
            "paired_validation_users": True,
            "test_data_read": False,
            "both_datasets_required": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
