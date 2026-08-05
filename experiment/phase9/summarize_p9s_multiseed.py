#!/usr/bin/env python3
"""Aggregate preregistered Toys/Beauty three-seed PCRF validation results."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(rows):
    by_dataset = {}
    for dataset in ("Toys", "Beauty"):
        group = [row for row in rows if row["dataset"] == dataset]
        values = [row["hit10_delta"] for row in group]
        by_dataset[dataset] = {
            "seeds": [row["seed"] for row in group],
            "Hit@10_delta_mean": statistics.mean(values),
            "Hit@10_delta_median": statistics.median(values),
            "Hit@10_delta_sample_std": statistics.stdev(values),
            "minimum_Hit@10_delta": min(values),
        }
    checks = {
        "four_new_item_heads_passed": all(row["item_gate"] == "passed" for row in rows if row["seed"] != 2023),
        "all_six_Hit10_deltas_positive": all(row["hit10_delta"] > 0 for row in rows),
        "both_dataset_median_Hit10_delta_at_least_0.002": all(value["Hit@10_delta_median"] >= 0.002 for value in by_dataset.values()),
        "all_six_NDCG10_non_degradation": all(row["ndcg10_delta"] >= 0 for row in rows),
        "all_six_tail_Hit10_non_degradation": all(row["tail_hit10_delta"] >= 0 for row in rows),
        "all_six_Hit50_identity": all(abs(row["hit50_delta"]) <= 1e-12 for row in rows),
    }
    return by_dataset, checks


def main():
    args = parse_args()
    rows = []
    for dataset in ("Toys", "Beauty"):
        for seed in (2023, 2024, 2025):
            base = args.root / dataset / f"seed{seed}"
            validation = json.load((base / "validation/summary.json").open())
            item_summary = base / "item_head/summary.json"
            item_gate = "frozen_existing"
            if item_summary.exists():
                item_gate = json.load(item_summary.open())["scientific_gate"]["status"]
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "item_gate": item_gate,
                "hit10_delta": validation["delta"]["Hit@10"],
                "ndcg10_delta": validation["delta"]["NDCG@10"],
                "tail_hit10_delta": validation["pcrf_subgroups"]["target_tail"]["Hit@10"] - validation["baseline_subgroups"]["target_tail"]["Hit@10"],
                "hit50_delta": validation["delta"]["Hit@50"],
            })
    by_dataset, checks = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = args.output / "seed_results.tsv"
    with evidence.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment_id": "GRAM_PHASE9_P9S_MULTISEED_VALIDATION_V1",
        "status": "completed",
        "datasets": ["Toys", "Beauty"],
        "seeds": [2023, 2024, 2025],
        "test_read": False,
        "frozen_pcrf": {"lambda": 1.0, "beta": 0.5, "gamma": 1.0},
        "rows": rows,
        "by_dataset": by_dataset,
        "robustness_gate": {"status": "passed" if all(checks.values()) else "failed", "checks": checks},
        "artifacts": {"seed_results_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()},
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"robustness_gate": summary["robustness_gate"], "by_dataset": by_dataset}), flush=True)


if __name__ == "__main__":
    main()
