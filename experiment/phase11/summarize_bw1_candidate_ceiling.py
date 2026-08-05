#!/usr/bin/env python3
"""Aggregate the two preregistered BW1 validation pilots."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_decision(rows):
    if not all(row["integrity_gate"] == "passed" for row in rows):
        return "integrity_failed"
    coverage = [row["coverage_headroom"] for row in rows]
    pcrf = [row["pcrf_headroom"] for row in rows]
    if all(value < 0.005 for value in coverage):
        return "beam_coverage_saturated"
    if any(value >= 0.005 for value in coverage) and all(value >= 0 for value in pcrf) and any(value >= 0.002 for value in pcrf):
        return "candidate_expansion_eligible"
    if min(pcrf) < 0 < max(pcrf):
        return "domain_dependent"
    return "coverage_not_converted_by_frozen_pcrf"


def minimum_width_at_90_percent(summary):
    by_width = {row["width"]: row for row in summary["widths"]}
    start = by_width[50]["pcrf"]["Hit@10"]
    end = by_width[200]["pcrf"]["Hit@10"]
    if end <= start:
        return 50
    target = start + 0.9 * (end - start)
    return next(width for width in (50, 100, 200) if by_width[width]["pcrf"]["Hit@10"] >= target)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, rows = {}, []
    for dataset in ("Toys", "Beauty"):
        path = args.root / dataset / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries[dataset] = summary
        by_width = {row["width"]: row for row in summary["widths"]}
        rows.append({
            "dataset": dataset,
            "integrity_gate": summary["integrity_gate"]["status"],
            "coverage_w50": by_width[50]["candidate_recall"],
            "coverage_w100": by_width[100]["candidate_recall"],
            "coverage_w200": by_width[200]["candidate_recall"],
            "coverage_headroom": summary["headroom"]["coverage_headroom"],
            "baseline_hit10_w50": by_width[50]["baseline"]["Hit@10"],
            "baseline_hit10_w100": by_width[100]["baseline"]["Hit@10"],
            "baseline_hit10_w200": by_width[200]["baseline"]["Hit@10"],
            "pcrf_hit10_w50": by_width[50]["pcrf"]["Hit@10"],
            "pcrf_hit10_w100": by_width[100]["pcrf"]["Hit@10"],
            "pcrf_hit10_w200": by_width[200]["pcrf"]["Hit@10"],
            "pcrf_headroom": summary["headroom"]["pcrf_headroom"],
            "minimum_width_at_90pct_pcrf_gain": minimum_width_at_90_percent(summary),
        })
    decision = choose_decision(rows)
    table_path = args.output / "dataset_results.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "experiment_id": "GRAM_PHASE11_BW1_CANDIDATE_CEILING_VALIDATION_V1",
        "status": "completed",
        "test_read": False,
        "sports_read": False,
        "rows": rows,
        "decision": decision,
        "all_integrity_gates_passed": all(row["integrity_gate"] == "passed" for row in rows),
        "artifacts": {"dataset_results_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()},
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": decision, "all_integrity_gates_passed": aggregate["all_integrity_gates_passed"]}), flush=True)


if __name__ == "__main__":
    main()
