#!/usr/bin/env python3
"""HBTR-B0 deterministic, CPU-only beam hierarchy feasibility diagnostic.

Only locked validation predictions are read. Test predictions are not accepted.
Popularity is computed from sequence[:-2], excluding validation and test targets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from s0_offline_diagnostics import (
    decode_item_ids,
    head_items,
    read_predictions,
    read_sequences,
    sha256,
    training_popularity,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "artifacts/phase3/hbtr_b0"
DATASETS = {
    "Toys": ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv",
    "Beauty": ROOT / "GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv",
}


def read_raw_semantic_ids(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            item, raw_id = line.rstrip("\n").split(" ", 1)
            tokens = tuple(token for token in raw_id.split("|") if token)
            if not tokens:
                raise ValueError(f"Empty semantic ID at {path}:{line_number}")
            result[item] = tokens
    return result


def common_prefix_depth(left: Sequence[str], right: Sequence[str]) -> int:
    depth = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        depth += 1
    return depth


def rank_of(items: Sequence[str], target: str) -> int | None:
    try:
        return items.index(target) + 1
    except ValueError:
        return None


def ndcg(rank: int | None, k: int = 10) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_group(rows: Iterable[dict]) -> dict:
    values = list(rows)
    n = len(values)
    hit10 = sum(row["rank"] is not None and row["rank"] <= 10 for row in values)
    hit50 = sum(row["rank"] is not None and row["rank"] <= 50 for row in values)
    miss10_hit50 = [row for row in values if row["rank"] is not None and 11 <= row["rank"] <= 50]
    shared = sum(row["max_top10_prefix_depth"] >= 1 for row in miss10_hit50)
    baseline_ndcg = sum(row["ndcg@10"] for row in values) / n if n else None
    oracle_ndcg = hit50 / n if n else None
    return {
        "n": n,
        "recall@10": rate(hit10, n),
        "recall@50": rate(hit50, n),
        "recall_gap_absolute": rate(hit50 - hit10, n),
        "miss10_hit50_n": len(miss10_hit50),
        "shared_prefix_n": shared,
        "shared_prefix_rate_in_miss10_hit50": rate(shared, len(miss10_hit50)),
        "mean_max_top10_prefix_depth_in_miss10_hit50": (
            sum(row["max_top10_prefix_depth"] for row in miss10_hit50) / len(miss10_hit50)
            if miss10_hit50
            else None
        ),
        "baseline_ndcg@10": baseline_ndcg,
        "oracle_promote_target_ndcg@10": oracle_ndcg,
        "oracle_ndcg@10_relative_headroom": (
            oracle_ndcg / baseline_ndcg - 1.0
            if baseline_ndcg not in (None, 0.0) and oracle_ndcg is not None
            else None
        ),
    }


def analyze(dataset: str, predictions: Path) -> dict:
    started = time.time()
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    index_paths = sorted(dataset_dir.glob("item_generative_indexing_hierarchy_*.txt"))
    if len(index_paths) != 1:
        raise ValueError(f"Expected one semantic index for {dataset}, got {index_paths}")
    index_path = index_paths[0]
    sequence_path = dataset_dir / "user_sequence.txt"
    inputs = {
        "predictions": predictions,
        "user_sequence": sequence_path,
        "item_index": index_path,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    _, text_to_item = decode_item_ids(index_path, "t5-small", True)
    prediction_rows, prediction_audit = read_predictions(predictions, text_to_item)
    sequences = read_sequences(sequence_path)
    semantic_ids = read_raw_semantic_ids(index_path)
    heads = head_items(training_popularity(sequences))

    rows = []
    target_mismatches = []
    missing_users = []
    for row in prediction_rows:
        sequence = sequences.get(row["user"])
        if sequence is None:
            missing_users.append(row["user"])
            continue
        expected_target = sequence[-2]
        if row["gold"] != expected_target:
            target_mismatches.append((row["user"], row["gold"], expected_target))
        target_sid = semantic_ids[expected_target]
        top10 = [item for item in row["pred_items"][:10] if item is not None]
        max_prefix = max(
            (common_prefix_depth(target_sid, semantic_ids[item]) for item in top10 if item != expected_target),
            default=0,
        )
        target_rank = rank_of(row["pred_items"][:50], expected_target)
        rows.append(
            {
                "group": "head" if expected_target in heads else "tail",
                "rank": target_rank,
                "ndcg@10": ndcg(target_rank),
                "max_top10_prefix_depth": max_prefix,
            }
        )
    if missing_users or target_mismatches:
        raise ValueError(
            f"Lineage failure for {dataset}: missing={missing_users[:3]}, "
            f"mismatch={target_mismatches[:3]}"
        )

    groups = {
        "overall": summarize_group(rows),
        "head": summarize_group(row for row in rows if row["group"] == "head"),
        "tail": summarize_group(row for row in rows if row["group"] == "tail"),
    }
    prefix_histogram = Counter(
        row["max_top10_prefix_depth"]
        for row in rows
        if row["rank"] is not None and 11 <= row["rank"] <= 50
    )
    gate = {
        "recall_gap_at_least_0.05": groups["overall"]["recall_gap_absolute"] >= 0.05,
        "tail_miss10_hit50_at_least_200": groups["tail"]["miss10_hit50_n"] >= 200,
        "shared_prefix_rate_at_least_0.25": (
            groups["overall"]["shared_prefix_rate_in_miss10_hit50"] is not None
            and groups["overall"]["shared_prefix_rate_in_miss10_hit50"] >= 0.25
        ),
        "oracle_ndcg_headroom_at_least_0.05": (
            groups["overall"]["oracle_ndcg@10_relative_headroom"] is not None
            and groups["overall"]["oracle_ndcg@10_relative_headroom"] >= 0.05
        ),
    }
    result = {
        "dataset": dataset,
        "mode": "validation_only",
        "groups": groups,
        "prefix_depth_histogram_miss10_hit50": dict(sorted(prefix_histogram.items())),
        "gate": {**gate, "passed": all(gate.values())},
        "audit": {
            **prediction_audit,
            "missing_users": len(missing_users),
            "target_mismatches": len(target_mismatches),
            "item_count": len(semantic_ids),
            "head_item_count": len(heads),
            "input_sha256": {name: sha256(path) for name, path in inputs.items()},
            "wall_time_seconds": time.time() - started,
        },
    }
    out_dir = OUTPUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diagnostic_summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def main() -> int:
    started = time.time()
    results = {dataset: analyze(dataset, path) for dataset, path in DATASETS.items()}
    diagnostic_pass = all(result["gate"]["passed"] for result in results.values())
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "hbtr_b0_diag_v1",
            "design_status": "PREREGISTERED_BEFORE_DIAGNOSTIC_READOUT",
        },
        "diagnostic_decision": "GO" if diagnostic_pass else "STOP",
        "datasets": results,
        "wall_time_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "test_predictions_read": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "diagnostic_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (OUTPUT_ROOT / "diagnostic_metrics.csv").open("w", newline="") as handle:
        fieldnames = [
            "dataset", "group", "n", "recall@10", "recall@50", "recall_gap_absolute",
            "miss10_hit50_n", "shared_prefix_rate_in_miss10_hit50",
            "mean_max_top10_prefix_depth_in_miss10_hit50", "baseline_ndcg@10",
            "oracle_promote_target_ndcg@10", "oracle_ndcg@10_relative_headroom",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, result in results.items():
            for group, metrics in result["groups"].items():
                writer.writerow({"dataset": dataset, "group": group, **{k: metrics[k] for k in fieldnames[2:]}})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
