#!/usr/bin/env python3
"""Evaluate target-blind, budgeted GRAM/CF candidate unions on Toys validation."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE10 = REPO_ROOT / "experiment/phase10"
PHASE9 = REPO_ROOT / "experiment/phase9"
for directory in (PHASE10, PHASE9):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_cf1_a_candidate_union import (  # noqa: E402
    coverage,
    load_item_model,
    retrieve_cf_top50,
)
from eval_cf0_b3_beamfusion import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    DEFAULT_PREDICTIONS,
    load_cached_beams,
    load_catalog,
    load_users,
)


DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_a2_toys_budgeted_union"
REFERENCE_G50 = 0.21193076447558212
REFERENCE_U50 = 0.2666907067793118
REFERENCE_TAIL_COMPLEMENT = 0.023449612403100777


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-users", type=int, default=0)
    return parser.parse_args()


def fill_cf_only(gram_ids, cf_ids, slots):
    """Keep G50 and append up to `slots` ranked CF candidates not already in G50."""
    selected = list(gram_ids)
    seen = set(selected)
    for item in cf_ids:
        if item not in seen:
            selected.append(item)
            seen.add(item)
            if len(selected) == len(gram_ids) + slots:
                break
    return selected


def fixed_prefix_union(gram_ids, cf_ids, cutoff):
    return list(dict.fromkeys([*gram_ids, *cf_ids[:cutoff]]))


def adaptive_history_slots(history_length):
    if history_length <= 5:
        return 25
    if history_length <= 10:
        return 30
    return 40


def policy_summary(rows, key):
    sizes = np.asarray([len(row[key]) for row in rows])
    hits = [row["target"] in row[key] for row in rows]
    complementary = [
        row["target"] in row[key] and not row["hit_g50"] for row in rows
    ]
    tail_rows = [row for row in rows if row["target_frequency"] <= row["q1"]]
    tail_complementary = [
        row["target"] in row[key] and not row["hit_g50"] for row in tail_rows
    ]
    return {
        "coverage": coverage(hits),
        "complementary_not_G50": coverage(complementary),
        "tail_complementary_not_G50": coverage(tail_complementary),
        "union_size_mean": float(sizes.mean()),
        "union_size_max": int(sizes.max()),
        "fraction_le_90": float(np.mean(sizes <= 90)),
        "cf_only_scoring_total": int(sum(len(row[key]) - 50 for row in rows)),
    }


def scientific_gate(metrics, full_run):
    if not full_run:
        return {"status": "not_evaluated_smoke", "checks": {}}
    primary = metrics["policies"]["fill_cf_only_40"]
    gain_retention = (
        (primary["coverage"] - metrics["G50_coverage"])
        / (metrics["U50_coverage"] - metrics["G50_coverage"])
    )
    tail_retention = (
        primary["tail_complementary_not_G50"]
        / metrics["U50_tail_complementary_not_G50"]
    )
    checks = {
        "reference_G50_identity": math.isclose(
            metrics["G50_coverage"], REFERENCE_G50, abs_tol=1e-12
        ),
        "reference_U50_identity": math.isclose(
            metrics["U50_coverage"], REFERENCE_U50, abs_tol=1e-12
        ),
        "hard_budget_all_users": primary["fraction_le_90"] == 1.0,
        "U50_gain_retention_at_least_0.80": gain_retention >= 0.80,
        "tail_complement_retention_at_least_0.80": tail_retention >= 0.80,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed_budgeted_union_gate",
        "checks": checks,
        "derived": {
            "U50_gain_retention": gain_retention,
            "tail_complement_retention": tail_retention,
        },
    }


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, raw_to_id, lexical_to_id = load_catalog(args.data_dir)
    users = load_users(args.data_dir, raw_to_id)
    cache, _ = load_cached_beams(args.predictions)
    if set(users) != set(cache):
        raise ValueError("validation cache/data user sets differ")
    if len(users) != 19412 or len(raw_to_id) != 11924:
        raise ValueError("unexpected validation user or catalog size")

    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    target_frequency_values = sorted(
        frequencies[sequence[-2]] for sequence in users.values()
    )
    q1 = target_frequency_values[len(target_frequency_values) // 4]
    q3 = target_frequency_values[3 * len(target_frequency_values) // 4]
    user_ids = list(users)
    if args.max_users:
        user_ids = user_ids[: args.max_users]
    full_run = len(user_ids) == 19412

    histories, targets, gram_top50 = [], [], []
    for user in user_ids:
        sequence = users[user]
        histories.append(sequence[max(0, len(sequence) - 22) : -2])
        targets.append(sequence[-2])
        gram_ids = [lexical_to_id[value] for value in cache[user]["candidates"]]
        if len(gram_ids) != 50 or len(set(gram_ids)) != 50:
            raise ValueError(f"{user}: invalid GRAM beam set")
        gram_top50.append(gram_ids)

    model, config = load_item_model(args.checkpoint)
    if config["num_items"] != len(raw_to_id):
        raise ValueError("checkpoint/catalog mismatch")
    cf_top50 = retrieve_cf_top50(model, config, histories, args.batch_size)

    rows = []
    for user, history, target, gram_ids, cf_ids in zip(
        user_ids, histories, targets, gram_top50, cf_top50
    ):
        row = {
            "user": user,
            "target": target,
            "history_length": len(history),
            "target_frequency": frequencies[target],
            "q1": q1,
            "hit_g50": target in gram_ids,
            "G50": gram_ids,
            "U50": list(dict.fromkeys([*gram_ids, *cf_ids])),
        }
        for slots in (10, 20, 30, 40):
            row[f"fixed_top_{slots}"] = fixed_prefix_union(gram_ids, cf_ids, slots)
            row[f"fill_cf_only_{slots}"] = fill_cf_only(gram_ids, cf_ids, slots)
        adaptive_slots = adaptive_history_slots(len(history))
        row["adaptive_history"] = fill_cf_only(gram_ids, cf_ids, adaptive_slots)
        row["adaptive_slots"] = adaptive_slots
        rows.append(row)

    policy_keys = [
        *(f"fixed_top_{slots}" for slots in (10, 20, 30, 40)),
        *(f"fill_cf_only_{slots}" for slots in (10, 20, 30, 40)),
        "adaptive_history",
    ]
    metrics = {
        "G50_coverage": coverage(row["hit_g50"] for row in rows),
        "U50_coverage": coverage(row["target"] in row["U50"] for row in rows),
        "U50_tail_complementary_not_G50": coverage(
            row["target"] in row["U50"] and not row["hit_g50"]
            for row in rows
            if row["target_frequency"] <= q1
        ),
        "policies": {key: policy_summary(rows, key) for key in policy_keys},
    }
    gate = scientific_gate(metrics, full_run)

    per_user_path = args.output_dir / "per_user_budget.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "user_id", "history_length", "target_frequency", "hit_g50", "hit_u50",
            "hit_primary", "primary_union_size", "adaptive_slots", "hit_adaptive",
            "adaptive_union_size",
        ])
        for row in rows:
            writer.writerow([
                row["user"], row["history_length"], row["target_frequency"],
                int(row["hit_g50"]), int(row["target"] in row["U50"]),
                int(row["target"] in row["fill_cf_only_40"]),
                len(row["fill_cf_only_40"]), row["adaptive_slots"],
                int(row["target"] in row["adaptive_history"]),
                len(row["adaptive_history"]),
            ])
    summary = {
        "experiment_id": "GRAM_PHASE10_CF1_A2_TOYS_BUDGETED_UNION_V1",
        "status": "completed",
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "full_run": full_run,
        "users": len(rows),
        "catalog_size": len(raw_to_id),
        "primary_policy": "fill_cf_only_40",
        "popularity_frequency_boundaries": {"q1": q1, "q3": q3},
        "metrics": metrics,
        "scientific_gate": gate,
        "artifacts": {
            "per_user_budget_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest()
        },
        "wall_time_seconds": time.time() - started,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": gate, "wall_time_seconds": summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()

