#!/usr/bin/env python3
"""One-shot frozen PCRF confirmation on the independent Toys test cache."""

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


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from eval_cf0_b3_beamfusion import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    bootstrap_hit10_delta,
    load_cached_beams,
    load_catalog,
    load_users,
    metrics_from_ranks,
    score_item_head,
    standardize,
    subgroup_metrics,
)
from eval_cf0_b4_reliability import delta_metrics, rank_matrix  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_PREDICTIONS = (
    REPO_ROOT / "GRAM/preds/20260722_094800_Toys_sequential_pred_test.tsv"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/cf0_b5_toys_pcrf_test_p2e"
FROZEN_PARAMS = (1.0, 0.5, 1.0)
FIXED_P9C_PARAMS = (0.75, 0.0, 0.0)
Q1 = 5
Q3 = 26


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_TEST_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def build_test_history_target(sequence, max_history=20):
    return sequence[max(0, len(sequence) - 1 - max_history) : -1], sequence[-1]


def prepare_test_records(data_dir, predictions, checkpoint, batch_size):
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir)
    users = load_users(data_dir, raw_to_id)
    cache, footer = load_cached_beams(predictions)
    if set(users) != set(cache) or len(users) != 19412 or len(raw_to_id) != 11924:
        raise ValueError("unexpected test cache/data identity or size")
    id_to_lexical = {
        raw_to_id[raw_item]: lexical for raw_item, lexical in raw_to_lexical.items()
    }
    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    records = []
    for user, sequence in users.items():
        history, target_id = build_test_history_target(sequence)
        cached = cache[user]
        if cached["gold"] != id_to_lexical[target_id]:
            raise ValueError(f"{user}: cached/current test gold mismatch")
        try:
            candidate_ids = [lexical_to_id[value] for value in cached["candidates"]]
        except KeyError as error:
            raise ValueError(f"{user}: unmapped test candidate {error}") from error
        target_position = candidate_ids.index(target_id) if target_id in candidate_ids else -1
        records.append(
            {
                "user": user,
                "history": history,
                "history_length": len(history),
                "target_id": target_id,
                "target_frequency": frequencies[target_id],
                "candidate_ids": candidate_ids,
                "candidate_frequencies": np.asarray(
                    [frequencies[item] for item in candidate_ids], dtype=np.float64
                ),
                "target_position": target_position,
                "seq": cached["seq"],
            }
        )
    score_item_head(records, checkpoint, batch_size)
    for record in records:
        record["seq_z"] = standardize(record["seq"])
        record["cf_z"] = standardize(record["cf"])
        record["pop_z"] = standardize(np.log1p(record["candidate_frequencies"]))
        record["tail_mass"] = float(
            np.mean(record["candidate_frequencies"][:10] <= Q1)
        )
    return records, footer


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, footer = prepare_test_records(
        args.data_dir, args.predictions, args.checkpoint, args.batch_size
    )
    seq_z = np.stack([record["seq_z"] for record in records])
    cf_z = np.stack([record["cf_z"] for record in records])
    pop_z = np.stack([record["pop_z"] for record in records])
    tail_mass = np.asarray([record["tail_mass"] for record in records])
    target_positions = np.asarray([record["target_position"] for record in records])
    target_frequencies = np.asarray([record["target_frequency"] for record in records])

    baseline_ranks, _ = rank_matrix(
        seq_z, cf_z, pop_z, tail_mass, target_positions, (0.0, 0.0, 0.0)
    )
    baseline = metrics_from_ranks(baseline_ranks)
    for key in ("hit@5", "hit@10", "hit@20", "hit@50", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50"):
        metric_key = key.replace("hit", "Hit").replace("ndcg", "NDCG")
        if not math.isclose(baseline[metric_key], footer[key], abs_tol=1e-12):
            raise ValueError(f"test baseline identity mismatch for {key}")

    pcrf_ranks, reliability = rank_matrix(
        seq_z, cf_z, pop_z, tail_mass, target_positions, FROZEN_PARAMS
    )
    pcrf, delta = delta_metrics(pcrf_ranks, baseline_ranks)
    fixed_ranks, _ = rank_matrix(
        seq_z, cf_z, pop_z, tail_mass, target_positions, FIXED_P9C_PARAMS
    )
    fixed, fixed_delta = delta_metrics(fixed_ranks, baseline_ranks)
    baseline_groups = subgroup_metrics(
        records, range(len(records)), baseline_ranks, Q1, Q3
    )
    pcrf_groups = subgroup_metrics(records, range(len(records)), pcrf_ranks, Q1, Q3)
    hit_ci = bootstrap_hit10_delta(
        baseline_ranks, pcrf_ranks, args.bootstrap_replicates, args.seed
    )
    tail_mask = target_frequencies <= Q1
    tail_ci = bootstrap_hit10_delta(
        baseline_ranks[tail_mask],
        pcrf_ranks[tail_mask],
        args.bootstrap_replicates,
        args.seed + 1,
    )
    tail_delta = (
        pcrf_groups["target_tail"]["Hit@10"]
        - baseline_groups["target_tail"]["Hit@10"]
    )
    checks = {
        "Hit@10_delta_at_least_0.002": delta["Hit@10"] >= 0.002,
        "Hit@10_bootstrap_lower_positive": hit_ci["lower"] > 0,
        "NDCG@10_non_degradation": delta["NDCG@10"] >= 0,
        "tail_Hit@10_non_degradation": tail_delta >= 0,
        "tail_Hit@10_noninferiority_lower_at_least_minus_0.002": tail_ci["lower"] >= -0.002,
        "Hit@1_delta_at_least_minus_0.001": delta["Hit@1"] >= -0.001,
        "Hit@50_identity": math.isclose(pcrf["Hit@50"], baseline["Hit@50"], abs_tol=1e-12),
    }
    confirmation_status = "confirmed" if all(checks.values()) else "failed_confirmation_gate"

    per_user_path = args.output_dir / "per_user_test.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "user_id", "history_length", "target_frequency", "tail_mass",
            "reliability", "baseline_rank", "fixed_p9c_rank", "pcrf_rank",
        ])
        for index, record in enumerate(records):
            writer.writerow([
                record["user"], record["history_length"], record["target_frequency"],
                f"{tail_mass[index]:.6f}", f"{reliability[index]:.6f}",
                baseline_ranks[index], fixed_ranks[index], pcrf_ranks[index],
            ])

    summary = {
        "experiment_id": "GRAM_PHASE9_CF0_B5_TOYS_PCRF_TEST_P2E_V1",
        "status": "completed",
        "evidence_class": "one_shot_independent_test_confirmation",
        "dataset": "Toys",
        "split": "test",
        "test_read": True,
        "sports_read": False,
        "integrity_gate": {
            "status": "passed",
            "users": len(records),
            "catalog_size": 11924,
            "beams_per_user": 50,
            "baseline_identity": baseline,
        },
        "frozen_params": {
            "lambda": FROZEN_PARAMS[0],
            "beta": FROZEN_PARAMS[1],
            "gamma": FROZEN_PARAMS[2],
            "q1": Q1,
            "q3": Q3,
        },
        "baseline": baseline,
        "fixed_p9c_diagnostic": {"metrics": fixed, "delta": fixed_delta},
        "pcrf": {
            "metrics": pcrf,
            "delta": delta,
            "baseline_subgroups": baseline_groups,
            "pcrf_subgroups": pcrf_groups,
            "Hit@10_paired_bootstrap_95ci": hit_ci,
            "tail_Hit@10_delta": tail_delta,
            "tail_Hit@10_paired_bootstrap_95ci": tail_ci,
        },
        "confirmation_gate": {"status": confirmation_status, "checks": checks},
        "artifacts": {
            "per_user_test_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest()
        },
        "wall_time_seconds": time.time() - started,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"confirmation_gate": summary["confirmation_gate"], "wall_time_seconds": summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
