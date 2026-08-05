#!/usr/bin/env python3
"""Evaluate the frozen Toys PCRF formula on another dataset/split."""

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
    bootstrap_hit10_delta,
    load_cached_beams,
    load_users,
    metrics_from_ranks,
    normalize_lexical_id,
    score_item_head,
    standardize,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--item-index-name", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--item-head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("validation", "test"), required=True)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--q1", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def load_catalog(data_dir, item_index_name):
    raw_to_lexical, lexical_to_raw = {}, {}
    with (data_dir / item_index_name).open(encoding="utf-8") as handle:
        for line in handle:
            raw_item, lexical = line.rstrip("\n").split(" ", 1)
            normalized = normalize_lexical_id(lexical)
            if normalized in lexical_to_raw:
                raise ValueError(f"duplicate normalized lexical ID: {normalized}")
            raw_to_lexical[raw_item] = normalized
            lexical_to_raw[normalized] = raw_item
    raw_to_id = {item: index + 1 for index, item in enumerate(sorted(raw_to_lexical))}
    lexical_to_id = {lexical: raw_to_id[item] for lexical, item in lexical_to_raw.items()}
    return raw_to_lexical, raw_to_id, lexical_to_id


def metric_delta(candidate, baseline):
    return {key: candidate[key] - baseline[key] for key in candidate if key != "count"}


def ranks_and_top10(records, weight, beta, gamma):
    ranks, top10 = [], []
    for record in records:
        seq_z = standardize(record["seq"])
        cf_z = standardize(record["cf"])
        pop_z = standardize(np.log1p(record["candidate_frequencies"]))
        adjusted = standardize(cf_z - beta * pop_z)
        reliability = (1.0 - record["tail_mass"]) ** gamma
        joint = seq_z + weight * reliability * adjusted
        order = np.argsort(-joint, kind="stable")
        top10.append(order[:10])
        target = record["target_position"]
        ranks.append(51 if target < 0 else int(np.flatnonzero(order == target)[0]) + 1)
    return np.asarray(ranks, dtype=np.int64), top10


def subgroup_metrics(records, ranks, q1):
    masks = {
        "target_tail": np.asarray([record["target_frequency"] <= q1 for record in records]),
        "target_non_tail": np.asarray([record["target_frequency"] > q1 for record in records]),
        "history_1-5": np.asarray([record["history_length"] <= 5 for record in records]),
        "history_6-10": np.asarray([6 <= record["history_length"] <= 10 for record in records]),
        "history_11-20": np.asarray([record["history_length"] >= 11 for record in records]),
    }
    return {name: metrics_from_ranks(ranks[mask]) for name, mask in masks.items() if mask.any()}


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(args.data_dir, args.item_index_name)
    users = load_users(args.data_dir, raw_to_id)
    beams, footer = load_cached_beams(args.predictions)
    if set(users) != set(beams):
        raise ValueError("prediction/data user identities differ")
    id_to_lexical = {raw_to_id[item]: lexical for item, lexical in raw_to_lexical.items()}
    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    validation_freqs = sorted(frequencies[sequence[-2]] for sequence in users.values())
    computed_q1 = validation_freqs[len(validation_freqs) // 4]
    q1 = computed_q1 if args.q1 is None else args.q1
    if args.mode == "validation" and args.q1 is not None and args.q1 != computed_q1:
        raise ValueError("validation q1 override differs from computed train-prefix q1")

    records = []
    for user, sequence in users.items():
        beam = beams[user]
        target_id = sequence[-2] if args.mode == "validation" else sequence[-1]
        history = sequence[-22:-2] if args.mode == "validation" else sequence[-21:-1]
        if beam["gold"] != id_to_lexical[target_id]:
            raise ValueError(f"{user}: gold mismatch")
        candidate_ids = [lexical_to_id[value] for value in beam["candidates"]]
        target_position = candidate_ids.index(target_id) if target_id in candidate_ids else -1
        candidate_frequencies = np.asarray([frequencies[item] for item in candidate_ids], dtype=np.float64)
        records.append({
            "user": user,
            "history": history,
            "history_length": len(history),
            "target_frequency": frequencies[target_id],
            "candidate_ids": candidate_ids,
            "candidate_frequencies": candidate_frequencies,
            "target_position": target_position,
            "seq": beam["seq"],
            "tail_mass": float(np.mean(candidate_frequencies[:10] <= q1)),
        })
    config = score_item_head(records, args.item_head, 512)
    if config["num_items"] != len(raw_to_id):
        raise ValueError("item-head/catalog size mismatch")
    baseline_ranks, _ = ranks_and_top10(records, 0.0, 0.0, 0.0)
    pcrf_ranks, _ = ranks_and_top10(records, args.lambda_weight, args.beta, args.gamma)
    baseline = metrics_from_ranks(baseline_ranks)
    pcrf = metrics_from_ranks(pcrf_ranks)
    for key in ("hit@5", "hit@10", "hit@20", "hit@50", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50"):
        metric_key = key.replace("hit", "Hit").replace("ndcg", "NDCG")
        if not math.isclose(baseline[metric_key], footer[key], abs_tol=1e-12):
            raise ValueError(f"baseline identity mismatch: {key}")
    delta = metric_delta(pcrf, baseline)
    baseline_groups = subgroup_metrics(records, baseline_ranks, q1)
    pcrf_groups = subgroup_metrics(records, pcrf_ranks, q1)
    ci = bootstrap_hit10_delta(baseline_ranks, pcrf_ranks, args.bootstrap_replicates, args.seed)
    tail_mask = np.asarray([record["target_frequency"] <= q1 for record in records])
    tail_ci = bootstrap_hit10_delta(baseline_ranks[tail_mask], pcrf_ranks[tail_mask], args.bootstrap_replicates, args.seed + 1)
    checks = {
        "Hit@10_delta_at_least_0.002": delta["Hit@10"] >= 0.002,
        "Hit@10_bootstrap_lower_positive": ci["lower"] > 0,
        "NDCG@10_non_degradation": delta["NDCG@10"] >= 0,
        "tail_Hit@10_non_degradation": pcrf_groups["target_tail"]["Hit@10"] >= baseline_groups["target_tail"]["Hit@10"],
        "tail_CI_lower_at_least_minus_0.002": tail_ci["lower"] >= -0.002,
        "Hit@1_delta_at_least_minus_0.001": delta["Hit@1"] >= -0.001,
        "Hit@50_identity": math.isclose(pcrf["Hit@50"], baseline["Hit@50"], abs_tol=1e-12),
    }
    gate_name = "validation_admission" if args.mode == "validation" else "external_confirmation"
    gate = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    per_user = args.output_dir / "per_user.tsv"
    with per_user.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "history_length", "target_frequency", "tail_mass", "baseline_rank", "pcrf_rank"])
        for record, base_rank, rank in zip(records, baseline_ranks, pcrf_ranks):
            writer.writerow([record["user"], record["history_length"], record["target_frequency"], record["tail_mass"], base_rank, rank])
    summary = {
        "experiment_id": f"GRAM_PHASE9_P9X_{args.dataset.upper()}_{args.mode.upper()}_FIXED_PCRF_V1",
        "status": "completed",
        "evidence_class": "external_validation_admission" if args.mode == "validation" else "one_shot_external_test_confirmation",
        "dataset": args.dataset,
        "split": args.mode,
        "test_read": args.mode == "test",
        "frozen_pcrf": {"lambda": args.lambda_weight, "beta": args.beta, "gamma": args.gamma, "q1": q1},
        "integrity": {"users": len(records), "catalog_size": len(raw_to_id), "beams_per_user": 50, "baseline_identity": True},
        "baseline": baseline,
        "pcrf": pcrf,
        "delta": delta,
        "baseline_subgroups": baseline_groups,
        "pcrf_subgroups": pcrf_groups,
        "Hit@10_paired_bootstrap_95ci": ci,
        "tail_Hit@10_paired_bootstrap_95ci": tail_ci,
        gate_name: gate,
        "artifacts": {"per_user_sha256": hashlib.sha256(per_user.read_bytes()).hexdigest()},
        "wall_time_seconds": time.time() - started,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({gate_name: gate, "delta": delta, "ci": ci, "q1": q1}), flush=True)


if __name__ == "__main__":
    main()
