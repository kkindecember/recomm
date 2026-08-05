#!/usr/bin/env python3
"""Cross-fitted popularity-calibrated reliability-aware BeamFusion."""

import argparse
import csv
import hashlib
import itertools
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
    DEFAULT_PREDICTIONS,
    bootstrap_hit10_delta,
    load_cached_beams,
    load_catalog,
    load_users,
    metrics_from_ranks,
    score_item_head,
    standardize,
    subgroup_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/cf0_b4_toys_reliability_p2d"
LAMBDAS = [0.5, 0.75, 1.0]
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
GAMMAS = [0.0, 1.0, 2.0, 4.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-seed", default="2023")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def make_folds(user_ids, num_folds=5, seed="2023"):
    ordered = sorted(
        user_ids,
        key=lambda user: (
            hashlib.sha256(f"P9-2D:{seed}:{user}".encode()).hexdigest(),
            user,
        ),
    )
    return {user: index % num_folds for index, user in enumerate(ordered)}


def prepare_records(data_dir, predictions, checkpoint, batch_size):
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir)
    users = load_users(data_dir, raw_to_id)
    cache, footer = load_cached_beams(predictions)
    if set(users) != set(cache) or len(users) != 19412 or len(raw_to_id) != 11924:
        raise ValueError("unexpected prediction/data identity or size")
    id_to_lexical = {
        raw_to_id[raw_item]: lexical for raw_item, lexical in raw_to_lexical.items()
    }
    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    frequency_values = sorted(frequencies[sequence[-2]] for sequence in users.values())
    q1 = frequency_values[len(frequency_values) // 4]
    q3 = frequency_values[3 * len(frequency_values) // 4]
    records = []
    for user, sequence in users.items():
        cached = cache[user]
        target_id = sequence[-2]
        if cached["gold"] != id_to_lexical[target_id]:
            raise ValueError(f"{user}: cached/current gold mismatch")
        try:
            candidate_ids = [lexical_to_id[value] for value in cached["candidates"]]
        except KeyError as error:
            raise ValueError(f"{user}: unmapped candidate {error}") from error
        target_position = candidate_ids.index(target_id) if target_id in candidate_ids else -1
        records.append(
            {
                "user": user,
                "history": sequence[max(0, len(sequence) - 22) : -2],
                "history_length": min(len(sequence) - 2, 20),
                "target_id": target_id,
                "target_frequency": frequencies[target_id],
                "candidate_ids": candidate_ids,
                "candidate_frequencies": np.asarray([frequencies[item] for item in candidate_ids], dtype=np.float64),
                "target_position": target_position,
                "seq": cached["seq"],
            }
        )
    score_item_head(records, checkpoint, batch_size)
    for record in records:
        record["seq_z"] = standardize(record["seq"])
        record["cf_z"] = standardize(record["cf"])
        record["pop_z"] = standardize(np.log1p(record["candidate_frequencies"]))
        record["tail_mass"] = float(np.mean(record["candidate_frequencies"][:10] <= q1))
    return records, footer, q1, q3


def rank_matrix(seq_z, cf_z, pop_z, tail_mass, target_positions, params):
    weight, beta, gamma = params
    adjusted = cf_z - beta * pop_z
    adjusted = (adjusted - adjusted.mean(axis=1, keepdims=True)) / np.maximum(
        adjusted.std(axis=1, keepdims=True), 1e-6
    )
    reliability = np.power(1.0 - tail_mass, gamma)
    joint = seq_z + weight * reliability[:, None] * adjusted
    order = np.argsort(-joint, axis=1, kind="stable")
    inverse = np.argsort(order, axis=1)
    safe_positions = np.maximum(target_positions, 0)
    ranks = inverse[np.arange(inverse.shape[0]), safe_positions] + 1
    return np.where(target_positions >= 0, ranks, 51).astype(np.int64), reliability


def delta_metrics(candidate_ranks, baseline_ranks):
    candidate = metrics_from_ranks(candidate_ranks)
    baseline = metrics_from_ranks(baseline_ranks)
    return candidate, {
        key: candidate[key] - baseline[key] for key in candidate if key != "count"
    }


def choose_params(rank_cache, train_indices, tail_mask, baseline_ranks):
    baseline_train = baseline_ranks[train_indices]
    tail_train = train_indices[tail_mask[train_indices]]
    baseline_tail = baseline_ranks[tail_train]
    feasible = []
    diagnostics = []
    for params, ranks in rank_cache.items():
        overall, delta = delta_metrics(ranks[train_indices], baseline_train)
        tail, tail_delta = delta_metrics(ranks[tail_train], baseline_tail)
        is_feasible = (
            delta["Hit@10"] >= 0.002
            and delta["NDCG@10"] >= 0
            and tail_delta["Hit@10"] >= 0
        )
        row = {
            "lambda": params[0],
            "beta": params[1],
            "gamma": params[2],
            "overall": overall,
            "delta": delta,
            "tail": tail,
            "tail_delta": tail_delta,
            "feasible": is_feasible,
        }
        diagnostics.append(row)
        if is_feasible:
            feasible.append((params, row))
    if not feasible:
        return None, diagnostics
    selected, _ = min(
        feasible,
        key=lambda item: (
            -item[1]["overall"]["Hit@10"],
            -item[1]["tail"]["Hit@10"],
            -item[1]["overall"]["NDCG@10"],
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    return selected, diagnostics


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, footer, q1, q3 = prepare_records(
        args.data_dir, args.predictions, args.checkpoint, args.batch_size
    )
    users = [record["user"] for record in records]
    folds = make_folds(users, args.num_folds, args.fold_seed)
    fold_array = np.asarray([folds[user] for user in users], dtype=np.int64)
    seq_z = np.stack([record["seq_z"] for record in records])
    cf_z = np.stack([record["cf_z"] for record in records])
    pop_z = np.stack([record["pop_z"] for record in records])
    tail_mass = np.asarray([record["tail_mass"] for record in records])
    target_positions = np.asarray([record["target_position"] for record in records])
    target_frequencies = np.asarray([record["target_frequency"] for record in records])
    baseline_ranks, _ = rank_matrix(
        seq_z, cf_z, pop_z, tail_mass, target_positions, (0.0, 0.0, 0.0)
    )
    baseline_full = metrics_from_ranks(baseline_ranks)
    for key in ("hit@5", "hit@10", "hit@20", "hit@50", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50"):
        metric_key = key.replace("hit", "Hit").replace("ndcg", "NDCG")
        if not math.isclose(baseline_full[metric_key], footer[key], abs_tol=1e-12):
            raise ValueError(f"baseline identity mismatch for {key}")

    params_grid = list(itertools.product(LAMBDAS, BETAS, GAMMAS))
    rank_cache = {}
    reliability_cache = {}
    for params in params_grid:
        ranks, reliability = rank_matrix(
            seq_z, cf_z, pop_z, tail_mass, target_positions, params
        )
        rank_cache[params] = ranks
        reliability_cache[params] = reliability

    tail_mask = target_frequencies <= q1
    oof_ranks = np.full(len(records), 51, dtype=np.int64)
    oof_effective_lambda = np.zeros(len(records), dtype=np.float64)
    oof_beta = np.zeros(len(records), dtype=np.float64)
    oof_gamma = np.zeros(len(records), dtype=np.float64)
    fold_results = []
    all_feasible = True
    for fold in range(args.num_folds):
        train_indices = np.flatnonzero(fold_array != fold)
        eval_indices = np.flatnonzero(fold_array == fold)
        selected, diagnostics = choose_params(
            rank_cache, train_indices, tail_mask, baseline_ranks
        )
        if selected is None:
            all_feasible = False
            selected_ranks = baseline_ranks
            reliability = np.zeros(len(records))
            selected_values = (0.0, 0.0, 0.0)
        else:
            selected_ranks = rank_cache[selected]
            reliability = reliability_cache[selected]
            selected_values = selected
        oof_ranks[eval_indices] = selected_ranks[eval_indices]
        oof_effective_lambda[eval_indices] = selected_values[0] * reliability[eval_indices]
        oof_beta[eval_indices] = selected_values[1]
        oof_gamma[eval_indices] = selected_values[2]
        eval_overall, eval_delta = delta_metrics(
            selected_ranks[eval_indices], baseline_ranks[eval_indices]
        )
        eval_tail_indices = eval_indices[tail_mask[eval_indices]]
        eval_tail, eval_tail_delta = delta_metrics(
            selected_ranks[eval_tail_indices], baseline_ranks[eval_tail_indices]
        )
        selected_diag = None
        if selected is not None:
            selected_diag = next(
                row for row in diagnostics
                if (row["lambda"], row["beta"], row["gamma"]) == selected
            )
        fold_results.append(
            {
                "fold": fold,
                "train_count": int(train_indices.size),
                "evaluation_count": int(eval_indices.size),
                "calibration_feasible": selected is not None,
                "selected": {
                    "lambda": selected_values[0],
                    "beta": selected_values[1],
                    "gamma": selected_values[2],
                },
                "train_selected_diagnostics": selected_diag,
                "evaluation": eval_overall,
                "evaluation_delta": eval_delta,
                "evaluation_tail": eval_tail,
                "evaluation_tail_delta": eval_tail_delta,
            }
        )

    oof_metrics, oof_delta = delta_metrics(oof_ranks, baseline_ranks)
    fixed_ranks = rank_cache[(0.75, 0.0, 0.0)]
    fixed_metrics, fixed_delta = delta_metrics(fixed_ranks, baseline_ranks)
    baseline_groups = subgroup_metrics(records, range(len(records)), baseline_ranks, q1, q3)
    oof_groups = subgroup_metrics(records, range(len(records)), oof_ranks, q1, q3)
    hit_ci = bootstrap_hit10_delta(
        baseline_ranks, oof_ranks, args.bootstrap_replicates, args.seed
    )
    tail_ci = bootstrap_hit10_delta(
        baseline_ranks[tail_mask], oof_ranks[tail_mask], args.bootstrap_replicates, args.seed + 1
    )
    checks = {
        "all_folds_calibration_feasible": all_feasible,
        "Hit@10_delta_at_least_0.002": oof_delta["Hit@10"] >= 0.002,
        "Hit@10_bootstrap_lower_positive": hit_ci["lower"] > 0,
        "NDCG@10_non_degradation": oof_delta["NDCG@10"] >= 0,
        "tail_Hit@10_non_degradation": oof_groups["target_tail"]["Hit@10"] >= baseline_groups["target_tail"]["Hit@10"],
        "Hit@50_identity": math.isclose(oof_metrics["Hit@50"], baseline_full["Hit@50"], abs_tol=1e-12),
    }
    gate = "passed" if all(checks.values()) else "failed_development_gate"

    fold_path = args.output_dir / "fold_assignments.tsv"
    with fold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "fold"])
        for user in users:
            writer.writerow([user, folds[user]])
    per_user_path = args.output_dir / "per_user_oof.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "user_id", "fold", "history_length", "target_frequency", "tail_mass",
            "effective_lambda", "beta", "gamma", "baseline_rank", "fixed_p9c_rank", "oof_rank",
        ])
        for index, record in enumerate(records):
            writer.writerow([
                record["user"], fold_array[index], record["history_length"],
                record["target_frequency"], f"{tail_mass[index]:.6f}",
                f"{oof_effective_lambda[index]:.8f}", oof_beta[index], oof_gamma[index],
                baseline_ranks[index], fixed_ranks[index], oof_ranks[index],
            ])

    summary = {
        "experiment_id": "GRAM_PHASE9_CF0_B4_TOYS_RELIABILITY_FUSION_P2D_V1",
        "status": "completed",
        "evidence_class": "cross_fitted_development_not_independent_confirmation",
        "dataset": "Toys",
        "test_read": False,
        "sports_read": False,
        "integrity_gate": {
            "status": "passed",
            "users": len(records),
            "catalog_size": 11924,
            "beams_per_user": 50,
            "baseline_identity": baseline_full,
        },
        "folds": fold_results,
        "grid": {"lambdas": LAMBDAS, "betas": BETAS, "gammas": GAMMAS, "count": len(params_grid)},
        "baseline": baseline_full,
        "fixed_p9c": {"metrics": fixed_metrics, "delta": fixed_delta},
        "oof": {
            "metrics": oof_metrics,
            "delta": oof_delta,
            "baseline_subgroups": baseline_groups,
            "oof_subgroups": oof_groups,
            "Hit@10_paired_bootstrap_95ci": hit_ci,
            "tail_Hit@10_paired_bootstrap_95ci": tail_ci,
        },
        "development_gate": {"status": gate, "checks": checks},
        "popularity_frequency_boundaries": {"q1": q1, "q3": q3},
        "artifacts": {
            "fold_assignments_sha256": hashlib.sha256(fold_path.read_bytes()).hexdigest(),
            "per_user_oof_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest(),
        },
        "wall_time_seconds": time.time() - started,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"development_gate": summary["development_gate"], "wall_time_seconds": summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
