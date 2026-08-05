#!/usr/bin/env python3
"""Five-fold cross-fitted monotone linear listwise calibration for CF1."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9 = REPO_ROOT / "experiment/phase9"
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from eval_cf0_b3_beamfusion import (  # noqa: E402
    DEFAULT_PREDICTIONS,
    bootstrap_hit10_delta,
    load_cached_beams,
    metrics_from_ranks,
    standardize,
)


DEFAULT_FEATURES = REPO_ROOT / "artifacts/phase10/cf1_c0_toys_feature_audit/feature_table.npz"
DEFAULT_C0_SUMMARY = REPO_ROOT / "artifacts/phase10/cf1_c0_toys_feature_audit/summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_c1_toys_crossfit_calibrator"
FEATURE_NAMES = [
    "gram_z", "corrected_item_z", "source_both", "source_cf_only",
    "gram_rr", "cf_rr", "agreement", "reliability_x_item",
    "short_history_x_item", "long_history_x_item", "item_log_frequency",
]
L2 = 1e-3
MAX_ITER = 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--c0-summary", type=Path, default=DEFAULT_C0_SUMMARY)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def listwise_loss_grad(weights, x, lengths, gold_local, l2=L2):
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    scores = x @ weights
    maximum = np.maximum.reduceat(scores, starts)
    shifted = scores - np.repeat(maximum, lengths)
    exp_score = np.exp(shifted)
    denominator = np.add.reduceat(exp_score, starts)
    probabilities = exp_score / np.repeat(denominator, lengths)
    gold_global = starts + gold_local
    loss = float(np.mean(np.log(denominator) + maximum - scores[gold_global]))
    gradient = (x.T @ probabilities - x[gold_global].sum(axis=0)) / len(lengths)
    loss += 0.5 * l2 * float(weights @ weights)
    gradient += l2 * weights
    return loss, gradient.astype(np.float64)


def candidate_indices(offsets, user_indices):
    return np.concatenate([np.arange(offsets[user], offsets[user + 1]) for user in user_indices])


def build_base_features(data):
    offsets = data["offsets"]
    lengths = np.diff(offsets)
    reliability = np.repeat(data["user_reliability"], lengths)
    history = np.repeat(data["user_history_length"], lengths)
    source = data["source"]
    corrected = data["corrected_item_z"].astype(np.float64)
    return np.column_stack([
        data["gram_z"], corrected, source == 1, source == 2,
        data["gram_rr"], data["cf_rr"], data["agreement"],
        reliability * corrected, (history <= 5) * corrected,
        (history >= 11) * corrected, data["item_log_frequency"],
    ]).astype(np.float64)


def rank_target(scores, target_position, missing_rank=91):
    if target_position < 0:
        return missing_rank
    order = np.argsort(-scores, kind="stable")
    return int(np.flatnonzero(order == target_position)[0]) + 1


def frozen_pcrf_ranks(data, cache):
    ranks = np.empty(len(data["users"]), dtype=np.int16)
    for index, raw_user in enumerate(data["users"]):
        user = str(raw_user)
        left = int(data["offsets"][index])
        target = int(data["user_target_position"][index])
        if target < 0 or target >= 50:
            ranks[index] = 51
            continue
        seq_z = standardize(np.asarray(cache[user]["seq"], dtype=np.float64))
        item_z = standardize(data["item_score"][left : left + 50].astype(np.float64))
        pop_z = standardize(data["item_log_frequency"][left : left + 50].astype(np.float64))
        adjusted = standardize(item_z - 0.5 * pop_z)
        score = seq_z + float(data["user_reliability"][index]) * adjusted
        ranks[index] = rank_target(score, target, 51)
    return ranks


def subgroup_metrics(ranks, history, target_frequency, q1=5, q3=26):
    masks = {
        "history_1-5": history <= 5,
        "history_6-10": (history >= 6) & (history <= 10),
        "history_11-20": history >= 11,
        "target_tail": target_frequency <= q1,
        "target_middle": (target_frequency > q1) & (target_frequency < q3),
        "target_head": target_frequency >= q3,
    }
    return {name: metrics_from_ranks(ranks[mask]) for name, mask in masks.items()}


def metric_delta(candidate, baseline):
    return {key: candidate[key] - baseline[key] for key in candidate if key != "count"}


def scientific_gate(summary):
    delta = summary["oof"]["delta"]
    checks = {
        "Hit@10_delta_at_least_0.003": delta["Hit@10"] >= 0.003,
        "Hit@50_delta_at_least_0.020": delta["Hit@50"] >= 0.020,
        "tail_Hit@10_non_degradation": summary["oof"]["subgroup_delta"]["target_tail"]["Hit@10"] >= 0,
        "Hit@1_delta_at_least_minus_0.001": delta["Hit@1"] >= -0.001,
        "Hit@10_bootstrap_lower_positive": summary["oof"]["Hit@10_paired_bootstrap_95ci"]["lower"] > 0,
        "at_least_four_folds_positive_Hit@10": sum(row["delta"]["Hit@10"] > 0 for row in summary["folds"]) >= 4,
        "all_folds_converged": all(row["optimizer"]["success"] for row in summary["folds"]),
        "all_oof_scores_finite": summary["oof"]["finite_fraction"] == 1.0,
        "train_only_scaling": True,
    }
    return {"status": "passed" if all(checks.values()) else "failed_development_gate", "checks": checks}


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    c0 = json.load(args.c0_summary.open())
    if c0["scientific_gate"]["status"] != "passed":
        raise ValueError("C0 gate not passed")
    if hashlib.sha256(args.features.read_bytes()).hexdigest() != c0["artifacts"]["feature_table_sha256"]:
        raise ValueError("C0 feature table hash mismatch")
    data = np.load(args.features, allow_pickle=False)
    if len(data["users"]) != 19412 or int(data["offsets"][-1]) != 1698905:
        raise ValueError("unexpected C0 feature identity")
    cache, _ = load_cached_beams(args.predictions)
    baseline_ranks = frozen_pcrf_ranks(data, cache)
    baseline = metrics_from_ranks(baseline_ranks)
    expected = c0["metrics"]["baselines"]["frozen_PCRF_1.0_0.5_1.0"]
    if any(not math.isclose(baseline[key], expected[key], abs_tol=1e-12) for key in expected if key != "count"):
        raise ValueError("frozen PCRF identity mismatch")

    offsets = data["offsets"].astype(np.int64)
    lengths_all = np.diff(offsets)
    folds = data["fold"].astype(np.int64)
    targets = data["user_target_position"].astype(np.int64)
    base_x = build_base_features(data)
    oof_ranks = np.full(len(folds), 91, dtype=np.int16)
    oof_scores_finite = 0
    oof_scores_total = 0
    fold_rows = []
    models = []

    for fold in range(5):
        train_users_all = np.flatnonzero(folds != fold)
        eval_users = np.flatnonzero(folds == fold)
        scale_indices = candidate_indices(offsets, train_users_all)
        log_mean = float(base_x[scale_indices, 10].mean())
        log_std = max(float(base_x[scale_indices, 10].std()), 1e-6)
        train_users = train_users_all[targets[train_users_all] >= 0]
        train_indices = candidate_indices(offsets, train_users)
        train_lengths = lengths_all[train_users]
        x_train = base_x[train_indices].copy()
        x_train[:, 10] = (x_train[:, 10] - log_mean) / log_std
        gold_local = targets[train_users]
        initial = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        initial[:2] = 1.0
        result = minimize(
            listwise_loss_grad, initial, args=(x_train, train_lengths, gold_local, L2),
            method="L-BFGS-B", jac=True,
            bounds=[(0.0, None), (0.0, None)] + [(None, None)] * (len(FEATURE_NAMES) - 2),
            options={"maxiter": MAX_ITER, "ftol": 1e-9, "gtol": 1e-6, "maxls": 30},
        )
        weights = result.x.astype(np.float64)
        for user in eval_users:
            left, right = offsets[user], offsets[user + 1]
            x_eval = base_x[left:right].copy()
            x_eval[:, 10] = (x_eval[:, 10] - log_mean) / log_std
            scores = x_eval @ weights
            oof_scores_finite += int(np.isfinite(scores).sum())
            oof_scores_total += scores.size
            oof_ranks[user] = rank_target(scores, int(targets[user]), 91)
        eval_metrics = metrics_from_ranks(oof_ranks[eval_users])
        eval_baseline = metrics_from_ranks(baseline_ranks[eval_users])
        fold_rows.append({
            "fold": fold, "train_users": int(train_users_all.size),
            "train_positive_users": int(train_users.size), "evaluation_users": int(eval_users.size),
            "metrics": eval_metrics, "baseline": eval_baseline,
            "delta": metric_delta(eval_metrics, eval_baseline),
            "optimizer": {"success": bool(result.success), "status": int(result.status),
                          "message": str(result.message), "iterations": int(result.nit),
                          "objective": float(result.fun)},
        })
        models.append({
            "fold": fold, "feature_names": FEATURE_NAMES,
            "weights": weights.tolist(), "item_log_frequency_mean": log_mean,
            "item_log_frequency_std": log_std,
        })

    oof = metrics_from_ranks(oof_ranks)
    delta = metric_delta(oof, baseline)
    history = data["user_history_length"].astype(np.int64)
    target_frequency = data["user_target_frequency"].astype(np.int64)
    baseline_groups = subgroup_metrics(baseline_ranks, history, target_frequency)
    oof_groups = subgroup_metrics(oof_ranks, history, target_frequency)
    subgroup_delta = {name: metric_delta(oof_groups[name], baseline_groups[name]) for name in oof_groups}
    hit_ci = bootstrap_hit10_delta(baseline_ranks, oof_ranks, args.bootstrap_replicates, args.seed)
    result_summary = {
        "experiment_id": "GRAM_PHASE10_CF1_C1_TOYS_CROSSFIT_CALIBRATOR_V1",
        "status": "completed", "evidence_class": "cross_fitted_development_not_independent_confirmation",
        "dataset": "Toys", "split": "validation", "test_read": False,
        "beauty_read": False, "sports_read": False,
        "model": {"type": "monotone_linear_listwise_softmax", "l2": L2,
                  "max_iter": MAX_ITER, "feature_names": FEATURE_NAMES,
                  "nonnegative_features": FEATURE_NAMES[:2]},
        "baseline": baseline,
        "folds": fold_rows,
        "oof": {"metrics": oof, "delta": delta, "baseline_subgroups": baseline_groups,
                "subgroups": oof_groups, "subgroup_delta": subgroup_delta,
                "Hit@10_paired_bootstrap_95ci": hit_ci,
                "finite_fraction": oof_scores_finite / oof_scores_total},
    }
    result_summary["development_gate"] = scientific_gate(result_summary)
    per_user_path = args.output_dir / "per_user_oof.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "fold", "history_length", "target_frequency", "target_in_union", "baseline_rank", "oof_rank"])
        for index, user in enumerate(data["users"]):
            writer.writerow([str(user), int(folds[index]), int(history[index]), int(target_frequency[index]),
                             int(targets[index] >= 0), int(baseline_ranks[index]), int(oof_ranks[index])])
    model_path = args.output_dir / "fold_models.json"
    with model_path.open("w", encoding="utf-8") as handle:
        json.dump(models, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    result_summary["artifacts"] = {
        "c0_feature_table_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "per_user_oof_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest(),
        "fold_models_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    result_summary["wall_time_seconds"] = time.time() - started
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"development_gate": result_summary["development_gate"],
                      "oof": result_summary["oof"], "wall_time_seconds": result_summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()

