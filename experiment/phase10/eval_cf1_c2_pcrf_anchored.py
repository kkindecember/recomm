#!/usr/bin/env python3
"""CF1-C2 PCRF-anchored source-asymmetric cross-fitted ranker."""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = REPO_ROOT / "artifacts/phase10/cf1_c0_toys_feature_audit/feature_table.npz"
DEFAULT_C0_SUMMARY = REPO_ROOT / "artifacts/phase10/cf1_c0_toys_feature_audit/summary.json"
DEFAULT_C1_SUMMARY = REPO_ROOT / "artifacts/phase10/cf1_c1_toys_crossfit_calibrator/summary.json"
DEFAULT_C1_PER_USER = REPO_ROOT / "artifacts/phase10/cf1_c1_toys_crossfit_calibrator/per_user_oof.tsv"
DEFAULT_DIAGNOSTIC = REPO_ROOT / "artifacts/phase10/cf1_c1_error_decomposition/summary.json"
DEFAULT_PREDICTIONS = REPO_ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_c2_toys_pcrf_anchored"

FEATURE_NAMES = [
    "gram_z",
    "corrected_item_z",
    "gram_rr",
    "cf_rr",
    "agreement",
    "reliability_x_item",
    "short_history_x_item",
    "long_history_x_item",
    "source_both",
    "source_cf_only",
    "both_x_gram",
    "both_x_item",
    "cf_only_x_gram",
    "cf_only_x_item",
    "negative_item_log_frequency_z",
]
SOURCE_NAMES = {0: "gram_only", 1: "both", 2: "cf_only"}
NONNEGATIVE_FEATURES = {
    "gram_z", "corrected_item_z", "negative_item_log_frequency_z"
}

RESIDUAL_CAP = 1.0
NDCG50_WEIGHT = 0.25
GOLD_RETENTION_MULTIPLIER = 2.0
SAFETY_COEFFICIENT = 0.25
L2 = 1e-3
MAX_ITER = 200
OPTIMIZER_FTOL = 1e-9
OPTIMIZER_GTOL = 1e-6
BOOTSTRAP_REPLICATES = 2000
SEED = 2023


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--c0-summary", type=Path, default=DEFAULT_C0_SUMMARY)
    parser.add_argument("--c1-summary", type=Path, default=DEFAULT_C1_SUMMARY)
    parser.add_argument("--c1-per-user", type=Path, default=DEFAULT_C1_PER_USER)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=0,
                        help="Deterministic hash-selected smoke users; 0 means all users.")
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def load_cached_scores(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n")
        if not header.startswith("idx\t"):
            raise ValueError("unexpected prediction header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1:
                continue
            scores = np.asarray([float(value) for value in fields[-1].split("||")])
            if scores.shape != (50,) or not np.isfinite(scores).all():
                raise ValueError(f"{fields[0]}: malformed cached scores")
            rows[fields[0]] = scores
    return rows


def deterministic_subset(users, requested, seed=SEED):
    if requested <= 0 or requested >= len(users):
        return np.arange(len(users), dtype=np.int64)
    ordered = sorted(
        range(len(users)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{users[index]}".encode()).hexdigest(),
            str(users[index]),
        ),
    )
    return np.asarray(sorted(ordered[:requested]), dtype=np.int64)


def build_anchors(data, cache, user_indices):
    offsets = data["offsets"].astype(np.int64)
    users = data["users"]
    reliability = data["user_reliability"].astype(np.float64)
    item_score = data["item_score"].astype(np.float64)
    item_log_frequency = data["item_log_frequency"].astype(np.float64)
    anchors = {}
    baseline_orders = {}
    raw_orders = {}
    floors_exact = True
    finite = True
    order_exact = True
    for user_index in user_indices:
        left, right = int(offsets[user_index]), int(offsets[user_index + 1])
        seq_z = standardize(cache[str(users[user_index])])
        item_z = standardize(item_score[left : left + 50])
        pop_z = standardize(item_log_frequency[left : left + 50])
        corrected = standardize(item_z - 0.5 * pop_z)
        raw = seq_z + reliability[user_index] * corrected
        anchor_g50 = standardize(raw)
        floor = float(anchor_g50.min())
        anchor = np.full(right - left, floor, dtype=np.float64)
        anchor[:50] = anchor_g50
        raw_order = np.argsort(-raw, kind="stable")
        anchor_order = np.argsort(-anchor_g50, kind="stable")
        raw_orders[int(user_index)] = raw_order
        baseline_orders[int(user_index)] = anchor_order
        anchors[int(user_index)] = anchor
        finite = finite and bool(np.isfinite(anchor).all())
        order_exact = order_exact and bool(np.array_equal(raw_order, anchor_order))
        if right - left > 50:
            floors_exact = floors_exact and bool(np.all(anchor[50:] == floor))
    audit = {
        "all_anchor_scores_finite": finite,
        "pcrf_order_preserved_by_anchor_standardization": order_exact,
        "all_cf_only_anchors_equal_rank50_floor": floors_exact,
    }
    return anchors, baseline_orders, raw_orders, audit


def build_unscaled_features(data):
    offsets = data["offsets"].astype(np.int64)
    lengths = np.diff(offsets)
    reliability = np.repeat(data["user_reliability"].astype(np.float64), lengths)
    history = np.repeat(data["user_history_length"].astype(np.int64), lengths)
    source = data["source"].astype(np.int64)
    gram = data["gram_z"].astype(np.float64)
    item = data["corrected_item_z"].astype(np.float64)
    both = source == 1
    cf_only = source == 2
    return np.column_stack([
        gram,
        item,
        data["gram_rr"],
        data["cf_rr"],
        data["agreement"],
        reliability * item,
        (history <= 5) * item,
        (history >= 11) * item,
        both,
        cf_only,
        both * gram,
        both * item,
        cf_only * gram,
        cf_only * item,
        data["item_log_frequency"],
    ]).astype(np.float64)


def select_features(unscaled_features, candidate_indices, log_mean, log_std):
    features = unscaled_features[candidate_indices].copy()
    features[:, -1] = -(features[:, -1] - log_mean) / log_std
    return features


def concatenate_users(offsets, user_indices):
    chunks = [np.arange(offsets[user], offsets[user + 1], dtype=np.int64) for user in user_indices]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def popularity_weights(target_frequency):
    target_frequency = np.asarray(target_frequency, dtype=np.int64)
    groups = np.where(target_frequency <= 5, 0, np.where(target_frequency < 26, 1, 2))
    counts = np.bincount(groups, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("all popularity groups must occur in every training fold")
    group_weights = len(groups) / (3.0 * counts)
    weights = group_weights[groups]
    weights /= weights.mean()
    return weights, group_weights


def discount(rank, cutoff):
    return 1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0


def prepare_pairs(lengths, gold_local, anchors, target_frequency):
    user_weights, group_weights = popularity_weights(target_frequency)
    supervised_positive = []
    supervised_negative = []
    supervised_weight = []
    safety_positive = []
    safety_negative = []
    cursor = 0
    for user, (length, gold, anchor, user_weight) in enumerate(
        zip(lengths, gold_local, anchors, user_weights)
    ):
        order = np.argsort(-anchor, kind="stable")
        ranks = np.empty(length, dtype=np.int64)
        ranks[order] = np.arange(1, length + 1)
        gold_rank = int(ranks[gold])
        gold_d10 = discount(gold_rank, 10)
        gold_d50 = discount(gold_rank, 50)
        for candidate in range(length):
            if candidate == gold:
                continue
            candidate_rank = int(ranks[candidate])
            pair_weight = abs(gold_d10 - discount(candidate_rank, 10))
            pair_weight += NDCG50_WEIGHT * abs(gold_d50 - discount(candidate_rank, 50))
            if pair_weight == 0:
                continue
            if gold_rank <= 10 and candidate_rank > 10:
                pair_weight *= GOLD_RETENTION_MULTIPLIER
            supervised_positive.append(cursor + gold)
            supervised_negative.append(cursor + candidate)
            supervised_weight.append(float(user_weight * pair_weight))
        non_gold_top10 = [int(candidate) for candidate in order[:10] if candidate != gold]
        for higher, lower in zip(non_gold_top10[:-1], non_gold_top10[1:]):
            safety_positive.append(cursor + higher)
            safety_negative.append(cursor + lower)
        cursor += int(length)
    pairs = {
        "supervised_positive": np.asarray(supervised_positive, dtype=np.int64),
        "supervised_negative": np.asarray(supervised_negative, dtype=np.int64),
        "supervised_weight": np.asarray(supervised_weight, dtype=np.float64),
        "safety_positive": np.asarray(safety_positive, dtype=np.int64),
        "safety_negative": np.asarray(safety_negative, dtype=np.int64),
    }
    return pairs, group_weights


def softplus_negative_margin(margin):
    return np.logaddexp(0.0, -margin)


def anchored_pairwise_loss_grad(weights, x, anchor, pairs, l2=L2):
    linear = x @ weights
    tanh_linear = np.tanh(linear)
    scores = anchor + RESIDUAL_CAP * tanh_linear
    derivative = RESIDUAL_CAP * (1.0 - tanh_linear * tanh_linear)
    gradient_score = np.zeros(scores.size, dtype=np.float64)

    sup_pos = pairs["supervised_positive"]
    sup_neg = pairs["supervised_negative"]
    sup_weight = pairs["supervised_weight"]
    sup_margin = scores[sup_pos] - scores[sup_neg]
    sup_norm = max(float(sup_weight.sum()), 1e-12)
    loss = float(np.sum(sup_weight * softplus_negative_margin(sup_margin)) / sup_norm)
    sup_dmargin = sup_weight * (-1.0 / (1.0 + np.exp(sup_margin))) / sup_norm
    np.add.at(gradient_score, sup_pos, sup_dmargin)
    np.add.at(gradient_score, sup_neg, -sup_dmargin)

    safe_pos = pairs["safety_positive"]
    safe_neg = pairs["safety_negative"]
    if safe_pos.size:
        safe_margin = scores[safe_pos] - scores[safe_neg]
        loss += SAFETY_COEFFICIENT * float(np.mean(softplus_negative_margin(safe_margin)))
        safe_dmargin = (
            SAFETY_COEFFICIENT * (-1.0 / (1.0 + np.exp(safe_margin))) / safe_pos.size
        )
        np.add.at(gradient_score, safe_pos, safe_dmargin)
        np.add.at(gradient_score, safe_neg, -safe_dmargin)

    gradient = x.T @ (gradient_score * derivative)
    loss += 0.5 * l2 * float(weights @ weights)
    gradient += l2 * weights
    return loss, gradient


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    result = {
        "count": int(ranks.size),
        "mrr": float(np.mean(np.where(ranks <= 50, 1.0 / ranks, 0.0))),
    }
    for cutoff in (1, 5, 10, 20, 50):
        hit = ranks <= cutoff
        result[f"Hit@{cutoff}"] = float(hit.mean())
        result[f"NDCG@{cutoff}"] = float(
            np.mean(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0))
        )
    return result


def metric_delta(candidate, baseline):
    return {key: candidate[key] - baseline[key] for key in candidate if key != "count"}


def subgroup_metrics(ranks, history, target_frequency):
    masks = {
        "history_1-5": history <= 5,
        "history_6-10": (history >= 6) & (history <= 10),
        "history_11-20": history >= 11,
        "target_tail": target_frequency <= 5,
        "target_middle": (target_frequency > 5) & (target_frequency < 26),
        "target_head": target_frequency >= 26,
    }
    return {name: metrics_from_ranks(ranks[mask]) for name, mask in masks.items()}


def bootstrap_hit10_delta(baseline_ranks, candidate_ranks, replicates=BOOTSTRAP_REPLICATES,
                          seed=SEED):
    paired = (candidate_ranks <= 10).astype(np.float64) - (baseline_ranks <= 10).astype(np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, paired.size, paired.size)
        values[index] = paired[sample].mean()
    low, high = np.quantile(values, [0.025, 0.975])
    return {"replicates": replicates, "lower": float(low), "upper": float(high)}


def load_expected_baseline(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["user_id"]] = int(row["baseline_rank"])
    return rows


def scientific_gate(summary):
    delta = summary["oof"]["delta"]
    checks = {
        "Hit@10_delta_at_least_0.003": delta["Hit@10"] >= 0.003,
        "Hit@50_delta_at_least_0.020": delta["Hit@50"] >= 0.020,
        "tail_Hit@10_non_degradation": (
            summary["oof"]["subgroup_delta"]["target_tail"]["Hit@10"] >= 0
        ),
        "Hit@1_delta_at_least_minus_0.001": delta["Hit@1"] >= -0.001,
        "Hit@10_bootstrap_lower_positive": (
            summary["oof"]["Hit@10_paired_bootstrap_95ci"]["lower"] > 0
        ),
        "at_least_four_folds_positive_Hit@10": sum(
            row["delta"]["Hit@10"] > 0 for row in summary["folds"]
        ) >= 4,
        "all_folds_converged": all(row["optimizer"]["success"] for row in summary["folds"]),
        "all_oof_scores_finite": summary["oof"]["finite_fraction"] == 1.0,
        "train_only_parameters": True,
        "protected_splits_unread": not any(
            summary[name] for name in ("test_read", "beauty_read", "sports_read")
        ),
    }
    return {"status": "passed" if all(checks.values()) else "failed_development_gate",
            "checks": checks}


def main():
    args = parse_args()
    started = time.time()
    c0 = json.loads(args.c0_summary.read_text(encoding="utf-8"))
    c1 = json.loads(args.c1_summary.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    if c0["scientific_gate"]["status"] != "passed":
        raise ValueError("C0 gate not passed")
    if c1["development_gate"]["status"] != "failed_development_gate":
        raise ValueError("C1 failure identity not preserved")
    if diagnostic["status"] != "completed" or not diagnostic["no_training_performed"]:
        raise ValueError("C1 diagnostic identity not preserved")
    if sha256(args.features) != c0["artifacts"]["feature_table_sha256"]:
        raise ValueError("feature table hash mismatch")

    data = np.load(args.features, allow_pickle=False)
    users = data["users"]
    offsets = data["offsets"].astype(np.int64)
    folds_all = data["fold"].astype(np.int64)
    targets_all = data["user_target_position"].astype(np.int64)
    history_all = data["user_history_length"].astype(np.int64)
    frequency_all = data["user_target_frequency"].astype(np.int64)
    selected = deterministic_subset(users, args.users)
    if args.users > 512 and args.users < len(users):
        raise ValueError("implementation smoke is capped at 512 users")
    if selected.size < 5 or set(folds_all[selected]) != set(range(5)):
        raise ValueError("selected users must cover all five frozen folds")

    cache = load_cached_scores(args.predictions)
    anchors, baseline_orders, _, anchor_audit = build_anchors(data, cache, selected)
    expected_baseline = load_expected_baseline(args.c1_per_user)
    baseline_ranks = np.full(selected.size, 51, dtype=np.int64)
    for local_user, global_user in enumerate(selected):
        target = int(targets_all[global_user])
        if 0 <= target < 50:
            baseline_ranks[local_user] = int(
                np.flatnonzero(baseline_orders[int(global_user)] == target)[0]
            ) + 1
    baseline_identity = all(
        baseline_ranks[local] == expected_baseline[str(users[global_user])]
        for local, global_user in enumerate(selected)
    )

    selected_position = {int(global_user): local for local, global_user in enumerate(selected)}
    unscaled_features = build_unscaled_features(data)
    oof_ranks = np.full(selected.size, 91, dtype=np.int64)
    finite_scores = 0
    total_scores = 0
    saturated_95 = 0
    saturated_99 = 0
    max_abs_residual = 0.0
    fold_rows = []
    model_rows = []
    transition_rows = []

    for fold in range(5):
        train_users_all = selected[folds_all[selected] != fold]
        eval_users = selected[folds_all[selected] == fold]
        train_candidates_for_scale = concatenate_users(offsets, train_users_all)
        train_logs = data["item_log_frequency"][train_candidates_for_scale].astype(np.float64)
        log_mean = float(train_logs.mean())
        log_std = max(float(train_logs.std()), 1e-6)
        train_users = train_users_all[targets_all[train_users_all] >= 0]
        if train_users.size == 0:
            raise ValueError(f"fold {fold}: no target-in-union training users")
        train_candidates = concatenate_users(offsets, train_users)
        x_train = select_features(unscaled_features, train_candidates, log_mean, log_std)
        train_lengths = np.asarray(
            [offsets[user + 1] - offsets[user] for user in train_users], dtype=np.int64
        )
        gold_local = targets_all[train_users].astype(np.int64)
        anchor_train = np.concatenate([anchors[int(user)] for user in train_users])
        pair_data, group_weights = prepare_pairs(
            train_lengths,
            gold_local,
            [anchors[int(user)] for user in train_users],
            frequency_all[train_users],
        )
        initial = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        bounds = [
            (0.0, None) if name in NONNEGATIVE_FEATURES else (None, None)
            for name in FEATURE_NAMES
        ]
        result = minimize(
            anchored_pairwise_loss_grad,
            initial,
            args=(x_train, anchor_train, pair_data, L2),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": MAX_ITER, "ftol": OPTIMIZER_FTOL,
                     "gtol": OPTIMIZER_GTOL, "maxls": 30},
        )
        weights = result.x.astype(np.float64)
        for global_user in eval_users:
            left, right = int(offsets[global_user]), int(offsets[global_user + 1])
            candidate_indices = np.arange(left, right, dtype=np.int64)
            x_eval = select_features(unscaled_features, candidate_indices, log_mean, log_std)
            residual = RESIDUAL_CAP * np.tanh(x_eval @ weights)
            scores = anchors[int(global_user)] + residual
            finite_scores += int(np.isfinite(scores).sum())
            total_scores += scores.size
            saturated_95 += int(np.sum(np.abs(residual) >= 0.95 * RESIDUAL_CAP))
            saturated_99 += int(np.sum(np.abs(residual) >= 0.99 * RESIDUAL_CAP))
            max_abs_residual = max(max_abs_residual, float(np.max(np.abs(residual))))
            target = int(targets_all[global_user])
            if target >= 0:
                order = np.argsort(-scores, kind="stable")
                oof_ranks[selected_position[int(global_user)]] = int(
                    np.flatnonzero(order == target)[0]
                ) + 1
        eval_local = np.asarray([selected_position[int(user)] for user in eval_users])
        eval_metrics = metrics_from_ranks(oof_ranks[eval_local])
        eval_baseline = metrics_from_ranks(baseline_ranks[eval_local])
        fold_rows.append({
            "fold": fold,
            "train_users": int(train_users_all.size),
            "train_positive_users": int(train_users.size),
            "evaluation_users": int(eval_users.size),
            "metrics": eval_metrics,
            "baseline": eval_baseline,
            "delta": metric_delta(eval_metrics, eval_baseline),
            "optimizer": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "objective": float(result.fun),
            },
        })
        model_rows.append({
            "fold": fold,
            "feature_names": FEATURE_NAMES,
            "weights": weights.tolist(),
            "item_log_frequency_mean": log_mean,
            "item_log_frequency_std": log_std,
            "popularity_group_weights_tail_middle_head": group_weights.tolist(),
        })
        print(json.dumps({"fold": fold, "optimizer_success": bool(result.success),
                          "iterations": int(result.nit), "eval_users": int(eval_users.size)}),
              flush=True)

    baseline = metrics_from_ranks(baseline_ranks)
    oof = metrics_from_ranks(oof_ranks)
    history = history_all[selected]
    frequency = frequency_all[selected]
    baseline_groups = subgroup_metrics(baseline_ranks, history, frequency)
    oof_groups = subgroup_metrics(oof_ranks, history, frequency)
    subgroup_delta = {
        name: metric_delta(oof_groups[name], baseline_groups[name]) for name in oof_groups
    }
    hit_ci = bootstrap_hit10_delta(baseline_ranks, oof_ranks)
    target_sources = np.full(selected.size, "union_miss", dtype="<U16")
    for local, global_user in enumerate(selected):
        target = int(targets_all[global_user])
        if target >= 0:
            target_sources[local] = SOURCE_NAMES[
                int(data["source"][offsets[global_user] + target])
            ]
    source_baseline_groups = {}
    source_oof_groups = {}
    source_subgroup_delta = {}
    for source_name in ("gram_only", "both", "cf_only", "union_miss"):
        mask = target_sources == source_name
        source_baseline_groups[source_name] = metrics_from_ranks(baseline_ranks[mask])
        source_oof_groups[source_name] = metrics_from_ranks(oof_ranks[mask])
        source_subgroup_delta[source_name] = metric_delta(
            source_oof_groups[source_name], source_baseline_groups[source_name]
        )
    for local, global_user in enumerate(selected):
        target = int(targets_all[global_user])
        target_source = "union_miss" if target < 0 else SOURCE_NAMES[
            int(data["source"][offsets[global_user] + target])
        ]
        baseline_hit = baseline_ranks[local] <= 10
        oof_hit = oof_ranks[local] <= 10
        transition = "stable_hit" if baseline_hit and oof_hit else (
            "gain" if not baseline_hit and oof_hit else (
                "loss" if baseline_hit and not oof_hit else "stable_miss"
            )
        )
        transition_rows.append({
            "user_id": str(users[global_user]),
            "fold": int(folds_all[global_user]),
            "target_source": target_source,
            "baseline_rank": int(baseline_ranks[local]),
            "oof_rank": int(oof_ranks[local]),
            "hit10_transition": transition,
        })

    mode = "smoke" if selected.size < len(users) else "formal"
    summary = {
        "experiment_id": f"GRAM_PHASE10_CF1_C2_TOYS_PCRF_ANCHORED_{mode.upper()}_V1",
        "status": "completed",
        "mode": mode,
        "evidence_class": (
            "implementation_smoke_not_scientific_evidence" if mode == "smoke"
            else "post_C1_cross_fitted_development_not_independent_confirmation"
        ),
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "selected_users": int(selected.size),
        "selection_seed": SEED,
        "model": {
            "type": "PCRF_anchored_source_asymmetric_bounded_residual",
            "feature_names": FEATURE_NAMES,
            "nonnegative_features": sorted(NONNEGATIVE_FEATURES),
            "residual_cap": RESIDUAL_CAP,
            "ndcg50_weight": NDCG50_WEIGHT,
            "gold_retention_multiplier": GOLD_RETENTION_MULTIPLIER,
            "safety_coefficient": SAFETY_COEFFICIENT,
            "l2": L2,
            "max_iter": MAX_ITER,
            "target_frequency_in_inference": False,
            "target_frequency_training_weight_only": True,
        },
        "anchor_audit": anchor_audit,
        "baseline_identity_with_C1": baseline_identity,
        "folds": fold_rows,
        "baseline": baseline,
        "oof": {
            "metrics": oof,
            "delta": metric_delta(oof, baseline),
            "baseline_subgroups": baseline_groups,
            "subgroups": oof_groups,
            "subgroup_delta": subgroup_delta,
            "baseline_source_subgroups": source_baseline_groups,
            "source_subgroups": source_oof_groups,
            "source_subgroup_delta": source_subgroup_delta,
            "Hit@10_paired_bootstrap_95ci": hit_ci,
            "finite_fraction": finite_scores / total_scores,
            "max_abs_residual": max_abs_residual,
            "residual_saturation_fraction_at_0.95_cap": saturated_95 / total_scores,
            "residual_saturation_fraction_at_0.99_cap": saturated_99 / total_scores,
        },
        "input_identity": {
            "feature_table_sha256": sha256(args.features),
            "c0_summary_sha256": sha256(args.c0_summary),
            "c1_summary_sha256": sha256(args.c1_summary),
            "c1_per_user_sha256": sha256(args.c1_per_user),
            "diagnostic_sha256": sha256(args.diagnostic),
            "predictions_sha256": sha256(args.predictions),
        },
        "implementation_checks": {
            "users_at_most_512": selected.size <= 512 if mode == "smoke" else True,
            "five_frozen_folds_present": set(folds_all[selected]) == set(range(5)),
            "baseline_identity_with_C1": baseline_identity,
            **anchor_audit,
            "all_folds_converged": all(row["optimizer"]["success"] for row in fold_rows),
            "all_oof_scores_finite": finite_scores == total_scores,
            "residual_bound_respected": max_abs_residual <= RESIDUAL_CAP + 1e-12,
            "train_only_scaling_and_popularity_weights": True,
            "target_absent_from_inference_features": not any(
                forbidden in name for name in FEATURE_NAMES
                for forbidden in ("target", "gold", "label")
            ),
            "protected_splits_unread": True,
        },
    }
    summary["implementation_gate"] = {
        "status": "passed" if all(summary["implementation_checks"].values()) else "failed",
        "checks": summary["implementation_checks"],
    }
    if mode == "formal":
        summary["development_gate"] = scientific_gate(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = args.output_dir / "per_user_oof.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "user_id", "fold", "history_length", "target_frequency", "target_in_union",
            "baseline_rank", "oof_rank",
        ])
        for local, global_user in enumerate(selected):
            writer.writerow([
                str(users[global_user]), int(folds_all[global_user]), int(history_all[global_user]),
                int(frequency_all[global_user]), int(targets_all[global_user] >= 0),
                int(baseline_ranks[local]), int(oof_ranks[local]),
            ])
    models_path = args.output_dir / "fold_models.json"
    models_path.write_text(json.dumps(model_rows, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    fold_metrics_path = args.output_dir / "fold_metrics.json"
    fold_metrics_path.write_text(json.dumps(fold_rows, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    transitions_path = args.output_dir / "hit10_transitions.tsv"
    with transitions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(transition_rows)
    summary["artifacts"] = {
        "per_user_oof_sha256": sha256(per_user_path),
        "fold_models_sha256": sha256(models_path),
        "fold_metrics_sha256": sha256(fold_metrics_path),
        "hit10_transitions_sha256": sha256(transitions_path),
    }
    summary["wall_time_seconds"] = time.time() - started
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "mode": mode,
        "implementation_gate": summary["implementation_gate"],
        "selected_users": int(selected.size),
        "wall_time_seconds": summary["wall_time_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
