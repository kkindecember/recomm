"""Unified item-level R2 evaluator with paired user uncertainty."""

from __future__ import annotations

import math
from statistics import mean
from typing import Sequence

import numpy as np


CUTOFFS = (5, 10, 20, 50)


def user_ranking_contribution(target: str, ranking: Sequence[str]) -> dict[str, float]:
    try:
        rank = list(ranking).index(target) + 1
    except ValueError:
        rank = None
    result: dict[str, float] = {}
    for cutoff in CUTOFFS:
        hit = rank is not None and rank <= cutoff
        result[f"hit@{cutoff}"] = float(hit)
        result[f"ndcg@{cutoff}"] = (
            1.0 / math.log2(rank + 1.0) if hit and rank is not None else 0.0
        )
    return result


def aggregate_user_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty user metric collection")
    return {key: mean(row[key] for row in rows) for key in sorted(rows[0])}


def paired_bootstrap_delta(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    replicates: int = 1000,
    seed: int = 2023,
) -> dict[str, float]:
    left = np.asarray(treatment, dtype=np.float64)
    right = np.asarray(control, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap needs equal non-empty one-dimensional arrays")
    differences = left - right
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(replicates, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {
        "delta": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "positive_user_fraction": float((differences > 0).mean()),
        "zero_user_fraction": float((differences == 0).mean()),
        "replicates": int(replicates),
    }


def mechanism_gate(family: str, metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    if metrics.get("valid_item_rate") != 1.0:
        failures.append("valid_item_rate_not_one")
    if family == "latte" and metrics.get("multi_path_item_rate", 0.0) <= 0.0:
        failures.append("no_multi_path_items")
    elif family == "gryphon":
        if metrics.get("candidate_sets_identical") is not True:
            failures.append("candidate_sets_not_identical")
        if metrics.get("mean_target_rank_gain", 0.0) <= 0.0:
            failures.append("nonpositive_same_candidate_rank_gain")
    elif family == "diffgrm":
        if metrics.get("treatment_generation_seconds", math.inf) >= metrics.get(
            "control_generation_seconds", -math.inf
        ):
            failures.append("masked_generation_not_faster_than_ar")
    elif family == "setrec":
        if metrics.get("treatment_generation_seconds", math.inf) >= metrics.get(
            "control_generation_seconds", -math.inf
        ):
            failures.append("simultaneous_generation_not_faster_than_ar")
        if metrics.get("set_token_recovery", 0.0) <= 0.0:
            failures.append("set_token_recovery_not_positive")
    return not failures, failures


def family_promotion_decision(
    *,
    cohort_deltas: Sequence[dict[str, float]],
    overall_ndcg: dict[str, float],
    overall_hit: dict[str, float],
    mechanism_pass: bool,
    minimum_ndcg_delta: float = 0.0015,
) -> str:
    positive_cohorts = sum(row["ndcg@10"] > 0.0 for row in cohort_deltas)
    minimum_cohort_hit = min(row["hit@10"] for row in cohort_deltas)
    strong = (
        overall_ndcg["delta"] >= minimum_ndcg_delta
        and positive_cohorts >= 2
        and overall_hit["delta"] >= 0.0
        and minimum_cohort_hit >= -0.002
        and mechanism_pass
    )
    if strong:
        return "STRONG_PROMOTE"
    borderline = overall_ndcg["delta"] > 0.0 and mechanism_pass
    return "BORDERLINE_ONE_REVISION" if borderline else "REJECT"


def compare_family_predictions(
    *,
    treatment: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    cohorts: Sequence[Sequence[str]],
    mechanism_metrics: dict,
    family: str,
    bootstrap_replicates: int = 1000,
    seed: int = 2023,
) -> dict:
    if set(treatment) != set(control):
        raise ValueError("treatment and control user predictions are not paired")
    cohort_results = []
    for index, cohort in enumerate(cohorts):
        if any(user not in treatment for user in cohort):
            raise ValueError(f"cohort c{index} includes a user without paired predictions")
        treatment_rows = [treatment[user] for user in cohort]
        control_rows = [control[user] for user in cohort]
        treatment_metrics = aggregate_user_metrics(treatment_rows)
        control_metrics = aggregate_user_metrics(control_rows)
        cohort_results.append(
            {
                "cohort_id": f"c{index}",
                "users": len(cohort),
                "treatment": treatment_metrics,
                "control": control_metrics,
                "delta": {
                    key: treatment_metrics[key] - control_metrics[key]
                    for key in treatment_metrics
                },
            }
        )
    user_ids = sorted(treatment)
    overall_ndcg = paired_bootstrap_delta(
        [treatment[user]["ndcg@10"] for user in user_ids],
        [control[user]["ndcg@10"] for user in user_ids],
        replicates=bootstrap_replicates,
        seed=seed,
    )
    overall_hit = paired_bootstrap_delta(
        [treatment[user]["hit@10"] for user in user_ids],
        [control[user]["hit@10"] for user in user_ids],
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    mechanism_pass, mechanism_failures = mechanism_gate(family, mechanism_metrics)
    decision = family_promotion_decision(
        cohort_deltas=[row["delta"] for row in cohort_results],
        overall_ndcg=overall_ndcg,
        overall_hit=overall_hit,
        mechanism_pass=mechanism_pass,
    )
    return {
        "family": family,
        "cohorts": cohort_results,
        "overall_paired_ndcg@10": overall_ndcg,
        "overall_paired_hit@10": overall_hit,
        "positive_ndcg_cohorts": sum(
            row["delta"]["ndcg@10"] > 0.0 for row in cohort_results
        ),
        "mechanism_metrics": mechanism_metrics,
        "mechanism_pass": mechanism_pass,
        "mechanism_failures": mechanism_failures,
        "decision": decision,
    }
