"""Frozen statistics and integrity checks for Stage17 FP1/FP2 external D0.

This module is deliberately data-source agnostic.  It consumes already
materialized examples and prediction rows, so importing or testing it can
never open the sealed external target file.  The family runtime owns the
single authorized materialization step.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .fullport_data import FullportExternalExample, FullportTrainUser


CUTOFFS = (5, 10, 20, 50)
PRIMARY_METRIC = "ndcg@10"
HISTORY_GROUPS = ("short_le3", "medium_4_9", "long_ge10")
FREQUENCY_GROUPS = ("tail", "mid", "head")
MEMORY_GROUPS = ("generalization", "memorization")


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rank(target: str, ranking: Sequence[str]) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def user_ranking_metrics(target: str, ranking: Sequence[str]) -> dict[str, float]:
    """Return the preregistered single-target user contribution metrics."""

    if not target:
        raise ValueError("target item must be non-empty")
    normalized = [str(item) for item in ranking]
    if len(normalized) > max(CUTOFFS):
        raise ValueError("prediction ranking exceeds the frozen top-50 contract")
    if len(normalized) != len(set(normalized)):
        raise ValueError("item-level prediction ranking contains duplicates")
    rank = _rank(target, normalized)
    result: dict[str, float] = {}
    for cutoff in CUTOFFS:
        hit = float(rank is not None and rank <= cutoff)
        result[f"hit@{cutoff}"] = hit
        result[f"ndcg@{cutoff}"] = (
            1.0 / math.log2(rank + 1) if hit else 0.0
        )
    result["mrr@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    result["target_rank"] = float(rank) if rank is not None else 0.0
    return result


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[FullportExternalExample],
    *,
    expected_arm_id: str | None = None,
    expected_variant: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate exact user/target alignment and unique top-50 item rankings."""

    expected = {example.user_id: example for example in examples}
    if len(expected) != len(examples):
        raise ValueError("external examples contain duplicate users")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        user_id = str(row.get("user_id", ""))
        if not user_id or user_id in indexed:
            raise ValueError(f"invalid or duplicate prediction user: {user_id!r}")
        if user_id not in expected:
            raise ValueError(f"prediction includes an unknown user: {user_id}")
        target = str(row.get("target", ""))
        if target != expected[user_id].target:
            raise ValueError(f"target alignment drift for user {user_id}")
        arm_id = str(row.get("arm_id", ""))
        variant = str(row.get("variant", ""))
        if expected_arm_id is not None and arm_id != expected_arm_id:
            raise ValueError(f"arm mismatch for user {user_id}: {arm_id}")
        if expected_variant is not None and variant != expected_variant:
            raise ValueError(f"variant mismatch for user {user_id}: {variant}")
        ranking = [str(item) for item in row.get("ranking", [])]
        metrics = user_ranking_metrics(target, ranking)
        recorded_rank = row.get("target_rank")
        if recorded_rank is not None and int(recorded_rank or 0) != int(
            metrics["target_rank"]
        ):
            raise ValueError(f"recorded target rank drift for user {user_id}")
        normalized = dict(row)
        normalized["ranking"] = ranking
        normalized["metrics"] = metrics
        indexed[user_id] = normalized
    missing = set(expected) - set(indexed)
    if missing:
        raise ValueError(f"predictions are missing {len(missing)} external users")
    return indexed


def aggregate_metrics(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty prediction cohort")
    metric_names = [
        *(f"hit@{cutoff}" for cutoff in CUTOFFS),
        *(f"ndcg@{cutoff}" for cutoff in CUTOFFS),
        "mrr@10",
    ]
    return {
        metric: fmean(float(row["metrics"][metric]) for row in rows.values())
        for metric in metric_names
    }


def paired_bootstrap_delta(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    replicates: int = 2000,
    seed: int = 2023,
    chunk_size: int = 64,
) -> dict[str, float | int]:
    """Deterministic paired user bootstrap without a full 3-D allocation."""

    treatment_values = np.asarray(treatment, dtype=np.float64)
    control_values = np.asarray(control, dtype=np.float64)
    if treatment_values.shape != control_values.shape or treatment_values.ndim != 1:
        raise ValueError("paired bootstrap inputs must be aligned one-dimensional arrays")
    if treatment_values.size == 0 or replicates <= 0 or chunk_size <= 0:
        raise ValueError("paired bootstrap requires users, replicates and chunks")
    delta = treatment_values - control_values
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        selected = rng.integers(
            0, delta.size, size=(count, delta.size), dtype=np.int64
        )
        samples[start : start + count] = delta[selected].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "paired_users": int(delta.size),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def compare_predictions(
    treatment: Mapping[str, Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
    *,
    treatment_label: str,
    control_label: str,
    replicates: int = 2000,
    seed: int = 2023,
) -> dict[str, Any]:
    """Produce paired effects plus user gain/loss/tie and ranking-change rates."""

    if set(treatment) != set(control):
        raise ValueError("paired prediction cohorts are not identical")
    users = sorted(treatment, key=lambda value: (len(value), value))
    metrics = [
        *(f"hit@{cutoff}" for cutoff in CUTOFFS),
        *(f"ndcg@{cutoff}" for cutoff in CUTOFFS),
        "mrr@10",
    ]
    effects: dict[str, Any] = {}
    for metric in metrics:
        treatment_values = [treatment[user]["metrics"][metric] for user in users]
        control_values = [control[user]["metrics"][metric] for user in users]
        effects[metric] = paired_bootstrap_delta(
            treatment_values,
            control_values,
            replicates=replicates,
            seed=_stable_seed(seed, f"{treatment_label}:{control_label}:{metric}"),
        )

    primary_delta = [
        treatment[user]["metrics"][PRIMARY_METRIC]
        - control[user]["metrics"][PRIMARY_METRIC]
        for user in users
    ]
    gain = sum(value > 0 for value in primary_delta)
    loss = sum(value < 0 for value in primary_delta)
    tie = len(primary_delta) - gain - loss
    ranking_changed = sum(
        treatment[user]["ranking"] != control[user]["ranking"] for user in users
    )
    rank_changed = sum(
        treatment[user]["metrics"]["target_rank"]
        != control[user]["metrics"]["target_rank"]
        for user in users
    )
    return {
        "treatment": treatment_label,
        "control": control_label,
        "paired_users": len(users),
        "effects": effects,
        "primary_user_outcomes": {
            "metric": PRIMARY_METRIC,
            "gain": gain,
            "loss": loss,
            "tie": tie,
            "gain_rate": gain / len(users),
            "loss_rate": loss / len(users),
            "tie_rate": tie / len(users),
        },
        "changed_ranking_rate": ranking_changed / len(users),
        "changed_target_rank_rate": rank_changed / len(users),
    }


def subgroup_assignments(
    train_users: Sequence[FullportTrainUser],
    examples: Sequence[FullportExternalExample],
) -> tuple[dict[str, dict[str, str]], dict[str, float]]:
    """Apply the frozen Stage17 history/frequency/memory subgroup rules."""

    histories = {user.user_id: user.train_items for user in train_users}
    if len(histories) != len(train_users):
        raise ValueError("train-prefix users contain duplicates")
    frequency: Counter[str] = Counter(
        item for user in train_users for item in user.train_items
    )
    if {example.user_id for example in examples} != set(histories):
        raise ValueError("external and train-prefix user cohorts are not identical")
    target_frequencies = np.asarray(
        [frequency[example.target] for example in examples], dtype=np.float64
    )
    q1, q2 = np.quantile(target_frequencies, [1 / 3, 2 / 3])
    assignments: dict[str, dict[str, str]] = {}
    for example in examples:
        full_history = histories[example.user_id]
        observed = min(20, len(full_history))
        history_group = (
            "short_le3"
            if observed <= 3
            else "medium_4_9"
            if observed <= 9
            else "long_ge10"
        )
        target_frequency = frequency[example.target]
        frequency_group = (
            "tail"
            if target_frequency <= q1
            else "mid"
            if target_frequency <= q2
            else "head"
        )
        assignments[example.user_id] = {
            "history_length": history_group,
            "target_frequency": frequency_group,
            "memory": (
                "memorization"
                if example.target in set(full_history)
                else "generalization"
            ),
        }
    return assignments, {
        "target_train_frequency_q1": float(q1),
        "target_train_frequency_q2": float(q2),
    }


def subgroup_comparison(
    treatment: Mapping[str, Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if set(treatment) != set(control) or set(treatment) != set(assignments):
        raise ValueError("subgroup inputs are not user-aligned")
    labels_by_dimension = {
        "history_length": HISTORY_GROUPS,
        "target_frequency": FREQUENCY_GROUPS,
        "memory": MEMORY_GROUPS,
    }
    output: dict[str, Any] = {}
    for dimension, labels in labels_by_dimension.items():
        output[dimension] = {}
        for label in labels:
            users = [
                user
                for user in treatment
                if assignments[user][dimension] == label
            ]
            if not users:
                output[dimension][label] = {"users": 0, "state": "EMPTY"}
                continue
            treatment_ndcg = fmean(
                treatment[user]["metrics"][PRIMARY_METRIC] for user in users
            )
            control_ndcg = fmean(
                control[user]["metrics"][PRIMARY_METRIC] for user in users
            )
            treatment_hit = fmean(
                treatment[user]["metrics"]["hit@10"] for user in users
            )
            control_hit = fmean(
                control[user]["metrics"]["hit@10"] for user in users
            )
            output[dimension][label] = {
                "users": len(users),
                "treatment_ndcg@10": treatment_ndcg,
                "control_ndcg@10": control_ndcg,
                "delta_ndcg@10": treatment_ndcg - control_ndcg,
                "treatment_hit@10": treatment_hit,
                "control_hit@10": control_hit,
                "delta_hit@10": treatment_hit - control_hit,
            }
    return output


def catastrophic_subgroups(
    subgroup_report: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    threshold: float = -0.003,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for dimension in ("history_length", "target_frequency"):
        for label, record in subgroup_report[dimension].items():
            if record.get("users", 0) and float(record["delta_ndcg@10"]) <= threshold:
                failures.append(
                    {
                        "dimension": dimension,
                        "label": label,
                        "users": int(record["users"]),
                        "delta_ndcg@10": float(record["delta_ndcg@10"]),
                    }
                )
    return failures


def summarize_mechanisms(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate path-level diagnostics emitted by a LATTE inference worker."""

    mechanisms = [row.get("mechanism") for row in rows.values()]
    mechanisms = [row for row in mechanisms if isinstance(row, Mapping)]
    if not mechanisms:
        return {"available": False, "users": 0}
    latent_counts: Counter[str] = Counter()
    for row in mechanisms:
        latent_counts.update(
            {str(key): int(value) for key, value in row.get("latent_counts", {}).items()}
        )
    latent_total = sum(latent_counts.values())
    probabilities = [value / latent_total for value in latent_counts.values()] if latent_total else []
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    normalized_entropy = (
        entropy / math.log(len(latent_counts)) if len(latent_counts) > 1 else 0.0
    )

    def mean_field(name: str) -> float:
        values = [float(row[name]) for row in mechanisms if row.get(name) is not None]
        return fmean(values) if values else 0.0

    collapsed_users = [
        int(row.get("latent_root_count", 0)) <= 1 for row in mechanisms
    ]

    return {
        "available": True,
        "users": len(mechanisms),
        "latent_counts": dict(sorted(latent_counts.items())),
        "latent_entropy": entropy,
        "latent_normalized_entropy": normalized_entropy,
        "latent_collapsed": len([value for value in latent_counts.values() if value]) <= 1,
        "latent_user_collapse_rate": fmean(collapsed_users),
        "mean_generated_paths": mean_field("generated_path_count"),
        "mean_valid_paths": mean_field("valid_path_count"),
        "valid_path_rate": mean_field("valid_path_rate"),
        "mean_unique_items": mean_field("unique_item_count"),
        "mean_duplicate_item_paths": mean_field("duplicate_item_path_count"),
        "mean_duplicate_path_rate": mean_field("duplicate_path_rate"),
        "multi_path_item_rate": mean_field("multi_path_item_rate"),
        "target_path_survival_rate": mean_field("target_path_survived"),
        "mean_target_root_count": mean_field("target_root_count"),
        "mean_pre_aggregation_target_rank": mean_field("pre_aggregation_target_rank"),
        "mean_post_aggregation_target_rank": mean_field("post_aggregation_target_rank"),
        "mean_pre_aggregation_ndcg@10": mean_field("pre_aggregation_ndcg@10"),
        "mean_post_aggregation_ndcg@10": mean_field("post_aggregation_ndcg@10"),
        "mean_aggregation_gain_ndcg@10": mean_field("aggregation_gain_ndcg@10"),
        "mean_tree_distance_score_correlation": mean_field(
            "tree_distance_score_correlation"
        ),
    }


def psid_collision_diagnostics(
    resolved_codes: Mapping[str, Sequence[int]],
    raw_codes: np.ndarray,
    centroids: np.ndarray,
    *,
    catalog_items: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Quantify frozen PSID reassignment count and reconstruction distortion."""

    items = list(catalog_items) if catalog_items is not None else list(resolved_codes)
    if set(items) != set(resolved_codes) or len(items) != len(resolved_codes):
        raise ValueError("PSID catalog order does not match resolved code keys")
    if raw_codes.shape != (len(items), 3) or centroids.ndim != 3:
        raise ValueError("PSID diagnostic artifacts have incompatible shapes")
    if centroids.shape[0] != 3:
        raise ValueError("PSID diagnostics require three RQ digits")
    resolved = np.asarray([resolved_codes[item] for item in items], dtype=np.int64)
    if resolved.shape != raw_codes.shape:
        raise ValueError("resolved/raw PSID code shapes differ")
    changed_mask = np.any(resolved != raw_codes, axis=1)
    distortions: list[float] = []
    hamming: list[int] = []
    for raw, updated in zip(raw_codes[changed_mask], resolved[changed_mask]):
        raw_reconstruction = sum(
            (centroids[digit, int(raw[digit])] for digit in range(3)),
            np.zeros(centroids.shape[-1], dtype=np.float64),
        )
        updated_reconstruction = sum(
            (centroids[digit, int(updated[digit])] for digit in range(3)),
            np.zeros(centroids.shape[-1], dtype=np.float64),
        )
        distortions.append(
            float(np.linalg.norm(updated_reconstruction - raw_reconstruction))
        )
        hamming.append(int(np.sum(raw != updated)))
    aliases_after = len(items) - len({tuple(row) for row in resolved.tolist()})
    return {
        "catalog_items": len(items),
        "reassigned_items": int(changed_mask.sum()),
        "reassigned_rate": float(changed_mask.mean()),
        "collision_aliases_after": aliases_after,
        "mean_reassigned_digit_hamming": fmean(hamming) if hamming else 0.0,
        "mean_reconstruction_l2_distortion": fmean(distortions) if distortions else 0.0,
        "max_reconstruction_l2_distortion": max(distortions, default=0.0),
    }


def fp1_gate(
    comparison: Mapping[str, Any],
    mechanisms: Mapping[str, Any],
    *,
    aggregate_item_valid: bool,
    integrity_valid: bool,
) -> dict[str, Any]:
    ndcg = comparison["effects"][PRIMARY_METRIC]
    hit = comparison["effects"]["hit@10"]
    checks = {
        "ndcg_delta_positive": float(ndcg["mean_delta"]) > 0.0,
        "ndcg_ci95_low_positive": float(ndcg["ci95_low"]) > 0.0,
        "hit_delta_nonnegative": float(hit["mean_delta"]) >= 0.0,
        "multi_path_item_rate_positive": float(
            mechanisms.get("multi_path_item_rate", 0.0)
        )
        > 0.0,
        "latent_not_collapsed": mechanisms.get("latent_collapsed") is False,
        "aggregate_item_valid": bool(aggregate_item_valid),
        "integrity_valid": bool(integrity_valid),
    }
    return {
        "verdict": "FP1_STRONG_PASS" if all(checks.values()) else "FP1_NOT_STRONG_PASS",
        "checks": checks,
    }


def fp2_gate(
    g2_vs_g1: Mapping[str, Any],
    g2_vs_g0: Mapping[str, Any],
    subgroup_g2_vs_g0: Mapping[str, Any],
    mechanisms: Mapping[str, Any],
    control_mechanisms: Mapping[str, Any],
    *,
    aggregate_item_valid: bool,
    integrity_valid: bool,
) -> dict[str, Any]:
    g21 = g2_vs_g1["effects"][PRIMARY_METRIC]
    g20 = g2_vs_g0["effects"][PRIMARY_METRIC]
    hit20 = g2_vs_g0["effects"]["hit@10"]
    catastrophes = catastrophic_subgroups(subgroup_g2_vs_g0)
    checks = {
        "g2_vs_g1_delta_ge_0.0015": float(g21["mean_delta"]) >= 0.0015,
        "g2_vs_g1_ci95_low_positive": float(g21["ci95_low"]) > 0.0,
        "g2_vs_g0_delta_ge_0.0015": float(g20["mean_delta"]) >= 0.0015,
        "g2_vs_g0_hit_delta_nonnegative": float(hit20["mean_delta"]) >= 0.0,
        "no_catastrophic_large_subgroup": not catastrophes,
        "multi_path_item_rate_positive": float(
            mechanisms.get("multi_path_item_rate", 0.0)
        )
        > 0.0,
        "latent_not_collapsed": mechanisms.get("latent_collapsed") is False,
        "item_aggregation_gain_positive": float(
            mechanisms.get("mean_aggregation_gain_ndcg@10", 0.0)
        )
        > 0.0,
        "tree_coupling_reduced_vs_g1": abs(
            float(mechanisms.get("mean_tree_distance_score_correlation", 0.0))
        )
        < abs(
            float(
                control_mechanisms.get(
                    "mean_tree_distance_score_correlation", 0.0
                )
            )
        ),
        "target_path_survival_positive": float(
            mechanisms.get("target_path_survival_rate", 0.0)
        )
        > 0.0,
        "aggregate_item_valid": bool(aggregate_item_valid),
        "integrity_valid": bool(integrity_valid),
    }
    return {
        "verdict": "FP2_STRONG_PASS" if all(checks.values()) else "FP2_NOT_STRONG_PASS",
        "checks": checks,
        "catastrophic_subgroups": catastrophes,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL artifact: {path}")
    return rows


def prediction_variants(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        variant = str(row.get("variant", ""))
        if not variant:
            raise ValueError("prediction row lacks a variant")
        grouped.setdefault(variant, []).append(dict(row))
    return grouped
