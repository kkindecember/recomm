"""Pure contracts for the Stage18 S18-1 actionability diagnostic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from experiment.phase18.core.contracts import internal_fold_view


S18_1_FOLDS = ("I-1", "I0")


def stable_cohort(
    domain: str,
    users: Iterable[str],
    count: int = 1024,
    seed: int = 2023,
) -> list[str]:
    return stable_cohort_slice(domain, users, start=0, count=count, seed=seed)


def stable_cohort_slice(
    domain: str,
    users: Iterable[str],
    start: int,
    count: int = 1024,
    seed: int = 2023,
) -> list[str]:
    """Select a predeclared contiguous slice of the stable S18-1 ranking."""
    if start < 0:
        raise ValueError("cohort start must be non-negative")
    if count <= 0:
        raise ValueError("cohort count must be positive")
    unique = set(users)
    stop = start + count
    if len(unique) < stop:
        raise ValueError(
            f"{domain}: only {len(unique)} eligible users for cohort slice "
            f"[{start},{stop})"
        )
    ranked = sorted(
        unique,
        key=lambda user: (
            hashlib.sha256(f"S18-1|{seed}|{domain}|{user}".encode()).hexdigest(),
            user,
        ),
    )
    return ranked[start:stop]


def cohort_sha256(users: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(users) + "\n").encode()).hexdigest()


def fold_views(
    histories: Mapping[str, Sequence[str]], fold: str
) -> dict[str, tuple[tuple[str, ...], str]]:
    if fold not in S18_1_FOLDS:
        raise PermissionError(f"S18-1 may not construct fold {fold}")
    views: dict[str, tuple[tuple[str, ...], str]] = {}
    for user, history in histories.items():
        try:
            views[user] = internal_fold_view(history, fold)
        except ValueError:
            continue
    return views


def lower_empirical_quartile(frequencies: Iterable[int]) -> int:
    values = sorted(int(value) for value in frequencies if int(value) > 0)
    if not values:
        raise ValueError("positive fold-visible frequencies are required")
    return values[math.floor(0.25 * (len(values) - 1))]


def first_drop_depth(
    active_prefixes: Mapping[int, Iterable[Sequence[int]]],
    target_path: Sequence[int],
) -> int | None:
    """Return first missing generated-token prefix; decoder start is excluded."""
    target = tuple(int(token) for token in target_path)
    if not target:
        raise ValueError("target path must contain at least one generated token")
    for depth in range(1, len(target) + 1):
        active = {tuple(int(token) for token in row) for row in active_prefixes.get(depth, ())}
        if target[:depth] not in active:
            return depth
    return None


def actual_pruner_items(
    returned_paths: Mapping[str, Sequence[int]],
    target_path: Sequence[int],
    drop_depth: int,
    legal_children: Iterable[int],
) -> set[str]:
    """Find legal returned item paths choosing a sibling at the first-drop node."""
    target = tuple(int(token) for token in target_path)
    if not 1 <= drop_depth <= len(target):
        raise ValueError("drop depth is outside target path")
    legal = {int(token) for token in legal_children}
    parent = target[: drop_depth - 1]
    target_child = target[drop_depth - 1]
    result = set()
    for item, raw_path in returned_paths.items():
        path = tuple(int(token) for token in raw_path)
        if (
            len(path) >= drop_depth
            and path[: drop_depth - 1] == parent
            and path[drop_depth - 1] != target_child
            and path[drop_depth - 1] in legal
        ):
            result.add(item)
    return result


def hard_negative_recall(selected: Iterable[str], actual: Iterable[str]) -> float:
    actual_set = set(actual)
    if not actual_set:
        raise ValueError("actual-pruner denominator is empty")
    return len(set(selected) & actual_set) / len(actual_set)


def capped_recall_metrics(
    events: Iterable[tuple[int, int]],
    k: int,
) -> dict[str, float | int]:
    """Summarize top-k coverage without penalizing an impossible denominator.

    Each event is ``(intersection, actual_pruner_count)``.  Raw micro recall is
    retained for continuity with S18-1, while capacity-normalized recall uses
    ``min(k, actual_pruner_count)`` as the per-event attainable denominator.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rows = [(int(intersection), int(actual)) for intersection, actual in events]
    if not rows:
        raise ValueError("at least one nonempty actual-pruner event is required")
    for intersection, actual in rows:
        if actual <= 0:
            raise ValueError("actual-pruner count must be positive")
        if intersection < 0 or intersection > min(k, actual):
            raise ValueError("intersection is outside the attainable top-k range")

    intersection_total = sum(intersection for intersection, _ in rows)
    actual_total = sum(actual for _, actual in rows)
    capacity_total = sum(min(k, actual) for _, actual in rows)
    return {
        "event_count": len(rows),
        "intersection_total": intersection_total,
        "actual_pruner_total": actual_total,
        "topk_capacity_total": capacity_total,
        "raw_micro_recall": intersection_total / actual_total,
        "raw_micro_mechanical_ceiling": capacity_total / actual_total,
        "capacity_normalized_recall": intersection_total / capacity_total,
        "macro_event_recall": sum(
            intersection / actual for intersection, actual in rows
        )
        / len(rows),
        "event_any_hit_rate": sum(intersection > 0 for intersection, _ in rows)
        / len(rows),
        "event_full_coverage_rate": sum(
            intersection == actual for intersection, actual in rows
        )
        / len(rows),
    }


def evaluate_capacity_gate(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "capacity_normalized_recall": float(metrics["capacity_normalized_recall"])
        >= float(gates["capacity_normalized_recall_min"]),
        "event_any_hit_rate": float(metrics["event_any_hit_rate"])
        >= float(gates["event_any_hit_rate_min"]),
    }
    return {
        "decision": (
            "CAPACITY_ADJUSTED_ACTIONABILITY_PASS"
            if all(checks.values())
            else "CAPACITY_ADJUSTED_ACTIONABILITY_FAIL"
        ),
        "checks": checks,
    }


def catalog_standardized_target(target_score: float, catalog_scores: Sequence[float]) -> float:
    values = [float(value) for value in catalog_scores]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("catalog scores must be finite and nonempty")
    target = float(target_score)
    if not math.isfinite(target):
        raise ValueError("target score must be finite")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0:
        raise ValueError("catalog score variance must be positive")
    return (target - mean) / math.sqrt(variance)


def evaluate_domain_gate(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, Any]:
    finite_legal = bool(metrics["finite_and_trie_legal"])
    checks = {
        "gate1_headroom": float(metrics["pooled_headroom"]) >= float(gates["per_domain_headroom_min"]),
        "gate2_beam200_only_events": int(metrics["beam200_only_events"]) >= int(gates["per_domain_beam200_only_events_min"]),
        "gate3_nonempty_actual_pruner": float(metrics["nonempty_actual_pruner_fraction"]) >= float(gates["per_domain_nonempty_actual_pruner_fraction_min"]),
        "gate4_k8_actual_pruner_recall": float(metrics["k8_actual_pruner_recall"]) >= float(gates["per_domain_k8_actual_pruner_recall_min"]),
        "gate5_finite_and_trie_legal": finite_legal,
        "gate6_cf_target_z_drift": abs(float(metrics["cf_target_z_mean_drift"])) < float(gates["absolute_cf_target_z_mean_drift_max_exclusive"]),
    }
    if not all(checks[name] for name in tuple(checks)[:5]):
        decision = str(gates["gate_1_5_failure"])
    elif not checks["gate6_cf_target_z_drift"]:
        decision = str(gates["gate_6_failure"])
    else:
        decision = "ACTIONABILITY_PASS"
    return {"decision": decision, "checks": checks}
