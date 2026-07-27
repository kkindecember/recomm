#!/usr/bin/env python3
"""Locked HBTR-B1 objective and cache validation helpers.

This module is deliberately independent of the GRAM runner so its leakage,
weighting, fallback, and monotonicity properties can be tested on CPU before
any GPU smoke is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


BASE_MARGIN = 0.1
PREFIX_DEPTH_CAP = 3
RANKING_LAMBDA = 0.1
NEGATIVE_COUNT = 4


def common_prefix_depth(left: Sequence[str], right: Sequence[str]) -> int:
    depth = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        depth += 1
    return depth


def prefix_weight(depth: int, cap: int = PREFIX_DEPTH_CAP) -> float:
    if depth < 0 or cap <= 0:
        raise ValueError("prefix depth must be non-negative and cap must be positive")
    return 1.0 + min(depth, cap) / cap


def tail_weight(positive_frequency: int, median_positive_frequency: float) -> float:
    if positive_frequency < 0 or median_positive_frequency < 0:
        raise ValueError("training frequencies must be non-negative")
    log_ratio = math.log(
        (float(median_positive_frequency) + 1.0) / (positive_frequency + 1.0)
    )
    return 1.0 + min(1.0, max(0.0, log_ratio))


def joint_margin(
    prefix_depth: int,
    positive_frequency: int,
    median_positive_frequency: float,
    *,
    base_margin: float = BASE_MARGIN,
) -> float:
    if base_margin < 0:
        raise ValueError("base margin must be non-negative")
    return (
        base_margin
        * prefix_weight(prefix_depth)
        * tail_weight(positive_frequency, median_positive_frequency)
    )


def component_margin(
    control: str,
    prefix_depth: int,
    positive_frequency: int,
    median_positive_frequency: float,
    *,
    base_margin: float = BASE_MARGIN,
) -> float:
    """Return the preregistered C1-C4 margin without scanning parameters."""
    if control == "C1":
        return base_margin
    if control == "C2":
        return base_margin * prefix_weight(prefix_depth)
    if control == "C3":
        return base_margin * tail_weight(
            positive_frequency, median_positive_frequency
        )
    if control == "C4":
        return joint_margin(
            prefix_depth,
            positive_frequency,
            median_positive_frequency,
            base_margin=base_margin,
        )
    raise ValueError(f"ranking margin is undefined for control {control!r}")


def sequence_log_scores(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return mean teacher-forced log p(label|input), including EOS.

    Padding labels must be -100. Rows with no valid token fail closed.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,L,V] and labels [B,L]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit and label batch/length dimensions do not match")
    mask = labels.ne(-100)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each sequence must contain at least one scored token")
    safe_labels = labels.masked_fill(~mask, 0)
    token_logp = F.log_softmax(logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logp * mask).sum(dim=1) / mask.sum(dim=1)


def pairwise_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margins: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Locked softplus margin loss over [B,K] negatives."""
    if positive_scores.ndim != 1 or negative_scores.ndim != 2:
        raise ValueError("expected positive [B] and negative [B,K] scores")
    if negative_scores.shape != margins.shape:
        raise ValueError("negative scores and margins must have the same shape")
    if positive_scores.shape[0] != negative_scores.shape[0]:
        raise ValueError("positive and negative batch sizes do not match")
    losses = F.softplus(margins + negative_scores - positive_scores[:, None])
    if valid_mask is None:
        valid_mask = torch.ones_like(losses, dtype=torch.bool)
    if valid_mask.shape != losses.shape:
        raise ValueError("valid mask shape does not match pairwise losses")
    if not torch.any(valid_mask):
        return positive_scores.sum() * 0.0
    return losses.masked_select(valid_mask).mean()


def total_loss(
    token_ce: torch.Tensor,
    ranking_loss: torch.Tensor,
    ranking_lambda: float = RANKING_LAMBDA,
) -> torch.Tensor:
    if ranking_lambda < 0:
        raise ValueError("ranking lambda must be non-negative")
    return token_ce + ranking_lambda * ranking_loss


def training_popularity(sequences: Mapping[str, Sequence[str]]) -> Counter:
    """Count only sequence[:-2], excluding validation and test targets."""
    counts: Counter = Counter()
    for items in sequences.values():
        counts.update(items[:-2])
    return counts


def validate_cache_row(
    row: Mapping,
    *,
    valid_items: set[str],
    expected_negative_count: int = NEGATIVE_COUNT,
) -> None:
    required = {
        "sample_key",
        "user_id",
        "positive_item",
        "positive_rank",
        "history_items",
        "negative_items",
        "prefix_depths",
        "positive_frequency",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"cache row missing fields: {sorted(missing)}")
    positive = row["positive_item"]
    negatives = list(row["negative_items"])
    history = set(row["history_items"])
    if not 11 <= int(row["positive_rank"]) <= 50:
        raise ValueError("HBTR cache requires positive rank in [11,50]")
    if len(negatives) != expected_negative_count:
        raise ValueError("cache row does not match locked negative count")
    if len(negatives) != len(set(negatives)):
        raise ValueError("cache row contains duplicate negatives")
    if positive in negatives or any(item in history for item in negatives):
        raise ValueError("cache row contains target/history leakage")
    if positive not in valid_items or any(item not in valid_items for item in negatives):
        raise ValueError("cache row contains item outside the locked Trie")
    if len(row["prefix_depths"]) != len(negatives):
        raise ValueError("prefix depths do not align with negatives")
    if int(row["positive_frequency"]) < 0:
        raise ValueError("positive training frequency must be non-negative")


def canonical_cache_sha256(rows: Iterable[Mapping]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cache(path: Path, valid_items: set[str]) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["rows"]
    for row in rows:
        validate_cache_row(row, valid_items=valid_items)
    expected = payload.get("rows_sha256")
    actual = canonical_cache_sha256(rows)
    if expected != actual:
        raise ValueError(f"cache hash mismatch: expected={expected} actual={actual}")
    return rows
