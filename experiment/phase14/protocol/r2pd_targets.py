"""Pure R2PD prefix-target construction used by Stage 14 M2.

The module deliberately has no GRAM or dataset dependency.  It converts a
stop-gradient item distribution into absolute-mass, prefix-conditional targets
and exposes small loss helpers that can be unit-tested before the training
stack is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import torch
import torch.nn.functional as F


Token = Hashable
Prefix = tuple[Token, ...]


@dataclass(frozen=True)
class PrefixTarget:
    prefix: Prefix
    mass: float
    children: tuple[Token, ...]
    probabilities: tuple[float, ...]


def assert_unique_paths(item_paths: Mapping[str, Sequence[Token]]) -> None:
    """Reject item aliases before any item mass is projected to a path."""
    reverse: dict[Prefix, str] = {}
    for item, raw_path in item_paths.items():
        path = tuple(raw_path)
        if not path:
            raise ValueError(f"Empty path for item {item}")
        previous = reverse.get(path)
        if previous is not None:
            raise ValueError(f"Path collision: {previous} and {item} share {path}")
        reverse[path] = item


def build_prefix_targets(
    item_mass: Mapping[str, float],
    item_paths: Mapping[str, Sequence[Token]],
    *,
    min_prefix_mass: float = 0.0,
    total_distribution_mass: float = 1.0,
) -> tuple[list[PrefixTarget], float]:
    """Project top-M item mass into absolute prefix mass and next-token Q.

    ``item_mass`` is not renormalized.  Consequently the returned prefix mass
    retains the teacher's absolute confidence, while ``probabilities`` are
    normalized only among children of the same prefix.  The missing catalog
    mass is returned as ``tail_mass`` for mandatory provenance reporting.
    """
    if min_prefix_mass < 0:
        raise ValueError("min_prefix_mass must be non-negative")
    if not 0 < total_distribution_mass <= 1.0 + 1e-9:
        raise ValueError("total_distribution_mass must be in (0, 1]")
    assert_unique_paths({item: item_paths[item] for item in item_mass})

    prefix_mass: dict[Prefix, float] = {}
    child_mass: dict[Prefix, dict[Token, float]] = {}
    observed_mass = 0.0
    for item, raw_mass in item_mass.items():
        if item not in item_paths:
            raise KeyError(f"Teacher item lacks a path: {item}")
        mass = float(raw_mass)
        if not torch.isfinite(torch.tensor(mass)) or mass < 0:
            raise ValueError(f"Invalid teacher mass for {item}: {mass}")
        if mass == 0:
            continue
        observed_mass += mass
        path = tuple(item_paths[item])
        for depth, child in enumerate(path):
            prefix = path[:depth]
            prefix_mass[prefix] = prefix_mass.get(prefix, 0.0) + mass
            bucket = child_mass.setdefault(prefix, {})
            bucket[child] = bucket.get(child, 0.0) + mass

    if observed_mass > total_distribution_mass + 1e-7:
        raise ValueError(
            f"Observed top-M mass {observed_mass} exceeds total mass "
            f"{total_distribution_mass}"
        )

    targets: list[PrefixTarget] = []
    for prefix in sorted(prefix_mass, key=lambda value: (len(value), repr(value))):
        mass = prefix_mass[prefix]
        if mass < min_prefix_mass:
            continue
        children = tuple(sorted(child_mass[prefix], key=repr))
        probabilities = tuple(child_mass[prefix][child] / mass for child in children)
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise RuntimeError(f"Conditional target does not normalize at {prefix}")
        targets.append(PrefixTarget(prefix, mass, children, probabilities))

    return targets, max(0.0, total_distribution_mass - observed_mass)


def prefix_distillation_loss(
    student_logits: Sequence[torch.Tensor],
    targets: Sequence[PrefixTarget],
    *,
    confidence: Sequence[float] | None = None,
) -> torch.Tensor:
    """Mass-weighted KL with an explicit zero-effective-prefix branch."""
    if len(student_logits) != len(targets):
        raise ValueError("student_logits and targets must have equal length")
    if confidence is None:
        confidence = [1.0] * len(targets)
    if len(confidence) != len(targets):
        raise ValueError("confidence and targets must have equal length")
    if not student_logits:
        return torch.zeros((), dtype=torch.float32)

    weighted: list[torch.Tensor] = []
    normalizer = 0.0
    for logits, target, raw_confidence in zip(student_logits, targets, confidence):
        if logits.ndim != 1 or logits.numel() != len(target.children):
            raise ValueError(f"Logit shape mismatch at prefix {target.prefix}")
        conf = float(raw_confidence)
        if not 0 <= conf <= 1 or not torch.isfinite(torch.tensor(conf)):
            raise ValueError(f"Invalid confidence at {target.prefix}: {conf}")
        weight = target.mass * conf
        if weight == 0:
            continue
        q = torch.tensor(target.probabilities, dtype=logits.dtype, device=logits.device)
        log_q = q.clamp_min(torch.finfo(q.dtype).tiny).log()
        kl = torch.sum(q * (log_q - F.log_softmax(logits, dim=-1)))
        weighted.append(weight * kl)
        normalizer += weight
    if not weighted:
        return student_logits[0].sum() * 0.0
    return torch.stack(weighted).sum() / normalizer


def retention_loss(
    student_logits: torch.Tensor,
    frozen_logits: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL[frozen-v0 || student] with the teacher forcibly detached."""
    if student_logits.shape != frozen_logits.shape:
        raise ValueError("student and frozen logits must have identical shapes")
    if legal_mask is not None:
        if legal_mask.shape != student_logits.shape or legal_mask.dtype != torch.bool:
            raise ValueError("legal_mask must be bool and match logits")
        if not legal_mask.any(dim=-1).all():
            raise ValueError("Every row needs at least one legal child")
        floor = torch.finfo(student_logits.dtype).min
        student_logits = student_logits.masked_fill(~legal_mask, floor)
        frozen_logits = frozen_logits.masked_fill(~legal_mask, floor)
    frozen_q = F.softmax(frozen_logits.detach(), dim=-1)
    frozen_log_q = F.log_softmax(frozen_logits.detach(), dim=-1)
    return torch.sum(
        frozen_q * (frozen_log_q - F.log_softmax(student_logits, dim=-1)), dim=-1
    ).mean()


def compose_r2pd_loss(
    warm_ce: torch.Tensor,
    cold_prefix_kd: torch.Tensor,
    warm_retention: torch.Tensor,
    *,
    lambda_cp: float,
    mu_keep: float,
) -> torch.Tensor:
    return warm_ce + float(lambda_cp) * cold_prefix_kd + float(mu_keep) * warm_retention
