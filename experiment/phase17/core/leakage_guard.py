"""Fail-closed data ACLs for rolling-origin Stage 17 experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_COMPONENTS = {
    "sports",
    "official_test",
    "test_predictions",
    "pred_test.tsv",
}


@dataclass(frozen=True)
class DatasetACL:
    allowed_roots: tuple[Path, ...]
    allowed_folds: frozenset[str] = frozenset({"D0"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_roots",
            tuple(path.resolve() for path in self.allowed_roots),
        )

    def authorize(self, path: str | os.PathLike[str], fold: str, purpose: str) -> Path:
        resolved = Path(path).resolve()
        if fold not in self.allowed_folds:
            raise PermissionError(f"fold {fold} is not authorized by this job")
        lowered = {part.lower() for part in resolved.parts}
        if any(token in lowered for token in FORBIDDEN_COMPONENTS):
            raise PermissionError(f"forbidden dataset component in {resolved}")
        if purpose not in {"train", "validation", "catalog", "manifest"}:
            raise PermissionError(f"unauthorized read purpose: {purpose}")
        if not any(resolved == root or root in resolved.parents for root in self.allowed_roots):
            raise PermissionError(f"path is outside the frozen dataset view: {resolved}")
        return resolved


def assert_no_future_read(event_positions: Iterable[int], cutoff: int) -> None:
    illegal = sorted(position for position in event_positions if position >= cutoff)
    if illegal:
        raise PermissionError(f"cutoff/future event positions were read: {illegal[:5]}")


def assert_fold_isolation(
    train_user_targets: dict[str, str], validation_user_targets: dict[str, str]
) -> None:
    leaked = [
        user
        for user, validation_target in validation_user_targets.items()
        if train_user_targets.get(user) == validation_target
    ]
    if leaked:
        raise PermissionError(f"validation targets entered the train fold: {leaked[:5]}")


def build_train_only_transitions(sequences: Iterable[list[str]], cutoffs: Iterable[int]):
    transitions: dict[str, dict[str, int]] = {}
    for sequence, cutoff in zip(sequences, cutoffs):
        assert 0 <= cutoff <= len(sequence)
        prefix = sequence[:cutoff]
        for left, right in zip(prefix, prefix[1:]):
            transitions.setdefault(left, {})[right] = transitions.setdefault(left, {}).get(right, 0) + 1
    return transitions
