"""Fail-closed evidence and data contracts for Stage18.

This module intentionally has no model or GPU dependency.  S18-0 uses it to
freeze what later Stage18 jobs may read and what historical claims they may
inherit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def json_pointer(payload: Any, pointer: str) -> Any:
    """Resolve the RFC 6901 subset used by the frozen contracts."""
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer}")
    value = payload
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise KeyError(f"cannot descend through {pointer!r} at {token!r}")
    return value


def values_match(actual: Any, expected: Any, tolerance: float) -> tuple[bool, float | None]:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return actual == expected, None
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False, None
    difference = abs(float(actual) - float(expected))
    return math.isfinite(difference) and difference <= tolerance, difference


def normalize_repo_relative(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise PermissionError(f"path is not a normalized repository-relative path: {raw}")
    return candidate.as_posix()


def authorize_path(path: str | Path, profile: str, contract: dict[str, Any]) -> str:
    normalized = normalize_repo_relative(path)
    profiles = contract.get("access_profiles", {})
    if profile not in profiles:
        raise PermissionError(f"unknown Stage18 access profile: {profile}")
    allowed = set(profiles[profile].get("exact_allowlist", []))
    if normalized not in allowed:
        raise PermissionError(f"Stage18 {profile} denied read: {normalized}")
    return normalized


def load_shadow_train_prefix_line(line: str) -> tuple[str, tuple[str, ...]]:
    """Return only user and D0 train-prefix; never expose the two sealed suffixes."""
    stripped = line.rstrip("\r\n")
    try:
        prefix_blob = stripped.rsplit(maxsplit=2)[0]
    except (AttributeError, IndexError) as error:
        raise ValueError("malformed Stage18 shadow row") from error
    fields = prefix_blob.split()
    if len(fields) < 2:
        raise ValueError("Stage18 shadow row has no train-prefix history")
    return fields[0], tuple(fields[1:])


FOLD_OFFSETS = {"I-1": 4, "I0": 3, "I1": 2, "I2": 1}


def internal_fold_view(history: Iterable[str], fold: str) -> tuple[tuple[str, ...], str]:
    items = tuple(history)
    if fold not in FOLD_OFFSETS:
        raise ValueError(f"unknown internal fold: {fold}")
    offset = FOLD_OFFSETS[fold]
    if len(items) <= offset:
        raise ValueError(f"history is ineligible for {fold}")
    return items[:-offset], items[-offset]


def metrics_from_ranks(ranks: Iterable[int]) -> dict[str, float | int]:
    values = tuple(int(rank) for rank in ranks)
    if not values or any(rank < 1 or rank > 51 for rank in values):
        raise ValueError("rank cache must contain integers in [1, 51]")
    count = len(values)
    result: dict[str, float | int] = {
        "count": count,
        "mrr": sum((1.0 / rank) if rank <= 50 else 0.0 for rank in values) / count,
    }
    for cutoff in (1, 5, 10, 20, 50):
        hits = [rank <= cutoff for rank in values]
        result[f"Hit@{cutoff}"] = sum(hits) / count
        result[f"NDCG@{cutoff}"] = sum(
            (1.0 / math.log2(rank + 1)) if hit else 0.0
            for rank, hit in zip(values, hits)
        ) / count
    return result
