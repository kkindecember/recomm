#!/usr/bin/env python3
"""Leak-safe, resumable Stage16 GenRecEdit request construction.

The builder deliberately consumes only the S16-1 student-readable interaction
train split.  Validation, internal-dev, held pseudo-cold ground truth, and test
artifacts are not accepted as occurrence sources.  Real-cold items enter only
as frozen catalog edit targets; their contexts are borrowed from content-nearest
retained-warm items with an actual train-only occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in an invalid runtime.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "stage16_s3_genrecedit_train_only_requests_v1"
TOYS_FROZEN_COUNTS = {"targets": 5963, "contexts": 59630, "requests": 302400}
TOYS_FROZEN_COVARIANCE_POSITION_COUNTS = {
    0: 27659,
    1: 27659,
    2: 27659,
    3: 27659,
    4: 27659,
    5: 2036,
}
FORBIDDEN_OCCURRENCE_PATH_PARTS = (
    "held_ground_truth",
    "internal_dev",
    "validation",
    "test",
    "pseudo_cold_events",
)


@dataclass(frozen=True)
class DatasetInputs:
    train_sequences: Path
    retained_warm_items: Path
    pseudo_cold_items: Path
    real_cold_items: Path
    lexical_paths: Path
    metadata: Path
    content_embeddings: Path
    split_manifest: Path | None = None
    expected_sha256: Mapping[str, str] | None = None


@dataclass(frozen=True)
class TrainOccurrence:
    user_id: str
    position: int
    history: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def _json_line(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    allowed_roots = [ROOT.resolve()]
    for relative in ("artifacts", "GRAM/rec_datasets"):
        shared = ROOT / relative
        if shared.is_symlink():
            allowed_roots.append(shared.resolve())
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError(f"Input escapes repository root: {value}")
    return path


def repo_relative_path(path: Path) -> str:
    """Return the logical repo path, including allowlisted parent symlinks."""

    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return str(resolved.relative_to(root))
    for relative in (Path("artifacts"), Path("GRAM/rec_datasets")):
        shared = ROOT / relative
        if not shared.is_symlink():
            continue
        target = shared.resolve()
        if resolved == target or target in resolved.parents:
            return str(relative / resolved.relative_to(target))
    raise ValueError(f"Input escapes repository root: {path}")


def _regular_input(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _assert_occurrence_path_is_train_only(path: Path) -> None:
    lowered_parts = [part.lower() for part in path.parts]
    lowered = "/".join(lowered_parts)
    if path.name != "interaction_train_sequences.jsonl":
        raise ValueError(
            "Occurrence source must be the S16-1 student-readable "
            "interaction_train_sequences.jsonl"
        )
    if any(token in lowered for token in FORBIDDEN_OCCURRENCE_PATH_PARTS):
        raise ValueError(f"Forbidden occurrence-source path: {path}")


def read_item_set(path: Path, label: str) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and duplicate-free")
    return set(values)


def read_lexical_paths(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    reverse: dict[tuple[str, ...], str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed lexical path at line {line_number}")
        item, serialized = fields
        lexical = tuple(token for token in serialized.split("|") if token)
        if not lexical or item in result:
            raise ValueError(f"Empty path or duplicate item at line {line_number}")
        if lexical in reverse:
            raise ValueError(f"Lexical collision between {reverse[lexical]} and {item}")
        result[item] = lexical
        reverse[lexical] = item
    if not result:
        raise ValueError("Empty lexical catalog")
    return result


def read_metadata_ids(path: Path) -> set[str]:
    ids: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        item, separator, text = raw.partition(" ")
        if not separator or not item or not text.strip():
            raise ValueError(f"Malformed metadata row {line_number}")
        ids.append(item)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Metadata item IDs must be non-empty and duplicate-free")
    return set(ids)


def read_train_sequences(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    _assert_occurrence_path_is_train_only(path)
    rows: list[tuple[str, tuple[str, ...]]] = []
    seen_users: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed train JSON at line {line_number}") from error
            if not isinstance(payload, dict) or set(payload) != {"user_id", "items"}:
                raise ValueError(f"Unexpected train schema at line {line_number}")
            user = payload["user_id"]
            items = payload["items"]
            if not isinstance(user, str) or not user or user in seen_users:
                raise ValueError(f"Invalid or duplicate train user at line {line_number}")
            if (
                not isinstance(items, list)
                or len(items) < 2
                or any(not isinstance(item, str) or not item for item in items)
            ):
                raise ValueError(f"Invalid train items at line {line_number}")
            seen_users.add(user)
            rows.append((user, tuple(items)))
    if not rows:
        raise ValueError("Empty S16-1 train sequence file")
    return rows


def collect_train_occurrences(
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    retained_warm: set[str],
    max_history: int,
) -> dict[str, tuple[TrainOccurrence, ...]]:
    """Collect non-empty-history, retained-warm occurrences from train only."""

    if max_history < 1:
        raise ValueError("max_history must be positive")
    occurrences: dict[str, list[TrainOccurrence]] = defaultdict(list)
    for user, items_like in sorted(rows, key=lambda row: row[0]):
        items = tuple(items_like)
        for position in range(1, len(items)):
            item = items[position]
            if item not in retained_warm:
                raise ValueError(f"Non-retained-warm train occurrence: {item}")
            history = tuple(items[max(0, position - max_history) : position])
            if not history or set(history) - retained_warm:
                raise ValueError("Train history escaped the retained-warm universe")
            occurrences[item].append(TrainOccurrence(user, position, history))
    return {
        item: tuple(sorted(values, key=lambda row: (row.user_id, row.position, row.history)))
        for item, values in occurrences.items()
    }


def iter_covariance_position_rows(
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    paths: Mapping[str, Sequence[str]],
    retained_warm: set[str],
    max_history: int,
    seed: int = 1502,
) -> Iterator[dict[str, Any]]:
    """Stream one legal covariance row per train transition and path position.

    This is intentionally independent of pseudo-context construction.  The row
    source is the same S16-1 retained-warm next-item transition population used
    by the adaptation protocol, never Stage15's context artifact.
    """

    if max_history < 1:
        raise ValueError("max_history must be positive")
    for user, items_like in sorted(rows, key=lambda row: row[0]):
        items = tuple(items_like)
        for transition_position in range(1, len(items)):
            target_item = items[transition_position]
            history = tuple(items[max(0, transition_position - max_history) : transition_position])
            if target_item not in retained_warm or not history or set(history) - retained_warm:
                raise ValueError("Covariance row escaped retained-warm train-only scope")
            if target_item not in paths:
                raise ValueError(f"Covariance target lacks lexical path: {target_item}")
            lexical = tuple(paths[target_item])
            occurrence_hash = _stable_id(
                seed, "covariance-occurrence", user, transition_position, target_item
            )
            for lexical_position, token in enumerate(lexical):
                yield {
                    "covariance_row_id": _stable_id(
                        seed, "covariance-row", occurrence_hash, lexical_position, token
                    ),
                    "lexical_position": lexical_position,
                    "occurrence_hash": occurrence_hash,
                    "prefix_tokens": list(lexical[:lexical_position]),
                    "target_item": target_item,
                    "target_token": token,
                    "train_context_items": list(history),
                }


def covariance_position_coverage(
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    paths: Mapping[str, Sequence[str]],
    retained_warm: set[str],
    max_history: int,
    seed: int = 1502,
) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for row in iter_covariance_position_rows(
        rows,
        paths=paths,
        retained_warm=retained_warm,
        max_history=max_history,
        seed=seed,
    ):
        counts[int(row["lexical_position"])] += 1
    return dict(sorted(counts.items()))


def deterministic_long_path_resource_subset(
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    paths: Mapping[str, Sequence[str]],
    retained_warm: set[str],
    max_history: int,
    seed: int = 1502,
    lexical_position: int = 5,
    minimum_rows: int = 8,
) -> list[dict[str, Any]]:
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    candidates = [
        row
        for row in iter_covariance_position_rows(
            rows,
            paths=paths,
            retained_warm=retained_warm,
            max_history=max_history,
            seed=seed,
        )
        if row["lexical_position"] == lexical_position
    ]
    if len(candidates) < minimum_rows:
        raise ValueError(
            f"Need at least {minimum_rows} legal position-{lexical_position} covariance rows, "
            f"found {len(candidates)}"
        )
    return sorted(
        candidates,
        key=lambda row: (stable_sha256([seed, "resource-subset", row["covariance_row_id"]]), row["covariance_row_id"]),
    )[:minimum_rows]


def choose_occurrence(
    occurrences: Sequence[TrainOccurrence],
    *,
    cold_item: str,
    warm_item: str,
    seed: int,
) -> TrainOccurrence:
    if not occurrences:
        raise ValueError(f"No train-only occurrence for retained-warm item {warm_item}")
    return min(
        occurrences,
        key=lambda row: (
            _stable_id(seed, "occurrence", cold_item, warm_item, row.user_id, row.position),
            row.user_id,
            row.position,
        ),
    )


def deterministic_topk(
    scores: "torch.Tensor", item_ids: Sequence[str], k: int
) -> list[tuple[int, float]]:
    """Return descending-score top-k with ascending item-ID tie breaking."""

    if torch is None:
        raise RuntimeError("PyTorch is required for content-nearest context selection")
    if scores.ndim != 1 or scores.numel() != len(item_ids):
        raise ValueError("Score vector and item IDs do not align")
    if k < 1 or k > len(item_ids):
        raise ValueError("Invalid deterministic top-k")
    tensor = scores.detach().cpu()
    values = [float(value) for value in tensor.tolist()]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite content similarity")
    cutoff = float(torch.topk(tensor, k=k, sorted=False).values.min())
    strict = [index for index, value in enumerate(values) if value > cutoff]
    tied = [index for index, value in enumerate(values) if value == cutoff]
    strict.sort(key=lambda index: (-values[index], item_ids[index]))
    tied.sort(key=lambda index: item_ids[index])
    chosen = (strict + tied)[:k]
    return [(index, values[index]) for index in chosen]


def load_content_embeddings(
    path: Path,
    *,
    metadata_path: Path,
    catalog: set[str],
) -> tuple[list[str], "torch.Tensor", dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to read the frozen content embeddings")
    payload = torch.load(path, map_location="cpu")
    required = {
        "item_ids",
        "embeddings",
        "model_name",
        "pooling",
        "l2_normalized",
        "text_source_sha256",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unexpected content-embedding manifest")
    item_ids = [str(item) for item in payload["item_ids"]]
    embeddings = payload["embeddings"].detach().float().cpu().contiguous()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(item_ids):
        raise ValueError("Content embeddings do not align with item IDs")
    if len(item_ids) != len(set(item_ids)) or set(item_ids) != catalog:
        raise ValueError("Content embedding item IDs do not exactly match the catalog")
    if payload["l2_normalized"] is not True:
        raise ValueError("Content embeddings must be frozen L2-normalized vectors")
    norms = embeddings.norm(dim=1)
    if not torch.isfinite(embeddings).all() or not torch.allclose(
        norms, torch.ones_like(norms), atol=2e-4, rtol=0
    ):
        raise ValueError("Content embeddings violate the finite L2-normalization contract")
    metadata_sha = sha256_file(metadata_path)
    if str(payload["text_source_sha256"]) != metadata_sha:
        raise ValueError("Embedding text provenance does not match frozen metadata")
    manifest = {
        "model_name": str(payload["model_name"]),
        "pooling": str(payload["pooling"]),
        "l2_normalized": True,
        "shape": [int(value) for value in embeddings.shape],
        "text_source_sha256": metadata_sha,
        "content_embeddings_sha256": sha256_file(path),
    }
    return item_ids, embeddings, manifest


def _verify_expected_hashes(inputs: DatasetInputs) -> dict[str, str]:
    paths = {
        "train_sequences": inputs.train_sequences,
        "retained_warm_items": inputs.retained_warm_items,
        "pseudo_cold_items": inputs.pseudo_cold_items,
        "real_cold_items": inputs.real_cold_items,
        "lexical_paths": inputs.lexical_paths,
        "metadata": inputs.metadata,
        "content_embeddings": inputs.content_embeddings,
    }
    if inputs.split_manifest is not None:
        paths["split_manifest"] = inputs.split_manifest
    actual: dict[str, str] = {}
    for label, path in paths.items():
        _regular_input(path, label)
        actual[label] = sha256_file(path)
    for label, expected in dict(inputs.expected_sha256 or {}).items():
        if label not in actual:
            raise ValueError(f"Expected SHA supplied for unknown input {label}")
        if actual[label] != expected:
            raise ValueError(
                f"Frozen SHA drift for {label}: expected {expected}, got {actual[label]}"
            )
    return actual


def _load_and_audit_inputs(
    inputs: DatasetInputs,
    *,
    max_history: int,
) -> dict[str, Any]:
    input_sha = _verify_expected_hashes(inputs)
    rows = read_train_sequences(inputs.train_sequences)
    retained_warm = read_item_set(inputs.retained_warm_items, "retained warm items")
    pseudo_cold = read_item_set(inputs.pseudo_cold_items, "pseudo-cold items")
    real_cold = read_item_set(inputs.real_cold_items, "real-cold items")
    paths = read_lexical_paths(inputs.lexical_paths)
    catalog = set(paths)
    metadata_ids = read_metadata_ids(inputs.metadata)
    if metadata_ids != catalog:
        raise ValueError("Metadata IDs do not exactly match lexical catalog")
    if retained_warm & pseudo_cold or retained_warm & real_cold or pseudo_cold & real_cold:
        raise ValueError("retained-warm, pseudo-cold, and real-cold sets must be disjoint")
    if retained_warm | pseudo_cold | real_cold != catalog:
        raise ValueError("Stage16 retained-warm/pseudo-cold/real-cold partition is incomplete")

    train_items = [item for _, items in rows for item in items]
    train_set = set(train_items)
    real_leaks = train_set & real_cold
    pseudo_leaks = train_set & pseudo_cold
    non_retained = train_set - retained_warm
    if real_leaks or pseudo_leaks or non_retained:
        raise ValueError(
            "Student-readable train leakage: "
            f"real_cold={len(real_leaks)}, pseudo_cold={len(pseudo_leaks)}, "
            f"non_retained={len(non_retained)}"
        )
    occurrences = collect_train_occurrences(
        rows, retained_warm=retained_warm, max_history=max_history
    )
    eligible_warm = sorted(occurrences)
    if not eligible_warm:
        raise ValueError("No retained-warm item has a train-only occurrence")

    embedding_ids, embeddings, embedding_manifest = load_content_embeddings(
        inputs.content_embeddings, metadata_path=inputs.metadata, catalog=catalog
    )
    embedding_index = {item: index for index, item in enumerate(embedding_ids)}

    split_manifest: dict[str, Any] | None = None
    if inputs.split_manifest is not None:
        split_manifest = json.loads(inputs.split_manifest.read_text(encoding="utf-8"))
        audit = split_manifest.get("leakage_audit", {})
        required_zero = (
            "real_cold_in_student_items",
            "pseudo_cold_in_student_items",
            "train_internal_dev_user_overlap",
        )
        if any(audit.get(key) != 0 for key in required_zero):
            raise ValueError("S16-1 split manifest does not carry a zero-leakage audit")
        if audit.get("test_files_opened") is not False:
            raise ValueError("S16-1 split manifest does not prove test sealing")

    return {
        "input_sha256": input_sha,
        "rows": rows,
        "retained_warm": retained_warm,
        "pseudo_cold": pseudo_cold,
        "real_cold": real_cold,
        "paths": paths,
        "occurrences": occurrences,
        "eligible_warm": eligible_warm,
        "embedding_ids": embedding_ids,
        "embeddings": embeddings,
        "embedding_index": embedding_index,
        "embedding_manifest": embedding_manifest,
        "split_manifest": split_manifest,
        "train_items": len(train_items),
        "train_transitions": sum(len(items) - 1 for _, items in rows),
    }


def _expected_counts(
    real_cold: set[str], paths: Mapping[str, Sequence[str]], contexts_per_target: int
) -> dict[str, int]:
    return {
        "targets": len(real_cold),
        "contexts": len(real_cold) * contexts_per_target,
        "requests": sum(len(paths[item]) for item in real_cold) * contexts_per_target,
    }


def _write_or_verify_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.candidate.{os.getpid()}")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_json_line(row))
                count += 1
        candidate_sha = sha256_file(temporary)
        if path.exists():
            if not path.is_file() or path.is_symlink() or sha256_file(path) != candidate_sha:
                raise ValueError(f"Existing uncheckpointed shard drift: {path}")
            temporary.unlink()
        else:
            temporary.replace(path)
        return candidate_sha, count
    except BaseException:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _iter_shard_rows(
    target_items: Sequence[str],
    *,
    seed: int,
    contexts_per_target: int,
    similarity_batch_size: int,
    paths: Mapping[str, tuple[str, ...]],
    eligible_warm: Sequence[str],
    occurrences: Mapping[str, Sequence[TrainOccurrence]],
    embeddings: "torch.Tensor",
    embedding_index: Mapping[str, int],
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for content similarity")
    eligible_matrix = embeddings[[embedding_index[item] for item in eligible_warm]]
    for batch_start in range(0, len(target_items), similarity_batch_size):
        batch = target_items[batch_start : batch_start + similarity_batch_size]
        cold_matrix = embeddings[[embedding_index[item] for item in batch]]
        similarities = cold_matrix @ eligible_matrix.T
        for row_index, cold_item in enumerate(batch):
            path = paths[cold_item]
            selected = deterministic_topk(
                similarities[row_index], eligible_warm, contexts_per_target
            )
            for rank, (warm_index, similarity) in enumerate(selected):
                warm_item = eligible_warm[warm_index]
                occurrence = choose_occurrence(
                    occurrences[warm_item],
                    cold_item=cold_item,
                    warm_item=warm_item,
                    seed=seed,
                )
                occurrence_hash = _stable_id(
                    seed,
                    "train-occurrence",
                    occurrence.user_id,
                    occurrence.position,
                    warm_item,
                )
                context_id = _stable_id(
                    seed, "pseudo-context", cold_item, rank, warm_item, occurrence_hash
                )
                context = {
                    "cold_item": cold_item,
                    "cold_path": list(path),
                    "context_id": context_id,
                    "neighbor_rank": rank,
                    "similarity": similarity,
                    "source_occurrence_hash": occurrence_hash,
                    "source_warm_item": warm_item,
                    "train_context_items": list(occurrence.history),
                }
                requests = [
                    {
                        "cold_item": cold_item,
                        "context_id": context_id,
                        "full_target_path": list(path),
                        "position": position,
                        "prefix_tokens": list(path[:position]),
                        "request_id": _stable_id(
                            seed, "edit-request", context_id, position, path[position]
                        ),
                        "source_warm_item": warm_item,
                        "target_token": path[position],
                        "train_context_items": list(occurrence.history),
                    }
                    for position in range(len(path))
                ]
                yield context, requests


def _write_shard(
    output_dir: Path,
    *,
    shard_index: int,
    target_items: Sequence[str],
    seed: int,
    contexts_per_target: int,
    similarity_batch_size: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    cached_rows = _iter_shard_rows(
        target_items,
        seed=seed,
        contexts_per_target=contexts_per_target,
        similarity_batch_size=similarity_batch_size,
        paths=data["paths"],
        eligible_warm=data["eligible_warm"],
        occurrences=data["occurrences"],
        embeddings=data["embeddings"],
        embedding_index=data["embedding_index"],
    )

    # A shard, not the full dataset, is retained to let both files be produced
    # deterministically and independently recovered after interruption.
    shard_contexts: list[dict[str, Any]] = []
    shard_requests: list[dict[str, Any]] = []
    for context, requests in cached_rows:
        shard_contexts.append(context)
        shard_requests.extend(requests)

    name = f"shard-{shard_index:05d}.jsonl"
    context_path = output_dir / "pseudo_contexts" / name
    request_path = output_dir / "position_requests" / name
    context_sha, context_count = _write_or_verify_jsonl(context_path, shard_contexts)
    request_sha, request_count = _write_or_verify_jsonl(request_path, shard_requests)
    return {
        "shard_index": shard_index,
        "first_target": target_items[0],
        "last_target": target_items[-1],
        "target_count": len(target_items),
        "context_count": context_count,
        "request_count": request_count,
        "pseudo_contexts": {
            "path": str(context_path.relative_to(output_dir)),
            "sha256": context_sha,
        },
        "position_requests": {
            "path": str(request_path.relative_to(output_dir)),
            "sha256": request_sha,
        },
    }


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _verify_completed_shard(
    output_dir: Path,
    row: Mapping[str, Any],
    *,
    expected_shard_index: int,
    expected_targets: Sequence[str],
    contexts_per_target: int,
    paths: Mapping[str, Sequence[str]],
) -> None:
    if not expected_targets:
        raise ValueError("Cannot verify an empty target shard")
    expected_contexts = len(expected_targets) * contexts_per_target
    expected_requests = (
        sum(len(paths[item]) for item in expected_targets) * contexts_per_target
    )
    identity = {
        "shard_index": expected_shard_index,
        "first_target": expected_targets[0],
        "last_target": expected_targets[-1],
        "target_count": len(expected_targets),
        "context_count": expected_contexts,
        "request_count": expected_requests,
    }
    for key, expected in identity.items():
        if row.get(key) != expected:
            raise ValueError(
                f"Checkpointed shard target/count contract drift for {key}: "
                f"expected {expected}, got {row.get(key)}"
            )
    for key in ("pseudo_contexts", "position_requests"):
        path = output_dir / row[key]["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing checkpointed {key} shard: {path}")
        if sha256_file(path) != row[key]["sha256"]:
            raise ValueError(f"Checkpointed {key} shard SHA drift: {path}")
    context_path = output_dir / row["pseudo_contexts"]["path"]
    request_path = output_dir / row["position_requests"]["path"]
    if _line_count(context_path) != expected_contexts:
        raise ValueError(f"Checkpointed context shard line-count drift: {context_path}")
    if _line_count(request_path) != expected_requests:
        raise ValueError(f"Checkpointed request shard line-count drift: {request_path}")


def build_sharded_dataset(
    inputs: DatasetInputs,
    output_dir: Path,
    *,
    seed: int = 1502,
    contexts_per_target: int = 10,
    max_history: int = 20,
    target_shard_size: int = 128,
    similarity_batch_size: int = 64,
    required_counts: Mapping[str, int] | None = None,
    required_covariance_position_counts: Mapping[int, int] | None = None,
    minimum_long_path_resource_rows: int = 8,
    long_path_resource_position: int = 5,
) -> dict[str, Any]:
    """Build or resume deterministic pseudo-context and full-position shards."""

    if contexts_per_target < 1 or max_history < 1:
        raise ValueError("contexts_per_target and max_history must be positive")
    if target_shard_size < 1 or similarity_batch_size < 1:
        raise ValueError("Shard and similarity batch sizes must be positive")
    data = _load_and_audit_inputs(inputs, max_history=max_history)
    if len(data["eligible_warm"]) < contexts_per_target:
        raise ValueError("Fewer eligible retained-warm occurrences than requested contexts")
    counts = _expected_counts(data["real_cold"], data["paths"], contexts_per_target)
    if required_counts is not None and counts != dict(required_counts):
        raise ValueError(f"Frozen workload count mismatch: expected {required_counts}, got {counts}")
    covariance_counts = covariance_position_coverage(
        data["rows"],
        paths=data["paths"],
        retained_warm=data["retained_warm"],
        max_history=max_history,
        seed=seed,
    )
    if required_covariance_position_counts is not None and covariance_counts != dict(
        required_covariance_position_counts
    ):
        raise ValueError(
            "Frozen covariance position coverage mismatch: "
            f"expected {required_covariance_position_counts}, got {covariance_counts}"
        )
    resource_subset = deterministic_long_path_resource_subset(
        data["rows"],
        paths=data["paths"],
        retained_warm=data["retained_warm"],
        max_history=max_history,
        seed=seed,
        lexical_position=long_path_resource_position,
        minimum_rows=minimum_long_path_resource_rows,
    )

    semantic_config = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "contexts_per_target": contexts_per_target,
        "max_history": max_history,
        "target_shard_size": target_shard_size,
        "similarity_batch_size": similarity_batch_size,
        "target_order": "ascending_item_id",
        "neighbor_order": "descending_cosine_then_ascending_item_id",
        "occurrence_choice": "sha256_rank_train_only",
        "positions": "all_non_eos_non_padding_lexical_path_positions",
        "covariance_rows": "all_train_only_retained_warm_transitions_per_legal_position",
        "minimum_long_path_resource_rows": minimum_long_path_resource_rows,
        "long_path_resource_position": long_path_resource_position,
    }
    build_id = stable_sha256(
        {
            "semantic_config": semantic_config,
            "input_sha256": data["input_sha256"],
            "counts": counts,
            "covariance_position_counts": covariance_counts,
        }
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    final_path = output_dir / "manifest.json"

    subset_path = output_dir / "covariance_resource_subset.jsonl"
    subset_sha, subset_count = _write_or_verify_jsonl(subset_path, resource_subset)

    if final_path.exists():
        manifest = json.loads(final_path.read_text(encoding="utf-8"))
        if manifest.get("build_id") != build_id:
            raise ValueError("Completed GenRecEdit manifest belongs to another frozen build")
        stable_payload = dict(manifest)
        recorded_stable_sha = stable_payload.pop("stable_manifest_payload_sha256", None)
        if recorded_stable_sha != stable_sha256(stable_payload):
            raise ValueError("Completed GenRecEdit stable manifest SHA drift")
        if manifest.get("counts") != counts:
            raise ValueError("Completed GenRecEdit manifest count drift")
        recorded_subset = manifest.get("covariance", {}).get("resource_subset", {})
        if recorded_subset.get("sha256") != subset_sha or recorded_subset.get("rows") != subset_count:
            raise ValueError("Completed covariance resource-subset manifest drift")
        target_items = sorted(data["real_cold"])
        shards = manifest.get("shards", [])
        if len(shards) != math.ceil(len(target_items) / target_shard_size):
            raise ValueError("Completed manifest has an unexpected shard count")
        for shard_index, shard in enumerate(shards):
            start = shard_index * target_shard_size
            _verify_completed_shard(
                output_dir,
                shard,
                expected_shard_index=shard_index,
                expected_targets=target_items[start : start + target_shard_size],
                contexts_per_target=contexts_per_target,
                paths=data["paths"],
            )
        expected_dataset_sha = stable_sha256(
            {
                "build_id": build_id,
                "totals": counts,
                "shards": [
                    {
                        "shard_index": row["shard_index"],
                        "pseudo_contexts_sha256": row["pseudo_contexts"]["sha256"],
                        "position_requests_sha256": row["position_requests"]["sha256"],
                    }
                    for row in shards
                ],
                "covariance_resource_subset_sha256": subset_sha,
            }
        )
        if manifest.get("dataset_sha256") != expected_dataset_sha:
            raise ValueError("Completed GenRecEdit dataset SHA drift")
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise ValueError("Completed manifest checkpoint is missing")
        if sha256_file(checkpoint_path) != manifest["resume_contract"]["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint SHA does not match final manifest")
        return manifest

    target_items = sorted(data["real_cold"])
    expected_shards = math.ceil(len(target_items) / target_shard_size)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("build_id") != build_id:
            raise ValueError("Checkpoint belongs to another frozen build")
    else:
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "expected_shards": expected_shards,
            "completed_shards": [],
            "status": "in_progress",
        }
        _atomic_json(checkpoint_path, checkpoint)

    if checkpoint.get("expected_shards") != expected_shards:
        raise ValueError("Checkpoint expected-shard count drift")

    completed = checkpoint.get("completed_shards")
    if not isinstance(completed, list):
        raise ValueError("Malformed checkpoint shard list")
    if [row.get("shard_index") for row in completed] != list(range(len(completed))):
        raise ValueError("Checkpointed shards must form a contiguous prefix")
    for shard_index, shard in enumerate(completed):
        start = shard_index * target_shard_size
        _verify_completed_shard(
            output_dir,
            shard,
            expected_shard_index=shard_index,
            expected_targets=target_items[start : start + target_shard_size],
            contexts_per_target=contexts_per_target,
            paths=data["paths"],
        )

    for shard_index in range(len(completed), expected_shards):
        start = shard_index * target_shard_size
        end = min(len(target_items), start + target_shard_size)
        shard = _write_shard(
            output_dir,
            shard_index=shard_index,
            target_items=target_items[start:end],
            seed=seed,
            contexts_per_target=contexts_per_target,
            similarity_batch_size=similarity_batch_size,
            data=data,
        )
        completed.append(shard)
        checkpoint["completed_shards"] = completed
        checkpoint["completed_targets"] = sum(row["target_count"] for row in completed)
        checkpoint["completed_contexts"] = sum(row["context_count"] for row in completed)
        checkpoint["completed_requests"] = sum(row["request_count"] for row in completed)
        _atomic_json(checkpoint_path, checkpoint)

    totals = {
        "targets": sum(row["target_count"] for row in completed),
        "contexts": sum(row["context_count"] for row in completed),
        "requests": sum(row["request_count"] for row in completed),
    }
    if totals != counts:
        raise ValueError(f"Constructed workload count mismatch: expected {counts}, got {totals}")
    shard_digest_rows = [
        {
            "shard_index": row["shard_index"],
            "pseudo_contexts_sha256": row["pseudo_contexts"]["sha256"],
            "position_requests_sha256": row["position_requests"]["sha256"],
        }
        for row in completed
    ]
    dataset_sha = stable_sha256(
        {
            "build_id": build_id,
            "totals": totals,
            "shards": shard_digest_rows,
            "covariance_resource_subset_sha256": subset_sha,
        }
    )
    leakage_audit = {
        "occurrence_source": "S16-1 student-readable interaction train only",
        "occurrence_source_sha256": data["input_sha256"]["train_sequences"],
        "validation_occurrence_files_opened": 0,
        "internal_dev_occurrence_files_opened": 0,
        "test_occurrence_files_opened": 0,
        "held_ground_truth_files_opened": 0,
        "real_cold_in_train_occurrences": 0,
        "pseudo_cold_in_train_occurrences": 0,
        "non_retained_warm_in_train_occurrences": 0,
        "real_cold_target_source": "frozen pre-Stage16 membership",
        "target_selection_uses_validation_or_test_occurrence": False,
        "context_source_membership": "retained_warm_with_train_occurrence",
        "eos_requests": 0,
        "padding_requests": 0,
    }
    # Finalize the checkpoint first.  The immutable SHA recorded below must be
    # the hash of the actual final checkpoint, never its prior in-progress form.
    checkpoint["status"] = "complete"
    checkpoint["dataset_sha256"] = dataset_sha
    _atomic_json(checkpoint_path, checkpoint)
    final_checkpoint_sha = sha256_file(checkpoint_path)

    manifest_base = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "build_id": build_id,
        "dataset_sha256": dataset_sha,
        "semantic_config": semantic_config,
        "input_sha256": data["input_sha256"],
        "embedding_manifest": data["embedding_manifest"],
        "counts": totals,
        "train_users": len(data["rows"]),
        "train_items": data["train_items"],
        "train_transitions": data["train_transitions"],
        "eligible_retained_warm_items_with_occurrence": len(data["eligible_warm"]),
        "path_length_counts": {
            str(length): sum(1 for item in data["real_cold"] if len(data["paths"][item]) == length)
            for length in sorted({len(data["paths"][item]) for item in data["real_cold"]})
        },
        "covariance": {
            "source": "S16-1 train-only retained-warm transitions",
            "iterator": "iter_covariance_position_rows",
            "mom2_n_samples_cap": 400000,
            "unique_train_transitions": data["train_transitions"],
            "position_counts": {str(key): value for key, value in covariance_counts.items()},
            "resource_subset": {
                "path": subset_path.name,
                "sha256": subset_sha,
                "rows": subset_count,
                "lexical_position": long_path_resource_position,
                "minimum_required_rows": minimum_long_path_resource_rows,
            },
            "stage15_context_artifact_reused": False,
        },
        "leakage_audit": leakage_audit,
        "resume_contract": {
            "checkpoint_manifest": checkpoint_path.name,
            "checkpoint_sha256": final_checkpoint_sha,
            "completed_shards": len(completed),
            "expected_shards": expected_shards,
        },
        "shards": completed,
    }
    manifest_base["stable_manifest_payload_sha256"] = stable_sha256(manifest_base)
    _atomic_json(final_path, manifest_base)
    return manifest_base


def resolve_stage16_toys_inputs(
    preflight_config_path: Path,
    preflight_root: Path | None = None,
) -> tuple[DatasetInputs, dict[str, int], int]:
    """Resolve the frozen Toys S16-1 inputs without opening any held event file."""

    config_path = _regular_input(preflight_config_path, "S16-1 config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    domains = [row for row in config.get("domains", []) if row.get("name") == "Toys_cold50"]
    if len(domains) != 1:
        raise ValueError("S16-1 config must contain exactly one Toys_cold50 domain")
    domain = domains[0]
    if preflight_root is None:
        preflight_root = _resolve_repo_path(config["output_dir"])
    else:
        preflight_root = preflight_root.resolve()
    split_root = preflight_root / "splits" / "Toys_cold50"
    split_manifest_path = _regular_input(split_root / "split_manifest.json", "split manifest")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("domain") != "Toys_cold50":
        raise ValueError("Unexpected S16-1 split-manifest domain")
    outputs = split_manifest.get("outputs", {})

    train = split_root / "student_readable" / "interaction_train_sequences.jsonl"
    retained = split_root / "retained_warm_items.txt"
    pseudo = split_root / "pseudo_cold_items.txt"

    def output_sha(path: Path) -> str:
        relative = repo_relative_path(path)
        expected = outputs.get(relative)
        if not isinstance(expected, str):
            raise ValueError(f"S16-1 manifest does not freeze output {relative}")
        return expected

    expected_sha = {
        "train_sequences": output_sha(train),
        "retained_warm_items": output_sha(retained),
        "pseudo_cold_items": output_sha(pseudo),
        "real_cold_items": domain["cold_items"]["sha256"],
        "lexical_paths": domain["lexical_paths"]["sha256"],
        "metadata": domain["metadata"]["sha256"],
        "content_embeddings": domain["content_embeddings"]["sha256"],
    }
    counts = {
        "targets": int(split_manifest["g_full_edit_targets"]),
        "contexts": int(split_manifest["g_full_contexts"]),
        "requests": int(split_manifest["g_full_prefix_next_token_requests"]),
    }
    if counts != TOYS_FROZEN_COUNTS:
        raise ValueError(f"Toys S16-1 G-FULL workload drift: {counts}")
    inputs = DatasetInputs(
        train_sequences=train,
        retained_warm_items=retained,
        pseudo_cold_items=pseudo,
        real_cold_items=_resolve_repo_path(domain["cold_items"]["path"]),
        lexical_paths=_resolve_repo_path(domain["lexical_paths"]["path"]),
        metadata=_resolve_repo_path(domain["metadata"]["path"]),
        content_embeddings=_resolve_repo_path(domain["content_embeddings"]["path"]),
        split_manifest=split_manifest_path,
        expected_sha256=expected_sha,
    )
    return inputs, counts, int(config["split_policy"]["maximum_history_items"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-config",
        type=Path,
        default=ROOT / "experiment/phase16/configs/stage16_s1_data_resource_preflight.json",
    )
    parser.add_argument("--preflight-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-shard-size", type=int, default=128)
    parser.add_argument("--similarity-batch-size", type=int, default=64)
    args = parser.parse_args()
    inputs, counts, max_history = resolve_stage16_toys_inputs(
        args.preflight_config.resolve(),
        args.preflight_root.resolve() if args.preflight_root else None,
    )
    manifest = build_sharded_dataset(
        inputs,
        args.output_dir,
        seed=1502,
        contexts_per_target=10,
        max_history=max_history,
        target_shard_size=args.target_shard_size,
        similarity_batch_size=args.similarity_batch_size,
        required_counts=counts,
        required_covariance_position_counts=TOYS_FROZEN_COVARIANCE_POSITION_COUNTS,
        minimum_long_path_resource_rows=8,
    )
    print(json.dumps({"verdict": "PASS_GFULL_DATA_REQUEST_BUILD", **manifest["counts"], "dataset_sha256": manifest["dataset_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
