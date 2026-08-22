"""Shared Stage15 data-boundary helpers for GRAM adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SEALED_SLOT = "__STAGE15_SEALED_TEST_SLOT__"


@dataclass(frozen=True)
class TrainTransition:
    """One train-only next-item example derived from the safe projection."""

    user_id: str
    history: tuple[str, ...]
    target: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_projected_sequences(path: Path) -> dict[str, list[str]]:
    if path.name != "user_sequence_train_validation.txt":
        raise ValueError(
            "Stage15 adapters require user_sequence_train_validation.txt"
        )
    rows: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            fields = raw.strip().split()
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: projected row too short")
            user, items = fields[0], fields[1:]
            if user in rows:
                raise ValueError(f"{path}:{line_number}: duplicate user {user}")
            if SEALED_SLOT in fields:
                raise ValueError(f"{path}:{line_number}: reserved sentinel present")
            rows[user] = items
    if not rows:
        raise ValueError(f"No projected rows in {path}")
    return rows


def stable_user_sample(users: list[str], sample_size: int, seed: int) -> list[str]:
    if sample_size < 1 or sample_size > len(users):
        raise ValueError("sample_size must be within the available user count")
    return sorted(
        users,
        key=lambda user: (
            hashlib.sha256(f"{seed}:{user}".encode("utf-8")).digest(),
            user,
        ),
    )[:sample_size]


def train_only_sequences(
    projected_rows: Mapping[str, list[str]],
) -> dict[str, tuple[str, ...]]:
    """Drop the validation target without exposing it to downstream builders."""

    train_rows: dict[str, tuple[str, ...]] = {}
    for user, projected in projected_rows.items():
        if len(projected) < 2:
            raise ValueError(f"Projected sequence for {user} has no train history")
        train = tuple(projected[:-1])
        if SEALED_SLOT in train:
            raise ValueError(f"Reserved sentinel in train sequence for {user}")
        train_rows[user] = train
    return train_rows


def iter_train_transitions(
    projected_rows: Mapping[str, list[str]],
) -> Iterable[TrainTransition]:
    """Yield chronological train-only supervision; validation is never a target."""

    for user, items in train_only_sequences(projected_rows).items():
        for target_index in range(1, len(items)):
            yield TrainTransition(
                user_id=user,
                history=items[:target_index],
                target=items[target_index],
            )


def read_validation_predictions(path: Path) -> dict[str, dict]:
    if "test" in path.name.lower():
        raise ValueError(f"Refusing test predictions: {path}")
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row["user_id"])
            if user in rows:
                raise ValueError(f"{path}:{line_number}: duplicate user {user}")
            ranking = row.get("v0_top50")
            if not isinstance(ranking, list) or len(ranking) != 50:
                raise ValueError(f"{path}:{line_number}: invalid v0_top50")
            if len(set(ranking)) != 50:
                raise ValueError(f"{path}:{line_number}: duplicate v0 item")
            rows[user] = row
    if not rows:
        raise ValueError(f"No validation predictions in {path}")
    return rows


def build_legacy_validation_view(
    *,
    projected_sequences: Path,
    selected_users: list[str],
    source_dataset_dir: Path,
    item_path_file: Path,
    view_dataset_dir: Path,
) -> dict:
    """Create a GRAM-compatible view with a non-catalog sealed slot.

    Historical GRAM validation slices ``items[-2]``.  The Stage15 projection
    ends at the validation target, so a fixed non-catalog sentinel is appended
    solely inside this isolated view.  The sentinel is never a history/target
    under validation mode and no test loader may be constructed.
    """

    rows = read_projected_sequences(projected_sequences)
    if len(set(selected_users)) != len(selected_users):
        raise ValueError("selected_users contains duplicates")
    missing = set(selected_users) - rows.keys()
    if missing:
        raise ValueError(f"Selected users absent from projection: {len(missing)}")
    if view_dataset_dir.exists() or view_dataset_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite dataset view {view_dataset_dir}")

    view_dataset_dir.mkdir(parents=True)
    sequence_path = view_dataset_dir / "user_sequence.txt"
    with sequence_path.open("x", encoding="utf-8") as handle:
        for user in selected_users:
            handle.write(f"{user} {' '.join(rows[user])} {SEALED_SLOT}\n")

    copy_pairs = [
        (source_dataset_dir / "item_plain_text.txt", view_dataset_dir / "item_plain_text.txt"),
        (source_dataset_dir / "similar_item_sasrec.txt", view_dataset_dir / "similar_item_sasrec.txt"),
        (item_path_file, view_dataset_dir / item_path_file.name),
        (
            source_dataset_dir / "cold_split_meta" / "cold_items.txt",
            view_dataset_dir / "cold_split_meta" / "cold_items.txt",
        ),
        (
            source_dataset_dir / "cold_split_meta" / "warm_items.txt",
            view_dataset_dir / "cold_split_meta" / "warm_items.txt",
        ),
    ]
    for source, destination in copy_pairs:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Input must be a regular non-symlink file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1 << 20)

    return {
        "status": "completed",
        "view_semantics": "projected train+validation plus fixed sealed slot",
        "selected_users": selected_users,
        "selected_users_sha256": hashlib.sha256(
            ("\n".join(selected_users) + "\n").encode("utf-8")
        ).hexdigest(),
        "sentinel": SEALED_SLOT,
        "sentinel_is_catalog_item": False,
        "test_target_materialized": False,
        "source_projection_sha256": sha256_file(projected_sequences),
        "view_sequence_sha256": sha256_file(sequence_path),
        "copied_inputs": {
            str(destination): sha256_file(destination)
            for _source, destination in copy_pairs
        },
    }


def compare_rankings(observed: list[str], expected: list[str]) -> dict:
    if len(observed) != 50 or len(expected) != 50:
        raise ValueError("Both rankings must contain exactly 50 items")
    if len(set(observed)) != 50:
        raise ValueError("Observed ranking contains duplicate items")
    first_mismatch = next(
        (index for index, pair in enumerate(zip(observed, expected), 1) if pair[0] != pair[1]),
        None,
    )
    return {
        "exact": observed == expected,
        "first_mismatch_rank": first_mismatch,
        "set_equal": set(observed) == set(expected),
        "prefix10_exact": observed[:10] == expected[:10],
    }
