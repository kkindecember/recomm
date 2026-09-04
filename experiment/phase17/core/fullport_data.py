"""Fold-safe full-data adapter for the Stage17 full-port experiments.

The training-facing API deliberately never materializes the external D0 target
or guard item.  Opening those values requires the separate, explicit
``read_external_examples`` entry point and an authorization flag.  This keeps
tokenizer fitting, internal early stopping, and model training structurally
separate from the external discovery evaluation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


APPROVED_D0_SUFFIX = Path(
    "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt"
)
FORBIDDEN_PATH_PARTS = (
    "/D1/",
    "/D2/",
    "/Sports/",
    "/GRAM/rec_datasets/",
)


@dataclass(frozen=True)
class FullportTrainUser:
    user_id: str
    train_items: tuple[str, ...]


@dataclass(frozen=True)
class FullportExample:
    user_id: str
    history: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class FullportExternalExample:
    user_id: str
    history: tuple[str, ...]
    target: str


def _assert_approved_d0_path(path: Path, root: Path) -> None:
    resolved = path.resolve()
    expected = (root.resolve() / APPROVED_D0_SUFFIX).resolve()
    normalized = resolved.as_posix()
    if any(part in normalized for part in FORBIDDEN_PATH_PARTS):
        raise PermissionError(f"forbidden Stage17 full-port data path: {path}")
    if resolved != expected:
        raise PermissionError(
            "Stage17 full-port may read only the frozen full Toys D0 projection: "
            f"{path}"
        )


def _validated_fields(
    path: Path, *, expected_sha256: str | None = None
) -> list[list[str]]:
    rows: list[list[str]] = []
    seen: set[str] = set()
    raw_bytes = path.read_bytes()
    if expected_sha256 is not None:
        observed = hashlib.sha256(raw_bytes).hexdigest()
        if observed != expected_sha256:
            raise RuntimeError(
                f"Stage17 D0 projection hash drift: {observed} != {expected_sha256}"
            )
    for line_number, raw in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) < 4:
            raise ValueError(
                f"line {line_number} lacks train-prefix/target/guard positions"
            )
        if fields[0] in seen:
            raise ValueError(f"duplicate user id: {fields[0]}")
        seen.add(fields[0])
        rows.append(fields)
    if not rows:
        raise ValueError("empty Stage17 full-port D0 projection")
    return rows


def read_train_prefix_users(path: Path, *, root: Path) -> list[FullportTrainUser]:
    """Read only user ids and train-prefix positions from the approved D0 file."""

    _assert_approved_d0_path(path, root)
    users: list[FullportTrainUser] = []
    for fields in _validated_fields(path):
        train_items = tuple(fields[1:-2])
        if not train_items:
            raise ValueError(f"user {fields[0]} has no train-prefix item")
        users.append(FullportTrainUser(user_id=fields[0], train_items=train_items))
    return users


def select_internal_dev_users(
    users: Sequence[FullportTrainUser], *, count: int, seed: int = 2023
) -> tuple[str, ...]:
    """Select a deterministic train-prefix-only early-stop cohort."""

    eligible = [user.user_id for user in users if len(user.train_items) >= 2]
    if count <= 0 or len(eligible) < count:
        raise ValueError(
            f"requested {count} internal-dev users from {len(eligible)} eligible users"
        )
    ranked = sorted(
        eligible,
        key=lambda user_id: (
            hashlib.sha256(
                f"s17-fp0-internal-dev:{seed}:{user_id}".encode("utf-8")
            ).hexdigest(),
            user_id,
        ),
    )
    return tuple(ranked[:count])


def build_train_and_internal_dev_examples(
    users: Sequence[FullportTrainUser],
    internal_dev_user_ids: Sequence[str],
    *,
    max_history_items: int = 20,
) -> tuple[list[FullportExample], list[FullportExample]]:
    """Build rolling train examples and a position-held-out internal dev set."""

    if max_history_items <= 0:
        raise ValueError("max_history_items must be positive")
    dev_ids = set(internal_dev_user_ids)
    known_ids = {user.user_id for user in users}
    if not dev_ids <= known_ids:
        raise ValueError("internal-dev ids include users outside the D0 projection")

    train: list[FullportExample] = []
    internal_dev: list[FullportExample] = []
    for user in users:
        training_positions = user.train_items
        if user.user_id in dev_ids:
            if len(user.train_items) < 2:
                raise ValueError(f"internal-dev user {user.user_id} lacks history")
            training_positions = user.train_items[:-1]
            internal_dev.append(
                FullportExample(
                    user_id=user.user_id,
                    history=user.train_items[:-1][-max_history_items:],
                    target=user.train_items[-1],
                )
            )
        for target_position in range(1, len(training_positions)):
            train.append(
                FullportExample(
                    user_id=user.user_id,
                    history=training_positions[:target_position][
                        -max_history_items:
                    ],
                    target=training_positions[target_position],
                )
            )
    if len(internal_dev) != len(dev_ids):
        raise AssertionError("internal-dev examples do not cover selected users")
    if not train:
        raise ValueError("no full-port rolling train examples were constructed")
    return train, internal_dev


def read_external_examples(
    path: Path,
    *,
    root: Path,
    external_target_authorized: bool = False,
    max_history_items: int = 20,
) -> list[FullportExternalExample]:
    """Open D0 external targets only after a family checkpoint is frozen."""

    _users, examples = materialize_external_evaluation_view(
        path,
        root=root,
        external_target_authorized=external_target_authorized,
        max_history_items=max_history_items,
    )
    return examples


def materialize_external_evaluation_view(
    path: Path,
    *,
    root: Path,
    external_target_authorized: bool = False,
    max_history_items: int = 20,
    expected_sha256: str | None = None,
) -> tuple[list[FullportTrainUser], list[FullportExternalExample]]:
    """Read train prefixes and external targets together in one authorized pass.

    The family evaluator uses this API exactly once to build its immutable
    shared bundle.  Returning both views avoids a second open of the D0 file
    when target-frequency and memorization subgroups are computed later.
    """

    if not external_target_authorized:
        raise PermissionError(
            "D0 external targets remain sealed until family checkpoint freeze"
        )
    if max_history_items <= 0:
        raise ValueError("max_history_items must be positive")
    _assert_approved_d0_path(path, root)
    users: list[FullportTrainUser] = []
    examples: list[FullportExternalExample] = []
    for fields in _validated_fields(path, expected_sha256=expected_sha256):
        train_items = tuple(fields[1:-2])
        users.append(FullportTrainUser(user_id=fields[0], train_items=train_items))
        examples.append(
            FullportExternalExample(
                user_id=fields[0],
                history=train_items[-max_history_items:],
                target=fields[-2],
            )
        )
    return users, examples
