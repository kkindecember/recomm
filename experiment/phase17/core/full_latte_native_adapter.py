"""Leakage-safe Stage17 Toys D0 adapter for the official LATTE codebase.

The official repository expects an ``AbstractDataset`` with Hugging Face Dataset
splits.  This module keeps the data contract independent of those optional
dependencies, then creates the official subclass lazily inside the pinned LATTE
environment.  The training adapter never calls the external-target reader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .fullport_data import FullportTrainUser, read_train_prefix_users


APPROVED_METADATA_SUFFIX = Path("GRAM/rec_datasets/Toys/item_plain_text.txt")
APPROVED_METADATA_SHA256 = "e2d4f59b59381a4519905c94409d334bcd092f060d4a1d52df1bb4d9b63a8507"
APPROVED_SEQUENCE_SUFFIX = Path(
    "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt"
)
APPROVED_DEV_IDS_SUFFIX = Path(
    "artifacts/phase17/fullport/manifests/toys_d0_internal_dev_user_ids.txt"
)
DEFAULT_NATIVE_CACHE_SUFFIX = Path(
    "artifacts/phase17/fullport/cache/latte_native_toys_d0"
)


@dataclass(frozen=True)
class LatteNativeSequence:
    user_id: str
    item_seq: tuple[str, ...]


@dataclass(frozen=True)
class LatteNativeDataBundle:
    id_mapping: dict[str, Any]
    item2meta: dict[str, str]
    train_sequences: tuple[LatteNativeSequence, ...]
    internal_dev_sequences: tuple[LatteNativeSequence, ...]
    all_item_seqs: dict[str, tuple[str, ...]]
    internal_dev_user_ids: tuple[str, ...]
    rolling_train_examples: int
    train_catalog_items: int
    tokenizer_fit_catalog_items: int
    catalog_items: int
    external_target_materialized: bool = False
    test_read: bool = False
    sports_read: bool = False
    d1_read: bool = False
    d2_read: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_item_metadata_catalog(path: Path, *, root: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    """Read the frozen item catalog without opening any interaction target."""

    resolved = path.resolve()
    expected = (root.resolve() / APPROVED_METADATA_SUFFIX).resolve()
    if resolved != expected:
        raise PermissionError(f"unexpected Stage17 LATTE metadata path: {path}")
    actual_sha256 = _sha256(resolved)
    if actual_sha256 != APPROVED_METADATA_SHA256:
        raise RuntimeError(f"Toys metadata hash drift: {actual_sha256}")
    ordered_items: list[str] = []
    item2meta: dict[str, str] = {}
    for line_number, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split(maxsplit=1)
        if len(fields) != 2 or not fields[1].strip():
            raise ValueError(f"metadata line {line_number} has no item text")
        item_id, text = fields[0], fields[1].strip()
        if item_id in item2meta:
            raise ValueError(f"duplicate metadata item id: {item_id}")
        ordered_items.append(item_id)
        item2meta[item_id] = text
    if not ordered_items:
        raise ValueError("empty Toys metadata catalog")
    return tuple(ordered_items), item2meta


def read_frozen_internal_dev_ids(path: Path, *, root: Path) -> tuple[str, ...]:
    resolved = path.resolve()
    expected = (root.resolve() / APPROVED_DEV_IDS_SUFFIX).resolve()
    if resolved != expected:
        raise PermissionError(f"unexpected Stage17 internal-dev id path: {path}")
    values = tuple(line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("internal-dev ids must be non-empty and unique")
    return values


def build_latte_native_bundle(
    *,
    root: Path,
    sequence_path: Path | None = None,
    metadata_path: Path | None = None,
    internal_dev_ids_path: Path | None = None,
) -> LatteNativeDataBundle:
    """Build native LATTE inputs from train prefix plus position-held-out dev only."""

    root = root.resolve()
    sequence_path = sequence_path or root / APPROVED_SEQUENCE_SUFFIX
    metadata_path = metadata_path or root / APPROVED_METADATA_SUFFIX
    internal_dev_ids_path = internal_dev_ids_path or root / APPROVED_DEV_IDS_SUFFIX
    users = read_train_prefix_users(sequence_path, root=root)
    catalog, item2meta = read_item_metadata_catalog(metadata_path, root=root)
    dev_ids = read_frozen_internal_dev_ids(internal_dev_ids_path, root=root)
    dev_set = set(dev_ids)
    user_ids = {user.user_id for user in users}
    if not dev_set <= user_ids:
        raise ValueError("frozen internal-dev ids contain users outside Toys D0")

    catalog_set = set(catalog)
    train_sequences: list[LatteNativeSequence] = []
    internal_dev_sequences: list[LatteNativeSequence] = []
    all_item_seqs: dict[str, tuple[str, ...]] = {}
    fold_train_catalog: set[str] = set()
    tokenizer_fit_catalog: set[str] = set()
    for user in users:
        unknown = sorted(set(user.train_items) - catalog_set)
        if unknown:
            raise ValueError(f"user {user.user_id} has items outside metadata catalog: {unknown[:3]}")
        if user.user_id in dev_set:
            if len(user.train_items) < 2:
                raise ValueError(f"internal-dev user {user.user_id} lacks a held-out train position")
            training_items = user.train_items[:-1]
            internal_dev_sequences.append(
                LatteNativeSequence(user_id=user.user_id, item_seq=user.train_items)
            )
        else:
            training_items = user.train_items
        train_sequences.append(
            LatteNativeSequence(user_id=user.user_id, item_seq=training_items)
        )
        all_item_seqs[user.user_id] = training_items
        fold_train_catalog.update(user.train_items)
        tokenizer_fit_catalog.update(training_items)

    dev_order = tuple(sequence.user_id for sequence in internal_dev_sequences)
    if set(dev_order) != dev_set or len(dev_order) != len(dev_ids):
        raise AssertionError("native LATTE internal-dev cohort is incomplete")
    rolling_examples = sum(max(0, len(row.item_seq) - 1) for row in train_sequences)
    id_mapping = {
        "user2id": {"[PAD]": 0},
        "item2id": {"[PAD]": 0},
        "id2user": ["[PAD]"],
        "id2item": ["[PAD]"],
    }
    for user in users:
        id_mapping["user2id"][user.user_id] = len(id_mapping["id2user"])
        id_mapping["id2user"].append(user.user_id)
    for item in catalog:
        id_mapping["item2id"][item] = len(id_mapping["id2item"])
        id_mapping["id2item"].append(item)

    return LatteNativeDataBundle(
        id_mapping=id_mapping,
        item2meta=item2meta,
        train_sequences=tuple(train_sequences),
        internal_dev_sequences=tuple(internal_dev_sequences),
        all_item_seqs=all_item_seqs,
        internal_dev_user_ids=dev_order,
        rolling_train_examples=rolling_examples,
        train_catalog_items=len(fold_train_catalog),
        tokenizer_fit_catalog_items=len(tokenizer_fit_catalog),
        catalog_items=len(catalog),
    )


def _split_rows(sequences: Sequence[LatteNativeSequence]) -> dict[str, list[Any]]:
    return {
        "user": [row.user_id for row in sequences],
        "item_seq": [list(row.item_seq) for row in sequences],
    }


def make_official_latte_dataset_class(
    *,
    root: Path,
    abstract_dataset_class: type | None = None,
    dataset_factory: Callable[[dict[str, list[Any]]], Any] | None = None,
) -> type:
    """Create the adapter subclass lazily inside the official Python environment.

    ``abstract_dataset_class`` and ``dataset_factory`` are injectable solely for
    dependency-free contract tests.  Formal execution leaves both unset and uses
    the pinned official ``genrec`` and Hugging Face ``datasets`` packages.
    """

    if abstract_dataset_class is None:
        from genrec.dataset import AbstractDataset as abstract_dataset_class
    if dataset_factory is None:
        from datasets import Dataset

        dataset_factory = Dataset.from_dict
    bundle = build_latte_native_bundle(root=root)
    resolved_root = root.resolve()

    class Stage17ToysD0(abstract_dataset_class):
        def __init__(self, config: dict[str, Any]):
            if config.get("external_target_authorized", False):
                raise PermissionError("the native training adapter cannot open external D0 targets")
            super().__init__(config)
            self.category = "Toys_D0_train_prefix"
            self.cache_dir = str(
                Path(config.get("stage17_native_cache_dir", resolved_root / DEFAULT_NATIVE_CACHE_SUFFIX))
            )
            self.id_mapping = bundle.id_mapping
            self.item2meta = bundle.item2meta
            self.all_item_seqs = bundle.all_item_seqs
            train = dataset_factory(_split_rows(bundle.train_sequences))
            internal_dev = dataset_factory(_split_rows(bundle.internal_dev_sequences))
            self.split_data = {
                "train": train,
                "val": internal_dev,
                # Official Pipeline requires a test key during tokenization.  It is
                # deliberately an internal-dev alias; the formal Stage17 runner
                # never reports it as external efficacy and uses a separate sealed
                # one-shot evaluator after both checkpoints are frozen.
                "test": dataset_factory(_split_rows(bundle.internal_dev_sequences)),
            }
            self.stage17_split_roles = {
                "train": "train_prefix_rolling_source",
                "val": "train_prefix_position_held_out_internal_dev",
                "test": "non_efficacy_internal_dev_alias",
            }
            self.stage17_external_target_materialized = False

        def _download_and_process_raw(self):
            raise PermissionError("Stage17 native adapter forbids official dataset download")

    Stage17ToysD0.__name__ = "Stage17ToysD0"
    Stage17ToysD0.__qualname__ = "Stage17ToysD0"
    return Stage17ToysD0


def install_official_latte_dataset_override(*, root: Path) -> tuple[type, type]:
    """Install the audited adapter under the official config-loading entry point."""

    import genrec.datasets as official_datasets

    replacement = make_official_latte_dataset_class(root=root)
    original = official_datasets.AmazonReviews2023
    official_datasets.AmazonReviews2023 = replacement
    return original, replacement
