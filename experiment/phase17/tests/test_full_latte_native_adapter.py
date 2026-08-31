from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_native_adapter import (
    APPROVED_DEV_IDS_SUFFIX,
    APPROVED_METADATA_SUFFIX,
    APPROVED_SEQUENCE_SUFFIX,
    build_latte_native_bundle,
    make_official_latte_dataset_class,
    read_item_metadata_catalog,
)


ROOT = Path(__file__).resolve().parents[3]


class FakeOfficialDataset:
    def __init__(self, config):
        self.config = config
        self.id_mapping = {}
        self.item2meta = None
        self.all_item_seqs = {}
        self.split_data = None

    @property
    def user2id(self):
        return self.id_mapping["user2id"]

    @property
    def item2id(self):
        return self.id_mapping["item2id"]


class FullLatteNativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = build_latte_native_bundle(root=ROOT)

    def test_real_bundle_matches_frozen_fp0_counts(self) -> None:
        self.assertEqual(len(self.bundle.train_sequences), 12833)
        self.assertEqual(len(self.bundle.internal_dev_sequences), 1283)
        self.assertEqual(self.bundle.rolling_train_examples, 56421)
        self.assertEqual(self.bundle.train_catalog_items, 11182)
        self.assertEqual(self.bundle.tokenizer_fit_catalog_items, 11138)
        self.assertEqual(self.bundle.catalog_items, 11924)
        self.assertFalse(self.bundle.external_target_materialized)
        self.assertFalse(self.bundle.test_read)
        self.assertFalse(self.bundle.sports_read)

    def test_dev_target_is_held_out_from_that_users_training_sequence(self) -> None:
        train_by_user = {row.user_id: row.item_seq for row in self.bundle.train_sequences}
        for row in self.bundle.internal_dev_sequences:
            self.assertEqual(train_by_user[row.user_id], row.item_seq[:-1])
            self.assertGreaterEqual(len(row.item_seq), 2)

    def test_mapping_is_one_based_and_covers_metadata(self) -> None:
        mapping = self.bundle.id_mapping
        self.assertEqual(mapping["id2item"][0], "[PAD]")
        self.assertEqual(mapping["id2user"][0], "[PAD]")
        self.assertEqual(len(mapping["id2item"]), self.bundle.catalog_items + 1)
        for index, item in enumerate(mapping["id2item"][1:], 1):
            self.assertEqual(mapping["item2id"][item], index)
            self.assertIn(item, self.bundle.item2meta)

    def test_dynamic_official_adapter_uses_no_external_split(self) -> None:
        adapter_class = make_official_latte_dataset_class(
            root=ROOT,
            abstract_dataset_class=FakeOfficialDataset,
            dataset_factory=lambda rows: rows,
        )
        dataset = adapter_class({"stage17_native_cache_dir": "/tmp/fake-cache"})
        self.assertEqual(dataset.__class__.__name__, "Stage17ToysD0")
        self.assertEqual(dataset.stage17_split_roles["test"], "non_efficacy_internal_dev_alias")
        self.assertEqual(dataset.split_data["val"], dataset.split_data["test"])
        self.assertFalse(dataset.stage17_external_target_materialized)
        with self.assertRaises(PermissionError):
            adapter_class({"external_target_authorized": True})

    def test_metadata_reader_rejects_a_copy_even_with_valid_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "item_plain_text.txt"
            copied.write_text("item title: x\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                read_item_metadata_catalog(copied, root=ROOT)


if __name__ == "__main__":
    unittest.main()
