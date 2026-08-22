from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from common_adapter import (  # noqa: E402
    SEALED_SLOT,
    build_legacy_validation_view,
    compare_rankings,
    iter_train_transitions,
    read_projected_sequences,
    stable_user_sample,
    train_only_sequences,
)


class TestCommonAdapter(unittest.TestCase):
    def test_user_sample_is_deterministic_and_order_independent(self):
        users = [f"u{i}" for i in range(20)]
        first = stable_user_sample(users, 5, 1502)
        second = stable_user_sample(list(reversed(users)), 5, 1502)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)

    def test_projection_reader_requires_frozen_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user_sequence.txt"
            path.write_text("u a validation\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "require"):
                read_projected_sequences(path)

    def test_legacy_view_preserves_validation_target_and_adds_only_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection = root / "user_sequence_train_validation.txt"
            projection.write_text("u train1 train2 validation\n", encoding="utf-8")
            source = root / "source"
            (source / "cold_split_meta").mkdir(parents=True)
            (source / "item_plain_text.txt").write_text("i text\n", encoding="utf-8")
            (source / "similar_item_sasrec.txt").write_text("i j\n", encoding="utf-8")
            (source / "cold_split_meta/cold_items.txt").write_text("i\n", encoding="utf-8")
            (source / "cold_split_meta/warm_items.txt").write_text("j\n", encoding="utf-8")
            item_path = source / "item_path.txt"
            item_path.write_text("i a|\nj b|\n", encoding="utf-8")
            view = root / "view"
            manifest = build_legacy_validation_view(
                projected_sequences=projection,
                selected_users=["u"],
                source_dataset_dir=source,
                item_path_file=item_path,
                view_dataset_dir=view,
            )
            fields = (view / "user_sequence.txt").read_text().split()
            self.assertEqual(fields[-2], "validation")
            self.assertEqual(fields[-1], SEALED_SLOT)
            self.assertFalse(manifest["test_target_materialized"])

    def test_ranking_contract_reports_first_mismatch(self):
        expected = [f"i{i}" for i in range(50)]
        observed = expected.copy()
        observed[10], observed[11] = observed[11], observed[10]
        report = compare_rankings(observed, expected)
        self.assertFalse(report["exact"])
        self.assertEqual(report["first_mismatch_rank"], 11)
        self.assertTrue(report["set_equal"])
        self.assertTrue(report["prefix10_exact"])

    def test_train_only_derivation_never_uses_validation_as_supervision(self):
        projected = {"u": ["train0", "train1", "validation"]}
        self.assertEqual(train_only_sequences(projected), {"u": ("train0", "train1")})
        transitions = list(iter_train_transitions(projected))
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].history, ("train0",))
        self.assertEqual(transitions[0].target, "train1")
        self.assertNotIn("validation", transitions[0].history)
        self.assertNotEqual(transitions[0].target, "validation")


if __name__ == "__main__":
    unittest.main()
