from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.leakage_guard import (
    DatasetACL,
    assert_fold_isolation,
    build_train_only_transitions,
)


ROOT = Path(__file__).resolve().parents[3]


class SplitGuardTests(unittest.TestCase):
    def test_frozen_d0_view_is_authorized_but_other_fold_is_not(self) -> None:
        root = ROOT / "artifacts/phase17/s0_audit/shadow_data/Toys/D0"
        acl = DatasetACL((root,), frozenset({"D0"}))
        self.assertEqual(acl.authorize(root / "user_sequence.txt", "D0", "train"), (root / "user_sequence.txt").resolve())
        with self.assertRaises(PermissionError):
            acl.authorize(root / "user_sequence.txt", "D1", "train")

    def test_sports_and_outside_path_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            acl = DatasetACL((root,), frozenset({"D0"}))
            with self.assertRaises(PermissionError):
                acl.authorize(root / "Sports" / "user_sequence.txt", "D0", "train")
            with self.assertRaises(PermissionError):
                acl.authorize(ROOT / "data" / "outside.txt", "D0", "train")

    def test_shadow_manifest_proves_official_positions_are_not_serialized(self) -> None:
        manifest = json.loads(
            (ROOT / "artifacts/phase17/s0_audit/shadow_data_manifest.json").read_text(encoding="utf-8")
        )
        for domain in ("Beauty", "Toys"):
            for fold in ("D0", "D1", "D2"):
                row = manifest["domains"][domain]["folds"][fold]
                self.assertFalse(row["official_validation_position_serialized"])
                self.assertFalse(row["official_test_position_serialized"])
                self.assertFalse(row["target_in_train_by_position"])

    def test_fold_isolation_and_train_only_transition_builder(self) -> None:
        assert_fold_isolation({"u": "old"}, {"u": "new"})
        with self.assertRaises(PermissionError):
            assert_fold_isolation({"u": "new"}, {"u": "new"})
        transitions = build_train_only_transitions([["a", "b", "future"]], [2])
        self.assertEqual(transitions, {"a": {"b": 1}})
        self.assertNotIn("future", transitions)
