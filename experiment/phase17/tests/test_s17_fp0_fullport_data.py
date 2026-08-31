from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from experiment.phase17.core.fullport_data import (
    build_train_and_internal_dev_examples,
    read_external_examples,
    read_train_prefix_users,
    select_internal_dev_users,
)


ROOT = Path(__file__).resolve().parents[3]
D0 = ROOT / "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt"


class S17FP0FullportDataTests(unittest.TestCase):
    def test_full_d0_train_prefix_is_frozen_and_deterministic(self) -> None:
        users = read_train_prefix_users(D0, root=ROOT)
        self.assertEqual(len(users), 12833)
        digest = hashlib.sha256(D0.read_bytes()).hexdigest()
        self.assertEqual(
            digest, "24e92f46fc21e0192f8f0764c2c79e166c3636c79fb2ef1a4119491dde7be1fa"
        )
        first = select_internal_dev_users(users, count=1283, seed=2023)
        second = select_internal_dev_users(users, count=1283, seed=2023)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 1283)

    def test_internal_dev_target_position_is_removed_from_training(self) -> None:
        users = read_train_prefix_users(D0, root=ROOT)
        selected = select_internal_dev_users(users, count=1283, seed=2023)
        train, internal_dev = build_train_and_internal_dev_examples(users, selected)
        self.assertEqual(len(internal_dev), 1283)
        train_counts = {}
        for example in train:
            train_counts[example.user_id] = train_counts.get(example.user_id, 0) + 1
        users_by_id = {user.user_id: user for user in users}
        for example in internal_dev[:50]:
            user = users_by_id[example.user_id]
            self.assertEqual(example.target, user.train_items[-1])
            self.assertEqual(example.history, user.train_items[:-1][-20:])
            self.assertEqual(train_counts.get(example.user_id, 0), max(0, len(user.train_items) - 2))

    def test_external_targets_are_fail_closed(self) -> None:
        with self.assertRaises(PermissionError):
            read_external_examples(D0, root=ROOT)

    def test_d1_and_monolithic_paths_are_forbidden(self) -> None:
        d1 = ROOT / "artifacts/phase17/s0_audit/shadow_data/Toys/D1/user_sequence.txt"
        monolithic = ROOT / "GRAM/rec_datasets/Toys/user_sequence.txt"
        with self.assertRaises(PermissionError):
            read_train_prefix_users(d1, root=ROOT)
        with self.assertRaises(PermissionError):
            read_train_prefix_users(monolithic, root=ROOT)


if __name__ == "__main__":
    unittest.main()
