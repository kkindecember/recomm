from __future__ import annotations

import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from pseudo_cold_audit import (  # noqa: E402
    audit_filtered_training,
    log_frequency_buckets,
    select_pseudo_cold_items,
)


class TestPseudoColdSelection(unittest.TestCase):
    def test_is_deterministic_disjoint_and_reference_stratified(self):
        warm = {f"w{i}" for i in range(20)}
        cold = {f"c{i}" for i in range(20)}
        frequencies = {
            **{f"w{i}": i + 1 for i in range(20)},
            **{f"c{i}": i + 1 for i in range(20)},
        }
        buckets = log_frequency_buckets(frequencies, warm | cold, 4)
        first, report = select_pseudo_cold_items(warm, cold, buckets, 0.25, 1401)
        second, _ = select_pseudo_cold_items(warm, cold, buckets, 0.25, 1401)
        self.assertEqual(first, second)
        self.assertFalse(first & cold)
        self.assertEqual(len(first), 5)
        self.assertEqual(sum(report["selected_bucket_counts"].values()), 5)


class TestLeakageAudit(unittest.TestCase):
    def test_removes_pseudo_items_from_student_histories_and_targets(self):
        sequences = [
            ("u1", ["warm0", "pseudo", "warm1", "warm2", "val", "test"]),
            ("u2", ["warm0", "warm1", "pseudo", "warm3", "val", "test"]),
        ]
        held, student_sequences, report = audit_filtered_training(
            sequences, {"pseudo"}, {"cold"}, max_history=20
        )
        self.assertEqual(len(held), 2)
        self.assertTrue(all("pseudo" not in row["visible_history"] for row in held))
        self.assertTrue(
            all("pseudo" not in row["train_items"] for row in student_sequences)
        )
        leak_keys = [key for key in report if key.endswith("_leaks")]
        self.assertTrue(all(report[key] == 0 for key in leak_keys))

    def test_real_cold_in_train_prefix_hard_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Real cold"):
            audit_filtered_training(
                [("u", ["warm", "cold", "warm2", "val", "test"])],
                {"pseudo"},
                {"cold"},
                20,
            )


if __name__ == "__main__":
    unittest.main()
