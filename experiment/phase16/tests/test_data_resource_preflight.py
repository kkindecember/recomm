from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from data_resource_preflight import allocate_stratified, bucket, quintile_thresholds, read_sequences


class DataResourcePreflightTests(unittest.TestCase):
    def test_validation_position_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe_projection.txt"
            path.write_text("u1 a b SECRET_VALIDATION\n", encoding="utf-8")
            self.assertEqual(read_sequences(path), [("u1", ["a", "b"])])

    def test_duplicate_users_hard_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe_projection.txt"
            path.write_text("u1 a b\nu1 c d\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate projected user"):
                read_sequences(path)

    def test_catalog_quintiles_are_deterministic(self) -> None:
        counts = {str(index): index for index in range(1, 11)}
        thresholds = quintile_thresholds(counts)
        self.assertEqual(thresholds, (2, 4, 6, 8))
        self.assertEqual([bucket(value, thresholds) for value in (1, 3, 5, 7, 10)], [0, 1, 2, 3, 4])

    def test_stratified_selection_is_reproducible_and_capacity_safe(self) -> None:
        eligible = {f"w{index}" for index in range(10)}
        reference = {f"c{index}" for index in range(6)}
        strata = {
            **{f"w{index}": (5, index % 2) for index in range(10)},
            **{f"c{index}": (5, index % 2) for index in range(6)},
        }
        first, audit = allocate_stratified(eligible, reference, strata, 4, 1502, "toy")
        second, _ = allocate_stratified(eligible, reference, strata, 4, 1502, "toy")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertEqual(audit["selected"], 4)


if __name__ == "__main__":
    unittest.main()
