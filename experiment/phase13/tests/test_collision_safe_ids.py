"""Unit tests for Phase-13 collision-safe cold-ID postprocessing."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
sys.path.insert(0, str(PROTOCOL_DIR))

from make_collision_safe_ids import make_collision_safe  # noqa: E402


class TestCollisionSafeIds(unittest.TestCase):
    def test_unique_ids_are_unchanged(self):
        rows = [("w", ("a", "b")), ("c", ("x", "y"))]
        output, report = make_collision_safe(rows, {"c"})
        self.assertEqual(output, rows)
        self.assertEqual(report["n_cold_modified"], 0)
        self.assertEqual(report["output_collision"]["duplicate_excess"], 0)

    def test_warm_exact_overlap_suffixes_cold_only(self):
        rows = [("w", ("a", "b")), ("c", ("a", "b"))]
        output, report = make_collision_safe(rows, {"c"})
        self.assertEqual(dict(output)["w"], ("a", "b"))
        self.assertEqual(dict(output)["c"], ("a", "b", "0"))
        self.assertEqual(report["n_warm_overlap_groups"], 1)

    def test_all_members_of_cold_collision_get_suffix(self):
        rows = [
            ("w", ("q", "r")),
            ("c1", ("a", "b")),
            ("c2", ("a", "b")),
        ]
        output, report = make_collision_safe(rows, {"c1", "c2"})
        self.assertEqual(dict(output)["c1"], ("a", "b", "0"))
        self.assertEqual(dict(output)["c2"], ("a", "b", "1"))
        self.assertEqual(report["n_cold_collision_groups"], 1)

    def test_existing_warm_suffixes_are_skipped(self):
        rows = [
            ("w0", ("a", "b", "0")),
            ("w2", ("a", "b", "2")),
            ("c1", ("a", "b")),
            ("c2", ("a", "b")),
        ]
        output, report = make_collision_safe(rows, {"c1", "c2"})
        result = dict(output)
        self.assertEqual(result["w0"], ("a", "b", "0"))
        self.assertEqual(result["w2"], ("a", "b", "2"))
        self.assertEqual(result["c1"], ("a", "b", "1"))
        self.assertEqual(result["c2"], ("a", "b", "3"))
        self.assertEqual(report["suffix_candidates_skipped"], 2)
        self.assertEqual(report["output_collision"]["duplicate_excess"], 0)

    def test_row_order_is_preserved(self):
        rows = [
            ("c2", ("a", "b")),
            ("w", ("a", "b")),
            ("c1", ("a", "b")),
        ]
        output, _report = make_collision_safe(rows, {"c1", "c2"})
        self.assertEqual([item for item, _tokens in output], ["c2", "w", "c1"])

    def test_missing_cold_item_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "absent from ID file"):
            make_collision_safe([("w", ("a",))], {"missing"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
