from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from capacity_aware_assign import TailCandidate, assign_unique_tails  # noqa: E402


def candidate(t4: str, t5: str, score: float, r4: int, r5: int) -> TailCandidate:
    return TailCandidate(t4, t5, score, r4, r5)


class CapacityAwareAssignmentTests(unittest.TestCase):
    def test_exact_minimum_cost_assignment_and_invariants(self):
        rows = [
            ("w", ("w0", "w1", "w2", "w3", "w4")),
            ("c1", ("p0", "p1", "p2", "a", "a")),
            ("c2", ("p0", "p1", "p2", "a", "a")),
        ]
        candidates = {
            "c1": [candidate("a", "a", 10.0, 1, 1), candidate("b", "b", 9.0, 2, 2)],
            "c2": [candidate("a", "a", 10.0, 1, 1), candidate("b", "b", 0.0, 2, 2)],
        }
        output, report = assign_unique_tails(rows, {"c1", "c2"}, candidates)
        output_map = dict(output)
        self.assertEqual(output_map["w"], rows[0][1])
        self.assertEqual(output_map["c1"], ("p0", "p1", "p2", "b", "b"))
        self.assertEqual(output_map["c2"], ("p0", "p1", "p2", "a", "a"))
        self.assertEqual(report["output_collision"]["duplicate_excess"], 0)
        self.assertEqual(report["cold_appended_suffix_count"], 0)
        self.assertEqual(report["cold_prefix_levels_preserved"], 3)

    def test_exact_warm_path_is_reserved(self):
        rows = [
            ("w", ("p0", "p1", "p2", "a", "a")),
            ("c", ("p0", "p1", "p2", "a", "a")),
        ]
        candidates = {
            "c": [candidate("a", "a", 10.0, 1, 1), candidate("b", "b", 8.0, 2, 2)]
        }
        output, _report = assign_unique_tails(rows, {"c"}, candidates)
        self.assertEqual(dict(output)["c"], ("p0", "p1", "p2", "b", "b"))

    def test_infeasible_topk_fails_closed(self):
        rows = [
            ("c1", ("p0", "p1", "p2", "a", "a")),
            ("c2", ("p0", "p1", "p2", "a", "a")),
        ]
        only = candidate("a", "a", 10.0, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            assign_unique_tails(rows, {"c1", "c2"}, {"c1": [only], "c2": [only]})

    def test_non_five_token_cold_input_rejected(self):
        rows = [("c", ("p0", "p1", "p2", "a", "a", "0"))]
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            assign_unique_tails(
                rows,
                {"c"},
                {"c": [candidate("a", "a", 10.0, 1, 1)]},
            )


if __name__ == "__main__":
    unittest.main()
