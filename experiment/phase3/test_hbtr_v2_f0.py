#!/usr/bin/env python3

import unittest
from collections import Counter

from hbtr_v2_f0 import (
    analyze_dataset,
    build_tail_quantiles,
    margins,
    prefix_weight,
    tail_weight,
)


class HbtrV2F0Test(unittest.TestCase):
    def test_quantile_head_tail_alignment_and_ties(self):
        popularity = Counter({"a": 10, "b": 8, "c": 8, "d": 4, "e": 1})
        quantiles, head_count = build_tail_quantiles(popularity)
        self.assertEqual(head_count, 1)
        self.assertEqual(quantiles["a"], 0.0)
        self.assertGreater(quantiles["b"], 0.0)
        self.assertLess(quantiles["b"], quantiles["c"])
        self.assertEqual(quantiles["e"], 1.0)

    def test_locked_weight_and_margin_caps(self):
        self.assertEqual(prefix_weight(3), 2.0)
        self.assertEqual(prefix_weight(99), 2.0)
        self.assertEqual(tail_weight(0.0), 1.0)
        self.assertEqual(tail_weight(1.0), 2.0)
        result = margins(3, 1.0)
        self.assertEqual(result["C1"], 0.1)
        self.assertEqual(result["C2"], 0.2)
        self.assertEqual(result["C3_v2"], 0.2)
        self.assertEqual(result["C4_v2"], 0.4)

    def test_invalid_weights_fail_closed(self):
        with self.assertRaises(ValueError):
            prefix_weight(-1)
        with self.assertRaises(ValueError):
            tail_weight(-0.1)
        with self.assertRaises(ValueError):
            tail_weight(1.1)

    def test_dataset_analysis_and_gates(self):
        popularity = Counter(
            {"a": 10, "b": 8, "c": 6, "d": 4, "e": 2, "f": 1}
        )
        cache = {
            "rows": [
                {
                    "positive_item": "c",
                    "positive_frequency": 6,
                    "prefix_depths": [0, 1],
                },
                {
                    "positive_item": "f",
                    "positive_frequency": 1,
                    "prefix_depths": [1, 2],
                },
            ]
        }
        gates = {
            "tail_nontrivial_row_rate_min_each_dataset": 0.20,
            "joint_nontrivial_row_rate_min_each_dataset": 0.10,
            "C4_v2_vs_C2_pair_margin_exact_equality_rate_max_each_dataset": 0.80,
            "tail_weight_max": 2.0,
            "margin_max": 0.4,
        }
        result, distributions = analyze_dataset(cache, popularity, gates)
        self.assertTrue(result["gates"]["passed"])
        self.assertEqual(len(distributions), 4)
        self.assertEqual(result["metrics"]["tail_nontrivial_row_rate"], 1.0)
        self.assertEqual(result["metrics"]["joint_nontrivial_row_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
