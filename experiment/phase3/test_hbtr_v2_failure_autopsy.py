#!/usr/bin/env python3

import unittest

from hbtr_v2_failure_autopsy import (
    activation_analysis,
    component_margin,
    percentile,
    transition_summary,
    validate_lineage,
)


class HbtrV2FailureAutopsyTest(unittest.TestCase):
    def test_locked_margin_values(self):
        self.assertEqual(component_margin("C1", 3, 1, 9), 0.1)
        self.assertEqual(component_margin("C2", 3, 1, 9), 0.2)
        self.assertEqual(component_margin("C3", 0, 1, 9), 0.2)
        self.assertEqual(component_margin("C4", 3, 1, 9), 0.4)

    def test_percentile_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)

    def test_activation_analysis(self):
        cache = {
            "samples": 4,
            "rows": [
                {
                    "prefix_depths": [0, 1],
                    "positive_frequency": 2,
                },
                {
                    "prefix_depths": [0, 0],
                    "positive_frequency": 8,
                },
            ],
        }
        metrics, margins = activation_analysis(cache, median_frequency=5.0)
        self.assertEqual(metrics["eligible_all_rate"], 0.5)
        self.assertEqual(metrics["prefix_nontrivial_pair_rate"], 0.25)
        self.assertEqual(metrics["prefix_nontrivial_row_rate"], 0.5)
        self.assertEqual(metrics["tail_nontrivial_row_rate"], 0.5)
        self.assertEqual(metrics["joint_nontrivial_row_rate"], 0.5)
        self.assertGreater(margins["C4"]["max"], margins["C1"]["max"])

    def test_lineage_and_transitions(self):
        base = {
            "u1": {
                "user_id": "u1",
                "target_item": "i1",
                "target_group": "tail",
                "history_bin": "1-5",
                "rank": "11",
                "Recall@10": "0",
                "NDCG@10": "0",
            },
            "u2": {
                "user_id": "u2",
                "target_item": "i2",
                "target_group": "head",
                "history_bin": "6-10",
                "rank": "2",
                "Recall@10": "1",
                "NDCG@10": "0.6309297536",
            },
        }
        candidate = {
            "u1": {
                **base["u1"],
                "rank": "10",
                "Recall@10": "1",
                "NDCG@10": "0.2890648263",
            },
            "u2": dict(base["u2"]),
        }
        rows = {"C0": base, "C1": candidate, "C2": candidate, "C3": candidate,
                "C4": candidate}
        self.assertEqual(validate_lineage(rows)["mismatches"], 0)
        overall = [
            row
            for row in transition_summary("Toys", "C0", "C4", rows)
            if row["group"] == "overall"
        ][0]
        self.assertEqual(overall["promoted_to_top10"], 1)
        self.assertEqual(overall["net_top10_promotions"], 1)


if __name__ == "__main__":
    unittest.main()
