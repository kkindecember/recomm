import unittest

import numpy as np

from cfsat_c0 import analyze, choose_donor, jaccard


class CFSATC0Test(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(jaccard({"a"}, {"b"}), 0.0)
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_choose_donor_excludes_target_and_overlap(self):
        neighbors = {
            "anchor": ["a", "b"],
            "bad_target": ["target", "c"],
            "bad_overlap": ["a", "d"],
            "good": ["e", "f"],
        }
        choice = choose_donor(
            seed=1,
            dataset="D",
            user="u",
            anchor="anchor",
            target="target",
            candidates=["bad_target", "bad_overlap", "good"],
            neighbors=neighbors,
            k=2,
            maximum_jaccard=0.2,
        )
        self.assertEqual(choice, ("good", 0.0))

    def test_analyze_uses_user_cluster_means(self):
        rows = []
        for user, margin in (("u1", 0.2), ("u2", 0.3)):
            for depth in (0, 1, 2):
                rows.append(
                    {
                        "user": user,
                        "split": "audit",
                        "depth": depth,
                        "cf_margin": margin,
                        "helpful": True,
                        "sensitivity_deficit": margin < 0.1,
                    }
                )
        config = {
            "bootstrap_iterations": 100,
            "scientific_gates_per_dataset": {
                "mean_margin_ci95_lower_strictly_greater_than": 0.0,
                "positive_user_mean_margin_rate_min": 0.55,
                "helpful_node_rate_min": 0.6,
                "sensitivity_deficit_margin": 0.1,
                "sensitivity_deficit_rate_among_helpful_min": 0.2,
                "deficit_user_coverage_min": 0.3,
                "minimum_nontrivial_depths_with_support": 2,
                "minimum_helpful_nodes_per_supported_depth": 1,
            },
        }
        result = analyze(rows, config, 1)
        self.assertAlmostEqual(
            result["user_cluster_mean_cf_margin"]["mean"], 0.25
        )
        self.assertEqual(result["audit_users"], 2)
        self.assertEqual(result["supported_nontrivial_depths"], 2)


if __name__ == "__main__":
    unittest.main()
