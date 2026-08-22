from __future__ import annotations

import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from toys_s3b_full_validation import (  # noqa: E402
    paired_bootstrap,
    portfolio_at_2,
    ranking_metrics,
    summarize_arm,
)


class TestToysS3BFullValidation(unittest.TestCase):
    def test_portfolio_at_2_preserves_prefix_and_inserts_at_ranks_9_10(self):
        gram = [f"g{i}" for i in range(50)]
        resolver = ["g0", "c1", "c2", *[f"r{i}" for i in range(47)]]
        ranking = portfolio_at_2(gram, resolver, {"c1", "c2"})
        self.assertEqual(ranking[:8], gram[:8])
        self.assertEqual(ranking[8:10], ["c1", "c2"])
        self.assertEqual(len(ranking), 50)
        self.assertEqual(len(set(ranking)), 50)

    def test_ranking_metrics_use_strict_rank_cutoffs(self):
        ranking = [f"i{i}" for i in range(50)]
        self.assertEqual(ranking_metrics(ranking, "i9")["ndcg@10"], 1.0 / __import__("math").log2(11))
        self.assertEqual(ranking_metrics(ranking, "i10")["ndcg@10"], 0.0)
        self.assertEqual(ranking_metrics(ranking, "missing")["hit@50"], 0)

    def test_paired_bootstrap_is_deterministic_and_paired(self):
        rows = []
        for index in range(20):
            rows.append({
                "is_cold": True,
                "target_item": str(index),
                "metrics": {
                    "b0": {"hit@50": 0.0, "ndcg@10": 0.0},
                    "b1": {"hit@50": 1.0, "ndcg@10": 0.5},
                },
            })
        first = paired_bootstrap(rows, "b1", "b0", "hit@50", "cold", resamples=100, seed=7)
        second = paired_bootstrap(rows, "b1", "b0", "hit@50", "cold", resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first["ci_low"], 0)

    def test_summary_counts_unique_targets_and_hit_events(self):
        rows = [
            {"is_cold": True, "target_item": "x", "metrics": {"b0": {"hit@50": 1, "ndcg@10": 0.5}}},
            {"is_cold": True, "target_item": "x", "metrics": {"b0": {"hit@50": 0, "ndcg@10": 0.0}}},
            {"is_cold": True, "target_item": "y", "metrics": {"b0": {"hit@50": 1, "ndcg@10": 1.0}}},
        ]
        summary = summarize_arm(rows, "b0", "cold")
        self.assertEqual(summary["hit_events"], 2)
        self.assertEqual(summary["unique_target_items"], 2)
        self.assertEqual(summary["unique_hit_target_items"], 2)


if __name__ == "__main__":
    unittest.main()
