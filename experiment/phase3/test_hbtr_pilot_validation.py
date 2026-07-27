#!/usr/bin/env python3

import unittest

from hbtr_pilot import build_validation_samples, summarize_metric_rows
from hbtr_pilot_split import history_bin


class HbtrPilotValidationTests(unittest.TestCase):
    def test_validation_keeps_raw_history_length_before_model_truncation(self):
        items = [f"i{index}" for index in range(25)]
        samples = build_validation_samples(
            {"u": items},
            {"u"},
            {item: f"text-{item}" for item in items},
            {item: f"<{item}>" for item in items},
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["raw_history_length"], 23)
        self.assertEqual(len(samples[0]["history_items"]), 20)
        self.assertEqual(history_bin(samples[0]["raw_history_length"]), "21+")

    def test_empty_subgroup_summary_is_explicit_and_safe(self):
        summary = summarize_metric_rows([])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["NDCG@10"])

    def test_nonempty_subgroup_summary_uses_all_rows(self):
        rows = [
            {"Recall@5": 0.0, "NDCG@5": 0.0, "Recall@10": 1.0, "NDCG@10": 0.5},
            {"Recall@5": 1.0, "NDCG@5": 0.5, "Recall@10": 1.0, "NDCG@10": 0.75},
        ]
        summary = summarize_metric_rows(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["Recall@5"], 0.5)
        self.assertEqual(summary["NDCG@10"], 0.625)


if __name__ == "__main__":
    unittest.main()
