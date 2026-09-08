#!/usr/bin/env python3
"""Contracts for the S18-1R capacity-adjusted Gate."""

from __future__ import annotations

import unittest

from experiment.phase18.core.s1_contracts import (
    capped_recall_metrics,
    evaluate_capacity_gate,
)


class CapacityAdjustedRecallTests(unittest.TestCase):
    def test_raw_recall_can_be_impossible_while_capacity_is_fully_covered(self) -> None:
        metrics = capped_recall_metrics([(8, 50), (8, 50)], k=8)
        self.assertAlmostEqual(metrics["raw_micro_recall"], 0.16)
        self.assertAlmostEqual(metrics["raw_micro_mechanical_ceiling"], 0.16)
        self.assertAlmostEqual(metrics["capacity_normalized_recall"], 1.0)
        self.assertAlmostEqual(metrics["event_any_hit_rate"], 1.0)

    def test_metrics_preserve_event_and_micro_views(self) -> None:
        metrics = capped_recall_metrics([(1, 1), (4, 8), (0, 2)], k=8)
        self.assertEqual(metrics["event_count"], 3)
        self.assertEqual(metrics["intersection_total"], 5)
        self.assertEqual(metrics["actual_pruner_total"], 11)
        self.assertEqual(metrics["topk_capacity_total"], 11)
        self.assertAlmostEqual(metrics["raw_micro_recall"], 5 / 11)
        self.assertAlmostEqual(metrics["event_any_hit_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["event_full_coverage_rate"], 1 / 3)

    def test_invalid_top_k_intersection_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            capped_recall_metrics([(9, 50)], k=8)


class CapacityGateTests(unittest.TestCase):
    def test_both_repaired_checks_are_required(self) -> None:
        gates = {
            "capacity_normalized_recall_min": 0.8,
            "event_any_hit_rate_min": 0.8,
        }
        passing = {
            "capacity_normalized_recall": 0.9,
            "event_any_hit_rate": 0.85,
        }
        self.assertEqual(
            evaluate_capacity_gate(passing, gates)["decision"],
            "CAPACITY_ADJUSTED_ACTIONABILITY_PASS",
        )
        failing = dict(passing, event_any_hit_rate=0.79)
        self.assertEqual(
            evaluate_capacity_gate(failing, gates)["decision"],
            "CAPACITY_ADJUSTED_ACTIONABILITY_FAIL",
        )


if __name__ == "__main__":
    unittest.main()
