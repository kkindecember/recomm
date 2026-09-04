#!/usr/bin/env python3
"""Unit tests for S18-1 cohort, prefix and scientific gate contracts."""

from __future__ import annotations

import unittest

from experiment.phase18.core.s1_contracts import (
    actual_pruner_items,
    catalog_standardized_target,
    cohort_sha256,
    evaluate_domain_gate,
    first_drop_depth,
    fold_views,
    hard_negative_recall,
    lower_empirical_quartile,
    stable_cohort,
)


class CohortAndFoldTests(unittest.TestCase):
    def test_cohort_is_order_independent_and_hashed_with_terminal_newline(self) -> None:
        users = [f"u{index}" for index in range(20)]
        left = stable_cohort("Toys", users, count=8)
        right = stable_cohort("Toys", reversed(users), count=8)
        self.assertEqual(left, right)
        self.assertEqual(len(cohort_sha256(left)), 64)

    def test_only_i_minus_1_and_i0_can_be_constructed(self) -> None:
        histories = {"u": tuple(f"i{index}" for index in range(8))}
        self.assertEqual(fold_views(histories, "I-1")["u"], (histories["u"][:-4], histories["u"][-4]))
        self.assertEqual(fold_views(histories, "I0")["u"], (histories["u"][:-3], histories["u"][-3]))
        for sealed in ("I1", "I2"):
            with self.assertRaises(PermissionError):
                fold_views(histories, sealed)

    def test_lower_positive_empirical_quartile(self) -> None:
        self.assertEqual(lower_empirical_quartile([0, 1, 2, 3, 4, 100]), 2)


class PrefixTests(unittest.TestCase):
    def test_first_drop_and_legal_actual_pruner(self) -> None:
        target = (10, 20, 30, 1)
        active = {
            1: {(10,), (11,)},
            2: {(10, 20), (10, 21)},
            3: {(10, 21, 40), (11, 22, 41)},
        }
        self.assertEqual(first_drop_depth(active, target), 3)
        returned = {
            "legal_sibling": (10, 20, 31, 1),
            "different_parent": (10, 21, 30, 1),
            "illegal_child": (10, 20, 99, 1),
        }
        actual = actual_pruner_items(returned, target, 3, legal_children={30, 31})
        self.assertEqual(actual, {"legal_sibling"})
        self.assertEqual(hard_negative_recall(["legal_sibling", "x"], actual), 1.0)

    def test_target_survives_complete_path(self) -> None:
        target = (4, 5, 1)
        active = {1: {(4,)}, 2: {(4, 5)}, 3: {(4, 5, 1)}}
        self.assertIsNone(first_drop_depth(active, target))


class TeacherAndGateTests(unittest.TestCase):
    def test_catalog_standardization_is_not_fold_vector_recentering(self) -> None:
        self.assertAlmostEqual(catalog_standardized_target(3.0, [1.0, 2.0, 3.0]), 1.224744871391589)

    def test_gate_precedence(self) -> None:
        gates = {
            "per_domain_headroom_min": 0.05,
            "per_domain_beam200_only_events_min": 100,
            "per_domain_nonempty_actual_pruner_fraction_min": 0.5,
            "per_domain_k8_actual_pruner_recall_min": 0.5,
            "absolute_cf_target_z_mean_drift_max_exclusive": 1.0,
            "gate_1_5_failure": "NO_ACTIONABLE_PREFIX_BOTTLENECK",
            "gate_6_failure": "CF_TEACHER_UNSTABLE",
        }
        passing = {
            "pooled_headroom": 0.06,
            "beam200_only_events": 120,
            "nonempty_actual_pruner_fraction": 0.6,
            "k8_actual_pruner_recall": 0.7,
            "finite_and_trie_legal": True,
            "cf_target_z_mean_drift": 0.2,
        }
        self.assertEqual(evaluate_domain_gate(passing, gates)["decision"], "ACTIONABILITY_PASS")
        unstable = dict(passing, cf_target_z_mean_drift=1.0)
        self.assertEqual(evaluate_domain_gate(unstable, gates)["decision"], "CF_TEACHER_UNSTABLE")
        no_action = dict(unstable, pooled_headroom=0.049)
        self.assertEqual(evaluate_domain_gate(no_action, gates)["decision"], "NO_ACTIONABLE_PREFIX_BOTTLENECK")


if __name__ == "__main__":
    unittest.main()
