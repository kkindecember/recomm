from __future__ import annotations

import unittest

from experiment.phase17.core.s2r_r2_evaluator import (
    compare_family_predictions,
    paired_bootstrap_delta,
    user_ranking_contribution,
)


class S2RR2EvaluatorTests(unittest.TestCase):
    def test_user_contribution_uses_item_rank(self) -> None:
        row = user_ranking_contribution("target", ["a", "target", "b"])
        self.assertEqual(row["hit@5"], 1.0)
        self.assertGreater(row["ndcg@5"], 0.0)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_delta([1, 0, 1], [0, 0, 0], seed=17)
        second = paired_bootstrap_delta([1, 0, 1], [0, 0, 0], seed=17)
        self.assertEqual(first, second)

    def test_strong_promotion_requires_mechanism_and_cohort_gates(self) -> None:
        users = [f"u{index}" for index in range(6)]
        treatment = {
            user: user_ranking_contribution("t", ["t"]) for user in users
        }
        control = {
            user: user_ranking_contribution("t", ["x"]) for user in users
        }
        result = compare_family_predictions(
            treatment=treatment,
            control=control,
            cohorts=[users[:2], users[2:4], users[4:]],
            mechanism_metrics={
                "valid_item_rate": 1.0,
                "multi_path_item_rate": 0.5,
            },
            family="latte",
            bootstrap_replicates=100,
            seed=17,
        )
        self.assertEqual(result["decision"], "STRONG_PROMOTE")
        self.assertEqual(result["positive_ndcg_cohorts"], 3)

    def test_positive_effect_without_mechanism_is_rejected(self) -> None:
        users = [f"u{index}" for index in range(6)]
        treatment = {
            user: user_ranking_contribution("t", ["t"]) for user in users
        }
        control = {
            user: user_ranking_contribution("t", ["x"]) for user in users
        }
        result = compare_family_predictions(
            treatment=treatment,
            control=control,
            cohorts=[users[:2], users[2:4], users[4:]],
            mechanism_metrics={"valid_item_rate": 1.0, "set_token_recovery": 0.0,
                               "treatment_generation_seconds": 1.0,
                               "control_generation_seconds": 2.0},
            family="setrec",
            bootstrap_replicates=100,
            seed=17,
        )
        self.assertEqual(result["decision"], "REJECT")


if __name__ == "__main__":
    unittest.main()
