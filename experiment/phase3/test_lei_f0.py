import unittest

import numpy as np

from lei_f0 import (
    decide,
    deterministic_metadata_positions,
    evaluate_gates,
    parse_role_char_spans,
    summarize_scores,
)


class TestLEIF0(unittest.TestCase):
    def test_locked_passage_parser(self):
        text = (
            "item: |a|b; similar items: |c|d, |e|f; "
            "title: a toy; brand: x"
        )
        spans = parse_role_char_spans(text)
        self.assertEqual([text[a:b] for a, b in spans["link"]], ["|a|b"])
        self.assertEqual(
            [text[a:b] for a, b in spans["cf"]], ["|c|d", "|e|f"]
        )
        self.assertEqual(
            [text[a:b] for a, b in spans["metadata"]],
            ["title: a toy; brand: x"],
        )

    def test_deterministic_controls(self):
        args = ([2, 3, 5, 8, 13], 3, 7, "Toys", "u", 2, 4)
        first = deterministic_metadata_positions(*args)
        second = deterministic_metadata_positions(*args)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_summary_sign_convention(self):
        rows = []
        for stratum, adjusted in (("tail_miss", 0.2), ("tail_hit", 0.1)):
            for index in range(8):
                rows.append(
                    {
                        "stratum": stratum,
                        "raw_link_harm": 0.1,
                        "matched_control_effect": -0.1,
                        "adjusted_link_echo": adjusted,
                        "metadata_benefit": 0.2,
                        "raw_cf_harm": 0.0,
                        "raw_all_id_harm": 0.0,
                    }
                )
        result = summarize_scores(rows, 100, 3)
        self.assertAlmostEqual(
            result["tail_miss_minus_tail_hit_adjusted_link_echo"]["mean"],
            0.1,
        )

    def test_gate_and_decision_order(self):
        summary = {
            "tail_miss": {
                "raw_link_harm": {
                    "mean": 0.02,
                    "positive_rate": 0.6,
                    "ci95": [0.001, 0.03],
                },
                "adjusted_link_echo": {
                    "mean": 0.03,
                    "ci95": [0.001, 0.04],
                },
                "metadata_benefit": {
                    "mean": 0.06,
                    "ci95": [0.001, 0.08],
                },
            },
            "tail_miss_minus_tail_hit_adjusted_link_echo": {
                "mean": 0.03,
                "ci95": [0.001, 0.04],
            },
        }
        thresholds = {
            "tail_miss_n": 256,
            "tail_hit_n": 256,
            "repeat_max_abs_error": 1e-7,
            "role_localization_rate": 1.0,
            "matched_control_eligibility_rate": 1.0,
            "raw_link_harm_mean": 0.01,
            "raw_link_harm_positive_rate": 0.55,
            "adjusted_link_echo_mean": 0.02,
            "metadata_benefit_mean": 0.05,
            "failure_association_mean": 0.02,
            "strict_bootstrap_lower_bound": 0.0,
        }
        gates = evaluate_gates(
            summary,
            {"tail_miss": 256, "tail_hit": 256},
            0.0,
            1.0,
            1.0,
            True,
            thresholds,
        )
        self.assertTrue(all(gates.values()))
        results = {"Toys": {"gates": gates}, "Beauty": {"gates": gates}}
        self.assertEqual(decide(results), "F0_MECHANISM_ALLOWED")
        failed = {**gates, "raw_link_harm": False}
        results["Beauty"] = {"gates": failed}
        self.assertEqual(decide(results), "STOP_LEI_NO_RAW_ECHO")


if __name__ == "__main__":
    unittest.main()
