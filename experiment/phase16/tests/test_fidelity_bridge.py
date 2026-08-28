from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from fidelity_bridge import (  # noqa: E402
    active_edit_position,
    fixed_width_probe,
    guided_redraft_candidates,
    lexical_probe,
    longest_warm_prefix,
    masked_mean_log_probability,
    optimizer_satisfied,
    run_bridge_checks,
    solve_delta,
    strict_accept,
)


class TestFidelityBridge(unittest.TestCase):
    def test_all_registered_bridge_checks_pass(self):
        results = run_bridge_checks()
        self.assertGreaterEqual(len(results), 15)
        self.assertEqual({row["status"] for row in results}, {"PASS"})

    def test_variable_score_matches_fixed_width_prefix_mean(self):
        values = [-0.1, -0.2, -8.0]
        self.assertAlmostEqual(masked_mean_log_probability(values, 2), -0.15)
        self.assertEqual(longest_warm_prefix((1, 2, 7), [(1, 2, 3)], minimum=2), 2)

    def test_acceptance_is_strict(self):
        self.assertFalse(strict_accept(-1.8, -1.8))
        self.assertTrue(strict_accept(-1.799, -1.8))

    def test_redraft_uses_live_prefix_and_no_repeat(self):
        paths = {"x": (1, 2), "y": (1, 3), "z": (4,)}
        self.assertEqual(
            guided_redraft_candidates(
                ["z", "x", "y"], paths, [(1,)], prefix_depth=1,
                already_drafted={"x"}, draft_size=2,
            ),
            ["y"],
        )

    def test_fixed_width_and_lexical_probe_ordering_match(self):
        logits = torch.full((513,), -9.0)
        start = 257
        logits[start + 4] = 3.0
        logits[start + 5] = 2.0
        fixed = fixed_width_probe(logits, position=1, target_code=4)
        lexical = lexical_probe(logits, target_token=start + 4, legal_tokens=range(start, start + 256))
        self.assertEqual(fixed.is_argmax, lexical.is_argmax)

    def test_probability_threshold_is_not_optimizer_gate(self):
        logits = torch.tensor([1.0, 0.99, 0.98, 0.97])
        probe = lexical_probe(logits, target_token=0, legal_tokens=(0, 1, 2, 3))
        self.assertTrue(probe.is_argmax)
        self.assertFalse(probe.cache_probe_pass)
        self.assertTrue(optimizer_satisfied(logits, 0))

    def test_closed_form_solve_matches_inverse_expression(self):
        residual = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
        keys = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        covariance = torch.eye(2)
        actual = solve_delta(residual, keys, covariance, 10.0)
        expected = (residual.double() @ keys.double().T) @ torch.linalg.inv(
            keys.double() @ keys.double().T + 10.0 * covariance.double()
        )
        self.assertTrue(torch.allclose(actual.double(), expected))

    def test_variable_paths_do_not_route_completed_or_eos_rows(self):
        paths = [(10,), (11, 12)]
        self.assertEqual(active_edit_position((), paths), 0)
        self.assertIsNone(active_edit_position((10,), paths))
        self.assertIsNone(active_edit_position((11,), paths, eos_seen=True))


if __name__ == "__main__":
    unittest.main()

