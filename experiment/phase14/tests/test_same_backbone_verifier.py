from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from same_backbone_verifier import (  # noqa: E402
    mean_path_log_likelihood,
    pad_candidate_paths,
    portfolio_at_2,
)


class TestSameBackboneVerifier(unittest.TestCase):
    def test_mean_path_log_likelihood_masks_padding_and_length_normalizes(self):
        labels = torch.tensor([[0, 1], [1, -100]])
        logits = torch.tensor(
            [
                [[3.0, 0.0], [0.0, 3.0]],
                [[0.0, 3.0], [100.0, -100.0]],
            ]
        )
        scores = mean_path_log_likelihood(logits, labels)
        self.assertAlmostEqual(float(scores[0]), float(scores[1]), places=6)

    def test_pad_candidate_paths_uses_minus_100(self):
        labels = pad_candidate_paths([(3, 4), (5,)], torch.device("cpu"))
        self.assertEqual(labels.tolist(), [[3, 4], [5, -100]])

    def test_portfolio_matches_frozen_top8_then_two_cold_rule(self):
        gram = [f"g{i}" for i in range(12)]
        resolver = ["g0", "c0", "c1", "c2"]
        ranking = portfolio_at_2(gram, resolver, {"c0", "c1", "c2"})
        self.assertEqual(ranking[:10], [*gram[:8], "c0", "c1"])
        self.assertEqual(len(ranking), len(set(ranking)))


if __name__ == "__main__":
    unittest.main()
