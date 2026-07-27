#!/usr/bin/env python3

import unittest

from s0b_reliability_probe import abstention_active, confidence_features, rerank


class ReliabilityProbeTests(unittest.TestCase):
    def test_inactive_preserves_exact_ranking(self):
        ranking = rerank(
            ["a", "b"], [-0.1, -0.2], [0.0, 1.0], [0, 1], False, 0.2, 1
        )
        self.assertEqual(ranking, ["a", "b"])

    def test_active_can_promote_supported_candidate(self):
        ranking = rerank(
            ["a", "b"], [-0.1, -0.2], [0.0, 1.0], [0, 1], True, 0.2, 1
        )
        self.assertEqual(ranking, ["b", "a"])

    def test_min_support_blocks_single_support(self):
        ranking = rerank(
            ["a", "b"], [-0.1, -0.2], [0.0, 1.0], [0, 1], True, 0.2, 2
        )
        self.assertEqual(ranking, ["a", "b"])

    def test_abstention_rule(self):
        self.assertTrue(abstention_active(0.8, 0.1, 1, 0.75))
        self.assertTrue(abstention_active(0.8, 0.0, 2, 0.75))
        self.assertFalse(abstention_active(0.7, 0.2, 2, 0.75))
        self.assertFalse(abstention_active(0.8, 0.01, 1, 0.75))

    def test_confidence_uses_history_and_neighbors_only(self):
        scores, supports, top1, gap, max_support = confidence_features(
            ["h1", "h2"], ["x", "z"], {"h1": ["x"], "h2": ["x", "y"]}, 0.25
        )
        self.assertEqual(supports, [2, 0])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(top1, scores[0])
        self.assertGreater(gap, 0)
        self.assertEqual(max_support, 2)


if __name__ == "__main__":
    unittest.main()
