from __future__ import annotations

import math
import unittest

import torch

from experiment.phase17.core.item_aggregation import (
    aggregate_item_scores,
    assert_k1_equivalence,
    ranked_items,
)


class ItemAggregationTests(unittest.TestCase):
    def test_logsumexp_matches_hand_calculation(self) -> None:
        scores = torch.tensor([math.log(0.2), math.log(0.3), math.log(0.4)])
        aggregated = aggregate_item_scores(["a", "a", "b"], scores)
        self.assertAlmostEqual(float(aggregated["a"].exp()), 0.5, places=6)
        self.assertEqual(ranked_items(aggregated), ["a", "b"])

    def test_k1_is_exact(self) -> None:
        scores = torch.tensor([-0.1, -2.0, -1.0])
        assert_k1_equivalence(["a", "b", "c"], scores)

    def test_shape_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_item_scores(["a"], torch.tensor([0.0, 1.0]))
