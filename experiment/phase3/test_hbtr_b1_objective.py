#!/usr/bin/env python3

import unittest

import torch

from hbtr_b1_objective import (
    canonical_cache_sha256,
    component_margin,
    common_prefix_depth,
    joint_margin,
    pairwise_ranking_loss,
    prefix_weight,
    sequence_log_scores,
    tail_weight,
    total_loss,
    training_popularity,
    validate_cache_row,
)


class HbtrB1ObjectiveTests(unittest.TestCase):
    def test_prefix_and_tail_weights_are_capped(self):
        self.assertEqual(common_prefix_depth(("a", "b"), ("a", "c")), 1)
        self.assertEqual(prefix_weight(0), 1.0)
        self.assertEqual(prefix_weight(99), 2.0)
        self.assertEqual(tail_weight(100, 10), 1.0)
        self.assertEqual(tail_weight(0, 1000), 2.0)
        self.assertLessEqual(joint_margin(99, 0, 1000), 0.4)

    def test_component_controls_are_distinct_and_nested(self):
        c1 = component_margin("C1", 2, 0, 100)
        c2 = component_margin("C2", 2, 0, 100)
        c3 = component_margin("C3", 2, 0, 100)
        c4 = component_margin("C4", 2, 0, 100)
        self.assertEqual(c1, 0.1)
        self.assertGreater(c2, c1)
        self.assertGreater(c3, c1)
        self.assertAlmostEqual(c4, c2 * c3 / c1)
        with self.assertRaises(ValueError):
            component_margin("C0", 0, 1, 1)

    def test_sequence_score_masks_padding_and_includes_valid_tokens(self):
        logits = torch.tensor(
            [[[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]], dtype=torch.float32
        )
        labels = torch.tensor([[0, 1, -100]])
        score = sequence_log_scores(logits, labels)
        expected = torch.log_softmax(logits[:, :2], dim=-1)[0, [0, 1], [0, 1]].mean()
        self.assertTrue(torch.allclose(score[0], expected))

    def test_ranking_loss_is_monotonic_and_empty_mask_is_zero(self):
        negatives = torch.tensor([[0.0, -0.2]])
        margins = torch.tensor([[0.1, 0.2]])
        low_positive = pairwise_ranking_loss(torch.tensor([0.1]), negatives, margins)
        high_positive = pairwise_ranking_loss(torch.tensor([1.1]), negatives, margins)
        self.assertLess(high_positive.item(), low_positive.item())
        zero = pairwise_ranking_loss(
            torch.tensor([0.1]), negatives, margins, torch.zeros_like(margins).bool()
        )
        self.assertEqual(zero.item(), 0.0)

    def test_lambda_zero_exactly_falls_back_to_ce(self):
        ce = torch.tensor(1.25, requires_grad=True)
        rank = torch.tensor(9.0, requires_grad=True)
        combined = total_loss(ce, rank, ranking_lambda=0.0)
        self.assertEqual(combined.item(), ce.item())
        combined.backward()
        self.assertEqual(ce.grad.item(), 1.0)
        self.assertEqual(rank.grad.item(), 0.0)

    def test_training_popularity_excludes_last_two_items(self):
        counts = training_popularity({"u": ["a", "b", "validation", "test"]})
        self.assertEqual(counts, {"a": 1, "b": 1})

    def test_cache_validation_and_hash_are_deterministic(self):
        row = {
            "sample_key": "u:target",
            "user_id": "u",
            "positive_item": "target",
            "positive_rank": 11,
            "history_items": ["seen"],
            "negative_items": ["n1", "n2", "n3", "n4"],
            "prefix_depths": [1, 0, 2, 1],
            "positive_frequency": 2,
        }
        valid = {"target", "seen", "n1", "n2", "n3", "n4"}
        validate_cache_row(row, valid_items=valid)
        self.assertEqual(canonical_cache_sha256([row]), canonical_cache_sha256([row]))
        bad = dict(row, negative_items=["target"] , prefix_depths=[1])
        with self.assertRaises(ValueError):
            validate_cache_row(bad, valid_items=valid)


if __name__ == "__main__":
    unittest.main()
