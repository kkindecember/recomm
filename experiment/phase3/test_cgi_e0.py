import unittest

import numpy as np
import torch

from cgi_e0 import (
    bootstrap_mean,
    condition_mask,
    lexical_mean_logprob,
    selection_hash,
)


class TestCGIE0(unittest.TestCase):
    def test_selection_hash_is_deterministic_and_dataset_scoped(self):
        value = selection_hash(20260724, "Toys", "tail_miss", "u1")
        self.assertEqual(value, selection_hash(20260724, "Toys", "tail_miss", "u1"))
        self.assertNotEqual(value, selection_hash(20260724, "Beauty", "tail_miss", "u1"))

    def test_masks_preserve_coarse_and_target_correct_passage(self):
        base = torch.ones((2, 5, 3), dtype=torch.bool)
        old = condition_mask(base, [2, 4], "minus_oldest")
        self.assertTrue(old[0, 0].all())
        self.assertFalse(old[0, 2].any())
        self.assertFalse(old[1, 4].any())
        newest = condition_mask(base, [2, 4], "minus_newest")
        self.assertFalse(newest[:, 1].any())
        coarse = condition_mask(base, [2, 4], "coarse_only")
        self.assertTrue(coarse[:, 0].all())
        self.assertFalse(coarse[:, 1:].any())

    def test_lexical_score_excludes_eos_and_padding(self):
        labels = torch.tensor([[2, 1, -100]])
        logits = torch.zeros((1, 3, 4))
        logits[0, 0, 2] = 3.0
        logits[0, 1, 1] = -100.0
        score = lexical_mean_logprob(logits, labels, eos_token_id=1)
        expected = torch.log_softmax(logits[0, 0], dim=-1)[2]
        self.assertAlmostEqual(score.item(), expected.item(), places=7)

    def test_bootstrap_is_deterministic(self):
        values = np.asarray([-1.0, 0.5, 2.0, 3.0])
        self.assertEqual(bootstrap_mean(values, 100, 9), bootstrap_mean(values, 100, 9))


if __name__ == "__main__":
    unittest.main()
