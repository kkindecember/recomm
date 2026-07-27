import unittest

import torch

from experiment.phase4.gcdh_p0_split import history_bin, largest_remainder


class GCDHP0Tests(unittest.TestCase):
    def test_largest_remainder_exact(self):
        result = largest_remainder({"a": 3, "b": 7}, 4)
        self.assertEqual(sum(result.values()), 4)
        self.assertLessEqual(result["a"], 3)
        self.assertLessEqual(result["b"], 7)

    def test_history_bins(self):
        self.assertEqual(history_bin(5), "1-5")
        self.assertEqual(history_bin(6), "6-10")
        self.assertEqual(history_bin(21), "21+")

    def test_masked_coarse_pool_formula(self):
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0]])))

    def test_balanced_softmax_adjusts_head_logit(self):
        logits = torch.zeros(1, 2)
        log_counts = torch.log(torch.tensor([10.0, 1.0]))
        adjusted = logits + log_counts
        self.assertGreater(adjusted[0, 0], adjusted[0, 1])


if __name__ == "__main__":
    unittest.main()
