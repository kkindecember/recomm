import unittest

import torch

from experiment.phase4.tcdr_s0 import (
    differentiable_correlation,
    differentiable_legal_path_score,
)


class FakeTrie:
    def get(self, prefix):
        return {(0,): [2, 3], (0, 2): [1]}[tuple(prefix)]


class TCDRS0Tests(unittest.TestCase):
    def test_correlation_gradient(self):
        left = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        right = torch.tensor([1.0, 3.0, 5.0])
        value = differentiable_correlation(left, right, 1e-8)
        value.backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertGreater(float(left.grad.norm()), 0)

    def test_legal_path_score_gradient(self):
        logits = torch.zeros(2, 5, requires_grad=True)
        logits.data[0, 2] = 2.0
        logits.data[0, 3] = 1.0
        labels = torch.tensor([2, 1])
        score = differentiable_legal_path_score(logits, labels, FakeTrie(), 1)
        score.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()

