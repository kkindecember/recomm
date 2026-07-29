import math
import unittest

import torch

from experiment.phase4.fpug_n1 import (
    legal_child_ce,
    normalized_entropy,
    recency_quartile,
)


class FakeTrie:
    def get(self, prefix):
        return {
            (0,): [2, 3],
            (0, 2): [1],
        }[tuple(prefix)]


class FPUGN1Tests(unittest.TestCase):
    def test_recency_quartiles(self):
        self.assertEqual(
            [recency_quartile(rank, 8) for rank in range(8)],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_entropy(self):
        self.assertAlmostEqual(normalized_entropy([10, 10, 10, 10]), 1.0)
        self.assertAlmostEqual(normalized_entropy([40, 0, 0, 0]), 0.0)

    def test_legal_child_ce(self):
        logits = torch.zeros(1, 2, 5)
        logits[0, 0, 2] = 2.0
        logits[0, 0, 3] = 1.0
        losses, counts = legal_child_ce(logits, [[0, 2, 1]], FakeTrie(), 1)
        self.assertEqual(counts, [1])
        self.assertAlmostEqual(losses[0], math.log1p(math.exp(-1.0)), places=6)


if __name__ == "__main__":
    unittest.main()
