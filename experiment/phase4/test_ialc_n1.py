import math
import unittest

import torch

from experiment.phase4.ialc_n1 import (
    select_unique_user_samples,
    support_metrics,
)


class IALCN1Tests(unittest.TestCase):
    def test_support_metrics_exact(self):
        logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        result = support_metrics(logits, [1, 3], 3)
        self.assertAlmostEqual(result["legal_mass"], 0.6, places=6)
        self.assertAlmostEqual(result["illegal_mass"], 0.4, places=6)
        self.assertAlmostEqual(result["loss_gap"], -math.log(0.6), places=6)
        self.assertEqual(result["full_rank"], 1)
        self.assertEqual(result["legal_rank"], 1)

    def test_gold_must_be_legal(self):
        with self.assertRaises(ValueError):
            support_metrics(torch.zeros(4), [0, 1], 3)

    def test_unique_user_stratification(self):
        rows = [
            {"sample_key": "a1", "user_id": "a", "positive_item": "h"},
            {"sample_key": "a2", "user_id": "a", "positive_item": "h"},
            {"sample_key": "b1", "user_id": "b", "positive_item": "h"},
            {"sample_key": "c1", "user_id": "c", "positive_item": "t"},
            {"sample_key": "d1", "user_id": "d", "positive_item": "t"},
        ]
        selected = select_unique_user_samples(rows, {"h"}, 7, "Toys", 2, 2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["user_id"] for row in selected}), 4)


if __name__ == "__main__":
    unittest.main()
