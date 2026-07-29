import unittest

import numpy as np
import torch

from experiment.phase4.lndr_n1 import (
    branch_bin,
    gold_margin,
    matched_readout,
    rank_auc,
)


class LNDRN1Tests(unittest.TestCase):
    def test_branch_bins(self):
        self.assertEqual([branch_bin(x) for x in (2, 3, 4, 5, 8, 9)], [
            "2", "3-4", "3-4", "5-8", "5-8", "9+"
        ])

    def test_gold_margin(self):
        logits = torch.tensor([0.0, 2.0, 1.5, 9.0])
        self.assertAlmostEqual(gold_margin(logits, [0, 1, 2], 1), 0.5)
        with self.assertRaises(ValueError):
            gold_margin(logits, [0, 2], 1)

    def test_rank_auc_exact(self):
        self.assertEqual(rank_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0)
        self.assertEqual(rank_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)
        self.assertEqual(rank_auc(np.array([1.0]), np.array([1.0])), 0.5)

    def test_matched_readout_exact_strata(self):
        rows = []
        for index, (cohort, margins) in enumerate(
            [("high_polysemy", [-1.0, -0.5, 1.0]), ("control", [-1.0, 1.0])]
        ):
            for offset, margin in enumerate(margins):
                rows.append({
                    "sample_key": f"{index}-{offset}",
                    "edge_id": f"e-{index}-{offset}",
                    "cohort": cohort,
                    "depth": 0,
                    "target_group": "tail",
                    "branch_bin": "3-4",
                    "gold_margin": margin,
                })
        result = matched_readout(rows, 7)
        self.assertEqual(result["matched_per_cohort"], 2)
        self.assertGreaterEqual(result["high_polysemy_deficit_rate"], 0.5)
        self.assertEqual(result["control_deficit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
