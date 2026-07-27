#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pens_h0_d import equalize_fine_norm, training_position_exposure  # noqa: E402


class PENSH0DTest(unittest.TestCase):
    def test_exposure_replays_prefix_augmentation(self):
        sequences = {
            "u1": ["a", "b", "c", "validation", "test"],
            "u2": ["a", "b", "validation", "test"],
        }
        exposure = training_position_exposure(sequences, max_history=3)
        np.testing.assert_array_equal(exposure, np.asarray([3, 3, 1, 0]))

    def test_final_two_items_do_not_change_exposure(self):
        first = {"u": ["a", "b", "c", "v1", "t1"]}
        second = {"u": ["a", "b", "c", "v2", "t2"]}
        np.testing.assert_array_equal(
            training_position_exposure(first, 3),
            training_position_exposure(second, 3),
        )

    def test_equal_norm_preserves_direction_and_coarse(self):
        generator = torch.Generator().manual_seed(7)
        table = torch.randn(21, 8, generator=generator)
        table[1:] *= torch.arange(1, 21).unsqueeze(1)
        equalized, audit = equalize_fine_norm(table)
        self.assertTrue(torch.equal(equalized[0], table[0]))
        self.assertLessEqual(audit["equal_norm_max_abs_error"], 1e-5)
        self.assertGreaterEqual(audit["direction_cosine_min"], 0.999999)
        expected = float(torch.quantile(table[1:].norm(dim=1), 0.5))
        self.assertAlmostEqual(audit["target_norm"], expected, places=6)

    def test_equal_norm_rejects_zero_vector(self):
        table = torch.ones(21, 4)
        table[3].zero_()
        with self.assertRaises(ValueError):
            equalize_fine_norm(table)


if __name__ == "__main__":
    unittest.main()
