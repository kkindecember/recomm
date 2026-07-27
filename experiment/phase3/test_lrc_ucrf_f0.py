#!/usr/bin/env python3

import unittest

import numpy as np

from lrc_ucrf_f0 import (
    FEATURE_NAMES,
    choose_threshold,
    retrieval_features,
    stable_user_is_calibration,
)


class LrcUcrfF0Tests(unittest.TestCase):
    def test_feature_schema_and_no_target_argument(self):
        features, items = retrieval_features(
            ["h1", "h2"], {"h1": ["a", "b"], "h2": ["a", "c"]}
        )
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(items[0], "a")
        self.assertTrue(np.isfinite(features).all())

    def test_user_split_is_deterministic(self):
        self.assertEqual(
            stable_user_is_calibration("user-1"),
            stable_user_is_calibration("user-1"),
        )

    def test_threshold_is_selected_from_four_locked_rates(self):
        y = np.array([0] * 80 + [1] * 20)
        probability = np.linspace(0, 1, 100)
        selected, candidates = choose_threshold(y, probability)
        self.assertEqual(len(candidates), 4)
        self.assertIn(selected["target_active_rate"], (0.1, 0.2, 0.3, 0.4))


if __name__ == "__main__":
    unittest.main()
