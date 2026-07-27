#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from smbr_i0_d import FEATURES, feature_row, split_for_user


class SMBRI0DTest(unittest.TestCase):
    def test_split_is_deterministic_and_total(self):
        config = {
            "split_hash_modulus": 100,
            "splits": {
                "fit": {"hash_buckets": [0, 60]},
                "calibration": {"hash_buckets": [60, 80]},
                "audit": {"hash_buckets": [80, 100]},
            },
        }
        for user in ("u1", "u2", "u3", "u4"):
            first = split_for_user(7, "Toys", user, config)
            second = split_for_user(7, "Toys", user, config)
            self.assertEqual(first, second)
            self.assertIn(first, config["splits"])

    def test_features_are_history_only_and_finite(self):
        template = {
            "recoverable_metadata_tokens": 8.0,
            "displaced_cf_tokens": 4.0,
            "current_metadata_lost": 10.0,
            "current_metadata_retention": 0.5,
            "current_metadata_visible": 10.0,
            "current_cf_visible": 20.0,
            "popularity_train": 3.0,
            "popularity_stratum": "nonzero_bottom50",
            "top_k_similar_item": 5.0,
        }
        row = feature_row(["a", "b"], {"a": template, "b": template})
        self.assertEqual(list(row), FEATURES)
        self.assertEqual(row["history_length"], 2.0)
        self.assertEqual(row["recoverable_sum"], 16.0)
        self.assertEqual(row["tail_item_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
