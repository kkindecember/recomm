import unittest

import numpy as np

from experiment.phase4.ccrr_r0 import (
    FEATURE_SCHEMA,
    candidate_union,
    feature_matrix,
    popularity_features,
    user_set_sha256,
)


class CCRRR0Tests(unittest.TestCase):
    def test_candidate_union_is_stable_and_deduplicated(self):
        self.assertEqual(candidate_union(["a", "b"], ["b", "c"]), ["a", "b", "c"])

    def test_features_are_target_free_and_finite(self):
        popularity = {"a": 0.8, "b": 0.2, "c": 0.5}
        matrix = feature_matrix(
            ["a", "b"],
            ["b", "c"],
            [2.0, 1.0],
            ["a"],
            ["a", "b", "c"],
            popularity,
            {"a"},
        )
        self.assertEqual(matrix.shape, (3, len(FEATURE_SCHEMA)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_popularity_uses_training_prefix(self):
        percentile, head = popularity_features(
            {"u": ["a", "a", "validation", "test"]}, ["a", "b"]
        )
        self.assertGreater(percentile["a"], percentile["b"])
        self.assertIn("a", head)

    def test_user_hash_is_order_invariant(self):
        self.assertEqual(user_set_sha256(["b", "a"]), user_set_sha256(["a", "b"]))


if __name__ == "__main__":
    unittest.main()
