import math
import unittest

import numpy as np

from experiment.phase4.fcrd_f0 import (
    fuse_indices,
    popularity_log_probability,
)


class FCRDF0Tests(unittest.TestCase):
    def test_fusion_endpoints(self):
        self.assertEqual(fuse_indices([1, 2], [3, 4], 0.0)[:2], [1, 2])
        self.assertEqual(fuse_indices([1, 2], [3, 4], 1.0)[:2], [3, 4])

    def test_smoothed_popularity(self):
        sequences = {"u": ["a", "a", "v", "t"]}
        values = popularity_log_probability(sequences, ["a", "b"])
        probabilities = np.exp(values)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_residual_boosts_rare(self):
        logits = np.asarray([0.0, 0.0])
        log_pop = np.log(np.asarray([0.9, 0.1]))
        residual = logits - log_pop
        self.assertGreater(residual[1], residual[0])


if __name__ == "__main__":
    unittest.main()
