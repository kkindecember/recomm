import unittest

import numpy as np

from experiment.phase4.tcdr_n1 import (
    bootstrap_mean,
    collaborative_cosine,
    frequency_bin,
    lcp_length,
)


class TCDRN1Tests(unittest.TestCase):
    def test_lcp(self):
        self.assertEqual(lcp_length((1, 2, 3), (1, 2, 4)), 2)
        self.assertEqual(lcp_length((1,), (2,)), 0)

    def test_collaborative_cosine(self):
        self.assertAlmostEqual(
            collaborative_cosine({"u1", "u2"}, {"u2", "u3"}), 0.5
        )

    def test_frequency_bin(self):
        self.assertEqual([frequency_bin(value) for value in (1, 2, 3, 4)], [0, 1, 1, 2])

    def test_bootstrap_positive(self):
        result = bootstrap_mean(np.ones(16), 100, 2023)
        self.assertGreater(result["ci_low"], 0)


if __name__ == "__main__":
    unittest.main()

