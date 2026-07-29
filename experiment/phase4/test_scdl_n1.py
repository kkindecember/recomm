import unittest

import numpy as np

from experiment.phase4.scdl_n1 import (
    current_metrics,
    joint_assignment,
    top_indices,
)


class SCDLN1Tests(unittest.TestCase):
    def test_current_margin(self):
        weights = np.array([[0.8, 0.2], [0.1, 0.7]])
        result = current_metrics(weights, [0, 1])
        np.testing.assert_allclose(result["margins"], [0.7, 0.5])

    def test_joint_assignment_corrects_swapped_tokens(self):
        weights = np.array([[0.8, 0.1, 0.0], [0.7, 0.9, 0.0]])
        result = joint_assignment(weights, 2, set(), 1.0, 1.0)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["tokens"], [0, 1])
        self.assertTrue((result["margins"] > 0).all())

    def test_top_indices_excludes_special(self):
        row = np.array([9.0, 8.0, 3.0, 2.0])
        self.assertEqual(top_indices(row, 2, {0, 1}).tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
