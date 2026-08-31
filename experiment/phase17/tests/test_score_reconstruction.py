from __future__ import annotations

import unittest

from experiment.phase17.core.metrics import assert_score_reconstruction, reconstruct_sequence_log_score


class ScoreReconstructionTests(unittest.TestCase):
    def test_sequence_score_reconstructs(self) -> None:
        values = [-0.1, -0.2, -0.3]
        self.assertAlmostEqual(reconstruct_sequence_log_score(values), -0.6)
        assert_score_reconstruction(values, -0.6)

    def test_mismatch_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_score_reconstruction([-0.1, -0.2], -1.0)
