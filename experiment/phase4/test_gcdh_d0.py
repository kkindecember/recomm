import unittest

import numpy as np
import torch

from experiment.phase4.gcdh_d0 import exact_rank, normalized_entropy, state_statistics


class GCDHD0Tests(unittest.TestCase):
    def test_exact_rank_uses_catalog_index_tie_break(self):
        logits = torch.tensor([1.0, 3.0, 3.0, 0.0])
        self.assertEqual(exact_rank(logits, 1), 1)
        self.assertEqual(exact_rank(logits, 2), 2)

    def test_normalized_entropy_bounds(self):
        value = normalized_entropy(torch.zeros(5))
        self.assertAlmostEqual(value, 1.0)

    def test_state_statistics_detects_nonconstant_states(self):
        states = np.eye(4, dtype=np.float32)
        result = state_statistics(states, 4)
        self.assertGreater(result["pooled_rms_feature_std"], 0)
        self.assertGreater(result["median_pairwise_cosine_distance"], 0)
        self.assertGreater(result["effective_rank"], 2)


if __name__ == "__main__":
    unittest.main()
