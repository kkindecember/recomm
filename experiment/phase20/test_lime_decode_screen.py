import unittest

import numpy as np
import torch

from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide, mixed_log_values


class TestProbabilityGuidance(unittest.TestCase):
    def setUp(self):
        self.paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 3, 1]]
        self.index = ExpertTrie(self.paths, [1, 2, 3])
        self.q = self.index.conditional_logs(np.log([.2, .3, .5]), 1.)

    def test_catalog_distribution_maps_to_paths(self):
        for path, expected in zip(self.paths, [.2, .3, .5]):
            logp = sum(self.q[tuple(path[:i])][path[i]] for i in range(1, len(path)))
            self.assertAlmostEqual(np.exp(logp), expected)

    def test_zero_identity_and_invalid_mask(self):
        score = torch.tensor([[-torch.inf, -torch.inf, -2., -3., -torch.inf, -torch.inf]])
        guide = ProbabilityGuide(self.index)
        torch.testing.assert_close(guide(torch.tensor([[0]]), score.clone()), score, rtol=0, atol=0)
        changed = ProbabilityGuide(self.index, self.q, .3)(torch.tensor([[0]]), score.clone())
        self.assertTrue(torch.equal(torch.isneginf(changed), torch.isneginf(score)))
        torch.testing.assert_close(changed.exp().sum(), score.exp().sum())

    def test_exact_late_rescoring(self):
        guide = ProbabilityGuide(self.index, self.q, .3)
        accumulated = 0.
        for prefix, tokens, vals, chosen in [([0], [2,3], [-2.,-3.], 2),
                    ([0,2], [4,5], [-1.,-2.], 4), ([0,2,4], [1], [-.5], 1)]:
            scores = torch.full((1, 6), -torch.inf)
            scores[0, tokens] = torch.tensor(vals)
            out = guide(torch.tensor([prefix]), scores)
            accumulated += float(out[0, chosen])
        actual = guide.rescore([self.paths[0]], self.q, .3)[0]
        self.assertAlmostEqual(actual, accumulated/3, places=6)

    def test_legal_mass_preserves_parent_comparison(self):
        p = np.log([.1, .2])
        out = mixed_log_values(p, np.log(.3), np.log([.8,.2]), .3)
        np.testing.assert_allclose(np.exp(out), .7*np.exp(p)+.3*.3*np.array([.8,.2]))


if __name__ == '__main__':
    unittest.main()
