import math
import unittest
import numpy as np
import torch
from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide
from experiment.phase20.lime_path_mixture_core import PathMixtureGuide, rescore_paths, next_weight


class PathMixtureTests(unittest.TestCase):
    def setUp(self):
        self.paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 3, 1]]
        self.index = ExpertTrie(self.paths, [1, 2, 3])
        self.q = self.index.conditional_logs(np.array([.1, .2, .05]))

    def inputs(self):
        for node, children in enumerate(self.index.children):
            if children:
                values = torch.full((1, 7), -float('inf'), dtype=torch.float64)
                for position, token in enumerate(children):
                    values[0, token] = -1. - .4 * position
                yield torch.tensor([self.index.nodes[node]]), values

    def test_rho_zero_is_exact_old_guide(self):
        old = ProbabilityGuide(self.index, self.q, .3)
        new = PathMixtureGuide(self.index, self.q, .3, 0.)
        for prefix, scores in self.inputs():
            self.assertTrue(torch.equal(old(prefix, scores.clone()), new(prefix, scores.clone())))
        np.testing.assert_allclose(old.rescore(self.paths, self.q, .3), new.rescore(self.paths), atol=1e-14)

    def test_alpha_zero_exact_for_all_rho(self):
        for rho in (.5, 1.):
            guide = PathMixtureGuide(self.index, self.q, 0., rho)
            for prefix, scores in self.inputs():
                self.assertTrue(torch.equal(guide(prefix, scores.clone()), scores))

    def test_mass_and_path_reconstruction(self):
        for rho in (.5, 1.):
            guide = PathMixtureGuide(self.index, self.q, .3, rho)
            actual = {}
            for prefix, scores in self.inputs():
                output = guide(prefix, scores.clone())
                torch.testing.assert_close(torch.logsumexp(output, -1), torch.logsumexp(scores, -1))
                self.assertTrue(torch.isneginf(output[torch.isneginf(scores)]).all())
                actual[tuple(prefix[0].tolist())] = output[0]
            expected = [sum(float(actual[tuple(p[:s])][p[s]]) for s in range(1,len(p)))/(len(p)-1)
                        for p in self.paths]
            np.testing.assert_allclose(guide.rescore(self.paths), expected, atol=1e-12)

    def test_rho_one_sequence_identity_including_legal_mass(self):
        guide = PathMixtureGuide(self.index, self.q, .3, 1.)
        for prefix, scores in self.inputs():
            guide(prefix, scores)
        expected = []
        for path in self.paths:
            lp = lq = lm = 0.
            for step in range(1, len(path)):
                prefix, token = tuple(path[:step]), path[step]
                tokens, native, mass = guide.trace[prefix]
                lp += native[tokens.index(token)]-mass
                lq += self.q[prefix][token]; lm += mass
            expected.append((lm + np.logaddexp(math.log(.7)+lp, math.log(.3)+lq))/(len(path)-1))
        np.testing.assert_allclose(guide.rescore(self.paths), expected, atol=1e-12)

    def test_equal_evidence_keeps_prior_and_rescoring_is_order_independent(self):
        for rho in (0., .5, 1.):
            self.assertAlmostEqual(next_weight(.3, -.7, -.7, .3, rho), .3)
        guide = PathMixtureGuide(self.index, self.q, .3, 1.)
        for prefix, scores in self.inputs():
            guide(prefix, scores)
        np.testing.assert_array_equal(guide.rescore(self.paths)[::-1], guide.rescore(self.paths[::-1]))


if __name__ == '__main__':
    unittest.main()
