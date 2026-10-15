"""Math and search-contract tests before freezing the GPU experiment."""
import unittest
import numpy as np
import torch
from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide
from experiment.phase20.lime_prefix_expert_core import Descendants, PrefixExperts


class PrefixExpertTest(unittest.TestCase):
    def setUp(self):
        self.paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 2, 6, 1], [0, 3, 7, 1]]
        self.index = ExpertTrie(self.paths, [1, 2, 3, 4])
        self.structure = Descendants(self.index)
        self.scores = np.asarray([[.9, .8, .2, .1], [.1, .3, .9, .8], [.3, .3, .3, .3]], dtype='float32')
        self.prior = np.asarray([.5, .3, .2], dtype='float32')

    def make(self, scores=None, coefficient=1.):
        return PrefixExperts(self.structure, self.scores if scores is None else scores,
                             self.prior, [1], .08, coefficient=coefficient)

    def test_uniform_scores_respect_branch_size(self):
        p = self.make(np.ones((3, 4), dtype='float32'))
        np.testing.assert_allclose(p.weights((0,)), self.prior, atol=1e-8)
        np.testing.assert_allclose(p.evidence_cache[self.index.root], 0, atol=1e-14)
        q = p.views[False][(0,)]
        self.assertAlmostEqual(np.exp(q[2]), .75)
        self.assertAlmostEqual(np.exp(q[3]), .25)

    def test_zero_coefficient_matches_original_fusion(self):
        p = self.make(coefficient=0.)
        fused = (self.prior[:, None] * self.scores).sum(0)
        for calibrated in (False, True):
            scores = fused.copy()
            if calibrated:
                scores[0] -= np.float32(.08)
            reference = self.index.conditional_logs(scores)
            for prefix, vals in reference.items():
                for token, expected in vals.items():
                    self.assertAlmostEqual(p.views[calibrated][prefix][token], expected, places=6)

    def test_local_fusion_matches_brute_force_and_bounds(self):
        p = self.make()
        for prefix in ((0,), (0, 2)):
            w = p.weights(prefix)
            ratio = w / p.prior
            self.assertLessEqual(ratio.max() / ratio.min(), 2 + 1e-12)
            self.assertAlmostEqual(w.sum(), 1.)
            for cal in (False, True):
                s = (w.astype('float32')[:, None] * self.scores).sum(0)
                if cal:
                    s[0] -= np.float32(.08)
                groups = {}
                for i, path in enumerate(self.paths):
                    if tuple(path[:len(prefix)]) == prefix:
                        groups.setdefault(path[len(prefix)], []).append(i)
                totals = {t: np.exp(20 * s.astype('float64')[ii]).sum() for t, ii in groups.items()}
                for t in groups:
                    self.assertAlmostEqual(p.views[cal][prefix][t], np.log(totals[t] / sum(totals.values())), places=10)
        self.assertGreater(abs(p.weights((0,)) - p.weights((0, 2))).max(), 1e-4)
        self.assertNotEqual(p.views[False][(0, 2)], p.views[True][(0, 2)])

    def test_guide_keeps_legal_mass_and_reconstructs_paths(self):
        p = self.make()
        guide = ProbabilityGuide(self.index, p.views[True], .3)
        accumulated = {}
        for prefix, children in zip(self.index.nodes, self.index.children):
            if not children:
                continue
            logits = torch.full((1, 10), -float('inf'), dtype=torch.float64)
            for k, t in enumerate(children):
                logits[0, t] = -1. - k
            oldmass = torch.logsumexp(logits, -1)
            out = guide(torch.tensor([prefix]), logits.clone())
            torch.testing.assert_close(torch.logsumexp(out, -1), oldmass)
            self.assertTrue(torch.isneginf(out[torch.isneginf(logits)]).all())
            accumulated[prefix] = out[0]
        manual = [sum(float(accumulated[tuple(path[:s])][path[s]]) for s in range(1, len(path))) / (len(path) - 1)
                  for path in self.paths]
        np.testing.assert_allclose(guide.rescore(self.paths, p.views[True], .3), manual, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
