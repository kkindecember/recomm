import unittest

import numpy as np
import torch

from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide
from experiment.phase20.lime_mechanism_core import ScheduledGuide, select_alpha


class TestScheduledGuide(unittest.TestCase):
    def setUp(self):
        # The first token is mandatory; the first real branch occurs after it.
        self.paths = [[0, 2, 3, 5, 1], [0, 2, 3, 6, 1], [0, 2, 4, 1]]
        self.index = ExpertTrie(self.paths, [1, 2, 3])
        self.q = self.index.conditional_logs(np.log([.2, .3, .5]), 1.)

    def score(self, legal, vals):
        score = torch.full((1, 7), -torch.inf)
        score[0, legal] = torch.tensor(vals)
        return score

    def test_delay_counts_branching_not_tokens(self):
        g = ScheduledGuide(self.index, self.q, .3, 'delayed_branch')
        self.assertEqual(g.prior_branches[(0, 2)], 0)
        self.assertEqual(g.prior_branches[(0, 2, 3)], 1)
        for prefix, legal, vals in [([0], [2], [-.4]), ([0, 2], [3, 4], [-2., -3.])]:
            x = self.score(legal, vals)
            torch.testing.assert_close(g(torch.tensor([prefix]), x.clone()), x, rtol=0, atol=0)
        x = self.score([5, 6], [-2., -3.])
        y = g(torch.tensor([[0, 2, 3]]), x.clone())
        self.assertFalse(torch.equal(x, y))
        torch.testing.assert_close(x.exp().sum(), y.exp().sum())

    def test_overlap_reduces_disagreement_without_labels(self):
        g = ScheduledGuide(self.index, self.q, .3, 'agreement')
        p = np.log([.9, .1])
        self.assertAlmostEqual(g.effective_alpha((0, 2), p, 0., p), .3)
        self.assertAlmostEqual(g.effective_alpha((0, 2), p, 0., np.log([.1, .9])), .06)

    def test_constant_and_zero_match_original(self):
        x = self.score([3, 4], [-2., -3.])
        prefix = torch.tensor([[0, 2]])
        expected = ProbabilityGuide(self.index, self.q, .3)(prefix, x.clone())
        actual = ScheduledGuide(self.index, self.q, .3, 'constant')(prefix, x.clone())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for policy in ('constant', 'agreement', 'delayed_branch'):
            torch.testing.assert_close(ScheduledGuide(self.index, self.q, 0., policy)(prefix, x.clone()),
                                       x, rtol=0, atol=0)

    def test_reconstruction_mask_mass_and_impossible_branch(self):
        for policy in ('constant', 'delayed_branch', 'agreement'):
            g = ScheduledGuide(self.index, self.q, .3, policy)
            total = 0.
            steps = [([0], [2], [-.4], 2), ([0, 2], [3, 4], [-2., -3.], 3),
                     ([0, 2, 3], [5, 6], [-1., -2.], 5), ([0, 2, 3, 5], [1], [-.5], 1)]
            for prefix, legal, vals, chosen in steps:
                x = self.score(legal, vals)
                y = g(torch.tensor([prefix]), x.clone())
                self.assertTrue(torch.equal(torch.isneginf(x), torch.isneginf(y)))
                torch.testing.assert_close(x.exp().sum(), y.exp().sum())
                total += float(y[0, chosen])
            self.assertAlmostEqual(g.rescore([self.paths[0]])[0], total / 4, places=6)
            x = torch.full((1, 7), -torch.inf)
            self.assertTrue(torch.isneginf(g(torch.tensor([[0, 6]]), x)).all())

    def test_selection_tie_and_zero_fallback(self):
        rows = [dict(arms={n: dict(rank=3) for n in ('GRAM', 'agreement_0.1', 'agreement_0.3')})]
        self.assertEqual(select_alpha(rows, 'agreement')[0], 0.)
        rows[0]['arms']['agreement_0.1']['rank'] = 1
        self.assertEqual(select_alpha(rows, 'agreement')[0], .1)


if __name__ == '__main__':
    unittest.main()
