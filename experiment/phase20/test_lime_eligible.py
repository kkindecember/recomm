import unittest
import numpy as np
import torch
from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide
from experiment.phase20.lime_eligible_core import EligibleTrie


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 3, 6, 1], [0, 7, 1]]
        self.index = ExpertTrie(self.paths, [1, 2, 3, 4])
        self.scores = np.asarray([.9, .2, .4, .5])

    def test_empty_filter_reproduces_original(self):
        index = EligibleTrie(self.index, [])
        self.assertEqual(index.children, self.index.children)
        self.assertEqual(index.conditional_logs(self.scores), self.index.conditional_logs(self.scores))

    def test_shared_branch_survives_and_forbidden_leaf_does_not(self):
        index = EligibleTrie(self.index, [1, 4])
        self.assertEqual(index.allowed_tokens(0, [0]), [2, 3])
        self.assertEqual(index.allowed_tokens(0, [0, 2]), [5])
        self.assertEqual(index.allowed_tokens(0, [0, 2, 4]), [])
        self.assertEqual(index.allowed_tokens(0, [0, 7]), [])
        self.assertEqual(index.allowed_tokens(0, [0, 2, 5]), [1])

    def test_global_mass_differs_from_plain_local_mask(self):
        index = EligibleTrie(self.index, [1, 4])
        q = index.conditional_logs(self.scores)
        plain = index.condition_original_logs(self.index.conditional_logs(self.scores))
        expected = np.exp(20 * self.scores[1:3]); expected /= expected.sum()
        np.testing.assert_allclose([np.exp(q[(0,)][t]) for t in [2, 3]], expected)
        # The excluded high-score item 1 no longer lends its mass to branch 2.
        self.assertLess(q[(0,)][2], plain[(0,)][2])
        for view in (q, plain):
            for prefix, values in view.items():
                self.assertAlmostEqual(sum(np.exp(list(values.values()))), 1.)

    def test_existing_guide_preserves_filtered_mass_and_scores(self):
        index = EligibleTrie(self.index, [1, 4])
        q = index.conditional_logs(self.scores)
        guide = ProbabilityGuide(index, q, .3)
        accumulated = {}
        for node, children in enumerate(index.children):
            if not children:
                continue
            prefix = index.nodes[node]
            scores = torch.full((1, 10), -float('inf'), dtype=torch.float64)
            for position, token in enumerate(children):
                scores[0, token] = -1. - position
            mass = torch.logsumexp(scores, -1)
            updated = guide(torch.tensor([prefix]), scores.clone())
            torch.testing.assert_close(torch.logsumexp(updated, -1), mass)
            self.assertTrue(torch.isneginf(updated[torch.isneginf(scores)]).all())
            accumulated[prefix] = updated[0]
        paths = [self.paths[1], self.paths[2]]
        manual = [sum(float(accumulated[tuple(p[:s])][p[s]]) for s in range(1, len(p))) / (len(p)-1) for p in paths]
        np.testing.assert_allclose(guide.rescore(paths, q, .3), manual, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
