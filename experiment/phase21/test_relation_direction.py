"""Scientific boundary checks: event direction, leakage, and legal rankings."""
import unittest
import numpy as np
from experiment.phase20.catalog_scope_core import topk, ranks
from experiment.phase21.relation_direction_core import (
    transition_counts, normalized_neighbors, history_matrix, fit_head, predict_head,
)


class RelationBoundaryTests(unittest.TestCase):
    def test_direction_and_gap_do_not_create_backward_events(self):
        counts = transition_counts([[1, 2, 3], [1, 2]], 5)
        self.assertEqual(counts[1, 2], 2.)
        self.assertEqual(counts[2, 1], 0.)
        self.assertEqual(counts[1, 3], .5)
        self.assertEqual(counts[2, 3], 1.)
        self.assertEqual(counts.diagonal().sum(), 0.)
        forward = normalized_neighbors(counts)
        reverse = normalized_neighbors(counts.T.tocsr())
        self.assertGreater(forward[1, 2], 0.)
        self.assertEqual(forward[2, 1], 0.)
        self.assertGreater(reverse[2, 1], 0.)

    def test_equal_topk_budget_ties_and_recent_history(self):
        counts = transition_counts([[1, 4], [1, 3], [1, 2]], 5)
        graph = normalized_neighbors(counts, k=2)
        self.assertEqual(graph[1].indices.tolist(), [2, 3])
        history = history_matrix([[1, 1, 2]], 5)
        self.assertAlmostEqual(history[0, 1], 1.)
        self.assertAlmostEqual(history[0, 2], 1.)
        long = history_matrix([list(range(1, 24))], 25)
        self.assertEqual(long[0, 3], 0.)
        self.assertAlmostEqual(long[0, 4], 1 / 20)

    def test_zero_support_and_history_exclusion_do_not_count_hits(self):
        score = np.array([[0., 1., -np.inf, 0.]])
        score[score <= 0] = -np.inf
        np.testing.assert_array_equal(topk(score, np.arange(1, 5), 3), [[2, 0, 0]])
        self.assertGreater(ranks(score, np.arange(1, 5), [1])[0], 50)

    def test_fold_exclusion_and_uncovered_rows(self):
        rng = np.random.RandomState(7)
        x = rng.normal(size=(10, 5, 3))
        prior = rng.normal(size=(10, 5))
        valid = np.ones((10, 5), bool)
        ids = np.tile(np.arange(1, 6), (10, 1))
        targets = np.array([1, 2, 3, 4, 5, 9, 2, 3, 4, 5])
        valid[4, 4] = False
        first = fit_head(x, prior, valid, ids, targets, np.arange(6))
        np.testing.assert_array_equal(first['used'], np.arange(4))
        x2 = x.copy(); x2[6:] += 100
        targets2 = targets.copy(); targets2[6:] = 1
        second = fit_head(x2, prior, valid, ids, targets2, np.arange(6))
        for key in ('mean', 'std', 'coef'):
            np.testing.assert_array_equal(first[key], second[key])
        original = predict_head(first, x, prior)
        perm = [4, 0, 2, 1, 3]
        changed = predict_head(first, x[:, perm], prior[:, perm])
        np.testing.assert_allclose(changed, original[:, perm], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
