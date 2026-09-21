import unittest
import numpy as np
from experiment.phase20.catalog_scope_core import (
    eligible_mask, masked_z, mixed_scores, ranks, restricted, topk, split_users, select_weight)


class ScopeChecks(unittest.TestCase):
    def test_scoring_identity_and_coverage_loss(self):
        d = np.array([[0., 4, 3, 2, 1], [0., 4, 3, 2, 1]])
        mask = eligible_mask([[], []], 5)
        score = mixed_scores(d, d, .3, mask)
        candidates = np.array([[4, 2, 3], [4, 2, 3]])
        targets = np.array([3, 1])
        local = restricted(score, candidates)
        np.testing.assert_array_equal(local, score[np.arange(2)[:, None], candidates])
        np.testing.assert_allclose(local, np.array([[1, 3, 2], [1, 3, 2]]), atol=1e-14, rtol=0)
        np.testing.assert_array_equal(ranks(score, np.arange(5), targets), [3, 1])
        self.assertEqual(ranks(local, candidates, targets)[0], 2)
        self.assertGreater(ranks(local, candidates, targets)[1], 50)

    def test_mask_and_ties(self):
        mask = eligible_mask([[4], [2]], 6)
        score = np.where(mask, np.ones((2, 6)), -np.inf)
        ids = np.arange(6)
        np.testing.assert_array_equal(topk(score, ids, 3), [[1, 2, 3], [1, 3, 4]])
        np.testing.assert_array_equal(ranks(score, ids, [3, 4]), [3, 3])
        perm = np.array([5, 3, 1, 0, 4, 2])
        np.testing.assert_array_equal(topk(score[:, perm], ids[perm], 3), topk(score, ids, 3))

    def test_normalization_excludes_ineligible(self):
        mask = eligible_mask([[1], [2]], 5)
        x = np.array([[999, -999, 1, 2, 3], [999, 1, -999, 2, 3]], float)
        z = masked_z(x, mask)
        np.testing.assert_allclose(z[mask].reshape(2, 3).mean(1), 0, atol=1e-15)
        np.testing.assert_allclose(z[mask].reshape(2, 3).std(1), 1, atol=1e-15)
        np.testing.assert_allclose(masked_z(x*7+4, mask), z, atol=1e-14)
        self.assertTrue(np.isfinite(masked_z(np.ones_like(x), mask)).all())

    def test_disjoint_split_and_frozen_selection(self):
        users = [f'u{x}' for x in range(20)]
        cal, evaluation = split_users(users, 'Toys', 7, 5)
        self.assertFalse(set(cal) & set(evaluation))
        self.assertEqual(len(set(cal) | set(evaluation)), 20)
        table = [dict(weight=w, B=dict(N10=.1, H10=.2)) for w in (0, .5, 1)]
        self.assertEqual(select_weight(table, 'B'), .5)


if __name__ == '__main__':
    unittest.main()
