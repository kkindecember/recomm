import unittest

import numpy as np

from experiment.phase20.personal_popularity_core import (
    reconstruct_training, incidence, neighborhoods, candidate_popularity,
    shuffled_users, donor_neighbors, ranks_from_scores, standardize_rows,
    crossfit, user_folds,
)


class PersonalPopularityTests(unittest.TestCase):
    def test_reconstruction_preserves_recent_order_and_detects_gaps(self):
        seq = [1, 2, 3, 1, 4, 5]
        rows = [{'user_id': 'u', 'history': seq[max(0, i-3):i], 'target': seq[i]}
                for i in range(1, len(seq))]
        users, sequences, recent, complete = reconstruct_training(rows, 3)
        self.assertEqual(sequences['u'], seq)
        self.assertEqual(recent, [[1, 4, 5]])
        self.assertEqual(complete, [[1, 2, 3, 4, 5]])
        with self.assertRaises(ValueError):
            reconstruct_training(rows[:2] + rows[3:], 3)

    def test_sparse_jaccard_matches_set_oracle_with_ties_and_self_exclusion(self):
        sets = [{1, 2}, {1, 2}, {1}, {2}, {3}, {4}]
        bank = incidence([sorted(x) for x in sets], 5)
        got, scores, positive = neighborhoods(bank, np.arange(len(sets)), 3, 2)
        for i, s in enumerate(sets):
            expected = sorted((j for j in range(len(sets)) if i != j),
                              key=lambda j: (-len(s & sets[j])/len(s | sets[j]), j))[:3]
            self.assertEqual(got[i].tolist(), expected)
            self.assertEqual(positive[i], sum(bool(s & t) for j, t in enumerate(sets) if i != j))

    def test_shuffled_pp_scores_actual_candidate_ids_and_excludes_query(self):
        sets = [{1, 2}, {1, 3}, {2}, {4}, {3, 4}, {1, 4}]
        bank = incidence([sorted(x) for x in sets], 5)
        ids, _, _ = neighborhoods(bank, np.arange(6), 3, 2)
        mapping = shuffled_users(list('abcdef'), 7)
        shuffled = donor_neighbors(ids, mapping, np.arange(6), 2)
        candidates = np.asarray([[1, 2, 4], [4, 3, 2], [2, 1, 3], [3, 4, 1], [1, 3, 4], [2, 4, 3]])
        pp = candidate_popularity(bank, shuffled, candidates, 2)
        for i in range(6):
            self.assertNotIn(i, shuffled[i])
            expected = [sum(c in sets[j] for j in shuffled[i])/2 for c in candidates[i]]
            np.testing.assert_array_equal(pp[i], expected)
        self.assertTrue(np.all(mapping != np.arange(6)))

    def test_crossfit_coefficient_never_uses_held_fold_outcomes(self):
        folds = user_folds([str(i) for i in range(20)], 5)
        baseline = np.full(20, 10)
        alternative = np.full(20, 8)
        _, first, _ = crossfit([baseline, alternative], folds, [0., .1])
        changed = alternative.copy()
        changed[folds == 0] = 51
        _, second, _ = crossfit([baseline, changed], folds, [0., .1])
        self.assertEqual(first[0], second[0])
        _, ties, full = crossfit([baseline, baseline], folds, [0., .1])
        self.assertEqual(full['eta'], 0.)
        self.assertTrue(all(x['eta'] == 0 for x in ties))

    def test_constant_feature_identity_and_missing_target(self):
        base = np.asarray([[2., 2., 1.], [1., 3., 2.]])
        constant = standardize_rows(np.ones_like(base))
        np.testing.assert_array_equal(constant, np.zeros_like(base))
        np.testing.assert_array_equal(ranks_from_scores(base + constant, [1, -1]), [2, 4])


if __name__ == '__main__':
    unittest.main()
