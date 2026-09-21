import unittest
import numpy as np

from experiment.phase20.harm_identifiability_core import (
    accept, features, fold_ids, item_ranks, partitions, select_threshold, target_ranks,
)


def fixture():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(8, 50))
    return dict(baseline=base, proposal=base + rng.normal(0, .2, base.shape),
        generation=rng.normal(size=base.shape), collaborative=rng.normal(size=base.shape),
        candidate_frequencies=rng.integers(1, 50, size=base.shape),
        history_frequencies=rng.integers(1, 50, size=(8, 20)),
        history_mask=np.tile(np.arange(20) < 7, (8, 1)))


class ContractTests(unittest.TestCase):
    def test_feature_interface_cannot_accept_labels(self):
        data = fixture()
        x, names, simple = features(**data)
        self.assertEqual(x.shape, (8, 36))
        self.assertEqual(len(simple), 9)
        self.assertTrue(all(names[i].startswith('pop_') for i in simple))
        for forbidden in ['targets', 'positions', 'target_frequencies', 'users', 'candidate_ids', 'coverage']:
            with self.assertRaises(TypeError):
                features(**data, **{forbidden: np.ones(8)})

    def test_joint_candidate_permutation_preserves_features_without_ties(self):
        data = fixture()
        x = features(**data)[0]
        permutation = np.random.default_rng(9).permutation(50)
        permuted = {k: (v[:, permutation] if v.shape[1] == 50 else v) for k, v in data.items()}
        np.testing.assert_allclose(features(**permuted)[0], x, rtol=1e-10, atol=1e-10)

    def test_identity_edit_has_zero_edit_and_popularity_movement(self):
        data = fixture()
        data['proposal'] = data['baseline'].copy()
        x, names, _ = features(**data)
        cols = [i for i, name in enumerate(names) if name.startswith('edit_') or name.endswith('_change')]
        np.testing.assert_array_equal(x[:, cols], 0.)

    def test_outer_evaluation_never_enters_fit_or_calibration(self):
        users = [f'u{i}' for i in range(500)]
        seen = np.zeros(len(users), dtype=int)
        for fold in range(5):
            fit, cal, ev = partitions(users, 20260917, fold)
            self.assertFalse(set(fit) & set(cal) or set(fit) & set(ev) or set(cal) & set(ev))
            self.assertEqual(len(fit) + len(cal) + len(ev), len(users))
            seen[ev] += 1
        np.testing.assert_array_equal(seen, 1)
        np.testing.assert_array_equal(fold_ids(users, 20260917), fold_ids(users[::-1], 20260917)[::-1])

    def test_threshold_rejects_nonpositive_or_tiny_selected_evidence(self):
        cfg = dict(threshold_quantiles=[.5, .9], minimum_calibration_ndcg=.0001,
                   minimum_calibration_changed_users=20)
        pred = np.linspace(-1, 1, 100)
        threshold, _ = select_threshold(pred, pred, np.full(100, -.1), np.zeros(100), cfg)
        self.assertIsNone(threshold)
        y = np.r_[np.full(99, -.01), .5]
        threshold, _ = select_threshold(pred, pred, y, np.zeros(100), cfg)
        self.assertIsNone(threshold)
        self.assertFalse(accept(pred, threshold).any())

    def test_threshold_can_select_outcome_positive_region_using_calibration(self):
        cfg = dict(threshold_quantiles=[.5, .9], minimum_calibration_ndcg=.0001,
                   minimum_calibration_changed_users=20)
        pred = np.linspace(-1, 1, 100)
        y = np.where(pred > 0, .01, -.02)
        threshold, _ = select_threshold(pred, pred, y, np.zeros(100), cfg)
        np.testing.assert_array_equal(accept(pred, threshold), pred > 0)

    def test_missing_target_and_stable_ties(self):
        score = np.zeros((3, 50))
        np.testing.assert_array_equal(target_ranks(score, np.array([0, 12, -1])), [1, 13, 51])
        np.testing.assert_array_equal(item_ranks(score)[0], np.arange(1, 51))


if __name__ == '__main__':
    unittest.main()
