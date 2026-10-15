"""Meaningful invariance and reproduction checks for the information boundary."""
import unittest
import numpy as np
import torch

from experiment.phase20.lime_clean_benefit_core import fit_head, predict_head
from experiment.phase20.lime_early_correction_core import GROUPS, fit, correction, add_prior


class EarlyCorrectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.RandomState(7)
        cls.d = dict(ids=np.tile(np.arange(1, 51), (12, 1)), targets=np.arange(1, 13),
                     valid=np.ones((12, 50), dtype=bool), x=rng.randn(12, 50, 10).astype('float32'),
                     mixed=rng.randn(12, 50) * .05)
        cls.d['valid'][:, -1] = False
        cls.ix = np.arange(8)
        cls.saved = fit(cls.d, cls.ix, 'early', 0)

    def test_full_matches_original_training_exactly(self):
        new = fit(self.d, self.ix, 'full', 1)
        old = fit_head(self.d, self.ix, 'mlp', 1)
        for k in new['state_dict']:
            self.assertTrue(torch.equal(new['state_dict'][k], old['state_dict'][k]))
        result = add_prior(self.d['mixed'], correction(new, self.d['x']))
        np.testing.assert_array_equal(result, predict_head(old, self.d['x'], self.d['mixed']))

    def test_training_ignores_unavailable_features_and_held_labels(self):
        changed = {k: v.copy() for k, v in self.d.items()}
        changed['x'][..., list(GROUPS['late'])] = np.nan
        changed['x'][8:] = np.nan
        changed['targets'][8:] = 9999
        changed['mixed'][8:] = np.nan
        saved = fit(changed, self.ix, 'early', 0)
        for k in saved['state_dict']:
            self.assertTrue(torch.equal(saved['state_dict'][k], self.saved['state_dict'][k]))

    def test_correction_pointwise_without_candidate_rank_or_prior(self):
        x = self.d['x'][..., list(GROUPS['early'])]
        y = correction(self.saved, x)
        perm = np.random.RandomState(1).permutation(50)
        np.testing.assert_allclose(correction(self.saved, x[:, perm]), y[:, perm], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(correction(self.saved, x[:, :1]), y[:, :1], atol=1e-6, rtol=1e-6)
        # GEMM batch shape changes float32 rounding, not feature availability.
        np.testing.assert_allclose(correction(self.saved, x, chunk=1), y, atol=1e-6, rtol=1e-6)
        with self.assertRaises(AssertionError):
            correction(self.saved, self.d['x'])

    def test_missing_target_retained_outside_fit_loss(self):
        changed = {k: v.copy() for k, v in self.d.items()}
        changed['targets'][0] = 9999
        changed['targets'][1] = 50  # History-masked candidate is also ineligible.
        saved = fit(changed, self.ix, 'early', 0)
        np.testing.assert_array_equal(saved['used'], np.arange(2, 8))
        x = self.d['x'][..., list(GROUPS['early'])]
        self.assertEqual(correction(saved, x).shape, (12, 50))


if __name__ == '__main__':
    unittest.main()
