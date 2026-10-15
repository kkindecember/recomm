import unittest
import numpy as np

from experiment.phase21.expert_mechanism_core import (
    transform_scores, shuffle_older_history, flat_neighbor_scores, ScoreOverride)


class MechanismControls(unittest.TestCase):
    def test_permutation_preserves_distribution_and_other_experts(self):
        raw = np.random.RandomState(2).rand(4, 3, 19).astype('float32')
        before = raw.copy(); donors = np.roll(np.arange(19), 3)
        for arm, channel in [('sasrec_misassigned', 0), ('itemcf_misassigned', 1)]:
            result = transform_scores(raw, arm, donors)
            np.testing.assert_array_equal(np.sort(result[:, channel]), np.sort(raw[:, channel]))
            self.assertFalse(np.array_equal(result[:, channel], raw[:, channel]))
            for other in set(range(3)) - {channel}:
                np.testing.assert_array_equal(result[:, other], raw[:, other])
        np.testing.assert_array_equal(raw, before)

    def test_order_preserves_last_event_and_visible_multiset(self):
        for h in [[], [2], [2, 4], [1, 2, 3, 4], [4]*12, [1, 1, 2, 2, 8], list(range(100))]:
            got = shuffle_older_history(h, 'u', 'Toys')
            self.assertEqual(sorted(got), sorted(h[-50:]))
            if h: self.assertEqual(got[-1], h[-1])
            self.assertEqual(got, shuffle_older_history(h, 'u', 'Toys'))
            if len(set(h[-50:-1])) > 1: self.assertNotEqual(got[:-1], h[-50:-1])

    def test_flat_strength_retains_mass_and_support(self):
        for a in [np.array([.1, .3, .6]), np.array([.2]), np.array([.5, .5])]:
            flat = flat_neighbor_scores(a)
            np.testing.assert_allclose(flat.sum(), a.sum(), rtol=2e-7)
            self.assertTrue((flat > 0).all())
            self.assertTrue((flat == flat[0]).all())

    def test_override_uses_source_indices_not_subset_positions(self):
        raw = np.zeros((9, 3, 5), dtype=np.float32)
        replacement = np.array([[1]*5, [4]*5, [8]*5], dtype=np.float32)
        bank = ScoreOverride(raw, np.array([7, 1, 4]), replacement, 0)
        got = bank[np.array([4, 7])]
        np.testing.assert_array_equal(got[:, 0], replacement[[2, 0]])
        np.testing.assert_array_equal(got[:, 1:], np.zeros((2, 2, 5)))
        with self.assertRaises(AssertionError): bank[np.array([0])]
        self.assertFalse(raw.any())


if __name__ == '__main__': unittest.main()
