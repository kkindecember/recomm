import copy
import unittest
import numpy as np
from experiment.phase20.lime_partial_state_core import prefix_features


class PrefixAvailabilityTest(unittest.TestCase):
    def setUp(self):
        self.q = {(0,): {2: np.log(.7), 3: np.log(.3)},
                  (0, 2): {4: np.log(.4), 5: np.log(.6)},
                  (0, 3): {6: np.log(.8), 7: np.log(.2)}}
        self.steps = [
            dict(depth=1, prefixes=[[0], [0]], parent_scores=[0., -1e9],
                 rows=[dict(tokens=[2, 3], native=[-1., -2.], legal_log_mass=float(np.logaddexp(-1., -2.)))] * 2),
            dict(depth=2, prefixes=[[0, 2], [0, 3]], parent_scores=[-1., -2.],
                 rows=[dict(tokens=[4, 5], native=[-1., -2.], legal_log_mass=float(np.logaddexp(-1., -2.))),
                       dict(tokens=[6, 7], native=[-.2, -1.], legal_log_mass=float(np.logaddexp(-.2, -1.)))])]

    def test_no_future_read(self):
        a = prefix_features(self.steps, self.q, 2)
        b = prefix_features(self.steps + [None, dict(target=999, final_score=1e9)], self.q, 2)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_all_competing_parents_used(self):
        f = prefix_features(self.steps, self.q, 2, beam_size=2)
        self.assertEqual(len(f), 4)
        self.assertEqual(f[(0, 2, 4)][2], 1.)
        self.assertEqual(f[(0, 3, 6)][2], .5)
        # Rank of the weaker sibling includes competition from the other parent.
        self.assertAlmostEqual(f[(0, 2, 5)][2], 1/3, places=6)

    def test_parent_order_and_duplicate_filler_do_not_change_features(self):
        before = prefix_features(self.steps, self.q, 2)
        changed = copy.deepcopy(self.steps)
        for key in ('prefixes', 'parent_scores', 'rows'):
            changed[1][key].reverse()
        after = prefix_features(changed, self.q, 2)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


if __name__ == '__main__':
    unittest.main()
