import unittest

from experiment.phase21.full_validation_core import build_samples


class FullValidationBoundaryTests(unittest.TestCase):
    def test_all_validation_users_with_observed_training_event(self):
        upstream = {'fit': [1, 2], 'new': [3, 4], 'old': [5, 6]}
        held = [dict(user_id='fit', history=[1, 2], target=7, role='gate_calibration')]
        validation = [dict(user_id='fit', history=[1, 2, 7], target=8),
                      dict(user_id='new', history=[3, 4], target=9),
                      dict(user_id='old', history=[5, 6], target=10)]
        prior = [dict(validation[2], role='development', full_history=[5, 6])]
        result = build_samples(validation, upstream, held, prior)
        self.assertEqual([r['user_id'] for r in result], ['fit', 'new', 'old'])
        self.assertEqual(result[0]['full_history'], [1, 2, 7])
        self.assertNotIn(8, result[0]['full_history'])
        self.assertEqual(sum(r['previously_evaluated'] for r in result), 1)
        changed = [dict(r, target=100 + i) for i, r in enumerate(validation)]
        other = build_samples(changed, upstream, held, [])
        self.assertEqual([r['full_history'] for r in result], [r['full_history'] for r in other])

    def test_reject_missing_users_and_wrong_history(self):
        with self.assertRaises(AssertionError):
            build_samples([dict(user_id='a', history=[1], target=2)], {'a': [1], 'b': [3]}, [], [])
        with self.assertRaises(AssertionError):
            build_samples([dict(user_id='a', history=[2], target=3)], {'a': [1]}, [], [])

    def test_repeated_target_does_not_remove_user(self):
        result = build_samples([dict(user_id='a', history=[1, 2], target=1)], {'a': [1, 2]}, [], [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['full_history'], [1, 2])


if __name__ == '__main__':
    unittest.main()
