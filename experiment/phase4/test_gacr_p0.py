import unittest

from experiment.phase4.gacr_p0 import (
    relative_gain,
    select_fresh_validation_users,
    split_training_samples,
    split_training_users,
)


class GACRP0Tests(unittest.TestCase):
    def test_fresh_validation_excludes_prior_users(self):
        selected = select_fresh_validation_users(
            {"a", "b", "c", "d"}, {"a", "b"}, "Toys", "salt", 2
        )
        self.assertEqual(set(selected), {"c", "d"})

    def test_split_is_exact_and_disjoint(self):
        rows = [
            {"sample_key": f"s{i}", "positive_item": "h" if i < 4 else "t"}
            for i in range(8)
        ]
        fit, calibration = split_training_samples(
            rows, {"h"}, 1, "Toys", 3, 1
        )
        self.assertEqual(len(fit), 6)
        self.assertEqual(len(calibration), 2)
        self.assertFalse(
            {row["sample_key"] for row in fit}
            & {row["sample_key"] for row in calibration}
        )

    def test_relative_gain(self):
        self.assertAlmostEqual(relative_gain(2.0, 2.2), 0.1)

    def test_training_user_split_is_disjoint(self):
        fit, calibration = split_training_users(
            {f"u{i}" for i in range(10)}, 1, "Toys"
        )
        self.assertEqual(len(fit), 8)
        self.assertEqual(len(calibration), 2)
        self.assertFalse(fit & calibration)


if __name__ == "__main__":
    unittest.main()
