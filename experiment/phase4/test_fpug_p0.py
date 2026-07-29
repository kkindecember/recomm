import unittest

from experiment.phase4.fpug_p0 import (
    bootstrap_relative,
    rank_metrics,
    validation_users_by_hash,
)


class FPUGP0Tests(unittest.TestCase):
    def test_rank_metrics(self):
        self.assertEqual(rank_metrics(["a", "b"], "a"), (1, 1.0, 1.0))
        self.assertEqual(rank_metrics(["a", "b"], "z"), (None, 0.0, 0.0))

    def test_validation_selection_deterministic(self):
        users = {"a", "b", "c", "d"}
        first = validation_users_by_hash(users, "Toys", "salt", 3)
        second = validation_users_by_hash(users, "Toys", "salt", 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_bootstrap_positive(self):
        result = bootstrap_relative([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], 100, 7)
        self.assertGreater(result["point"], 0)


if __name__ == "__main__":
    unittest.main()
