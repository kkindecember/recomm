import unittest

from experiment.phase4.prpd_r0 import (
    popularity_midrank_percentile,
    residual_fuse,
)
from experiment.phase4.rpcd_t0 import fuse


class PRPDR0Tests(unittest.TestCase):
    def test_gamma_zero_exactly_matches_rpcd(self):
        gram = ["a", "b", "c"]
        sasrec = ["d", "b", "e"]
        popularity = {"a": 0.8, "b": 0.9, "c": 0.1, "d": 1.0, "e": 0.0}
        for weight in (0.0, 0.2, 0.5, 1.0):
            self.assertEqual(
                residual_fuse(gram, sasrec, popularity, 0.0, weight),
                fuse(gram, sasrec, weight),
            )

    def test_popularity_midrank_ties(self):
        sequences = {
            "u1": ["a", "a", "x", "y"],
            "u2": ["b", "c", "x", "y"],
        }
        values = popularity_midrank_percentile(sequences, ["a", "b", "c", "d"])
        self.assertGreater(values["a"], values["b"])
        self.assertEqual(values["b"], values["c"])
        self.assertGreater(values["b"], values["d"])

    def test_positive_gamma_demotes_popular_sasrec_item(self):
        gram = ["g"]
        sasrec = ["popular", "tail"]
        popularity = {"popular": 1.0, "tail": 0.0}
        ranking = residual_fuse(gram, sasrec, popularity, 1.0, 1.0)
        self.assertLess(ranking.index("tail"), ranking.index("popular"))


if __name__ == "__main__":
    unittest.main()
