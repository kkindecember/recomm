import unittest

from experiment.phase4.cpgv_v0 import rank_from_scores, summarize_rows


class CPGVV0Tests(unittest.TestCase):
    def test_rank_is_descending_and_stable(self):
        self.assertEqual(
            rank_from_scores(["a", "b", "c"], [0.2, 0.5, 0.5]),
            ["b", "c", "a"],
        )

    def test_summary(self):
        rows = [
            {
                "sasrec_rank": 20,
                "exact_rank": 5,
                "pairwise_concordance": 0.8,
            },
            {
                "sasrec_rank": 5,
                "exact_rank": 20,
                "pairwise_concordance": 0.2,
            },
        ]
        result = summarize_rows(rows)
        self.assertEqual(result["sasrec_recall@10"], 0.5)
        self.assertEqual(result["exact_rescore_recall@10"], 0.5)
        self.assertEqual(result["mean_rank_improvement"], 0.0)
        self.assertEqual(result["pairwise_concordance"], 0.5)

    def test_duplicate_candidates_fail(self):
        with self.assertRaises(ValueError):
            rank_from_scores(["a", "a"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
