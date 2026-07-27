#!/usr/bin/env python3

import unittest

from hbtr_b0_diag import common_prefix_depth, ndcg, rank_of


class HbtrB0DiagnosticTest(unittest.TestCase):
    def test_common_prefix_depth(self):
        self.assertEqual(common_prefix_depth(("a", "b", "c"), ("a", "b", "d")), 2)
        self.assertEqual(common_prefix_depth(("a",), ("b",)), 0)

    def test_rank_and_ndcg(self):
        self.assertEqual(rank_of(["x", "y"], "y"), 2)
        self.assertIsNone(rank_of(["x", "y"], "z"))
        self.assertEqual(ndcg(None), 0.0)
        self.assertEqual(ndcg(11), 0.0)
        self.assertAlmostEqual(ndcg(1), 1.0)


if __name__ == "__main__":
    unittest.main()
