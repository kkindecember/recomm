from __future__ import annotations

import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from specgr_gram_adapter import PathCatalog  # noqa: E402
from toys_s3a_admission import rank_metrics, verifier_score_lengths  # noqa: E402


class TestToysS3AAdmission(unittest.TestCase):
    def test_verifier_lengths_use_full_warm_and_shared_cold_prefix(self):
        catalog = PathCatalog.build(
            {
                "w1": "a|b|c",
                "w2": "a|d|e",
                "c1": "a|b|x|y",
                "c2": "q|r|s",
            },
            {"w1", "w2"},
            {"c1", "c2"},
        )
        self.assertEqual(
            verifier_score_lengths(catalog),
            {"w1": 3, "w2": 3, "c1": 2, "c2": 2},
        )

    def test_rank_metrics_do_not_collapse_missing_targets(self):
        self.assertEqual(rank_metrics(["a", "b"], "b"), {"rank": 2, "hit50": 1, "mrr": 0.5})
        self.assertEqual(rank_metrics(["a", "b"], "c"), {"rank": None, "hit50": 0, "mrr": 0.0})


if __name__ == "__main__":
    unittest.main()
