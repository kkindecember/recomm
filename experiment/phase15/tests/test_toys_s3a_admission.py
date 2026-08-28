from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from specgr_gram_adapter import PathCatalog  # noqa: E402
from toys_s3a_admission import (  # noqa: E402
    admission_verdict,
    align_batch_to_canonical_targets,
    rank_metrics,
    verifier_score_lengths,
)


class TestToysS3AAdmission(unittest.TestCase):
    def test_canonical_targets_remove_split_token_position_drift(self):
        split_collator_batch = {
            "target_ids": torch.tensor(
                [
                    [3, 22188, 3820, 986, 8058, 340, 2934, 18257, 536, 1],
                    [7334, 768, 9319, 3829, 1023, 18312, 12663, 632, 1, -100],
                ],
                dtype=torch.long,
            ),
            "item_text_ids": torch.ones((2, 1, 1), dtype=torch.long),
        }
        encoded_paths = {
            "split-leading-token": (22188, 3820, 986, 8058, 340, 2934, 18257, 536),
            "canonical-token": (7334, 768, 9319, 3829, 1023, 18312, 12663, 632),
        }

        aligned = align_batch_to_canonical_targets(
            split_collator_batch,
            target_items=["split-leading-token", "canonical-token"],
            encoded_paths=encoded_paths,
            eos_token_id=1,
        )

        self.assertEqual(tuple(aligned["target_ids"].shape), (2, 9))
        self.assertEqual(aligned["target_ids"][:, :8].tolist(), list(map(list, encoded_paths.values())))
        self.assertEqual(aligned["target_ids"][:, 8].tolist(), [1, 1])
        self.assertIs(aligned["item_text_ids"], split_collator_batch["item_text_ids"])

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

    def test_admission_verdict_requires_positive_safety_checks(self):
        checks = {
            "complete_path": True,
            "held_ground_truth_not_used_for_training_or_state_selection": True,
            "test_not_opened": True,
        }
        self.assertEqual(
            admission_verdict(False, checks),
            "PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION",
        )
        checks["test_not_opened"] = False
        self.assertEqual(
            admission_verdict(False, checks),
            "FAIL_S15_3A_B2_ITEM_DISJOINT_ADMISSION",
        )


if __name__ == "__main__":
    unittest.main()
