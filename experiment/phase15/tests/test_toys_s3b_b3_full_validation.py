from __future__ import annotations

import argparse
import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from toys_s3b_b3_full_validation import (  # noqa: E402
    EXPECTED_ADMISSION_VERDICT,
    validate_frozen_contract,
)


def frozen_args() -> argparse.Namespace:
    return argparse.Namespace(
        train_transitions=4096,
        covariance_transitions=256,
        covariance_long_path_minimum=32,
        covariance_batch_size=32,
        contexts_per_pseudo_cold=10,
        requests_per_position=4,
        z_steps=30,
        beam_size=50,
        bootstrap_resamples=10_000,
        bootstrap_seed=20260822,
        seed=1502,
    )


def passing_admission() -> dict:
    checks = {
        "all_rankings_unique_known_top50": True,
        "base_hash_unchanged": True,
        "held_ground_truth_not_used_for_training_or_state_selection": True,
        "test_not_opened": True,
        "b3_complete_one_one_edited_beam_path": True,
        "b3_delta_finite": True,
        "b3_delta_nonzero": True,
        "b3_every_position_exercised": True,
    }
    return {
        "verdict": EXPECTED_ADMISSION_VERDICT,
        "admission_checks": checks,
        "seed": 1502,
        "beam_size": 50,
        "b3_covariance_transitions": 256,
        "b3_requests_per_position": 4,
    }


class TestToysS3BB3FullValidation(unittest.TestCase):
    def test_frozen_contract_accepts_admitted_b3(self):
        validate_frozen_contract(frozen_args(), passing_admission())

    def test_frozen_contract_rejects_drift_or_missing_gate(self):
        args = frozen_args()
        args.requests_per_position = 5
        with self.assertRaisesRegex(ValueError, "contract drift"):
            validate_frozen_contract(args, passing_admission())
        admission = passing_admission()
        admission["admission_checks"]["b3_delta_nonzero"] = False
        with self.assertRaisesRegex(ValueError, "checks are incomplete"):
            validate_frozen_contract(frozen_args(), admission)


if __name__ == "__main__":
    unittest.main()
