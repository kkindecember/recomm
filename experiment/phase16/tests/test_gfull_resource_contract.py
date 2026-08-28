from __future__ import annotations

import unittest

import torch

from experiment.phase16.protocol.finalize_s3_gfull_resource_sweep import (
    REQUIRED_CANDIDATE_SEMANTIC_KEYS,
    expected_selected_microbatch,
    observed_full_30_step_path,
)
from experiment.phase16.protocol.gfull_objective_resource_sweep import (
    choose_candidate,
    covariance_convergence_diagnostics,
)


def candidate(microbatch: int, throughput: float, peak: float = 4000.0) -> dict:
    semantics = {key: True for key in REQUIRED_CANDIDATE_SEMANTIC_KEYS}
    return {
        "microbatch": microbatch,
        "steady_request_steps_per_second": throughput,
        "peak_reserved_mib": peak,
        "eligible": peak <= 8192,
        "semantic_checks": semantics,
    }


class ResourceContractTests(unittest.TestCase):
    def test_selection_recomputed_with_two_percent_smaller_batch_tie_break(self) -> None:
        rows = [candidate(4, 98.5), candidate(8, 100.0), candidate(16, 99.0)]
        self.assertEqual(choose_candidate(rows)["microbatch"], 4)
        self.assertEqual(expected_selected_microbatch(rows, 8192), 4)

    def test_selection_rejects_self_reported_eligibility_or_missing_semantics(self) -> None:
        row = candidate(4, 10.0)
        row["eligible"] = False
        self.assertIsNone(expected_selected_microbatch([row], 8192))
        row = candidate(4, 10.0)
        row["semantic_checks"].pop("full_30_step_path_observed")
        self.assertIsNone(expected_selected_microbatch([row], 8192))

    def test_step_29_is_observed_not_inferred_from_config(self) -> None:
        trace = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        row = candidate(4, 10.0)
        row["batch_records"] = [
            {
                "lifecycle_check_steps": trace,
                "observed_step_29": True,
                "forward_calls": 30,
            }
        ]
        self.assertTrue(observed_full_30_step_path([row]))
        row["batch_records"][0]["lifecycle_check_steps"] = [10, 20]
        row["batch_records"][0]["forward_calls"] = 21
        self.assertFalse(observed_full_30_step_path([row]))

    def test_covariance_convergence_ends_at_zero_reference_drift(self) -> None:
        activations = {
            0: torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [1.0, 1.0]]),
            1: torch.tensor([[1.0, 1.0], [2.0, 0.0]]),
        }
        result = covariance_convergence_diagnostics(
            activations, {0: [2, 4], 1: [1, 2]}
        )
        self.assertGreater(
            result["0"][0][
                "relative_frobenius_drift_to_largest_resource_checkpoint"
            ],
            0.0,
        )
        self.assertEqual(
            result["0"][-1][
                "relative_frobenius_drift_to_largest_resource_checkpoint"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
