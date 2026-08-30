from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from experiment.phase16.protocol.finalize_s3b_rank_sufficiency_recovery import (
    INCONCLUSIVE_PSD,
    STRUCTURAL_BLOCKED,
    VALID_Z_REQUIRED,
    adjudicate_positions,
)


def diagnostic(width: int, rank: int, *, negative: int = 0) -> dict:
    return {
        "width": width,
        "rank": rank,
        "nullity": width - rank,
        "tolerance": 1e-10,
        "min_eigenvalue": -1e-12 if negative == 0 else -1e-4,
        "max_abs_eigenvalue": 10.0,
        "significant_negative_eigenvalues": negative,
        "tolerance_rule": "max(matrix_shape)*float64_eps*max_abs_eigenvalue",
    }


def position_rows(
    ranks: list[int],
    *,
    width: int = 4,
    covariance_negative_position: int | None = None,
    prefix_negative_position: int | None = None,
) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for position, rank in enumerate(ranks):
        total = 8 + position
        prefix_system_negative = 2 if prefix_negative_position == position else 0
        curve = [
            {
                "request_count": 2,
                "key_gram": diagnostic(width, min(2, width)),
                "system": diagnostic(
                    width, rank, negative=prefix_system_negative
                ),
            },
            {
                "request_count": total,
                "key_gram": diagnostic(width, rank),
                "system": diagnostic(width, rank),
            },
        ]
        rows[str(position)] = {
            "position": position,
            "layer": position % 4,
            "request_count": total,
            "covariance_rows": 20 + position,
            "effective_checkpoints": [2, total],
            "full_covariance_universe_processed": True,
            "full_request_key_universe_processed": True,
            "all_request_key_superset": True,
            "valid_z_filter_applied": False,
            "z_optimization_run": False,
            "weight_delta_solve_run": False,
            "ridge_added": False,
            "pseudoinverse_used": False,
            "jitter_fallback_used": False,
            "outcome_resampling_used": False,
            "covariance": diagnostic(
                width,
                rank,
                negative=2 if covariance_negative_position == position else 0,
            ),
            "rank_curve": curve,
            "final_key_rank": rank,
            "final_system_rank": rank,
            "final_system_nullity": width - rank,
        }
    return rows


def adjudicate(rows: dict[str, dict], *, width: int = 4) -> dict:
    request_counts = {
        position: rows[str(position)]["request_count"] for position in range(6)
    }
    covariance_rows = {
        position: rows[str(position)]["covariance_rows"] for position in range(6)
    }
    return adjudicate_positions(
        rows,
        width=width,
        expected_request_counts=request_counts,
        expected_covariance_rows=covariance_rows,
        configured_checkpoints=[2, "full"],
    )


class RankSufficiencyRecoveryTests(unittest.TestCase):
    def test_one_eligible_deficiency_is_sufficient(self) -> None:
        result = adjudicate(position_rows([4, 4, 3, 4, 4, 4]))
        self.assertEqual(result["classification"], STRUCTURAL_BLOCKED)
        self.assertEqual(result["structurally_blocked_positions"], [2])
        self.assertEqual(result["proof_eligible_positions"], list(range(6)))
        self.assertFalse(result["faithful_gate_promoted"])

    def test_ineligible_covariance_is_excluded_not_relabelled_psd(self) -> None:
        result = adjudicate(
            position_rows([3, 4, 4, 4, 4, 3], covariance_negative_position=5)
        )
        self.assertEqual(result["classification"], STRUCTURAL_BLOCKED)
        self.assertEqual(result["structurally_blocked_positions"], [0])
        self.assertEqual(result["proof_ineligible_positions"], [5])
        p5 = result["position_adjudications"]["5"]
        self.assertFalse(p5["proof_eligible"])
        self.assertIn(
            "covariance_significant_negative_eigenvalues",
            p5["proof_ineligibility_reasons"],
        )

    def test_prefix_system_negative_is_diagnostic_only(self) -> None:
        result = adjudicate(
            position_rows([3, 4, 4, 4, 4, 4], prefix_negative_position=0)
        )
        self.assertEqual(result["classification"], STRUCTURAL_BLOCKED)
        p0 = result["position_adjudications"]["0"]
        self.assertTrue(p0["proof_eligible"])
        self.assertEqual(
            p0["intermediate_prefix_system_negatives_diagnostic_only"],
            [{"request_count": 2, "significant_negative_eigenvalues": 2}],
        )

    def test_ineligible_without_eligible_deficiency_is_inconclusive(self) -> None:
        result = adjudicate(
            position_rows([4, 4, 4, 4, 4, 3], covariance_negative_position=5)
        )
        self.assertEqual(result["classification"], INCONCLUSIVE_PSD)
        self.assertEqual(result["structurally_blocked_positions"], [])

        resolved = adjudicate(position_rows([4, 4, 4, 4, 4, 4]))
        self.assertEqual(resolved["classification"], VALID_Z_REQUIRED)

    def test_config_freezes_b1_failed_artifacts_and_cpu_only_scope(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config = json.loads(
            (
                root
                / "experiment/phase16/configs/stage16_s3b_rank_sufficiency_recovery_c1_cpu.json"
            ).read_text()
        )
        self.assertTrue(config["resources"]["cpu_only"])
        self.assertEqual(config["resources"]["gpu_count"], 0)
        self.assertEqual(
            config["source_b1_contract"]["expected_failed_raw_checks"],
            ["positive_semidefinite_evidence"],
        )
        self.assertFalse(config["adjudication"]["recompute_covariance"])
        self.assertFalse(config["adjudication"]["recompute_keys"])
        self.assertFalse(config["adjudication"]["recompute_eigenvalues"])
        self.assertFalse(config["adjudication"]["recompute_ranks"])
        for spec in config["frozen_inputs"].values():
            observed = hashlib.sha256((root / spec["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, spec["sha256"])

    def test_runner_forbids_gpu_and_existing_attempt_overwrite(self) -> None:
        root = Path(__file__).resolve().parents[3]
        runner = (
            root
            / "experiment/phase16/run_stage16_s3b_rank_sufficiency_recovery_c1_cpu.sh"
        ).read_text()
        self.assertIn('env CUDA_VISIBLE_DEVICES=""', runner)
        self.assertIn('if [[ -e "$OUTPUT" ]]', runner)
        self.assertNotIn("nvidia-smi", runner)
        self.assertNotIn("tmux", runner)


if __name__ == "__main__":
    unittest.main()
