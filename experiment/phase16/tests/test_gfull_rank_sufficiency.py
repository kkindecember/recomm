from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

from experiment.phase16.protocol.genrecedit_faithful import FullTargetRequest
from experiment.phase16.protocol.genrecedit_rank_sufficiency import (
    RANK_TOLERANCE_RULE,
    STRUCTURAL_BLOCKED,
    StreamingKeyGram,
    VALID_Z_REQUIRED,
    classify_all_request_upper_bound,
    deterministic_request_order,
    effective_checkpoints,
    ordered_request_sha256,
    symmetric_rank_diagnostics,
)
from experiment.phase16.protocol.gfull_rank_sufficiency_diagnostic import (
    EXECUTED_CODE_PATHS,
)


def request(index: int, *, position: int = 0) -> FullTargetRequest:
    return FullTargetRequest(
        cold_item=f"cold-{index}",
        source_warm_item=f"warm-{index}",
        context_items=(f"context-{index}",),
        full_target_path=(2, 3, 4, 5, 6, 7),
        prefix_token_ids=(2, 3, 4, 5, 6)[:position],
        target_token_id=(2, 3, 4, 5, 6, 7)[position],
        legal_token_ids=((2, 8) if position == 0 else ((2, 3, 4, 5, 6, 7)[position], 8)),
        position=position,
    )


def position_rows(ranks: list[int], width: int = 4) -> dict[str, dict]:
    return {
        str(position): {
            "full_covariance_universe_processed": True,
            "full_request_key_universe_processed": True,
            "all_request_key_superset": True,
            "final_system_rank": ranks[position],
        }
        for position in range(6)
    }


class RankSufficiencyTests(unittest.TestCase):
    def test_request_order_is_seeded_stable_and_duplicate_closed(self) -> None:
        rows = [request(index) for index in range(8)]
        first = deterministic_request_order(rows, seed=1502)
        second = deterministic_request_order(tuple(reversed(rows)), seed=1502)
        self.assertEqual(first, second)
        self.assertEqual(ordered_request_sha256(first), ordered_request_sha256(second))
        self.assertNotEqual(
            first, deterministic_request_order(rows, seed=1503)
        )
        with self.assertRaises(ValueError):
            deterministic_request_order([rows[0], rows[0]], seed=1502)

    def test_effective_checkpoints_are_capped_and_end_at_full(self) -> None:
        configured = [16, 64, 256, 1024, 4096, 16384, "full"]
        self.assertEqual(
            effective_checkpoints(configured, total=4250),
            (16, 64, 256, 1024, 4096, 4250),
        )
        self.assertEqual(
            effective_checkpoints(configured, total=59630)[-1], 59630
        )
        with self.assertRaises(ValueError):
            effective_checkpoints([0, "full"], total=10)

    def test_frozen_symmetric_rank_rule_and_streaming_key_gram(self) -> None:
        singular = torch.diag(torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64))
        diagnostic = symmetric_rank_diagnostics(singular)
        self.assertEqual(diagnostic.rank, 2)
        self.assertEqual(diagnostic.nullity, 1)
        self.assertEqual(diagnostic.tolerance_rule, RANK_TOLERANCE_RULE)

        accumulator = StreamingKeyGram(3, device="cpu")
        accumulator.update(torch.tensor([[0.0, 0.0, 1.0]]))
        combined = singular + accumulator.gram
        self.assertEqual(symmetric_rank_diagnostics(combined).rank, 3)
        self.assertEqual(accumulator.count, 1)

    def test_all_request_upper_bound_classification_is_one_way(self) -> None:
        blocked = classify_all_request_upper_bound(
            position_rows([4, 4, 3, 4, 4, 4]), width=4
        )
        self.assertEqual(blocked["classification"], STRUCTURAL_BLOCKED)
        self.assertEqual(blocked["structurally_blocked_positions"], [2])
        self.assertFalse(blocked["faithful_gate_promoted"])

        unresolved = classify_all_request_upper_bound(
            position_rows([4, 4, 4, 4, 4, 4]), width=4
        )
        self.assertEqual(unresolved["classification"], VALID_Z_REQUIRED)
        self.assertFalse(unresolved["faithful_gate_promoted"])

    def test_psd_all_key_superset_can_only_reduce_nullspace(self) -> None:
        covariance = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        valid_keys = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        all_keys = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        valid_system = covariance + valid_keys.T @ valid_keys
        all_system = covariance + all_keys.T @ all_keys
        self.assertEqual(symmetric_rank_diagnostics(valid_system).rank, 2)
        self.assertEqual(symmetric_rank_diagnostics(all_system).rank, 3)

    def test_b1_config_freezes_full_train_only_upper_bound_and_gpu4(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config_path = (
            root
            / "experiment/phase16/configs/stage16_s3b_gfull_rank_sufficiency_b1_gpu4.json"
        )
        config = json.loads(config_path.read_text())
        diagnostic = config["diagnostic"]
        self.assertEqual(
            sum(diagnostic["full_request_counts_by_position"].values()), 302400
        )
        self.assertEqual(
            diagnostic["full_covariance_rows_by_position"],
            {"0": 27659, "1": 27659, "2": 27659, "3": 27659, "4": 27659, "5": 2036},
        )
        for forbidden in (
            "valid_z_filter_applied",
            "z_optimization_run",
            "weight_delta_solve_run",
            "ridge_added",
            "pseudoinverse_used",
            "jitter_fallback_used",
            "outcome_resampling_used",
            "scientific_efficacy_metric",
            "faithful_gate_promotion_allowed",
        ):
            self.assertFalse(diagnostic[forbidden])
        resources = config["resources"]
        self.assertEqual(resources["fixed_physical_gpu"], 4)
        self.assertEqual(resources["minimum_free_mib"], 18432)
        self.assertEqual(resources["expected_peak_mib"], 12288)
        self.assertEqual(resources["hard_timeout_seconds"], 10800)
        self.assertFalse(config["validation_used"])
        self.assertFalse(config["test_read"])

    def test_parent_artifact_hashes_and_execution_paths_are_frozen(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config = json.loads(
            (
                root
                / "experiment/phase16/configs/stage16_s3b_gfull_rank_sufficiency_b1_gpu4.json"
            ).read_text()
        )
        for label in (
            "parent_a4_raw",
            "parent_a4_status",
            "parent_request_manifest",
            "parent_request_checkpoint",
        ):
            spec = config["inputs"][label]
            observed = hashlib.sha256((root / spec["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, spec["sha256"])
        for relative in EXECUTED_CODE_PATHS:
            self.assertTrue((root / relative).is_file(), relative)
        self.assertIn(
            "experiment/phase16/protocol/finalize_s3b_rank_sufficiency.py",
            EXECUTED_CODE_PATHS,
        )

    def test_gpu4_launcher_is_background_and_has_no_auto_switch(self) -> None:
        root = Path(__file__).resolve().parents[3]
        launcher = (
            root
            / "experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency_b1_gpu4.sh"
        ).read_text()
        inner = (
            root
            / "experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency_b1_gpu4_inner.sh"
        ).read_text()
        generic = (
            root / "experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency.sh"
        ).read_text()
        self.assertIn("tmux new-session -d", launcher)
        self.assertIn("S16_S3B_FIXED_GPU=4", inner)
        self.assertIn("S16_S3B_HARD_TIMEOUT=10800", inner)
        self.assertIn('[[ "$index" == "$FIXED_GPU" ]]', generic)
        self.assertIn("no automatic retry/resume", generic)
        self.assertIn("PASS_S16_3B_RANK_DIAGNOSTIC_COMPLETE", generic)


if __name__ == "__main__":
    unittest.main()
