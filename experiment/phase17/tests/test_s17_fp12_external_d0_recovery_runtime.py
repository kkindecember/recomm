from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.protocol.s17_fp12_external_d0_recovery_runtime import (
    ATTEMPT_ID,
    RECOVERY_ARM_IDS,
    RECOVERY_GPU_BY_ARM,
    arm_paths,
    paths,
    source_paths,
    verify_attempt_001_evidence,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class ExternalD0RecoveryRuntimeTests(unittest.TestCase):
    def test_scope_is_only_failed_or_unlaunched_gram_arms(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_002")
        self.assertEqual(
            set(RECOVERY_ARM_IDS),
            {"G0_GRAM_B0_FRESH", "G1_GRAM_PSID_FULL", "G2_GRAM_LATTE_FULL"},
        )
        self.assertEqual(RECOVERY_GPU_BY_ARM["G0_GRAM_B0_FRESH"], 5)
        self.assertEqual(RECOVERY_GPU_BY_ARM["G1_GRAM_PSID_FULL"], 5)
        self.assertEqual(RECOVERY_GPU_BY_ARM["G2_GRAM_LATTE_FULL"], 6)

    def test_recovery_paths_are_disjoint_and_bundle_is_reused(self) -> None:
        resolved = paths(ROOT)
        self.assertIn("recovery/attempt_002", str(resolved["result"]))
        self.assertIn("external_d0/attempt_001", str(resolved["original_bundle"]))
        for arm_id in RECOVERY_ARM_IDS:
            self.assertTrue(str(arm_paths(ROOT, arm_id)["result"]).startswith(str(resolved["result"])))

    def test_evidence_verification_never_calls_raw_target_materializer(self) -> None:
        with patch(
            "experiment.phase17.protocol.s17_fp12_external_d0_runtime."
            "materialize_external_evaluation_view",
            side_effect=AssertionError("controlled recovery must not reopen raw D0"),
        ):
            evidence = verify_attempt_001_evidence(ROOT)
        self.assertEqual(evidence["single_materialization_count"], 1)
        self.assertFalse(evidence["raw_external_projection_reopened"])
        self.assertEqual(
            set(evidence["failed_before_predictions"]),
            {"G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL"},
        )

    def test_worker_commands_are_snapshot_bound_and_gpu_isolated(self) -> None:
        for arm_id in RECOVERY_ARM_IDS:
            gpu = RECOVERY_GPU_BY_ARM[arm_id]
            command = worker_command(ROOT, arm_id, gpu)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={gpu}", command)
            self.assertIn(str(arm_paths(ROOT, arm_id)["snapshot_worker"]), command)
            self.assertIn(str(paths(ROOT)["snapshot"]), command)

    def test_snapshot_sources_exist(self) -> None:
        for source in source_paths(ROOT):
            self.assertTrue(source.is_file(), source)


if __name__ == "__main__":
    unittest.main()
