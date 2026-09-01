from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as v1
from experiment.phase17.protocol.s17_fp12_resource_profile_r2_runtime import (
    ATTEMPT_ID,
    PROFILE_SPECS,
    SAFETY_MARGIN_MIB,
    experiment_id,
    frozen_config,
    paths,
    two_snapshot_admission,
)


ROOT = Path(__file__).resolve().parents[3]


class FP12ResourceProfileR2RuntimeTests(unittest.TestCase):
    def test_r2_import_does_not_mutate_attempt_001_module(self) -> None:
        self.assertEqual(v1.ATTEMPT_ID, "attempt_001")
        self.assertEqual(v1.PROFILE_SPECS["G0_GRAM_B0_FRESH"].train_batch_size, 16)

    def test_r2_contract_is_memory_only_and_preserves_effective_batch(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_002")
        self.assertEqual(set(PROFILE_SPECS), set(ARM_IDS))
        for arm_id, spec in PROFILE_SPECS.items():
            self.assertEqual(spec.minimum_free_mib, spec.peak_cap_mib + SAFETY_MARGIN_MIB)
            if spec.family == "gram":
                self.assertEqual(spec.train_batch_size, 2)
                self.assertEqual(128 // spec.train_batch_size, 64)
            else:
                self.assertEqual(spec.train_batch_size, 256)

    def test_r2_names_and_paths_do_not_collide_with_attempt_001(self) -> None:
        for arm_id in ARM_IDS:
            self.assertIn("_r2_", experiment_id(arm_id))
            self.assertEqual(paths(ROOT, arm_id)["result"].name, "attempt_002")
            self.assertIn("attempt_002", str(paths(ROOT, arm_id)["snapshot"]))

    def test_r2_config_records_utilization_without_gating(self) -> None:
        config = frozen_config(ROOT, "G0_GRAM_B0_FRESH", {"state": "PASS_CPU_PREFLIGHT"})
        self.assertEqual(config["train_batch_size"], 2)
        self.assertEqual(config["safety_margin_mib"], 4096)
        self.assertFalse(config["utilization_hard_gate"])
        self.assertTrue(config["utilization_recorded_only"])
        self.assertTrue(config["preserve_all_preexisting_processes"])

    def test_high_utilization_is_not_an_admission_failure(self) -> None:
        spec = PROFILE_SPECS["G0_GRAM_B0_FRESH"]
        snapshot = {
            "selected": {"free_mib": spec.minimum_free_mib, "utilization_percent": 100},
            "selected_compute_processes": [{"pid": 123, "used_memory_mib": 100}],
        }
        with mock.patch(
            "experiment.phase17.protocol.s17_fp12_resource_profile_r2_runtime.gpu_snapshot_once",
            side_effect=[snapshot, snapshot],
        ), mock.patch(
            "experiment.phase17.protocol.s17_fp12_resource_profile_r2_runtime.v1.time.sleep"
        ):
            admission = two_snapshot_admission(spec, {})
        self.assertTrue(admission["utilization_recorded_only"])
        self.assertEqual(admission["observed_preexisting_compute_pids"], [123])


if __name__ == "__main__":
    unittest.main()
