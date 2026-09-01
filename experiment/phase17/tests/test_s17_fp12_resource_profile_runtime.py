from __future__ import annotations

import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.protocol.s17_fp12_resource_profile_runtime import (
    GRAM_PYTHON,
    PROFILE_SPECS,
    experiment_id,
    frozen_config,
    paths,
    selected_python,
    snapshot_sources,
    verify_dependencies,
    verify_launch_authorization,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class FP12ResourceProfileRuntimeTests(unittest.TestCase):
    def test_all_five_specs_match_frozen_gpu_and_memory_contract(self) -> None:
        self.assertEqual(set(PROFILE_SPECS), set(ARM_IDS))
        self.assertEqual(PROFILE_SPECS["G0_GRAM_B0_FRESH"].physical_gpu, 1)
        self.assertEqual(PROFILE_SPECS["G1_GRAM_PSID_FULL"].physical_gpu, 0)
        self.assertEqual(PROFILE_SPECS["G2_GRAM_LATTE_FULL"].physical_gpu, 7)
        for arm in ("N0_NATIVE_PSID", "N1_NATIVE_LATTE"):
            self.assertEqual(PROFILE_SPECS[arm].physical_gpu, 4)
            self.assertEqual(PROFILE_SPECS[arm].peak_cap_mib, 16384)
            self.assertEqual(PROFILE_SPECS[arm].minimum_free_mib, 19456)
        for arm in ARM_IDS[2:]:
            self.assertEqual(PROFILE_SPECS[arm].peak_cap_mib, 20480)
            self.assertEqual(PROFILE_SPECS[arm].minimum_free_mib, 23552)

    def test_dependencies_include_completed_tokenizer_and_vocab_amendment(self) -> None:
        dependencies = verify_dependencies(ROOT)
        self.assertIn("tokenizer_manifest_sha256", dependencies)
        self.assertIn("complete_vocabulary_sha256", dependencies)
        self.assertEqual(len(dependencies["complete_vocabulary_sha256"]), 64)

    def test_configs_are_resource_only_and_launch_closed(self) -> None:
        for arm in ARM_IDS:
            config = frozen_config(ROOT, arm, {"state": "PASS_CPU_PREFLIGHT"})
            self.assertTrue(config["resource_only"])
            self.assertTrue(config["effect_metrics_forbidden"])
            self.assertFalse(config["external_target_materialized"])
            self.assertFalse(config["launch_authorized"])
            self.assertFalse(config["automatic_retry"])
            self.assertFalse(config["automatic_process_termination"])
            self.assertEqual(config["primary_beam"], 500)
            self.assertEqual(config["top_k"], 50)

    def test_worker_commands_pin_physical_gpu_and_correct_environment(self) -> None:
        for arm in ARM_IDS:
            spec = PROFILE_SPECS[arm]
            command = worker_command(ROOT, arm)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={spec.physical_gpu}", command)
            self.assertIn("HF_HUB_OFFLINE=1", command)
            self.assertIn(str(selected_python(ROOT, spec)), command)
            self.assertIn(str(paths(ROOT, arm)["snapshot_worker"]), command)
        self.assertEqual(selected_python(ROOT, PROFILE_SPECS["G0_GRAM_B0_FRESH"]), GRAM_PYTHON)

    def test_launch_fails_before_gpu_query_without_explicit_arm_authorization(self) -> None:
        for arm in ARM_IDS:
            with self.assertRaises(PermissionError):
                verify_launch_authorization(ROOT, arm)

    def test_snapshot_sources_exist_and_names_are_neutral(self) -> None:
        for arm in ARM_IDS:
            self.assertNotIn("repeat", experiment_id(arm))
            self.assertNotIn("occupancy", experiment_id(arm))
            for source in snapshot_sources(ROOT, PROFILE_SPECS[arm]):
                self.assertTrue(source.is_file(), source)


if __name__ == "__main__":
    unittest.main()
