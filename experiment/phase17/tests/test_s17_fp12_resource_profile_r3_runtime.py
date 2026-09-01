from __future__ import annotations

import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.protocol import s17_fp12_resource_profile_r2_runtime as r2
from experiment.phase17.protocol.s17_fp12_resource_profile_r3_runtime import (
    ATTEMPT_ID,
    PROFILE_SPECS,
    experiment_id,
    frozen_config,
    paths,
)


ROOT = Path(__file__).resolve().parents[3]


class FP12ResourceProfileR3RuntimeTests(unittest.TestCase):
    def test_r3_import_does_not_mutate_attempt_002_module(self) -> None:
        self.assertEqual(r2.ATTEMPT_ID, "attempt_002")
        self.assertEqual(r2.PROFILE_SPECS["G0_GRAM_B0_FRESH"].physical_gpu, 1)

    def test_five_arms_use_five_distinct_cards(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_003")
        self.assertEqual(set(PROFILE_SPECS), set(ARM_IDS))
        self.assertEqual(
            {spec.physical_gpu for spec in PROFILE_SPECS.values()},
            {2, 3, 4, 5, 6},
        )

    def test_r3_contract_keeps_memory_only_scientific_settings(self) -> None:
        for arm_id, spec in PROFILE_SPECS.items():
            self.assertEqual(spec.minimum_free_mib, spec.peak_cap_mib + 4096)
            if spec.family == "gram":
                self.assertEqual(spec.train_batch_size, 2)
                self.assertEqual(128 // spec.train_batch_size, 64)
            else:
                self.assertEqual(spec.train_batch_size, 256)
        config = frozen_config(ROOT, "G0_GRAM_B0_FRESH", {"state": "PASS_CPU_PREFLIGHT"})
        self.assertTrue(config["all_five_profiles_may_run_concurrently"])
        self.assertFalse(config["utilization_hard_gate"])
        self.assertEqual(config["supersedes_attempt_id"], "attempt_002")

    def test_r3_paths_are_isolated(self) -> None:
        for arm_id in ARM_IDS:
            self.assertIn("_r3_", experiment_id(arm_id))
            self.assertEqual(paths(ROOT, arm_id)["result"].name, "attempt_003")


if __name__ == "__main__":
    unittest.main()
