from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment.phase17.protocol import s17_fp12_resource_profile_r3_runtime as r3
from experiment.phase17.protocol.s17_fp12_resource_profile_r4_runtime import (
    ATTEMPT_ID,
    PROFILE_SPECS,
    REVISION_ID,
    experiment_id,
    frozen_config,
    paths,
    snapshot_sources,
)


ROOT = Path(__file__).resolve().parents[3]


class FP12ResourceProfileR4RuntimeTests(unittest.TestCase):
    def test_import_does_not_mutate_attempt_003(self) -> None:
        self.assertEqual(r3.ATTEMPT_ID, "attempt_003")
        self.assertEqual(r3.PROFILE_SPECS["G2_GRAM_LATTE_FULL"].physical_gpu, 3)

    def test_researcher_selected_live_assignment(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_004")
        self.assertEqual(
            {
                arm: spec.physical_gpu
                for arm, spec in PROFILE_SPECS.items()
            },
            {
                "G0_GRAM_B0_FRESH": 5,
                "G1_GRAM_PSID_FULL": 3,
                "G2_GRAM_LATTE_FULL": 1,
                "N0_NATIVE_PSID": 2,
                "N1_NATIVE_LATTE": 6,
            },
        )

    def test_gram_contract_only_changes_execution_memory_strategy(self) -> None:
        config = frozen_config(
            ROOT, "G0_GRAM_B0_FRESH", {"state": "PASS_CPU_PREFLIGHT"}
        )
        self.assertEqual(config["revision_id"], REVISION_ID)
        self.assertEqual(config["primary_beam"], 500)
        self.assertEqual(config["top_k"], 50)
        self.assertFalse(config["generation_kv_cache"])
        self.assertEqual(config["scientific_protocol_changes"], [])
        self.assertEqual(config["train_batch_size"], 2)
        self.assertEqual(config["supersedes_attempt_id"], "attempt_003")

    def test_allocation_and_snapshot_contract(self) -> None:
        allocation = json.loads(
            (ROOT / "experiment/phase17/config/s17_fp_resource_allocation.json")
            .read_text(encoding="utf-8")
        )["arm_specific_resource_profiles_r4"]
        self.assertEqual(allocation["attempt_id"], ATTEMPT_ID)
        self.assertFalse(allocation["gram_generation_kv_cache"])
        for arm, spec in PROFILE_SPECS.items():
            self.assertEqual(allocation["physical_gpu_by_arm"][arm], spec.physical_gpu)
        for source in snapshot_sources(ROOT, PROFILE_SPECS["G0_GRAM_B0_FRESH"]):
            self.assertTrue(source.is_file(), source)

    def test_r4_paths_are_isolated(self) -> None:
        for arm in PROFILE_SPECS:
            self.assertIn("_r4_", experiment_id(arm))
            self.assertEqual(paths(ROOT, arm)["result"].name, ATTEMPT_ID)


if __name__ == "__main__":
    unittest.main()
