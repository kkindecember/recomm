from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_formal_executor import _ndcg_at_10
from experiment.phase17.protocol.s17_fp12_formal_runtime import (
    ATTEMPT_ID,
    FORMAL_SPECS,
    GRAM_ACCUMULATION_BY_ARM,
    GRAM_MICROBATCH_BY_ARM,
    PROFILE_EVIDENCE,
    RESEARCHER_DIRECTION,
    frozen_config,
    paths,
    snapshot_sources,
    verify_profile_evidence,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class FP12FormalRuntimeTests(unittest.TestCase):
    def test_all_five_profiles_are_pass_dependencies(self) -> None:
        self.assertEqual(set(FORMAL_SPECS), set(PROFILE_EVIDENCE))
        for arm, spec in FORMAL_SPECS.items():
            evidence = verify_profile_evidence(ROOT, arm)
            self.assertEqual(
                evidence["profile_peak_reserved_mib"],
                spec.profile_peak_reserved_mib,
            )
            self.assertEqual(evidence["formal_minimum_free_mib"], spec.minimum_free_mib)

    def test_live_assignment_and_researcher_authorization_match(self) -> None:
        allocation = json.loads(
            (ROOT / "experiment/phase17/config/s17_fp_resource_allocation.json")
            .read_text(encoding="utf-8")
        )["fp1_fp2_formal_attempt_004"]
        self.assertEqual(ATTEMPT_ID, "attempt_004")
        self.assertEqual(allocation["researcher_direction"], RESEARCHER_DIRECTION)
        self.assertFalse(allocation["external_d0_evaluation_authorized"])
        for arm, spec in FORMAL_SPECS.items():
            if not arm.startswith("G"):
                continue
            self.assertEqual(allocation["physical_gpu_by_arm"][arm], spec.physical_gpu)
            self.assertTrue(allocation["launch_authorized_by_arm"][arm])

    def test_frozen_training_contracts_keep_external_target_sealed(self) -> None:
        for arm in FORMAL_SPECS:
            config = frozen_config(ROOT, arm, {"profile_peak_reserved_mib": 1})
            self.assertEqual(config["precision"], "fp32")
            self.assertFalse(config["external_target_materialized"])
            self.assertTrue(
                config["external_evaluation_deferred_until_all_family_checkpoints_frozen"]
            )
            if arm.startswith("G"):
                self.assertEqual(config["training"]["effective_batch"], 128)
                self.assertEqual(
                    config["training"]["train_microbatch"],
                    GRAM_MICROBATCH_BY_ARM[arm],
                )
                self.assertEqual(
                    config["training"]["gradient_accumulation"],
                    GRAM_ACCUMULATION_BY_ARM[arm],
                )
                self.assertEqual(config["training"]["primary_final_beam"], 500)
                self.assertFalse(config["training"]["generation_kv_cache"])
            else:
                self.assertEqual(config["training"]["train_batch_size"], 256)

    def test_snapshot_and_worker_are_gpu_isolated(self) -> None:
        for arm, spec in FORMAL_SPECS.items():
            for source in snapshot_sources(ROOT, spec):
                self.assertTrue(source.is_file(), source)
            command = worker_command(ROOT, arm)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={spec.physical_gpu}", command)
            self.assertIn(str(paths(ROOT, arm)["snapshot_worker"]), command)

    def test_single_relevant_item_ndcg(self) -> None:
        self.assertEqual(_ndcg_at_10(None), 0.0)
        self.assertEqual(_ndcg_at_10(11), 0.0)
        self.assertEqual(_ndcg_at_10(1), 1.0)
        self.assertGreater(_ndcg_at_10(2), 0.0)


if __name__ == "__main__":
    unittest.main()
