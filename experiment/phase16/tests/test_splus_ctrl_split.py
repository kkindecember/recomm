from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from finalize_splus_ctrl_split import scientific_core, validate_arm_summary, validate_pair


def frozen_config() -> dict:
    budget = {
        "pretrain": {"optimizer_steps": 10},
        "finetune": {"optimizer_steps": 2},
        "internal_dev_transitions": 3,
        "pseudo_cold_events": 4,
        "full_catalog_items": 5,
    }
    return {
        "seed": 1502,
        "domain": "Toys_cold50",
        "inputs": {"train": {"path": "train.jsonl", "sha256": "x"}},
        "model": {"execution_precision": "fp32"},
        "formal_budget": budget,
        "admission": {"maximum_eligible_peak_reserved_mib": 28672},
        "resource_evidence": {"selected_candidate": "e16_g4_a64"},
        "batching_adaptation": {"generation_microbatch": 4},
        "compatibility_patch": {"algorithm_objective_unchanged": True},
        "attempt_id": "attempt-a",
        "output_dir": "output-a",
        "resources": {"physical_gpu": 5},
        "execution": {"exact_start_command": "run-a"},
    }


def arm_summary(arm: str) -> dict:
    pseudo = None
    if arm == "S-PLUS":
        pseudo = {"all_finite": True, "events": 4, "candidate_items": 5}
    budget = {"dataset": "same", "optimizer_steps": 10}
    return {
        "verdict": f"PASS_S16_2_{arm.replace('-', '_')}_FORMAL_EXECUTION",
        "arm": arm,
        "arm_optimizer_steps": 12,
        "internal_dev_generation_admission": {"all_finite": True, "events": 3},
        "pseudo_cold_full_catalog_admission": pseudo,
        "base_checkpoint_unchanged": True,
        "base_checkpoint_sha256_before": "checkpoint",
        "test_read": False,
        "validation_used": False,
        "peak_cuda_reserved_mib": 2000,
        "budget_audit": {
            "pretrain": {"budget": budget},
            "finetune": {"budget": {"dataset": "same", "optimizer_steps": 2}},
        },
    }


class SPlusCtrlSplitTests(unittest.TestCase):
    def test_execution_only_differences_do_not_change_scientific_core(self) -> None:
        plus = frozen_config()
        control = copy.deepcopy(plus)
        control["attempt_id"] = "attempt-b"
        control["output_dir"] = "output-b"
        control["resources"]["physical_gpu"] = 7
        control["execution"]["exact_start_command"] = "run-b"
        self.assertEqual(scientific_core(plus), scientific_core(control))

    def test_budget_change_is_not_an_allowed_split_difference(self) -> None:
        plus = frozen_config()
        control = copy.deepcopy(plus)
        control["formal_budget"]["pretrain"]["optimizer_steps"] = 9
        self.assertNotEqual(scientific_core(plus), scientific_core(control))

    def test_valid_split_pair_passes(self) -> None:
        plus_config = frozen_config()
        control_config = copy.deepcopy(plus_config)
        control_config["attempt_id"] = "attempt-b"
        control_config["output_dir"] = "output-b"
        control_config["resources"]["physical_gpu"] = 7
        audit = validate_pair(
            arm_summary("S-PLUS"),
            arm_summary("S-PLUS-CTRL"),
            plus_config,
            control_config,
        )
        self.assertTrue(audit["pretrain"]["matched"])
        self.assertTrue(audit["finetune"]["matched"])

    def test_ctrl_pseudo_cold_output_is_rejected(self) -> None:
        control = arm_summary("S-PLUS-CTRL")
        control["pseudo_cold_full_catalog_admission"] = {"all_finite": True}
        with self.assertRaisesRegex(ValueError, "unexpectedly produced"):
            validate_pair(arm_summary("S-PLUS"), control, frozen_config(), frozen_config())

    def test_nonfinite_arm_is_rejected(self) -> None:
        control = arm_summary("S-PLUS-CTRL")
        control["internal_dev_generation_admission"]["all_finite"] = False
        with self.assertRaisesRegex(ValueError, "incomplete or non-finite"):
            validate_arm_summary(control, frozen_config(), "S-PLUS-CTRL")


if __name__ == "__main__":
    unittest.main()
