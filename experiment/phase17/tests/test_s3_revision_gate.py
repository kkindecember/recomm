from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment.phase17.protocol.s3_revision_gate_runtime import evaluate_arm


ROOT = Path(__file__).resolve().parents[3]


class S3RevisionGateTests(unittest.TestCase):
    def test_budget_freezes_treatment_and_controls(self) -> None:
        config = json.loads(
            (ROOT / "experiment/phase17/config/s17_s3_revision_gate.json").read_text()
        )
        modules = [arm["module_id"] for arm in config["arms"]]
        self.assertEqual(
            modules,
            [
                "A0_bear_proxy",
                "A1_prefixcurr",
                "E0_shortcut_fid",
                "E0_shortcut_fid_full_control",
                "E0_shortcut_fid_random_control",
            ],
        )
        self.assertFalse(config["test_read"])
        self.assertFalse(config["sports_read"])

    def test_e0_gate_parses_non_degenerate_ratio(self) -> None:
        arm = {"module_id": "E0_shortcut_fid"}
        parsed = {
            "traceback": False,
            "forbidden_test_evidence": [],
            "validation_metrics": {"ndcg@10": 0.1},
            "mechanism_metric_lines": [
                "S17_MECHANISM_METRIC epoch=1 E0_shortcut_fid/selected_history_ratio=0.50000000"
            ],
            "peak_reserved_mib": 20000.0,
        }
        self.assertTrue(all(evaluate_arm(arm, parsed, 0).values()))

    def test_random_control_requires_same_size(self) -> None:
        arm = {"module_id": "E0_shortcut_fid_random_control"}
        parsed = {
            "traceback": False,
            "forbidden_test_evidence": [],
            "validation_metrics": {"ndcg@10": 0.1},
            "mechanism_metric_lines": [
                "S17_MECHANISM_METRIC epoch=1 "
                "E0_shortcut_fid_random_control/selected_history_ratio=0.50000000 "
                "E0_shortcut_fid_random_control/adaptive_target_ratio=0.50000000"
            ],
            "peak_reserved_mib": 20000.0,
        }
        self.assertTrue(all(evaluate_arm(arm, parsed, 0).values()))

    def test_exploration_budget_runs_one_matched_epoch_for_ten_arms(self) -> None:
        config = json.loads(
            (ROOT / "experiment/phase17/config/s17_s3_formal_budget.json").read_text()
        )
        self.assertFalse(config["historical_baseline"]["training_rerun_required"])
        self.assertEqual(config["historical_baseline"]["rec_epochs"], 30)
        self.assertEqual(config["exploration_stage"]["rec_epochs"], 1)
        self.assertEqual(len(config["exploration_stage"]["arms"]), 10)
        self.assertEqual(
            config["exploration_stage"]["arms"][0]["arm_id"], "gram_continue"
        )
        self.assertEqual(config["resources"]["preferred_gpu_count"], 4)
        self.assertFalse(config["test_read"])
        self.assertFalse(config["sports_read"])


if __name__ == "__main__":
    unittest.main()
