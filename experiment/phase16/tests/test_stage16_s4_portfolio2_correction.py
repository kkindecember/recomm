from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from experiment.phase16.protocol.finalize_stage16_s4_portfolio2_correction import (
    CORRECT_COMPARATOR,
    INCORRECT_COMPARATOR,
    _build_corrected_events,
    _calculate_corrected_results,
    _load_frozen_inputs,
    _verify_a8_lineage,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT
    / "experiment/phase16/configs/stage16_s4_portfolio2_correction_a9_cpu.json"
)
RUNNER = ROOT / "experiment/phase16/run_stage16_s4_portfolio2_correction_a9_cpu.sh"
PLAN = (
    ROOT
    / "plan/第十六阶段/GRAM_第十六阶段_SpecGR与GenRecEdit忠实迁移及条件式组合方法开发计划v0.1.md"
)


class Stage16S4Portfolio2CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_correction_is_cpu_only_write_once_and_preserves_a7_a8(self) -> None:
        self.assertTrue(self.config["resources"]["cpu_only"])
        self.assertEqual(self.config["resources"]["gpu_count"], 0)
        self.assertFalse(self.config["gpu_scientific_inference_recompute"])
        self.assertFalse(self.config["correction_scope"]["source_attempt_modified"])
        self.assertFalse(self.config["correction_scope"]["source_a8_modified"])
        self.assertEqual(
            self.config["correction_scope"]["corrected_arm_method"],
            CORRECT_COMPARATOR,
        )

    def test_every_frozen_input_matches_its_declared_sha(self) -> None:
        for declaration in self.config["frozen_inputs"].values():
            path = ROOT / declaration["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, declaration["sha256"])

    def test_plan_expected_portfolio_but_a8_used_p0_r2(self) -> None:
        plan = PLAN.read_text(encoding="utf-8")
        self.assertIn("| `R2` | 冻结 R² portfolio@2 |", plan)
        self.assertEqual(
            self.config["correction_scope"]["a8_actual_comparator"],
            INCORRECT_COMPARATOR,
        )
        p0_path = self.config["frozen_inputs"]["phase13_p0_predictions"]["path"]
        self.assertIn("v1_r2_toys_p0/predictions_validation.jsonl", p0_path)

    def test_corrected_results_match_phase13_and_change_only_saux_label(self) -> None:
        paths, _ = _load_frozen_inputs(self.config)
        a8_summary = _verify_a8_lineage(paths)
        events, predictions, reconstruction = _build_corrected_events(
            self.config, paths, a8_summary
        )
        result = _calculate_corrected_results(self.config, events, a8_summary)
        self.assertEqual(len(events), 8789)
        self.assertEqual(len(predictions), 8789)
        self.assertEqual(reconstruction["candidate_mismatches"], 0)
        self.assertEqual(reconstruction["f0_metric_mismatches"], 0)
        self.assertLessEqual(reconstruction["max_abs_error"], 1e-15)
        self.assertAlmostEqual(
            result["metrics"]["R2"]["cold"]["hit@50"],
            0.029768719945042363,
        )
        self.assertEqual(
            result["standalone_gates"]["S-AUX"]["label"],
            "PASS_STANDALONE_PARETO",
        )
        self.assertEqual(
            result["correction_impact"]["changed_standalone_labels"],
            {
                "S-AUX": {
                    "a8": "PASS_STANDALONE_COLD_SIGNAL",
                    "a9": "PASS_STANDALONE_PARETO",
                }
            },
        )
        overlap = result["cold_hit_complementarity_diagnostic"]["S-AUX_vs_R2"]
        self.assertEqual(overlap["treatment_only_hits"], 218)
        self.assertEqual(overlap["control_only_hits"], 85)
        self.assertEqual(overlap["both_hit"], 45)

    def test_runner_forbids_gpu_tmux_and_automatic_retry(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('env CUDA_VISIBLE_DEVICES=""', runner)
        self.assertIn('if [[ -e "$OUTPUT" ]]', runner)
        self.assertNotIn("nvidia-smi", runner)
        self.assertNotIn("tmux", runner)
        self.assertNotIn("while ", runner)


if __name__ == "__main__":
    unittest.main()
