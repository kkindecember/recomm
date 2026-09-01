from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from experiment.phase16.protocol.finalize_stage16_s4_toys import (
    verify_formal_runtime_identity,
)
from experiment.phase16.protocol.finalize_stage16_s4_toys_recovery import (
    _calculate_results,
    _load_frozen_inputs,
    _verify_source_attempt,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT
    / "experiment/phase16/configs/stage16_s4_toys_recovery_gpu4_a7_cpu_a8.json"
)
RUNNER = ROOT / "experiment/phase16/run_stage16_s4_toys_recovery_gpu4_a7_cpu_a8.sh"


class Stage16S4ToysRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_recovery_is_cpu_only_write_once_and_preserves_a7(self) -> None:
        self.assertTrue(self.config["resources"]["cpu_only"])
        self.assertEqual(self.config["resources"]["gpu_count"], 0)
        self.assertFalse(self.config["gpu_scientific_inference_recompute"])
        self.assertTrue(
            self.config["derived_statistical_finalization_from_frozen_predictions"]
        )
        self.assertFalse(self.config["source_attempt"]["source_attempt_modified"])
        self.assertNotEqual(
            self.config["output_dir"], self.config["source_attempt"]["output_dir"]
        )

    def test_every_frozen_a7_input_matches_its_declared_sha(self) -> None:
        for declaration in self.config["frozen_inputs"].values():
            path = ROOT / declaration["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, declaration["sha256"])

    def test_a7_failure_lineage_and_runtime_identity_are_valid(self) -> None:
        paths, _ = _load_frozen_inputs(self.config)
        source_config, source_status, source_root = _verify_source_attempt(
            self.config, paths
        )
        self.assertEqual(source_status["status"], "FAILED")
        self.assertEqual(source_status["status_code"], "ARTIFACT_CONTRACT_FAILED")
        manifest_path, manifest = verify_formal_runtime_identity(
            paths["source_config"], source_config, source_root
        )
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(
            manifest["schema_version"],
            "stage16_s4_toys_gpu4_a7_isolated_runtime_v1",
        )

    def test_frozen_results_produce_expected_holm_corrected_labels(self) -> None:
        paths, _ = _load_frozen_inputs(self.config)
        source_config, _, source_root = _verify_source_attempt(self.config, paths)
        result = _calculate_results(self.config, source_config, source_root)
        labels = {
            arm: gate["label"] for arm, gate in result["standalone_gates"].items()
        }
        self.assertEqual(
            labels,
            {
                "S-AUX": "PASS_STANDALONE_COLD_SIGNAL",
                "S-PLUS-CTRL": "FAIL_STANDALONE",
                "S-PLUS": "PASS_STANDALONE_COLD_SIGNAL",
                "G-RIDGE": "FAIL_STANDALONE",
            },
        )
        tests = result["multiplicity"]["primary_tests"]
        self.assertTrue(tests["S-AUX"]["reject_at_alpha"])
        self.assertTrue(tests["S-PLUS"]["reject_at_alpha"])
        self.assertFalse(tests["S-PLUS-CTRL"]["reject_at_alpha"])
        self.assertFalse(tests["G-RIDGE"]["reject_at_alpha"])

    def test_runner_forbids_gpu_tmux_and_automatic_retry(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('env CUDA_VISIBLE_DEVICES=""', runner)
        self.assertIn('if [[ -e "$OUTPUT" ]]', runner)
        self.assertNotIn("nvidia-smi", runner)
        self.assertNotIn("tmux", runner)
        self.assertNotIn("while ", runner)


if __name__ == "__main__":
    unittest.main()
