from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_CONFIG = ROOT / "experiment/phase16/configs/stage16_s2_splus_ctrl_split_pair_gpu5_a3_gpu7_a4.json"
RECOVERY_CONFIG = ROOT / "experiment/phase16/configs/stage16_s2_splus_ctrl_split_pair_gpu5_a3_gpu7_a4_a2.json"
RECOVERY_RUNNER = ROOT / "experiment/phase16/run_stage16_s2_splus_ctrl_split_pair_finalize_a2.sh"


class SPlusCtrlSplitPairRecoveryTests(unittest.TestCase):
    def test_recovery_sources_are_identical_and_output_is_isolated(self) -> None:
        original = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
        recovery = json.loads(RECOVERY_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(recovery["sources"], original["sources"])
        self.assertNotEqual(recovery["attempt_id"], original["attempt_id"])
        self.assertNotEqual(recovery["output_dir"], original["output_dir"])

    def test_recovery_changes_only_cpu_finalizer_timeout(self) -> None:
        recovery = json.loads(RECOVERY_CONFIG.read_text(encoding="utf-8"))
        declaration = recovery["engineering_recovery"]
        self.assertEqual(declaration["previous_timeout_seconds"], 600)
        self.assertEqual(declaration["pair_finalizer_timeout_seconds"], 1800)
        self.assertFalse(declaration["scientific_configuration_modified"])
        self.assertFalse(declaration["source_artifacts_modified"])

    def test_runner_preserves_full_hashing_and_classifies_timeout(self) -> None:
        runner = RECOVERY_RUNNER.read_text(encoding="utf-8")
        self.assertIn("PAIR_FINALIZER_TIMEOUT_SECONDS=1800", runner)
        self.assertIn("finalize_splus_ctrl_split.py", runner)
        self.assertIn("PAIR_FINALIZER_TIMEOUT", runner)
        self.assertNotIn("--resume", runner)


if __name__ == "__main__":
    unittest.main()
