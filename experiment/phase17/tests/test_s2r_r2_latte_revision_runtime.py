from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment.phase17.protocol.s2r_r2_latte_revision_runtime import (
    CONFIG_PATH,
    RECOVERY_CONFIG_PATH,
    REVISION_ADMISSION_FREE_MIB,
)


class S2RR2LatteRevisionRuntimeTests(unittest.TestCase):
    def test_revision_is_single_scope_and_compute_matched(self) -> None:
        config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertEqual(config["revision_index"], 1)
        self.assertEqual(config["maximum_revisions_for_family"], 1)
        self.assertFalse(config["change_scope"]["training_changed"])
        self.assertTrue(
            config["change_scope"]["same_beam_budget_for_treatment_and_control"]
        )
        self.assertEqual(config["change_scope"]["num_beams_after"], 200)
        self.assertTrue(config["promotion"]["no_further_revision"])

    def test_profile_recovery_changes_only_evaluation_batch_size(self) -> None:
        recovery = json.loads(Path(RECOVERY_CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertFalse(recovery["scientific_configuration_changed"])
        self.assertEqual(recovery["engineering_change"]["num_beams"], 200)
        self.assertEqual(
            recovery["engineering_change"]["evaluation_batch_size_after"], 4
        )
        self.assertFalse(recovery["automatic_retry"])
        self.assertGreaterEqual(REVISION_ADMISSION_FREE_MIB, 4096)


if __name__ == "__main__":
    unittest.main()
