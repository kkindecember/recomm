from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.status_writer import assert_neutral_name
from experiment.phase17.protocol.s3_formal_parent_runtime import (
    EXPERIMENT_ID,
    PHYSICAL_GPU,
    build_command,
    latest_completed_epoch,
)


ROOT = Path(__file__).resolve().parents[3]


class S3FormalParentTests(unittest.TestCase):
    def test_parent_is_fixed_to_researcher_allocated_gpu1(self) -> None:
        self.assertEqual(PHYSICAL_GPU, 1)
        assert_neutral_name(EXPERIMENT_ID)

    def test_parent_command_is_validation_only_and_thirty_epochs(self) -> None:
        command = build_command(ROOT, ROOT / "artifacts/example", 30)
        self.assertEqual(command[command.index("--rec_epochs") + 1], "30")
        self.assertEqual(command[command.index("--test_epoch_rec") + 1], "0")
        self.assertEqual(command[command.index("--save_predictions") + 1], "0")
        self.assertEqual(command[command.index("--s17_modules") + 1], "")
        self.assertIn("Toys_s17_d0", command)
        self.assertNotIn("Sports", command)

    def test_formal_budget_parent_matches_runner(self) -> None:
        budget = json.loads(
            (ROOT / "experiment/phase17/config/s17_s3_confirmation_budget.json").read_text()
        )
        self.assertEqual(budget["parent_stage"]["rec_epochs"], 30)
        self.assertEqual(budget["resources"]["minimum_free_mib_at_admission"], 27648)
        self.assertFalse(budget["test_read"])
        self.assertFalse(budget["sports_read"])

    def test_epoch_progress_parser(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-parent-progress-") as temporary:
            log = Path(temporary) / "run.log"
            log.write_text(
                "The average training loss for rec phase 1 epoch 1 is 4.0\n"
                "The average training loss for rec phase 1 epoch 7 is 2.0\n",
                encoding="utf-8",
            )
            self.assertEqual(latest_completed_epoch(log), 7)


if __name__ == "__main__":
    unittest.main()
