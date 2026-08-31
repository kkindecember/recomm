from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.status_writer import assert_neutral_name
from experiment.phase17.protocol.s3_exploration_portfolio_runtime import (
    EXPERIMENT_ID,
    PHYSICAL_GPU,
    build_arm_command,
)


ROOT = Path(__file__).resolve().parents[3]


class S3ExplorationPortfolioTests(unittest.TestCase):
    def test_queue_is_neutral_and_fixed_to_allocated_gpu1(self) -> None:
        self.assertEqual(PHYSICAL_GPU, 1)
        assert_neutral_name(EXPERIMENT_ID)

    def test_budget_freezes_matched_one_epoch_queue(self) -> None:
        budget = json.loads(
            (ROOT / "experiment/phase17/config/s17_s3_formal_budget.json").read_text()
        )
        stage = budget["exploration_stage"]
        self.assertEqual(stage["rec_epochs"], 1)
        self.assertEqual(len(stage["arms"]), 10)
        self.assertEqual(stage["arms"][0]["arm_id"], "gram_continue")
        self.assertFalse(budget["test_read"])
        self.assertFalse(budget["sports_read"])

    def test_b1_command_loads_history_and_never_tests(self) -> None:
        arm = {"arm_id": "b1_latte", "track_id": "B1", "module_id": "B1_latte"}
        with tempfile.TemporaryDirectory(prefix="s17-s3-command-") as temporary:
            command = build_arm_command(
                ROOT,
                Path(temporary),
                arm,
                1,
                Path(temporary) / "teacher.json",
            )
        self.assertEqual(command[command.index("--rec_epochs") + 1], "1")
        self.assertEqual(command[command.index("--test_epoch_rec") + 1], "0")
        self.assertEqual(command[command.index("--save_predictions") + 1], "0")
        self.assertEqual(command[command.index("--s17_modules") + 1], "B1_latte")
        self.assertTrue(any(value.endswith("model_rec_phase_1_epoch_30.pt") for value in command))
        self.assertIn("Toys", command)
        self.assertNotIn("Sports", command)


if __name__ == "__main__":
    unittest.main()
