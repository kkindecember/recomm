from __future__ import annotations

import unittest
from pathlib import Path

from experiment.phase17.core.status_writer import assert_neutral_name
from experiment.phase17.protocol.s3_dual_gpu_handoff_runtime import (
    RUNTIME_GPU,
    SCIENCE_GPUS,
    TMUX_SESSION,
    arm_environment,
    pending_arm_ids,
)


ROOT = Path(__file__).resolve().parents[3]


class S3DualGpuHandoffTests(unittest.TestCase):
    def test_two_science_gpus_but_only_gpu1_runtime(self) -> None:
        self.assertEqual(SCIENCE_GPUS, (0, 1))
        self.assertEqual(RUNTIME_GPU, 1)
        assert_neutral_name(TMUX_SESSION)

    def test_pending_queue_excludes_completed_and_adopted_arm(self) -> None:
        config = {
            "arms": [
                {"arm_id": "gram_continue"},
                {"arm_id": "b1_latte"},
                {"arm_id": "b0_mvi"},
                {"arm_id": "c0_biflow"},
            ]
        }
        self.assertEqual(
            pending_arm_ids(config, {"gram_continue"}, "b1_latte"),
            ["b0_mvi", "c0_biflow"],
        )

    def test_physical_gpu_isolated_by_visible_device(self) -> None:
        self.assertEqual(arm_environment(ROOT, 0)["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(arm_environment(ROOT, 1)["CUDA_VISIBLE_DEVICES"], "1")


if __name__ == "__main__":
    unittest.main()
