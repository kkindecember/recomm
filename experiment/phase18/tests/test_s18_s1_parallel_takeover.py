from __future__ import annotations

import inspect
import unittest

from experiment.phase18.protocol import s18_s1_gpu0_postrun_guard as guard
from experiment.phase18.protocol import s18_s1_parallel_takeover as takeover


class S18S1ParallelTakeoverTests(unittest.TestCase):
    def test_authorized_parallel_layout_is_result_preserving(self) -> None:
        _, _, authorization = takeover.verify_authorization()
        scope = authorization["correction_scope"]
        runtime = authorization["runtime"]
        self.assertTrue(scope["parallel_takeover"])
        self.assertTrue(scope["preserve_active_toys_worker"])
        self.assertFalse(scope["scientific_config_changes"])
        self.assertFalse(scope["beam_changes"])
        self.assertFalse(scope["score_changes_allowed"])
        self.assertEqual(runtime["beauty_physical_gpu"], 0)
        self.assertEqual(runtime["beauty_units"], ["Beauty:I0", "Beauty:I-1"])
        self.assertTrue(runtime["beauty_units_serial"])
        self.assertTrue(runtime["toys_and_beauty_parallel"])

    def test_beauty_worker_is_checkpoint_only_and_single_gpu(self) -> None:
        source = inspect.getsource(takeover.run_beauty_unit)
        self.assertIn("load_frozen_models", source)
        self.assertNotIn("train_parent(", source)
        self.assertNotIn("train_item_head(", source)
        self.assertIn("torch.cuda.device_count() != 1", source)
        self.assertIn("generation_use_cache=True", source)
        self.assertIn("cross_attention_cache=True", source)
        self.assertIn("release_cuda_cache_per_user=True", source)

    def test_completed_toys_units_are_carried_without_recompute(self) -> None:
        source = inspect.getsource(takeover.master)
        self.assertIn('carry_forward("Toys:I0"', source)
        self.assertIn('carry_forward("Toys:I-1"', source)
        self.assertEqual(takeover.ATTEMPT_ID, "run-0004")

    def test_occupancy_is_gpu0_post_science_and_preemptible(self) -> None:
        _, authorization, _ = guard.occupancy_config()
        occupancy = authorization["postrun_occupancy"]
        self.assertEqual(occupancy["physical_gpu"], 0)
        self.assertTrue(occupancy["fresh_cuda_process_per_cycle"])
        self.assertTrue(occupancy["normal_priority_preemption"])
        self.assertFalse(occupancy["result_selection_eligible"])
        self.assertTrue(occupancy["repeat_metrics_ignored"])
        self.assertFalse(occupancy["affects_scientific_result"])
        self.assertTrue(
            guard.science_launch_gate_satisfied(
                {"scientific_state": "COMPLETED", "process_alive": False},
                authorization,
            )
        )
        self.assertFalse(
            guard.science_launch_gate_satisfied(
                {"scientific_state": "RUNNING", "process_alive": True},
                authorization,
            )
        )

    def test_occupancy_checks_normal_priority_inside_each_cycle(self) -> None:
        source = inspect.getsource(guard.cycle_worker)
        self.assertIn("normal_priority_reason", source)
        self.assertIn('"PREEMPTED_NORMAL_PRIORITY"', source)
        command = guard.cycle_command(3)
        self.assertEqual(command[:3], ["nice", "-n", "19"])


if __name__ == "__main__":
    unittest.main()
