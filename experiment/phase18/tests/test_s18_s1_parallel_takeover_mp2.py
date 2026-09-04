from __future__ import annotations

import inspect
import unittest

from experiment.phase18.protocol import s18_s1_gpu0_postrun_guard as guard
from experiment.phase18.protocol import s18_s1_parallel_takeover_mp2 as takeover


class S18S1ParallelTakeoverMp2Tests(unittest.TestCase):
    def test_gpu0_gpu6_layout_is_exact_path(self) -> None:
        _, _, authorization = takeover.verify_authorization()
        runtime = authorization["runtime"]
        self.assertEqual(runtime["beauty_physical_gpus"], [0, 6])
        self.assertEqual(runtime["required_free_mib_by_gpu"], {"0": 19178, "6": 17934})
        source = inspect.getsource(takeover.run_beauty_unit)
        self.assertIn("enable_two_gpu_decoder_parallel", source)
        self.assertIn("generation_use_cache=True", source)
        self.assertIn("cross_attention_cache=True", source)
        self.assertIn("release_cuda_cache_per_user=True", source)
        self.assertNotIn("train_parent(", source)
        self.assertNotIn("train_item_head(", source)

    def test_occupancy_launches_after_beauty_lane_not_aggregate(self) -> None:
        source = inspect.getsource(takeover.master)
        launch_position = source.index("launch_early_occupancy")
        wait_position = source.index("wait_for_toys_im1")
        aggregate_position = source.index("aggregate(config)")
        self.assertLess(launch_position, wait_position)
        self.assertLess(launch_position, aggregate_position)
        authorization = takeover.verify_authorization()[2]
        self.assertTrue(authorization["postrun_occupancy"]["launch_after_beauty_lane"])

    def test_guard_gate_supports_completed_beauty_lane(self) -> None:
        authorization = takeover.verify_authorization()[2]
        science = {
            "scientific_state": "RUNNING",
            "process_alive": True,
            "beauty_lane_state": "COMPLETED",
            "beauty_lane_process_alive": False,
        }
        self.assertTrue(guard.science_launch_gate_satisfied(science, authorization))
        science["beauty_lane_process_alive"] = True
        self.assertFalse(guard.science_launch_gate_satisfied(science, authorization))


if __name__ == "__main__":
    unittest.main()
