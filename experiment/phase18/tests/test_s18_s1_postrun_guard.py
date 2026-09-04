from __future__ import annotations

import inspect
import unittest

from experiment.phase18.protocol import s18_s1_postrun_guard as guard


class S18S1PostrunGuardTests(unittest.TestCase):
    def test_fixed_gpu_and_fresh_process_contract(self) -> None:
        _, authorization = guard.occupancy_config()
        occupancy = authorization["postrun_occupancy"]
        self.assertEqual(occupancy["physical_gpu"], 4)
        self.assertTrue(occupancy["fresh_cuda_process_per_cycle"])
        self.assertFalse(occupancy["result_selection_eligible"])
        self.assertTrue(occupancy["repeat_metrics_ignored"])

    def test_cycle_paths_are_disjoint_and_numbered(self) -> None:
        first = guard.cycle_dir(1)
        second = guard.cycle_dir(2)
        self.assertNotEqual(first, second)
        self.assertTrue(str(first).endswith("run-0001"))
        self.assertTrue(str(second).endswith("run-0002"))
        with self.assertRaises(ValueError):
            guard.cycle_dir(0)

    def test_worker_spawns_cycle_worker_subprocess(self) -> None:
        command = guard.cycle_command(7)
        self.assertIn("cycle-worker", command)
        self.assertEqual(command[-2], "--cycle-dir")
        self.assertTrue(command[-1].endswith("run-0007"))

    def test_cycle_is_scientifically_isolated(self) -> None:
        source = inspect.getsource(guard.cycle_worker)
        self.assertIn('"result_selection_eligible": False', source)
        self.assertIn('"repeat_metrics_ignored": True', source)
        self.assertIn('"affects_scientific_result": False', source)


if __name__ == "__main__":
    unittest.main()
