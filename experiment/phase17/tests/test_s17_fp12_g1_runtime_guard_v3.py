from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.protocol import s17_fp12_g1_runtime_guard_v3 as guard_v3


ROOT = Path(__file__).resolve().parents[3]


class G1RuntimeGuardV3Tests(unittest.TestCase):
    def test_v3_is_gpu4_and_fresh_process_per_cycle(self) -> None:
        self.assertEqual(guard_v3.PHYSICAL_GPU, 4)
        self.assertEqual(guard_v3.PROFILE_PEAK_RESERVED_MIB, 15892)
        self.assertEqual(guard_v3.MINIMUM_FREE_MIB, 18000)
        self.assertLess(
            guard_v3.PROFILE_PEAK_RESERVED_MIB,
            guard_v3.MINIMUM_FREE_MIB,
        )

    def test_cycle_command_uses_frozen_worker_and_disjoint_cycle(self) -> None:
        manifest = ROOT / guard_v3.SNAPSHOT_SUFFIX
        cycle_dir = ROOT / guard_v3.RESULT_SUFFIX / "run-0002"
        command = guard_v3.cycle_command(ROOT, manifest, cycle_dir, 2)
        self.assertEqual(command[0], str(guard_v3.profile_base.GRAM_PYTHON))
        self.assertIn("000_s17_fp12_g1_runtime_guard_v3.py", command[1])
        self.assertIn("cycle-worker", command)
        self.assertIn(str(cycle_dir), command)
        self.assertEqual(command[-1], "2")

    def test_cycle_directory_cannot_escape_v3_tree(self) -> None:
        with self.assertRaisesRegex(PermissionError, "escaped"):
            guard_v3._validate_cycle_dir(ROOT, ROOT / "run-0002", 2)
        with self.assertRaisesRegex(ValueError, "iteration"):
            guard_v3._validate_cycle_dir(
                ROOT, ROOT / guard_v3.RESULT_SUFFIX / "run-0003", 2
            )

    def test_resource_probe_timeout_remains_non_terminal(self) -> None:
        timeout = subprocess.TimeoutExpired(["nvidia-smi"], 15)
        with patch.object(guard_v3, "_admitted", side_effect=timeout):
            admitted, snapshots, error = guard_v3._probe_admission_fail_soft()
        self.assertIsNone(admitted)
        self.assertIsNone(snapshots)
        self.assertIn("TimeoutExpired", error or "")

    def test_programming_errors_are_not_masked(self) -> None:
        with patch.object(
            guard_v3, "_admitted", side_effect=AssertionError("unexpected defect")
        ):
            with self.assertRaisesRegex(AssertionError, "unexpected defect"):
                guard_v3._probe_admission_fail_soft()

    def test_v2_stop_validation_is_exact_and_science_isolated(self) -> None:
        status = {
            "experiment_id": guard_v3.V2_EXPERIMENT_ID,
            "scientific_state": "COMPLETED",
            "affects_scientific_result": False,
            "workload_pid": 123,
            "tmux_session": guard_v3.V2_EXPERIMENT_ID,
        }
        self.assertEqual(guard_v3._validate_v2_status(status), (123, guard_v3.V2_EXPERIMENT_ID))
        status["affects_scientific_result"] = True
        with self.assertRaisesRegex(RuntimeError, "not isolated"):
            guard_v3._validate_v2_status(status)


if __name__ == "__main__":
    unittest.main()
