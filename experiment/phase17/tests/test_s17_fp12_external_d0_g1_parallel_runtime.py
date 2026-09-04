from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.protocol import s17_fp12_g1_runtime_guard_v2 as guard_v2
from experiment.phase17.protocol.s17_fp12_external_d0_g1_parallel_runtime import (
    ARM_ID,
    ATTEMPT_ID,
    MINIMUM_FREE_MIB,
    PHYSICAL_GPU,
    paths,
    source_paths,
    verify_prerequisites,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class G1ParallelRuntimeTests(unittest.TestCase):
    def test_scope_gpu_and_attempt_are_frozen(self) -> None:
        self.assertEqual(ARM_ID, "G1_GRAM_PSID_FULL")
        self.assertEqual(ATTEMPT_ID, "attempt_003")
        self.assertEqual(PHYSICAL_GPU, 4)
        self.assertEqual(MINIMUM_FREE_MIB, 18968)

    def test_attempt_is_disjoint_and_reuses_attempt_001_bundle(self) -> None:
        resolved = paths(ROOT)
        self.assertIn("recovery/attempt_003", str(resolved["result"]))
        self.assertIn("external_d0/attempt_001", str(resolved["original_bundle"]))

    def test_prerequisites_do_not_reopen_raw_d0(self) -> None:
        with patch(
            "experiment.phase17.protocol.s17_fp12_external_d0_runtime."
            "materialize_external_evaluation_view",
            side_effect=AssertionError("G1 parallel recovery must not reopen raw D0"),
        ):
            evidence = verify_prerequisites(ROOT)
        self.assertEqual(evidence["single_materialization_count"], 1)
        self.assertFalse(evidence["raw_external_projection_reopened"])
        self.assertFalse(evidence["attempt_002_g1_worker_started"])

    def test_worker_is_gpu4_snapshot_bound(self) -> None:
        command = worker_command(ROOT)
        self.assertIn("CUDA_VISIBLE_DEVICES=4", command)
        self.assertIn(str(paths(ROOT)["snapshot_worker"]), command)
        self.assertIn(str(paths(ROOT)["snapshot"]), command)

    def test_guard_and_scientific_sources_are_snapshot_inputs(self) -> None:
        sources = source_paths(ROOT)
        self.assertTrue(all(path.is_file() for path in sources))
        self.assertEqual(sources[0].name, "s17_fp12_external_d0_g1_parallel_runtime.py")
        self.assertEqual(sources[1].name, "s17_fp12_g1_runtime_guard.py")

    def test_guard_v2_keeps_resource_probe_timeout_non_terminal(self) -> None:
        timeout = subprocess.TimeoutExpired(["nvidia-smi"], 15)
        with patch.object(guard_v2, "_admitted", side_effect=timeout):
            admitted, snapshots, error = guard_v2._probe_admission_fail_soft()
        self.assertIsNone(admitted)
        self.assertIsNone(snapshots)
        self.assertIn("TimeoutExpired", error or "")

    def test_guard_v2_does_not_mask_programming_errors(self) -> None:
        with patch.object(
            guard_v2, "_admitted", side_effect=AssertionError("unexpected defect")
        ):
            with self.assertRaisesRegex(AssertionError, "unexpected defect"):
                guard_v2._probe_admission_fail_soft()


if __name__ == "__main__":
    unittest.main()
