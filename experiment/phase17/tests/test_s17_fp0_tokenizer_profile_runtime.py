from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment.phase17.protocol.s17_fp0_tokenizer_profile_runtime import (
    ATTEMPT_ID,
    BATCH_SIZE,
    EXPECTED_PEAK_MIB,
    RESERVED_GPU_ID,
    SAMPLE_SIZE,
    choose_safe_non_gpu1,
    profile_script,
    paths,
    select_profile_sample,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


def gpu(index: int, free: int, utilization: int = 0):
    return SimpleNamespace(index=index, free_mib=free, utilization_percent=utilization)


class S17FP0TokenizerProfileRuntimeTests(unittest.TestCase):
    def test_sample_is_fixed_and_target_independent(self) -> None:
        first = select_profile_sample(ROOT)
        second = select_profile_sample(ROOT)
        self.assertEqual(first, second)
        self.assertEqual(len(first), SAMPLE_SIZE)
        self.assertEqual(len({row["item_id"] for row in first}), SAMPLE_SIZE)
        self.assertTrue(all(row["text"] for row in first))

    def test_gpu1_and_busy_cards_are_never_selected(self) -> None:
        records = [gpu(RESERVED_GPU_ID, 40000), gpu(2, 30000), gpu(3, 35000)]
        selected = choose_safe_non_gpu1(records, {1: [], 2: [{"pid": 7}], 3: []})
        self.assertEqual(selected.index, 3)

    def test_insufficient_or_utilized_cards_block(self) -> None:
        self.assertIsNone(choose_safe_non_gpu1([gpu(0, 12000)], {0: []}))
        self.assertIsNone(choose_safe_non_gpu1([gpu(0, 40000, utilization=10)], {0: []}))

    def test_profile_is_bounded_and_offline(self) -> None:
        self.assertEqual(SAMPLE_SIZE, 512)
        self.assertEqual(BATCH_SIZE, 32)
        self.assertLessEqual(EXPECTED_PEAK_MIB, 10240)
        compile(profile_script(), "<s17-fp0-tokenizer-profile>", "exec")
        self.assertIn("local_files_only=True", profile_script())

    def test_recovery_worker_bootstraps_project_imports(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_005")
        command = worker_command(ROOT, paths(ROOT))
        self.assertEqual(command[:2], ["/usr/bin/env", f"PYTHONPATH={ROOT}"])


if __name__ == "__main__":
    unittest.main()
