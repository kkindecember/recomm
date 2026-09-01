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
    TARGET_GPU_ID,
    choose_authorized_shared_gpu0,
    choose_authorized_shared_gpu1,
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

    def test_shared_gpu1_requires_existing_repeat_and_headroom(self) -> None:
        selected, code = choose_authorized_shared_gpu1([gpu(1, 40000, 99)], {1: []})
        self.assertIsNone(selected)
        self.assertEqual(code, "BLOCKED_GPU1_REPEAT_NOT_PRESENT")

        selected, code = choose_authorized_shared_gpu1(
            [gpu(1, 14000, 99)], {1: [{"pid": 123, "used_memory_mib": 18000}]}
        )
        self.assertIsNone(selected)
        self.assertEqual(code, "BLOCKED_GPU1_SHARED_HEADROOM_INSUFFICIENT")

    def test_shared_gpu1_allows_busy_card_with_sufficient_remaining_memory(self) -> None:
        selected, code = choose_authorized_shared_gpu1(
            [gpu(1, 30000, 99)], {1: [{"pid": 123, "used_memory_mib": 18000}]}
        )
        self.assertEqual(selected.index, 1)
        self.assertEqual(code, "GPU1_SHARED_AUTHORIZED_WITH_EXISTING_REPEAT_AND_MEMORY_MARGIN")

    def test_researcher_selected_gpu0_requires_profile_headroom(self) -> None:
        selected, code = choose_authorized_shared_gpu0([gpu(TARGET_GPU_ID, 14000, 99)])
        self.assertIsNone(selected)
        self.assertEqual(code, "BLOCKED_GPU0_SHARED_HEADROOM_INSUFFICIENT")

        selected, code = choose_authorized_shared_gpu0([gpu(TARGET_GPU_ID, 30000, 99)])
        self.assertEqual(selected.index, TARGET_GPU_ID)
        self.assertEqual(code, "GPU0_SHARED_AUTHORIZED_WITH_MEMORY_MARGIN")

    def test_profile_is_bounded_and_offline(self) -> None:
        self.assertEqual(SAMPLE_SIZE, 512)
        self.assertEqual(BATCH_SIZE, 32)
        self.assertLessEqual(EXPECTED_PEAK_MIB, 10240)
        compile(profile_script(), "<s17-fp0-tokenizer-profile>", "exec")
        self.assertIn("local_files_only=True", profile_script())

    def test_recovery_worker_bootstraps_project_imports(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_010")
        command = worker_command(ROOT, paths(ROOT))
        self.assertEqual(command[:2], ["/usr/bin/env", f"PYTHONPATH={ROOT}"])


if __name__ == "__main__":
    unittest.main()
