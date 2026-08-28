from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from splus_formal_train_accel import accumulation_batch_window, generation_microbatches


class SPlusFormalAcceleratedTests(unittest.TestCase):
    def test_generation_batches_are_complete_and_have_expected_tail(self) -> None:
        batches = generation_microbatches(list(range(27659)), 4)
        self.assertEqual(len(batches), 6915)
        self.assertTrue(all(len(batch) == 4 for batch in batches[:-1]))
        self.assertEqual(len(batches[-1]), 3)
        self.assertEqual([index for batch in batches for index in batch], list(range(27659)))

    def test_accumulation_preserves_optimizer_steps_and_effective_batches(self) -> None:
        batches = generation_microbatches(list(range(27659)), 4)
        windows = [accumulation_batch_window(batches, 64, start) for start in range(0, len(batches), 64)]
        self.assertEqual(len(windows), 109)
        self.assertEqual(windows[0], (64, 256))
        self.assertEqual(windows[-1], (3, 11))
        self.assertEqual(sum(examples for _, examples in windows), 27659)
        self.assertEqual(math.ceil(len(batches) / 64), 109)

    def test_preregistered_effective_embedding_batch_is_preserved(self) -> None:
        self.assertEqual(16 * 64, 1024)
        self.assertEqual(4 * 64, 256)

    def test_tail_generation_weights_sum_to_one(self) -> None:
        batches = generation_microbatches(list(range(27659)), 4)
        tail = batches[-3:]
        _, examples = accumulation_batch_window(batches, 64, len(batches) - 3)
        self.assertEqual(examples, 11)
        self.assertAlmostEqual(sum(len(batch) / examples for batch in tail), 1.0)


if __name__ == "__main__":
    unittest.main()
