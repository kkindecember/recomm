from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from splus_formal_train import accumulation_window_size, cuda_device_index, embedding_microbatch_indices, epoch_permutation, reset_peak_memory_stats_compat


class SPlusFormalTests(unittest.TestCase):
    def test_cuda_device_index_is_torch_1_11_compatible(self) -> None:
        self.assertEqual(cuda_device_index(torch.device("cuda")), 0)
        self.assertEqual(cuda_device_index(torch.device("cuda:0")), 0)
        self.assertEqual(cuda_device_index(torch.device("cuda:5")), 5)
        with self.assertRaises(ValueError):
            cuda_device_index(torch.device("cpu"))

    def test_peak_reset_initializes_cuda_context_first(self) -> None:
        calls = []
        with patch("splus_formal_train.torch.cuda.init", side_effect=lambda: calls.append("init")), patch(
            "splus_formal_train.torch.cuda.reset_peak_memory_stats",
            side_effect=lambda index: calls.append(("reset", index)),
        ):
            reset_peak_memory_stats_compat(torch.device("cuda:0"))
        self.assertEqual(calls, ["init", ("reset", 0)])

    def test_epoch_tail_produces_frozen_optimizer_step_count(self) -> None:
        windows = [accumulation_window_size(27659, 256, start) for start in range(0, 27659, 256)]
        self.assertEqual(len(windows), 109)
        self.assertEqual(windows[-1], 11)
        self.assertEqual(sum(windows), 27659)

    def test_embedding_microbatch_cycles_deterministically(self) -> None:
        permutation = [4, 2, 0, 3, 1]
        self.assertEqual(embedding_microbatch_indices(permutation, 0, 4), [4, 2, 0, 3])
        self.assertEqual(embedding_microbatch_indices(permutation, 1, 4), [1, 4, 2, 0])

    def test_epoch_permutation_is_reproducible_and_complete(self) -> None:
        first = epoch_permutation(19, 1502, "S-PLUS", "pretrain", 7)
        second = epoch_permutation(19, 1502, "S-PLUS", "pretrain", 7)
        other = epoch_permutation(19, 1502, "S-PLUS", "pretrain", 8)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(19)))
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
