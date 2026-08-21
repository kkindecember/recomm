from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from pseudo_cold_teacher import build_examples, fold  # noqa: E402


class TestTeacherIsolation(unittest.TestCase):
    def test_fold_is_deterministic(self):
        self.assertEqual(fold(1401, "u", 3, 10), fold(1401, "u", 3, 10))

    def test_forbidden_item_hard_fails_before_training(self):
        embeddings = torch.eye(4)
        with self.assertRaisesRegex(RuntimeError, "Forbidden"):
            build_examples(
                [("u", ["a", "forbidden", "b"])],
                {"a": 0, "forbidden": 1, "b": 2, "c": 3},
                embeddings,
                {"forbidden"},
                20,
                0.85,
                1401,
                2,
            )


if __name__ == "__main__":
    unittest.main()
