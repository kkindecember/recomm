from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from toys_b2_verifier_probe_smoke import (  # noqa: E402
    _configure_deterministic_smoke_math,
    _train_sample_lookup,
)


class TestToysB2VerifierProbeSmoke(unittest.TestCase):
    def test_chronological_lookup_survives_max_history_truncation(self):
        samples = [
            {"user_id": "u", "target": f"i{position}", "history_item_ids": list(range(min(position, 20)))}
            for position in range(1, 34)
        ]
        lookup = _train_sample_lookup(samples)
        self.assertEqual(lookup[("u", 1)], (0, "i1"))
        self.assertEqual(lookup[("u", 20)], (19, "i20"))
        self.assertEqual(lookup[("u", 32)], (31, "i32"))
        self.assertEqual(lookup[("u", 33)], (32, "i33"))

    def test_deterministic_math_requires_cublas_workspace_contract(self):
        original = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        original_deterministic = torch.are_deterministic_algorithms_enabled()
        original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
                _configure_deterministic_smoke_math()
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            mode = _configure_deterministic_smoke_math()
            self.assertEqual(mode["cublas_workspace_config"], ":4096:8")
            self.assertFalse(mode["cuda_matmul_allow_tf32"])
            self.assertTrue(mode["deterministic_algorithms"])
        finally:
            if original is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = original
            torch.use_deterministic_algorithms(original_deterministic)
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


if __name__ == "__main__":
    unittest.main()
