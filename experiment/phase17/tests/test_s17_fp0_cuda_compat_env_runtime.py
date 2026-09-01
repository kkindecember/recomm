from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment.phase17.protocol.s17_fp0_cuda_compat_env_runtime import (
    ATTEMPT_ID,
    CUDA_SMOKE_MIN_FREE_MIB,
    PROFILE_ADMISSION_MIB,
    TORCH_CUDA_VERSION,
    TORCH_VERSION,
    TORCH_WHEEL_SHA256,
    TORCH_WHEEL_URL,
    compatible_requirements,
    cuda_smoke_script,
    paths,
    select_authorized_gpu1,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


def gpu(index: int, free: int):
    return SimpleNamespace(index=index, free_mib=free, utilization_percent=99)


class S17FP0CudaCompatEnvRuntimeTests(unittest.TestCase):
    def test_cuda13_packages_are_removed_but_regular_freeze_is_retained(self) -> None:
        source = "\n".join(
            (
                "torch==2.13.0",
                "triton==3.7.1",
                "cuda-toolkit==13.0.3.0",
                "nvidia-cublas==13.1.1.3",
                "nvidia-ml-py==13.610.43",
                "sentence-transformers==5.1.0",
            )
        )
        self.assertEqual(
            compatible_requirements(source),
            ["nvidia-ml-py==13.610.43", "sentence-transformers==5.1.0"],
        )

    def test_official_torch_wheel_is_exactly_pinned(self) -> None:
        self.assertEqual(TORCH_VERSION, "2.7.1+cu126")
        self.assertEqual(TORCH_CUDA_VERSION, "12.6")
        self.assertIn(TORCH_WHEEL_SHA256, TORCH_WHEEL_URL)
        self.assertIn("mirrors.aliyun.com/pytorch-wheels/cu126", TORCH_WHEEL_URL)
        self.assertEqual(len(TORCH_WHEEL_SHA256), 64)

    def test_gpu1_smoke_requires_repeat_and_only_tiny_smoke_headroom(self) -> None:
        selected, code = select_authorized_gpu1([gpu(1, 30000)], {1: []})
        self.assertIsNone(selected)
        self.assertEqual(code, "BLOCKED_GPU1_REPEAT_NOT_PRESENT")
        selected, code = select_authorized_gpu1([gpu(1, 512)], {1: [{"pid": 1}]})
        self.assertIsNone(selected)
        self.assertEqual(code, "BLOCKED_GPU1_SHARED_HEADROOM_INSUFFICIENT")
        selected, code = select_authorized_gpu1([gpu(1, 14000)], {1: [{"pid": 1}]})
        self.assertEqual(selected.index, 1)
        self.assertEqual(code, "GPU1_SHARED_AUTHORIZED_FOR_CUDA_SMOKE")
        self.assertEqual(CUDA_SMOKE_MIN_FREE_MIB, 1024)
        self.assertEqual(PROFILE_ADMISSION_MIB, 14336)

    def test_smoke_and_snapshot_command_are_valid(self) -> None:
        compile(cuda_smoke_script(), "<s17-fp0-cuda-smoke>", "exec")
        self.assertEqual(ATTEMPT_ID, "attempt_004")
        command = worker_command(ROOT, paths(ROOT))
        self.assertEqual(command[:2], ["/usr/bin/env", f"PYTHONPATH={ROOT}"])


if __name__ == "__main__":
    unittest.main()
