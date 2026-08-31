#!/usr/bin/env python3
"""Build and verify the immutable S16-4 GPU4 a7 runtime."""

from experiment.phase16.protocol import prepare_stage16_s4_gpu4_a4_runtime as base


base.DEFAULT_SNAPSHOT = base.ROOT / ".runtime/phase16_s4_toys_gpu4_a7_runtime"
base.CONFIG = "experiment/phase16/configs/stage16_s4_toys_standalone_gpu4_a7.json"
base.EXPECTED_SNAPSHOT_REL = ".runtime/phase16_s4_toys_gpu4_a7_runtime"
base.EXPECTED_PHYSICAL_GPU = 4
base.EXPECTED_MINIMUM_FREE_MIB = 19000
base.MANIFEST_SCHEMA_VERSION = "stage16_s4_toys_gpu4_a7_isolated_runtime_v1"
base.FORMAL_OUTPUT_WRITE_SCOPE = (
    "artifacts/phase16/s4_toys_standalone/formal/toys_seed1502_gpu4_a7"
)
base.OVERLAYS = (
    base.CONFIG,
    "experiment/phase16/configs/stage16_s2_saux_formal_toys_a2.json",
    "experiment/phase16/protocol/stage16_s4_toys_validation.py",
    "experiment/phase16/protocol/check_stage16_s4_saux_cpu_materialization.py",
    "experiment/phase16/protocol/finalize_stage16_s4_toys.py",
    "experiment/phase16/protocol/prepare_stage16_s4_gpu4_a4_runtime.py",
    "experiment/phase16/protocol/prepare_stage16_s4_gpu4_a7_runtime.py",
    "experiment/phase16/tests/test_stage16_s4_toys_frozen_preflight.py",
    "experiment/phase16/tests/test_stage16_s4_toys_validation.py",
    "experiment/phase16/configs/stage16_s4_toys_frozen_preflight.json",
    "experiment/phase16/protocol/stage16_s4_toys_frozen_preflight.py",
    "experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh",
    "experiment/phase16/run_stage16_s4_toys_standalone_gpu4_a7.sh",
    "experiment/phase16/run_stage16_s4_toys_standalone_gpu4_a7_inner.sh",
    "experiment/phase16/run_stage16_s4_toys_repeat_gpu4_a7_inner.sh",
)


if __name__ == "__main__":
    raise SystemExit(base.main())
