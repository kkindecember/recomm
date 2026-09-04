#!/usr/bin/env python3
"""High-throughput revision of the FP3 four-arm resource profile."""

from __future__ import annotations

from pathlib import Path

from experiment.phase17.protocol import s17_fp3_setrec_resource_profile_runtime as base


base.EXPERIMENT_ID = "s17_fp3_setrec_resource_profiles_upscale"
base.ATTEMPT_ID = "attempt_002"
base.TMUX_SESSION = base.EXPERIMENT_ID
base.MINIMUM_FREE_MIB = 32768
base.RUNTIME_SOURCE = Path(__file__).resolve()
base.CONFIRMED_COMMAND = (
    "bash experiment/phase17/run_stage17_fp3_setrec_resource_profiles_upscale.sh launch"
)


if __name__ == "__main__":
    raise SystemExit(base.main())
