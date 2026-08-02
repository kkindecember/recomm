#!/usr/bin/env python3
"""Hold the non-workload portion of a cooperative GPU memory lease.

An experiment declares its expected peak.  This sidecar then reserves the
difference between that peak and the requested total lease before the workload
starts.  The workload and sidecar therefore jointly hold the requested capacity
without reserving that capacity twice.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import torch


def write_status(path: Path, state: str, **extra: int | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state": state, **extra}, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--total-lease-mib", type=int, default=30720)
    parser.add_argument("--expected-workload-peak-mib", type=int, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU memory lease requires CUDA")
    if not 0 < args.expected_workload_peak_mib <= args.total_lease_mib:
        raise ValueError("expected workload peak must be in (0, total lease]")

    reserve_mib = args.total_lease_mib - args.expected_workload_peak_mib
    torch.cuda.set_device(args.gpu)
    # float32 keeps the allocation size exact in MiB.  The reference remains
    # live until the runner terminates this sidecar.
    held = torch.empty(reserve_mib * 1024 * 1024 // 4, device=f"cuda:{args.gpu}")
    torch.cuda.synchronize(args.gpu)
    write_status(
        args.status_path,
        "holding",
        gpu=args.gpu,
        total_lease_mib=args.total_lease_mib,
        expected_workload_peak_mib=args.expected_workload_peak_mib,
        sidecar_reserved_mib=reserve_mib,
    )
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        time.sleep(1)
    del held
    torch.cuda.empty_cache()
    write_status(args.status_path, "released", gpu=args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
