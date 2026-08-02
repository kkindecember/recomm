#!/usr/bin/env python3
"""Load a local CodeLlama checkpoint and retain a bounded CUDA cache while idle."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def gpu_snapshot(torch: Any, device: Any) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": int(torch.cuda.current_device()),
        "device_name": torch.cuda.get_device_name(device),
        "memory_allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
        "memory_reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
        "memory_free_mib": round(free / 1024**2, 2),
        "memory_total_mib": round(total / 1024**2, 2),
    }


def reserve_cuda_memory(torch: Any, target_mib: int, device: Any) -> None:
    target_bytes = target_mib * 1024**2
    reserved = torch.cuda.memory_reserved(device)
    free, _ = torch.cuda.mem_get_info(device)
    needed = max(0, target_bytes - reserved)
    if needed > free:
        raise RuntimeError(
            f"insufficient free CUDA memory: need {needed / 1024**2:.0f} MiB, "
            f"free {free / 1024**2:.0f} MiB"
        )
    if needed:
        reservation = torch.empty(needed, dtype=torch.uint8, device=device)
        del reservation
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reserve-gpu-memory-mib", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.reserve_gpu_memory_mib <= 0 or args.heartbeat_seconds <= 0:
        parser.error("reservation and heartbeat values must be positive")

    device = torch.device("cuda:0")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGHUP, request_stop)

    write_json_atomic(
        args.state,
        {
            "state": "reserving_cuda_cache",
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "reserve_mib": args.reserve_gpu_memory_mib,
        },
    )
    reserve_cuda_memory(torch, args.reserve_gpu_memory_mib, device)

    write_json_atomic(
        args.state,
        {
            "state": "loading_model",
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "reserve_mib": args.reserve_gpu_memory_mib,
            "gpu": gpu_snapshot(torch, device),
        },
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    torch.cuda.synchronize(device)

    while not stop_requested:
        state = {
            "state": "holding",
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "reserve_mib": args.reserve_gpu_memory_mib,
            "hf_home": os.environ.get("HF_HOME"),
            "transformers_version": transformers.__version__,
            "gpu": gpu_snapshot(torch, device),
        }
        write_json_atomic(args.state, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        for _ in range(args.heartbeat_seconds):
            if stop_requested:
                break
            time.sleep(1)

    write_json_atomic(
        args.state,
        {
            "state": "stopping",
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "reserve_mib": args.reserve_gpu_memory_mib,
            "gpu": gpu_snapshot(torch, device),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
