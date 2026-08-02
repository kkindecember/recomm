#!/usr/bin/env python3
"""Resource-audited entry point for the frozen ST-GCGD-v2.1 P0-G2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase7.st_gcgd_v2 import write_json
from experiment.phase7.st_gcgd_v21 import run_p0_g2, validate_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P0-G2 requires CUDA_VISIBLE_DEVICES=0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    summary = run_p0_g2(args.dataset, config, args.output_root)
    torch.cuda.synchronize(device)
    actual = int(torch.cuda.max_memory_reserved(device) / (1024 * 1024))
    budget = int(config["execution"]["domain_gpu_lease_mib"][args.dataset]["workload_budget_mib"])
    sidecar = int(config["execution"]["domain_gpu_lease_mib"][args.dataset]["sidecar_mib"])
    tolerance = int(config["execution"]["measurement_tolerance_mib"])
    resource_pass = abs(actual - budget) <= tolerance and budget + sidecar == 30720
    summary["resource_audit"] = {
        "actual_workload_peak_reserved_mib": actual,
        "frozen_workload_budget_mib": budget,
        "sidecar_mib": sidecar,
        "declared_total_mib": budget + sidecar,
        "measurement_tolerance_mib": tolerance,
        "passed": resource_pass,
    }
    if not resource_pass:
        summary["status"] = "RESOURCE_LEASE_MISMATCH"
    write_json(args.output_root / args.dataset / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if resource_pass else 9


if __name__ == "__main__":
    raise SystemExit(main())
