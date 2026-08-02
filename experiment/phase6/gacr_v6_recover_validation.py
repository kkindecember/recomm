#!/usr/bin/env python3
"""Validation-only recovery for the interrupted GACR-v6 full-fit run.

This program never trains or overwrites a residual checkpoint.  It verifies the
six checkpoint hashes recorded after the interrupted run, constructs the frozen
fresh cohort, and writes a separate recovery result directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gcdh_p0 import ROOT, sha256, write_json  # noqa: E402
from experiment.phase6.gacr_v2 import (  # noqa: E402
    build_validation_records,
    method_result,
    serializable_rows,
    validate_checkpoint_lineage,
)
from experiment.phase6.gacr_v6 import (  # noqa: E402
    add_standard_metrics,
    compare_methods,
    load_v3_state,
)


def load_reused_state(recovery: dict, dataset: str, seed: int) -> dict:
    path = ROOT / recovery["reused_checkpoint_root"] / dataset / f"residual_seed{seed}.pt"
    expected = recovery["expected_reused_residual_sha256"][dataset][str(seed)]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"reused v6 residual SHA mismatch {dataset}/{seed}: "
            f"expected={expected}:actual={actual}"
        )
    return torch.load(path, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v6 recovery validation requires CUDA")

    recovery = json.loads(args.recovery_config.read_text())
    config_path = ROOT / recovery["original_preregistered_config"]
    if sha256(config_path) != recovery["original_preregistered_config_sha256"]:
        raise RuntimeError("original preregistered config SHA mismatch")
    config = json.loads(config_path.read_text())
    validate_checkpoint_lineage(config)
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")

    validation = {}
    reused = {}
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(dataset, config, p0_config, device)
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = {}
        reused[dataset] = {}
        for seed in config["training_seeds"]:
            v3_state = load_v3_state(config, dataset, int(seed))
            v6_state = load_reused_state(recovery, dataset, int(seed))
            reused[dataset][str(seed)] = sha256(
                ROOT / recovery["reused_checkpoint_root"] / dataset / f"residual_seed{seed}.pt"
            )
            v3_result, v3_rows = method_result(records, v3_state, config, 1.0, int(seed), device)
            v6_result, v6_rows = method_result(records, v6_state, config, 1.0, int(seed) + 1000, device)
            v3_result, v3_rows = add_standard_metrics(v3_result, v3_rows)
            v6_result, v6_rows = add_standard_metrics(v6_result, v6_rows)
            seeds[str(seed)] = {
                "gacr_v3": v3_result,
                "gacr_v6": v6_result,
                "incremental_v6_vs_v3": compare_methods(v3_rows, v6_rows),
            }
            for method, rows in (("gacr_v3", v3_rows), ("gacr_v6", v6_rows)):
                path = output_dir / f"{method}_seed{seed}_per_user.csv"
                serial = serializable_rows(rows)
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
                    writer.writeheader()
                    writer.writerows(serial)
                seeds[str(seed)][method]["per_user_sha256"] = sha256(path)
        validation[dataset] = metadata | {"seeds": seeds}
        del records
        torch.cuda.empty_cache()

    integrity = {
        "recovery_mode": "validation_only_reuse_completed_v6_residuals",
        "retrained_residuals": False,
        "reused_residual_sha256": reused,
        "parent_checkpoint_sha_unchanged": all(
            validation[d]["parent_checkpoint_sha256_before"]
            == validation[d]["parent_checkpoint_sha256_after"]
            for d in config["datasets"]
        ),
        "fresh_validation_zero_overlap": all(
            validation[d]["gcdh_or_training_overlap"] == 0
            and validation[d]["prior_gacr_p0_overlap"] == 0
            for d in config["datasets"]
        ),
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "recovery": recovery,
        "single_changed_factor": config["single_changed_factor"],
        "validation": validation,
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
