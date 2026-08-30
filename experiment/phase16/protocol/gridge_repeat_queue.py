#!/usr/bin/env python3
"""Prepare/finalize non-promotional G-RIDGE repeats after any f2 terminal state."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from experiment.phase16.protocol.gfull_objective_resource_sweep import (
    sha256,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[3]
FORMAL_CONFIG_REL = os.environ.get(
    "PHASE16_FORMAL_CONFIG",
    "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f2.json",
)
FORMAL_STATUS_REL = os.environ.get(
    "PHASE16_FORMAL_STATUS_REL",
    "artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2/status.json",
)
FORMAL_ATTEMPT_ID = os.environ.get(
    "PHASE16_FORMAL_ATTEMPT_ID", "s16_s3r_gridge_formal_gpu5_f2"
)
REPEAT_ATTEMPT_PREFIX = os.environ.get(
    "PHASE16_REPEAT_ATTEMPT_PREFIX", "s16_s3r_gridge_repeat_gpu5_f2"
)
REPEAT_EXACT_COMMAND = os.environ.get(
    "PHASE16_REPEAT_EXACT_COMMAND",
    "bash experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2.sh",
)
REPEAT_CONFIG_ROOT_REL = os.environ.get(
    "PHASE16_REPEAT_CONFIG_ROOT_REL",
    ".runtime/phase16_s3r_gridge_f2_repeat_configs",
)
FORMAL_CONFIG = ROOT / FORMAL_CONFIG_REL
FORMAL_STATUS = ROOT / FORMAL_STATUS_REL
TERMINAL_STATES = {"COMPLETED", "FAILED", "BLOCKED", "TIMEOUT", "KILLED_TARGET_LEAKAGE"}


def read_formal_terminal(path: Path = FORMAL_STATUS) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("formal terminal status is unavailable")
    status = json.loads(path.read_text(encoding="utf-8"))
    if (
        status.get("attempt_id") != FORMAL_ATTEMPT_ID
        or status.get("status") not in TERMINAL_STATES
        or status.get("process_alive") is not False
        or status.get("test_read") is not False
        or status.get("validation_used") is not False
    ):
        raise ValueError("Repeat queue requires a sealed formal terminal status")
    return status


def build_cycle_config(
    *,
    formal_config: Mapping[str, Any],
    formal_status: Mapping[str, Any],
    cycle: int,
    queue_root: str,
) -> dict[str, Any]:
    if cycle < 1:
        raise ValueError("Repeat cycle must be positive")
    config = copy.deepcopy(dict(formal_config))
    suffix = f"cycle_{cycle:04d}"
    config.update(
        {
            "schema_version": "stage16_s3r_gridge_nonpromotional_repeat_v1",
            "experiment_id": "GRAM_PHASE16_S3R_GRIDGE_NONPROMOTIONAL_REPEAT",
            "attempt_id": f"{REPEAT_ATTEMPT_PREFIX}_c{cycle:04d}",
            "run_role": "stability_repeat",
            "output_dir": f"{queue_root}/{suffix}",
            "exact_start_command": (
                f"{REPEAT_EXACT_COMMAND} (planned independent cycle {cycle})"
            ),
        }
    )
    config["stability"] = {
        "cycle": cycle,
        "planned_independent_repetition": True,
        "full_reexecution": True,
        "launch_after_any_formal_terminal": True,
        "authoritative_stage_status": formal_status["status"],
        "authoritative_status_code": formal_status["status_code"],
        "formal_parent_attempt_id": formal_status["attempt_id"],
        "affects_scientific_results": False,
        "promotion_eligible": False,
        "formal_inputs_read_only": True,
        "normal_experiment_priority": True,
        "new_gpu5_process_causes_repeat_cycle_yield": True,
        "continue_after_cycle_failure_by_user_authorization": True,
        "automatic_retry": False,
    }
    config["automatic_retry"] = False
    config["scientific_efficacy_metric"] = False
    return config


def prepare_cycle(cycle: int, queue_root: str) -> Path:
    formal_status = read_formal_terminal()
    formal_config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    config = build_cycle_config(
        formal_config=formal_config,
        formal_status=formal_status,
        cycle=cycle,
        queue_root=queue_root,
    )
    config_root = ROOT / REPEAT_CONFIG_ROOT_REL
    config_path = config_root / f"cycle_{cycle:04d}.json"
    output = ROOT / config["output_dir"]
    if config_path.exists() or output.exists():
        raise FileExistsError("Repeat cycle path already exists; no overwrite/retry")
    write_json(config_path, config)
    return config_path


def finalize_cycle(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    raw_path = output / "formal_admission_summary.json"
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError("Refusing to overwrite repeat summary")
    if not raw_path.is_file() or raw_path.is_symlink():
        raise ValueError("Repeat raw computation is missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    repeat = config.get("stability", {})
    checks = raw.get("contract_checks")
    if (
        config.get("run_role") != "stability_repeat"
        or raw.get("run_role") != "stability_repeat"
        or raw.get("verdict") != "STABILITY_CYCLE_COMPUTE_COMPLETE"
        or raw.get("formal_gate") != "NOT_APPLICABLE_STABILITY_REPEAT"
        or raw.get("affects_scientific_results") is not False
        or raw.get("promotion_eligible") is not False
        or raw.get("authoritative_stage_status")
        != repeat.get("authoritative_stage_status")
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or repeat.get("affects_scientific_results") is not False
        or repeat.get("promotion_eligible") is not False
        or repeat.get("formal_inputs_read_only") is not True
        or repeat.get("normal_experiment_priority") is not True
        or repeat.get("automatic_retry") is not False
    ):
        raise ValueError("Non-promotional repeat contract failed closed")
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "cycle": repeat["cycle"],
        "generated_at_utc": utc_now(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "formal_parent_attempt_id": repeat["formal_parent_attempt_id"],
        "formal_parent_status": repeat["authoritative_stage_status"],
        "formal_parent_status_code": repeat["authoritative_status_code"],
        "affects_scientific_results": False,
        "promotion_eligible": False,
        "normal_experiment_priority": True,
        "automatic_retry": False,
        "validation_used": False,
        "test_read": False,
    }
    write_json(
        summary_path,
        {
            **common,
            "status": "COMPLETED",
            "status_code": "NONPROMOTIONAL_REPEAT_COMPUTE_COMPLETE",
            "compute_mode": "full G-RIDGE reexecution isolated from the formal Gate",
            "full_universe": raw["full_universe"],
            "position_diagnostics": {
                position: {
                    key: row[key]
                    for key in (
                        "request_count",
                        "valid_z_count",
                        "failed_z_count",
                        "regularized_condition",
                        "solve_relative_residual",
                        "delta_norm",
                    )
                }
                for position, row in raw["position_diagnostics"].items()
            },
            "item_disjoint_admission_non_promotional": raw[
                "item_disjoint_admission_non_promotional"
            ],
            "warm_preservation_non_promotional": raw[
                "warm_preservation_non_promotional"
            ],
            "resource_summary": raw["resource_summary"],
            "raw_path": str(raw_path.relative_to(ROOT)),
            "raw_sha256": sha256(raw_path),
        },
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cycle", type=int, required=True)
    prepare.add_argument("--queue-root", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_cycle(args.cycle, args.queue_root).relative_to(ROOT))
    else:
        print(finalize_cycle(args.config).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
