#!/usr/bin/env python3
"""Prepare and finalize isolated full-compute G-RIDGE stability cycles."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from experiment.phase16.protocol.gfull_objective_resource_sweep import sha256, utc_now, write_json


ROOT = Path(__file__).resolve().parents[3]
FORMAL_CONFIG = ROOT / "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f1.json"
FORMAL_OUTPUT = ROOT / "artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f1"


def verify_authoritative_completion() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    summary_path = FORMAL_OUTPUT / "summary.json"
    completion_path = FORMAL_OUTPUT / "authoritative_completion.json"
    if any(not path.is_file() or path.is_symlink() for path in (summary_path, completion_path)):
        raise ValueError("Authoritative formal completion artifacts are missing")
    summary_sha = sha256(summary_path)
    completion_sha = sha256(completion_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("verdict") != "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION"
        or completion.get("authoritative_stage_status") != "COMPLETED"
        or completion.get("authoritative_status_code")
        != "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION"
        or completion.get("summary_sha256") != summary_sha
        or summary.get("test_read") is not False
        or summary.get("validation_used") is not False
    ):
        raise ValueError("Authoritative formal completion is not an admissible stability parent")
    return summary, completion, summary_sha, completion_sha


def build_cycle_config(
    *,
    formal_config: Mapping[str, Any],
    cycle: int,
    queue_root: str,
    summary_sha: str,
    completion_sha: str,
) -> dict[str, Any]:
    if cycle < 1:
        raise ValueError("Stability cycle must be positive")
    config = copy.deepcopy(dict(formal_config))
    suffix = f"cycle_{cycle:04d}"
    config.update(
        {
            "schema_version": "stage16_s3r_gridge_stability_cycle_v1",
            "experiment_id": "GRAM_PHASE16_S3R_GRIDGE_STABILITY",
            "attempt_id": f"s16_s3r_gridge_stability_gpu5_c{cycle:04d}",
            "run_role": "stability_repeat",
            "output_dir": f"{queue_root}/{suffix}",
            "exact_start_command": (
                "bash experiment/phase16/run_stage16_s3r_gridge_stability_gpu5.sh "
                f"(planned cycle {cycle})"
            ),
        }
    )
    config["inputs"]["authoritative_formal_summary"] = {
        "path": str((FORMAL_OUTPUT / "summary.json").relative_to(ROOT)),
        "sha256": summary_sha,
    }
    config["inputs"]["authoritative_completion"] = {
        "path": str((FORMAL_OUTPUT / "authoritative_completion.json").relative_to(ROOT)),
        "sha256": completion_sha,
    }
    config["stability"] = {
        "cycle": cycle,
        "planned_repeat_queue": True,
        "full_reexecution": True,
        "authoritative_stage_status": "COMPLETED",
        "authoritative_status_code": "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION",
        "authoritative_summary_sha256": summary_sha,
        "authoritative_completion_sha256": completion_sha,
        "affects_scientific_results": False,
        "promotion_eligible": False,
        "formal_inputs_read_only": True,
        "stop_on_cycle_failure": True,
        "automatic_retry": False,
    }
    config["automatic_retry"] = False
    config["scientific_efficacy_metric"] = False
    return config


def prepare_cycle(cycle: int, queue_root: str) -> Path:
    _summary, _completion, summary_sha, completion_sha = verify_authoritative_completion()
    formal = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    config = build_cycle_config(
        formal_config=formal,
        cycle=cycle,
        queue_root=queue_root,
        summary_sha=summary_sha,
        completion_sha=completion_sha,
    )
    root = ROOT / queue_root
    config_path = root / "configs" / f"cycle_{cycle:04d}.json"
    output = ROOT / config["output_dir"]
    if config_path.exists() or output.exists():
        raise FileExistsError("Stability cycle config/output already exists; no retry or overwrite")
    write_json(config_path, config)
    return config_path


def finalize_cycle(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    raw_path = output / "formal_admission_summary.json"
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError("Refusing to overwrite completed stability cycle summary")
    if not raw_path.is_file() or raw_path.is_symlink():
        raise ValueError("Stability cycle raw computation is missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    stability = config.get("stability", {})
    if (
        config.get("run_role") != "stability_repeat"
        or raw.get("run_role") != "stability_repeat"
        or raw.get("verdict") != "STABILITY_CYCLE_COMPUTE_COMPLETE"
        or raw.get("formal_gate") != "NOT_APPLICABLE_STABILITY_REPEAT"
        or raw.get("authoritative_stage_status") != "COMPLETED"
        or raw.get("affects_scientific_results") is not False
        or raw.get("promotion_eligible") is not False
        or not raw.get("contract_checks")
        or not all(value is True for value in raw["contract_checks"].values())
        or stability.get("affects_scientific_results") is not False
        or stability.get("promotion_eligible") is not False
        or stability.get("formal_inputs_read_only") is not True
        or stability.get("automatic_retry") is not False
    ):
        raise ValueError("Stability cycle fail-closed contract did not pass")
    _summary, _completion, summary_sha, completion_sha = verify_authoritative_completion()
    if (
        summary_sha != stability.get("authoritative_summary_sha256")
        or completion_sha != stability.get("authoritative_completion_sha256")
    ):
        raise ValueError("Authoritative formal artifacts changed during stability cycle")
    positions = {
        position: {
            "request_count": row["request_count"],
            "valid_z_count": row["valid_z_count"],
            "failed_z_count": row["failed_z_count"],
            "regularized_condition": row["regularized_condition"],
            "solve_relative_residual": row["solve_relative_residual"],
            "delta_norm": row["delta_norm"],
        }
        for position, row in raw["position_diagnostics"].items()
    }
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "cycle": stability["cycle"],
        "generated_at_utc": utc_now(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "authoritative_stage_status": "COMPLETED",
        "authoritative_status_code": "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION",
        "authoritative_summary_sha256": summary_sha,
        "authoritative_completion_sha256": completion_sha,
        "affects_scientific_results": False,
        "promotion_eligible": False,
        "formal_inputs_read_only": True,
        "automatic_retry": False,
        "validation_used": False,
        "test_read": False,
    }
    write_json(
        summary_path,
        {
            **common,
            "status": "COMPLETED",
            "status_code": "STABILITY_CYCLE_COMPUTE_COMPLETE",
            "compute_mode": "full G-RIDGE reexecution with the authoritative workload",
            "full_universe": raw["full_universe"],
            "position_diagnostics": positions,
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
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "source": "read-only authoritative formal inputs and frozen Stage16 train-only artifacts",
            "result_use": "stability/occupancy evidence only; excluded from scientific result tables and promotion",
        },
    )
    write_json(
        output / "resource_summary.json", {**common, **raw["resource_summary"]}
    )
    write_json(
        output / "command_manifest.json",
        {**common, "exact_start_command": config["exact_start_command"]},
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
