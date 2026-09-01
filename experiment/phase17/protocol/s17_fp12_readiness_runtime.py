#!/usr/bin/env python3
"""Read-only readiness gate for FP1/FP2 arm-specific profiles and formal runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_latte_arm_contracts import (
    ARM_IDS,
    load_and_validate_arm_matrix,
)
from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import utc_now


ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_EXPERIMENT_ID = "s17_fp0_full_data_tokenizer"
TOKENIZER_PASS_CODE = "PASS_S17_FP0_FULL_DATA_TOKENIZER"
PROFILE_READY_CODE = "S17_FP12_RESOURCE_PROFILE_READY_AUTHORIZATION_REQUIRED"


def _profile_experiment_id(arm_id: str) -> str:
    return f"s17_fp12_profile_{arm_id.lower()}"


def paths(root: Path) -> dict[str, Path]:
    return {
        "matrix": root / "experiment/phase17/config/s17_fp12_latte_arm_matrix.json",
        "allocation": root / "experiment/phase17/config/s17_fp_resource_allocation.json",
        "tokenizer_status": root
        / f"artifacts/phase17/status/{TOKENIZER_EXPERIMENT_ID}.status.json",
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_contract(arm_id: str, physical_gpu: int) -> dict[str, Any]:
    native = arm_id.startswith("N")
    peak_cap = 16384 if native else 20480
    minimum_free = 19456 if native else 23552
    return {
        "arm_id": arm_id,
        "physical_gpu": physical_gpu,
        "peak_reserved_cap_mib": peak_cap,
        "minimum_free_mib": minimum_free,
        "safety_margin_mib": 3072,
        "external_target_materialized": False,
        "effect_metrics_forbidden": True,
        "profile_adjustments_allowed": [
            "reduce_microbatch_and_increase_accumulation_preserving_effective_batch",
            "reduce_eval_batch_preserving_beam_topk_aggregation",
            "matched_activation_checkpointing_only",
        ],
        "automatic_retry": False,
        "automatic_process_termination": False,
        "launch_authorized": False,
    }


def inspect_readiness(root: Path) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    matrix = load_and_validate_arm_matrix(resolved["matrix"])
    allocation = _read(resolved["allocation"])
    blockers: list[dict[str, str]] = []
    tokenizer_status: dict[str, Any] | None = None
    if not resolved["tokenizer_status"].is_file():
        blockers.append(
            {
                "code": "BLOCKED_FULL_DATA_TOKENIZER_STATUS_MISSING",
                "resolution": "prepare and complete the authorized full-data tokenizer",
            }
        )
    else:
        tokenizer_status = _read(resolved["tokenizer_status"])
        if (
            tokenizer_status["scientific_state"] != "COMPLETED"
            or tokenizer_status["status_code"] != TOKENIZER_PASS_CODE
        ):
            blockers.append(
                {
                    "code": "BLOCKED_FULL_DATA_TOKENIZER_NOT_COMPLETE",
                    "resolution": "obtain explicit GPU0 authorization and complete tokenizer attempt",
                }
            )
        else:
            manifest_path = root / tokenizer_status["tokenizer_manifest_path"]
            if sha256(manifest_path) != tokenizer_status["tokenizer_manifest_sha256"]:
                blockers.append(
                    {
                        "code": "BLOCKED_FULL_DATA_TOKENIZER_MANIFEST_DRIFT",
                        "resolution": "audit tokenizer artifact integrity before any profile",
                    }
                )
    if allocation["arm_specific_resource_profiles"]["formal_launch_authorized"]:
        blockers.append(
            {
                "code": "INVALID_PREAUTHORIZED_RESOURCE_PROFILE",
                "resolution": "profiles require attempt-specific authorization",
            }
        )
    if allocation["formal_launch_authorized"]:
        blockers.append(
            {
                "code": "INVALID_PREAUTHORIZED_FORMAL_RUN",
                "resolution": "formal runs require profile evidence and a new researcher handoff",
            }
        )
    gpu_by_arm = allocation["arm_specific_resource_profiles"]["physical_gpu_by_arm"]
    if set(gpu_by_arm) != set(ARM_IDS):
        blockers.append(
            {
                "code": "BLOCKED_ARM_GPU_ALLOCATION_INCOMPLETE",
                "resolution": "freeze all five arm-to-GPU assignments",
            }
        )
    profiles = {
        arm_id: _profile_contract(arm_id, int(gpu_by_arm[arm_id]))
        for arm_id in ARM_IDS
        if arm_id in gpu_by_arm
    }
    profile_preparation: dict[str, Any] = {}
    prepared_arms: list[str] = []
    for arm_id in ARM_IDS:
        status_path = (
            root
            / f"artifacts/phase17/status/{_profile_experiment_id(arm_id)}.status.json"
        )
        if not status_path.is_file():
            profile_preparation[arm_id] = {"prepared": False, "status_path": None}
            continue
        status = _read(status_path)
        prepared = (
            status.get("scientific_state") == "PREFLIGHT"
            and status.get("execution_state") == "PREFLIGHT"
            and status.get("status_code") == PROFILE_READY_CODE
            and status.get("launch_authorized") is False
        )
        if prepared:
            prepared_arms.append(arm_id)
        profile_preparation[arm_id] = {
            "prepared": prepared,
            "scientific_state": status.get("scientific_state"),
            "execution_state": status.get("execution_state"),
            "status_code": status.get("status_code"),
            "launch_authorized": status.get("launch_authorized"),
            "status_path": str(status_path.relative_to(root)),
            "status_sha256": sha256(status_path),
        }
    all_profiles_prepared = set(prepared_arms) == set(ARM_IDS)
    if blockers:
        state = blockers[0]["code"]
    elif all_profiles_prepared:
        state = "READY_FOR_ARM_SPECIFIC_PROFILE_AUTHORIZATION"
    else:
        state = "READY_TO_IMPLEMENT_AND_PREPARE_PROFILES"
    return {
        "schema_version": "phase17.s17_fp12_readiness.v1",
        "captured_at": utc_now(),
        "state": state,
        "blockers": blockers,
        "arm_matrix_path": str(resolved["matrix"].relative_to(root)),
        "arm_matrix_sha256": sha256(resolved["matrix"]),
        "allocation_path": str(resolved["allocation"].relative_to(root)),
        "allocation_sha256": sha256(resolved["allocation"]),
        "tokenizer_status": (
            None
            if tokenizer_status is None
            else {
                "attempt_id": tokenizer_status["attempt_id"],
                "scientific_state": tokenizer_status["scientific_state"],
                "execution_state": tokenizer_status["execution_state"],
                "status_code": tokenizer_status["status_code"],
                "status_sha256": sha256(resolved["tokenizer_status"]),
            }
        ),
        "resource_profile_contracts": profiles,
        "resource_profile_preparation": profile_preparation,
        "prepared_profile_arms": prepared_arms,
        "all_profiles_prepared": all_profiles_prepared,
        "formal_wave_plan": allocation["fp1_fp2_formal"],
        "next_action": (
            "authorize_and_complete_full_data_tokenizer_attempt_001"
            if blockers
            and blockers[0]["code"] == "BLOCKED_FULL_DATA_TOKENIZER_NOT_COMPLETE"
            else (
                "freeze_current_gpu_pid_handoff_and_request_attempt_specific_profile_authorization"
                if all_profiles_prepared
                else "implement_and_prepare_arm_specific_resource_profile_executors"
            )
        ),
        "writes_performed": False,
        "gpu_used": False,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "effect_experiment_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(inspect_readiness(args.root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
