#!/usr/bin/env python3
"""Resource-only GRAM microbatch escalation profile for formal attempt_002."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.full_latte_profile_executor import run_resource_profile
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as profile_base


ARM_ID = "G2_GRAM_LATTE_FULL"
DEFAULT_PHYSICAL_GPU = 5
MARGIN_MIB = 3072


def experiment_id(microbatch: int, training_only: bool = False) -> str:
    kind = "microbatch_training_profile" if training_only else "microbatch_profile"
    return f"s17_fp12_{kind}_g2_mb{microbatch}"


def profile_spec(microbatch: int, physical_gpu: int) -> profile_base.ProfileSpec:
    return profile_base.ProfileSpec(
        ARM_ID,
        "S17-FP2",
        physical_gpu,
        49140,
        0,
        microbatch,
        1,
        "gram",
    )


def run(
    root: Path,
    microbatch: int,
    attempt_id: str,
    physical_gpu: int,
    training_only: bool,
) -> int:
    root = root.resolve()
    exp_id = experiment_id(microbatch, training_only)
    profile_kind = "microbatch_training_upscale" if training_only else "microbatch_upscale"
    result = (
        root
        / "artifacts/phase17/fullport/profiles/g2_gram_latte_full"
        / f"{profile_kind}/mb{microbatch}/{attempt_id}"
    )
    status_dir = root / "artifacts/phase17/status"
    status_path = status_dir / f"{exp_id}.status.json"
    if result.exists():
        raise FileExistsError(f"microbatch profile already exists: {result}")
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("scientific_state") not in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError("previous microbatch profile status is not terminal")
        if previous.get("attempt_id") == attempt_id:
            raise FileExistsError(f"microbatch profile attempt already exists: {attempt_id}")
    result.mkdir(parents=True, exist_ok=False)
    writer = StatusWriter(status_dir, exp_id)
    writer.initialize(
        step_id="S17-FP2",
        attempt_id=attempt_id,
        track_id=f"G2_MICROBATCH_{microbatch}",
        canonical_result_dir=str(result.relative_to(root)),
        log_path=None,
        extra={
            "stage": "microbatch_profile_ready",
            "progress": {"current": 0, "total": 1, "unit": "profile"},
            "gpu_ids": [physical_gpu],
            "target_gpu_id": physical_gpu,
            "train_microbatch": microbatch,
            "gradient_accumulation": 128 // microbatch,
            "effective_batch": 128,
            "resource_only": True,
            "training_only": training_only,
            "result_selection_eligible": False,
            "affects_scientific_result": False,
            "external_target_materialized": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_MICROBATCH_RESOURCE_PROFILE_READY",
        stage="microbatch_profile_preflight_complete",
        process_alive=False,
    )
    before = profile_base.gpu_snapshot_once(profile_spec(microbatch, physical_gpu))
    writer.transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_FP12_MICROBATCH_RESOURCE_PROFILE_RUNNING",
        stage="training_and_primary_beam_resource_profile",
        process_alive=True,
        workload_pid=os.getpid(),
        gpu_snapshot={"before": before},
    )
    started = time.monotonic()
    try:
        measured = run_resource_profile(
            root,
            ARM_ID,
            train_batch_size=microbatch,
            eval_batch_size=1,
            include_primary_generation=not training_only,
            heartbeat=lambda stage, progress: writer.heartbeat(
                stage=stage, progress=progress
            ),
        )
        after = profile_base.gpu_snapshot_once(profile_spec(microbatch, physical_gpu))
        peak = float(measured["peak_reserved_mib"])
        summary = {
            "schema_version": "phase17.s17_fp12_microbatch_profile.v1",
            "verdict": "PASS_S17_FP12_MICROBATCH_RESOURCE_PROFILE",
            "completed_at": utc_now(),
            "arm_id": ARM_ID,
            "physical_gpu": physical_gpu,
            "train_microbatch": microbatch,
            "gradient_accumulation": 128 // microbatch,
            "effective_batch": 128,
            "measured": measured,
            "profile_peak_reserved_mib": peak,
            "recommended_minimum_free_mib": round(peak + MARGIN_MIB),
            "margin_mib": MARGIN_MIB,
            "wall_seconds": time.monotonic() - started,
            "gpu_snapshot": {"before": before, "after": after},
            "resource_only": True,
            "training_only": training_only,
            "result_selection_eligible": False,
            "affects_scientific_result": False,
            "external_target_materialized": False,
            "automatic_retry": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        atomic_json(result / "summary.json", summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP12_MICROBATCH_RESOURCE_PROFILE",
            stage="microbatch_profile_complete",
            process_alive=False,
            workload_pid=0,
            gpu_ids=[],
            progress={"current": 1, "total": 1, "unit": "profile"},
            summary_path=str((result / "summary.json").relative_to(root)),
            profile_peak_reserved_mib=peak,
            recommended_minimum_free_mib=round(peak + MARGIN_MIB),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "train_microbatch": microbatch,
            "automatic_retry": False,
            "resource_only": True,
            "result_selection_eligible": False,
            "affects_scientific_result": False,
            "external_target_materialized": False,
        }
        atomic_json(result / "failure.json", failure)
        writer.transition(
            "FAILED",
            "SCIENTIFIC_FAILED",
            "S17_FP12_MICROBATCH_RESOURCE_PROFILE_FAILED_NO_RETRY",
            stage="microbatch_profile_terminal_failure_no_retry",
            process_alive=False,
            workload_pid=0,
            gpu_ids=[],
            terminal_error=repr(error),
            automatic_retry=False,
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


def finalize_infrastructure_failure(root: Path, microbatch: int) -> int:
    root = root.resolve()
    exp_id = experiment_id(microbatch)
    writer = StatusWriter(root / "artifacts/phase17/status", exp_id)
    status = writer.read()
    if status.get("attempt_id") != "attempt_001" or status.get("scientific_state") != "PENDING":
        raise RuntimeError("expected the pre-GPU attempt_001 PENDING status")
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_MICROBATCH_PROFILE_INFRASTRUCTURE_FAILURE",
        stage="pre_gpu_status_transition_failure",
        process_alive=False,
    )
    final = writer.transition(
        "FAILED",
        "SCIENTIFIC_FAILED",
        "S17_FP12_MICROBATCH_PROFILE_INFRASTRUCTURE_FAILED_NO_GPU_WORKLOAD",
        stage="terminal_infrastructure_failure_no_gpu_workload",
        process_alive=False,
        workload_pid=0,
        gpu_ids=[],
        gpu_workload_started=False,
        automatic_retry=False,
        result_selection_eligible=False,
        affects_scientific_result=False,
        terminal_error="illegal status transition PENDING -> RUNNING",
    )
    result = (
        root
        / "artifacts/phase17/fullport/profiles/g2_gram_latte_full"
        / f"microbatch_upscale/mb{microbatch}/attempt_001"
    )
    atomic_json(
        result / "infrastructure_failure.json",
        {
            "schema_version": "phase17.infrastructure_failure.v1",
            "failed_at": utc_now(),
            "error": "illegal status transition PENDING -> RUNNING",
            "gpu_workload_started": False,
            "scientific_profile_started": False,
            "automatic_retry": False,
        },
    )
    atomic_json(result / "final_status.json", final)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--attempt-id", choices=("attempt_001", "attempt_002"), default="attempt_001")
    parser.add_argument("--finalize-infrastructure-failure", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--physical-gpu", type=int, choices=tuple(range(8)), default=DEFAULT_PHYSICAL_GPU)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if 128 % args.microbatch:
        raise ValueError("microbatch must divide effective batch 128")
    if args.finalize_infrastructure_failure:
        return finalize_infrastructure_failure(args.root, args.microbatch)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"expected CUDA_VISIBLE_DEVICES={args.physical_gpu}, observed {visible!r}"
        )
    return run(
        args.root,
        args.microbatch,
        args.attempt_id,
        args.physical_gpu,
        args.training_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
