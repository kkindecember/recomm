#!/usr/bin/env python3
"""Authorization-gated four-arm FP3 Full SETRec resource profile."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_setrec_backend import SETREC_ARMS
from experiment.phase17.core.full_setrec_executor import (
    GLOBAL_BATCH_SIZE,
    PROFILE_EVAL_BATCH_BY_ARM,
    PROFILE_MICROBATCH_BY_ARM,
    cpu_preflight_setrec,
    run_setrec_resource_profile,
)
from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    verify_run_snapshot,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import (
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)
from experiment.phase17.protocol import s17_fp3_setrec_tokenizer_runtime as tokenizer_runtime


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SOURCE = Path(__file__).resolve()
EXPERIMENT_ID = "s17_fp3_setrec_resource_profiles"
STEP_ID = "S17-FP3-PROFILE"
ATTEMPT_ID = "attempt_001"
TMUX_SESSION = EXPERIMENT_ID
TARGET_GPU_ID = 7
MINIMUM_FREE_MIB = 24576
SAFETY_MARGIN_MIB = 4096
TOKENIZER_STATUS_CODE = "PASS_S17_FP3_SETREC_CF_TOKENIZER"
NATIVE_PYTHON_SUFFIX = tokenizer_runtime.NATIVE_PYTHON_SUFFIX
CONFIRMED_COMMAND = (
    "bash experiment/phase17/run_stage17_fp3_setrec_resource_profiles.sh launch"
)


def paths(root: Path) -> dict[str, Path]:
    result = root / f"artifacts/phase17/fullport/fp3_setrec/profiles/{ATTEMPT_ID}"
    snapshot = root / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json"
    return {
        "result": result,
        "config": result / "config.json",
        "cpu_preflight": result / "cpu_preflight.json",
        "partial": result / "profiles.partial.json",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "authorization": root
        / f"artifacts/phase17/authorizations/{EXPERIMENT_ID}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP3.attempts.jsonl",
        "snapshot": snapshot,
        "snapshot_worker": snapshot.parent / f"src/000_{RUNTIME_SOURCE.name}",
        "native_python": root / NATIVE_PYTHON_SUFFIX,
        "fp3_config": root / "experiment/phase17/config/s17_fp3_setrec.json",
        "tokenizer_status": root
        / "artifacts/phase17/status/s17_fp3_setrec_cf_tokenizer.status.json",
        "tokenizer_summary": root
        / "artifacts/phase17/fullport/fp3_setrec/tokenizer/attempt_001/summary.json",
        "tokenizer_artifact": root
        / "artifacts/phase17/fullport/fp3_setrec/tokenizer/attempt_001/sasrec_item_embeddings.pt",
    }


def verify_tokenizer(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    status = json.loads(resolved["tokenizer_status"].read_text(encoding="utf-8"))
    summary = json.loads(resolved["tokenizer_summary"].read_text(encoding="utf-8"))
    if (
        status.get("scientific_state") != "COMPLETED"
        or status.get("status_code") != TOKENIZER_STATUS_CODE
        or summary.get("verdict") != TOKENIZER_STATUS_CODE
        or summary.get("catalog_items") != 11924
    ):
        raise RuntimeError("FP3 CF tokenizer evidence is not PASS")
    if sha256(resolved["tokenizer_artifact"]) != summary.get("artifact_sha256"):
        raise RuntimeError("FP3 CF tokenizer artifact hash drifted")
    return {
        "status_sha256": sha256(resolved["tokenizer_status"]),
        "summary_sha256": sha256(resolved["tokenizer_summary"]),
        "artifact_sha256": sha256(resolved["tokenizer_artifact"]),
    }


def gpu_admission() -> dict[str, Any]:
    first = tokenizer_runtime._gpu_state()
    time.sleep(5)
    second = tokenizer_runtime._gpu_state()
    for name, snapshot in (("first", first), ("second", second)):
        selected = [
            row for row in snapshot["devices"] if row["index"] == TARGET_GPU_ID
        ]
        if len(selected) != 1:
            raise RuntimeError(f"GPU{TARGET_GPU_ID} not uniquely visible")
        if selected[0]["free_mib"] < MINIMUM_FREE_MIB:
            raise RuntimeError(
                f"GPU{TARGET_GPU_ID} {name} free {selected[0]['free_mib']} MiB "
                f"is below {MINIMUM_FREE_MIB} MiB"
            )
    return {
        "selected_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "first": first,
        "second": second,
        "preexisting_compute_processes": second["compute_processes"].get(
            TARGET_GPU_ID, []
        ),
        "automatic_process_termination": False,
    }


def frozen_config(root: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    resolved = paths(root)
    return {
        "schema_version": "phase17.s17_fp3_setrec_resource_profile_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "step_id": STEP_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "arms": list(SETREC_ARMS),
        "profile_order": list(SETREC_ARMS),
        "train_microbatch_by_arm": PROFILE_MICROBATCH_BY_ARM,
        "eval_batch_by_arm": PROFILE_EVAL_BATCH_BY_ARM,
        "effective_global_batch": GLOBAL_BATCH_SIZE,
        "precision": "fp16_autocast_with_fp32_forward_parity_gate",
        "profile_workload": "one_worst_case_train_step_plus_one_internal_dev_eval_step",
        "cpu_preflight": preflight,
        "dependencies": verify_tokenizer(root),
        "fp3_config_sha256": sha256(resolved["fp3_config"]),
        "formal_effect_experiment_authorized": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


def worker_command(root: Path, resolved: dict[str, Path]) -> list[str]:
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={TARGET_GPU_ID}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(resolved["native_python"]),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def prepare(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    if resolved["result"].exists() or resolved["snapshot"].exists():
        raise FileExistsError("FP3 resource profile attempt_001 already exists")
    verify_tokenizer(root)
    preflight = cpu_preflight_setrec(root)
    resolved["result"].mkdir(parents=True, exist_ok=False)
    atomic_json(resolved["cpu_preflight"], preflight)
    config = frozen_config(root, preflight)
    atomic_json(resolved["config"], config)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=worker_command(root, resolved),
        source_paths=[
            RUNTIME_SOURCE,
            Path(__file__).resolve(),
            root / "experiment/phase17/core/full_setrec_executor.py",
            root / "experiment/phase17/core/full_setrec_backend.py",
            root / "experiment/phase17/core/full_setrec_contracts.py",
            root / "experiment/phase17/core/full_setrec_cf_tokenizer.py",
            root / "experiment/phase17/core/full_latte_gram_backend.py",
            root / "experiment/phase17/core/full_latte_native_adapter.py",
            root / "experiment/phase17/core/fullport_data.py",
            root / "experiment/phase17/core/run_manager.py",
            root / "experiment/phase17/core/status_writer.py",
            root
            / "experiment/phase17/protocol/s17_fp3_setrec_tokenizer_runtime.py",
        ],
        config=config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": f"profiles_{ATTEMPT_ID}",
            "experiment_id": EXPERIMENT_ID,
            "step_id": STEP_ID,
            "kind": "four_arm_resource_profile",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_EXACT_COMMAND_CONFIRMATION_REQUIRED",
            "scientific_result_eligible": False,
            "automatic_retry": False,
            "gpu_ids": [],
            "snapshot_manifest": str(manifest.relative_to(root)),
        }
    )
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id=STEP_ID,
        attempt_id=ATTEMPT_ID,
        track_id="FP3-PROFILE",
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "cpu_preflight_complete_waiting_exact_command_confirmation",
            "progress": {"current": 0, "total": 4, "unit": "arm"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "gpu_ids": [],
            "target_gpu_id": TARGET_GPU_ID,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "launch_authorized": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "external_target_materialized": False,
            "d1_read": False,
            "d2_read": False,
            "result_selection_eligible": False,
            "affects_scientific_result": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP3_RESOURCE_PROFILES_READY_EXACT_COMMAND_CONFIRMATION_REQUIRED",
        process_alive=False,
    )
    print(manifest)
    return 0


def authorize(root: Path, researcher_direction: str) -> int:
    root = root.resolve()
    resolved = paths(root)
    if not researcher_direction.strip():
        raise ValueError("profile authorization requires researcher direction")
    if resolved["authorization"].exists():
        raise FileExistsError(resolved["authorization"])
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("FP3 resource profile is not in PREFLIGHT")
    snapshot = tokenizer_runtime._gpu_state()
    selected = [row for row in snapshot["devices"] if row["index"] == TARGET_GPU_ID]
    if len(selected) != 1 or selected[0]["free_mib"] < MINIMUM_FREE_MIB:
        raise RuntimeError("GPU7 lacks FP3 resource-profile headroom")
    payload = {
        "schema_version": "phase17.s17_fp3_setrec_resource_profile_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "authorized": True,
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "researcher_direction": researcher_direction,
        "confirmed_exact_command": CONFIRMED_COMMAND,
        "resource_profile_only": True,
        "formal_effect_experiment_authorized": False,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "gpu_snapshot": snapshot,
    }
    atomic_json(resolved["authorization"], payload)
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP3_RESOURCE_PROFILES_AUTHORIZED_READY_TO_LAUNCH",
        launch_authorized=True,
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        gpu_snapshot=snapshot,
    )
    print(resolved["authorization"])
    return 0


def verify_authorization(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    payload = json.loads(resolved["authorization"].read_text(encoding="utf-8"))
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "authorized": True,
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "resource_profile_only": True,
        "formal_effect_experiment_authorized": False,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"invalid FP3 profile authorization field: {key}")
    if not payload.get("researcher_direction"):
        raise PermissionError("FP3 profile authorization lacks researcher direction")
    return payload


def launch(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("FP3 resource profile is not launchable")
    verify_authorization(root)
    admission = gpu_admission()
    session = launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=worker_command(root, resolved),
        cwd=root,
        tmux_session=TMUX_SESSION,
        startup_log_path=resolved["log"],
    )
    writer.transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP3_RESOURCE_PROFILES_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="background_started",
        gpu_ids=[TARGET_GPU_ID],
        gpu_snapshot=admission,
        launch_authorized=True,
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP3_RESOURCE_PROFILES_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
            )
        raise RuntimeError("FP3 resource profile exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, manifest_path: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    started = time.monotonic()
    try:
        verify_run_snapshot(root, manifest_path)
        verify_authorization(root)
        verify_tokenizer(root)
        admission = gpu_admission()
        import torch

        torch.cuda.set_device(0)
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP3_RESOURCE_PROFILES_RUNNING",
            workload_pid=os.getpid(),
            process_alive=True,
            stage="profiling_s0",
            gpu_ids=[TARGET_GPU_ID],
            gpu_snapshot=admission,
        )
        profiles: dict[str, Any] = {}
        for index, arm_id in enumerate(SETREC_ARMS, 1):
            writer.heartbeat(
                stage=f"profiling_{arm_id.lower()}",
                progress={"current": index - 1, "total": 4, "unit": "arm"},
            )
            profiles[arm_id] = run_setrec_resource_profile(
                root, arm_id, device=torch.device("cuda")
            )
            atomic_json(
                resolved["partial"],
                {
                    "schema_version": "phase17.s17_fp3_profiles_partial.v1",
                    "updated_at": utc_now(),
                    "profiles": profiles,
                },
            )
            writer.heartbeat(
                stage=f"profiled_{arm_id.lower()}",
                progress={"current": index, "total": 4, "unit": "arm"},
            )
        parameter_counts = {row["parameter_count"] for row in profiles.values()}
        if len(parameter_counts) != 1:
            raise RuntimeError("FP3 profile parameter-count parity failed")
        summary = {
            "schema_version": "phase17.s17_fp3_setrec_resource_profile_summary.v1",
            "verdict": "PASS_S17_FP3_SETREC_RESOURCE_PROFILES",
            "completed_at": utc_now(),
            "physical_gpu": TARGET_GPU_ID,
            "profiles": profiles,
            "formal_minimum_free_mib_by_arm": {
                arm_id: int(profile["peak_reserved_mib"] + SAFETY_MARGIN_MIB + 0.999)
                for arm_id, profile in profiles.items()
            },
            "recommended_formal_precision": "fp16",
            "matched_parameter_count": parameter_counts.pop(),
            "wall_seconds": time.monotonic() - started,
            "formal_effect_experiment_authorized": False,
            "automatic_process_termination": False,
            "automatic_retry": False,
            "external_target_materialized": False,
            "effect_metrics_computed": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
            "next_gate": "S17_FP3_FORMAL_FOUR_GPU_ALLOCATION_REQUEST",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP3_SETREC_RESOURCE_PROFILES",
            process_alive=False,
            workload_pid=0,
            stage="four_arm_profiles_complete",
            progress={"current": 4, "total": 4, "unit": "arm"},
            gpu_ids=[],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            formal_effect_experiment_authorized=False,
            result_selection_eligible=False,
            affects_scientific_result=False,
        )
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "automatic_process_termination": False,
            "formal_effect_experiment_authorized": False,
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        atomic_json(resolved["failure"], failure)
        with resolved["log"].open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] terminal_error={error!r}\n")
            handle.write(failure["traceback"])
        current = writer.read()
        if current["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP3_RESOURCE_PROFILES_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                stage="terminal_failure_no_retry",
                gpu_ids=[],
                failure_path=str(resolved["failure"].relative_to(root)),
                failure_sha256=sha256(resolved["failure"]),
                terminal_error=repr(error),
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
        return 1


def inspect(root: Path) -> int:
    resolved = paths(root.resolve())
    payload: dict[str, Any] = {}
    status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    for key, path in (
        ("status", status_path),
        ("partial", resolved["partial"]),
        ("summary", resolved["summary"]),
        ("failure", resolved["failure"]),
    ):
        if path.is_file():
            payload[key] = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "authorize", "launch", "worker", "inspect")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--researcher-direction", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root)
    if args.action == "authorize":
        return authorize(root, args.researcher_direction)
    if args.action == "launch":
        return launch(root)
    if args.action == "inspect":
        return inspect(root)
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
