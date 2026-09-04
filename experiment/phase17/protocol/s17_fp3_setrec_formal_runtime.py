#!/usr/bin/env python3
"""Parallel four-GPU formal training runtime for Stage17 FP3 Full SETRec."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_setrec_backend import SETREC_ARMS
from experiment.phase17.core.full_setrec_executor import (
    PROFILE_EVAL_BATCH_BY_ARM,
    PROFILE_MICROBATCH_BY_ARM,
    SetRecFormalSpec,
    train_setrec_formal,
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
from experiment.phase17.protocol import s17_fp3_setrec_tokenizer_runtime as gpu_runtime


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "attempt_002"
SEED = 2023
STEP_ID = "S17-FP3-FORMAL"
NATIVE_PYTHON_SUFFIX = gpu_runtime.NATIVE_PYTHON_SUFFIX
PROFILE_EXPERIMENT_ID = "s17_fp3_setrec_resource_profiles_upscale"
PROFILE_STATUS_CODE = "PASS_S17_FP3_SETREC_RESOURCE_PROFILES"
PROFILE_SUMMARY_SUFFIX = Path(
    "artifacts/phase17/fullport/fp3_setrec/profiles/attempt_002/summary.json"
)
ALLOCATION_SUFFIX = Path(
    "artifacts/phase17/fullport/fp3_setrec/formal_allocation_attempt_002.json"
)
GPU_BY_ARM = {
    "S0_SETREC_ORDERED_CONTROL": 3,
    "S1R_SETREC_REPO_PARITY": 5,
    "S1P_SETREC_PAPER_FAITHFUL": 6,
    "S2_GRAM_SETREC_PAPER_FULL": 7,
}
MINIMUM_FREE_MIB_BY_ARM = {
    "S0_SETREC_ORDERED_CONTROL": 9326,
    "S1R_SETREC_REPO_PARITY": 9322,
    "S1P_SETREC_PAPER_FAITHFUL": 9330,
    "S2_GRAM_SETREC_PAPER_FULL": 16366,
}


@dataclass(frozen=True)
class FormalArmSpec:
    arm_id: str
    physical_gpu: int
    minimum_free_mib: int
    train_microbatch: int
    eval_batch_size: int


FORMAL_SPECS = {
    arm_id: FormalArmSpec(
        arm_id=arm_id,
        physical_gpu=GPU_BY_ARM[arm_id],
        minimum_free_mib=MINIMUM_FREE_MIB_BY_ARM[arm_id],
        train_microbatch=PROFILE_MICROBATCH_BY_ARM[arm_id],
        eval_batch_size=PROFILE_EVAL_BATCH_BY_ARM[arm_id],
    )
    for arm_id in SETREC_ARMS
}


def arm_slug(arm_id: str) -> str:
    if arm_id not in FORMAL_SPECS:
        raise ValueError(f"unknown FP3 formal arm: {arm_id}")
    return arm_id.lower()


def experiment_id(arm_id: str) -> str:
    return f"s17_fp3_formal_{arm_slug(arm_id)}_seed{SEED}"


def paths(root: Path, arm_id: str) -> dict[str, Path]:
    exp_id = experiment_id(arm_id)
    result = root / (
        f"artifacts/phase17/fullport/fp3_setrec/formal/{arm_slug(arm_id)}/{ATTEMPT_ID}"
    )
    snapshot = root / f"artifacts/phase17/snapshots/{exp_id}/{ATTEMPT_ID}/manifest.json"
    return {
        "result": result,
        "config": result / "config.json",
        "summary": result / "training_summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "best_checkpoint": result / "best_checkpoint.pt",
        "latest_checkpoint": result / "latest_checkpoint.pt",
        "learning_curve": result / "learning_curve.json",
        "authorization": root
        / f"artifacts/phase17/authorizations/{exp_id}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP3-FORMAL.attempts.jsonl",
        "snapshot": snapshot,
        "snapshot_worker": snapshot.parent / "src/000_s17_fp3_setrec_formal_runtime.py",
        "native_python": root / NATIVE_PYTHON_SUFFIX,
        "profile_status": root
        / f"artifacts/phase17/status/{PROFILE_EXPERIMENT_ID}.status.json",
        "profile_summary": root / PROFILE_SUMMARY_SUFFIX,
        "allocation": root / ALLOCATION_SUFFIX,
        "fp3_config": root / "experiment/phase17/config/s17_fp3_setrec.json",
        "tokenizer_artifact": root
        / "artifacts/phase17/fullport/fp3_setrec/tokenizer/attempt_001/sasrec_item_embeddings.pt",
    }


def verify_profile(root: Path) -> dict[str, Any]:
    resolved = paths(root, SETREC_ARMS[0])
    status = json.loads(resolved["profile_status"].read_text(encoding="utf-8"))
    summary = json.loads(resolved["profile_summary"].read_text(encoding="utf-8"))
    if (
        status.get("scientific_state") != "COMPLETED"
        or status.get("status_code") != PROFILE_STATUS_CODE
        or summary.get("verdict") != PROFILE_STATUS_CODE
        or summary.get("formal_effect_experiment_authorized") is not False
    ):
        raise RuntimeError("FP3 high-throughput profile evidence is not PASS")
    for arm_id, spec in FORMAL_SPECS.items():
        profile = summary["profiles"][arm_id]
        if (
            profile["train_microbatch"] != spec.train_microbatch
            or profile["eval_batch_size"] != spec.eval_batch_size
            or profile["effective_global_batch"] != 512
            or profile["fp16_parity"]["pass"] is not True
            or summary["formal_minimum_free_mib_by_arm"][arm_id]
            != spec.minimum_free_mib
        ):
            raise RuntimeError(f"FP3 formal profile contract drifted for {arm_id}")
    return {
        "status_path": str(resolved["profile_status"].relative_to(root)),
        "status_sha256": sha256(resolved["profile_status"]),
        "summary_path": str(resolved["profile_summary"].relative_to(root)),
        "summary_sha256": sha256(resolved["profile_summary"]),
    }


def _live_snapshot() -> dict[str, Any]:
    return gpu_runtime._gpu_state()


def _device(snapshot: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    matches = [row for row in snapshot["devices"] if row["index"] == gpu_id]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU{gpu_id} not uniquely visible")
    return matches[0]


def validate_allocation_snapshot(snapshot: dict[str, Any]) -> None:
    for arm_id, spec in FORMAL_SPECS.items():
        free_mib = _device(snapshot, spec.physical_gpu)["free_mib"]
        if free_mib < spec.minimum_free_mib:
            raise RuntimeError(
                f"GPU{spec.physical_gpu} free {free_mib} MiB is below "
                f"{arm_id} admission {spec.minimum_free_mib} MiB"
            )


def two_snapshot_admission() -> dict[str, Any]:
    first = _live_snapshot()
    time.sleep(5)
    second = _live_snapshot()
    validate_allocation_snapshot(first)
    validate_allocation_snapshot(second)
    return {
        "captured_at": utc_now(),
        "first": first,
        "second": second,
        "gpu_by_arm": GPU_BY_ARM,
        "minimum_free_mib_by_arm": MINIMUM_FREE_MIB_BY_ARM,
        "preexisting_compute_processes_by_arm": {
            arm_id: second["compute_processes"].get(spec.physical_gpu, [])
            for arm_id, spec in FORMAL_SPECS.items()
        },
        "shared_server_coexistence": True,
        "automatic_process_termination": False,
    }


def worker_command(root: Path, arm_id: str) -> list[str]:
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={spec.physical_gpu}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(resolved["native_python"]),
        str(resolved["snapshot_worker"]),
        "worker",
        "--arm",
        arm_id,
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def frozen_config(
    root: Path,
    arm_id: str,
    *,
    profile_evidence: dict[str, Any],
    allocation_sha256: str,
) -> dict[str, Any]:
    spec = FORMAL_SPECS[arm_id]
    training = SetRecFormalSpec()
    return {
        "schema_version": "phase17.s17_fp3_setrec_formal_config.v1",
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "step_id": STEP_ID,
        "arm_id": arm_id,
        "seed": SEED,
        "prepared_at": utc_now(),
        "physical_gpu": spec.physical_gpu,
        "minimum_free_mib": spec.minimum_free_mib,
        "train_microbatch": spec.train_microbatch,
        "gradient_accumulation": 512 // spec.train_microbatch,
        "eval_batch_size": spec.eval_batch_size,
        "training": asdict(training),
        "checkpoint_rule": "maximum train-prefix internal-dev NDCG@10; ties choose earlier evaluation",
        "beta_rule": "maximum train-prefix internal-dev NDCG@10; ties Hit@10 then smaller beta",
        "save_total_limit": 1,
        "profile_evidence": profile_evidence,
        "allocation_sha256": allocation_sha256,
        "fp3_config_sha256": sha256(paths(root, arm_id)["fp3_config"]),
        "tokenizer_artifact_sha256": sha256(
            paths(root, arm_id)["tokenizer_artifact"]
        ),
        "external_target_materialized": False,
        "external_d0_evaluation_deferred_until_all_four_checkpoints_frozen": True,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
    }


def prepare_all(root: Path) -> int:
    root = root.resolve()
    profile = verify_profile(root)
    for arm_id in SETREC_ARMS:
        resolved = paths(root, arm_id)
        if resolved["result"].exists() or resolved["snapshot"].exists():
            raise FileExistsError(f"FP3 formal attempt already exists for {arm_id}")
        if not resolved["native_python"].is_file():
            raise FileNotFoundError(resolved["native_python"])
    snapshot = _live_snapshot()
    validate_allocation_snapshot(snapshot)
    allocation = {
        "schema_version": "phase17.s17_fp3_formal_allocation.v1",
        "attempt_id": ATTEMPT_ID,
        "captured_at": utc_now(),
        "selection_policy": "fastest_four_way_parallel_assignment_from_live_free_memory_after_upscale_profile",
        "gpu_by_arm": GPU_BY_ARM,
        "minimum_free_mib_by_arm": MINIMUM_FREE_MIB_BY_ARM,
        "snapshot": snapshot,
        "formal_launch_authorized": False,
        "shared_server_coexistence": True,
        "preserve_all_preexisting_processes": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    allocation_path = root / ALLOCATION_SUFFIX
    if allocation_path.exists():
        raise FileExistsError(allocation_path)
    atomic_json(allocation_path, allocation)
    allocation_hash = sha256(allocation_path)
    for arm_id in SETREC_ARMS:
        resolved = paths(root, arm_id)
        resolved["result"].mkdir(parents=True, exist_ok=False)
        config = frozen_config(
            root,
            arm_id,
            profile_evidence=profile,
            allocation_sha256=allocation_hash,
        )
        atomic_json(resolved["config"], config)
        manifest = freeze_run_snapshot(
            root=root,
            experiment_id=experiment_id(arm_id),
            attempt_id=ATTEMPT_ID,
            command=worker_command(root, arm_id),
            source_paths=[
                Path(__file__),
                root / "experiment/phase17/core/full_setrec_executor.py",
                root / "experiment/phase17/core/full_setrec_backend.py",
                root / "experiment/phase17/core/full_setrec_contracts.py",
                root / "experiment/phase17/core/full_latte_gram_backend.py",
                root / "experiment/phase17/core/full_latte_native_adapter.py",
                root / "experiment/phase17/core/fullport_data.py",
                root / "experiment/phase17/core/status_writer.py",
                root / "experiment/phase17/core/run_manager.py",
            ],
            config=config,
        )
        AttemptLedger(resolved["ledger"]).append(
            {
                "attempt_id": f"{arm_slug(arm_id)}_{ATTEMPT_ID}",
                "experiment_id": experiment_id(arm_id),
                "step_id": STEP_ID,
                "arm_id": arm_id,
                "kind": "formal_internal_dev_checkpoint_selection",
                "started_at": utc_now(),
                "state": "PREFLIGHT_READY_EXACT_COMMAND_CONFIRMATION_REQUIRED",
                "scientific_result_eligible": True,
                "automatic_retry": False,
                "gpu_ids": [],
                "snapshot_manifest": str(manifest.relative_to(root)),
            }
        )
        writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
        writer.initialize(
            step_id=STEP_ID,
            attempt_id=ATTEMPT_ID,
            track_id=arm_id,
            canonical_result_dir=str(resolved["result"].relative_to(root)),
            log_path=str(resolved["log"].relative_to(root)),
            extra={
                "stage": "formal_preflight_waiting_exact_command_confirmation",
                "progress": {"current": 0, "total": 3330, "unit": "optimizer_step"},
                "run_snapshot_manifest": str(manifest.relative_to(root)),
                "profile_evidence": profile,
                "gpu_ids": [],
                "target_gpu_id": FORMAL_SPECS[arm_id].physical_gpu,
                "minimum_free_mib": FORMAL_SPECS[arm_id].minimum_free_mib,
                "launch_authorized": False,
                "automatic_retry": False,
                "automatic_process_termination": False,
                "external_target_materialized": False,
                "checkpoint_frozen": False,
                "d1_read": False,
                "d2_read": False,
                "result_selection_eligible": True,
                "affects_scientific_result": True,
            },
        )
        writer.transition(
            "PREFLIGHT",
            "PREFLIGHT",
            "S17_FP3_FORMAL_READY_EXACT_COMMAND_CONFIRMATION_REQUIRED",
            process_alive=False,
        )
    print(allocation_path)
    return 0


def supersede_attempt_001(root: Path) -> int:
    """Close the unlaunched GPU1 allocation before preparing attempt_002."""

    root = root.resolve()
    for arm_id in SETREC_ARMS:
        writer = StatusWriter(
            root / "artifacts/phase17/status", experiment_id(arm_id)
        )
        status_path = writer.paths.status(experiment_id(arm_id))
        if not status_path.is_file():
            continue
        status = writer.read()
        if (
            status.get("attempt_id") == "attempt_001"
            and status.get("scientific_state") == "PREFLIGHT"
        ):
            writer.transition(
                "STOPPED",
                "STOPPED",
                "S17_FP3_FORMAL_ATTEMPT_001_SUPERSEDED_TO_AVOID_GPU1",
                process_alive=False,
                workload_pid=0,
                stage="superseded_before_launch",
                gpu_ids=[],
                superseded_by_attempt_id=ATTEMPT_ID,
                superseded_reason=(
                    "GPU1 hosts unrelated researchers; use admitted GPU3 for S0"
                ),
                launch_authorized=False,
                automatic_retry=False,
                automatic_process_termination=False,
                external_target_materialized=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
    return 0


def authorize_all(root: Path, researcher_direction: str) -> int:
    root = root.resolve()
    if not researcher_direction.strip():
        raise ValueError("formal authorization requires researcher direction")
    snapshot = _live_snapshot()
    validate_allocation_snapshot(snapshot)
    for arm_id in SETREC_ARMS:
        resolved = paths(root, arm_id)
        if resolved["authorization"].exists():
            raise FileExistsError(resolved["authorization"])
        writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
        if writer.read()["scientific_state"] != "PREFLIGHT":
            raise RuntimeError(f"formal arm is not authorizable: {arm_id}")
        payload = {
            "schema_version": "phase17.s17_fp3_formal_authorization.v1",
            "authorized_at": utc_now(),
            "experiment_id": experiment_id(arm_id),
            "attempt_id": ATTEMPT_ID,
            "arm_id": arm_id,
            "authorized": True,
            "physical_gpu": FORMAL_SPECS[arm_id].physical_gpu,
            "minimum_free_mib": FORMAL_SPECS[arm_id].minimum_free_mib,
            "researcher_direction": researcher_direction,
            "confirmed_exact_command": (
                "bash experiment/phase17/run_stage17_fp3_setrec_formal.sh launch-all"
            ),
            "formal_effect_experiment_authorized": True,
            "external_target_materialization_authorized": False,
            "automatic_process_termination": False,
            "automatic_retry": False,
            "gpu_snapshot": snapshot,
        }
        atomic_json(resolved["authorization"], payload)
        writer.transition(
            "PREFLIGHT",
            "PREFLIGHT",
            "S17_FP3_FORMAL_AUTHORIZED_READY_TO_LAUNCH",
            launch_authorized=True,
            authorization_path=str(resolved["authorization"].relative_to(root)),
            authorization_sha256=sha256(resolved["authorization"]),
            gpu_snapshot=snapshot,
        )
    print(root / ALLOCATION_SUFFIX)
    return 0


def verify_authorization(root: Path, arm_id: str) -> dict[str, Any]:
    resolved = paths(root, arm_id)
    payload = json.loads(resolved["authorization"].read_text(encoding="utf-8"))
    spec = FORMAL_SPECS[arm_id]
    expected = {
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "arm_id": arm_id,
        "authorized": True,
        "physical_gpu": spec.physical_gpu,
        "minimum_free_mib": spec.minimum_free_mib,
        "formal_effect_experiment_authorized": True,
        "external_target_materialization_authorized": False,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"invalid FP3 formal authorization field: {key}")
    return payload


def launch_all(root: Path) -> int:
    root = root.resolve()
    for arm_id in SETREC_ARMS:
        writer = StatusWriter(paths(root, arm_id)["status_dir"], experiment_id(arm_id))
        if writer.read()["scientific_state"] != "PREFLIGHT":
            raise RuntimeError(f"FP3 formal arm is not launchable: {arm_id}")
        verify_authorization(root, arm_id)
    admission = two_snapshot_admission()
    launched: list[str] = []
    try:
        # Launch the expensive FiD arm first, then the three matched SETRec arms.
        launch_order = (
            "S2_GRAM_SETREC_PAPER_FULL",
            "S0_SETREC_ORDERED_CONTROL",
            "S1R_SETREC_REPO_PARITY",
            "S1P_SETREC_PAPER_FAITHFUL",
        )
        for arm_id in launch_order:
            resolved = paths(root, arm_id)
            session = experiment_id(arm_id)
            launch_background_tmux(
                experiment_id=experiment_id(arm_id),
                argv=worker_command(root, arm_id),
                cwd=root,
                tmux_session=session,
                startup_log_path=resolved["log"],
            )
            writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
            writer.transition(
                "RUNNING",
                "BACKGROUND_STARTED",
                "S17_FP3_FORMAL_BACKGROUND_STARTED",
                tmux_session=session,
                launcher_pid=os.getpid(),
                process_alive=True,
                stage="background_started",
                gpu_ids=[FORMAL_SPECS[arm_id].physical_gpu],
                gpu_snapshot=admission,
                launch_authorized=True,
            )
            if not wait_for_tmux_startup(session):
                latest = writer.read()
                if latest["scientific_state"] == "RUNNING":
                    writer.transition(
                        "FAILED",
                        "SCIENTIFIC_FAILED",
                        "S17_FP3_FORMAL_STARTUP_FAILED_NO_RETRY",
                        process_alive=False,
                        workload_pid=0,
                        gpu_ids=[],
                    )
                raise RuntimeError(f"FP3 formal worker exited at startup: {arm_id}")
            launched.append(arm_id)
    except BaseException:
        # Already-started scientific arms remain untouched; no automatic stop/retry.
        raise
    print(json.dumps({"launched": launched, "gpu_by_arm": GPU_BY_ARM}, indent=2))
    return 0


def worker(root: Path, arm_id: str, manifest_path: Path) -> int:
    root = root.resolve()
    resolved = paths(root, arm_id)
    spec = FORMAL_SPECS[arm_id]
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    started = time.monotonic()
    try:
        verify_run_snapshot(root, manifest_path)
        verify_profile(root)
        verify_authorization(root, arm_id)
        live = _live_snapshot()
        if _device(live, spec.physical_gpu)["free_mib"] < spec.minimum_free_mib:
            raise RuntimeError(f"GPU{spec.physical_gpu} lost formal headroom")
        import torch

        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP3_FORMAL_RUNNING",
            workload_pid=os.getpid(),
            process_alive=True,
            stage="initializing_formal_training",
            gpu_ids=[spec.physical_gpu],
            gpu_snapshot=live,
        )

        def heartbeat(record: dict[str, Any]) -> None:
            current = int(record["global_step"])
            total = int(record["total_steps"])
            stage = (
                f"internal_dev_evaluation_step_{current}"
                if "latest_internal_dev" in record
                else f"training_step_{current}"
            )
            writer.heartbeat(
                stage=stage,
                progress={
                    "current": current,
                    "total": total,
                    "unit": "optimizer_step",
                    "latest": record,
                },
            )

        training = train_setrec_formal(
            root,
            arm_id,
            output_dir=resolved["result"],
            device=torch.device("cuda"),
            spec=SetRecFormalSpec(),
            heartbeat=heartbeat,
        )
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / 1048576
        peak_reserved = torch.cuda.max_memory_reserved() / 1048576
        summary = {
            "schema_version": "phase17.s17_fp3_setrec_formal_training_summary.v1",
            "verdict": "PASS_S17_FP3_FORMAL_CHECKPOINT_FROZEN",
            "completed_at": utc_now(),
            "arm_id": arm_id,
            "physical_gpu": spec.physical_gpu,
            "training": training,
            "best_checkpoint_path": str(resolved["best_checkpoint"].relative_to(root)),
            "best_checkpoint_sha256": sha256(resolved["best_checkpoint"]),
            "latest_checkpoint_path": str(resolved["latest_checkpoint"].relative_to(root)),
            "latest_checkpoint_sha256": sha256(resolved["latest_checkpoint"]),
            "learning_curve_path": str(resolved["learning_curve"].relative_to(root)),
            "learning_curve_sha256": sha256(resolved["learning_curve"]),
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "wall_seconds": time.monotonic() - started,
            "external_target_materialized": False,
            "external_evaluation_pending_all_four_checkpoint_freeze": True,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP3_FORMAL_CHECKPOINT_FROZEN",
            process_alive=False,
            workload_pid=0,
            stage="best_checkpoint_frozen_external_target_still_sealed",
            progress={
                "current": training["global_steps_completed"],
                "total": training["total_planned_steps"],
                "unit": "optimizer_step",
            },
            gpu_ids=[],
            checkpoint_frozen=True,
            checkpoint_path=summary["best_checkpoint_path"],
            checkpoint_sha256=summary["best_checkpoint_sha256"],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            external_target_materialized=False,
            result_selection_eligible=True,
            affects_scientific_result=True,
        )
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "arm_id": arm_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "automatic_process_termination": False,
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
                "S17_FP3_FORMAL_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                stage="terminal_failure_no_retry",
                gpu_ids=[],
                failure_path=str(resolved["failure"].relative_to(root)),
                failure_sha256=sha256(resolved["failure"]),
                terminal_error=repr(error),
                automatic_retry=False,
                automatic_process_termination=False,
                external_target_materialized=False,
            )
        return 1


def inspect(root: Path, arm_id: str) -> dict[str, Any]:
    resolved = paths(root.resolve(), arm_id)
    status_path = resolved["status_dir"] / f"{experiment_id(arm_id)}.status.json"
    payload: dict[str, Any] = {
        "arm_id": arm_id,
        "experiment_id": experiment_id(arm_id),
        "physical_gpu": FORMAL_SPECS[arm_id].physical_gpu,
    }
    for key, path in (
        ("status", status_path),
        ("summary", resolved["summary"]),
        ("failure", resolved["failure"]),
    ):
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if key == "status":
                value = {
                    field: value.get(field)
                    for field in (
                        "scientific_state",
                        "execution_state",
                        "status_code",
                        "stage",
                        "process_alive",
                        "progress",
                        "heartbeat_at",
                    )
                }
            payload[key] = value
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare-all",
            "supersede-attempt-001",
            "authorize-all",
            "launch-all",
            "worker",
            "inspect",
            "inspect-all",
        ),
    )
    parser.add_argument("--arm", choices=SETREC_ARMS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--researcher-direction", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare-all":
        return prepare_all(root)
    if args.action == "supersede-attempt-001":
        return supersede_attempt_001(root)
    if args.action == "authorize-all":
        return authorize_all(root, args.researcher_direction)
    if args.action == "launch-all":
        return launch_all(root)
    if args.action == "inspect-all":
        print(
            json.dumps(
                [inspect(root, arm_id) for arm_id in SETREC_ARMS],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.arm is None:
        raise ValueError(f"{args.action} requires --arm")
    if args.action == "inspect":
        print(json.dumps(inspect(root, args.arm), ensure_ascii=False, indent=2))
        return 0
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.arm, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
