#!/usr/bin/env python3
"""Authorization-gated formal FP1/FP2 checkpoint-selection training."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
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
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as profile_base


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "attempt_004"
SEED = 2023
ALLOCATION_SUFFIX = Path("experiment/phase17/config/s17_fp_resource_allocation.json")
MATRIX_SUFFIX = Path("experiment/phase17/config/s17_fp12_latte_arm_matrix.json")
LEDGER_SUFFIX = Path("artifacts/phase17/attempts/S17-FP12-FORMAL.attempts.jsonl")
RESEARCHER_DIRECTION = (
    "当前microbatch=2显存占用太少，提高后从头重跑；能提多高提多高，"
    "只要不OOM。正式实测确认GPU5:G0用16，GPU3:G1和GPU1:G2用8为逐卡最高安全档。"
)
GRAM_MICROBATCH_BY_ARM = {
    "G0_GRAM_B0_FRESH": 16,
    "G1_GRAM_PSID_FULL": 8,
    "G2_GRAM_LATTE_FULL": 8,
}
GRAM_ACCUMULATION_BY_ARM = {
    arm_id: 128 // microbatch
    for arm_id, microbatch in GRAM_MICROBATCH_BY_ARM.items()
}
GRAM_EFFECTIVE_BATCH = 128
GRAM_UPSCALE_MINIMUM_FREE_MIB = 18968
GRAM_UPSCALE_PROFILE_PEAK_MIB_BY_BATCH = {8: 8088, 16: 15506}


@dataclass(frozen=True)
class FormalSpec:
    arm_id: str
    step_id: str
    physical_gpu: int
    minimum_free_mib: int
    profile_peak_reserved_mib: int
    timeout_seconds: int
    family: str


FORMAL_SPECS = {
    "G0_GRAM_B0_FRESH": FormalSpec(
        "G0_GRAM_B0_FRESH", "S17-FP2", 5, GRAM_UPSCALE_MINIMUM_FREE_MIB, 15884, 7 * 24 * 3600, "gram"
    ),
    "G1_GRAM_PSID_FULL": FormalSpec(
        "G1_GRAM_PSID_FULL", "S17-FP2", 3, GRAM_UPSCALE_MINIMUM_FREE_MIB, 15892, 7 * 24 * 3600, "gram"
    ),
    "G2_GRAM_LATTE_FULL": FormalSpec(
        "G2_GRAM_LATTE_FULL", "S17-FP2", 1, GRAM_UPSCALE_MINIMUM_FREE_MIB, 15896, 7 * 24 * 3600, "gram"
    ),
    "N0_NATIVE_PSID": FormalSpec(
        "N0_NATIVE_PSID", "S17-FP1", 2, 6076, 1980, 3 * 24 * 3600, "native"
    ),
    "N1_NATIVE_LATTE": FormalSpec(
        "N1_NATIVE_LATTE", "S17-FP1", 6, 7020, 2924, 3 * 24 * 3600, "native"
    ),
}


PROFILE_EVIDENCE = {
    "G0_GRAM_B0_FRESH": (
        "s17_fp12_profile_r4_g0_gram_b0_fresh",
        "artifacts/phase17/fullport/profiles/g0_gram_b0_fresh/attempt_004/summary.json",
    ),
    "G1_GRAM_PSID_FULL": (
        "s17_fp12_profile_r4_g1_gram_psid_full",
        "artifacts/phase17/fullport/profiles/g1_gram_psid_full/attempt_004/summary.json",
    ),
    "G2_GRAM_LATTE_FULL": (
        "s17_fp12_profile_r4_g2_gram_latte_full",
        "artifacts/phase17/fullport/profiles/g2_gram_latte_full/attempt_004/summary.json",
    ),
    "N0_NATIVE_PSID": (
        "s17_fp12_profile_r4_n0_native_psid",
        "artifacts/phase17/fullport/profiles/n0_native_psid/attempt_004/summary.json",
    ),
    "N1_NATIVE_LATTE": (
        "s17_fp12_profile_r3_n1_native_latte",
        "artifacts/phase17/fullport/profiles/n1_native_latte/attempt_003/summary.json",
    ),
}


def arm_slug(arm_id: str) -> str:
    if arm_id not in FORMAL_SPECS:
        raise ValueError(f"unknown formal arm: {arm_id}")
    return arm_id.lower()


def experiment_id(arm_id: str) -> str:
    return f"s17_fp12_formal_{arm_slug(arm_id)}_seed{SEED}"


def paths(root: Path, arm_id: str) -> dict[str, Path]:
    exp_id = experiment_id(arm_id)
    result = root / f"artifacts/phase17/fullport/formal/{arm_slug(arm_id)}/{ATTEMPT_ID}"
    snapshot = root / f"artifacts/phase17/snapshots/{exp_id}/{ATTEMPT_ID}/manifest.json"
    return {
        "result": result,
        "config": result / "config.json",
        "summary": result / "training_summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "allocation": root / ALLOCATION_SUFFIX,
        "matrix": root / MATRIX_SUFFIX,
        "authorization": root
        / f"artifacts/phase17/authorizations/{exp_id}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / LEDGER_SUFFIX,
        "snapshot": snapshot,
        "snapshot_worker": snapshot.parent / "src/000_s17_fp12_formal_runtime.py",
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_python(root: Path, spec: FormalSpec) -> Path:
    if spec.family == "gram":
        return profile_base.GRAM_PYTHON
    return root / profile_base.NATIVE_PYTHON_SUFFIX


def verify_profile_evidence(root: Path, arm_id: str) -> dict[str, Any]:
    status_id, summary_suffix = PROFILE_EVIDENCE[arm_id]
    status_path = root / f"artifacts/phase17/status/{status_id}.status.json"
    summary_path = root / summary_suffix
    status = _read(status_path)
    summary = _read(summary_path)
    spec = FORMAL_SPECS[arm_id]
    if (
        status.get("scientific_state") != "COMPLETED"
        or status.get("status_code") != "PASS_S17_FP12_ARM_RESOURCE_PROFILE"
        or summary.get("verdict") != "PASS_S17_FP12_ARM_RESOURCE_PROFILE"
        or summary.get("arm_id") != arm_id
    ):
        raise RuntimeError(f"formal profile evidence is not PASS for {arm_id}")
    measured = round(float(summary["measured"]["peak_reserved_mib"]))
    if measured != spec.profile_peak_reserved_mib:
        raise RuntimeError(
            f"profile peak drift for {arm_id}: {measured} != {spec.profile_peak_reserved_mib}"
        )
    evidence = {
        "profile_status_path": str(status_path.relative_to(root)),
        "profile_status_sha256": sha256(status_path),
        "profile_summary_path": str(summary_path.relative_to(root)),
        "profile_summary_sha256": sha256(summary_path),
        "profile_peak_reserved_mib": measured,
        "formal_minimum_free_mib": spec.minimum_free_mib,
    }
    if spec.family == "gram":
        microbatch = GRAM_MICROBATCH_BY_ARM[arm_id]
        accumulation = GRAM_ACCUMULATION_BY_ARM[arm_id]
        upscale_peak = GRAM_UPSCALE_PROFILE_PEAK_MIB_BY_BATCH[microbatch]
        upscale_status_path = root / (
            "artifacts/phase17/status/"
            f"s17_fp12_microbatch_training_profile_g2_mb{microbatch}.status.json"
        )
        upscale_summary_path = root / (
            "artifacts/phase17/fullport/profiles/g2_gram_latte_full/"
            f"microbatch_training_upscale/mb{microbatch}/attempt_001/summary.json"
        )
        upscale_status = _read(upscale_status_path)
        upscale_summary = _read(upscale_summary_path)
        if (
            upscale_status.get("scientific_state") != "COMPLETED"
            or upscale_status.get("status_code")
            != "PASS_S17_FP12_MICROBATCH_RESOURCE_PROFILE"
            or upscale_summary.get("verdict")
            != "PASS_S17_FP12_MICROBATCH_RESOURCE_PROFILE"
            or upscale_summary.get("arm_id") != "G2_GRAM_LATTE_FULL"
            or upscale_summary.get("train_microbatch") != microbatch
            or upscale_summary.get("gradient_accumulation") != accumulation
            or upscale_summary.get("effective_batch") != GRAM_EFFECTIVE_BATCH
            or upscale_summary.get("training_only") is not True
            or upscale_summary.get("measured", {}).get("primary_generation_included")
            is not False
            or round(float(upscale_summary.get("profile_peak_reserved_mib", -1)))
            != upscale_peak
        ):
            raise RuntimeError("GRAM microbatch-upscale profile evidence is not PASS")
        evidence["microbatch_upscale"] = {
            "profile_status_path": str(upscale_status_path.relative_to(root)),
            "profile_status_sha256": sha256(upscale_status_path),
            "profile_summary_path": str(upscale_summary_path.relative_to(root)),
            "profile_summary_sha256": sha256(upscale_summary_path),
            "train_microbatch": microbatch,
            "gradient_accumulation": accumulation,
            "effective_batch": GRAM_EFFECTIVE_BATCH,
            "profile_peak_reserved_mib": upscale_peak,
            "formal_minimum_free_mib": GRAM_UPSCALE_MINIMUM_FREE_MIB,
        }
    return evidence


def snapshot_sources(root: Path, spec: FormalSpec) -> list[Path]:
    sources = [
        Path(__file__).resolve(),
        root / "experiment/phase17/core/full_latte_formal_executor.py",
        root / "experiment/phase17/core/full_latte_arm_contracts.py",
        root / "experiment/phase17/core/full_latte_contracts.py",
        root / "experiment/phase17/core/fullport_data.py",
        root / "experiment/phase17/core/full_latte_native_adapter.py",
        root / "experiment/phase17/core/status_writer.py",
        root / "experiment/phase17/core/run_manager.py",
        root / "experiment/phase17/core/resource_profiler.py",
    ]
    if spec.family == "gram":
        sources.extend(
            [
                root / "experiment/phase17/core/full_latte_gram_backend.py",
                root / "GRAM/src/model/gram.py",
                root / "GRAM/src/model/gram_t5.py",
                root / "GRAM/src/model/gram_t5_modeling.py",
                root / "GRAM/src/processor/Collator.py",
            ]
        )
    else:
        official = root / (
            "artifacts/phase17/fullport/sources/"
            "latte_05e4e6d983225bcb7172f148a076890e80c524d1_attempt_003"
        )
        sources.extend(
            [
                root / "experiment/phase17/core/full_latte_native_backend.py",
                official / "genrec/models/PSID/model.py",
                official / "genrec/models/PSID/tokenizer.py",
                official / "genrec/models/Latte/model.py",
                official / "genrec/models/Latte/tokenizer.py",
                official / "genrec/evaluator.py",
            ]
        )
    return sources


def worker_command(root: Path, arm_id: str) -> list[str]:
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={spec.physical_gpu}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(selected_python(root, spec)),
        str(resolved["snapshot_worker"]),
        "worker",
        "--arm",
        arm_id,
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def frozen_config(root: Path, arm_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    spec = FORMAL_SPECS[arm_id]
    common = {
        "schema_version": "phase17.s17_fp12_formal_config.v1",
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "step_id": spec.step_id,
        "arm_id": arm_id,
        "family": spec.family,
        "seed": SEED,
        "physical_gpu": spec.physical_gpu,
        "minimum_free_mib": spec.minimum_free_mib,
        "profile_peak_reserved_mib": spec.profile_peak_reserved_mib,
        "timeout_seconds": spec.timeout_seconds,
        "precision": "fp32",
        "checkpoint_selection": "train_prefix_internal_dev_ndcg@10_only",
        "external_target_materialized": False,
        "external_evaluation_deferred_until_all_family_checkpoints_frozen": True,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "profile_evidence": evidence,
    }
    if spec.family == "gram":
        microbatch = GRAM_MICROBATCH_BY_ARM[arm_id]
        common["training"] = {
            "maximum_epochs": 50,
            "minimum_epochs": 20,
            "evaluation_interval_epochs": 5,
            "early_stop_patience_evaluations": 3,
            "train_microbatch": microbatch,
            "gradient_accumulation": GRAM_ACCUMULATION_BY_ARM[arm_id],
            "effective_batch": GRAM_EFFECTIVE_BATCH,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_fraction": 0.05,
            "gradient_clip": 1.0,
            "internal_eval_batch_size": 1,
            "internal_eval_beam": 50,
            "primary_final_beam": 500,
            "generation_kv_cache": False,
        }
    else:
        common["training"] = {
            "maximum_epochs": 150,
            "early_stop_patience_epochs": 50,
            "train_batch_size": 256,
            "learning_rate": 0.003,
            "weight_decay": 0.05,
            "warmup_steps": 10000,
            "gradient_clip": 1.0,
            "internal_eval_batch_size": 1,
            "internal_eval_beam": 50,
            "primary_final_beam": 500,
        }
    return common


def prepare(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    status_path = resolved["status_dir"] / f"{experiment_id(arm_id)}.status.json"
    if resolved["result"].exists():
        raise FileExistsError(f"formal attempt already exists for {arm_id}")
    if status_path.exists():
        previous = _read(status_path)
        if previous.get("scientific_state") not in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError("previous formal attempt status is not terminal")
        if previous.get("attempt_id") == ATTEMPT_ID:
            raise FileExistsError(f"formal attempt already exists for {arm_id}")
    if not selected_python(root, spec).is_file():
        raise FileNotFoundError("formal Python environment is missing")
    evidence = verify_profile_evidence(root, arm_id)
    allocation = _read(resolved["allocation"])[f"fp1_fp2_formal_{ATTEMPT_ID}"]
    if allocation["physical_gpu_by_arm"].get(arm_id) != spec.physical_gpu:
        raise PermissionError("formal arm-to-GPU allocation drifted")
    if allocation["launch_authorized_by_arm"].get(arm_id) is not True:
        raise PermissionError(f"formal allocation is not authorized for {arm_id}")
    resolved["result"].mkdir(parents=True, exist_ok=False)
    config = frozen_config(root, arm_id, evidence)
    config["allocation_sha256"] = sha256(resolved["allocation"])
    config["matrix_sha256"] = sha256(resolved["matrix"])
    atomic_json(resolved["config"], config)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=experiment_id(arm_id),
        attempt_id=ATTEMPT_ID,
        command=worker_command(root, arm_id),
        source_paths=snapshot_sources(root, spec),
        config=config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": f"{arm_slug(arm_id)}_{ATTEMPT_ID}",
            "formal_attempt_id": ATTEMPT_ID,
            "experiment_id": experiment_id(arm_id),
            "step_id": spec.step_id,
            "arm_id": arm_id,
            "kind": "formal",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_AUTHORIZATION_REQUIRED",
            "scientific_result_eligible": True,
            "automatic_retry": False,
            "gpu_ids": [],
            "snapshot_manifest": str(manifest.relative_to(root)),
        }
    )
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    writer.initialize(
        step_id=spec.step_id,
        attempt_id=ATTEMPT_ID,
        track_id=arm_id,
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "formal_preflight_complete_waiting_launch_authorization",
            "progress": {"current": 0, "total": 150 if spec.family == "native" else 50, "unit": "epoch"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "profile_evidence": evidence,
            "gpu_ids": [],
            "target_gpu_id": spec.physical_gpu,
            "minimum_free_mib": spec.minimum_free_mib,
            "launch_authorized": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "external_target_materialized": False,
            "checkpoint_frozen": False,
            "result_selection_eligible": True,
            "affects_scientific_result": True,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_FORMAL_READY_AUTHORIZATION_REQUIRED",
        process_alive=False,
    )
    print(json.dumps({"arm_id": arm_id, "manifest": str(manifest)}, indent=2))
    return 0


def gpu_snapshot(spec: FormalSpec) -> dict[str, Any]:
    proxy = profile_base.ProfileSpec(
        spec.arm_id,
        spec.step_id,
        spec.physical_gpu,
        spec.profile_peak_reserved_mib,
        spec.minimum_free_mib,
        1,
        1,
        spec.family,
    )
    return profile_base.gpu_snapshot_once(proxy)


def admission(spec: FormalSpec) -> dict[str, Any]:
    first = gpu_snapshot(spec)
    time.sleep(5)
    second = gpu_snapshot(spec)
    for row in (first, second):
        free = int(row["selected"]["free_mib"])
        if free < spec.minimum_free_mib:
            raise RuntimeError(
                f"GPU{spec.physical_gpu} free={free} MiB below formal gate {spec.minimum_free_mib} MiB"
            )
    return {
        "policy": "two_snapshot_free_memory_only",
        "interval_seconds": 5,
        "minimum_free_mib": spec.minimum_free_mib,
        "first": first,
        "second": second,
        "utilization_recorded_only": True,
        "preexisting_processes_preserved": True,
        "automatic_process_termination": False,
    }


def authorize(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    if resolved["authorization"].exists():
        raise FileExistsError("formal authorization already exists")
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("formal attempt is not authorizable")
    current = gpu_snapshot(spec)
    if current["selected"]["free_mib"] < spec.minimum_free_mib:
        raise RuntimeError(f"GPU{spec.physical_gpu} is below formal memory gate")
    payload = {
        "schema_version": "phase17.s17_fp12_formal_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "arm_id": arm_id,
        "authorized": True,
        "physical_gpu": spec.physical_gpu,
        "minimum_free_mib": spec.minimum_free_mib,
        "researcher_direction": RESEARCHER_DIRECTION,
        "formal_checkpoint_selection_authorized": True,
        "external_target_evaluation_authorized": False,
        "observed_preexisting_compute_pids": sorted(
            int(row["pid"]) for row in current["selected_compute_processes"]
        ),
        "preserve_all_preexisting_compute_processes": True,
        "automatic_retry": False,
        "automatic_process_termination": False,
    }
    atomic_json(resolved["authorization"], payload)
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_FORMAL_AUTHORIZED_WAITING_LAUNCH",
        launch_authorized=True,
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        gpu_snapshot={"authorization": current},
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_authorization(root: Path, arm_id: str) -> dict[str, Any]:
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    payload = _read(resolved["authorization"])
    expected = {
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "arm_id": arm_id,
        "authorized": True,
        "physical_gpu": spec.physical_gpu,
        "formal_checkpoint_selection_authorized": True,
        "external_target_evaluation_authorized": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"invalid formal authorization field: {key}")
    if payload.get("researcher_direction") != RESEARCHER_DIRECTION:
        raise PermissionError("formal authorization lost researcher direction")
    return payload


def launch(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("formal attempt is not launchable")
    verify_authorization(root, arm_id)
    snapshots = admission(spec)
    session = launch_background_tmux(
        experiment_id=experiment_id(arm_id),
        argv=worker_command(root, arm_id),
        cwd=root,
        tmux_session=experiment_id(arm_id),
        startup_log_path=resolved["log"],
    )
    writer.transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP12_FORMAL_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="background_started",
        gpu_ids=[spec.physical_gpu],
        gpu_snapshot=snapshots,
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_FORMAL_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                automatic_retry=False,
            )
        raise RuntimeError("formal worker exited during startup handshake")
    print(session)
    return 0


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("formal arm exceeded its frozen hard timeout")


def worker(root: Path, arm_id: str, manifest: Path) -> int:
    root = root.resolve()
    spec = FORMAL_SPECS[arm_id]
    resolved = paths(root, arm_id)
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    started = time.monotonic()
    try:
        verify_run_snapshot(root, manifest)
        verify_profile_evidence(root, arm_id)
        verify_authorization(root, arm_id)
        snapshots = admission(spec)
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP12_FORMAL_TRAINING",
            workload_pid=os.getpid(),
            process_alive=True,
            stage="initializing_formal_training",
            gpu_ids=[spec.physical_gpu],
            gpu_snapshot=snapshots,
            external_target_materialized=False,
        )
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(spec.timeout_seconds)
        from experiment.phase17.core.full_latte_formal_executor import train_formal_arm

        def heartbeat(stage: str, progress: dict[str, Any]) -> None:
            writer.heartbeat(stage=stage, progress=progress)

        training = train_formal_arm(
            root, arm_id, resolved["result"], heartbeat=heartbeat
        )
        signal.alarm(0)
        checkpoint = root / training["checkpoint_path"]
        curve = root / training["learning_curve_path"]
        summary = {
            "schema_version": "phase17.s17_fp12_formal_training_summary.v1",
            "verdict": "PASS_S17_FP12_FORMAL_CHECKPOINT_FROZEN",
            "completed_at": utc_now(),
            "arm_id": arm_id,
            "physical_gpu": spec.physical_gpu,
            "training": training,
            "checkpoint_sha256": sha256(checkpoint),
            "learning_curve_sha256": sha256(curve),
            "wall_seconds": time.monotonic() - started,
            "external_target_materialized": False,
            "external_evaluation_pending_family_checkpoint_freeze": True,
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
            "PASS_S17_FP12_FORMAL_CHECKPOINT_FROZEN",
            process_alive=False,
            workload_pid=0,
            stage="best_checkpoint_frozen_external_target_still_sealed",
            progress={
                "current": training["epochs_completed"],
                "total": training["epochs_completed"],
                "unit": "epoch",
            },
            gpu_ids=[],
            checkpoint_frozen=True,
            checkpoint_path=training["checkpoint_path"],
            checkpoint_sha256=summary["checkpoint_sha256"],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            external_target_materialized=False,
            result_selection_eligible=True,
            affects_scientific_result=True,
        )
        return 0
    except BaseException as error:
        signal.alarm(0)
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
                "S17_FP12_FORMAL_FAILED_NO_RETRY",
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
    result = {
        "arm_id": arm_id,
        "experiment_id": experiment_id(arm_id),
        "prepared": resolved["config"].is_file() and resolved["snapshot"].is_file(),
        "authorization_present": resolved["authorization"].is_file(),
        "summary_present": resolved["summary"].is_file(),
        "failure_present": resolved["failure"].is_file(),
    }
    if status_path.is_file():
        status = _read(status_path)
        result["status"] = {
            key: status.get(key)
            for key in (
                "scientific_state",
                "execution_state",
                "status_code",
                "stage",
                "process_alive",
                "progress",
            )
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "authorize", "launch", "worker", "inspect"))
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root, args.arm)
    if args.action == "authorize":
        return authorize(root, args.arm)
    if args.action == "launch":
        return launch(root, args.arm)
    if args.action == "inspect":
        print(json.dumps(inspect(root, args.arm), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.arm, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
