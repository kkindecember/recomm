#!/usr/bin/env python3
"""Researcher-authorized G1-only parallel recovery on a newly available GPU."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Mapping

from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import (
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)
from experiment.phase17.protocol import s17_fp12_external_d0_recovery_runtime as recovery
from experiment.phase17.protocol import s17_fp12_external_d0_runtime as external


ROOT = Path(__file__).resolve().parents[3]
ARM_ID = "G1_GRAM_PSID_FULL"
ATTEMPT_ID = "attempt_003"
EXPERIMENT_ID = "s17_fp12_external_d0_recovery_g1_parallel"
STEP_ID = "S17-FP12-EXTERNAL-D0-G1-PARALLEL"
PHYSICAL_GPU = 4
MINIMUM_FREE_MIB = 18968
RESULT_SUFFIX = Path("artifacts/phase17/fullport/external_d0/recovery/attempt_003")
AUTHORIZATION_SUFFIX = Path(
    "artifacts/phase17/authorizations/"
    "s17_fp12_external_d0_recovery_g1_parallel_attempt_003.json"
)
LEDGER_SUFFIX = Path("artifacts/phase17/attempts/S17-FP12-EXTERNAL-D0.attempts.jsonl")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    original = external.paths(root)
    result = root / RESULT_SUFFIX
    return {
        "result": result,
        "arm_result": result / "arms" / external.arm_slug(ARM_ID),
        "config": result / "config.json",
        "provenance": result / "provenance.json",
        "predictions": result / "arms" / external.arm_slug(ARM_ID) / "predictions.jsonl",
        "summary": result / "arms" / external.arm_slug(ARM_ID) / "summary.json",
        "failure": result / "arms" / external.arm_slug(ARM_ID) / "failure.json",
        "log": result / "arms" / external.arm_slug(ARM_ID) / "run.log",
        "authorization": root / AUTHORIZATION_SUFFIX,
        "status_dir": original["status_dir"],
        "ledger": root / LEDGER_SUFFIX,
        "snapshot": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "snapshot_worker": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/000_s17_fp12_external_d0_g1_parallel_runtime.py",
        "snapshot_guard": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/001_s17_fp12_g1_runtime_guard.py",
        "original_bundle": original["bundle"],
        "original_seal": original["seal"],
        "original_config": original["config"],
        "analysis": original["analysis"],
    }


def source_paths(root: Path) -> list[Path]:
    extra = [
        root / "experiment/phase17/protocol/s17_fp12_g1_runtime_guard.py",
        Path(recovery.__file__).resolve(),
        root / "experiment/phase17/protocol/s17_fp12_resource_profile_runtime.py",
        root / "experiment/phase17/core/full_latte_profile_executor.py",
    ]
    original = external.source_paths(root)
    seen: set[Path] = set()
    output: list[Path] = []
    for path in [Path(__file__).resolve(), *extra, *original]:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(resolved)
    return output


def verify_prerequisites(root: Path) -> dict[str, Any]:
    root = root.resolve()
    original = external.paths(root)
    seal = external.verify_seal(root)
    original_config = _read(original["config"])
    checkpoints = external.verify_checkpoint_freeze(root, original_config)
    recovery.verify_recovery(root)
    attempt_002_g1 = recovery.arm_paths(root, ARM_ID)
    attempt_002_status = original["status_dir"] / (
        f"{recovery.arm_experiment_id(ARM_ID)}.status.json"
    )
    if attempt_002_status.exists() or attempt_002_g1["result"].exists():
        raise RuntimeError("attempt_002 G1 already has worker artifacts")
    active: dict[str, Any] = {}
    for arm_id in ("G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL"):
        status_path = original["status_dir"] / (
            f"{recovery.arm_experiment_id(arm_id)}.status.json"
        )
        status = _read(status_path)
        if status.get("scientific_state") not in {"RUNNING", "COMPLETED"}:
            raise RuntimeError(f"attempt_002 arm is not healthy: {arm_id}")
        active[arm_id] = {
            "status_path": str(status_path.relative_to(root)),
            "scientific_state": status["scientific_state"],
            "left_untouched": True,
        }
    return {
        "schema_version": "phase17.s17_fp12_g1_parallel_evidence.v1",
        "verified_at": utc_now(),
        "arm_id": ARM_ID,
        "checkpoint_evidence": checkpoints[ARM_ID],
        "attempt_002_active_arms": active,
        "attempt_002_g1_worker_started": False,
        "bundle_path": seal["bundle_path"],
        "bundle_sha256": seal["bundle_sha256"],
        "materialization_seal_path": str(original["seal"].relative_to(root)),
        "materialization_seal_sha256": sha256(original["seal"]),
        "single_materialization_count": 1,
        "raw_external_projection_reopened": False,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }


def prepare(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if any(
        path.exists()
        for path in (
            resolved["result"],
            resolved["authorization"],
            resolved["snapshot"],
            status_path,
        )
    ):
        raise FileExistsError("G1 parallel attempt_003 already exists")
    evidence = verify_prerequisites(root)
    resolved["result"].mkdir(parents=True, exist_ok=False)
    authorization = {
        "schema_version": "phase17.s17_fp12_g1_parallel_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "original_researcher_direction": (
            "那你找一张gpu把g0运行起来吧 然后给他做个重复轮 跑完之后把资源占住"
        ),
        "researcher_correction": "哦哦g1 我说错了",
        "resolved_direction": (
            "找一张GPU运行G1；G1完成后在同卡启动隔离重复轮持续占用资源"
        ),
        "arm_id": ARM_ID,
        "physical_gpu": PHYSICAL_GPU,
        "post_completion_runtime_guard_authorized": True,
        "reuse_sealed_bundle_only": True,
        "raw_external_projection_reopen_authorized": False,
        "preserve_all_preexisting_compute_processes": True,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["authorization"], authorization)
    config = {
        "schema_version": "phase17.s17_fp12_g1_parallel_config.v1",
        "attempt_id": ATTEMPT_ID,
        "arm_id": ARM_ID,
        "physical_gpu": PHYSICAL_GPU,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "bundle_path": evidence["bundle_path"],
        "bundle_sha256": evidence["bundle_sha256"],
        "materialization_seal_path": evidence["materialization_seal_path"],
        "materialization_seal_sha256": evidence["materialization_seal_sha256"],
        "original_config_path": str(resolved["original_config"].relative_to(root)),
        "original_config_sha256": sha256(resolved["original_config"]),
        "authorization_path": str(resolved["authorization"].relative_to(root)),
        "authorization_sha256": sha256(resolved["authorization"]),
        "supersedes_attempt_002_unlaunched_g1_schedule": True,
        "post_completion_runtime_guard": True,
        "single_materialization_count": 1,
        "raw_external_projection_reopened": False,
    }
    atomic_json(resolved["config"], config)
    atomic_json(resolved["provenance"], evidence)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=["researcher_authorized_g1_parallel_and_post_completion_guard"],
        source_paths=source_paths(root),
        config=config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "experiment_id": EXPERIMENT_ID,
            "step_id": STEP_ID,
            "kind": "researcher_authorized_g1_parallel_schedule",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY",
            "scientific_result_eligible": True,
            "snapshot_manifest": str(manifest.relative_to(root)),
            "external_target_materialized": True,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
            "automatic_retry": False,
        }
    )
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id=STEP_ID,
        attempt_id=ATTEMPT_ID,
        track_id=ARM_ID,
        canonical_result_dir=str(resolved["arm_result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "g1_parallel_preflight",
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "authorization_path": str(resolved["authorization"].relative_to(root)),
            "authorization_sha256": sha256(resolved["authorization"]),
            "target_gpu_id": PHYSICAL_GPU,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "external_target_materialized": True,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
            "post_completion_runtime_guard_authorized": True,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_G1_PARALLEL_PREFLIGHT_COMPLETE",
        process_alive=False,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify(root: Path, manifest: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    config = _read(resolved["config"])
    authorization = _read(resolved["authorization"])
    if (
        authorization.get("researcher_correction") != "哦哦g1 我说错了"
        or authorization.get("arm_id") != ARM_ID
        or authorization.get("physical_gpu") != PHYSICAL_GPU
        or authorization.get("post_completion_runtime_guard_authorized") is not True
        or authorization.get("reuse_sealed_bundle_only") is not True
        or authorization.get("raw_external_projection_reopen_authorized") is not False
    ):
        raise PermissionError("G1 parallel authorization drifted")
    seal = external.verify_seal(root)
    if (
        sha256(resolved["original_bundle"]) != config["bundle_sha256"]
        or seal["bundle_sha256"] != config["bundle_sha256"]
        or sha256(resolved["original_seal"]) != config["materialization_seal_sha256"]
        or sha256(resolved["authorization"]) != config["authorization_sha256"]
    ):
        raise RuntimeError("G1 parallel sealed inputs drifted")
    external._verify_snapshot_and_live(root, manifest or resolved["snapshot"])
    return config


def worker_command(root: Path) -> list[str]:
    resolved = paths(root)
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(external.selected_python(root, ARM_ID)),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def launch(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    verify(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("G1 parallel attempt is not in PREFLIGHT")
    if resolved["arm_result"].exists():
        raise FileExistsError("G1 parallel arm result already exists")
    snapshots = external.gpu_admission(ARM_ID, PHYSICAL_GPU)
    resolved["arm_result"].mkdir(parents=True, exist_ok=False)
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_G1_PARALLEL_GPU_ADMITTED",
        gpu_snapshot=snapshots,
    )
    launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=worker_command(root),
        cwd=root,
        tmux_session=EXPERIMENT_ID,
        startup_log_path=resolved["log"],
    )
    writer.transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP12_G1_PARALLEL_BACKGROUND_STARTED",
        tmux_session=EXPERIMENT_ID,
        gpu_ids=[PHYSICAL_GPU],
        process_alive=True,
        stage="background_started",
    )
    if not wait_for_tmux_startup(EXPERIMENT_ID):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_G1_PARALLEL_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                gpu_ids=[],
            )
        raise RuntimeError("G1 parallel worker exited during startup handshake")
    print(EXPERIMENT_ID)
    return 0


def worker(root: Path, manifest: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    try:
        config = verify(root, manifest)
        original_config = _read(resolved["original_config"])
        checkpoint_record = original_config["checkpoints"][ARM_ID]
        checkpoint = root / checkpoint_record["path"]
        if sha256(checkpoint) != checkpoint_record["sha256"]:
            raise RuntimeError("G1 checkpoint hash drifted")
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP12_G1_PARALLEL_INFERENCE",
            workload_pid=os.getpid(),
            process_alive=True,
            gpu_ids=[PHYSICAL_GPU],
            stage="external_inference_from_sealed_bundle",
        )
        from experiment.phase17.core.full_latte_external_inference import (
            evaluate_external_arm,
        )

        def heartbeat(stage: str, current: int, total: int) -> None:
            writer.heartbeat(
                stage=stage,
                progress={"current": current, "total": total, "unit": "external_user"},
            )

        result = evaluate_external_arm(
            root,
            ARM_ID,
            checkpoint,
            resolved["original_bundle"],
            resolved["predictions"],
            heartbeat=heartbeat,
        )
        summary = {
            "schema_version": "phase17.s17_fp12_external_arm_summary.v1",
            "verdict": "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN",
            "completed_at": utc_now(),
            "attempt_id": ATTEMPT_ID,
            "arm_id": ARM_ID,
            "physical_gpu": PHYSICAL_GPU,
            "checkpoint_path": checkpoint_record["path"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "bundle_path": str(resolved["original_bundle"].relative_to(root)),
            "bundle_sha256": config["bundle_sha256"],
            "predictions_path": str(resolved["predictions"].relative_to(root)),
            "predictions_sha256": sha256(resolved["predictions"]),
            "result": result,
            "post_completion_runtime_guard_authorized": True,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
            "automatic_retry": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN",
            process_alive=False,
            workload_pid=0,
            gpu_ids=[],
            stage="external_predictions_frozen",
            progress={
                "current": result["external_users"],
                "total": result["external_users"],
                "unit": "external_user",
            },
            predictions_path=summary["predictions_path"],
            predictions_sha256=summary["predictions_sha256"],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            result_selection_eligible=True,
        )
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "attempt_id": ATTEMPT_ID,
            "arm_id": ARM_ID,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
        }
        atomic_json(resolved["failure"], failure)
        current = writer.read()
        if current["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_G1_PARALLEL_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                failure_path=str(resolved["failure"].relative_to(root)),
                failure_sha256=sha256(resolved["failure"]),
                terminal_error=repr(error),
            )
        return 1


def analyze(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    config = verify(root)
    sources: dict[str, tuple[Mapping[str, Path], str]] = {
        "N0_NATIVE_PSID": (
            external.arm_paths(root, "N0_NATIVE_PSID"),
            external.arm_experiment_id("N0_NATIVE_PSID"),
        ),
        "N1_NATIVE_LATTE": (
            external.arm_paths(root, "N1_NATIVE_LATTE"),
            external.arm_experiment_id("N1_NATIVE_LATTE"),
        ),
        "G0_GRAM_B0_FRESH": (
            recovery.arm_paths(root, "G0_GRAM_B0_FRESH"),
            recovery.arm_experiment_id("G0_GRAM_B0_FRESH"),
        ),
        ARM_ID: (
            {
                "predictions": resolved["predictions"],
                "summary": resolved["summary"],
            },
            EXPERIMENT_ID,
        ),
        "G2_GRAM_LATTE_FULL": (
            recovery.arm_paths(root, "G2_GRAM_LATTE_FULL"),
            recovery.arm_experiment_id("G2_GRAM_LATTE_FULL"),
        ),
    }
    provenance = {
        "attempt_id": ATTEMPT_ID,
        "researcher_direction": (
            "找一张GPU运行G1；G1完成后在同卡启动隔离重复轮持续占用资源"
        ),
        "authorization_path": str(resolved["authorization"].relative_to(root)),
        "authorization_sha256": sha256(resolved["authorization"]),
        "recovery_provenance_path": str(resolved["provenance"].relative_to(root)),
        "recovery_provenance_sha256": sha256(resolved["provenance"]),
        "attempt_001_arms": ["N0_NATIVE_PSID", "N1_NATIVE_LATTE"],
        "attempt_002_arms": ["G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL"],
        "attempt_003_arms": [ARM_ID],
        "bundle_sha256": config["bundle_sha256"],
        "single_materialization_count": 1,
        "raw_external_projection_reopened": False,
        "automatic_retry": False,
    }
    result = external.analyze_selected(
        root,
        artifact_sources=sources,
        analysis_path=resolved["analysis"],
        manifest_path=resolved["snapshot"],
        recovery_provenance=provenance,
    )
    recovery_writer = StatusWriter(
        external.paths(root)["status_dir"], recovery.EXPERIMENT_ID
    )
    if recovery_writer.read()["scientific_state"] == "RUNNING":
        recovery_writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP12_CONTROLLED_RECOVERY_ANALYZED",
            process_alive=False,
            stage="controlled_recovery_analysis_complete",
            analysis_path=str(resolved["analysis"].relative_to(root)),
            analysis_sha256=sha256(resolved["analysis"]),
            result_selection_eligible=True,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "worker", "analyze"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.root)
    if args.action == "launch":
        return launch(args.root)
    if args.action == "worker":
        if args.manifest is None:
            parser.error("worker requires --manifest")
        return worker(args.root, args.manifest)
    return analyze(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
