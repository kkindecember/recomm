#!/usr/bin/env python3
"""Researcher-authorized recovery for the three interrupted GRAM external arms."""

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
from experiment.phase17.protocol import s17_fp12_external_d0_runtime as external


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "attempt_002"
EXPERIMENT_ID = "s17_fp12_external_d0_recovery"
STEP_ID = "S17-FP12-EXTERNAL-D0-RECOVERY"
RECOVERY_ARM_IDS = (
    "G0_GRAM_B0_FRESH",
    "G1_GRAM_PSID_FULL",
    "G2_GRAM_LATTE_FULL",
)
RECOVERY_GPU_BY_ARM = {
    "G0_GRAM_B0_FRESH": 5,
    "G1_GRAM_PSID_FULL": 5,
    "G2_GRAM_LATTE_FULL": 6,
}
RESULT_SUFFIX = Path("artifacts/phase17/fullport/external_d0/recovery/attempt_002")
AUTHORIZATION_SUFFIX = Path(
    "artifacts/phase17/authorizations/s17_fp12_external_d0_recovery_attempt_002.json"
)
LEDGER_SUFFIX = Path("artifacts/phase17/attempts/S17-FP12-EXTERNAL-D0.attempts.jsonl")
EXPECTED_COMPATIBILITY_ERROR = (
    "TypeError(\"'weights_only' is an invalid keyword argument for Unpickler()\")"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def arm_experiment_id(arm_id: str) -> str:
    if arm_id not in RECOVERY_ARM_IDS:
        raise ValueError(f"arm is outside the controlled recovery: {arm_id}")
    return f"{EXPERIMENT_ID}_{external.arm_slug(arm_id)}"


def paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    result = root / RESULT_SUFFIX
    original = external.paths(root)
    return {
        "result": result,
        "config": result / "config.json",
        "provenance": result / "recovery_provenance.json",
        "g1_admission_failure": result / "g1_attempt_001_admission_failure.json",
        "analysis": original["analysis"],
        "authorization": root / AUTHORIZATION_SUFFIX,
        "status_dir": original["status_dir"],
        "ledger": root / LEDGER_SUFFIX,
        "snapshot": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "original_bundle": original["bundle"],
        "original_seal": original["seal"],
        "original_authorization": original["authorization"],
        "original_config": original["config"],
    }


def arm_paths(root: Path, arm_id: str) -> dict[str, Path]:
    resolved = paths(root)
    result = resolved["result"] / "arms" / external.arm_slug(arm_id)
    return {
        "result": result,
        "predictions": result / "predictions.jsonl",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "snapshot_worker": resolved["snapshot"].parent
        / "src/000_s17_fp12_external_d0_recovery_runtime.py",
    }


def source_paths(root: Path) -> list[Path]:
    original_sources = external.source_paths(root)
    return [Path(__file__).resolve(), *original_sources]


def _original_failure_evidence(root: Path, arm_id: str) -> dict[str, Any]:
    original = external.arm_paths(root, arm_id)
    status_path = external.paths(root)["status_dir"] / (
        f"{external.arm_experiment_id(arm_id)}.status.json"
    )
    if not status_path.is_file() or not original["failure"].is_file():
        raise RuntimeError(f"attempt_001 failure evidence is missing for {arm_id}")
    status = _read(status_path)
    failure = _read(original["failure"])
    if (
        status.get("scientific_state") != "FAILED"
        or status.get("status_code") != "S17_FP12_EXTERNAL_ARM_FAILED_NO_RETRY"
        or status.get("terminal_error") != EXPECTED_COMPATIBILITY_ERROR
        or failure.get("error") != EXPECTED_COMPATIBILITY_ERROR
        or failure.get("automatic_retry") is not False
        or original["predictions"].exists()
        or original["summary"].exists()
    ):
        raise RuntimeError(f"attempt_001 is not an eligible pre-prediction failure: {arm_id}")
    return {
        "arm_id": arm_id,
        "status_path": str(status_path.relative_to(root)),
        "status_sha256": sha256(status_path),
        "failure_path": str(original["failure"].relative_to(root)),
        "failure_sha256": sha256(original["failure"]),
        "terminal_error": EXPECTED_COMPATIBILITY_ERROR,
        "predictions_created": False,
    }


def verify_attempt_001_evidence(root: Path) -> dict[str, Any]:
    """Verify recovery eligibility without reopening the raw external projection."""

    root = root.resolve()
    original = external.paths(root)
    seal = external.verify_seal(root)
    config = _read(original["config"])
    checkpoint_evidence = external.verify_checkpoint_freeze(root, config)
    failed = {
        arm_id: _original_failure_evidence(root, arm_id)
        for arm_id in ("G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL")
    }
    g1 = external.arm_paths(root, "G1_GRAM_PSID_FULL")
    g1_status = original["status_dir"] / (
        f"{external.arm_experiment_id('G1_GRAM_PSID_FULL')}.status.json"
    )
    if g1_status.exists() or g1["result"].exists():
        raise RuntimeError("G1 attempt_001 unexpectedly has worker artifacts")
    native: dict[str, Any] = {}
    for arm_id in ("N0_NATIVE_PSID", "N1_NATIVE_LATTE"):
        status_path = original["status_dir"] / (
            f"{external.arm_experiment_id(arm_id)}.status.json"
        )
        status = _read(status_path)
        if status.get("scientific_state") not in {"RUNNING", "COMPLETED"}:
            raise RuntimeError(f"native attempt_001 arm is not healthy: {arm_id}")
        native[arm_id] = {
            "status_path": str(status_path.relative_to(root)),
            "status_sha256_at_recovery_prepare": sha256(status_path),
            "scientific_state": status["scientific_state"],
            "left_untouched": True,
        }
    return {
        "schema_version": "phase17.s17_fp12_recovery_evidence.v1",
        "verified_at": utc_now(),
        "original_attempt_id": external.ATTEMPT_ID,
        "original_authorization_path": str(original["authorization"].relative_to(root)),
        "original_authorization_sha256": sha256(original["authorization"]),
        "materialization_seal_path": str(original["seal"].relative_to(root)),
        "materialization_seal_sha256": sha256(original["seal"]),
        "bundle_path": seal["bundle_path"],
        "bundle_sha256": seal["bundle_sha256"],
        "single_materialization_count": seal["single_materialization_count"],
        "failed_before_predictions": failed,
        "g1_attempt_001": {
            "worker_started": False,
            "predictions_created": False,
            "reason": "GPU0 failed the 18968 MiB two-snapshot admission gate",
        },
        "native_attempt_001": native,
        "checkpoint_evidence": checkpoint_evidence,
        "raw_external_projection_reopened": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }


def prepare(root: Path, researcher_direction: str) -> int:
    root = root.resolve()
    resolved = paths(root)
    status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if (
        resolved["result"].exists()
        or resolved["authorization"].exists()
        or resolved["snapshot"].exists()
        or status_path.exists()
    ):
        raise FileExistsError("controlled recovery attempt_002 already exists")
    if researcher_direction.strip() != "同意受控恢复":
        raise PermissionError("exact researcher direction '同意受控恢复' is required")
    evidence = verify_attempt_001_evidence(root)
    resolved["result"].mkdir(parents=True, exist_ok=False)
    authorization = {
        "schema_version": "phase17.s17_fp12_recovery_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "researcher_direction": researcher_direction.strip(),
        "authorized_arms": list(RECOVERY_ARM_IDS),
        "physical_gpu_by_arm": RECOVERY_GPU_BY_ARM,
        "reuse_sealed_bundle_only": True,
        "raw_external_projection_reopen_authorized": False,
        "preserve_native_attempt_001_workers": True,
        "preserve_all_preexisting_compute_processes": True,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["authorization"], authorization)
    recovery_config = {
        "schema_version": "phase17.s17_fp12_recovery_config.v1",
        "attempt_id": ATTEMPT_ID,
        "original_attempt_id": external.ATTEMPT_ID,
        "authorized_arms": list(RECOVERY_ARM_IDS),
        "physical_gpu_by_arm": RECOVERY_GPU_BY_ARM,
        "scheduling": {
            "parallel_first": ["G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL"],
            "serial_after_g0": "G1_GRAM_PSID_FULL",
        },
        "compatibility_change": (
            "inspect torch.load signature and pass weights_only=False only when supported"
        ),
        "original_config_path": str(resolved["original_config"].relative_to(root)),
        "original_config_sha256": sha256(resolved["original_config"]),
        "bundle_path": evidence["bundle_path"],
        "bundle_sha256": evidence["bundle_sha256"],
        "materialization_seal_path": evidence["materialization_seal_path"],
        "materialization_seal_sha256": evidence["materialization_seal_sha256"],
        "authorization_path": str(resolved["authorization"].relative_to(root)),
        "authorization_sha256": sha256(resolved["authorization"]),
        "single_materialization_count": 1,
        "raw_external_projection_reopened": False,
    }
    atomic_json(resolved["config"], recovery_config)
    atomic_json(resolved["provenance"], evidence)
    atomic_json(
        resolved["g1_admission_failure"],
        {
            "schema_version": "phase17.s17_fp12_admission_failure.v1",
            "recorded_at": utc_now(),
            "attempt_id": external.ATTEMPT_ID,
            "arm_id": "G1_GRAM_PSID_FULL",
            "physical_gpu": 0,
            "minimum_free_mib": external.MINIMUM_FREE_MIB["G1_GRAM_PSID_FULL"],
            "worker_started": False,
            "predictions_created": False,
            "reason": evidence["g1_attempt_001"]["reason"],
            "evidence_kind": "control_plane_launcher_exception_reconstruction",
        },
    )
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=["researcher_authorized_controlled_recovery"],
        source_paths=source_paths(root),
        config=recovery_config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "experiment_id": EXPERIMENT_ID,
            "step_id": STEP_ID,
            "kind": "researcher_authorized_infrastructure_recovery",
            "started_at": utc_now(),
            "state": "RECOVERY_PREFLIGHT_COMPLETE_WAITING_GPU",
            "scientific_result_eligible": True,
            "snapshot_manifest": str(manifest.relative_to(root)),
            "original_attempt_preserved": True,
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
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=None,
        extra={
            "stage": "controlled_recovery_preflight",
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "authorization_path": str(resolved["authorization"].relative_to(root)),
            "authorization_sha256": sha256(resolved["authorization"]),
            "recovery_provenance_path": str(resolved["provenance"].relative_to(root)),
            "recovery_provenance_sha256": sha256(resolved["provenance"]),
            "authorized_arms": list(RECOVERY_ARM_IDS),
            "physical_gpu_by_arm": RECOVERY_GPU_BY_ARM,
            "external_target_materialized": True,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
            "controlled_recovery": True,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "result_selection_eligible": True,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_RECOVERY_PREFLIGHT_COMPLETE",
        process_alive=False,
    )
    writer.transition(
        "RUNNING",
        "WAITING_FOR_GPU",
        "S17_FP12_RECOVERY_AUTHORIZED_WAITING_ARM_LAUNCH",
        process_alive=False,
    )
    print(json.dumps(recovery_config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_recovery(root: Path, manifest: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    config = _read(resolved["config"])
    authorization = _read(resolved["authorization"])
    if (
        authorization.get("researcher_direction") != "同意受控恢复"
        or authorization.get("reuse_sealed_bundle_only") is not True
        or authorization.get("raw_external_projection_reopen_authorized") is not False
        or authorization.get("automatic_retry") is not False
        or tuple(authorization.get("authorized_arms", ())) != RECOVERY_ARM_IDS
    ):
        raise PermissionError("controlled recovery authorization drifted")
    seal = external.verify_seal(root)
    if (
        sha256(resolved["original_bundle"]) != config["bundle_sha256"]
        or seal["bundle_sha256"] != config["bundle_sha256"]
        or sha256(resolved["original_seal"]) != config["materialization_seal_sha256"]
        or sha256(resolved["authorization"]) != config["authorization_sha256"]
    ):
        raise RuntimeError("controlled recovery sealed inputs drifted")
    external._verify_snapshot_and_live(root, manifest or resolved["snapshot"])
    return config


def worker_command(root: Path, arm_id: str, gpu: int) -> list[str]:
    resolved = paths(root)
    arm = arm_paths(root, arm_id)
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(external.selected_python(root, arm_id)),
        str(arm["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--arm",
        arm_id,
        "--manifest",
        str(resolved["snapshot"]),
    ]


def _ensure_serial_schedule(root: Path, arm_id: str, gpu: int) -> None:
    resolved = paths(root)
    for other in RECOVERY_ARM_IDS:
        if other == arm_id or RECOVERY_GPU_BY_ARM[other] != gpu:
            continue
        status_path = resolved["status_dir"] / f"{arm_experiment_id(other)}.status.json"
        if status_path.is_file() and _read(status_path)["scientific_state"] == "RUNNING":
            raise RuntimeError(f"GPU{gpu} already hosts recovery arm {other}")
    if arm_id == "G1_GRAM_PSID_FULL":
        g0_status = resolved["status_dir"] / (
            f"{arm_experiment_id('G0_GRAM_B0_FRESH')}.status.json"
        )
        if (
            not g0_status.is_file()
            or _read(g0_status).get("scientific_state") != "COMPLETED"
        ):
            raise RuntimeError("G1 recovery must wait for G0 completion on shared GPU5")


def launch_arm(root: Path, arm_id: str) -> int:
    root = root.resolve()
    if arm_id not in RECOVERY_ARM_IDS:
        raise ValueError("only the three GRAM arms are authorized for recovery")
    resolved = paths(root)
    config = verify_recovery(root)
    arm = arm_paths(root, arm_id)
    if arm["result"].exists():
        raise FileExistsError(f"recovery arm result already exists: {arm_id}")
    gpu = int(config["physical_gpu_by_arm"][arm_id])
    _ensure_serial_schedule(root, arm_id, gpu)
    if not external.selected_python(root, arm_id).is_file():
        raise FileNotFoundError(f"frozen {arm_id} Python environment is missing")
    snapshots = external.gpu_admission(arm_id, gpu)
    arm["result"].mkdir(parents=True, exist_ok=False)
    writer = StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id))
    writer.initialize(
        step_id=STEP_ID,
        attempt_id=ATTEMPT_ID,
        track_id=arm_id,
        canonical_result_dir=str(arm["result"].relative_to(root)),
        log_path=str(arm["log"].relative_to(root)),
        extra={
            "stage": "controlled_recovery_arm_preflight",
            "original_attempt_preserved": True,
            "external_target_materialized": True,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
            "target_gpu_id": gpu,
            "controlled_recovery": True,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_RECOVERY_ARM_READY",
        gpu_snapshot=snapshots,
    )
    session = arm_experiment_id(arm_id)
    launch_background_tmux(
        experiment_id=session,
        argv=worker_command(root, arm_id, gpu),
        cwd=root,
        tmux_session=session,
        startup_log_path=arm["log"],
    )
    writer.transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP12_RECOVERY_ARM_BACKGROUND_STARTED",
        tmux_session=session,
        gpu_ids=[gpu],
        process_alive=True,
        stage="background_started",
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_RECOVERY_ARM_STARTUP_FAILED_NO_AUTOMATIC_RETRY",
                process_alive=False,
                gpu_ids=[],
            )
        raise RuntimeError("recovery worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, arm_id: str, manifest: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    arm = arm_paths(root, arm_id)
    writer = StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id))
    try:
        config = verify_recovery(root, manifest)
        original_config = _read(resolved["original_config"])
        checkpoint_record = original_config["checkpoints"][arm_id]
        checkpoint = root / checkpoint_record["path"]
        if sha256(checkpoint) != checkpoint_record["sha256"]:
            raise RuntimeError(f"checkpoint hash drift for {arm_id}")
        gpu = int(config["physical_gpu_by_arm"][arm_id])
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP12_RECOVERY_ARM_INFERENCE",
            workload_pid=os.getpid(),
            process_alive=True,
            gpu_ids=[gpu],
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
            arm_id,
            checkpoint,
            resolved["original_bundle"],
            arm["predictions"],
            heartbeat=heartbeat,
        )
        summary = {
            "schema_version": "phase17.s17_fp12_external_arm_summary.v1",
            "verdict": "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN",
            "completed_at": utc_now(),
            "attempt_id": ATTEMPT_ID,
            "recovered_from_attempt_id": external.ATTEMPT_ID,
            "arm_id": arm_id,
            "physical_gpu": gpu,
            "checkpoint_path": checkpoint_record["path"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "bundle_path": str(resolved["original_bundle"].relative_to(root)),
            "bundle_sha256": config["bundle_sha256"],
            "predictions_path": str(arm["predictions"].relative_to(root)),
            "predictions_sha256": sha256(arm["predictions"]),
            "result": result,
            "controlled_recovery": True,
            "researcher_authorized": True,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
            "automatic_retry": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        atomic_json(arm["summary"], summary)
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
            summary_path=str(arm["summary"].relative_to(root)),
            summary_sha256=sha256(arm["summary"]),
            result_selection_eligible=True,
        )
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "attempt_id": ATTEMPT_ID,
            "arm_id": arm_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "controlled_recovery": True,
            "automatic_retry": False,
            "single_materialization_count": 1,
            "raw_external_projection_reopened": False,
        }
        atomic_json(arm["failure"], failure)
        current = writer.read()
        if current["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_RECOVERY_ARM_FAILED_NO_AUTOMATIC_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                failure_path=str(arm["failure"].relative_to(root)),
                failure_sha256=sha256(arm["failure"]),
                terminal_error=repr(error),
                automatic_retry=False,
            )
        return 1


def analyze(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    config = verify_recovery(root)
    sources: dict[str, tuple[Mapping[str, Path], str]] = {}
    for arm_id in ("N0_NATIVE_PSID", "N1_NATIVE_LATTE"):
        sources[arm_id] = (
            external.arm_paths(root, arm_id),
            external.arm_experiment_id(arm_id),
        )
    for arm_id in RECOVERY_ARM_IDS:
        sources[arm_id] = (arm_paths(root, arm_id), arm_experiment_id(arm_id))
    provenance = {
        "attempt_id": ATTEMPT_ID,
        "researcher_direction": "同意受控恢复",
        "authorization_path": str(resolved["authorization"].relative_to(root)),
        "authorization_sha256": sha256(resolved["authorization"]),
        "recovery_provenance_path": str(resolved["provenance"].relative_to(root)),
        "recovery_provenance_sha256": sha256(resolved["provenance"]),
        "recovered_arms": list(RECOVERY_ARM_IDS),
        "preserved_attempt_001_arms": ["N0_NATIVE_PSID", "N1_NATIVE_LATTE"],
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
    StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
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


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    output: dict[str, Any] = {
        "attempt_id": ATTEMPT_ID,
        "prepared": resolved["result"].is_dir(),
        "authorized": resolved["authorization"].is_file(),
        "analyzed": resolved["analysis"].is_file(),
        "arms": {},
    }
    family_status = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    output["status"] = _read(family_status) if family_status.is_file() else None
    for arm_id in RECOVERY_ARM_IDS:
        arm = arm_paths(root, arm_id)
        status = resolved["status_dir"] / f"{arm_experiment_id(arm_id)}.status.json"
        output["arms"][arm_id] = {
            "predictions": arm["predictions"].is_file(),
            "summary": arm["summary"].is_file(),
            "failure": arm["failure"].is_file(),
            "status": _read(status) if status.is_file() else None,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "launch-arm", "worker", "analyze", "inspect")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=RECOVERY_ARM_IDS)
    parser.add_argument("--researcher-direction", default="")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.root, args.researcher_direction)
    if args.action == "launch-arm":
        if args.arm is None:
            parser.error("launch-arm requires --arm")
        return launch_arm(args.root, args.arm)
    if args.action == "worker":
        if args.arm is None or args.manifest is None:
            parser.error("worker requires --arm and --manifest")
        return worker(args.root, args.arm, args.manifest)
    if args.action == "analyze":
        return analyze(args.root)
    print(json.dumps(inspect(args.root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
