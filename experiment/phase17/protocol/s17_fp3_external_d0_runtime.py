#!/usr/bin/env python3
"""Authorization-gated one-shot external D0 evaluation for Stage17 FP3."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from experiment.phase17.core.full_latte_external_evaluator import (
    aggregate_metrics,
    compare_predictions,
    prediction_variants,
    read_jsonl,
    subgroup_assignments,
    subgroup_comparison,
    validate_prediction_rows,
)
from experiment.phase17.core.full_setrec_backend import SETREC_ARMS
from experiment.phase17.core.full_setrec_external import (
    evaluate_external_arm,
    fp3_gate,
    read_sealed_bundle_views,
    summarize_mechanisms,
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
EXPERIMENT_ID = "s17_fp3_external_d0"
ATTEMPT_ID = "attempt_001"
STEP_ID = "S17-FP3-EXTERNAL-D0"
CONFIG_SUFFIX = Path("experiment/phase17/config/s17_fp3_external_d0.json")
RESULT_SUFFIX = Path(
    "artifacts/phase17/fullport/fp3_setrec/external_d0/attempt_001"
)
AUTHORIZATION_SUFFIX = Path(
    "artifacts/phase17/authorizations/s17_fp3_external_d0_attempt_001.json"
)
LEDGER_SUFFIX = Path(
    "artifacts/phase17/attempts/S17-FP3-EXTERNAL-D0.attempts.jsonl"
)
REPORT_SUFFIX = Path(
    "report/第十七阶段/Stage17_FP3_FullSETRec正式结果报告.md"
)
NATIVE_PYTHON_SUFFIX = gpu_runtime.NATIVE_PYTHON_SUFFIX


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def arm_slug(arm_id: str) -> str:
    if arm_id not in SETREC_ARMS:
        raise ValueError(f"unknown FP3 arm: {arm_id}")
    return arm_id.lower()


def arm_experiment_id(arm_id: str) -> str:
    return f"{EXPERIMENT_ID}_{arm_slug(arm_id)}"


def paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    result = root / RESULT_SUFFIX
    snapshot_path = (
        root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json"
    )
    return {
        "source_config": root / CONFIG_SUFFIX,
        "result": result,
        "config": result / "config.json",
        "readiness": result / "readiness.json",
        "analysis": result / "analysis.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "authorization": root / AUTHORIZATION_SUFFIX,
        "ledger": root / LEDGER_SUFFIX,
        "status_dir": root / "artifacts/phase17/status",
        "snapshot": snapshot_path,
        "snapshot_worker": snapshot_path.parent
        / "src/000_s17_fp3_external_d0_runtime.py",
        "native_python": root / NATIVE_PYTHON_SUFFIX,
        "report": root / REPORT_SUFFIX,
    }


def arm_paths(root: Path, arm_id: str) -> dict[str, Path]:
    base = paths(root)
    result = base["result"] / "arms" / arm_slug(arm_id)
    return {
        "result": result,
        "predictions": result / "predictions.jsonl",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
    }


def source_paths(root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        root / "experiment/phase17/core/full_setrec_external.py",
        root / "experiment/phase17/core/full_setrec_executor.py",
        root / "experiment/phase17/core/full_setrec_backend.py",
        root / "experiment/phase17/core/full_setrec_contracts.py",
        root / "experiment/phase17/core/full_latte_external_evaluator.py",
        root / "experiment/phase17/core/full_latte_external_inference.py",
        root / "experiment/phase17/core/full_latte_gram_backend.py",
        root / "experiment/phase17/core/full_latte_native_adapter.py",
        root / "experiment/phase17/core/fullport_data.py",
        root / "experiment/phase17/core/status_writer.py",
        root / "experiment/phase17/core/run_manager.py",
    ]


def _verify_snapshot_and_live(root: Path, manifest_path: Path) -> None:
    verify_run_snapshot(root, manifest_path)
    manifest = _read(manifest_path)
    for record in manifest["files"]:
        live = root / record["source_path"]
        if sha256(live) != record["sha256"]:
            raise RuntimeError(
                f"live source drifted after FP3 external preflight: {record['source_path']}"
            )


def _gpu_state() -> dict[str, Any]:
    return gpu_runtime._gpu_state()


def _gpu(snapshot: Mapping[str, Any], gpu_id: int) -> dict[str, Any]:
    matches = [row for row in snapshot["devices"] if int(row["index"]) == gpu_id]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU{gpu_id} is not uniquely visible")
    return dict(matches[0])


def verify_checkpoint_freeze(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if set(config["checkpoints"]) != set(SETREC_ARMS):
        raise RuntimeError("FP3 external checkpoint inventory is not the four arms")
    evidence: dict[str, Any] = {}
    for arm_id in SETREC_ARMS:
        record = config["checkpoints"][arm_id]
        status_path = root / record["status_path"]
        summary_path = root / record["summary_path"]
        checkpoint_path = root / record["path"]
        status = _read(status_path)
        summary = _read(summary_path)
        internal = summary["training"]["best_internal_dev"]
        if (
            status.get("attempt_id") != "attempt_002"
            or status.get("scientific_state") != "COMPLETED"
            or status.get("status_code") != "PASS_S17_FP3_FORMAL_CHECKPOINT_FROZEN"
            or status.get("checkpoint_frozen") is not True
            or status.get("external_target_materialized") is not False
            or status.get("checkpoint_path") != record["path"]
            or status.get("checkpoint_sha256") != record["sha256"]
            or summary.get("verdict") != "PASS_S17_FP3_FORMAL_CHECKPOINT_FROZEN"
            or summary.get("best_checkpoint_sha256") != record["sha256"]
            or int(summary["training"]["best_epoch"]) != int(record["best_epoch"])
            or float(internal["selected_beta"]) != float(record["selected_beta"])
            or sha256(summary_path) != record["summary_sha256"]
            or sha256(checkpoint_path) != record["sha256"]
        ):
            raise RuntimeError(f"frozen FP3 checkpoint evidence drifted for {arm_id}")
        evidence[arm_id] = {
            "checkpoint_path": record["path"],
            "checkpoint_sha256": record["sha256"],
            "status_path": record["status_path"],
            "status_sha256": sha256(status_path),
            "summary_path": record["summary_path"],
            "summary_sha256": record["summary_sha256"],
            "best_epoch": int(record["best_epoch"]),
            "selected_beta": float(record["selected_beta"]),
            "internal_dev_full_set_recovery_rate": float(
                internal["full_set_recovery_rate"]
            ),
        }
    return evidence


def verify_sealed_bundle_without_content_read(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    data = config["data"]
    seal_path = root / data["upstream_seal_path"]
    bundle_path = root / data["sealed_bundle_path"]
    seal = _read(seal_path)
    if (
        sha256(seal_path) != data["upstream_seal_sha256"]
        or seal.get("bundle_path") != data["sealed_bundle_path"]
        or seal.get("bundle_sha256") != data["sealed_bundle_sha256"]
        or int(seal.get("external_users", -1)) != int(data["expected_users"])
        or int(seal.get("single_materialization_count", -1)) != 1
        or seal.get("guard_item_serialized") is not False
        or seal.get("test_read") is not False
        or seal.get("sports_read") is not False
        or not bundle_path.is_file()
    ):
        raise RuntimeError("upstream sealed D0 bundle evidence drifted")
    return {
        "bundle_path": data["sealed_bundle_path"],
        "expected_bundle_sha256": data["sealed_bundle_sha256"],
        "bundle_size_bytes": bundle_path.stat().st_size,
        "seal_path": data["upstream_seal_path"],
        "seal_sha256": data["upstream_seal_sha256"],
        "upstream_single_materialization_count": 1,
        "bundle_content_read": False,
        "raw_external_projection_reopened": False,
    }


def verify_g0_control_without_prediction_read(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    record = config["g0_control"]
    status_path = root / record["status_path"]
    summary_path = root / record["summary_path"]
    predictions_path = root / record["predictions_path"]
    checkpoint_path = root / record["checkpoint_path"]
    status = _read(status_path)
    summary = _read(summary_path)
    if (
        sha256(status_path) != record["status_sha256"]
        or sha256(summary_path) != record["summary_sha256"]
        or sha256(checkpoint_path) != record["checkpoint_sha256"]
        or status.get("scientific_state") != "COMPLETED"
        or status.get("status_code")
        != "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN"
        or summary.get("verdict")
        != "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN"
        or summary.get("predictions_path") != record["predictions_path"]
        or summary.get("predictions_sha256") != record["predictions_sha256"]
        or not predictions_path.is_file()
    ):
        raise RuntimeError("frozen fresh G0 control evidence drifted")
    return {
        "predictions_path": record["predictions_path"],
        "expected_predictions_sha256": record["predictions_sha256"],
        "predictions_size_bytes": predictions_path.stat().st_size,
        "prediction_content_read": False,
        "summary_path": record["summary_path"],
        "status_path": record["status_path"],
    }


def verify_readiness(root: Path) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    config = _read(resolved["source_config"])
    if (
        config.get("state") != "PREPARED_NOT_AUTHORIZED"
        or config["authorization"].get("external_bundle_content_read_authorized")
        is not False
        or config["authorization"].get("gpu_execution_authorized") is not False
        or config["data"].get("raw_external_projection_reopen_forbidden") is not True
    ):
        raise PermissionError("FP3 external source config is not sealed/unauthorized")
    source_config = config["formal_source_config"]
    if sha256(root / source_config["path"]) != source_config["sha256"]:
        raise RuntimeError("FP3 formal preregistration config drifted")
    if not resolved["native_python"].is_file():
        raise FileNotFoundError(resolved["native_python"])
    return {
        "schema_version": "phase17.s17_fp3_external_readiness.v1",
        "verdict": "READY_AWAITING_EXPLICIT_GPU_AND_BUNDLE_READ_AUTHORIZATION",
        "verified_at": utc_now(),
        "source_config_sha256": sha256(resolved["source_config"]),
        "checkpoint_evidence": verify_checkpoint_freeze(root, config),
        "sealed_bundle_evidence": verify_sealed_bundle_without_content_read(root, config),
        "g0_control_evidence": verify_g0_control_without_prediction_read(root, config),
        "external_bundle_content_read": False,
        "raw_external_projection_reopened": False,
        "gpu_used": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }


def prepare(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    family_status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if resolved["result"].exists() or resolved["snapshot"].exists() or family_status_path.exists():
        raise FileExistsError("FP3 external D0 attempt_001 is already prepared")
    readiness = verify_readiness(root)
    resolved["result"].mkdir(parents=True, exist_ok=False)
    frozen = _read(resolved["source_config"])
    frozen["source_config_sha256"] = readiness["source_config_sha256"]
    frozen["prepared_at"] = utc_now()
    atomic_json(resolved["config"], frozen)
    atomic_json(resolved["readiness"], readiness)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=["future_authorized_fp3_external_family_worker"],
        source_paths=source_paths(root),
        config=frozen,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "experiment_id": EXPERIMENT_ID,
            "step_id": STEP_ID,
            "kind": "one_shot_external_d0_family_evaluation",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_AUTHORIZATION_REQUIRED",
            "scientific_result_eligible": True,
            "snapshot_manifest": str(manifest.relative_to(root)),
            "external_bundle_content_read": False,
            "raw_external_projection_reopened": False,
            "automatic_retry": False,
        }
    )
    family_writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    family_writer.initialize(
        step_id=STEP_ID,
        attempt_id=ATTEMPT_ID,
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "preflight_complete_waiting_explicit_gpu_and_bundle_read_authorization",
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "readiness_path": str(resolved["readiness"].relative_to(root)),
            "readiness_sha256": sha256(resolved["readiness"]),
            "external_bundle_content_read_authorized": False,
            "gpu_execution_authorized": False,
            "external_bundle_content_read": False,
            "raw_external_projection_reopened": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "result_selection_eligible": True,
            "d1_read": False,
            "d2_read": False,
        },
    )
    family_writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP3_EXTERNAL_D0_READY_AUTHORIZATION_REQUIRED",
        process_alive=False,
    )
    for arm_id in SETREC_ARMS:
        arm = arm_paths(root, arm_id)
        arm["result"].mkdir(parents=True, exist_ok=False)
        writer = StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id))
        writer.initialize(
            step_id=STEP_ID,
            attempt_id=ATTEMPT_ID,
            track_id=arm_id,
            canonical_result_dir=str(arm["result"].relative_to(root)),
            log_path=str(resolved["log"].relative_to(root)),
            extra={
                "stage": "family_preflight_waiting_authorization",
                "progress": {"current": 0, "total": 12833, "unit": "external_user"},
                "run_snapshot_manifest": str(manifest.relative_to(root)),
                "external_bundle_content_read": False,
                "raw_external_projection_reopened": False,
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
            "S17_FP3_EXTERNAL_ARM_READY_AUTHORIZATION_REQUIRED",
            process_alive=False,
        )
    print(json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def authorize(root: Path, gpu_id: int, researcher_direction: str) -> int:
    root = root.resolve()
    resolved = paths(root)
    if gpu_id < 0 or gpu_id == 1:
        raise ValueError("FP3 external attempt_001 requires one non-GPU1 physical GPU")
    if not researcher_direction.strip():
        raise ValueError("explicit researcher direction is required")
    if resolved["authorization"].exists():
        raise FileExistsError(resolved["authorization"])
    family_writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    if family_writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("FP3 external family is not in authorizable PREFLIGHT")
    _verify_snapshot_and_live(root, resolved["snapshot"])
    config = _read(resolved["config"])
    verify_checkpoint_freeze(root, config)
    verify_sealed_bundle_without_content_read(root, config)
    verify_g0_control_without_prediction_read(root, config)
    live = _gpu_state()
    _gpu(live, gpu_id)
    payload = {
        "schema_version": "phase17.s17_fp3_external_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "researcher_direction": researcher_direction.strip(),
        "physical_gpu": int(gpu_id),
        "minimum_free_mib": int(config["resources"]["minimum_free_mib"]),
        "confirmed_exact_command": (
            "bash experiment/phase17/run_stage17_fp3_external_d0.sh launch"
        ),
        "external_bundle_content_read_authorized": True,
        "gpu_execution_authorized": True,
        "raw_external_projection_reopen_authorized": False,
        "preserve_all_preexisting_compute_processes": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "authorization_snapshot": live,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["authorization"], payload)
    family_writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP3_EXTERNAL_D0_AUTHORIZED_READY_TO_LAUNCH",
        external_bundle_content_read_authorized=True,
        gpu_execution_authorized=True,
        physical_gpu=gpu_id,
        minimum_free_mib=payload["minimum_free_mib"],
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        gpu_snapshot=live,
    )
    for arm_id in SETREC_ARMS:
        StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id)).transition(
            "PREFLIGHT",
            "PREFLIGHT",
            "S17_FP3_EXTERNAL_ARM_AUTHORIZED_WAITING_FAMILY_LAUNCH",
            physical_gpu=gpu_id,
            external_bundle_content_read_authorized=True,
            gpu_execution_authorized=True,
            authorization_path=str(resolved["authorization"].relative_to(root)),
            authorization_sha256=sha256(resolved["authorization"]),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_authorization(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    payload = _read(resolved["authorization"])
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "external_bundle_content_read_authorized": True,
        "gpu_execution_authorized": True,
        "raw_external_projection_reopen_authorized": False,
        "preserve_all_preexisting_compute_processes": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"invalid FP3 external authorization field: {key}")
    if int(payload.get("physical_gpu", -1)) < 0 or int(payload["physical_gpu"]) == 1:
        raise PermissionError("FP3 external authorization requires a non-GPU1 card")
    if not str(payload.get("researcher_direction", "")).strip():
        raise PermissionError("FP3 external authorization lacks researcher direction")
    return payload


def two_snapshot_admission(root: Path) -> dict[str, Any]:
    authorization = verify_authorization(root)
    gpu_id = int(authorization["physical_gpu"])
    minimum = int(authorization["minimum_free_mib"])
    first = _gpu_state()
    time.sleep(5)
    second = _gpu_state()
    for label, value in (("first", first), ("second", second)):
        free_mib = int(_gpu(value, gpu_id)["free_mib"])
        if free_mib < minimum:
            raise RuntimeError(
                f"GPU{gpu_id} {label} free {free_mib} MiB is below {minimum} MiB"
            )
    return {
        "captured_at": utc_now(),
        "physical_gpu": gpu_id,
        "minimum_free_mib": minimum,
        "first": first,
        "second": second,
        "preexisting_compute_processes": second["compute_processes"].get(gpu_id, []),
        "preserve_all_preexisting_compute_processes": True,
        "automatic_process_termination": False,
    }


def worker_command(root: Path, gpu_id: int) -> list[str]:
    resolved = paths(root)
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
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


def launch(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("FP3 external family is not launchable")
    _verify_snapshot_and_live(root, resolved["snapshot"])
    authorization = verify_authorization(root)
    admission = two_snapshot_admission(root)
    gpu_id = int(authorization["physical_gpu"])
    session = EXPERIMENT_ID
    launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=worker_command(root, gpu_id),
        cwd=root,
        tmux_session=session,
        startup_log_path=resolved["log"],
    )
    writer.transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP3_EXTERNAL_D0_BACKGROUND_STARTED",
        process_alive=True,
        launcher_pid=os.getpid(),
        tmux_session=session,
        gpu_ids=[gpu_id],
        gpu_snapshot=admission,
        stage="background_worker_started_bundle_still_unread_by_launcher",
        external_bundle_content_read=False,
        raw_external_projection_reopened=False,
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP3_EXTERNAL_D0_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
            )
        raise RuntimeError("FP3 external family worker exited during startup")
    print(
        json.dumps(
            {"tmux_session": session, "physical_gpu": gpu_id},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _verified_g0_predictions(
    root: Path,
    config: Mapping[str, Any],
    examples: list[Any],
) -> dict[str, dict[str, Any]]:
    record = config["g0_control"]
    predictions = root / record["predictions_path"]
    if sha256(predictions) != record["predictions_sha256"]:
        raise RuntimeError("frozen G0 prediction hash drifted")
    grouped = prediction_variants(read_jsonl(predictions))
    variant = config["inference"]["primary_variant_by_arm"]["G0_GRAM_B0_FRESH"]
    return validate_prediction_rows(
        grouped[variant],
        examples,
        expected_arm_id="G0_GRAM_B0_FRESH",
        expected_variant=variant,
    )


def _verified_setrec_predictions(
    root: Path,
    arm_id: str,
    config: Mapping[str, Any],
    examples: list[Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved = paths(root)
    arm = arm_paths(root, arm_id)
    status = StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id)).read()
    summary = _read(arm["summary"])
    if (
        status.get("scientific_state") != "COMPLETED"
        or status.get("status_code")
        != "PASS_S17_FP3_EXTERNAL_ARM_PREDICTIONS_FROZEN"
        or summary.get("verdict")
        != "PASS_S17_FP3_EXTERNAL_ARM_PREDICTIONS_FROZEN"
        or summary.get("predictions_sha256") != sha256(arm["predictions"])
        or summary.get("checkpoint_sha256")
        != config["checkpoints"][arm_id]["sha256"]
    ):
        raise RuntimeError(f"FP3 external prediction evidence drifted for {arm_id}")
    grouped = prediction_variants(read_jsonl(arm["predictions"]))
    variant = config["inference"]["primary_variant_by_arm"][arm_id]
    return (
        validate_prediction_rows(
            grouped[variant],
            examples,
            expected_arm_id=arm_id,
            expected_variant=variant,
        ),
        summary,
    )


def _fallacy_scan(integrity_valid: bool) -> dict[str, Any]:
    return {
        "coverage": "11/11 checked",
        "items": {
            "simpsons_paradox": "NOTE: overall and preregistered subgroups are retained for direction checks",
            "ecological_fallacy": "NOTE: inference is paired at the user level",
            "berksons_paradox": "NOTE: the full frozen D0 cohort is used without efficacy-based selection",
            "collider_bias": "NOTE: no post-treatment covariate conditioning is used",
            "base_rate_neglect": "NOTE: not a diagnostic-classification study",
            "regression_to_mean": "NOTE: checkpoint selection used train-prefix internal dev; D0 is external",
            "survivorship_bias": (
                "NOTE: exact complete user alignment"
                if integrity_valid
                else "RED_FLAG: user-alignment integrity failed"
            ),
            "look_elsewhere_effect": "NOTE: primary contrasts and thresholds were preregistered",
            "garden_of_forking_paths": "CAUTION: engineering attempts exist, but beta and best checkpoints were frozen before D0",
            "correlation_not_causation": "NOTE: matched arm interventions support only this implementation and fold",
            "reverse_causality": "NOTE: not applicable to assigned model-arm comparisons",
        },
    }


def analyze_from_views(
    root: Path,
    train_users: list[Any],
    examples: list[Any],
) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    config = _read(resolved["config"])
    if len(examples) != int(config["data"]["expected_users"]):
        raise RuntimeError("FP3 external bundle user count drifted")
    primary: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    mechanisms: dict[str, Any] = {}
    artifact_sources: dict[str, Any] = {}
    for arm_id in SETREC_ARMS:
        indexed, summary = _verified_setrec_predictions(
            root, arm_id, config, examples
        )
        primary[arm_id] = indexed
        summaries[arm_id] = summary
        mechanisms[arm_id] = summarize_mechanisms(
            indexed, attention_contract=summary["result"]["attention_contract"]
        )
        arm = arm_paths(root, arm_id)
        artifact_sources[arm_id] = {
            "predictions_path": str(arm["predictions"].relative_to(root)),
            "predictions_sha256": sha256(arm["predictions"]),
            "summary_path": str(arm["summary"].relative_to(root)),
            "summary_sha256": sha256(arm["summary"]),
        }
    primary["G0_GRAM_B0_FRESH"] = _verified_g0_predictions(root, config, examples)
    artifact_sources["G0_GRAM_B0_FRESH"] = dict(config["g0_control"])
    overall = {
        arm_id: {
            "variant": config["inference"]["primary_variant_by_arm"][arm_id],
            "metrics": aggregate_metrics(rows),
            "mechanisms": mechanisms.get(arm_id),
        }
        for arm_id, rows in primary.items()
    }
    comparisons: dict[str, Any] = {}
    subgroups: dict[str, Any] = {}
    assignments, thresholds = subgroup_assignments(train_users, examples)
    for label, (treatment, control) in config["comparisons"].items():
        comparisons[label] = compare_predictions(
            primary[treatment],
            primary[control],
            treatment_label=treatment,
            control_label=control,
            replicates=int(config["statistics"]["paired_bootstrap_replicates"]),
            seed=int(config["statistics"]["paired_bootstrap_seed"]),
        )
        subgroups[label] = subgroup_comparison(
            primary[treatment], primary[control], assignments
        )
    exact_alignment = all(
        set(rows) == {example.user_id for example in examples}
        for rows in primary.values()
    )
    rankings_valid = {
        arm_id: all(
            len(row["ranking"]) == 50 and len(set(row["ranking"])) == 50
            for row in rows.values()
        )
        for arm_id, rows in primary.items()
    }
    seal = _read(root / config["data"]["upstream_seal_path"])
    integrity = {
        "exact_user_alignment": exact_alignment,
        "all_primary_rankings_valid_top50": rankings_valid,
        "external_users": len(examples),
        "bundle_sha256": sha256(root / config["data"]["sealed_bundle_path"]),
        "bundle_sha256_matches": sha256(root / config["data"]["sealed_bundle_path"])
        == config["data"]["sealed_bundle_sha256"],
        "upstream_single_materialization_count": seal["single_materialization_count"],
        "fp3_family_bundle_read_count": 1,
        "guard_item_serialized": seal["guard_item_serialized"],
        "raw_external_projection_reopened": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    integrity_valid = bool(
        exact_alignment
        and all(rankings_valid.values())
        and integrity["bundle_sha256_matches"]
        and integrity["upstream_single_materialization_count"] == 1
        and integrity["guard_item_serialized"] is False
    )
    gate = fp3_gate(
        comparisons,
        subgroups["S2_MINUS_S0"],
        mechanisms,
        integrity_valid=integrity_valid,
    )
    next_action = (
        "freeze S2 standalone winner; keep FP4 closed; prepare FP5 D1 authorization without opening D1"
        if gate["verdict"] == "FP3_STRONG_PASS"
        else "stop FP3 SETRec promotion; keep D1/D2 and FP4 closed; do not tune on D0"
    )
    latency = {
        "ordered_control_seconds_per_user": mechanisms[
            "S0_SETREC_ORDERED_CONTROL"
        ]["mean_latency_seconds_per_user"],
        "repo_parity_seconds_per_user": mechanisms[
            "S1R_SETREC_REPO_PARITY"
        ]["mean_latency_seconds_per_user"],
        "paper_faithful_seconds_per_user": mechanisms[
            "S1P_SETREC_PAPER_FAITHFUL"
        ]["mean_latency_seconds_per_user"],
        "gram_paper_full_seconds_per_user": mechanisms[
            "S2_GRAM_SETREC_PAPER_FULL"
        ]["mean_latency_seconds_per_user"],
    }
    analysis = {
        "schema_version": "phase17.s17_fp3_external_analysis.v1",
        "completed_at": utc_now(),
        "verification_status": "ANALYZED",
        "external_users": len(examples),
        "overall": overall,
        "comparisons": comparisons,
        "subgroup_thresholds": thresholds,
        "subgroups": subgroups,
        "mechanisms": mechanisms,
        "latency": latency,
        "integrity": integrity,
        "gate": gate,
        "next_action": next_action,
        "arm_artifact_sources": artifact_sources,
        "fallacy_scan": _fallacy_scan(integrity_valid),
        "raw_external_projection_reopened": False,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["analysis"], analysis)
    return analysis


def _format_effect(comparison: Mapping[str, Any], metric: str) -> str:
    value = comparison["effects"][metric]
    return (
        f"{float(value['mean_delta']):+.6f} "
        f"[{float(value['ci95_low']):+.6f}, {float(value['ci95_high']):+.6f}]"
    )


def render_report(analysis: Mapping[str, Any]) -> str:
    names = {
        "S0_SETREC_ORDERED_CONTROL": "S0 Ordered Control",
        "S1R_SETREC_REPO_PARITY": "S1R Repo-Parity",
        "S1P_SETREC_PAPER_FAITHFUL": "S1P Paper-Faithful",
        "S2_GRAM_SETREC_PAPER_FULL": "S2 GRAM-SETRec-Paper-Full",
        "G0_GRAM_B0_FRESH": "G0 GRAM-B0-Fresh",
    }
    overall_rows = []
    for arm_id in (*SETREC_ARMS, "G0_GRAM_B0_FRESH"):
        metrics = analysis["overall"][arm_id]["metrics"]
        overall_rows.append(
            f"| {names[arm_id]} | {metrics['hit@10']:.6f} | "
            f"{metrics['ndcg@10']:.6f} | {metrics['mrr@10']:.6f} | "
            f"{metrics['hit@50']:.6f} | {metrics['ndcg@50']:.6f} |"
        )
    comparison_rows = []
    for label in (
        "S1R_MINUS_S0",
        "S1P_MINUS_S1R",
        "S1P_MINUS_S0",
        "S2_MINUS_S0",
        "S2_MINUS_G0",
    ):
        comparison = analysis["comparisons"][label]
        outcomes = comparison["primary_user_outcomes"]
        comparison_rows.append(
            f"| {label} | {_format_effect(comparison, 'ndcg@10')} | "
            f"{_format_effect(comparison, 'hit@10')} | "
            f"{outcomes['gain']}/{outcomes['loss']}/{outcomes['tie']} |"
        )
    mechanism_rows = []
    for arm_id in SETREC_ARMS:
        mechanism = analysis["mechanisms"][arm_id]
        mechanism_rows.append(
            f"| {names[arm_id]} | {mechanism['full_set_recovery_rate']:.6f} | "
            f"{mechanism['valid_item_rate']:.6f} | "
            f"{str(mechanism['query_norms_finite_nonzero'])} | "
            f"{mechanism['attention_contract']['forbidden_visibility_count']} | "
            f"{mechanism['mean_latency_seconds_per_user']:.6f} |"
        )
    gate_rows = [
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in analysis["gate"]["checks"].items()
    ]
    fallacy_rows = [
        f"| {name} | {finding} |"
        for name, finding in analysis["fallacy_scan"]["items"].items()
    ]
    return "\n".join(
        [
            "# Stage17 FP3 Full SETRec 正式结果报告",
            "",
            "## Material Passport",
            "",
            "- Origin Skill：`academic-research-suite / experiment-agent`",
            "- Origin Mode：`run → validate`",
            "- Step：`S17-FP3`",
            f"- 状态：`COMPLETED / {analysis['gate']['verdict']}`",
            f"- Verification Status：`{analysis['verification_status']}`",
            f"- 外部评估用户：{analysis['external_users']}",
            "- 数据边界：复用哈希冻结的 D0 bundle；未重开 raw projection；D1/D2、official test、Sports 未读",
            "",
            "## 1. 正式判定",
            "",
            f"FP3 Gate：`{analysis['gate']['verdict']}`。",
            f"下一步：{analysis['next_action']}。",
            "",
            "## 2. 主结果",
            "",
            "| Arm | Hit@10 | NDCG@10 | MRR@10 | Hit@50 | NDCG@50 |",
            "|---|---:|---:|---:|---:|---:|",
            *overall_rows,
            "",
            "## 3. Paired effects",
            "",
            "| Comparison | ΔNDCG@10 [95% CI] | ΔHit@10 [95% CI] | Gain/Loss/Tie |",
            "|---|---:|---:|---:|",
            *comparison_rows,
            "",
            "## 4. 机制与效率",
            "",
            "| Arm | Full-set recovery | Valid item | Query norms | Forbidden visibility | sec/user |",
            "|---|---:|---:|---|---:|---:|",
            *mechanism_rows,
            "",
            "每个 query 的 target grounding rank/recovery、semantic reconstruction、完整 latency 与分组结果见 canonical `analysis.json`。",
            "",
            "## 5. Gate 审计",
            "",
            "| Check | Status |",
            "|---|---|",
            *gate_rows,
            "",
            "## 6. 完整性",
            "",
            f"- 用户严格对齐：`{analysis['integrity']['exact_user_alignment']}`。",
            f"- 四臂及 G0 top-50 合法：`{all(analysis['integrity']['all_primary_rankings_valid_top50'].values())}`。",
            f"- Bundle SHA 匹配：`{analysis['integrity']['bundle_sha256_matches']}`。",
            "- Raw external projection reopened：`false`。",
            "- D1/D2、official test、Sports read：`false`。",
            "",
            "## 7. 统计谬误扫描",
            "",
            f"Coverage：`{analysis['fallacy_scan']['coverage']}`。",
            "",
            "| Type | Finding |",
            "|---|---|",
            *fallacy_rows,
            "",
        ]
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def worker(root: Path, manifest_path: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    family_writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    current_arm: str | None = None
    started = time.monotonic()
    try:
        _verify_snapshot_and_live(root, manifest_path)
        authorization = verify_authorization(root)
        config = _read(resolved["config"])
        verify_checkpoint_freeze(root, config)
        verify_sealed_bundle_without_content_read(root, config)
        verify_g0_control_without_prediction_read(root, config)
        gpu_id = int(authorization["physical_gpu"])
        live = _gpu_state()
        if int(_gpu(live, gpu_id)["free_mib"]) < int(
            authorization["minimum_free_mib"]
        ):
            raise RuntimeError(f"GPU{gpu_id} lost FP3 external admission headroom")
        import torch

        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        family_writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP3_EXTERNAL_D0_RUNNING",
            process_alive=True,
            workload_pid=os.getpid(),
            gpu_ids=[gpu_id],
            gpu_snapshot=live,
            stage="authorized_sealed_bundle_read",
            external_bundle_content_read=True,
            raw_external_projection_reopened=False,
        )
        bundle = root / config["data"]["sealed_bundle_path"]
        if sha256(bundle) != config["data"]["sealed_bundle_sha256"]:
            raise RuntimeError("sealed D0 bundle hash drifted at authorized read")
        train_users, examples = read_sealed_bundle_views(bundle)
        if len(examples) != int(config["data"]["expected_users"]):
            raise RuntimeError("authorized FP3 bundle user count drifted")
        for arm_id in SETREC_ARMS:
            current_arm = arm_id
            arm = arm_paths(root, arm_id)
            arm_writer = StatusWriter(
                resolved["status_dir"], arm_experiment_id(arm_id)
            )
            arm_writer.transition(
                "RUNNING",
                "RUNNING_SCIENTIFIC",
                "S17_FP3_EXTERNAL_ARM_RUNNING",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[gpu_id],
                stage="loading_frozen_checkpoint",
                external_bundle_content_read=True,
                raw_external_projection_reopened=False,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def heartbeat(stage: str, done: int, total: int) -> None:
                arm_writer.heartbeat(
                    stage=stage,
                    progress={"current": done, "total": total, "unit": "external_user"},
                )
                family_writer.heartbeat(
                    stage=f"{arm_slug(arm_id)}:{stage}",
                    progress={
                        "arm": arm_id,
                        "current": done,
                        "total": total,
                        "unit": "external_user",
                    },
                )

            record = config["checkpoints"][arm_id]
            result = evaluate_external_arm(
                root,
                arm_id,
                root / record["path"],
                examples,
                arm["predictions"],
                selected_beta=float(record["selected_beta"]),
                batch_size=int(config["inference"]["batch_size_by_arm"][arm_id]),
                device=torch.device("cuda"),
                heartbeat=heartbeat,
            )
            torch.cuda.synchronize()
            summary = {
                "schema_version": "phase17.s17_fp3_external_arm_summary.v1",
                "verdict": "PASS_S17_FP3_EXTERNAL_ARM_PREDICTIONS_FROZEN",
                "completed_at": utc_now(),
                "arm_id": arm_id,
                "physical_gpu": gpu_id,
                "checkpoint_path": record["path"],
                "checkpoint_sha256": record["sha256"],
                "bundle_path": config["data"]["sealed_bundle_path"],
                "bundle_sha256": config["data"]["sealed_bundle_sha256"],
                "predictions_path": str(arm["predictions"].relative_to(root)),
                "predictions_sha256": sha256(arm["predictions"]),
                "result": result,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1048576,
                "raw_external_projection_reopened": False,
                "fp3_family_bundle_read_count": 1,
                "automatic_retry": False,
                "automatic_process_termination": False,
                "d1_read": False,
                "d2_read": False,
                "test_read": False,
                "sports_read": False,
            }
            atomic_json(arm["summary"], summary)
            arm_writer.transition(
                "COMPLETED",
                "SCIENTIFIC_COMPLETED",
                "PASS_S17_FP3_EXTERNAL_ARM_PREDICTIONS_FROZEN",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                stage="external_predictions_frozen",
                progress={
                    "current": len(examples),
                    "total": len(examples),
                    "unit": "external_user",
                },
                predictions_path=summary["predictions_path"],
                predictions_sha256=summary["predictions_sha256"],
                summary_path=str(arm["summary"].relative_to(root)),
                summary_sha256=sha256(arm["summary"]),
                external_bundle_content_read=True,
                raw_external_projection_reopened=False,
                result_selection_eligible=True,
            )
        current_arm = None
        family_writer.heartbeat(stage="paired_analysis_and_gate")
        analysis = analyze_from_views(root, train_users, examples)
        _atomic_text(resolved["report"], render_report(analysis))
        family_writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP3_EXTERNAL_D0_ANALYZED",
            process_alive=False,
            workload_pid=0,
            gpu_ids=[],
            stage="external_analysis_and_report_complete",
            progress={"current": 4, "total": 4, "unit": "arm"},
            analysis_path=str(resolved["analysis"].relative_to(root)),
            analysis_sha256=sha256(resolved["analysis"]),
            report_path=str(resolved["report"].relative_to(root)),
            report_sha256=sha256(resolved["report"]),
            gate=analysis["gate"],
            next_action=analysis["next_action"],
            wall_seconds=time.monotonic() - started,
            external_bundle_content_read=True,
            fp3_family_bundle_read_count=1,
            raw_external_projection_reopened=False,
            result_selection_eligible=True,
        )
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "current_arm": current_arm,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "automatic_process_termination": False,
            "raw_external_projection_reopened": False,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
        }
        atomic_json(resolved["failure"], failure)
        if current_arm is not None:
            arm = arm_paths(root, current_arm)
            atomic_json(arm["failure"], failure)
            arm_writer = StatusWriter(
                resolved["status_dir"], arm_experiment_id(current_arm)
            )
            if arm_writer.read()["scientific_state"] == "RUNNING":
                arm_writer.transition(
                    "FAILED",
                    "SCIENTIFIC_FAILED",
                    "S17_FP3_EXTERNAL_ARM_FAILED_NO_RETRY",
                    process_alive=False,
                    workload_pid=0,
                    gpu_ids=[],
                    stage="terminal_failure_no_retry",
                    failure_path=str(arm["failure"].relative_to(root)),
                    failure_sha256=sha256(arm["failure"]),
                    terminal_error=repr(error),
                    automatic_retry=False,
                )
        if family_writer.read()["scientific_state"] == "RUNNING":
            family_writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP3_EXTERNAL_D0_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                stage="terminal_failure_no_retry",
                failure_path=str(resolved["failure"].relative_to(root)),
                failure_sha256=sha256(resolved["failure"]),
                terminal_error=repr(error),
                automatic_retry=False,
            )
        return 1


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    output: dict[str, Any] = {
        "source_config": str(resolved["source_config"]),
        "prepared": resolved["result"].is_dir(),
        "authorized": resolved["authorization"].is_file(),
        "analyzed": resolved["analysis"].is_file(),
        "report": resolved["report"].is_file(),
        "arms": {},
    }
    family_status = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if family_status.is_file():
        output["status"] = _read(family_status)
    for arm_id in SETREC_ARMS:
        arm = arm_paths(root, arm_id)
        status = resolved["status_dir"] / f"{arm_experiment_id(arm_id)}.status.json"
        output["arms"][arm_id] = {
            "predictions": arm["predictions"].is_file(),
            "summary": arm["summary"].is_file(),
            "status": _read(status) if status.is_file() else None,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "authorize", "launch", "worker", "inspect")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--researcher-direction", default="")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        return prepare(args.root)
    if args.action == "authorize":
        if args.gpu is None:
            raise SystemExit("authorize requires --gpu")
        return authorize(args.root, args.gpu, args.researcher_direction)
    if args.action == "launch":
        return launch(args.root)
    if args.action == "worker":
        if args.manifest is None:
            raise SystemExit("worker requires --manifest")
        return worker(args.root, args.manifest)
    print(json.dumps(inspect(args.root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
