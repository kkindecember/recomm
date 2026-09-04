#!/usr/bin/env python3
"""One-shot, authorization-gated external D0 evaluation for Stage17 FP1/FP2."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.core.full_latte_native_adapter import (
    APPROVED_METADATA_SUFFIX,
    read_item_metadata_catalog,
)
from experiment.phase17.core.full_latte_external_evaluator import (
    aggregate_metrics,
    compare_predictions,
    fp1_gate,
    fp2_gate,
    psid_collision_diagnostics,
    prediction_variants,
    read_jsonl,
    subgroup_assignments,
    subgroup_comparison,
    summarize_mechanisms,
    validate_prediction_rows,
)
from experiment.phase17.core.fullport_data import (
    APPROVED_D0_SUFFIX,
    FullportExternalExample,
    FullportTrainUser,
    materialize_external_evaluation_view,
)
from experiment.phase17.core.resource_profiler import query_gpus, snapshot
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
from experiment.phase17.protocol import s17_fp12_formal_runtime as formal


ROOT = Path(__file__).resolve().parents[3]
CONFIG_SUFFIX = Path("experiment/phase17/config/s17_fp12_external_d0.json")
RESULT_SUFFIX = Path("artifacts/phase17/fullport/external_d0/attempt_001")
STATUS_DIR_SUFFIX = Path("artifacts/phase17/status")
AUTHORIZATION_SUFFIX = Path(
    "artifacts/phase17/authorizations/s17_fp12_external_d0_attempt_001.json"
)
LEDGER_SUFFIX = Path("artifacts/phase17/attempts/S17-FP12-EXTERNAL-D0.attempts.jsonl")
EXPERIMENT_ID = "s17_fp12_external_d0"
ATTEMPT_ID = "attempt_001"
STEP_ID = "S17-FP12-EXTERNAL-D0"
MINIMUM_FREE_MIB = {
    "N0_NATIVE_PSID": 6076,
    "N1_NATIVE_LATTE": 7020,
    "G0_GRAM_B0_FRESH": 18968,
    "G1_GRAM_PSID_FULL": 18968,
    "G2_GRAM_LATTE_FULL": 18968,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def arm_slug(arm_id: str) -> str:
    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown Stage17 FP arm: {arm_id}")
    return arm_id.lower()


def arm_experiment_id(arm_id: str) -> str:
    return f"{EXPERIMENT_ID}_{arm_slug(arm_id)}"


def paths(root: Path) -> dict[str, Path]:
    result = root.resolve() / RESULT_SUFFIX
    return {
        "source_config": root / CONFIG_SUFFIX,
        "result": result,
        "config": result / "config.json",
        "readiness": result / "readiness.json",
        "bundle": result / "external_examples.jsonl",
        "materialization_attempt": result / "materialization_attempt.json",
        "seal": result / "materialization_seal.json",
        "analysis": result / "analysis.json",
        "failure": result / "failure.json",
        "authorization": root / AUTHORIZATION_SUFFIX,
        "status_dir": root / STATUS_DIR_SUFFIX,
        "ledger": root / LEDGER_SUFFIX,
        "snapshot": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
    }


def arm_paths(root: Path, arm_id: str) -> dict[str, Path]:
    base = paths(root)
    result = base["result"] / "arms" / arm_slug(arm_id)
    snapshot_worker = base["snapshot"].parent / "src/000_s17_fp12_external_d0_runtime.py"
    return {
        "result": result,
        "predictions": result / "predictions.jsonl",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "snapshot_worker": snapshot_worker,
    }


def source_paths(root: Path) -> list[Path]:
    official = root / (
        "artifacts/phase17/fullport/sources/"
        "latte_05e4e6d983225bcb7172f148a076890e80c524d1_attempt_003"
    )
    return [
        Path(__file__).resolve(),
        root / "experiment/phase17/core/full_latte_external_inference.py",
        root / "experiment/phase17/core/full_latte_external_evaluator.py",
        root / "experiment/phase17/core/fullport_data.py",
        root / "experiment/phase17/core/full_latte_native_backend.py",
        root / "experiment/phase17/core/full_latte_native_adapter.py",
        root / "experiment/phase17/core/full_latte_gram_backend.py",
        root / "experiment/phase17/core/full_latte_arm_contracts.py",
        root / "experiment/phase17/core/full_latte_contracts.py",
        root / "experiment/phase17/core/status_writer.py",
        root / "experiment/phase17/core/run_manager.py",
        official / "genrec/models/PSID/model.py",
        official / "genrec/models/PSID/tokenizer.py",
        official / "genrec/models/Latte/model.py",
        official / "genrec/models/Latte/tokenizer.py",
    ]


def _verify_snapshot_and_live(root: Path, manifest_path: Path) -> None:
    verify_run_snapshot(root, manifest_path)
    manifest = _read(manifest_path)
    for record in manifest["files"]:
        live = root / record["source_path"]
        if sha256(live) != record["sha256"]:
            raise RuntimeError(f"live source drifted after preflight: {record['source_path']}")


def verify_checkpoint_freeze(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if set(config["checkpoints"]) != set(ARM_IDS):
        raise RuntimeError("external evaluator checkpoint inventory is not the five frozen arms")
    for arm_id in ARM_IDS:
        record = config["checkpoints"][arm_id]
        status_path = root / record["status_path"]
        checkpoint = root / record["path"]
        status = _read(status_path)
        if (
            status.get("scientific_state") != "COMPLETED"
            or status.get("status_code") != "PASS_S17_FP12_FORMAL_CHECKPOINT_FROZEN"
            or status.get("checkpoint_frozen") is not True
            or status.get("external_target_materialized") is not False
            or status.get("checkpoint_path") != record["path"]
            or status.get("checkpoint_sha256") != record["sha256"]
        ):
            raise RuntimeError(f"formal checkpoint status is not frozen/sealed for {arm_id}")
        observed = sha256(checkpoint)
        if observed != record["sha256"]:
            raise RuntimeError(f"checkpoint hash drift for {arm_id}")
        evidence[arm_id] = {
            "status_path": record["status_path"],
            "status_sha256": sha256(status_path),
            "checkpoint_path": record["path"],
            "checkpoint_sha256": observed,
        }
    return evidence


def verify_readiness(root: Path) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    config = _read(resolved["source_config"])
    if (
        config.get("state") != "PREPARED_NOT_AUTHORIZED"
        or config["authorization"].get("external_d0_evaluation_authorized") is not False
        or config["authorization"].get("gpu_execution_authorized") is not False
    ):
        raise PermissionError("source config must remain prepared and unauthorized")
    manifest_path = root / "artifacts/phase17/s0_audit/shadow_data_manifest.json"
    manifest = _read(manifest_path)
    d0 = manifest["domains"]["Toys"]["folds"]["D0"]
    if (
        d0["output_path"] != config["data"]["projection_path"]
        or d0["output_sha256"] != config["data"]["projection_sha256"]
        or int(d0["output_users"]) != int(config["data"]["expected_users"])
    ):
        raise RuntimeError("safe D0 manifest metadata drifted")
    projection = root / config["data"]["projection_path"]
    if projection.resolve() != (root / APPROVED_D0_SUFFIX).resolve() or not projection.is_file():
        raise PermissionError("approved D0 projection path is missing or drifted")
    evidence = verify_checkpoint_freeze(root, config)
    psid_evidence = {}
    for key in ("summary", "resolved_codes", "raw_codes", "centroids"):
        artifact = root / config["psid_diagnostics"][f"{key}_path"]
        expected = config["psid_diagnostics"][f"{key}_sha256"]
        if sha256(artifact) != expected:
            raise RuntimeError(f"frozen PSID diagnostic artifact drifted: {key}")
        psid_evidence[key] = {
            "path": str(artifact.relative_to(root)),
            "sha256": expected,
        }
    return {
        "schema_version": "phase17.s17_fp12_external_readiness.v1",
        "verdict": "READY_AWAITING_EXPLICIT_D0_GPU_AUTHORIZATION",
        "verified_at": utc_now(),
        "source_config_sha256": sha256(resolved["source_config"]),
        "shadow_manifest_path": str(manifest_path.relative_to(root)),
        "shadow_manifest_sha256": sha256(manifest_path),
        "checkpoint_evidence": evidence,
        "psid_diagnostic_evidence": psid_evidence,
        "external_projection_content_read": False,
        "external_target_materialized": False,
        "gpu_used": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


def prepare(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if resolved["result"].exists() or status_path.exists() or resolved["snapshot"].exists():
        raise FileExistsError("external D0 attempt_001 has already been prepared")
    readiness = verify_readiness(root)
    resolved["result"].mkdir(parents=True, exist_ok=False)
    source_config = _read(resolved["source_config"])
    frozen = dict(source_config)
    frozen["source_config_sha256"] = readiness["source_config_sha256"]
    frozen["prepared_at"] = utc_now()
    atomic_json(resolved["config"], frozen)
    atomic_json(resolved["readiness"], readiness)
    manifest_path = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=["future_authorized_external_d0_worker"],
        source_paths=source_paths(root),
        config=frozen,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "experiment_id": EXPERIMENT_ID,
            "step_id": STEP_ID,
            "kind": "one_shot_external_evaluation",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_AUTHORIZATION_REQUIRED",
            "scientific_result_eligible": True,
            "snapshot_manifest": str(manifest_path.relative_to(root)),
            "external_target_materialized": False,
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
            "stage": "evaluator_preflight_complete_waiting_explicit_authorization",
            "run_snapshot_manifest": str(manifest_path.relative_to(root)),
            "readiness_path": str(resolved["readiness"].relative_to(root)),
            "readiness_sha256": sha256(resolved["readiness"]),
            "external_d0_evaluation_authorized": False,
            "gpu_execution_authorized": False,
            "external_target_materialized": False,
            "single_materialization_count": 0,
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
        "S17_FP12_EXTERNAL_D0_READY_AUTHORIZATION_REQUIRED",
        process_alive=False,
    )
    print(json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_gpu_map(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        arm_id, separator, raw_gpu = value.partition("=")
        if not separator or arm_id not in ARM_IDS or arm_id in mapping:
            raise ValueError(f"invalid --gpu assignment: {value}")
        gpu = int(raw_gpu)
        if gpu < 0:
            raise ValueError("GPU ids must be non-negative")
        mapping[arm_id] = gpu
    if set(mapping) != set(ARM_IDS):
        raise ValueError("authorization requires an explicit GPU assignment for all five arms")
    return mapping


def authorize(root: Path, gpu_map: Mapping[str, int], researcher_direction: str) -> int:
    """Record a future explicit user authorization; never called by prepare."""

    root = root.resolve()
    resolved = paths(root)
    if not researcher_direction.strip():
        raise ValueError("explicit researcher direction is required")
    if resolved["authorization"].exists():
        raise FileExistsError("external D0 authorization already exists")
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("external evaluator is not in authorizable PREFLIGHT state")
    _verify_snapshot_and_live(root, resolved["snapshot"])
    verify_checkpoint_freeze(root, _read(resolved["config"]))
    payload = {
        "schema_version": "phase17.s17_fp12_external_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "researcher_direction": researcher_direction.strip(),
        "external_d0_evaluation_authorized": True,
        "gpu_execution_authorized": True,
        "single_materialization_authorized": True,
        "physical_gpu_by_arm": {arm: int(gpu_map[arm]) for arm in ARM_IDS},
        "preserve_all_preexisting_compute_processes": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["authorization"], payload)
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_EXTERNAL_D0_AUTHORIZED_WAITING_MATERIALIZATION",
        external_d0_evaluation_authorized=True,
        gpu_execution_authorized=True,
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        physical_gpu_by_arm=payload["physical_gpu_by_arm"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_authorization(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    payload = _read(resolved["authorization"])
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "external_d0_evaluation_authorized": True,
        "gpu_execution_authorized": True,
        "single_materialization_authorized": True,
        "preserve_all_preexisting_compute_processes": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise PermissionError(f"invalid external authorization field: {key}")
    mapping = payload.get("physical_gpu_by_arm", {})
    if set(mapping) != set(ARM_IDS) or any(int(value) < 0 for value in mapping.values()):
        raise PermissionError("external authorization lacks the exact five-arm GPU map")
    if not str(payload.get("researcher_direction", "")).strip():
        raise PermissionError("external authorization lacks explicit researcher direction")
    return payload


def _atomic_bundle(
    path: Path,
    train_users: list[FullportTrainUser],
    examples: list[FullportExternalExample],
) -> None:
    if len(train_users) != len(examples):
        raise ValueError("train/external views are not aligned")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for user, example in zip(train_users, examples):
            if user.user_id != example.user_id:
                raise ValueError("single-pass D0 views lost user order")
            handle.write(
                json.dumps(
                    {
                        "user_id": example.user_id,
                        "train_items": list(user.train_items),
                        "history": list(example.history),
                        "target": example.target,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize(root: Path) -> int:
    """Perform the sole authorized open of the external D0 projection."""

    root = root.resolve()
    resolved = paths(root)
    authorization = verify_authorization(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("external D0 materialization is not in PREFLIGHT")
    if any(
        path.exists()
        for path in (resolved["materialization_attempt"], resolved["bundle"], resolved["seal"])
    ):
        raise FileExistsError("single D0 materialization was already attempted; retry is forbidden")
    _verify_snapshot_and_live(root, resolved["snapshot"])
    config = _read(resolved["config"])
    verify_checkpoint_freeze(root, config)
    atomic_json(
        resolved["materialization_attempt"],
        {
            "schema_version": "phase17.s17_fp12_external_materialization_attempt.v1",
            "started_at": utc_now(),
            "attempt_id": ATTEMPT_ID,
            "authorization_sha256": sha256(resolved["authorization"]),
            "single_materialization_count": 1,
            "automatic_retry": False,
        },
    )
    projection = root / config["data"]["projection_path"]
    train_users, examples = materialize_external_evaluation_view(
        projection,
        root=root,
        external_target_authorized=True,
        max_history_items=int(config["data"]["max_history_items"]),
        expected_sha256=config["data"]["projection_sha256"],
    )
    if len(examples) != int(config["data"]["expected_users"]):
        raise RuntimeError("external D0 user count drifted")
    _atomic_bundle(resolved["bundle"], train_users, examples)
    seal = {
        "schema_version": "phase17.s17_fp12_external_materialization_seal.v1",
        "sealed_at": utc_now(),
        "attempt_id": ATTEMPT_ID,
        "source_path": config["data"]["projection_path"],
        "source_sha256": config["data"]["projection_sha256"],
        "bundle_path": str(resolved["bundle"].relative_to(root)),
        "bundle_sha256": sha256(resolved["bundle"]),
        "external_users": len(examples),
        "single_materialization_count": 1,
        "authorization_sha256": sha256(resolved["authorization"]),
        "researcher_direction": authorization["researcher_direction"],
        "guard_item_serialized": False,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["seal"], seal)
    writer.transition(
        "RUNNING",
        "WAITING_FOR_GPU",
        "S17_FP12_EXTERNAL_D0_MATERIALIZED_ONCE_WAITING_ARM_LAUNCH",
        stage="external_target_materialized_once_workers_pending",
        external_target_materialized=True,
        single_materialization_count=1,
        materialization_seal_path=str(resolved["seal"].relative_to(root)),
        materialization_seal_sha256=sha256(resolved["seal"]),
        process_alive=False,
    )
    print(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_seal(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    seal = _read(resolved["seal"])
    config = _read(resolved["config"])
    expected = {
        "attempt_id": ATTEMPT_ID,
        "source_path": config["data"]["projection_path"],
        "source_sha256": config["data"]["projection_sha256"],
        "bundle_path": str(resolved["bundle"].relative_to(root)),
        "external_users": int(config["data"]["expected_users"]),
        "single_materialization_count": 1,
        "guard_item_serialized": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise PermissionError(f"invalid materialization seal field: {key}")
    if sha256(resolved["bundle"]) != seal.get("bundle_sha256"):
        raise RuntimeError("materialized D0 bundle hash drifted")
    return seal


def selected_python(root: Path, arm_id: str) -> Path:
    return formal.selected_python(root, formal.FORMAL_SPECS[arm_id])


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
        str(selected_python(root, arm_id)),
        str(arm["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--arm",
        arm_id,
        "--manifest",
        str(resolved["snapshot"]),
    ]


def gpu_admission(arm_id: str, gpu: int) -> dict[str, Any]:
    first_records = query_gpus()
    time.sleep(5)
    second_records = query_gpus()
    snapshots = {"first": snapshot(first_records), "second": snapshot(second_records)}
    for records in (first_records, second_records):
        selected = next((row for row in records if row.index == gpu), None)
        if selected is None or selected.free_mib < MINIMUM_FREE_MIB[arm_id]:
            raise RuntimeError(
                f"GPU{gpu} does not satisfy {arm_id} free-memory gate "
                f"{MINIMUM_FREE_MIB[arm_id]} MiB"
            )
    snapshots.update(
        physical_gpu=gpu,
        minimum_free_mib=MINIMUM_FREE_MIB[arm_id],
        utilization_recorded_only=True,
        preserve_preexisting_processes=True,
    )
    return snapshots


def launch_arm(root: Path, arm_id: str) -> int:
    root = root.resolve()
    resolved = paths(root)
    arm = arm_paths(root, arm_id)
    authorization = verify_authorization(root)
    verify_seal(root)
    _verify_snapshot_and_live(root, resolved["snapshot"])
    if arm["result"].exists():
        raise FileExistsError(f"external arm result already exists: {arm_id}")
    gpu = int(authorization["physical_gpu_by_arm"][arm_id])
    if not selected_python(root, arm_id).is_file():
        raise FileNotFoundError(f"frozen {arm_id} Python environment is missing")
    snapshots = gpu_admission(arm_id, gpu)
    arm["result"].mkdir(parents=True, exist_ok=False)
    writer = StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id))
    writer.initialize(
        step_id=STEP_ID,
        attempt_id=ATTEMPT_ID,
        track_id=arm_id,
        canonical_result_dir=str(arm["result"].relative_to(root)),
        log_path=str(arm["log"].relative_to(root)),
        extra={
            "stage": "external_arm_preflight",
            "external_target_materialized": True,
            "single_materialization_count": 1,
            "target_gpu_id": gpu,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_EXTERNAL_ARM_READY",
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
        "S17_FP12_EXTERNAL_ARM_BACKGROUND_STARTED",
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
                "S17_FP12_EXTERNAL_ARM_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                gpu_ids=[],
            )
        raise RuntimeError("external arm worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, arm_id: str, manifest: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    arm = arm_paths(root, arm_id)
    writer = StatusWriter(resolved["status_dir"], arm_experiment_id(arm_id))
    try:
        authorization = verify_authorization(root)
        seal = verify_seal(root)
        _verify_snapshot_and_live(root, manifest)
        config = _read(resolved["config"])
        checkpoint_record = config["checkpoints"][arm_id]
        checkpoint = root / checkpoint_record["path"]
        if sha256(checkpoint) != checkpoint_record["sha256"]:
            raise RuntimeError(f"checkpoint hash drift for {arm_id}")
        gpu = int(authorization["physical_gpu_by_arm"][arm_id])
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP12_EXTERNAL_ARM_INFERENCE",
            workload_pid=os.getpid(),
            process_alive=True,
            gpu_ids=[gpu],
            stage="external_inference",
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
            resolved["bundle"],
            arm["predictions"],
            heartbeat=heartbeat,
        )
        summary = {
            "schema_version": "phase17.s17_fp12_external_arm_summary.v1",
            "verdict": "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN",
            "completed_at": utc_now(),
            "arm_id": arm_id,
            "physical_gpu": gpu,
            "checkpoint_path": checkpoint_record["path"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "bundle_sha256": seal["bundle_sha256"],
            "predictions_path": str(arm["predictions"].relative_to(root)),
            "predictions_sha256": sha256(arm["predictions"]),
            "result": result,
            "single_materialization_count": 1,
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
            "arm_id": arm_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "single_materialization_count": 1,
        }
        atomic_json(arm["failure"], failure)
        current = writer.read()
        if current["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_EXTERNAL_ARM_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                failure_path=str(arm["failure"].relative_to(root)),
                failure_sha256=sha256(arm["failure"]),
                terminal_error=repr(error),
                automatic_retry=False,
            )
        return 1


def _bundle_views(path: Path) -> tuple[list[FullportTrainUser], list[FullportExternalExample]]:
    train_users: list[FullportTrainUser] = []
    examples: list[FullportExternalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            train_users.append(
                FullportTrainUser(
                    user_id=str(row["user_id"]),
                    train_items=tuple(str(item) for item in row["train_items"]),
                )
            )
            examples.append(
                FullportExternalExample(
                    user_id=str(row["user_id"]),
                    history=tuple(str(item) for item in row["history"]),
                    target=str(row["target"]),
                )
            )
    return train_users, examples


def _verified_arm_predictions(
    root: Path,
    arm_id: str,
    examples: list[FullportExternalExample],
    *,
    artifact_paths: Mapping[str, Path] | None = None,
    status_experiment_id: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    resolved = paths(root)
    arm = dict(artifact_paths or arm_paths(root, arm_id))
    status = StatusWriter(
        resolved["status_dir"], status_experiment_id or arm_experiment_id(arm_id)
    ).read()
    summary = _read(arm["summary"])
    if (
        status["scientific_state"] != "COMPLETED"
        or status["status_code"] != "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN"
        or summary.get("verdict") != "PASS_S17_FP12_EXTERNAL_ARM_PREDICTIONS_FROZEN"
        or summary.get("predictions_sha256") != sha256(arm["predictions"])
    ):
        raise RuntimeError(f"external prediction artifact is not frozen for {arm_id}")
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, rows in prediction_variants(read_jsonl(arm["predictions"])).items():
        output[variant] = validate_prediction_rows(
            rows,
            examples,
            expected_arm_id=arm_id,
            expected_variant=variant,
        )
    return output


def analyze_selected(
    root: Path,
    *,
    artifact_sources: Mapping[str, tuple[Mapping[str, Path], str]] | None = None,
    analysis_path: Path | None = None,
    manifest_path: Path | None = None,
    recovery_provenance: Mapping[str, Any] | None = None,
) -> int:
    root = root.resolve()
    resolved = paths(root)
    selected_analysis = analysis_path or resolved["analysis"]
    if selected_analysis.exists():
        raise FileExistsError("external D0 analysis already exists")
    verify_authorization(root)
    seal = verify_seal(root)
    _verify_snapshot_and_live(root, manifest_path or resolved["snapshot"])
    config = _read(resolved["config"])
    train_users, examples = _bundle_views(resolved["bundle"])
    if len(examples) != int(config["data"]["expected_users"]):
        raise RuntimeError("analysis bundle user count drifted")
    sources = dict(artifact_sources or {})
    variants = {}
    selected_sources: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        artifact, status_id = sources.get(
            arm_id, (arm_paths(root, arm_id), arm_experiment_id(arm_id))
        )
        variants[arm_id] = _verified_arm_predictions(
            root,
            arm_id,
            examples,
            artifact_paths=artifact,
            status_experiment_id=status_id,
        )
        selected_sources[arm_id] = {
            "predictions_path": str(Path(artifact["predictions"]).relative_to(root)),
            "predictions_sha256": sha256(Path(artifact["predictions"])),
            "summary_path": str(Path(artifact["summary"]).relative_to(root)),
            "summary_sha256": sha256(Path(artifact["summary"])),
            "status_experiment_id": status_id,
        }
    primary = {
        arm_id: variants[arm_id][config["primary_variants"][arm_id]]
        for arm_id in ARM_IDS
    }
    overall = {
        arm_id: {
            variant: {
                "metrics": aggregate_metrics(rows),
                "mechanisms": summarize_mechanisms(rows),
            }
            for variant, rows in arm_variants.items()
        }
        for arm_id, arm_variants in variants.items()
    }
    comparisons: dict[str, Any] = {}
    subgroups: dict[str, Any] = {}
    assignments, thresholds = subgroup_assignments(train_users, examples)
    for label, (treatment_arm, control_arm) in config["comparisons"].items():
        comparisons[label] = compare_predictions(
            primary[treatment_arm],
            primary[control_arm],
            treatment_label=treatment_arm,
            control_label=control_arm,
            replicates=int(config["statistics"]["paired_bootstrap_replicates"]),
            seed=int(config["statistics"]["paired_bootstrap_seed"]),
        )
        subgroups[label] = subgroup_comparison(
            primary[treatment_arm], primary[control_arm], assignments
        )
    ablations = {
        "N1_AGG_SUM_MINUS_N0": compare_predictions(
            variants["N1_NATIVE_LATTE"]["beam500_agg_sum"],
            primary["N0_NATIVE_PSID"],
            treatment_label="N1_NATIVE_LATTE:beam500_agg_sum",
            control_label="N0_NATIVE_PSID:beam500_identity",
        ),
        "G2_AGG_SUM_MINUS_G1": compare_predictions(
            variants["G2_GRAM_LATTE_FULL"]["beam500_agg_sum"],
            primary["G1_GRAM_PSID_FULL"],
            treatment_label="G2_GRAM_LATTE_FULL:beam500_agg_sum",
            control_label="G1_GRAM_PSID_FULL:beam500_identity",
        ),
    }
    n1_mechanisms = overall["N1_NATIVE_LATTE"]["beam500_agg_max"]["mechanisms"]
    g2_mechanisms = overall["G2_GRAM_LATTE_FULL"]["beam500_agg_max"]["mechanisms"]
    g1_mechanisms = overall["G1_GRAM_PSID_FULL"]["beam500_identity"]["mechanisms"]
    psid_paths = config["psid_diagnostics"]
    for key in ("summary", "resolved_codes", "raw_codes", "centroids"):
        artifact = root / psid_paths[f"{key}_path"]
        if sha256(artifact) != psid_paths[f"{key}_sha256"]:
            raise RuntimeError(f"frozen PSID diagnostic artifact drifted: {key}")
    import numpy as np

    resolved_codes = _read(root / psid_paths["resolved_codes_path"])
    psid_diagnostics = psid_collision_diagnostics(
        resolved_codes,
        np.load(root / psid_paths["raw_codes_path"]),
        np.load(root / psid_paths["centroids_path"]),
        catalog_items=read_item_metadata_catalog(
            root / APPROVED_METADATA_SUFFIX, root=root
        )[0],
    )
    tokenizer_summary = _read(root / psid_paths["summary_path"])
    collision_contract = tokenizer_summary["collision_resolution"]
    if (
        psid_diagnostics["reassigned_items"] != collision_contract["reassigned_items"]
        or psid_diagnostics["collision_aliases_after"] != collision_contract["collisions_after"]
    ):
        raise RuntimeError("computed PSID collision diagnostics drifted from FP0")
    all_primary_nonempty = {
        arm_id: all(bool(row["ranking"]) for row in rows.values())
        for arm_id, rows in primary.items()
    }
    g2_legal = all(
        float(row["mechanism"]["valid_path_rate"]) == 1.0
        for row in primary["G2_GRAM_LATTE_FULL"].values()
    )
    integrity = {
        "exact_user_alignment": all(len(rows) == len(examples) for rows in primary.values()),
        "all_primary_rankings_nonempty": all_primary_nonempty,
        "all_g2_constrained_paths_legal": g2_legal,
        "single_materialization_count": seal["single_materialization_count"],
        "guard_item_serialized": seal["guard_item_serialized"],
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    shared_integrity = (
        integrity["exact_user_alignment"]
        and all(all_primary_nonempty.values())
        and seal["single_materialization_count"] == 1
        and seal["guard_item_serialized"] is False
        and psid_diagnostics["collision_aliases_after"] == 0
    )
    gates = {
        "FP1": fp1_gate(
            comparisons["FP1_N1_MINUS_N0"],
            n1_mechanisms,
            aggregate_item_valid=all_primary_nonempty["N1_NATIVE_LATTE"],
            integrity_valid=shared_integrity,
        ),
        "FP2": fp2_gate(
            comparisons["FP2_G2_MINUS_G1"],
            comparisons["FP2_G2_MINUS_G0"],
            subgroups["FP2_G2_MINUS_G0"],
            g2_mechanisms,
            g1_mechanisms,
            aggregate_item_valid=all_primary_nonempty["G2_GRAM_LATTE_FULL"],
            integrity_valid=shared_integrity and g2_legal,
        ),
    }
    if gates["FP1"]["verdict"] == "FP1_STRONG_PASS" and gates["FP2"]["verdict"] == "FP2_STRONG_PASS":
        next_action = "freeze standalone LATTE candidates; proceed to preregistered FP3, keep D1 locked"
    elif gates["FP2"]["verdict"] == "FP2_STRONG_PASS":
        next_action = "freeze G2 standalone candidate; proceed to FP3, keep D1 locked"
    elif gates["FP1"]["verdict"] == "FP1_STRONG_PASS":
        next_action = "retain native LATTE evidence only; stop GRAM-LATTE branch and keep D1 locked"
    else:
        next_action = "stop standalone LATTE migration; do not open D1 and do not tune on D0"
    result = {
        "schema_version": "phase17.s17_fp12_external_analysis.v1",
        "completed_at": utc_now(),
        "external_users": len(examples),
        "primary_variants": config["primary_variants"],
        "overall": overall,
        "comparisons": comparisons,
        "ablations": ablations,
        "subgroup_thresholds": thresholds,
        "subgroups": subgroups,
        "psid_collision_diagnostics": psid_diagnostics,
        "integrity": integrity,
        "gates": gates,
        "next_action": next_action,
        "arm_artifact_sources": selected_sources,
        "controlled_recovery": recovery_provenance is not None,
        "recovery_provenance": dict(recovery_provenance or {}),
        "single_materialization_count": 1,
        "automatic_retry": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(selected_analysis, result)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.transition(
        "COMPLETED",
        "SCIENTIFIC_COMPLETED",
        "PASS_S17_FP12_EXTERNAL_D0_ANALYZED",
        process_alive=False,
        stage="external_analysis_complete",
        analysis_path=str(selected_analysis.relative_to(root)),
        analysis_sha256=sha256(selected_analysis),
        gates=gates,
        next_action=next_action,
        result_selection_eligible=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def analyze(root: Path) -> int:
    return analyze_selected(root)


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve()
    resolved = paths(root)
    output: dict[str, Any] = {
        "source_config": str(resolved["source_config"]),
        "prepared": resolved["result"].is_dir(),
        "authorized": resolved["authorization"].is_file(),
        "materialization_attempted": resolved["materialization_attempt"].is_file(),
        "materialized": resolved["seal"].is_file(),
        "analyzed": resolved["analysis"].is_file(),
        "arms": {},
    }
    family_status = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if family_status.is_file():
        output["status"] = _read(family_status)
    for arm_id in ARM_IDS:
        arm = arm_paths(root, arm_id)
        arm_status = resolved["status_dir"] / f"{arm_experiment_id(arm_id)}.status.json"
        output["arms"][arm_id] = {
            "predictions": arm["predictions"].is_file(),
            "summary": arm["summary"].is_file(),
            "status": _read(arm_status) if arm_status.is_file() else None,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "authorize", "materialize", "launch-arm", "worker", "analyze", "inspect"),
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=ARM_IDS)
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--researcher-direction", default="")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.root)
    if args.action == "authorize":
        return authorize(args.root, _parse_gpu_map(args.gpu), args.researcher_direction)
    if args.action == "materialize":
        return materialize(args.root)
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
