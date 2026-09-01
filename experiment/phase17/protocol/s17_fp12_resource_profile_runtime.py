#!/usr/bin/env python3
"""Immutable, authorization-gated resource profiles for FP1/FP2 arms."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_latte_arm_contracts import (
    ARM_IDS,
    load_and_validate_arm_matrix,
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


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "attempt_001"
GRAM_PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
NATIVE_PYTHON_SUFFIX = Path(
    "artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python"
)
MATRIX_SUFFIX = Path("experiment/phase17/config/s17_fp12_latte_arm_matrix.json")
ALLOCATION_SUFFIX = Path("experiment/phase17/config/s17_fp_resource_allocation.json")
TOKENIZER_STATUS_SUFFIX = Path(
    "artifacts/phase17/status/s17_fp0_full_data_tokenizer.status.json"
)
VOCAB_MANIFEST_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/amendment_001/manifest.json"
)
LEDGER_SUFFIX = Path(
    "artifacts/phase17/attempts/S17-FP12-RESOURCE-PROFILES.attempts.jsonl"
)
SAFETY_MARGIN_MIB = 3072


@dataclass(frozen=True)
class ProfileSpec:
    arm_id: str
    step_id: str
    physical_gpu: int
    peak_cap_mib: int
    minimum_free_mib: int
    train_batch_size: int
    eval_batch_size: int
    family: str


PROFILE_SPECS = {
    "G0_GRAM_B0_FRESH": ProfileSpec(
        "G0_GRAM_B0_FRESH", "S17-FP2-PROFILE", 1, 20480, 23552, 16, 1, "gram"
    ),
    "G1_GRAM_PSID_FULL": ProfileSpec(
        "G1_GRAM_PSID_FULL", "S17-FP2-PROFILE", 0, 20480, 23552, 16, 1, "gram"
    ),
    "G2_GRAM_LATTE_FULL": ProfileSpec(
        "G2_GRAM_LATTE_FULL", "S17-FP2-PROFILE", 7, 20480, 23552, 16, 1, "gram"
    ),
    "N0_NATIVE_PSID": ProfileSpec(
        "N0_NATIVE_PSID", "S17-FP1-PROFILE", 4, 16384, 19456, 256, 1, "native"
    ),
    "N1_NATIVE_LATTE": ProfileSpec(
        "N1_NATIVE_LATTE", "S17-FP1-PROFILE", 4, 16384, 19456, 256, 1, "native"
    ),
}


def arm_slug(arm_id: str) -> str:
    if arm_id not in PROFILE_SPECS:
        raise ValueError(f"unknown profile arm: {arm_id}")
    return arm_id.lower()


def experiment_id(arm_id: str) -> str:
    return f"s17_fp12_profile_{arm_slug(arm_id)}"


def paths(root: Path, arm_id: str) -> dict[str, Path]:
    slug = arm_slug(arm_id)
    exp_id = experiment_id(arm_id)
    result = root / f"artifacts/phase17/fullport/profiles/{slug}/{ATTEMPT_ID}"
    snapshot_manifest = (
        root / f"artifacts/phase17/snapshots/{exp_id}/{ATTEMPT_ID}/manifest.json"
    )
    return {
        "result": result,
        "config": result / "config.json",
        "cpu_preflight": result / "cpu_preflight.json",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "matrix": root / MATRIX_SUFFIX,
        "allocation": root / ALLOCATION_SUFFIX,
        "tokenizer_status": root / TOKENIZER_STATUS_SUFFIX,
        "vocab_manifest": root / VOCAB_MANIFEST_SUFFIX,
        "authorization": root
        / f"artifacts/phase17/authorizations/{exp_id}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / LEDGER_SUFFIX,
        "snapshot": snapshot_manifest,
        "snapshot_worker": snapshot_manifest.parent
        / "src/000_s17_fp12_resource_profile_runtime.py",
        "native_python": root / NATIVE_PYTHON_SUFFIX,
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dependencies(root: Path) -> dict[str, Any]:
    tokenizer_status_path = root / TOKENIZER_STATUS_SUFFIX
    tokenizer = _read(tokenizer_status_path)
    if (
        tokenizer.get("scientific_state") != "COMPLETED"
        or tokenizer.get("status_code") != "PASS_S17_FP0_FULL_DATA_TOKENIZER"
    ):
        raise RuntimeError("full-data tokenizer is not a completed PASS dependency")
    tokenizer_manifest = root / tokenizer["tokenizer_manifest_path"]
    if sha256(tokenizer_manifest) != tokenizer["tokenizer_manifest_sha256"]:
        raise RuntimeError("full-data tokenizer manifest hash drifted")
    vocab_manifest_path = root / VOCAB_MANIFEST_SUFFIX
    vocabulary = _read(vocab_manifest_path)
    vocabulary_path = root / vocabulary["complete_vocabulary_path"]
    if sha256(vocabulary_path) != vocabulary["complete_vocabulary_sha256"]:
        raise RuntimeError("complete vocabulary amendment hash drifted")
    matrix_path = root / MATRIX_SUFFIX
    load_and_validate_arm_matrix(matrix_path)
    return {
        "tokenizer_status_sha256": sha256(tokenizer_status_path),
        "tokenizer_manifest_path": tokenizer["tokenizer_manifest_path"],
        "tokenizer_manifest_sha256": tokenizer["tokenizer_manifest_sha256"],
        "vocabulary_manifest_sha256": sha256(vocab_manifest_path),
        "complete_vocabulary_sha256": vocabulary["complete_vocabulary_sha256"],
        "arm_matrix_sha256": sha256(matrix_path),
        "allocation_sha256": sha256(root / ALLOCATION_SUFFIX),
    }


def cpu_preflight(root: Path, arm_id: str) -> dict[str, Any]:
    if PROFILE_SPECS[arm_id].family == "gram":
        from experiment.phase17.core.full_latte_gram_backend import (
            cpu_preflight_gram_arm,
        )

        return cpu_preflight_gram_arm(root, arm_id)
    from experiment.phase17.core.full_latte_native_backend import (
        cpu_preflight_native_arm,
    )

    return cpu_preflight_native_arm(root, arm_id)


def selected_python(root: Path, spec: ProfileSpec) -> Path:
    return GRAM_PYTHON if spec.family == "gram" else root / NATIVE_PYTHON_SUFFIX


def worker_command(
    root: Path, arm_id: str, *, frozen_worker: bool = True
) -> list[str]:
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    worker = resolved["snapshot_worker"] if frozen_worker else Path(__file__).resolve()
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
        str(worker),
        "worker",
        "--arm",
        arm_id,
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def frozen_config(root: Path, arm_id: str, preflight: dict[str, Any]) -> dict[str, Any]:
    spec = PROFILE_SPECS[arm_id]
    matrix = load_and_validate_arm_matrix(root / MATRIX_SUFFIX)
    return {
        "schema_version": "phase17.s17_fp12_resource_profile_config.v1",
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "step_id": spec.step_id,
        "arm_id": arm_id,
        "family": spec.family,
        "prepared_at": utc_now(),
        "physical_gpu": spec.physical_gpu,
        "peak_reserved_cap_mib": spec.peak_cap_mib,
        "minimum_free_mib": spec.minimum_free_mib,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "train_batch_size": spec.train_batch_size,
        "eval_batch_size": spec.eval_batch_size,
        "primary_beam": 500,
        "top_k": 50,
        "aggregation": matrix["arms"][arm_id]["aggregation"],
        "cpu_preflight": preflight,
        "resource_only": True,
        "effect_metrics_forbidden": True,
        "external_target_materialized": False,
        "launch_requires_attempt_authorization": True,
        "launch_authorized": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "background_required": True,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "effect_experiment_started": False,
    }


def snapshot_sources(root: Path, spec: ProfileSpec) -> list[Path]:
    common = [
        Path(__file__).resolve(),
        root / "experiment/phase17/core/full_latte_profile_executor.py",
        root / "experiment/phase17/core/full_latte_arm_contracts.py",
        root / "experiment/phase17/core/full_latte_contracts.py",
        root / "experiment/phase17/core/fullport_data.py",
        root / "experiment/phase17/core/full_latte_native_adapter.py",
        root / "experiment/phase17/core/status_writer.py",
        root / "experiment/phase17/core/run_manager.py",
        root / "experiment/phase17/core/resource_profiler.py",
    ]
    if spec.family == "gram":
        common.extend(
            [
                root / "experiment/phase17/core/full_latte_gram_backend.py",
                root / "GRAM/src/model/gram.py",
                root / "GRAM/src/model/gram_t5.py",
                root / "GRAM/src/processor/Collator.py",
            ]
        )
    else:
        source = root / (
            "artifacts/phase17/fullport/sources/"
            "latte_05e4e6d983225bcb7172f148a076890e80c524d1_attempt_003"
        )
        common.extend(
            [
                root / "experiment/phase17/core/full_latte_native_backend.py",
                source / "genrec/models/PSID/model.py",
                source / "genrec/models/PSID/tokenizer.py",
                source / "genrec/models/Latte/model.py",
                source / "genrec/models/Latte/tokenizer.py",
            ]
        )
    return common


def prepare(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    dependencies = verify_dependencies(root)
    python_path = selected_python(root, spec)
    if not python_path.is_file():
        raise FileNotFoundError(f"profile Python is missing: {python_path}")
    status_path = resolved["status_dir"] / f"{experiment_id(arm_id)}.status.json"
    if status_path.exists():
        raise FileExistsError(f"resource profile {arm_id} already has a status record")
    partial = resolved["result"].exists() or resolved["snapshot"].exists()
    if partial:
        required = (
            resolved["result"],
            resolved["config"],
            resolved["cpu_preflight"],
            resolved["snapshot"],
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                f"incomplete, non-recoverable profile preparation for {arm_id}: {missing}"
            )
        # A ledger write can fail after the immutable snapshot is complete.  In
        # that narrow case, close out the prepared state without deleting or
        # regenerating any frozen artifact.
        verify_run_snapshot(root, resolved["snapshot"])
        preflight = _read(resolved["cpu_preflight"])
        config = _read(resolved["config"])
        if config.get("arm_id") != arm_id or preflight.get("state") != "PASS_CPU_PREFLIGHT":
            raise RuntimeError("partial profile preparation artifacts failed validation")
        manifest = resolved["snapshot"]
    else:
        preflight = cpu_preflight(root, arm_id)
        if preflight["state"] != "PASS_CPU_PREFLIGHT":
            raise RuntimeError(f"CPU preflight did not pass for {arm_id}")
        resolved["result"].mkdir(parents=True, exist_ok=False)
        atomic_json(resolved["cpu_preflight"], preflight)
        config = frozen_config(root, arm_id, preflight)
        config["dependencies"] = dependencies
        config["python_path"] = str(python_path)
        atomic_json(resolved["config"], config)
        manifest = freeze_run_snapshot(
            root=root,
            experiment_id=experiment_id(arm_id),
            attempt_id=ATTEMPT_ID,
            command=worker_command(root, arm_id),
            source_paths=snapshot_sources(root, spec),
            config=config,
        )
    ledger_attempt_id = f"{arm_slug(arm_id)}_{ATTEMPT_ID}"
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ledger_attempt_id,
            "profile_attempt_id": ATTEMPT_ID,
            "experiment_id": experiment_id(arm_id),
            "step_id": spec.step_id,
            "arm_id": arm_id,
            "kind": "arm_specific_resource_profile",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_AUTHORIZATION_REQUIRED",
            "scientific_result_eligible": False,
            "automatic_retry": False,
            "gpu_ids": [],
            "snapshot_manifest": str(manifest.relative_to(root)),
            "recovered_after_ledger_collision": partial,
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
            "stage": "cpu_preflight_complete_waiting_researcher_authorization",
            "progress": {"current": 1, "total": 4, "unit": "profile_gate"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "dependencies": dependencies,
            "cpu_preflight_path": str(resolved["cpu_preflight"].relative_to(root)),
            "cpu_preflight_sha256": sha256(resolved["cpu_preflight"]),
            "gpu_ids": [],
            "target_gpu_id": spec.physical_gpu,
            "minimum_free_mib": spec.minimum_free_mib,
            "peak_reserved_cap_mib": spec.peak_cap_mib,
            "launch_authorized": False,
            "resource_only": True,
            "effect_metrics_forbidden": True,
            "automatic_process_termination": False,
            "automatic_retry": False,
            "external_target_materialized": False,
            "effect_experiment_started": False,
            "affects_scientific_result": False,
            "result_selection_eligible": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_RESOURCE_PROFILE_READY_AUTHORIZATION_REQUIRED",
        process_alive=False,
    )
    print(json.dumps({"arm_id": arm_id, "manifest": str(manifest)}, indent=2))
    return 0


def query_compute_processes() -> dict[int, list[dict[str, Any]]]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    uuid_to_index = {
        row[1].strip(): int(row[0].strip())
        for row in csv.reader(io.StringIO(gpu_rows))
        if len(row) == 2
    }
    process_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    result = {index: [] for index in uuid_to_index.values()}
    for row in csv.reader(io.StringIO(process_rows)):
        if len(row) != 4 or row[0].strip() not in uuid_to_index:
            continue
        result[uuid_to_index[row[0].strip()]].append(
            {
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "used_memory_mib": int(row[3].strip()),
            }
        )
    return result


def gpu_snapshot_once(spec: ProfileSpec) -> dict[str, Any]:
    records = query_gpus()
    matches = [row for row in records if row.index == spec.physical_gpu]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU{spec.physical_gpu} is not uniquely visible")
    processes = query_compute_processes()
    return {
        "captured_at": utc_now(),
        "devices": snapshot(records),
        "selected": matches[0].__dict__,
        "selected_compute_processes": processes.get(spec.physical_gpu, []),
    }


def two_snapshot_admission(
    spec: ProfileSpec, authorization: dict[str, Any]
) -> dict[str, Any]:
    first = gpu_snapshot_once(spec)
    time.sleep(5)
    second = gpu_snapshot_once(spec)
    for row in (first, second):
        selected = row["selected"]
        if selected["free_mib"] < spec.minimum_free_mib:
            raise RuntimeError(
                f"GPU{spec.physical_gpu} free={selected['free_mib']} MiB below "
                f"profile gate {spec.minimum_free_mib} MiB"
            )
        if spec.physical_gpu != 4 and selected["utilization_percent"] > 20:
            raise RuntimeError(
                f"GPU{spec.physical_gpu} utilization exceeds the frozen 20% gate"
            )
    approved = {
        int(pid) for pid in authorization.get("approved_preexisting_compute_pids", [])
    }
    observed = {
        int(process["pid"])
        for row in (first, second)
        for process in row["selected_compute_processes"]
    }
    if not observed <= approved:
        raise PermissionError(
            f"GPU{spec.physical_gpu} has unapproved compute PIDs: {sorted(observed - approved)}"
        )
    return {
        "required_interval_seconds": 5,
        "first": first,
        "second": second,
        "approved_preexisting_compute_pids": sorted(approved),
        "observed_preexisting_compute_pids": sorted(observed),
        "automatic_process_termination": False,
    }


def verify_launch_authorization(root: Path, arm_id: str) -> dict[str, Any]:
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    allocation = _read(resolved["allocation"])
    profile_allocation = allocation["arm_specific_resource_profiles"]
    if profile_allocation["physical_gpu_by_arm"].get(arm_id) != spec.physical_gpu:
        raise PermissionError("arm-to-GPU allocation drifted")
    authorized_by_arm = profile_allocation.get("profile_launch_authorized_by_arm", {})
    if authorized_by_arm.get(arm_id) is not True:
        raise PermissionError(
            f"resource allocation has not authorized profile launch for {arm_id}"
        )
    if not resolved["authorization"].is_file():
        raise PermissionError(
            f"missing attempt-specific authorization: {resolved['authorization']}"
        )
    authorization = _read(resolved["authorization"])
    expected = {
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "arm_id": arm_id,
        "authorized": True,
        "physical_gpu": spec.physical_gpu,
        "resource_profile_only": True,
        "effect_experiment_authorized": False,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"invalid profile authorization field: {key}")
    if not authorization.get("researcher_direction"):
        raise PermissionError("profile authorization lacks researcher_direction")
    if arm_id == "G0_GRAM_B0_FRESH" and authorization.get("gpu1_handoff_completed") is not True:
        raise PermissionError("G0 profile requires an explicit GPU1 handoff record")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256(resolved["authorization"]),
        "allocation_sha256": sha256(resolved["allocation"]),
    }


def _assert_native_profiles_sequential(root: Path, arm_id: str) -> None:
    if arm_id not in {"N0_NATIVE_PSID", "N1_NATIVE_LATTE"}:
        return
    other = "N1_NATIVE_LATTE" if arm_id == "N0_NATIVE_PSID" else "N0_NATIVE_PSID"
    other_status = (
        root / f"artifacts/phase17/status/{experiment_id(other)}.status.json"
    )
    if other_status.is_file() and _read(other_status).get("scientific_state") == "RUNNING":
        raise RuntimeError("GPU4 native resource profiles must run sequentially")


def launch(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"profile is not launchable: {status['scientific_state']}")
    _assert_native_profiles_sequential(root, arm_id)
    authorization = verify_launch_authorization(root, arm_id)
    admission = two_snapshot_admission(spec, authorization["authorization"])
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
        "S17_FP12_RESOURCE_PROFILE_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="background_started",
        gpu_ids=[spec.physical_gpu],
        launch_authorized=True,
        allocation_sha256=authorization["allocation_sha256"],
        authorization_sha256=authorization["authorization_sha256"],
        gpu_snapshot=admission,
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP12_RESOURCE_PROFILE_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                automatic_retry=False,
            )
        raise RuntimeError("resource profile worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, arm_id: str, manifest_path: Path) -> int:
    root = root.resolve()
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    started = time.monotonic()
    try:
        verify_run_snapshot(root, manifest_path)
        verify_dependencies(root)
        authorization = verify_launch_authorization(root, arm_id)
        admission = two_snapshot_admission(spec, authorization["authorization"])
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP12_RESOURCE_PROFILE_RUNNING",
            workload_pid=os.getpid(),
            process_alive=True,
            stage="initializing_resource_workload",
            gpu_ids=[spec.physical_gpu],
            gpu_snapshot=admission,
        )
        from experiment.phase17.core.full_latte_profile_executor import (
            run_resource_profile,
        )

        def heartbeat(stage: str, progress: dict[str, Any]) -> None:
            writer.heartbeat(stage=stage, progress=progress)

        measurements = run_resource_profile(
            root,
            arm_id,
            train_batch_size=spec.train_batch_size,
            eval_batch_size=spec.eval_batch_size,
            heartbeat=heartbeat,
        )
        if measurements["peak_reserved_mib"] > spec.peak_cap_mib:
            raise RuntimeError(
                f"measured peak {measurements['peak_reserved_mib']:.1f} MiB exceeds "
                f"frozen cap {spec.peak_cap_mib} MiB"
            )
        post_snapshot = gpu_snapshot_once(spec)
        summary = {
            "schema_version": "phase17.s17_fp12_resource_profile_summary.v1",
            "verdict": "PASS_S17_FP12_ARM_RESOURCE_PROFILE",
            "completed_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "arm_id": arm_id,
            "physical_gpu": spec.physical_gpu,
            "peak_reserved_cap_mib": spec.peak_cap_mib,
            "measured": measurements,
            "formal_minimum_free_mib": (
                measurements["peak_reserved_mib"] + SAFETY_MARGIN_MIB
            ),
            "authorization_sha256": authorization["authorization_sha256"],
            "allocation_sha256": authorization["allocation_sha256"],
            "post_gpu_snapshot": post_snapshot,
            "resource_only": True,
            "effect_metrics_computed": False,
            "automatic_process_termination": False,
            "automatic_retry": False,
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
            "effect_experiment_started": False,
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP12_ARM_RESOURCE_PROFILE",
            process_alive=False,
            workload_pid=0,
            stage="resource_profile_complete",
            progress={"current": 4, "total": 4, "unit": "profile_gate"},
            gpu_ids=[],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            profiled_physical_gpu=spec.physical_gpu,
            peak_allocated_mib=measurements["peak_allocated_mib"],
            peak_reserved_mib=measurements["peak_reserved_mib"],
            formal_minimum_free_mib=summary["formal_minimum_free_mib"],
            external_target_materialized=False,
            result_selection_eligible=False,
            affects_scientific_result=False,
        )
        return 0
    except BaseException as error:
        resolved["result"].mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "arm_id": arm_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "automatic_process_termination": False,
            "external_target_materialized": False,
            "effect_metrics_computed": False,
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
                "S17_FP12_RESOURCE_PROFILE_FAILED_NO_RETRY",
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
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
        return 1


def inspect(root: Path, arm_id: str) -> dict[str, Any]:
    resolved = paths(root.resolve(), arm_id)
    result = {
        "arm_id": arm_id,
        "experiment_id": experiment_id(arm_id),
        "prepared": resolved["config"].is_file() and resolved["snapshot"].is_file(),
        "authorization_present": resolved["authorization"].is_file(),
        "summary_present": resolved["summary"].is_file(),
        "failure_present": resolved["failure"].is_file(),
        "gpu_used_by_inspection": False,
    }
    status_path = resolved["status_dir"] / f"{experiment_id(arm_id)}.status.json"
    if status_path.is_file():
        status = _read(status_path)
        result["status"] = {
            key: status.get(key)
            for key in (
                "scientific_state",
                "execution_state",
                "status_code",
                "stage",
                "launch_authorized",
            )
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "worker", "inspect"))
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root, args.arm)
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
