#!/usr/bin/env python3
"""Authorized background runner for the Stage17 full-data LATTE tokenizer.

Preparation is CPU-only and safe.  Launch fails closed unless both the frozen
resource allocation and an attempt-specific researcher authorization record
explicitly permit physical GPU0.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_latte_tokenizer import (
    FullLatteTokenizerSpec,
    build_full_data_tokenizer,
)
from experiment.phase17.core.resource_profiler import query_gpus, snapshot
from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    verify_run_snapshot,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import AttemptLedger, StatusWriter, atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
LAUNCH_PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
NATIVE_PYTHON = Path(
    "artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python"
)
EXPERIMENT_ID = "s17_fp0_full_data_tokenizer"
ATTEMPT_ID = "attempt_001"
STEP_ID = "S17-FP0-FULL-DATA-TOKENIZER"
TMUX_SESSION = EXPERIMENT_ID
TARGET_GPU_ID = 0
MINIMUM_FREE_MIB = 5080
EXPECTED_CATALOG_ITEMS = 11924
EXPECTED_FIT_ITEMS = 11138
MODEL_REVISION = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
DEPENDENCIES = {
    "s17_fp0_native_data_adapter_audit": "PASS_S17_FP0_NATIVE_DATA_ADAPTER",
    "s17_fp0_sentence_t5_cache": "PASS_S17_FP0_SENTENCE_T5_CACHE_READY",
    "s17_fp0_cuda_compat_env": "PASS_S17_FP0_CUDA_COMPAT_ENV_READY",
    "s17_fp0_tokenizer_bounded_profile": "PASS_S17_FP0_TOKENIZER_BOUNDED_PROFILE",
}


def paths(root: Path) -> dict[str, Path]:
    result = root / f"artifacts/phase17/fullport/fp0/full_data_tokenizer/{ATTEMPT_ID}"
    return {
        "result": result,
        "output": result / "tokenizer",
        "config": result / "config.json",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "model": root
        / f"artifacts/phase17/fullport/models/sentence-t5-base_{MODEL_REVISION}",
        "native_cache": root / "artifacts/phase17/fullport/cache/latte_native_toys_d0",
        "native_python": root / NATIVE_PYTHON,
        "allocation": root / "experiment/phase17/config/s17_fp_resource_allocation.json",
        "authorization": root
        / f"artifacts/phase17/authorizations/{EXPERIMENT_ID}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root
        / "artifacts/phase17/attempts/S17-FP0-FULL-DATA-TOKENIZER.attempts.jsonl",
        "snapshot": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "snapshot_worker": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/000_s17_fp0_full_data_tokenizer_runtime.py",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dependencies(root: Path) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for experiment_id, pass_code in DEPENDENCIES.items():
        path = root / f"artifacts/phase17/status/{experiment_id}.status.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing tokenizer dependency status: {path}")
        status = _read_json(path)
        if status["scientific_state"] != "COMPLETED" or status["status_code"] != pass_code:
            raise RuntimeError(
                f"tokenizer dependency is not PASS: {experiment_id} "
                f"{status['scientific_state']} {status['status_code']}"
            )
        resolved[experiment_id] = {
            "status_code": status["status_code"],
            "status_sha256": sha256(path),
        }
    return resolved


def verify_launch_authorization(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    allocation = _read_json(resolved["allocation"])
    tokenizer_allocation = allocation["full_data_tokenizer"]
    if tokenizer_allocation.get("physical_gpu") != TARGET_GPU_ID:
        raise PermissionError("resource allocation does not assign tokenizer to GPU0")
    if tokenizer_allocation.get("minimum_free_mib") != MINIMUM_FREE_MIB:
        raise RuntimeError("tokenizer memory admission contract drifted")
    if tokenizer_allocation.get("launch_authorized") is not True:
        raise PermissionError(
            "full-data tokenizer launch is not authorized in s17_fp_resource_allocation.json"
        )
    if not resolved["authorization"].is_file():
        raise PermissionError(
            f"missing attempt-specific researcher authorization: {resolved['authorization']}"
        )
    authorization = _read_json(resolved["authorization"])
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "authorized": True,
        "physical_gpu": TARGET_GPU_ID,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"invalid launch authorization field {key}")
    if not authorization.get("researcher_direction"):
        raise PermissionError("authorization lacks researcher_direction")
    return {
        "allocation_sha256": sha256(resolved["allocation"]),
        "authorization_sha256": sha256(resolved["authorization"]),
        "authorization": authorization,
    }


def query_compute_processes() -> dict[int, list[dict[str, Any]]]:
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    uuid_to_index: dict[str, int] = {}
    for row in csv.reader(io.StringIO(gpu_rows)):
        if len(row) == 2:
            uuid_to_index[row[1].strip()] = int(row[0].strip())
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


def gpu_admission_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = query_gpus()
    matches = [row for row in records if row.index == TARGET_GPU_ID]
    if len(matches) != 1:
        raise RuntimeError("physical GPU0 is not uniquely visible")
    processes = query_compute_processes()
    payload = {
        "captured_at": utc_now(),
        "devices": snapshot(records),
        "compute_processes": processes,
        "selected_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "selection_rule": "researcher_allocated_physical_gpu0_with_free_mib_gate",
    }
    if matches[0].free_mib < MINIMUM_FREE_MIB:
        raise RuntimeError(
            f"GPU0 free memory {matches[0].free_mib} MiB is below {MINIMUM_FREE_MIB} MiB"
        )
    return payload, list(processes.get(TARGET_GPU_ID, []))


def controlled_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(TARGET_GPU_ID),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(root),
        }
    )
    return env


def worker_command(root: Path, resolved: dict[str, Path]) -> list[str]:
    return [
        "/usr/bin/env",
        f"PYTHONPATH={root}",
        str(resolved["native_python"]),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def frozen_config(root: Path) -> dict[str, Any]:
    spec = FullLatteTokenizerSpec()
    return {
        "schema_version": "phase17.s17_fp0_full_data_tokenizer_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "expected_catalog_items": EXPECTED_CATALOG_ITEMS,
        "expected_fit_items": EXPECTED_FIT_ITEMS,
        "tokenizer_spec": spec.__dict__,
        "fit_contract": {
            "sentence_embedding_assignment": "complete_metadata_catalog",
            "pca_fit": "train_prefix_mask_only",
            "rqkmeans_fit": "same_train_prefix_mask_only",
            "semantic_code_assignment": "complete_metadata_catalog",
            "collision_resolution": "conflict_free_psid_top5",
        },
        "official_cache_prewrite_required": True,
        "launch_requires_external_authorization_record": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "background_required": True,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "effect_experiment_started": False,
    }


def prepare(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    if resolved["result"].exists() or resolved["snapshot"].exists():
        raise FileExistsError("full-data tokenizer attempt_001 already exists")
    dependencies = verify_dependencies(root)
    if not resolved["model"].is_dir() or not resolved["native_python"].is_file():
        raise FileNotFoundError("full-data tokenizer environment/model is missing")
    resolved["result"].mkdir(parents=True, exist_ok=False)
    config = frozen_config(root)
    config["dependencies"] = dependencies
    atomic_json(resolved["config"], config)
    command = worker_command(root, resolved)
    source_paths = [
        Path(__file__),
        root / "experiment/phase17/core/full_latte_tokenizer.py",
        root / "experiment/phase17/core/full_latte_contracts.py",
        root / "experiment/phase17/core/full_latte_native_adapter.py",
        root / "experiment/phase17/core/fullport_data.py",
        root / "experiment/phase17/core/status_writer.py",
        root / "experiment/phase17/core/run_manager.py",
    ]
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=command,
        source_paths=source_paths,
        config=config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": STEP_ID,
            "kind": "full_catalog_sentence_t5_train_only_pca_rqkmeans",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_AUTHORIZATION_REQUIRED",
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
        track_id="FP0-TOKENIZER",
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "preflight_complete_waiting_researcher_authorization",
            "progress": {"current": 0, "total": 4, "unit": "tokenizer_gate"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "dependencies": dependencies,
            "gpu_ids": [],
            "target_gpu_id": TARGET_GPU_ID,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "launch_authorized": False,
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
        "S17_FP0_FULL_DATA_TOKENIZER_READY_AUTHORIZATION_REQUIRED",
        process_alive=False,
    )
    print(manifest)
    return 0


def launch(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"full-data tokenizer is not launchable: {status['scientific_state']}")
    authorization = verify_launch_authorization(root)
    admission, preexisting = gpu_admission_snapshot()
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
        "S17_FP0_FULL_DATA_TOKENIZER_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="background_started",
        gpu_ids=[TARGET_GPU_ID],
        launch_authorized=True,
        allocation_sha256=authorization["allocation_sha256"],
        authorization_sha256=authorization["authorization_sha256"],
        gpu_snapshot=admission,
        gpu0_preexisting_processes=preexisting,
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_FULL_DATA_TOKENIZER_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                automatic_retry=False,
            )
        raise RuntimeError("full-data tokenizer worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, manifest_path: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    started = time.monotonic()
    try:
        verify_run_snapshot(root, manifest_path)
        authorization = verify_launch_authorization(root)
        verify_dependencies(root)
        admission, preexisting = gpu_admission_snapshot()
        import torch

        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP0_FULL_DATA_TOKENIZER_RUNNING",
            workload_pid=os.getpid(),
            process_alive=True,
            stage="initializing_sentence_t5",
            gpu_ids=[TARGET_GPU_ID],
            gpu_snapshot=admission,
            gpu0_preexisting_processes=preexisting,
        )

        def heartbeat(stage: str, progress: dict[str, Any]) -> None:
            writer.heartbeat(stage=stage, progress=progress)

        result = build_full_data_tokenizer(
            root=root,
            model_path=resolved["model"],
            output_dir=resolved["output"],
            official_cache_dir=resolved["native_cache"],
            spec=FullLatteTokenizerSpec(),
            device="cuda",
            heartbeat=heartbeat,
        )
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / 1048576
        peak_reserved = torch.cuda.max_memory_reserved() / 1048576
        if result.manifest["catalog_items"] != EXPECTED_CATALOG_ITEMS:
            raise RuntimeError("full tokenizer catalog count drifted")
        if result.manifest["fit_catalog_items"] != EXPECTED_FIT_ITEMS:
            raise RuntimeError("full tokenizer fit-mask count drifted")
        if result.resolution.collisions_after != 0:
            raise RuntimeError("full tokenizer retained semantic aliases")
        post_admission, _ = gpu_admission_snapshot()
        summary = {
            "schema_version": "phase17.s17_fp0_full_data_tokenizer_summary.v1",
            "verdict": "PASS_S17_FP0_FULL_DATA_TOKENIZER",
            "completed_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "physical_gpu": TARGET_GPU_ID,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "tokenizer_manifest_path": str((resolved["output"] / "manifest.json").relative_to(root)),
            "tokenizer_manifest_sha256": sha256(resolved["output"] / "manifest.json"),
            "catalog_items": result.manifest["catalog_items"],
            "fit_catalog_items": result.manifest["fit_catalog_items"],
            "collision_resolution": result.manifest["collision_resolution"],
            "allocation_sha256": authorization["allocation_sha256"],
            "authorization_sha256": authorization["authorization_sha256"],
            "gpu0_preexisting_processes": preexisting,
            "post_gpu_snapshot": post_admission,
            "automatic_process_termination": False,
            "automatic_retry": False,
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
            "effect_experiment_started": False,
            "next_gate": "S17_FP1_FP2_ARM_SPECIFIC_RESOURCE_PROFILES",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP0_FULL_DATA_TOKENIZER",
            process_alive=False,
            workload_pid=0,
            stage="full_data_tokenizer_complete",
            progress={"current": 4, "total": 4, "unit": "tokenizer_gate"},
            gpu_ids=[],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            tokenizer_manifest_path=summary["tokenizer_manifest_path"],
            tokenizer_manifest_sha256=summary["tokenizer_manifest_sha256"],
            profiled_physical_gpu=TARGET_GPU_ID,
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
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
        if current["scientific_state"] not in {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_FULL_DATA_TOKENIZER_FAILED_NO_RETRY",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root)
    if args.action == "launch":
        return launch(root)
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
