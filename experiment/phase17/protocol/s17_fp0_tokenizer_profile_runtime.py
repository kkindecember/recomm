#!/usr/bin/env python3
"""Bounded researcher-authorized GPU0 SentenceT5 tokenizer profile for Stage17 FP0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_latte_native_adapter import read_item_metadata_catalog
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
EXPERIMENT_ID = "s17_fp0_tokenizer_bounded_profile"
ATTEMPT_ID = "attempt_010"
PRIOR_ATTEMPT_ID = "attempt_009"
STEP_ID = "S17-FP0-TOKENIZER-PROFILE"
TMUX_SESSION = EXPERIMENT_ID
DEPENDENCY_EXPERIMENT_ID = "s17_fp0_sentence_t5_cache"
DEPENDENCY_PASS_CODE = "PASS_S17_FP0_SENTENCE_T5_CACHE_READY"
ENV_DEPENDENCY_EXPERIMENT_ID = "s17_fp0_cuda_compat_env"
ENV_DEPENDENCY_PASS_CODE = "PASS_S17_FP0_CUDA_COMPAT_ENV_READY"
MODEL_REVISION = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
SAMPLE_SIZE = 512
BATCH_SIZE = 32
EXPECTED_EMBEDDING_DIM = 768
EXPECTED_PEAK_MIB = 10240
SAFETY_MARGIN_MIB = 4096
MAX_GPU_UTILIZATION_PERCENT = 5
PROFILE_TIMEOUT_SECONDS = 600
DEPENDENCY_WAIT_TIMEOUT_SECONDS = 21600
GPU_ADMISSION_WAIT_TIMEOUT_SECONDS = 21600
GPU_ADMISSION_POLL_SECONDS = 60
HEARTBEAT_SECONDS = 30
RESERVED_GPU_ID = 1
TARGET_GPU_ID = 0
GPU1_SHARED_AUTHORIZATION = (
    "researcher_confirmed_2026-08-31_attempt004_smoke_and_attempt009_headroom_wait"
)
GPU0_SHARED_AUTHORIZATION = "researcher_directed_2026-08-31_attempt010_use_gpu0"
GPU_QUERY_ATTEMPTS = 3
GPU_QUERY_RETRY_SECONDS = 2


class DependencyBlockedError(RuntimeError):
    """A prerequisite is blocked, so this attempt must also be BLOCKED, not FAILED."""


def paths(root: Path) -> dict[str, Path]:
    result = root / f"artifacts/phase17/fullport/fp0/tokenizer_profile/{ATTEMPT_ID}"
    return {
        "result": result,
        "config": result / "config.json",
        "sample": result / "sample.json",
        "summary": result / "summary.json",
        "log": result / "run.log",
        "model": root
        / f"artifacts/phase17/fullport/models/sentence-t5-base_{MODEL_REVISION}",
        "native_env": root
        / "artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126",
        "dependency_status": root
        / f"artifacts/phase17/status/{DEPENDENCY_EXPERIMENT_ID}.status.json",
        "env_dependency_status": root
        / f"artifacts/phase17/status/{ENV_DEPENDENCY_EXPERIMENT_ID}.status.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP0-TOKENIZER-PROFILE.attempts.jsonl",
        "snapshot": root / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "snapshot_worker": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/000_s17_fp0_tokenizer_profile_runtime.py",
    }


def select_profile_sample(root: Path) -> list[dict[str, str]]:
    catalog, item2meta = read_item_metadata_catalog(
        root / "GRAM/rec_datasets/Toys/item_plain_text.txt", root=root
    )
    ranked = sorted(
        catalog,
        key=lambda item: (
            hashlib.sha256(f"s17-fp0-tokenizer-profile:{item}".encode("utf-8")).hexdigest(),
            item,
        ),
    )
    selected = ranked[:SAMPLE_SIZE]
    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError("Toys metadata catalog is too small for the profile")
    return [{"item_id": item, "text": item2meta[item]} for item in selected]


def query_compute_processes() -> dict[int, list[dict[str, Any]]]:
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    uuid_to_index: dict[str, int] = {}
    for row in csv.reader(io.StringIO(gpu_query.stdout)):
        if len(row) == 2:
            uuid_to_index[row[1].strip()] = int(row[0].strip())
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    processes = {index: [] for index in uuid_to_index.values()}
    for row in csv.reader(io.StringIO(process_query.stdout)):
        if len(row) != 4 or row[0].strip() not in uuid_to_index:
            continue
        processes[uuid_to_index[row[0].strip()]].append(
            {
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "used_memory_mib": int(row[3].strip()),
            }
        )
    return processes


def query_gpu_state_with_retries() -> tuple[list[Any], dict[int, list[dict[str, Any]]]]:
    """Retry transient read-only NVML/nvidia-smi failures without retrying the profile."""
    errors: list[str] = []
    for attempt in range(1, GPU_QUERY_ATTEMPTS + 1):
        try:
            return query_gpus(), query_compute_processes()
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"attempt_{attempt}={error!r}")
            if attempt < GPU_QUERY_ATTEMPTS:
                time.sleep(GPU_QUERY_RETRY_SECONDS)
    raise RuntimeError("GPU state query failed after bounded read retries: " + "; ".join(errors))


def choose_safe_non_gpu1(
    gpu_records: list[Any], compute_processes: dict[int, list[dict[str, Any]]]
) -> Any | None:
    needed = EXPECTED_PEAK_MIB + SAFETY_MARGIN_MIB
    eligible = [
        row
        for row in gpu_records
        if row.index != RESERVED_GPU_ID
        and not compute_processes.get(row.index, [])
        and row.utilization_percent <= MAX_GPU_UTILIZATION_PERCENT
        and row.free_mib >= needed
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row.utilization_percent, -row.free_mib, row.index))


def choose_authorized_shared_gpu1(
    gpu_records: list[Any], compute_processes: dict[int, list[dict[str, Any]]]
) -> tuple[Any | None, str]:
    """Admit GPU1 only when its existing workload is present and headroom is sufficient."""
    matches = [row for row in gpu_records if row.index == RESERVED_GPU_ID]
    if len(matches) != 1:
        return None, "BLOCKED_GPU1_NOT_VISIBLE"
    if not compute_processes.get(RESERVED_GPU_ID, []):
        return None, "BLOCKED_GPU1_REPEAT_NOT_PRESENT"
    needed = EXPECTED_PEAK_MIB + SAFETY_MARGIN_MIB
    if matches[0].free_mib < needed:
        return None, "BLOCKED_GPU1_SHARED_HEADROOM_INSUFFICIENT"
    return matches[0], "GPU1_SHARED_AUTHORIZED_WITH_EXISTING_REPEAT_AND_MEMORY_MARGIN"


def choose_authorized_shared_gpu0(gpu_records: list[Any]) -> tuple[Any | None, str]:
    """Admit only researcher-selected GPU0 when bounded-profile headroom is sufficient."""

    matches = [row for row in gpu_records if row.index == TARGET_GPU_ID]
    if len(matches) != 1:
        return None, "BLOCKED_GPU0_NOT_VISIBLE"
    needed = EXPECTED_PEAK_MIB + SAFETY_MARGIN_MIB
    if matches[0].free_mib < needed:
        return None, "BLOCKED_GPU0_SHARED_HEADROOM_INSUFFICIENT"
    return matches[0], "GPU0_SHARED_AUTHORIZED_WITH_MEMORY_MARGIN"


def controlled_environment(root: Path, gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
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
        str(LAUNCH_PYTHON),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def profile_script() -> str:
    return (
        "import json, sys, time, numpy as np, torch; "
        "from sentence_transformers import SentenceTransformer; "
        "sample=json.load(open(sys.argv[1], encoding='utf-8')); "
        "texts=[row['text'] for row in sample]; "
        "torch.cuda.set_device(0); torch.cuda.empty_cache(); "
        "load_start=time.perf_counter(); "
        "model=SentenceTransformer(sys.argv[2], local_files_only=True, device='cuda'); "
        "torch.cuda.synchronize(); load_seconds=time.perf_counter()-load_start; "
        "torch.cuda.reset_peak_memory_stats(); encode_start=time.perf_counter(); "
        "values=model.encode(texts, batch_size=int(sys.argv[3]), convert_to_numpy=True, "
        "show_progress_bar=False, device='cuda'); "
        "torch.cuda.synchronize(); encode_seconds=time.perf_counter()-encode_start; "
        "print(json.dumps({'sample_size': len(texts), 'batch_size': int(sys.argv[3]), "
        "'shape': list(values.shape), 'finite': bool(np.isfinite(values).all()), "
        "'dtype': str(values.dtype), 'load_seconds': load_seconds, "
        "'encode_seconds': encode_seconds, 'items_per_second': len(texts)/encode_seconds, "
        "'peak_allocated_mib': torch.cuda.max_memory_allocated()/1048576, "
        "'peak_reserved_mib': torch.cuda.max_memory_reserved()/1048576, "
        "'device_name': torch.cuda.get_device_name(0)}, sort_keys=True))"
    )


def terminate_exact_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def wait_for_dependency(
    writer: StatusWriter,
    status_path: Path,
    *,
    pass_code: str,
    label: str,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        dependency = json.loads(status_path.read_text(encoding="utf-8"))
        if dependency["scientific_state"] == "COMPLETED":
            if dependency["status_code"] != pass_code:
                raise RuntimeError(
                    f"{label} dependency completed without PASS: {dependency['status_code']}"
                )
            return dependency
        if dependency["scientific_state"] == "BLOCKED":
            raise DependencyBlockedError(
                f"{label} dependency terminal: BLOCKED {dependency['status_code']}"
            )
        if dependency["scientific_state"] in {"FAILED", "STOPPED"}:
            raise RuntimeError(
                f"{label} dependency terminal: {dependency['scientific_state']} "
                f"{dependency['status_code']}"
            )
        elapsed = time.monotonic() - started
        if elapsed > DEPENDENCY_WAIT_TIMEOUT_SECONDS:
            raise TimeoutError(f"timed out waiting for {label}")
        writer.heartbeat(
            stage=f"waiting_for_{label.lower().replace(' ', '_')}",
            progress={
                "current": min(int(elapsed), DEPENDENCY_WAIT_TIMEOUT_SECONDS),
                "total": DEPENDENCY_WAIT_TIMEOUT_SECONDS,
                "unit": "seconds_until_dependency_timeout",
            },
        )
        time.sleep(60)


def wait_for_authorized_shared_gpu0(
    writer: StatusWriter,
) -> tuple[Any | None, dict[int, list[dict[str, Any]]], dict[str, Any], str]:
    """Wait in the background for bounded GPU0 profile headroom without changing its jobs."""

    started = time.monotonic()
    while True:
        gpu_records, compute_processes = query_gpu_state_with_retries()
        selected, admission_code = choose_authorized_shared_gpu0(gpu_records)
        gpu_snapshot = {
            "captured_at": utc_now(),
            "devices": snapshot(gpu_records),
            "compute_processes": compute_processes,
            "selection_rule": "researcher_selected_gpu0_free_gte_14336_mib",
            "shared_gpu0_authorization": GPU0_SHARED_AUTHORIZATION,
        }
        if selected is not None:
            return selected, compute_processes, gpu_snapshot, admission_code
        elapsed = time.monotonic() - started
        if elapsed >= GPU_ADMISSION_WAIT_TIMEOUT_SECONDS:
            writer.transition(
                "BLOCKED",
                "BLOCKED",
                "BLOCKED_GPU0_HEADROOM_WAIT_TIMEOUT",
                process_alive=False,
                workload_pid=0,
                stage="gpu0_profile_headroom_wait_timeout",
                gpu_ids=[],
                gpu_snapshot=gpu_snapshot,
                selection_block_reason=admission_code,
                automatic_retry=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
            return None, compute_processes, gpu_snapshot, admission_code
        writer.transition(
            "RUNNING",
            "WAITING_FOR_GPU",
            "S17_FP0_TOKENIZER_PROFILE_WAITING_FOR_GPU0_HEADROOM",
            process_alive=True,
            workload_pid=os.getpid(),
            stage="waiting_for_gpu0_profile_headroom",
            gpu_ids=[],
            gpu_snapshot=gpu_snapshot,
            selection_block_reason=admission_code,
            progress={
                "current": min(int(elapsed), GPU_ADMISSION_WAIT_TIMEOUT_SECONDS),
                "total": GPU_ADMISSION_WAIT_TIMEOUT_SECONDS,
                "unit": "seconds_until_gpu_admission_timeout",
            },
        )
        time.sleep(GPU_ADMISSION_POLL_SECONDS)


def run_profile(
    *,
    root: Path,
    resolved: dict[str, Path],
    writer: StatusWriter,
    gpu_id: int,
) -> tuple[dict[str, Any], float]:
    native_python = resolved["native_env"] / "bin/python"
    command = [
        str(native_python),
        "-c",
        profile_script(),
        str(resolved["sample"]),
        str(resolved["model"]),
        str(BATCH_SIZE),
    ]
    started = time.monotonic()
    with resolved["log"].open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            env=controlled_environment(root, gpu_id),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP0_TOKENIZER_BOUNDED_PROFILE_RUNNING",
            workload_pid=process.pid,
            process_alive=True,
            gpu_ids=[gpu_id],
            stage="sentence_t5_512_item_profile",
        )
        while True:
            try:
                return_code = process.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                writer.heartbeat(
                    stage="sentence_t5_512_item_profile",
                    progress={
                        "current": min(int(elapsed), PROFILE_TIMEOUT_SECONDS),
                        "total": PROFILE_TIMEOUT_SECONDS,
                        "unit": "seconds_until_hard_timeout",
                    },
                )
                if elapsed > PROFILE_TIMEOUT_SECONDS:
                    terminate_exact_process_group(process)
                    raise TimeoutError(f"bounded tokenizer profile exceeded {PROFILE_TIMEOUT_SECONDS}s")
    if return_code != 0:
        raise RuntimeError(f"bounded tokenizer profile exited with code {return_code}")
    lines = [line for line in resolved["log"].read_text(encoding="utf-8").splitlines() if line.strip()]
    result = json.loads(lines[-1])
    return result, time.monotonic() - started


def prepare(root: Path) -> int:
    resolved = paths(root)
    prior_status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if prior_status_path.exists():
        prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
        if prior_status["scientific_state"] not in {"FAILED", "STOPPED", "BLOCKED"}:
            raise FileExistsError("prior tokenizer profile attempt is not terminal")
        if prior_status["attempt_id"] != PRIOR_ATTEMPT_ID:
            raise FileExistsError("unexpected prior tokenizer profile attempt id")
    if not resolved["dependency_status"].is_file():
        raise FileNotFoundError("SentenceT5 cache status must exist before profile preparation")
    if not resolved["env_dependency_status"].is_file():
        raise FileNotFoundError("CUDA compatibility status must exist before profile preparation")
    resolved["result"].mkdir(parents=True, exist_ok=False)
    sample = select_profile_sample(root)
    with resolved["sample"].open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    config = {
        "schema_version": "phase17.s17_fp0_tokenizer_profile_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "dependency_experiment_id": DEPENDENCY_EXPERIMENT_ID,
        "dependency_pass_code": DEPENDENCY_PASS_CODE,
        "env_dependency_experiment_id": ENV_DEPENDENCY_EXPERIMENT_ID,
        "env_dependency_pass_code": ENV_DEPENDENCY_PASS_CODE,
        "model_revision": MODEL_REVISION,
        "sample_size": SAMPLE_SIZE,
        "sample_sha256": sha256(resolved["sample"]),
        "batch_size": BATCH_SIZE,
        "expected_embedding_dim": EXPECTED_EMBEDDING_DIM,
        "expected_peak_mib": EXPECTED_PEAK_MIB,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "max_gpu_utilization_percent": MAX_GPU_UTILIZATION_PERCENT,
        "profile_timeout_seconds": PROFILE_TIMEOUT_SECONDS,
        "dependency_wait_timeout_seconds": DEPENDENCY_WAIT_TIMEOUT_SECONDS,
        "gpu_admission_wait_timeout_seconds": GPU_ADMISSION_WAIT_TIMEOUT_SECONDS,
        "gpu_admission_poll_seconds": GPU_ADMISSION_POLL_SECONDS,
        "gpu_admission_waits_in_background": True,
        "background_required": True,
        "gpu_selection": "researcher_selected_shared_gpu0_free_gte_14336_mib",
        "target_gpu_id": TARGET_GPU_ID,
        "gpu0_shared_authorized": True,
        "gpu0_shared_authorization": GPU0_SHARED_AUTHORIZATION,
        "gpu1_allowed": False,
        "full_data_tokenizer_started": False,
        "effect_experiment_started": False,
        "automatic_retry": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
    atomic_json(resolved["config"], config)
    command = worker_command(root, resolved)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=command,
        source_paths=[Path(__file__)],
        config=config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": STEP_ID,
            "kind": "bounded_single_gpu_tokenizer_profile",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY",
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
        track_id="FP0-PROFILE",
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": 3, "unit": "profile_gate"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "dependency_experiment_id": DEPENDENCY_EXPERIMENT_ID,
            "env_dependency_experiment_id": ENV_DEPENDENCY_EXPERIMENT_ID,
            "sample_sha256": sha256(resolved["sample"]),
            "expected_peak_mib": EXPECTED_PEAK_MIB,
            "safety_margin_mib": SAFETY_MARGIN_MIB,
            "gpu_ids": [],
            "target_gpu_id": TARGET_GPU_ID,
            "gpu0_shared_authorized": True,
            "gpu0_shared_authorization": GPU0_SHARED_AUTHORIZATION,
            "gpu0_preexisting_processes_preserved": None,
            "gpu1_allowed": False,
            "automatic_retry": False,
            "full_data_tokenizer_started": False,
            "effect_experiment_started": False,
            "affects_scientific_result": False,
            "result_selection_eligible": False,
            "d1_read": False,
            "d2_read": False,
            "prior_attempt_id": PRIOR_ATTEMPT_ID,
            "prior_failure_path": "artifacts/phase17/fullport/fp0/tokenizer_profile/attempt_009/run.log",
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP0_TOKENIZER_PROFILE_READY_TO_LAUNCH",
        process_alive=False,
    )
    print(manifest)
    return 0


def launch(root: Path) -> int:
    resolved = paths(root)
    status = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"tokenizer profile is not launchable: {status['scientific_state']}")
    command = worker_command(root, resolved)
    session = launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=command,
        cwd=root,
        tmux_session=TMUX_SESSION,
        startup_log_path=resolved["log"],
    )
    StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP0_TOKENIZER_PROFILE_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="waiting_for_cuda_compat_env",
    )
    if not wait_for_tmux_startup(session):
        latest = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
        if latest["scientific_state"] == "RUNNING":
            StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_TOKENIZER_PROFILE_STARTUP_HANDSHAKE_FAILED",
                process_alive=False,
                workload_pid=0,
                stage="startup_handshake_failed_no_retry",
                automatic_retry=False,
            )
        raise RuntimeError("tokenizer profile worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, snapshot_manifest: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    preexisting_target_processes: list[dict[str, Any]] = []
    try:
        verify_run_snapshot(root, snapshot_manifest)
        frozen_config = json.loads(
            snapshot_manifest.parent.joinpath("config.json").read_text(encoding="utf-8")
        )
        if sha256(resolved["sample"]) != frozen_config["sample_sha256"]:
            raise RuntimeError("bounded tokenizer sample changed after preflight")
        dependency = wait_for_dependency(
            writer,
            resolved["dependency_status"],
            pass_code=DEPENDENCY_PASS_CODE,
            label="SentenceT5 cache",
        )
        environment_dependency = wait_for_dependency(
            writer,
            resolved["env_dependency_status"],
            pass_code=ENV_DEPENDENCY_PASS_CODE,
            label="CUDA compatibility environment",
        )
        if not resolved["model"].is_dir() or not (resolved["native_env"] / "bin/python").is_file():
            raise FileNotFoundError("profile dependencies are not materialized")
        selected, compute_processes, gpu_snapshot, admission_code = (
            wait_for_authorized_shared_gpu0(writer)
        )
        if selected is None:
            return 0
        preexisting_target_processes = [
            dict(row) for row in compute_processes.get(TARGET_GPU_ID, [])
        ]
        preexisting_target_pids = sorted(row["pid"] for row in preexisting_target_processes)
        gpu_snapshot["selected_gpu"] = selected.index
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP0_TOKENIZER_PROFILE_GPU_ADMITTED",
            process_alive=True,
            stage="gpu_admitted",
            gpu_ids=[selected.index],
            gpu_snapshot=gpu_snapshot,
            selection_reason=admission_code,
            gpu0_preexisting_processes=preexisting_target_processes,
            gpu0_preexisting_pids=preexisting_target_pids,
            gpu0_preexisting_processes_preserved=None,
        )
        result, wall_seconds = run_profile(
            root=root,
            resolved=resolved,
            writer=writer,
            gpu_id=selected.index,
        )
        if result["shape"] != [SAMPLE_SIZE, EXPECTED_EMBEDDING_DIM] or not result["finite"]:
            raise RuntimeError(f"invalid bounded tokenizer profile output: {result}")
        if result["peak_reserved_mib"] > EXPECTED_PEAK_MIB:
            raise RuntimeError(
                f"profile peak {result['peak_reserved_mib']:.1f} MiB exceeds "
                f"pre-registered {EXPECTED_PEAK_MIB} MiB"
            )
        post_gpu_records, post_compute_processes = query_gpu_state_with_retries()
        post_target_pids = sorted(
            row["pid"] for row in post_compute_processes.get(TARGET_GPU_ID, [])
        )
        missing_preexisting_target_pids = sorted(
            set(preexisting_target_pids) - set(post_target_pids)
        )
        target_processes_preserved = not missing_preexisting_target_pids
        if not target_processes_preserved:
            raise RuntimeError(
                "pre-existing GPU0 process disappeared during bounded profile: "
                f"{missing_preexisting_target_pids}"
            )
        post_gpu_snapshot = {
            "captured_at": utc_now(),
            "devices": snapshot(post_gpu_records),
            "compute_processes": post_compute_processes,
        }
        estimated_full_seconds = result["encode_seconds"] * (11924 / SAMPLE_SIZE)
        summary = {
            "schema_version": "phase17.s17_fp0_tokenizer_profile_summary.v1",
            "verdict": "PASS_S17_FP0_TOKENIZER_BOUNDED_PROFILE",
            "completed_at": utc_now(),
            "dependency_status_code": dependency["status_code"],
            "dependency_status_sha256": sha256(resolved["dependency_status"]),
            "environment_dependency_status_code": environment_dependency["status_code"],
            "environment_dependency_status_sha256": sha256(resolved["env_dependency_status"]),
            "physical_gpu": selected.index,
            "sample_sha256": sha256(resolved["sample"]),
            "profile": result,
            "worker_wall_seconds": wall_seconds,
            "linear_full_catalog_encode_estimate_seconds": estimated_full_seconds,
            "estimate_scope_warning": "encoding-only linear estimate; excludes PCA/RQ/conflict-resolution and startup",
            "full_data_tokenizer_started": False,
            "effect_experiment_started": False,
            "gpu0_used": True,
            "gpu0_shared_authorization": GPU0_SHARED_AUTHORIZATION,
            "gpu0_preexisting_processes": preexisting_target_processes,
            "gpu0_preexisting_pids": preexisting_target_pids,
            "gpu0_post_profile_snapshot": post_gpu_snapshot,
            "gpu0_preexisting_processes_preserved": target_processes_preserved,
            "automatic_retry": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
            "next_gate": "REQUEST_FULL_DATA_TOKENIZER_GPU_ALLOCATION",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP0_TOKENIZER_BOUNDED_PROFILE",
            process_alive=False,
            workload_pid=0,
            stage="bounded_tokenizer_profile_complete",
            progress={"current": 3, "total": 3, "unit": "profile_gate"},
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            gpu_ids=[],
            profiled_physical_gpu=selected.index,
            profiled_peak_allocated_mib=result["peak_allocated_mib"],
            profiled_peak_reserved_mib=result["peak_reserved_mib"],
            gpu0_shared_authorized=True,
            gpu0_preexisting_pids=preexisting_target_pids,
            gpu0_post_profile_snapshot=post_gpu_snapshot,
            gpu0_preexisting_processes_preserved=target_processes_preserved,
            result_selection_eligible=False,
            affects_scientific_result=False,
        )
        return 0
    except DependencyBlockedError as error:
        resolved["result"].mkdir(parents=True, exist_ok=True)
        with resolved["log"].open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] dependency_blocked={error!r}\n")
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}:
            writer.transition(
                "BLOCKED",
                "BLOCKED",
                "BLOCKED_CUDA_COMPAT_ENV_DEPENDENCY",
                process_alive=False,
                workload_pid=0,
                stage="dependency_blocked_no_profile",
                dependency_block_reason=repr(error),
                automatic_retry=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
                gpu_ids=[],
            )
        return 0
    except BaseException as error:
        resolved["result"].mkdir(parents=True, exist_ok=True)
        with resolved["log"].open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] terminal_error={error!r}\n")
            log.write(traceback.format_exc())
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}:
            preservation_fields: dict[str, Any] = {}
            if preexisting_target_processes:
                try:
                    post_compute_processes = query_compute_processes()
                    preexisting_pids = sorted(row["pid"] for row in preexisting_target_processes)
                    post_pids = sorted(
                        row["pid"] for row in post_compute_processes.get(TARGET_GPU_ID, [])
                    )
                    preservation_fields = {
                        "gpu0_preexisting_pids": preexisting_pids,
                        "gpu0_post_failure_compute_processes": post_compute_processes,
                        "gpu0_preexisting_processes_preserved": set(preexisting_pids).issubset(
                            post_pids
                        ),
                    }
                except BaseException as preservation_error:
                    preservation_fields = {
                        "gpu0_preexisting_processes_preserved": None,
                        "gpu0_preservation_check_error": repr(preservation_error),
                    }
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_TOKENIZER_BOUNDED_PROFILE_FAILED",
                process_alive=False,
                workload_pid=0,
                stage="terminal_failure_no_retry",
                terminal_error=repr(error),
                automatic_retry=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
                gpu_ids=[],
                **preservation_fields,
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
