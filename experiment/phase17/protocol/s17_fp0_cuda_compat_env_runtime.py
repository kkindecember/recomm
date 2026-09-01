#!/usr/bin/env python3
"""Build and validate the pinned CUDA 12.6 LATTE environment for Stage17 FP0."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

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
UV = Path("/home/jiangtangyunzhi/miniconda3/bin/uv")
BOOTSTRAP_PYTHON = Path(
    "/home/jiangtangyunzhi/.local/share/uv/python/"
    "cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
)
EXPERIMENT_ID = "s17_fp0_cuda_compat_env"
ATTEMPT_ID = "attempt_004"
PRIOR_ATTEMPT_ID = "attempt_003"
STEP_ID = "S17-FP0-CUDA-COMPAT-ENV"
TMUX_SESSION = EXPERIMENT_ID
LATTE_COMMIT = "05e4e6d983225bcb7172f148a076890e80c524d1"
TORCH_VERSION = "2.7.1+cu126"
TORCH_CUDA_VERSION = "12.6"
TORCH_WHEEL_SHA256 = "63bce0590bc540fc16139e2be0177847585182b8c5e68d7f9213789d1d96c978"
TORCH_WHEEL_URL = (
    "https://mirrors.aliyun.com/pytorch-wheels/cu126/"
    "torch-2.7.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
    f"#sha256={TORCH_WHEEL_SHA256}"
)
TORCH_INDEX_URL = "https://mirrors.aliyun.com/pytorch-wheels/cu126"
PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
GPU1_ID = 1
CUDA_SMOKE_MIN_FREE_MIB = 1024
PROFILE_ADMISSION_MIB = 14336
GPU_QUERY_ATTEMPTS = 3
GPU_QUERY_RETRY_SECONDS = 2
COMMAND_TIMEOUT_SECONDS = 7200
SMOKE_TIMEOUT_SECONDS = 120
HEARTBEAT_SECONDS = 60
AUTHORIZATION = (
    "researcher_confirmed_2026-08-31_attempt004_smoke_and_attempt009_headroom_wait"
)
EXCLUDED_REQUIREMENT_NAMES = {
    "cuda-bindings",
    "cuda-pathfinder",
    "cuda-toolkit",
    "torch",
    "triton",
}


def paths(root: Path) -> dict[str, Path]:
    result = root / f"artifacts/phase17/fullport/fp0/cuda_compat_env/{ATTEMPT_ID}"
    return {
        "result": result,
        "config": result / "config.json",
        "requirements": result / "requirements.cuda126.txt",
        "freeze": result / "requirements.freeze.txt",
        "environment_manifest": result / "environment_manifest.json",
        "summary": result / "summary.json",
        "log": result / "run.log",
        "env": root
        / "artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126",
        "source": root
        / "artifacts/phase17/fullport/sources/"
        "latte_05e4e6d983225bcb7172f148a076890e80c524d1_attempt_003",
        "source_freeze": root
        / "artifacts/phase17/fullport/fp0/native_env_setup/attempt_003/requirements.freeze.txt",
        "source_manifest": root
        / "artifacts/phase17/fullport/fp0/native_env_setup/attempt_003/environment_manifest.json",
        "uv_cache": root / "artifacts/phase17/fullport/cache/uv_cuda126",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP0-CUDA-COMPAT-ENV.attempts.jsonl",
        "snapshot": root / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "snapshot_worker": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/"
        "000_s17_fp0_cuda_compat_env_runtime.py",
    }


def requirement_name(line: str) -> str:
    return line.split("==", 1)[0].strip().lower().replace("_", "-")


def compatible_requirements(source_text: str) -> list[str]:
    retained: list[str] = []
    for raw in source_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = requirement_name(line)
        if name in EXCLUDED_REQUIREMENT_NAMES:
            continue
        if name.startswith("nvidia-") and name != "nvidia-ml-py":
            continue
        retained.append(line)
    return retained


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
    errors: list[str] = []
    for attempt in range(1, GPU_QUERY_ATTEMPTS + 1):
        try:
            return query_gpus(), query_compute_processes()
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"attempt_{attempt}={error!r}")
            if attempt < GPU_QUERY_ATTEMPTS:
                time.sleep(GPU_QUERY_RETRY_SECONDS)
    raise RuntimeError("GPU state query failed after bounded read retries: " + "; ".join(errors))


def select_authorized_gpu1(
    gpu_records: list[Any], compute_processes: dict[int, list[dict[str, Any]]]
) -> tuple[Any | None, str]:
    matches = [row for row in gpu_records if row.index == GPU1_ID]
    if len(matches) != 1:
        return None, "BLOCKED_GPU1_NOT_VISIBLE"
    if not compute_processes.get(GPU1_ID, []):
        return None, "BLOCKED_GPU1_REPEAT_NOT_PRESENT"
    if matches[0].free_mib < CUDA_SMOKE_MIN_FREE_MIB:
        return None, "BLOCKED_GPU1_SHARED_HEADROOM_INSUFFICIENT"
    return matches[0], "GPU1_SHARED_AUTHORIZED_FOR_CUDA_SMOKE"


def controlled_environment(root: Path, *, gpu_id: int | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if gpu_id is None else str(gpu_id),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "UV_CACHE_DIR": str(paths(root)["uv_cache"]),
            "UV_LINK_MODE": "copy",
            "UV_NO_PROGRESS": "1",
            "UV_HTTP_TIMEOUT": "300",
            "NO_PROXY": "*",
            "no_proxy": "*",
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


def terminate_exact_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_command(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    resolved: dict[str, Path],
    writer: StatusWriter,
    stage: str,
    status_code: str,
    timeout_seconds: int,
) -> float:
    started = time.monotonic()
    with resolved["log"].open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            status_code,
            process_alive=True,
            workload_pid=process.pid,
            stage=stage,
        )
        while True:
            try:
                return_code = process.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                writer.heartbeat(
                    stage=stage,
                    progress={
                        "current": min(int(elapsed), timeout_seconds),
                        "total": timeout_seconds,
                        "unit": "seconds_until_hard_timeout",
                    },
                )
                if elapsed > timeout_seconds:
                    terminate_exact_process_group(process)
                    raise TimeoutError(f"{stage} exceeded {timeout_seconds}s")
    if return_code != 0:
        raise RuntimeError(f"{stage} exited with code {return_code}")
    return time.monotonic() - started


def cpu_validation_script() -> str:
    return (
        "import json, platform, torch, transformers, sentence_transformers, faiss, numpy, sklearn; "
        "print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, "
        "'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), "
        "'transformers': transformers.__version__, "
        "'sentence_transformers': sentence_transformers.__version__, "
        "'faiss': getattr(faiss, '__version__', 'unknown'), 'numpy': numpy.__version__, "
        "'sklearn': sklearn.__version__}, sort_keys=True))"
    )


def cuda_smoke_script() -> str:
    return (
        "import json, torch; torch.cuda.set_device(0); torch.cuda.empty_cache(); "
        "torch.cuda.reset_peak_memory_stats(); x=torch.arange(4096, dtype=torch.float32, device='cuda'); "
        "value=(x.square().mean()).item(); torch.cuda.synchronize(); "
        "print(json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), 'device_name': torch.cuda.get_device_name(0), "
        "'finite': bool(torch.isfinite(torch.tensor(value))), 'value': value, "
        "'peak_reserved_mib': torch.cuda.max_memory_reserved()/1048576}, sort_keys=True))"
    )


def read_last_json(log_path: Path) -> dict[str, Any]:
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def prepare(root: Path) -> int:
    resolved = paths(root)
    for required in (LAUNCH_PYTHON, UV, BOOTSTRAP_PYTHON, resolved["source_freeze"]):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not resolved["source"].is_dir() or not resolved["source_manifest"].is_file():
        raise FileNotFoundError("frozen LATTE source or environment manifest is missing")
    if resolved["result"].exists():
        raise FileExistsError("CUDA compatibility result already exists")
    prior_status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if not prior_status_path.is_file():
        raise FileNotFoundError("prior terminal status is required")
    prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
    if (
        prior_status["attempt_id"] != PRIOR_ATTEMPT_ID
        or prior_status["scientific_state"] != "BLOCKED"
    ):
        raise FileExistsError("prior attempt must be terminal BLOCKED before recovery")
    python = resolved["env"] / "bin/python"
    if not python.is_file():
        raise FileNotFoundError("prior empty recovery environment is missing")
    torch_probe = subprocess.run(
        [
            str(python),
            "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, "
            "'torch_cuda': torch.version.cuda}, sort_keys=True))",
        ],
        capture_output=True,
        text=True,
        env=controlled_environment(root, gpu_id=None),
        timeout=30,
        check=True,
    )
    installed_torch = json.loads(torch_probe.stdout.strip())
    if installed_torch != {"torch": TORCH_VERSION, "torch_cuda": TORCH_CUDA_VERSION}:
        raise RuntimeError(f"reusable environment has unexpected torch build: {installed_torch}")

    resolved["result"].mkdir(parents=True, exist_ok=False)
    source_text = resolved["source_freeze"].read_text(encoding="utf-8")
    requirements = compatible_requirements(source_text)
    resolved["requirements"].write_text("\n".join(requirements) + "\n", encoding="utf-8")
    config = {
        "schema_version": "phase17.s17_fp0_cuda_compat_env_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "official_source_commit": LATTE_COMMIT,
        "source_environment_manifest_sha256": sha256(resolved["source_manifest"]),
        "source_requirements_sha256": sha256(resolved["source_freeze"]),
        "compatible_requirements_sha256": sha256(resolved["requirements"]),
        "compatible_requirements_count": len(requirements),
        "torch_version": TORCH_VERSION,
        "torch_cuda_version": TORCH_CUDA_VERSION,
        "torch_wheel_url": TORCH_WHEEL_URL,
        "torch_wheel_sha256": TORCH_WHEEL_SHA256,
        "torch_index_url": TORCH_INDEX_URL,
        "python": "3.12.12",
        "environment_path": str(resolved["env"].relative_to(root)),
        "environment_reused": True,
        "reused_ready_environment": True,
        "network_install": False,
        "download_skipped": True,
        "prior_attempt_id": PRIOR_ATTEMPT_ID,
        "background_required": True,
        "gpu1_shared_authorized": True,
        "gpu1_shared_authorization": AUTHORIZATION,
        "gpu1_repeat_must_remain": True,
        "gpu1_minimum_free_mib": CUDA_SMOKE_MIN_FREE_MIB,
        "profile_minimum_free_mib": PROFILE_ADMISSION_MIB,
        "profile_attempt_after_pass": "attempt_009",
        "download_mirror": "aliyun_pytorch_wheels_cu126_direct",
        "uv_http_timeout_seconds": 300,
        "sentence_t5_redownload": False,
        "automatic_retry": False,
        "effect_experiment_started": False,
        "full_data_tokenizer_started": False,
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
            "kind": "cuda_compatible_environment_and_gpu1_smoke",
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
        track_id="FP0-INFRASTRUCTURE",
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": 5, "unit": "environment_gate"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "environment_path": str(resolved["env"].relative_to(root)),
            "environment_reused": True,
            "reused_ready_environment": True,
            "network_install": False,
            "download_skipped": True,
            "prior_attempt_id": PRIOR_ATTEMPT_ID,
            "prior_failure_path": "artifacts/phase17/fullport/fp0/cuda_compat_env/attempt_003/run.log",
            "download_mirror": "aliyun_pytorch_wheels_cu126_direct",
            "uv_http_timeout_seconds": 300,
            "torch_version": TORCH_VERSION,
            "torch_cuda_version": TORCH_CUDA_VERSION,
            "torch_wheel_sha256": TORCH_WHEEL_SHA256,
            "gpu_ids": [],
            "gpu1_handoff_used": False,
            "gpu1_repeat_restored": None,
            "gpu1_repeat_preserved": None,
            "gpu1_shared_authorized": True,
            "gpu1_shared_authorization": AUTHORIZATION,
            "sentence_t5_redownload": False,
            "automatic_retry": False,
            "effect_experiment_started": False,
            "full_data_tokenizer_started": False,
            "affects_scientific_result": False,
            "result_selection_eligible": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP0_CUDA_COMPAT_ENV_READY_TO_LAUNCH",
        process_alive=False,
    )
    print(manifest)
    return 0


def launch(root: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"CUDA compatibility environment is not launchable: {status['scientific_state']}")
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
        "S17_FP0_CUDA_COMPAT_ENV_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="validate_reused_environment",
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_CUDA_COMPAT_ENV_STARTUP_HANDSHAKE_FAILED",
                process_alive=False,
                workload_pid=0,
                stage="startup_handshake_failed_no_retry",
                automatic_retry=False,
            )
        raise RuntimeError("CUDA compatibility worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, snapshot_manifest: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    preexisting_gpu1_processes: list[dict[str, Any]] = []
    try:
        verify_run_snapshot(root, snapshot_manifest)
        frozen_config = json.loads(
            snapshot_manifest.parent.joinpath("config.json").read_text(encoding="utf-8")
        )
        if sha256(resolved["requirements"]) != frozen_config["compatible_requirements_sha256"]:
            raise RuntimeError("CUDA compatibility requirements changed after preflight")
        resolved["uv_cache"].mkdir(parents=True, exist_ok=True)

        python = resolved["env"] / "bin/python"
        if not python.is_file():
            raise FileNotFoundError("CUDA compatibility environment Python is missing")

        run_command(
            command=[str(UV), "pip", "check", "--python", str(python)],
            cwd=root,
            env=controlled_environment(root, gpu_id=None),
            resolved=resolved,
            writer=writer,
            stage="validate_dependency_graph",
            status_code="S17_FP0_CUDA_COMPAT_ENV_CHECKING_DEPENDENCIES",
            timeout_seconds=300,
        )
        run_command(
            command=[str(python), "-c", cpu_validation_script()],
            cwd=root,
            env=controlled_environment(root, gpu_id=None),
            resolved=resolved,
            writer=writer,
            stage="validate_cpu_imports_and_pins",
            status_code="S17_FP0_CUDA_COMPAT_ENV_VALIDATING_IMPORTS",
            timeout_seconds=300,
        )
        cpu_validation = read_last_json(resolved["log"])
        if cpu_validation["torch"] != TORCH_VERSION:
            raise RuntimeError(f"unexpected torch version: {cpu_validation['torch']}")
        if cpu_validation["torch_cuda"] != TORCH_CUDA_VERSION:
            raise RuntimeError(f"unexpected torch CUDA version: {cpu_validation['torch_cuda']}")
        if cpu_validation["cuda_available"]:
            raise RuntimeError("CPU-only import validation unexpectedly exposed a GPU")

        freeze_completed = subprocess.run(
            [str(UV), "pip", "freeze", "--python", str(python)],
            cwd=root,
            env=controlled_environment(root, gpu_id=None),
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        resolved["freeze"].write_text(freeze_completed.stdout, encoding="utf-8")

        writer.heartbeat(
            stage="gpu1_read_only_admission",
            progress={"current": 4, "total": 5, "unit": "environment_gate"},
        )
        gpu_records, compute_processes = query_gpu_state_with_retries()
        selected, admission_code = select_authorized_gpu1(gpu_records, compute_processes)
        gpu_snapshot = {
            "captured_at": utc_now(),
            "devices": snapshot(gpu_records),
            "compute_processes": compute_processes,
            "selection_rule": "gpu1_existing_compute_pid_free_gte_1024_mib",
            "shared_gpu1_authorization": AUTHORIZATION,
        }
        if selected is None:
            writer.transition(
                "BLOCKED",
                "BLOCKED",
                admission_code,
                process_alive=False,
                workload_pid=0,
                stage="gpu1_cuda_smoke_admission_blocked",
                gpu_ids=[],
                gpu_snapshot=gpu_snapshot,
                gpu1_repeat_preserved=None,
                automatic_retry=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
            return 0
        preexisting_gpu1_processes = [dict(row) for row in compute_processes[GPU1_ID]]
        preexisting_pids = sorted(row["pid"] for row in preexisting_gpu1_processes)
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP0_CUDA_COMPAT_ENV_GPU1_SMOKE_ADMITTED",
            process_alive=True,
            workload_pid=os.getpid(),
            stage="gpu1_minimal_cuda_smoke",
            gpu_ids=[GPU1_ID],
            gpu_snapshot=gpu_snapshot,
            gpu1_preexisting_processes=preexisting_gpu1_processes,
            gpu1_preexisting_pids=preexisting_pids,
            gpu1_repeat_preserved=None,
        )
        run_command(
            command=[str(python), "-c", cuda_smoke_script()],
            cwd=root,
            env=controlled_environment(root, gpu_id=GPU1_ID),
            resolved=resolved,
            writer=writer,
            stage="gpu1_minimal_cuda_smoke",
            status_code="S17_FP0_CUDA_COMPAT_ENV_GPU1_SMOKE_RUNNING",
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
        )
        smoke = read_last_json(resolved["log"])
        if smoke["torch"] != TORCH_VERSION or smoke["torch_cuda"] != TORCH_CUDA_VERSION:
            raise RuntimeError(f"CUDA smoke used unexpected torch build: {smoke}")
        if not smoke["cuda_available"] or not smoke["finite"]:
            raise RuntimeError(f"CUDA smoke failed validity checks: {smoke}")

        post_records, post_processes = query_gpu_state_with_retries()
        post_pids = sorted(row["pid"] for row in post_processes.get(GPU1_ID, []))
        missing_pids = sorted(set(preexisting_pids) - set(post_pids))
        if missing_pids:
            raise RuntimeError(f"pre-existing GPU1 processes disappeared during smoke: {missing_pids}")
        post_snapshot = {
            "captured_at": utc_now(),
            "devices": snapshot(post_records),
            "compute_processes": post_processes,
        }
        environment_manifest = {
            "schema_version": "phase17.s17_fp0_cuda_compat_env_manifest.v1",
            "generated_at": utc_now(),
            "environment_path": str(resolved["env"].relative_to(root)),
            "official_source_commit": LATTE_COMMIT,
            "torch_wheel_url": TORCH_WHEEL_URL,
            "torch_wheel_sha256": TORCH_WHEEL_SHA256,
            "download_mirror": "aliyun_pytorch_wheels_cu126_direct",
            "environment_reused": True,
            "network_install_performed": False,
            "download_skipped": True,
            "uv_http_timeout_seconds": 300,
            "requirements_freeze_path": str(resolved["freeze"].relative_to(root)),
            "requirements_freeze_sha256": sha256(resolved["freeze"]),
            "cpu_validation": cpu_validation,
            "cuda_smoke": smoke,
            "physical_gpu": GPU1_ID,
            "gpu1_preexisting_processes": preexisting_gpu1_processes,
            "gpu1_post_smoke_snapshot": post_snapshot,
            "gpu1_repeat_preserved": True,
            "sentence_t5_redownload": False,
            "automatic_retry": False,
            "effect_experiment_started": False,
            "full_data_tokenizer_started": False,
        }
        atomic_json(resolved["environment_manifest"], environment_manifest)
        summary = {
            "schema_version": "phase17.s17_fp0_cuda_compat_env_summary.v1",
            "verdict": "PASS_S17_FP0_CUDA_COMPAT_ENV_READY",
            "completed_at": utc_now(),
            "environment_manifest_path": str(resolved["environment_manifest"].relative_to(root)),
            "environment_manifest_sha256": sha256(resolved["environment_manifest"]),
            "torch": TORCH_VERSION,
            "torch_cuda": TORCH_CUDA_VERSION,
            "download_mirror": "aliyun_pytorch_wheels_cu126_direct",
            "environment_reused": True,
            "network_install_performed": False,
            "download_skipped": True,
            "cuda_smoke": smoke,
            "gpu1_repeat_preserved": True,
            "sentence_t5_redownload": False,
            "effect_experiment_started": False,
            "full_data_tokenizer_started": False,
            "next_gate": "S17_FP0_TOKENIZER_PROFILE_ATTEMPT_009",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP0_CUDA_COMPAT_ENV_READY",
            process_alive=False,
            workload_pid=0,
            stage="cuda_compatible_environment_ready",
            progress={"current": 5, "total": 5, "unit": "environment_gate"},
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            environment_manifest_path=str(resolved["environment_manifest"].relative_to(root)),
            environment_manifest_sha256=sha256(resolved["environment_manifest"]),
            gpu_ids=[],
            gpu1_preexisting_pids=preexisting_pids,
            gpu1_post_smoke_snapshot=post_snapshot,
            gpu1_repeat_preserved=True,
            result_selection_eligible=False,
            affects_scientific_result=False,
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
            if preexisting_gpu1_processes:
                try:
                    _, post_processes = query_gpu_state_with_retries()
                    preexisting_pids = sorted(row["pid"] for row in preexisting_gpu1_processes)
                    post_pids = sorted(row["pid"] for row in post_processes.get(GPU1_ID, []))
                    preservation_fields = {
                        "gpu1_preexisting_pids": preexisting_pids,
                        "gpu1_post_failure_compute_processes": post_processes,
                        "gpu1_repeat_preserved": set(preexisting_pids).issubset(post_pids),
                    }
                except BaseException as preservation_error:
                    preservation_fields = {
                        "gpu1_repeat_preserved": None,
                        "gpu1_preservation_check_error": repr(preservation_error),
                    }
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_CUDA_COMPAT_ENV_FAILED",
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
