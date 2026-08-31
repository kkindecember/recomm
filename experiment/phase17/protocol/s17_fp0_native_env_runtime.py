#!/usr/bin/env python3
"""Prepare the pinned official LATTE Python environment without using a GPU.

This is infrastructure preparation, not an efficacy experiment.  The unknown-duration
dependency installation is always launched in tmux and reports progress through the
Stage17 atomic status file.  A failed attempt is terminal and is never retried here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    verify_run_snapshot,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import AttemptLedger, StatusWriter, atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
UV = Path("/home/jiangtangyunzhi/miniconda3/bin/uv")
EXPERIMENT_ID = "s17_fp0_native_env_setup"
ATTEMPT_ID = "attempt_003"
PRIOR_ATTEMPT_ID = "attempt_002"
STEP_ID = "S17-FP0-NATIVE-ENV"
TMUX_SESSION = EXPERIMENT_ID
LATTE_COMMIT = "05e4e6d983225bcb7172f148a076890e80c524d1"
LATTE_ARCHIVE_SHA256 = "43ead8c1dd7dacf8a06c4bc4b6bce7b7f7645451f3733140f4aada05cf68f242"
BOOTSTRAP_PYTHON = Path(
    "/home/jiangtangyunzhi/.local/share/uv/python/"
    "cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
)
BOOTSTRAP_PYTHON_SHA256 = "956a763aa0a77c239c1925c16f3922f292a8271b85ed44633c590f7aaece029a"
DEFAULT_INPUT_ARCHIVE = Path("/tmp/s17_fp0_latte_main.tar.gz")
HARD_TIMEOUT_SECONDS = 7200
HEARTBEAT_SECONDS = 60


def paths(root: Path) -> dict[str, Path]:
    source_base = root / "artifacts/phase17/fullport/sources"
    result = root / f"artifacts/phase17/fullport/fp0/native_env_setup/{ATTEMPT_ID}"
    return {
        "source_base": source_base,
        "archive": source_base / f"latte_{LATTE_COMMIT}.tar.gz",
        "source": source_base / f"latte_{LATTE_COMMIT}_{ATTEMPT_ID}",
        "env": root / "artifacts/phase17/fullport/envs" / f"latte_{LATTE_COMMIT[:12]}",
        "uv_python": root / "artifacts/phase17/fullport/envs/uv_managed_python",
        "uv_cache": root / "artifacts/phase17/fullport/cache/uv",
        "result": result,
        "config": result / "config.json",
        "summary": result / "summary.json",
        "log": result / "run.log",
        "freeze": result / "requirements.freeze.txt",
        "environment_manifest": result / "environment_manifest.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP0-NATIVE-ENV.attempts.jsonl",
        "snapshot": root / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "snapshot_worker": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/000_s17_fp0_native_env_runtime.py",
        "source_manifest": root / "artifacts/phase17/fullport/manifests/latte_source_manifest.json",
    }


def build_uv_command(resolved: dict[str, Path]) -> list[str]:
    return [
        str(UV),
        "sync",
        "--project",
        str(resolved["source"]),
        "--python",
        str(BOOTSTRAP_PYTHON),
        "--managed-python",
        "--no-dev",
    ]


def worker_command(root: Path, resolved: dict[str, Path]) -> list[str]:
    return [
        "/usr/bin/env",
        f"PYTHONPATH={root}",
        str(PYTHON),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def controlled_environment(resolved: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "UV_PROJECT_ENVIRONMENT": str(resolved["env"]),
            "UV_PYTHON_INSTALL_DIR": str(resolved["uv_python"]),
            "UV_CACHE_DIR": str(resolved["uv_cache"]),
            "UV_LINK_MODE": "copy",
            "UV_NO_PROGRESS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def archive_member_is_safe(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if member.issym() or member.islnk() or member.isdev():
        return False
    return member.isdir() or member.isfile()


def extract_official_source(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"official source destination already exists: {destination}")
    staging = destination.with_name(f"{destination.name}.extracting.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            if not members or any(not archive_member_is_safe(member) for member in members):
                raise PermissionError("LATTE archive contains an unsafe member")
            roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
            if len(roots) != 1:
                raise RuntimeError(f"LATTE archive must have exactly one root: {sorted(roots)}")
            handle.extractall(staging, members=members)
        extracted_root = staging / next(iter(roots))
        required = ("LICENSE", "README.md", "pyproject.toml", "genrec/default.yaml")
        missing = [name for name in required if not (extracted_root / name).is_file()]
        if missing:
            raise RuntimeError(f"LATTE archive is missing required files: {missing}")
        extracted_root.rename(destination)
        staging.rmdir()
    except BaseException:
        # Preserve a failed extraction for forensic inspection; never overwrite or retry it.
        raise


def terminate_exact_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_with_heartbeat(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    writer: StatusWriter,
    stage: str,
    timeout_seconds: int,
) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
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
            "S17_FP0_NATIVE_ENV_INSTALLING",
            workload_pid=process.pid,
            process_alive=True,
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
                    raise TimeoutError(f"command exceeded hard timeout of {timeout_seconds}s")
    return return_code, time.monotonic() - started


def validation_script() -> str:
    return (
        "import json, platform; "
        "import torch, transformers, sentence_transformers, faiss, numpy, sklearn; "
        "print(json.dumps({"
        "'python': platform.python_version(), "
        "'torch': torch.__version__, "
        "'transformers': transformers.__version__, "
        "'sentence_transformers': sentence_transformers.__version__, "
        "'faiss': getattr(faiss, '__version__', 'unknown'), "
        "'numpy': numpy.__version__, "
        "'sklearn': sklearn.__version__, "
        "'cuda_visible': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'), "
        "'cuda_available': torch.cuda.is_available()} , sort_keys=True))"
    )


def validate_environment(root: Path, resolved: dict[str, Path], env: dict[str, str]) -> dict[str, Any]:
    python = resolved["env"] / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"native environment Python is missing: {python}")
    completed = subprocess.run(
        [str(python), "-c", validation_script()],
        cwd=resolved["source"],
        env={**env, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"native import validation failed: {completed.stderr[-4000:]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    versions = json.loads(lines[-1])
    if tuple(int(value) for value in versions["python"].split(".")[:2]) < (3, 12):
        raise RuntimeError(f"LATTE requires Python >=3.12, got {versions['python']}")
    if versions["cuda_visible"] != "" or versions["cuda_available"]:
        raise RuntimeError("native environment preparation unexpectedly exposed a GPU")

    freeze_completed = subprocess.run(
        [str(UV), "pip", "freeze", "--python", str(python)],
        cwd=resolved["source"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if freeze_completed.returncode != 0:
        raise RuntimeError(f"uv pip freeze failed: {freeze_completed.stderr[-4000:]}")
    resolved["freeze"].write_text(freeze_completed.stdout, encoding="utf-8")
    lock_path = resolved["source"] / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("uv sync did not create uv.lock")
    source_manifest = json.loads(resolved["source_manifest"].read_text(encoding="utf-8"))
    expected_license = source_manifest["license_sha256"]
    actual_license = sha256(resolved["source"] / "LICENSE")
    if actual_license != expected_license:
        raise RuntimeError("LATTE MIT license hash changed after extraction")
    manifest = {
        "schema_version": "phase17.s17_fp0_native_env.v1",
        "generated_at": utc_now(),
        "official_source_commit": LATTE_COMMIT,
        "official_archive_sha256": sha256(resolved["archive"]),
        "official_license_sha256": actual_license,
        "source_path": str(resolved["source"].relative_to(root)),
        "environment_path": str(resolved["env"].relative_to(root)),
        "uv_lock_path": str(lock_path.relative_to(root)),
        "uv_lock_sha256": sha256(lock_path),
        "requirements_freeze_path": str(resolved["freeze"].relative_to(root)),
        "requirements_freeze_sha256": sha256(resolved["freeze"]),
        "versions": versions,
        "gpu_used": False,
        "gpu_ids": [],
        "model_weights_downloaded": False,
        "effect_experiment_started": False,
        "automatic_retry": False,
    }
    atomic_json(resolved["environment_manifest"], manifest)
    return manifest


def prepare(root: Path, input_archive: Path) -> int:
    resolved = paths(root)
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not UV.is_file():
        raise FileNotFoundError(UV)
    if not BOOTSTRAP_PYTHON.is_file():
        raise FileNotFoundError(BOOTSTRAP_PYTHON)
    if sha256(BOOTSTRAP_PYTHON) != BOOTSTRAP_PYTHON_SHA256:
        raise RuntimeError("local uv-managed Python 3.12.12 hash mismatch")
    prior_status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if prior_status_path.exists():
        prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
        if prior_status["scientific_state"] not in {"FAILED", "STOPPED", "BLOCKED"}:
            raise FileExistsError("prior native environment attempt is not terminal")
        if prior_status["attempt_id"] != PRIOR_ATTEMPT_ID:
            raise FileExistsError("unexpected prior native environment attempt id")
    if not input_archive.is_file():
        raise FileNotFoundError(input_archive)
    if sha256(input_archive) != LATTE_ARCHIVE_SHA256:
        raise RuntimeError("input LATTE archive hash does not match the frozen official source")
    source_manifest = json.loads(resolved["source_manifest"].read_text(encoding="utf-8"))
    if source_manifest["commit"] != LATTE_COMMIT:
        raise RuntimeError("frozen LATTE source manifest commit mismatch")
    if source_manifest["downloaded_archive_sha256"] != LATTE_ARCHIVE_SHA256:
        raise RuntimeError("frozen LATTE source manifest archive hash mismatch")

    resolved["result"].mkdir(parents=True, exist_ok=False)
    resolved["source_base"].mkdir(parents=True, exist_ok=True)
    if resolved["archive"].exists():
        if sha256(resolved["archive"]) != LATTE_ARCHIVE_SHA256:
            raise RuntimeError("persistent LATTE archive from prior attempt has drifted")
    else:
        shutil.copy2(input_archive, resolved["archive"])
        resolved["archive"].chmod(0o444)

    config = {
        "schema_version": "phase17.s17_fp0_native_env_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "official_source_commit": LATTE_COMMIT,
        "official_archive_sha256": LATTE_ARCHIVE_SHA256,
        "archive_path": str(resolved["archive"].relative_to(root)),
        "source_path": str(resolved["source"].relative_to(root)),
        "environment_path": str(resolved["env"].relative_to(root)),
        "uv_command": build_uv_command(resolved),
        "python_requirement": ">=3.12",
        "bootstrap_python_path": str(BOOTSTRAP_PYTHON),
        "bootstrap_python_sha256": BOOTSTRAP_PYTHON_SHA256,
        "bootstrap_python_reused": True,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "background_required": True,
        "gpu_ids": [],
        "gpu1_allowed": False,
        "cuda_visible_devices": "",
        "effect_experiment_started": False,
        "model_weights_downloaded": False,
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
            "kind": "cpu_native_environment_setup",
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
            "progress": {"current": 0, "total": 4, "unit": "setup_gate"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "official_source_commit": LATTE_COMMIT,
            "official_archive_sha256": LATTE_ARCHIVE_SHA256,
            "gpu_ids": [],
            "gpu1_handoff_used": False,
            "gpu1_repeat_restored": None,
            "automatic_retry": False,
            "effect_experiment_started": False,
            "model_weights_downloaded": False,
            "affects_scientific_result": False,
            "result_selection_eligible": False,
            "d1_read": False,
            "d2_read": False,
            "prior_attempt_id": PRIOR_ATTEMPT_ID,
            "prior_failure_path": "artifacts/phase17/fullport/fp0/native_env_setup/attempt_002/run.log",
            "bootstrap_python_path": str(BOOTSTRAP_PYTHON),
            "bootstrap_python_sha256": BOOTSTRAP_PYTHON_SHA256,
            "bootstrap_python_reused": True,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP0_NATIVE_ENV_READY_TO_LAUNCH",
        process_alive=False,
    )
    print(manifest)
    return 0


def launch(root: Path) -> int:
    resolved = paths(root)
    if not resolved["config"].is_file() or not resolved["snapshot"].is_file():
        raise FileNotFoundError("run prepare before launch")
    status = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"native environment setup is not launchable: {status['scientific_state']}")
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
        "S17_FP0_NATIVE_ENV_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="verify_archive",
        progress={"current": 0, "total": 4, "unit": "setup_gate"},
    )
    if not wait_for_tmux_startup(session):
        latest = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
        if latest["scientific_state"] == "RUNNING":
            StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_NATIVE_ENV_STARTUP_HANDSHAKE_FAILED",
                process_alive=False,
                workload_pid=0,
                stage="startup_handshake_failed_no_retry",
                automatic_retry=False,
            )
        raise RuntimeError("native environment worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, manifest_path: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    try:
        verify_run_snapshot(root, manifest_path)
        if not BOOTSTRAP_PYTHON.is_file():
            raise FileNotFoundError(BOOTSTRAP_PYTHON)
        if sha256(BOOTSTRAP_PYTHON) != BOOTSTRAP_PYTHON_SHA256:
            raise RuntimeError("local uv-managed Python changed after preflight")
        if sha256(resolved["archive"]) != LATTE_ARCHIVE_SHA256:
            raise RuntimeError("persistent LATTE archive hash mismatch")
        writer.heartbeat(stage="archive_verified", progress={"current": 1, "total": 4, "unit": "setup_gate"})

        extract_official_source(resolved["archive"], resolved["source"])
        writer.heartbeat(stage="official_source_extracted", progress={"current": 2, "total": 4, "unit": "setup_gate"})

        env = controlled_environment(resolved)
        resolved["env"].parent.mkdir(parents=True, exist_ok=True)
        resolved["uv_python"].mkdir(parents=True, exist_ok=True)
        resolved["uv_cache"].mkdir(parents=True, exist_ok=True)
        return_code, wall_seconds = run_with_heartbeat(
            command=build_uv_command(resolved),
            cwd=resolved["source"],
            env=env,
            log_path=resolved["log"],
            writer=writer,
            stage="uv_sync_official_dependencies",
            timeout_seconds=HARD_TIMEOUT_SECONDS,
        )
        if return_code != 0:
            raise RuntimeError(f"uv sync exited with code {return_code}")
        writer.heartbeat(stage="validate_native_imports", progress={"current": 3, "total": 4, "unit": "setup_gate"})
        environment_manifest = validate_environment(root, resolved, env)
        summary = {
            "schema_version": "phase17.s17_fp0_native_env_summary.v1",
            "verdict": "PASS_S17_FP0_NATIVE_ENV_READY",
            "completed_at": utc_now(),
            "wall_seconds_uv_sync": wall_seconds,
            "environment_manifest_path": str(resolved["environment_manifest"].relative_to(root)),
            "environment_manifest_sha256": sha256(resolved["environment_manifest"]),
            "versions": environment_manifest["versions"],
            "gpu_used": False,
            "gpu_ids": [],
            "effect_experiment_started": False,
            "model_weights_downloaded": False,
            "automatic_retry": False,
            "next_gate": "S17-FP0-TOKENIZER-MODEL-INTEGRATION",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP0_NATIVE_ENV_READY",
            process_alive=False,
            workload_pid=0,
            stage="native_environment_ready",
            progress={"current": 4, "total": 4, "unit": "setup_gate"},
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            environment_manifest_path=str(resolved["environment_manifest"].relative_to(root)),
            environment_manifest_sha256=sha256(resolved["environment_manifest"]),
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
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_NATIVE_ENV_SETUP_FAILED",
                process_alive=False,
                workload_pid=0,
                stage="terminal_failure_no_retry",
                terminal_error=repr(error),
                automatic_retry=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
                gpu_ids=[],
            )
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input-archive", type=Path, default=DEFAULT_INPUT_ARCHIVE)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root, args.input_archive.resolve())
    if args.action == "launch":
        return launch(root)
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
