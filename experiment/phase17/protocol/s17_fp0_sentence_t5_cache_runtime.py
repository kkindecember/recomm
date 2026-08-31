#!/usr/bin/env python3
"""Freeze the official SentenceT5 model snapshot for LATTE, CPU-only in tmux."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
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
LAUNCH_PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
EXPERIMENT_ID = "s17_fp0_sentence_t5_cache"
ATTEMPT_ID = "attempt_005"
PRIOR_ATTEMPT_ID = "attempt_004"
STEP_ID = "S17-FP0-SENTENCE-T5-CACHE"
TMUX_SESSION = EXPERIMENT_ID
DEPENDENCY_EXPERIMENT_ID = "s17_fp0_native_env_setup"
DEPENDENCY_PASS_CODE = "PASS_S17_FP0_NATIVE_ENV_READY"
MODEL_ID = "sentence-transformers/sentence-t5-base"
MODEL_REVISION = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
EXPECTED_EMBEDDING_DIM = 768
WAIT_TIMEOUT_SECONDS = 10800
DOWNLOAD_TIMEOUT_SECONDS = 7200
HEARTBEAT_SECONDS = 60
CURL = Path("/usr/bin/curl")
TMUX_NETWORK_PROXY = "http://127.0.0.1:7899"
TRANSFER_ATTEMPTS = 5
TRANSFER_RETRY_DELAY_SECONDS = 5
TRANSFER_MAX_SECONDS = 3600
ALLOW_PATTERNS = (
    "README.md",
    "modules.json",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
)


def paths(root: Path) -> dict[str, Path]:
    result = root / f"artifacts/phase17/fullport/fp0/sentence_t5_cache/{ATTEMPT_ID}"
    model_root = root / "artifacts/phase17/fullport/models"
    return {
        "result": result,
        "config": result / "config.json",
        "summary": result / "summary.json",
        "manifest": result / "model_manifest.json",
        "log": result / "run.log",
        "model_root": model_root,
        "model": model_root / f"sentence-t5-base_{MODEL_REVISION}",
        "staging": model_root / f"sentence-t5-base_{MODEL_REVISION}.downloading.{ATTEMPT_ID}",
        "hf_cache": root / "artifacts/phase17/fullport/cache/huggingface",
        "native_env": root / "artifacts/phase17/fullport/envs/latte_05e4e6d98322",
        "dependency_status": root
        / f"artifacts/phase17/status/{DEPENDENCY_EXPERIMENT_ID}.status.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP0-SENTENCE-T5-CACHE.attempts.jsonl",
        "snapshot": root / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json",
        "snapshot_worker": root
        / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/src/000_s17_fp0_sentence_t5_cache_runtime.py",
    }


def controlled_environment(resolved: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(resolved["hf_cache"]),
            "HF_HUB_CACHE": str(resolved["hf_cache"] / "hub"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def worker_command(root: Path, resolved: dict[str, Path]) -> list[str]:
    return [
        "/usr/bin/env",
        f"PYTHONPATH={root}",
        f"http_proxy={TMUX_NETWORK_PROXY}",
        f"https_proxy={TMUX_NETWORK_PROXY}",
        str(LAUNCH_PYTHON),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def download_script() -> str:
    return (
        "import json, pathlib, subprocess, sys, time, urllib.parse; "
        "curl, repo, revision, local_dir = sys.argv[1:5]; "
        "files=json.loads(sys.argv[5]); attempts=int(sys.argv[6]); delay=int(sys.argv[7]); "
        "max_seconds=int(sys.argv[8]); root=pathlib.Path(local_dir); "
        "root.mkdir(parents=True, exist_ok=True); "
        "\nfor index, name in enumerate(files, 1):\n"
        " target=root / name; target.parent.mkdir(parents=True, exist_ok=True)\n"
        " encoded=urllib.parse.quote(name, safe='/')\n"
        " url=f'https://huggingface.co/{repo}/resolve/{revision}/{encoded}?download=true'\n"
        " succeeded=False\n"
        " for attempt in range(1, attempts + 1):\n"
        "  print(json.dumps({'event':'transfer_start','file':name,'index':index,'total':len(files),'attempt':attempt,'resume_bytes':target.stat().st_size if target.exists() else 0}), flush=True)\n"
        "  command=[curl,'--fail','--location','--silent','--show-error','--connect-timeout','30','--max-time',str(max_seconds),'--continue-at','-','--output',str(target),url]\n"
        "  result=subprocess.run(command, check=False)\n"
        "  if result.returncode == 0:\n"
        "   succeeded=True; print(json.dumps({'event':'transfer_complete','file':name,'size_bytes':target.stat().st_size}), flush=True); break\n"
        "  print(json.dumps({'event':'transfer_retry','file':name,'attempt':attempt,'return_code':result.returncode}), flush=True)\n"
        "  if attempt < attempts: time.sleep(delay)\n"
        " if not succeeded: raise SystemExit(result.returncode or 1)\n"
    )


def validation_script() -> str:
    return (
        "import json, numpy as np, sys; "
        "from sentence_transformers import SentenceTransformer; "
        "model=SentenceTransformer(sys.argv[1], local_files_only=True, device='cpu'); "
        "values=model.encode(['stage seventeen adapter check', 'latent semantic identifier'], "
        "convert_to_numpy=True, show_progress_bar=False); "
        "print(json.dumps({'shape': list(values.shape), 'finite': bool(np.isfinite(values).all()), "
        "'dtype': str(values.dtype)}))"
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


def run_command_with_heartbeat(
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
            "S17_FP0_SENTENCE_T5_DOWNLOADING",
            stage=stage,
            workload_pid=process.pid,
            process_alive=True,
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
                    raise TimeoutError(f"SentenceT5 download exceeded {timeout_seconds}s")
    return return_code, time.monotonic() - started


def wait_for_native_environment(writer: StatusWriter, resolved: dict[str, Path]) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        if not resolved["dependency_status"].is_file():
            raise FileNotFoundError("native environment status is missing")
        dependency = json.loads(resolved["dependency_status"].read_text(encoding="utf-8"))
        if dependency["scientific_state"] == "COMPLETED":
            if dependency["status_code"] != DEPENDENCY_PASS_CODE:
                raise RuntimeError(f"native environment completed without PASS: {dependency['status_code']}")
            python = resolved["native_env"] / "bin/python"
            if not python.is_file():
                raise FileNotFoundError(f"native environment Python is missing: {python}")
            return dependency
        if dependency["scientific_state"] in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(
                f"native environment dependency is terminal: {dependency['scientific_state']} "
                f"{dependency['status_code']}"
            )
        elapsed = time.monotonic() - started
        if elapsed > WAIT_TIMEOUT_SECONDS:
            raise TimeoutError("timed out waiting for the native LATTE environment")
        writer.heartbeat(
            stage="waiting_for_native_environment",
            progress={
                "current": min(int(elapsed), WAIT_TIMEOUT_SECONDS),
                "total": WAIT_TIMEOUT_SECONDS,
                "unit": "seconds_until_dependency_timeout",
            },
        )
        time.sleep(HEARTBEAT_SECONDS)


def file_inventory(directory: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative.startswith(".cache/"):
            continue
        inventory.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    return inventory


def prepare(root: Path) -> int:
    resolved = paths(root)
    if not LAUNCH_PYTHON.is_file():
        raise FileNotFoundError(LAUNCH_PYTHON)
    if not CURL.is_file():
        raise FileNotFoundError(CURL)
    prior_status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if prior_status_path.exists():
        prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
        if prior_status["scientific_state"] not in {"FAILED", "STOPPED", "BLOCKED"}:
            raise FileExistsError("prior SentenceT5 cache attempt is not terminal")
        if prior_status["attempt_id"] != PRIOR_ATTEMPT_ID:
            raise FileExistsError("unexpected prior SentenceT5 cache attempt id")
    if not resolved["dependency_status"].is_file():
        raise FileNotFoundError("launch the native environment task first")
    resolved["result"].mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "phase17.s17_fp0_sentence_t5_cache_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": "apache-2.0",
        "model_path": str(resolved["model"].relative_to(root)),
        "allow_patterns": list(ALLOW_PATTERNS),
        "download_transport": "curl_fixed_revision_resume",
        "curl_path": str(CURL),
        "tmux_proxy_endpoint": TMUX_NETWORK_PROXY,
        "tmux_proxy_has_credentials": False,
        "transfer_attempts": TRANSFER_ATTEMPTS,
        "transfer_retry_delay_seconds": TRANSFER_RETRY_DELAY_SECONDS,
        "transfer_max_seconds": TRANSFER_MAX_SECONDS,
        "expected_embedding_dim": EXPECTED_EMBEDDING_DIM,
        "dependency_experiment_id": DEPENDENCY_EXPERIMENT_ID,
        "dependency_pass_code": DEPENDENCY_PASS_CODE,
        "wait_timeout_seconds": WAIT_TIMEOUT_SECONDS,
        "download_timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
        "background_required": True,
        "gpu_ids": [],
        "gpu1_allowed": False,
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
            "kind": "cpu_model_snapshot_freeze",
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
            "progress": {"current": 0, "total": 3, "unit": "cache_gate"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "dependency_experiment_id": DEPENDENCY_EXPERIMENT_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "gpu_ids": [],
            "gpu1_handoff_used": False,
            "gpu1_repeat_restored": None,
            "automatic_retry": False,
            "effect_experiment_started": False,
            "affects_scientific_result": False,
            "result_selection_eligible": False,
            "d1_read": False,
            "d2_read": False,
            "prior_attempt_id": PRIOR_ATTEMPT_ID,
            "prior_failure_path": "artifacts/phase17/fullport/fp0/sentence_t5_cache/attempt_004/run.log",
            "download_transport": "curl_fixed_revision_resume",
            "transfer_attempts": TRANSFER_ATTEMPTS,
            "tmux_proxy_endpoint": TMUX_NETWORK_PROXY,
            "tmux_proxy_has_credentials": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP0_SENTENCE_T5_CACHE_READY_TO_LAUNCH",
        process_alive=False,
    )
    print(manifest)
    return 0


def launch(root: Path) -> int:
    resolved = paths(root)
    status = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"SentenceT5 cache is not launchable: {status['scientific_state']}")
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
        "S17_FP0_SENTENCE_T5_CACHE_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="waiting_for_native_environment",
    )
    if not wait_for_tmux_startup(session):
        latest = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
        if latest["scientific_state"] == "RUNNING":
            StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP0_SENTENCE_T5_STARTUP_HANDSHAKE_FAILED",
                process_alive=False,
                workload_pid=0,
                stage="startup_handshake_failed_no_retry",
                automatic_retry=False,
            )
        raise RuntimeError("SentenceT5 cache worker exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, snapshot_manifest: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    try:
        verify_run_snapshot(root, snapshot_manifest)
        dependency = wait_for_native_environment(writer, resolved)
        writer.heartbeat(
            stage="native_environment_ready",
            progress={"current": 1, "total": 3, "unit": "cache_gate"},
        )
        if resolved["model"].exists() or resolved["staging"].exists():
            raise FileExistsError("SentenceT5 model/staging path already exists; no overwrite")
        resolved["model_root"].mkdir(parents=True, exist_ok=True)
        resolved["hf_cache"].mkdir(parents=True, exist_ok=True)
        resolved["staging"].mkdir(parents=False, exist_ok=False)
        native_python = resolved["native_env"] / "bin/python"
        command = [
            str(native_python),
            "-c",
            download_script(),
            str(CURL),
            MODEL_ID,
            MODEL_REVISION,
            str(resolved["staging"]),
            json.dumps(list(ALLOW_PATTERNS)),
            str(TRANSFER_ATTEMPTS),
            str(TRANSFER_RETRY_DELAY_SECONDS),
            str(TRANSFER_MAX_SECONDS),
        ]
        return_code, wall_seconds = run_command_with_heartbeat(
            command=command,
            cwd=root,
            env=controlled_environment(resolved),
            log_path=resolved["log"],
            writer=writer,
            stage="download_exact_sentence_t5_revision",
            timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
        )
        if return_code != 0:
            raise RuntimeError(f"SentenceT5 snapshot download exited with code {return_code}")
        resolved["staging"].rename(resolved["model"])
        writer.heartbeat(
            stage="offline_sentence_t5_validation",
            progress={"current": 2, "total": 3, "unit": "cache_gate"},
        )
        validation = subprocess.run(
            [str(native_python), "-c", validation_script(), str(resolved["model"])],
            cwd=root,
            env={
                **controlled_environment(resolved),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if validation.returncode != 0:
            raise RuntimeError(f"offline SentenceT5 validation failed: {validation.stderr[-4000:]}")
        validation_lines = [line for line in validation.stdout.splitlines() if line.strip()]
        validation_result = json.loads(validation_lines[-1])
        if validation_result["shape"] != [2, EXPECTED_EMBEDDING_DIM] or not validation_result["finite"]:
            raise RuntimeError(f"invalid SentenceT5 validation output: {validation_result}")
        inventory = file_inventory(resolved["model"])
        required = set(ALLOW_PATTERNS)
        present = {row["path"] for row in inventory}
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(f"SentenceT5 snapshot is missing allowed required files: {missing}")
        model_manifest = {
            "schema_version": "phase17.s17_fp0_sentence_t5_model_manifest.v1",
            "generated_at": utc_now(),
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "apache-2.0",
            "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "download_transport": "curl_fixed_revision_resume",
            "transfer_attempts": TRANSFER_ATTEMPTS,
            "local_path": str(resolved["model"].relative_to(root)),
            "files": inventory,
            "total_size_bytes": sum(row["size_bytes"] for row in inventory),
            "offline_validation": validation_result,
            "gpu_used": False,
            "gpu_ids": [],
            "effect_experiment_started": False,
            "automatic_retry": False,
        }
        atomic_json(resolved["manifest"], model_manifest)
        summary = {
            "schema_version": "phase17.s17_fp0_sentence_t5_cache_summary.v1",
            "verdict": "PASS_S17_FP0_SENTENCE_T5_CACHE_READY",
            "completed_at": utc_now(),
            "download_wall_seconds": wall_seconds,
            "dependency_status_sha256": sha256(resolved["dependency_status"]),
            "dependency_status_code": dependency["status_code"],
            "model_manifest_path": str(resolved["manifest"].relative_to(root)),
            "model_manifest_sha256": sha256(resolved["manifest"]),
            "offline_validation": validation_result,
            "gpu_used": False,
            "gpu_ids": [],
            "effect_experiment_started": False,
            "automatic_retry": False,
            "next_gate": "S17-FP0-TOKENIZER-BOUNDED-PROFILE",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP0_SENTENCE_T5_CACHE_READY",
            process_alive=False,
            workload_pid=0,
            stage="sentence_t5_cache_ready",
            progress={"current": 3, "total": 3, "unit": "cache_gate"},
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            model_manifest_path=str(resolved["manifest"].relative_to(root)),
            model_manifest_sha256=sha256(resolved["manifest"]),
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
                "S17_FP0_SENTENCE_T5_CACHE_FAILED",
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
