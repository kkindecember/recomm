#!/usr/bin/env python3
"""Prepare and execute the bounded S17-1 contract/GPU smoke attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT_HINT = os.environ.get("S17_REPOSITORY_ROOT")
if ROOT_HINT and ROOT_HINT not in sys.path:
    sys.path.insert(0, ROOT_HINT)

from experiment.phase17.core.resource_profiler import (  # noqa: E402
    MAX_USABLE_MIB_PER_JOB,
    choose_idle_gpu,
    query_gpus,
    snapshot,
)
from experiment.phase17.core.run_manager import (  # noqa: E402
    freeze_run_snapshot,
    sha256,
    verify_run_snapshot,
)
from experiment.phase17.core.status_writer import (  # noqa: E402
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)


EXPERIMENT_ID = "s17_s1_public_framework"
ATTEMPT_ID = "attempt_001"
EXPECTED_PEAK_MIB = 21916
SAFETY_MARGIN_MIB = 4096
TIMEOUT_SECONDS = 600
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")


def paths(root: Path) -> dict[str, Path]:
    output = root / "artifacts/phase17/s1_contract" / ATTEMPT_ID
    return {
        "output": output,
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-1.attempts.jsonl",
        "cpu_log": output / "cpu_contract_tests.log",
        "gpu_log": output / "gpu_smoke.log",
        "summary": output / "summary.json",
        "config": output / "config.json",
        "snapshot": root / "artifacts/phase17/snapshots" / EXPERIMENT_ID / ATTEMPT_ID / "manifest.json",
    }


def canonical_command(root: Path, physical_gpu: int) -> list[str]:
    return [
        str(PYTHON),
        "../src/main_generative_gram.py",
        "--data_path", str(root / "artifacts/phase17/s0_audit/profile_data"),
        "--datasets", "Toys_s17_d0_100",
        "--distributed", "0", "--gpu", "0", "--seed", "2023", "--train", "1",
        "--resource_metrics", "1",
        "--log_dir", str(paths(root)["output"] / "gram_logs"),
        "--prediction_dir", str(paths(root)["output"] / "predictions"),
        "--item_prompt_max_len", "128", "--item_prompt", "all_text",
        "--cf_model", "sasrec", "--id_linking", "1", "--max_his", "20",
        "--rec_batch_size", "16", "--gradient_accumulation_steps", "8",
        "--rec_lr", "1e-3", "--rec_epochs", "1", "--test_epoch_rec", "0",
        "--save_rec_epochs", "1", "--save_predictions", "0", "--beam_size", "50",
        "--top_k_similar_item", "5", "--item_id_type", "split",
        "--hierarchical_id_type", "hierarchy_v1_c32_l5_len32768_split",
        "--debug_train_100", "0", "--debug_test_100", "0",
        "--cf0_arm", "A", "--cf0_phase9", "1", "--hi_gram_enabled", "0",
        "--s17_modules", "",
    ]


def source_paths(root: Path) -> list[Path]:
    return [
        root / "experiment/phase17/protocol/s1_contract_runtime.py",
        root / "experiment/phase17/run_stage17_s1_contract_smoke.sh",
        *sorted((root / "experiment/phase17/core").glob("*.py")),
        root / "experiment/phase17/registry/module_registry.py",
        root / "GRAM/src/model/gram.py",
        root / "GRAM/src/arguments.py",
        root / "GRAM/src/main_generative_gram.py",
    ]


def prepare(root: Path, requested_gpu: int) -> int:
    resolved = paths(root)
    if resolved["output"].exists() or resolved["snapshot"].exists():
        raise FileExistsError("attempt_001 already exists; no automatic retry is allowed")
    records = query_gpus()
    selected = choose_idle_gpu(
        records,
        expected_peak_mib=EXPECTED_PEAK_MIB,
        safety_margin_mib=SAFETY_MARGIN_MIB,
    )
    if selected is None:
        raise RuntimeError("BLOCKED_WAITING_IDLE_GPU: no card satisfies the frozen admission gate")
    if requested_gpu >= 0 and selected.index != requested_gpu:
        raise RuntimeError(
            f"GPU{requested_gpu} is not the current lowest-utilization eligible card; selected GPU{selected.index}"
        )
    command = canonical_command(root, selected.index)
    config = {
        "schema_version": "phase17.s1_contract_smoke.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "step_id": "S17-1",
        "purpose": "public migration/runtime contract plus 100-sample parent-GRAM smoke; no effect comparison",
        "dataset_view": "Toys D0 100-user target-independent profile view",
        "enabled_s17_modules": [],
        "physical_gpu": selected.index,
        "expected_peak_mib": EXPECTED_PEAK_MIB,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "usable_memory_ceiling_mib": MAX_USABLE_MIB_PER_JOB,
        "timeout_seconds": TIMEOUT_SECONDS,
        "test_read": False,
        "sports_read": False,
        "scientific_result_eligible": False,
        "command": command,
    }
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=command,
        source_paths=source_paths(root),
        config=config,
    )
    resolved["output"].mkdir(parents=True)
    atomic_json(resolved["config"], config)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-1",
        attempt_id=ATTEMPT_ID,
        canonical_result_dir=str(resolved["output"].relative_to(root)),
        log_path=str(resolved["gpu_log"].relative_to(root)),
        extra={
            "stage": "preflight",
            "progress": {"current": 0, "total": 2},
            "gpu_ids": [selected.index],
            "gpu_snapshot": {
                "captured_at": utc_now(),
                "devices": snapshot(records),
                "selected_gpu": selected.index,
                "selection_reason": "lowest utilization among cards satisfying expected peak plus 4096 MiB margin",
            },
            "predicted_peak_mib": EXPECTED_PEAK_MIB,
            "usable_memory_ceiling_mib": MAX_USABLE_MIB_PER_JOB,
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
        },
    )
    writer.transition("PREFLIGHT", "PREFLIGHT", "CPU_CONTRACT_TESTS", process_alive=True)
    print(manifest)
    return 0


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    timeout: int,
    on_start: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        if on_start is not None:
            on_start(process.pid)
        try:
            return process.wait(timeout=timeout), process.pid
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124, process.pid


def append_attempt(root: Path, state: str, failure_reason: str | None) -> None:
    resolved = paths(root)
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": "S17-1",
            "track_id": None,
            "kind": "smoke",
            "started_at": json.loads(
                (resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json").read_text(encoding="utf-8")
            )["started_at"],
            "ended_at": utc_now(),
            "state": state,
            "config_sha256": sha256(resolved["config"]),
            "data_manifest_sha256": sha256(root / "artifacts/phase17/s0_audit/shadow_data_manifest.json"),
            "source_sha256": sha256(resolved["snapshot"]),
            "scientific_result_eligible": False,
            "failure_reason": failure_reason,
            "artifact_dir": str(resolved["output"].relative_to(root)),
            "command_sha256": hashlib.sha256("\0".join(config["command"]).encode()).hexdigest(),
        }
    )


def worker(root: Path) -> int:
    resolved = paths(root)
    verify_run_snapshot(root, resolved["snapshot"])
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cpu_command = [
        str(PYTHON), "-m", "unittest", "discover", "-v",
        "-s", "experiment/phase17/tests", "-p", "test_*.py",
    ]
    cpu_rc, cpu_pid = run_logged(cpu_command, root, resolved["cpu_log"], env, timeout=120)
    if cpu_rc != 0:
        writer.transition(
            "FAILED", "SCIENTIFIC_FAILED", "CPU_CONTRACT_TESTS_FAILED",
            workload_pid=0, process_alive=False, stage="failed_cpu_contracts",
        )
        append_attempt(root, "FAILED", "CPU contract tests failed; see log; no automatic retry")
        return cpu_rc

    physical_gpu = int(config["physical_gpu"])
    writer.transition(
        "RUNNING", "RUNNING_SCIENTIFIC", "GPU_SMOKE_RUNNING",
        workload_pid=0, process_alive=True, stage="gram_100_sample_smoke",
        progress={"current": 1, "total": 2},
    )
    gpu_env = env.copy()
    gpu_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "HF_HUB_CACHE": str(root / ".cache/huggingface"),
            "TRANSFORMERS_CACHE": str(root / ".cache/huggingface/transformers"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    started = datetime.now(timezone.utc)
    gpu_rc, gpu_pid = run_logged(
        config["command"],
        root / "GRAM/command",
        resolved["gpu_log"],
        gpu_env,
        TIMEOUT_SECONDS,
        on_start=lambda pid: writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "GPU_SMOKE_RUNNING",
            workload_pid=pid,
            process_alive=True,
            stage="gram_100_sample_smoke",
            progress={"current": 1, "total": 2},
        ),
    )
    ended = datetime.now(timezone.utc)
    text = resolved["gpu_log"].read_text(encoding="utf-8", errors="replace")
    metric_pattern = re.compile(
        r"RESOURCE_METRIC phase=(\S+) wall_time_seconds=([0-9.]+) "
        r"peak_allocated_mib=([0-9.]+) peak_reserved_mib=([0-9.]+)"
    )
    metrics = [
        {
            "phase": match.group(1),
            "wall_time_seconds": float(match.group(2)),
            "peak_allocated_mib": float(match.group(3)),
            "peak_reserved_mib": float(match.group(4)),
        }
        for match in metric_pattern.finditer(text)
    ]
    forbidden = [token for token in ("automatic_last_checkpoint_test", "[test] testing", "_pred_test.tsv") if token in text]
    passed = (
        gpu_rc == 0
        and {row["phase"] for row in metrics} == {"training", "automatic_last_checkpoint_validation"}
        and not forbidden
        and "Traceback" not in text
    )
    summary = {
        "schema_version": "phase17.s1_contract_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "verdict": "PASS_S17_1_CONTRACT_AND_GPU_SMOKE" if passed else "FAIL_S17_1_GPU_SMOKE",
        "cpu_contract_tests": {"return_code": cpu_rc, "log_path": str(resolved["cpu_log"].relative_to(root))},
        "gpu_smoke": {
            "physical_gpu": physical_gpu,
            "workload_pid": gpu_pid,
            "return_code": gpu_rc,
            "wall_seconds": (ended - started).total_seconds(),
            "resource_metrics": metrics,
            "forbidden_test_evidence": forbidden,
            "log_path": str(resolved["gpu_log"].relative_to(root)),
            "log_sha256": sha256(resolved["gpu_log"]),
        },
        "run_snapshot_manifest": str(resolved["snapshot"].relative_to(root)),
        "run_snapshot_sha256": sha256(resolved["snapshot"]),
        "test_read": False,
        "sports_read": False,
        "scientific_result_eligible": False,
    }
    atomic_json(resolved["summary"], summary)
    if passed:
        writer.transition(
            "COMPLETED", "SCIENTIFIC_COMPLETED", "S17_1_CONTRACTS_COMPLETE",
            workload_pid=0, process_alive=False, stage="complete",
            progress={"current": 2, "total": 2},
            result_selection_eligible=False,
            summary_path=str(resolved["summary"].relative_to(root)),
        )
        append_attempt(root, "COMPLETED", None)
        print(summary["verdict"])
        return 0
    code = "GPU_SMOKE_TIMEOUT" if gpu_rc == 124 else "GPU_SMOKE_FAILED"
    writer.transition(
        "FAILED", "SCIENTIFIC_FAILED", code,
        workload_pid=0, process_alive=False, stage="failed_gpu_smoke",
        progress={"current": 1, "total": 2}, summary_path=str(resolved["summary"].relative_to(root)),
    )
    append_attempt(root, "FAILED", f"{code}; see log; no automatic retry")
    return gpu_rc or 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root, args.gpu)
    if args.action == "worker":
        return worker(root)
    status = paths(root)["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    print(status.read_text(encoding="utf-8") if status.exists() else '{"scientific_state":"PENDING"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
