#!/usr/bin/env python3
"""Hand an active S17-3 serial queue to a two-GPU scientific scheduler.

The active arm is adopted without restarting it.  GPU0 and GPU1 then draw
different remaining arms from one queue.  After canonical science completes,
GPU0 is released and only GPU1 enters the isolated runtime cycle.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_HINT = os.environ.get("S17_REPOSITORY_ROOT") or str(Path(__file__).resolve().parents[3])
if ROOT_HINT not in sys.path:
    sys.path.insert(0, ROOT_HINT)

from experiment.phase17.core.resource_profiler import query_gpus, snapshot  # noqa: E402
from experiment.phase17.core.run_manager import sha256  # noqa: E402
from experiment.phase17.core.status_writer import (  # noqa: E402
    AttemptLedger,
    StatusWriter,
    assert_neutral_name,
    atomic_json,
    utc_now,
)
from experiment.phase17.protocol.s2_probe_runtime import parse_log  # noqa: E402
from experiment.phase17.protocol.s3_exploration_portfolio_runtime import (  # noqa: E402
    ARM_HARD_TIMEOUT_SECONDS,
    ATTEMPT_ID,
    EXPERIMENT_ID,
    MONITOR_SECONDS,
    PYTHON,
    STALL_CHECKS,
    find_checkpoint,
    paths,
    runtime_loop,
)


SCIENCE_GPUS = (0, 1)
RUNTIME_GPU = 1
TMUX_SESSION = "s17_s3_dual_lane_seed2023"
HANDOFF_SCHEMA = "phase17.s17_3_dual_gpu_handoff.v1"


def arm_environment(root: Path, physical_gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root),
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "HF_HUB_CACHE": str(root / ".cache/huggingface"),
            "TRANSFORMERS_CACHE": str(root / ".cache/huggingface/transformers"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def pending_arm_ids(config: dict, completed: set[str], active_arm: str | None) -> list[str]:
    return [
        arm["arm_id"]
        for arm in config["arms"]
        if arm["arm_id"] not in completed and arm["arm_id"] != active_arm
    ]


def proc_state(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = raw.rfind(")")
    return raw[closing + 2 :].split()[0]


def proc_parent(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split()
    return int(fields[1])


def proc_exit_code(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split()
    if fields[0] != "Z" or len(fields) < 50:
        return None
    try:
        return os.waitstatus_to_exitcode(int(fields[49]))
    except (AttributeError, ValueError):
        return int(fields[49]) >> 8


def proc_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return ""


def iso_elapsed_seconds(started_at: str) -> float:
    started = datetime.fromisoformat(started_at)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def make_arm_config(root: Path, config: dict, arm: dict) -> Path:
    arm_dir = paths(root)["output"] / "arms" / arm["arm_id"]
    if arm_dir.exists():
        raise FileExistsError(f"canonical arm output already exists: {arm_dir}")
    arm_dir.mkdir(parents=True)
    arm_config_path = arm_dir / "config.json"
    atomic_json(
        arm_config_path,
        {
            "schema_version": "phase17.s17_3_one_epoch_arm.v1",
            "attempt_id": f"s3_{arm['arm_id']}_001",
            **arm,
            "epochs": config["epochs_per_arm"],
            "seed": config["seed"],
            "command": config["commands"][arm["arm_id"]],
            "baseline_checkpoint": config["baseline_checkpoint"],
            "historical_validation": config["historical_baseline"]["validation_metrics"],
            "test_read": False,
            "sports_read": False,
        },
    )
    return arm_config_path


def append_attempt(root: Path, config: dict, arm: dict, result: dict, arm_config: Path) -> None:
    AttemptLedger(paths(root)["ledger"]).append(
        {
            "attempt_id": result["attempt_id"],
            "step_id": "S17-3",
            "track_id": arm["track_id"],
            "kind": "one_epoch_exploration",
            "started_at": result["started_at"],
            "ended_at": result["ended_at"],
            "state": result["state"],
            "config_sha256": sha256(arm_config),
            "data_manifest_sha256": config["data_audit_sha256"],
            "source_sha256": sha256(paths(root)["snapshot"]),
            "scientific_result_eligible": result["state"] == "COMPLETED",
            "failure_reason": None
            if result["state"] == "COMPLETED"
            else "one-epoch arm completion contract failed",
            "artifact_dir": str((paths(root)["output"] / "arms" / arm["arm_id"]).relative_to(root)),
            "physical_gpu": result["physical_gpu"],
        }
    )


def collect_result(
    *,
    root: Path,
    config: dict,
    arm: dict,
    physical_gpu: int,
    admission_free_mib: int,
    started_at: str,
    wall_seconds: float,
    workload_pid: int,
    return_code: int,
) -> dict:
    arm_dir = paths(root)["output"] / "arms" / arm["arm_id"]
    log_path = arm_dir / "run.log"
    arm_config = arm_dir / "config.json"
    parsed = parse_log(log_path)
    checkpoint = find_checkpoint(arm_dir, int(config["epochs_per_arm"]))
    checks = {
        "exit_zero": return_code == 0,
        "no_traceback": not parsed["traceback"],
        "no_forbidden_test_evidence": not parsed["forbidden_test_evidence"],
        "one_training_epoch": len(parsed["training_losses"]) == 1,
        "validation_completed": bool(parsed["validation_metrics"]),
        "mechanism_metric_present": arm["module_id"] == ""
        or bool(parsed["mechanism_metric_lines"]),
        "within_memory_ceiling": parsed["peak_reserved_mib"] is not None
        and parsed["peak_reserved_mib"] <= int(config["usable_memory_ceiling_mib"]),
        "checkpoint_present": checkpoint is not None,
    }
    passed = all(checks.values())
    historical = config["historical_baseline"]["validation_metrics"]
    validation = parsed["validation_metrics"]
    result = {
        "schema_version": "phase17.s17_3_one_epoch_result.v1",
        "attempt_id": f"s3_{arm['arm_id']}_001",
        **arm,
        "state": "COMPLETED" if passed else "FAILED",
        "started_at": started_at,
        "ended_at": utc_now(),
        "return_code": return_code,
        "workload_pid": workload_pid,
        "physical_gpu": physical_gpu,
        "admission_free_mib": admission_free_mib,
        "wall_seconds": wall_seconds,
        "checks": checks,
        "parsed": parsed,
        "checkpoint": str(checkpoint.relative_to(root)) if checkpoint else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "historical_validation": historical,
        "delta_vs_historical": {
            metric: validation.get(metric) - value if validation.get(metric) is not None else None
            for metric, value in historical.items()
        },
        "official_result_claim": False,
        "test_read": False,
        "sports_read": False,
        "execution_topology": "dual_gpu_handoff",
    }
    atomic_json(arm_dir / "result.json", result)
    append_attempt(root, config, arm, result, arm_config)
    return result


def gpu_map() -> tuple[dict[int, Any], list[Any]]:
    records = query_gpus()
    return {record.index: record for record in records}, records


def start_arm(root: Path, config: dict, arm: dict, gpu: int, admission: int) -> dict:
    arm_config = make_arm_config(root, config, arm)
    arm_dir = arm_config.parent
    log_handle = (arm_dir / "run.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        config["commands"][arm["arm_id"]],
        cwd=root / "GRAM/command",
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=arm_environment(root, gpu),
        start_new_session=True,
    )
    return {
        "kind": "owned",
        "arm": arm,
        "gpu": gpu,
        "pid": process.pid,
        "process": process,
        "log_handle": log_handle,
        "log_path": arm_dir / "run.log",
        "started_at": utc_now(),
        "started_monotonic": time.monotonic(),
        "admission_free_mib": admission,
        "last_size": -1,
        "unchanged_checks": 0,
        "stall_advisory": False,
    }


def active_status(writer: StatusWriter, jobs: dict[int, dict], arm_states: dict, completed: list[str], failed: list[str], pending: list[str], records: list[Any], stage: str) -> None:
    current_arms = {str(gpu): job["arm"]["arm_id"] for gpu, job in sorted(jobs.items())}
    workload_pids = {str(gpu): job["pid"] for gpu, job in sorted(jobs.items())}
    primary_gpu = RUNTIME_GPU if RUNTIME_GPU in jobs else next(iter(sorted(jobs)), None)
    primary = jobs.get(primary_gpu) if primary_gpu is not None else None
    writer.transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC" if jobs else "WAITING_FOR_GPU",
        "S17_3_DUAL_GPU_ARMS_RUNNING" if jobs else "S17_3_DUAL_GPU_WAITING",
        heartbeat_at=utc_now(),
        process_alive=True,
        workload_pid=primary["pid"] if primary else 0,
        workload_pids=workload_pids,
        gpu_ids=sorted(jobs),
        allocated_gpu_ids=list(SCIENCE_GPUS),
        gpu_snapshot={"captured_at": utc_now(), "devices": snapshot(records)},
        stage=stage,
        progress={
            "current": len(completed) + len(failed),
            "total": len(arm_states),
            "unit": "arm",
            "running": len(jobs),
            "queued": len(pending),
        },
        current_arm=primary["arm"]["arm_id"] if primary else None,
        current_arms=current_arms,
        completed_arms=completed,
        failed_arms=failed,
        arm_states=arm_states,
        queue_mode="dynamic_dual_gpu",
        gpu0_post_science="release",
        gpu1_post_science="runtime_cycle",
        handoff_active=True,
    )


def retire_serial_parent(parent_pid: int) -> None:
    if proc_state(parent_pid) is None:
        return
    os.kill(parent_pid, signal.SIGTERM)
    os.kill(parent_pid, signal.SIGCONT)
    deadline = time.monotonic() + 15
    while proc_state(parent_pid) is not None and time.monotonic() < deadline:
        time.sleep(0.25)
    if proc_state(parent_pid) is not None:
        raise RuntimeError("serial orchestrator did not exit after SIGTERM")


def adopt_active_job(root: Path, status: dict, config: dict) -> tuple[int, dict, int]:
    active_arm_id = status.get("current_arm")
    active_pid = int(status.get("workload_pid") or 0)
    if not active_arm_id or active_pid <= 0 or proc_state(active_pid) in {None, "Z", "X"}:
        raise RuntimeError("no live S17-3 arm is available for zero-restart handoff")
    parent_pid = proc_parent(active_pid)
    if "s3_exploration_portfolio_runtime.py" not in proc_command(parent_pid):
        raise RuntimeError("active arm parent is not the expected S17-3 serial orchestrator")
    arm = next(arm for arm in config["arms"] if arm["arm_id"] == active_arm_id)
    gpu = int(status.get("gpu_ids", [1])[0])
    started_at = status.get("gpu_snapshot", {}).get("captured_at") or utc_now()
    admission = int(status.get("gpu_snapshot", {}).get("admission_free_mib") or 0)
    return parent_pid, {
        "kind": "adopted",
        "arm": arm,
        "gpu": gpu,
        "pid": active_pid,
        "process": None,
        "log_handle": None,
        "log_path": paths(root)["output"] / "arms" / active_arm_id / "run.log",
        "started_at": started_at,
        "started_monotonic": None,
        "admission_free_mib": admission,
        "last_size": -1,
        "unchanged_checks": 0,
        "stall_advisory": False,
    }, active_pid


def job_return_code(job: dict) -> int | None:
    if job["kind"] == "owned":
        return job["process"].poll()
    state = proc_state(job["pid"])
    if state not in {None, "Z", "X"}:
        return None
    code = proc_exit_code(job["pid"])
    return 0 if code is None else code


def finish_job(root: Path, config: dict, job: dict, return_code: int) -> dict:
    if job["log_handle"] is not None:
        job["log_handle"].close()
    wall_seconds = (
        time.monotonic() - job["started_monotonic"]
        if job["started_monotonic"] is not None
        else iso_elapsed_seconds(job["started_at"])
    )
    return collect_result(
        root=root,
        config=config,
        arm=job["arm"],
        physical_gpu=job["gpu"],
        admission_free_mib=job["admission_free_mib"],
        started_at=job["started_at"],
        wall_seconds=wall_seconds,
        workload_pid=job["pid"],
        return_code=return_code,
    )


def write_handoff_record(root: Path, config: dict, status: dict, parent_pid: int, active_pid: int) -> Path:
    target = paths(root)["output"] / "preflight/dual_gpu_handoff.json"
    if target.exists():
        raise FileExistsError(f"handoff record already exists: {target}")
    source = Path(__file__).resolve()
    payload = {
        "schema_version": HANDOFF_SCHEMA,
        "created_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "science_gpus": list(SCIENCE_GPUS),
        "runtime_gpu": RUNTIME_GPU,
        "active_arm": status["current_arm"],
        "active_pid": active_pid,
        "serial_parent_pid": parent_pid,
        "commands_unchanged": True,
        "portfolio_config_sha256": sha256(paths(root)["config"]),
        "handoff_source": str(source.relative_to(root)),
        "handoff_source_sha256": sha256(source),
        "gpu0_post_science": "release",
        "gpu1_post_science": "runtime_cycle",
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(target, payload)
    return target


def write_summary(root: Path, config: dict, completed: list[str], failed: list[str]) -> dict:
    results_by_id = {}
    for arm in config["arms"]:
        result_path = paths(root)["output"] / "arms" / arm["arm_id"] / "result.json"
        if result_path.exists():
            results_by_id[arm["arm_id"]] = json.loads(result_path.read_text(encoding="utf-8"))
    results = [results_by_id[arm["arm_id"]] for arm in config["arms"] if arm["arm_id"] in results_by_id]
    summary = {
        "schema_version": "phase17.s17_3_one_epoch_portfolio_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "step_id": "S17-3",
        "substep": "one_epoch_exploration_screen",
        "queue_completed": True,
        "all_arms_passed": not failed,
        "completed_arms": completed,
        "failed_arms": failed,
        "historical_baseline": config["historical_baseline"],
        "results": results,
        "execution_topology": "dynamic_dual_gpu_handoff",
        "science_gpus": list(SCIENCE_GPUS),
        "runtime_gpu": RUNTIME_GPU,
        "official_result_claim": False,
        "test_read": False,
        "sports_read": False,
        "completed_at": utc_now(),
    }
    atomic_json(paths(root)["summary"], summary)
    return summary


def handoff(root: Path) -> int:
    assert_neutral_name(TMUX_SESSION)
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    status = writer.read()
    if status["scientific_state"] != "RUNNING":
        raise RuntimeError("S17-3 scientific queue is not running")

    parent_pid, adopted, active_pid = adopt_active_job(root, status, config)
    handoff_record = write_handoff_record(root, config, status, parent_pid, active_pid)
    # The active workload was launched in its own session, so retiring only the
    # serial parent leaves that workload alive and prevents duplicate dispatch.
    retire_serial_parent(parent_pid)

    completed = list(status.get("completed_arms", []))
    failed = list(status.get("failed_arms", []))
    completed_set = set(completed) | set(failed)
    pending = pending_arm_ids(config, completed_set, adopted["arm"]["arm_id"])
    arm_by_id = {arm["arm_id"]: arm for arm in config["arms"]}
    arm_states = {arm["arm_id"]: "QUEUED" for arm in config["arms"]}
    for arm_id in completed:
        arm_states[arm_id] = "COMPLETED"
    for arm_id in failed:
        arm_states[arm_id] = "FAILED"
    arm_states[adopted["arm"]["arm_id"]] = "RUNNING"
    jobs: dict[int, dict] = {adopted["gpu"]: adopted}
    serial_parent_retired = True

    writer.transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_3_DUAL_GPU_HANDOFF_ACTIVE",
        heartbeat_at=utc_now(),
        tmux_session=TMUX_SESSION,
        launcher_pid=os.getpid(),
        allocated_gpu_ids=list(SCIENCE_GPUS),
        queue_mode="dynamic_dual_gpu",
        handoff_active=True,
        handoff_record=str(handoff_record.relative_to(root)),
        serial_parent_pid=parent_pid,
        serial_parent_state="retired_after_zero_restart_handoff",
        gpu0_post_science="release",
        gpu1_post_science="runtime_cycle",
    )

    while pending or jobs:
        try:
            records_by_gpu, records = gpu_map()
        except Exception as error:
            writer.transition(
                "RUNNING",
                "WAITING_FOR_GPU",
                "S17_3_GPU_TELEMETRY_WAITING",
                heartbeat_at=utc_now(),
                process_alive=True,
                stage="gpu_telemetry_waiting",
                telemetry_advisory=f"{type(error).__name__}: {error}",
            )
            time.sleep(MONITOR_SECONDS)
            continue
        for gpu in SCIENCE_GPUS:
            if gpu in jobs or not pending:
                continue
            selected = records_by_gpu[gpu]
            if selected.free_mib < int(config["minimum_free_mib_at_admission"]):
                continue
            arm_id = pending.pop(0)
            arm = arm_by_id[arm_id]
            try:
                job = start_arm(root, config, arm, gpu, selected.free_mib)
            except Exception as error:
                failed.append(arm_id)
                arm_states[arm_id] = "FAILED"
                arm_dir = resolved["output"] / "arms" / arm_id
                arm_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    arm_dir / "result.json",
                    {
                        "schema_version": "phase17.s17_3_one_epoch_result.v1",
                        "attempt_id": f"s3_{arm_id}_001",
                        **arm,
                        "state": "FAILED",
                        "failure_reason": f"{type(error).__name__}: {error}",
                        "test_read": False,
                        "sports_read": False,
                    },
                )
                continue
            jobs[gpu] = job
            arm_states[arm_id] = "RUNNING"

        active_status(
            writer,
            jobs,
            arm_states,
            completed,
            failed,
            pending,
            records,
            "dual_gpu_science_active" if jobs else "dual_gpu_waiting_for_memory",
        )

        time.sleep(MONITOR_SECONDS)
        for gpu, job in list(jobs.items()):
            elapsed = (
                time.monotonic() - job["started_monotonic"]
                if job["started_monotonic"] is not None
                else iso_elapsed_seconds(job["started_at"])
            )
            size = job["log_path"].stat().st_size if job["log_path"].exists() else 0
            job["unchanged_checks"] = job["unchanged_checks"] + 1 if size == job["last_size"] else 0
            job["last_size"] = size
            job["stall_advisory"] = job["unchanged_checks"] >= STALL_CHECKS
            return_code = job_return_code(job)
            if return_code is None and elapsed <= int(config.get("arm_hard_timeout_seconds", ARM_HARD_TIMEOUT_SECONDS)):
                continue
            if return_code is None:
                os.killpg(job["pid"], signal.SIGTERM)
                return_code = 124
            try:
                result = finish_job(root, config, job, return_code)
            except Exception as error:
                result = {
                    "state": "FAILED",
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            arm_id = job["arm"]["arm_id"]
            if result["state"] == "COMPLETED":
                completed.append(arm_id)
                arm_states[arm_id] = "COMPLETED"
            else:
                failed.append(arm_id)
                arm_states[arm_id] = "FAILED"
            del jobs[gpu]
            if job["kind"] == "adopted":
                serial_parent_retired = True

    if not serial_parent_retired:
        retire_serial_parent(parent_pid)

    write_summary(root, config, completed, failed)
    scientific_state = "COMPLETED" if completed else "FAILED"
    writer.transition(
        scientific_state,
        "SCIENTIFIC_COMPLETED" if completed else "SCIENTIFIC_FAILED",
        "S17_3_ONE_EPOCH_QUEUE_COMPLETE" if not failed else "S17_3_ONE_EPOCH_QUEUE_COMPLETE_WITH_FAILURES",
        heartbeat_at=utc_now(),
        workload_pid=0,
        workload_pids={},
        process_alive=False,
        gpu_ids=[],
        allocated_gpu_ids=list(SCIENCE_GPUS),
        stage="scientific_queue_complete",
        progress={"current": len(completed) + len(failed), "total": len(config["arms"]), "unit": "arm"},
        current_arm=None,
        current_arms={},
        completed_arms=completed,
        failed_arms=failed,
        arm_states=arm_states,
        result_selection_eligible=bool(completed),
        affects_scientific_result=True,
        summary_path=str(resolved["summary"].relative_to(root)),
        handoff_active=False,
        gpu0_state="released_after_science",
        gpu1_state="ready_for_runtime_cycle" if completed else "released",
    )
    if completed:
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "S17_3_GPU1_RUNTIME_CYCLE_PENDING",
            gpu_ids=[RUNTIME_GPU],
            allocated_gpu_ids=[RUNTIME_GPU],
            gpu0_state="released_after_science",
            gpu1_state="runtime_cycle_pending",
        )
        source = next(arm for arm in reversed(config["arms"]) if arm["arm_id"] in completed)
        runtime_loop(root, writer, config, source)
    return 0 if completed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["handoff"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    return handoff(args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
