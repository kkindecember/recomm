#!/usr/bin/env python3
"""Repeat the non-scientific G1 workload across fresh CUDA processes.

V2 executed every cycle in the long-lived guard process.  PyTorch's CUDA
caching allocator consequently kept the first cycle's reservation alive, so
the guard counted its own memory against the next admission check and waited
forever.  V3 keeps the lightweight controller CPU-only and runs each resource
cycle in a child process.  A successful child exit is the CUDA-release
boundary before the next two-snapshot admission check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    tmux_session_exists,
    verify_run_snapshot,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_external_d0_g1_parallel_runtime as g1
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as profile_base


ROOT = Path(__file__).resolve().parents[3]
SOURCE_BASENAME = "s17_fp12_g1_runtime_guard_v3.py"
EXPERIMENT_ID = "s17_fp12_external_d0_g1_guard_v3"
ATTEMPT_ID = "attempt_001"
RESULT_SUFFIX = Path("artifacts/phase17/runtime/s17_fp12_external_d0_g1_guard_v3/attempt_001")
STATUS_DIR_SUFFIX = Path("artifacts/phase17/status")
SNAPSHOT_SUFFIX = Path(
    "artifacts/phase17/snapshots/s17_fp12_external_d0_g1_guard_v3/attempt_001/manifest.json"
)
V2_EXPERIMENT_ID = "s17_fp12_external_d0_g1_guard_v2"
PHYSICAL_GPU = g1.PHYSICAL_GPU

# The canonical GPU4 profile measured 15,892 MiB peak reserved and the live v2
# process measured about 17,862 MiB including CUDA context overhead.  The old
# 18,968 MiB threshold cannot be met after a new pre-existing 2.7 GiB process
# appeared, even though the already-running workload still fits.  V3 uses a
# two-snapshot 18,000 MiB gate and never evicts or modifies pre-existing jobs.
PROFILE_PEAK_RESERVED_MIB = 15892
MINIMUM_FREE_MIB = 18000
ADMISSION_POLL_SECONDS = 30
INTER_CYCLE_COOLDOWN_SECONDS = 2
V2_STOP_TIMEOUT_SECONDS = 30
RESOURCE_PROBE_ERRORS = (
    subprocess.SubprocessError,
    OSError,
    RuntimeError,
    ValueError,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _profile_spec():
    return profile_base.ProfileSpec(
        g1.ARM_ID,
        "S17-FP12-EXTERNAL-D0-G1-GUARD-V3",
        PHYSICAL_GPU,
        PROFILE_PEAK_RESERVED_MIB,
        MINIMUM_FREE_MIB,
        2,
        1,
        "gram",
    )


def _admitted() -> tuple[bool, dict[str, Any]]:
    first = profile_base.gpu_snapshot_once(_profile_spec())
    time.sleep(5)
    second = profile_base.gpu_snapshot_once(_profile_spec())
    admitted = all(
        row["selected"]["free_mib"] >= MINIMUM_FREE_MIB
        for row in (first, second)
    )
    return admitted, {
        "first": first,
        "second": second,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "profile_peak_reserved_mib": PROFILE_PEAK_RESERVED_MIB,
        "utilization_recorded_only": True,
        "preexisting_processes_preserved": True,
        "controller_is_cpu_only": True,
    }


def _probe_admission_fail_soft() -> tuple[
    bool | None, dict[str, Any] | None, str | None
]:
    try:
        admitted, snapshots = _admitted()
    except RESOURCE_PROBE_ERRORS as error:
        return None, None, repr(error)
    return admitted, snapshots, None


def _verify_self(root: Path, manifest: Path) -> None:
    verify_run_snapshot(root, manifest)
    record = _read(manifest)["files"][0]
    if sha256(root / record["source_path"]) != record["sha256"]:
        raise RuntimeError("G1 guard v3 live source drifted")


def _cycle_dir(root: Path, iteration: int) -> Path:
    if iteration < 2:
        raise ValueError("run-0001 remains reserved; guard cycles begin at run-0002")
    return root / RESULT_SUFFIX / f"run-{iteration:04d}"


def _validate_cycle_dir(root: Path, cycle_dir: Path, iteration: int) -> Path:
    resolved = cycle_dir.resolve()
    expected_parent = (root / RESULT_SUFFIX).resolve()
    if resolved.parent != expected_parent:
        raise PermissionError("cycle directory escaped the isolated v3 result tree")
    if resolved.name != f"run-{iteration:04d}" or not re.fullmatch(r"run-\d{4}", resolved.name):
        raise ValueError("cycle directory does not match its iteration")
    return resolved


def _snapshot_worker(manifest: Path) -> Path:
    return manifest.parent / f"src/000_{SOURCE_BASENAME}"


def cycle_command(
    root: Path, manifest: Path, cycle_dir: Path, iteration: int
) -> list[str]:
    return [
        str(profile_base.GRAM_PYTHON),
        str(_snapshot_worker(manifest)),
        "cycle-worker",
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--cycle-dir",
        str(cycle_dir),
        "--iteration",
        str(iteration),
    ]


def prepare(root: Path) -> int:
    root = root.resolve()
    result = root / RESULT_SUFFIX
    status_path = root / STATUS_DIR_SUFFIX / f"{EXPERIMENT_ID}.status.json"
    manifest = root / SNAPSHOT_SUFFIX
    if result.exists() or status_path.exists() or manifest.exists():
        raise FileExistsError("G1 guard v3 already exists")
    g1.verify(root)
    config = {
        "schema_version": "phase17.s17_fp12_g1_guard_v3_config.v1",
        "attempt_id": ATTEMPT_ID,
        "arm_id": g1.ARM_ID,
        "physical_gpu": PHYSICAL_GPU,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "profile_peak_reserved_mib": PROFILE_PEAK_RESERVED_MIB,
        "g1_authorization_path": str(g1.paths(root)["authorization"].relative_to(root)),
        "g1_authorization_sha256": sha256(g1.paths(root)["authorization"]),
        "g1_snapshot_manifest": str(g1.paths(root)["snapshot"].relative_to(root)),
        "controller_is_cpu_only": True,
        "fresh_cuda_process_per_cycle": True,
        "cuda_release_boundary": "successful_cycle_child_exit",
        "admission_snapshots_per_cycle": 2,
        "resource_probe_timeout_is_fail_soft": True,
        "scientific_cycle_retry": False,
        "result_selection_eligible": False,
        "repeat_metrics_ignored": True,
        "affects_scientific_result": False,
        "raw_external_projection_reopened": False,
    }
    result.mkdir(parents=True, exist_ok=False)
    atomic_json(result / "config.json", config)
    frozen = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=["repeat_non_scientific_g1_profile_in_fresh_cuda_processes"],
        source_paths=[Path(__file__).resolve()],
        config=config,
    )
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-FP12-EXTERNAL-D0-G1-GUARD",
        attempt_id=ATTEMPT_ID,
        track_id="G1_RUNTIME_GUARD_V3",
        canonical_result_dir=str(RESULT_SUFFIX),
        log_path=str(RESULT_SUFFIX / "guard.log"),
        extra={
            "stage": "prepared_waiting_exact_migration_command",
            "run_snapshot_manifest": str(frozen.relative_to(root)),
            "target_gpu_id": PHYSICAL_GPU,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "profile_peak_reserved_mib": PROFILE_PEAK_RESERVED_MIB,
            "process_alive": False,
            "result_selection_eligible": False,
            "repeat_metrics_ignored": True,
            "affects_scientific_result": False,
            "external_target_materialized": False,
            "raw_external_projection_reopened": False,
            "fresh_cuda_process_per_cycle": True,
            "resource_probe_timeout_is_fail_soft": True,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_G1_GUARD_V3_PREFLIGHT_COMPLETE",
        process_alive=False,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cycle_worker(
    root: Path, manifest: Path, cycle_dir: Path, iteration: int
) -> int:
    root = root.resolve()
    _verify_self(root, manifest)
    g1.verify(root, g1.paths(root)["snapshot"])
    cycle_dir = _validate_cycle_dir(root, cycle_dir, iteration)
    try:
        from experiment.phase17.core.full_latte_profile_executor import (
            run_resource_profile,
        )

        measurements = run_resource_profile(
            root,
            g1.ARM_ID,
            train_batch_size=2,
            eval_batch_size=1,
            heartbeat=None,
        )
        atomic_json(
            cycle_dir / "cycle.json",
            {
                "schema_version": "phase17.g1_runtime_cycle.v3",
                "completed_at": utc_now(),
                "iteration": iteration,
                "worker_pid": os.getpid(),
                "measurements": measurements,
                "cuda_release_boundary": "cycle_worker_process_exit",
                "result_selection_eligible": False,
                "repeat_metrics_ignored": True,
                "affects_scientific_result": False,
                "external_target_materialized": False,
                "raw_external_projection_reopened": False,
                "test_read": False,
                "sports_read": False,
                "d1_read": False,
                "d2_read": False,
            },
        )
        return 0
    except BaseException as error:
        atomic_json(
            cycle_dir / "failure.json",
            {
                "failed_at": utc_now(),
                "iteration": iteration,
                "worker_pid": os.getpid(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
                "result_selection_eligible": False,
                "affects_scientific_result": False,
            },
        )
        return 1


def _run_cycle_subprocess(
    root: Path, manifest: Path, writer: StatusWriter, iteration: int
) -> int:
    cycle_dir = _cycle_dir(root, iteration)
    cycle_dir.mkdir(parents=True, exist_ok=False)
    log_path = cycle_dir / "cycle.log"
    command = cycle_command(root, manifest, cycle_dir, iteration)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(root)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        writer.start_runtime_cycle(
            iteration=iteration,
            runtime_result_dir=str(cycle_dir.relative_to(root)),
            workload_pid=process.pid,
        )
        writer.heartbeat(
            stage="isolated_g1_resource_cycle_child_process",
            progress={"current": iteration - 1, "total": 0, "unit": "cycle"},
        )
        return_code = process.wait()
    if return_code != 0 or not (cycle_dir / "cycle.json").is_file():
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "S17_FP12_G1_GUARD_V3_CYCLE_FAILED_NO_RETRY",
            process_alive=False,
            workload_pid=0,
            gpu_ids=[],
            failed_iteration=iteration,
            cycle_return_code=return_code,
            cycle_failure_path=str((cycle_dir / "failure.json").relative_to(root)),
            result_selection_eligible=False,
            repeat_metrics_ignored=True,
            affects_scientific_result=False,
        )
        return 1
    writer.transition(
        "COMPLETED",
        "WAITING_FOR_GPU",
        "PASS_S17_FP12_G1_GUARD_V3_CYCLE_COMPLETE_CUDA_RELEASED",
        process_alive=True,
        workload_pid=os.getpid(),
        gpu_ids=[],
        stage="cycle_child_exited_cuda_released",
        completed_iteration=iteration,
        progress={"current": iteration - 1, "total": 0, "unit": "cycle"},
        result_selection_eligible=False,
        repeat_metrics_ignored=True,
        affects_scientific_result=False,
    )
    return 0


def worker(root: Path, manifest: Path) -> int:
    root = root.resolve()
    _verify_self(root, manifest)
    g1.verify(root, g1.paths(root)["snapshot"])
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    iteration = 2
    resource_probe_failures = 0
    while True:
        g1_status = _read(root / STATUS_DIR_SUFFIX / f"{g1.EXPERIMENT_ID}.status.json")
        if g1_status["scientific_state"] in {"FAILED", "BLOCKED", "STOPPED"}:
            writer.transition(
                "BLOCKED",
                "BLOCKED",
                "S17_FP12_G1_GUARD_V3_BLOCKED_BY_G1_TERMINAL_STATE",
                process_alive=False,
                workload_pid=0,
                terminal_g1_state=g1_status["scientific_state"],
            )
            return 2
        if g1_status["scientific_state"] != "COMPLETED":
            writer.heartbeat(
                stage="waiting_for_g1_scientific_completion",
                progress=g1_status.get(
                    "progress", {"current": 0, "total": 12833, "unit": "external_user"}
                ),
            )
            time.sleep(ADMISSION_POLL_SECONDS)
            continue
        if writer.read()["scientific_state"] == "PREFLIGHT":
            writer.transition(
                "RUNNING",
                "RUNNING_SCIENTIFIC",
                "S17_FP12_G1_GUARD_V3_ACTIVATING",
                process_alive=True,
            )
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "PASS_S17_FP12_G1_GUARD_V3_READY",
                process_alive=True,
                workload_pid=os.getpid(),
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
        admitted, snapshots, probe_error = _probe_admission_fail_soft()
        if admitted is None:
            resource_probe_failures += 1
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "S17_FP12_G1_GUARD_V3_RESOURCE_PROBE_WAITING",
                process_alive=True,
                workload_pid=os.getpid(),
                stage="resource_probe_transient_failure_waiting",
                progress={"current": iteration - 2, "total": 0, "unit": "cycle"},
                resource_probe_failures=resource_probe_failures,
                last_resource_probe_error=probe_error,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(ADMISSION_POLL_SECONDS)
            continue
        assert snapshots is not None
        if not admitted:
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "S17_FP12_G1_GUARD_V3_WAITING_FOR_MEMORY",
                process_alive=True,
                workload_pid=os.getpid(),
                stage="waiting_for_gpu_memory",
                progress={"current": iteration - 2, "total": 0, "unit": "cycle"},
                gpu_snapshot=snapshots,
                resource_probe_failures=resource_probe_failures,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(ADMISSION_POLL_SECONDS)
            continue
        if _run_cycle_subprocess(root, manifest, writer, iteration) != 0:
            return 1
        iteration += 1
        time.sleep(INTER_CYCLE_COOLDOWN_SECONDS)


def launch(root: Path) -> int:
    root = root.resolve()
    manifest = root / SNAPSHOT_SUFFIX
    _verify_self(root, manifest)
    g1.verify(root, g1.paths(root)["snapshot"])
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("G1 guard v3 is not in PREFLIGHT")
    command = [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(profile_base.GRAM_PYTHON),
        str(_snapshot_worker(manifest)),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(manifest),
    ]
    launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=command,
        cwd=root,
        tmux_session=EXPERIMENT_ID,
        startup_log_path=root / RESULT_SUFFIX / "guard.log",
    )
    writer.transition(
        "PREFLIGHT",
        "WAITING_FOR_GPU",
        "S17_FP12_G1_GUARD_V3_WAITING_FOR_G1_COMPLETION",
        tmux_session=EXPERIMENT_ID,
        launcher_pid=os.getpid(),
        process_alive=True,
    )
    if not wait_for_tmux_startup(EXPERIMENT_ID):
        writer.transition(
            "BLOCKED",
            "BLOCKED",
            "S17_FP12_G1_GUARD_V3_STARTUP_FAILED",
            process_alive=False,
        )
        raise RuntimeError("G1 guard v3 exited during startup handshake")
    print(EXPERIMENT_ID)
    return 0


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or not Path(f"/proc/{pid}").exists():
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _validate_v2_status(status: dict[str, Any]) -> tuple[int, str]:
    if status.get("experiment_id") != V2_EXPERIMENT_ID:
        raise ValueError("unexpected v2 experiment id")
    if status.get("scientific_state") != "COMPLETED":
        raise RuntimeError("v2 scientific state is not COMPLETED")
    if status.get("affects_scientific_result") is not False:
        raise RuntimeError("v2 is not isolated from scientific result selection")
    pid = int(status.get("workload_pid") or 0)
    session = str(status.get("tmux_session") or "")
    if pid <= 0 or session != V2_EXPERIMENT_ID:
        raise RuntimeError("v2 live process/session identity is incomplete")
    return pid, session


def stop_v2_for_migration(root: Path) -> int:
    root = root.resolve()
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, V2_EXPERIMENT_ID)
    status = writer.read()
    pid, session = _validate_v2_status(status)
    if not _pid_alive(pid):
        raise RuntimeError("v2 status says live but its recorded PID is not alive")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    )
    if "s17_fp12_g1_runtime_guard_v2.py" not in cmdline or " worker " not in f" {cmdline} ":
        raise RuntimeError("refusing to stop a PID that is not the frozen v2 worker")
    if not tmux_session_exists(session):
        raise RuntimeError("v2 tmux session is not alive")
    v2_result = root / Path(status["canonical_result_dir"])
    before_path = v2_result / "status_before_v3_migration.json"
    supersession_path = v2_result / "supersession_by_v3.json"
    if before_path.exists() or supersession_path.exists():
        raise FileExistsError("v2 migration evidence already exists")
    atomic_json(before_path, status)
    subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    deadline = time.monotonic() + V2_STOP_TIMEOUT_SECONDS
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if _pid_alive(pid):
        raise TimeoutError("v2 worker did not exit after its tmux session was stopped")
    stopped = writer.transition(
        "COMPLETED",
        "STOPPED",
        "S17_FP12_G1_GUARD_V2_SUPERSEDED_BY_FRESH_PROCESS_V3",
        process_alive=False,
        workload_pid=0,
        gpu_ids=[],
        tmux_session=None,
        occupancy_mode="none",
        superseded_by_experiment_id=EXPERIMENT_ID,
        result_selection_eligible=False,
        repeat_metrics_ignored=True,
        affects_scientific_result=False,
    )
    atomic_json(
        supersession_path,
        {
            "schema_version": "phase17.g1_guard_supersession.v1",
            "stopped_at": utc_now(),
            "stopped_pid": pid,
            "stopped_tmux_session": session,
            "superseded_by_experiment_id": EXPERIMENT_ID,
            "v2_status_after_stop": stopped,
            "scientific_result_affected": False,
        },
    )
    return 0


def _prepared(root: Path) -> bool:
    paths = (
        root / RESULT_SUFFIX,
        root / STATUS_DIR_SUFFIX / f"{EXPERIMENT_ID}.status.json",
        root / SNAPSHOT_SUFFIX,
    )
    present = [path.exists() for path in paths]
    if any(present) and not all(present):
        raise RuntimeError("partial v3 preparation artifacts exist")
    return all(present)


def migrate_v2(root: Path) -> int:
    root = root.resolve()
    # Validate the exact live target before writing any v3 runtime artifacts.
    _validate_v2_status(
        _read(root / STATUS_DIR_SUFFIX / f"{V2_EXPERIMENT_ID}.status.json")
    )
    if not _prepared(root):
        prepare(root)
    stop_v2_for_migration(root)
    return launch(root)


def inspect(root: Path) -> int:
    root = root.resolve()
    status_path = root / STATUS_DIR_SUFFIX / f"{EXPERIMENT_ID}.status.json"
    if not status_path.exists():
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "state": "NOT_PREPARED"}))
        return 0
    print(json.dumps(_read(status_path), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "launch",
            "worker",
            "cycle-worker",
            "stop-v2",
            "migrate-v2",
            "inspect",
        ),
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cycle-dir", type=Path)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.root)
    if args.action == "launch":
        return launch(args.root)
    if args.action == "stop-v2":
        return stop_v2_for_migration(args.root)
    if args.action == "migrate-v2":
        return migrate_v2(args.root)
    if args.action == "inspect":
        return inspect(args.root)
    if args.manifest is None:
        parser.error(f"{args.action} requires --manifest")
    if args.action == "worker":
        return worker(args.root, args.manifest)
    if args.cycle_dir is None or args.iteration is None:
        parser.error("cycle-worker requires --cycle-dir and --iteration")
    return cycle_worker(
        args.root, args.manifest, args.cycle_dir, args.iteration
    )


if __name__ == "__main__":
    raise SystemExit(main())
