#!/usr/bin/env python3
"""Queue the S17-3 one-epoch exploration arms on the allocated GPU."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT_HINT = os.environ.get("S17_REPOSITORY_ROOT") or str(Path(__file__).resolve().parents[3])
if ROOT_HINT not in sys.path:
    sys.path.insert(0, ROOT_HINT)

from experiment.phase17.core.resource_profiler import query_gpus, snapshot  # noqa: E402
from experiment.phase17.core.run_manager import (  # noqa: E402
    assert_runtime_isolation,
    freeze_run_snapshot,
    isolated_runtime_dir,
    launch_background_tmux,
    sha256,
    verify_run_snapshot,
)
from experiment.phase17.core.status_writer import (  # noqa: E402
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)
from experiment.phase17.protocol.s2_probe_runtime import (  # noqa: E402
    build_transition_teacher,
    command_template,
    parse_log,
)


EXPERIMENT_ID = "s17_s3_one_epoch_portfolio_seed2023"
ATTEMPT_ID = "run-0001"
TMUX_SESSION = EXPERIMENT_ID
PHYSICAL_GPU = 1
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
GPU_WAIT_SECONDS = 30
MONITOR_SECONDS = 30
STALL_CHECKS = 10
ARM_HARD_TIMEOUT_SECONDS = 12 * 60 * 60
PREDICTED_PEAK_MIB = 27000


def paths(root: Path) -> dict[str, Path]:
    output = root / "artifacts/phase17/s3_exploration" / ATTEMPT_ID
    return {
        "output": output,
        "status_dir": root / "artifacts/phase17/status",
        "status": root / "artifacts/phase17/status" / f"{EXPERIMENT_ID}.status.json",
        "ledger": root / "artifacts/phase17/attempts/S17-3.attempts.jsonl",
        "summary": output / "summary.json",
        "config": output / "portfolio_config.json",
        "worker_log": output / "portfolio.log",
        "snapshot": root / "artifacts/phase17/snapshots" / EXPERIMENT_ID / ATTEMPT_ID / "manifest.json",
        "budget": root / "experiment/phase17/config/s17_s3_formal_budget.json",
        "baseline": root / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt",
        "data_audit": root / "artifacts/phase2_toys/data_audit.md",
        "transition_teacher": output / "preflight/transition_teacher.json",
    }


def source_paths(root: Path, transition_teacher: Path) -> list[Path]:
    return [
        root / "experiment/phase17/protocol/s3_exploration_portfolio_runtime.py",
        root / "experiment/phase17/protocol/s2_probe_runtime.py",
        root / "experiment/phase17/run_stage17_s3_one_epoch_portfolio_gpu1.sh",
        root / "experiment/phase17/config/s17_s3_formal_budget.json",
        root / "artifacts/phase2_toys/data_audit.md",
        transition_teacher,
        *sorted((root / "experiment/phase17/core").glob("*.py")),
        root / "experiment/phase17/registry/module_registry.py",
        *sorted((root / "experiment/phase17/registry/migration_cards").glob("*.yaml")),
        root / "GRAM/src/model/gram.py",
        root / "GRAM/src/arguments.py",
        root / "GRAM/src/main_generative_gram.py",
        root / "GRAM/src/data/multi_task_dataset_gram.py",
        root / "GRAM/src/data/test_dataset_gram.py",
        root / "GRAM/src/runner/single_runner_gram.py",
    ]


def build_arm_command(
    root: Path,
    output_dir: Path,
    arm: dict,
    epochs: int,
    transition_teacher: Path,
) -> list[str]:
    command = command_template(
        root,
        track_id=arm["track_id"],
        module_id=arm["module_id"],
        dataset="Toys",
        epochs=epochs,
        output_dir=output_dir,
        transition_map=transition_teacher,
    )
    data_index = command.index("--data_path") + 1
    command[data_index] = str(root / "GRAM/rec_datasets")
    command.extend(["--rec_model_path", str(paths(root)["baseline"])])
    return command


def prepare(root: Path) -> int:
    resolved = paths(root)
    if resolved["output"].exists() or resolved["snapshot"].exists() or resolved["status"].exists():
        raise FileExistsError("S17-3 one-epoch portfolio already exists; automatic retry is forbidden")
    budget = json.loads(resolved["budget"].read_text(encoding="utf-8"))
    stage = budget["exploration_stage"]
    arms = stage["arms"]
    if int(stage["rec_epochs"]) != 1 or len(arms) != 10:
        raise ValueError("S17-3 exploration budget must freeze ten one-epoch arms")
    if arms[0]["arm_id"] != "gram_continue":
        raise ValueError("the matched GRAM continuation must be the first queued arm")
    if not resolved["baseline"].is_file() or resolved["baseline"].stat().st_size != 242132665:
        raise FileNotFoundError("frozen Phase2 Toys checkpoint is missing or has unexpected size")
    if not resolved["data_audit"].is_file():
        raise FileNotFoundError("Phase2 Toys data audit is missing")

    resolved["output"].mkdir(parents=True)
    (resolved["output"] / "preflight").mkdir()
    teacher = build_transition_teacher(
        root / "GRAM/rec_datasets/Toys",
        resolved["transition_teacher"],
    )
    worker_command = [
        str(PYTHON),
        str(root / "experiment/phase17/protocol/s3_exploration_portfolio_runtime.py"),
        "worker",
        "--root",
        str(root),
    ]
    commands = {
        arm["arm_id"]: build_arm_command(
            root,
            resolved["output"] / "arms" / arm["arm_id"],
            arm,
            int(stage["rec_epochs"]),
            resolved["transition_teacher"],
        )
        for arm in arms
    }
    config = {
        "schema_version": "phase17.s17_3_one_epoch_portfolio.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "step_id": "S17-3",
        "dataset": budget["dataset"],
        "fold": budget["fold"],
        "seed": budget["seed"],
        "arms": arms,
        "epochs_per_arm": int(stage["rec_epochs"]),
        "commands": commands,
        "worker_command": worker_command,
        "historical_baseline": budget["historical_baseline"],
        "allocated_physical_gpu": PHYSICAL_GPU,
        "researcher_allocation_confirmed": True,
        "minimum_free_mib_at_admission": budget["resources"]["minimum_free_mib_at_admission"],
        "usable_memory_ceiling_mib": budget["resources"]["usable_memory_mib_per_gpu"],
        "predicted_peak_mib": PREDICTED_PEAK_MIB,
        "arm_hard_timeout_seconds": ARM_HARD_TIMEOUT_SECONDS,
        "budget_path": str(resolved["budget"].relative_to(root)),
        "budget_sha256": sha256(resolved["budget"]),
        "baseline_checkpoint": str(resolved["baseline"].relative_to(root)),
        "baseline_checkpoint_sha256_from_phase2": budget["historical_baseline"]["checkpoint_sha256"],
        "baseline_checkpoint_size": resolved["baseline"].stat().st_size,
        "data_audit_path": str(resolved["data_audit"].relative_to(root)),
        "data_audit_sha256": sha256(resolved["data_audit"]),
        "transition_teacher": teacher,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["config"], config)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=worker_command,
        source_paths=source_paths(root, resolved["transition_teacher"]),
        config=config,
    )
    arm_states = {arm["arm_id"]: "QUEUED" for arm in arms}
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-3",
        attempt_id=ATTEMPT_ID,
        track_id="P0-PORTFOLIO",
        canonical_result_dir=str(resolved["output"].relative_to(root)),
        log_path=str(resolved["worker_log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": len(arms), "unit": "arm"},
            "current_arm": None,
            "completed_arms": [],
            "failed_arms": [],
            "arm_states": arm_states,
            "predicted_peak_mib": PREDICTED_PEAK_MIB,
            "usable_memory_ceiling_mib": config["usable_memory_ceiling_mib"],
            "minimum_free_mib_at_admission": config["minimum_free_mib_at_admission"],
            "allocated_gpu_ids": [PHYSICAL_GPU],
            "queue_mode": "serial_no_gap",
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "budget_path": str(resolved["budget"].relative_to(root)),
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_3_ONE_EPOCH_QUEUE_READY",
        process_alive=True,
    )
    print(manifest)
    return 0


def launch(root: Path) -> int:
    resolved = paths(root)
    if not resolved["config"].exists():
        prepare(root)
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    session = launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=config["worker_command"],
        cwd=root,
        tmux_session=TMUX_SESSION,
    )
    StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_3_ONE_EPOCH_QUEUE_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="cpu_contracts",
    )
    print(session)
    return 0


def fixed_gpu_record(config: dict):
    records = query_gpus()
    selected = [row for row in records if row.index == int(config["allocated_physical_gpu"])]
    if len(selected) != 1:
        raise RuntimeError("allocated physical GPU is not visible")
    return selected[0], records


def wait_for_gpu(writer: StatusWriter, config: dict, *, runtime: bool = False):
    while True:
        selected, records = fixed_gpu_record(config)
        if selected.free_mib >= int(config["minimum_free_mib_at_admission"]):
            return selected, records
        state = writer.read()
        writer.transition(
            state["scientific_state"],
            "WAITING_FOR_GPU",
            "SCIENTIFIC_COMPLETED_WAITING_FOR_RUNTIME_GPU" if runtime else "S17_3_WAITING_ALLOCATED_GPU",
            workload_pid=0,
            process_alive=True,
            gpu_ids=[],
            gpu_snapshot={"captured_at": utc_now(), "devices": snapshot(records)},
            stage="runtime_waiting_for_gpu" if runtime else "waiting_for_allocated_gpu",
        )
        time.sleep(GPU_WAIT_SECONDS)


def run_cpu_contracts(root: Path, writer: StatusWriter) -> dict:
    log_path = paths(root)["output"] / "preflight/cpu_contract_tests.log"
    command = [
        str(PYTHON), "-m", "unittest", "discover", "-v",
        "-s", "experiment/phase17/tests", "-p", "test_*.py",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=300,
            check=False,
        )
    writer.heartbeat(stage="cpu_contracts_complete")
    return {
        "return_code": completed.returncode,
        "log_path": str(log_path.relative_to(root)),
        "log_sha256": sha256(log_path),
    }


def find_checkpoint(arm_dir: Path, epochs: int) -> Path | None:
    matches = list(arm_dir.glob(f"gram_logs/**/model_rec_phase_1_epoch_{epochs}.pt"))
    return matches[0] if len(matches) == 1 else None


def arm_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root),
            "CUDA_VISIBLE_DEVICES": str(PHYSICAL_GPU),
            "HF_HUB_CACHE": str(root / ".cache/huggingface"),
            "TRANSFORMERS_CACHE": str(root / ".cache/huggingface/transformers"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def run_process(
    *,
    command: list[str],
    log_path: Path,
    root: Path,
    writer: StatusWriter,
    config: dict,
    arm: dict,
    index: int,
    completed_arms: list[str],
    failed_arms: list[str],
    arm_states: dict[str, str],
    records: list,
    admission_free_mib: int,
) -> tuple[int, int, float]:
    started = time.monotonic()
    last_size = -1
    unchanged_checks = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root / "GRAM/command",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=arm_environment(root),
            start_new_session=True,
        )
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_3_ONE_EPOCH_ARM_RUNNING",
            workload_pid=process.pid,
            process_alive=True,
            gpu_ids=[PHYSICAL_GPU],
            gpu_snapshot={
                "captured_at": utc_now(),
                "devices": snapshot(records),
                "selected_gpu": PHYSICAL_GPU,
                "admission_free_mib": admission_free_mib,
                "selection_reason": "researcher-authorized serial queue on GPU1",
            },
            stage=f"arm_{index + 1}_of_{len(config['arms'])}_training",
            progress={"current": index, "total": len(config["arms"]), "unit": "arm"},
            current_arm=arm["arm_id"],
            current_arm_log=str(log_path.relative_to(root)),
            completed_arms=completed_arms,
            failed_arms=failed_arms,
            arm_states=arm_states,
        )
        while True:
            try:
                return_code = process.wait(timeout=MONITOR_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                size = log_path.stat().st_size
                unchanged_checks = unchanged_checks + 1 if size == last_size else 0
                last_size = size
                stalled = unchanged_checks >= STALL_CHECKS
                writer.transition(
                    "RUNNING",
                    "RUNNING_SCIENTIFIC",
                    "S17_3_ARM_OUTPUT_STALL_SUSPECTED" if stalled else "S17_3_ONE_EPOCH_ARM_RUNNING",
                    workload_pid=process.pid,
                    process_alive=True,
                    stage=f"arm_{index + 1}_of_{len(config['arms'])}_{'stall_advisory' if stalled else 'active'}",
                    progress={
                        "current": index,
                        "total": len(config["arms"]),
                        "unit": "arm",
                        "current_arm": arm["arm_id"],
                        "elapsed_seconds": round(elapsed, 1),
                    },
                )
                if elapsed > int(config["arm_hard_timeout_seconds"]):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return_code = 124
                    break
    return return_code, process.pid, time.monotonic() - started


def run_arm(
    root: Path,
    writer: StatusWriter,
    config: dict,
    arm: dict,
    index: int,
    completed_arms: list[str],
    failed_arms: list[str],
    arm_states: dict[str, str],
) -> dict:
    resolved = paths(root)
    arm_dir = resolved["output"] / "arms" / arm["arm_id"]
    if arm_dir.exists():
        raise FileExistsError(f"canonical arm output already exists: {arm_dir}")
    arm_dir.mkdir(parents=True)
    command = config["commands"][arm["arm_id"]]
    arm_config_path = arm_dir / "config.json"
    arm_config = {
        "schema_version": "phase17.s17_3_one_epoch_arm.v1",
        "attempt_id": f"s3_{arm['arm_id']}_001",
        **arm,
        "epochs": config["epochs_per_arm"],
        "seed": config["seed"],
        "command": command,
        "baseline_checkpoint": config["baseline_checkpoint"],
        "historical_validation": config["historical_baseline"]["validation_metrics"],
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(arm_config_path, arm_config)
    arm_states[arm["arm_id"]] = "WAITING_FOR_GPU"
    selected, records = wait_for_gpu(writer, config)
    arm_states[arm["arm_id"]] = "RUNNING"
    started_at = utc_now()
    return_code, pid, wall_seconds = run_process(
        command=command,
        log_path=arm_dir / "run.log",
        root=root,
        writer=writer,
        config=config,
        arm=arm,
        index=index,
        completed_arms=completed_arms,
        failed_arms=failed_arms,
        arm_states=arm_states,
        records=records,
        admission_free_mib=selected.free_mib,
    )
    parsed = parse_log(arm_dir / "run.log")
    checkpoint = find_checkpoint(arm_dir, int(config["epochs_per_arm"]))
    checks = {
        "exit_zero": return_code == 0,
        "no_traceback": not parsed["traceback"],
        "no_forbidden_test_evidence": not parsed["forbidden_test_evidence"],
        "one_training_epoch": len(parsed["training_losses"]) == 1,
        "validation_completed": bool(parsed["validation_metrics"]),
        "mechanism_metric_present": arm["module_id"] == "" or bool(parsed["mechanism_metric_lines"]),
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
        "workload_pid": pid,
        "physical_gpu": PHYSICAL_GPU,
        "admission_free_mib": selected.free_mib,
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
    }
    atomic_json(arm_dir / "result.json", result)
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": result["attempt_id"],
            "step_id": "S17-3",
            "track_id": arm["track_id"],
            "kind": "one_epoch_exploration",
            "started_at": result["started_at"],
            "ended_at": result["ended_at"],
            "state": result["state"],
            "config_sha256": sha256(arm_config_path),
            "data_manifest_sha256": config["data_audit_sha256"],
            "source_sha256": sha256(resolved["snapshot"]),
            "scientific_result_eligible": passed,
            "failure_reason": None if passed else "one-epoch arm completion contract failed",
            "artifact_dir": str(arm_dir.relative_to(root)),
            "physical_gpu": PHYSICAL_GPU,
        }
    )
    return result


def runtime_loop(root: Path, writer: StatusWriter, config: dict, source_arm: dict) -> None:
    canonical = paths(root)["output"]
    iteration = 2
    while True:
        runtime_dir = isolated_runtime_dir(root, EXPERIMENT_ID, iteration)
        assert_runtime_isolation(canonical, runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=False)
        selected, records = wait_for_gpu(writer, config, runtime=True)
        command = build_arm_command(
            root,
            runtime_dir,
            source_arm,
            int(config["epochs_per_arm"]),
            paths(root)["transition_teacher"],
        )
        atomic_json(
            runtime_dir / "runtime_config.json",
            {
                "schema_version": "phase17.runtime_cycle.v1",
                "source_arm": source_arm["arm_id"],
                "iteration": iteration,
                "command": command,
                "result_selection_eligible": False,
                "affects_scientific_result": False,
                "test_read": False,
                "sports_read": False,
            },
        )
        writer.start_runtime_cycle(
            iteration=iteration,
            runtime_result_dir=str(runtime_dir.relative_to(root)),
            workload_pid=0,
        )
        with (runtime_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=root / "GRAM/command",
                stdout=log,
                stderr=subprocess.STDOUT,
                env=arm_environment(root),
                start_new_session=True,
            )
            writer.start_runtime_cycle(
                iteration=iteration,
                runtime_result_dir=str(runtime_dir.relative_to(root)),
                workload_pid=process.pid,
            )
            started = time.monotonic()
            while True:
                try:
                    return_code = process.wait(timeout=MONITOR_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    writer.heartbeat(
                        stage="runtime_cycle_active",
                        progress={
                            "current": len(config["arms"]),
                            "total": len(config["arms"]),
                            "unit": "arm",
                            "runtime_iteration": iteration,
                            "source_arm": source_arm["arm_id"],
                            "elapsed_seconds": round(elapsed, 1),
                        },
                    )
                    if elapsed > int(config["arm_hard_timeout_seconds"]):
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        return_code = 124
                        break
        if return_code != 0:
            writer.transition(
                "COMPLETED",
                "SCIENTIFIC_COMPLETED",
                "SCIENTIFIC_COMPLETED_RUNTIME_CYCLE_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="runtime_cycle_failed",
                occupancy_mode="stopped_after_runtime_failure",
                repeat_metrics_ignored=True,
                result_selection_eligible=False,
                affects_scientific_result=False,
                runtime_return_code=return_code,
            )
            return
        iteration += 1


def worker(root: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    try:
        verify_run_snapshot(root, resolved["snapshot"])
        cpu = run_cpu_contracts(root, writer)
        if cpu["return_code"] != 0:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_3_CPU_CONTRACTS_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="failed_cpu_contracts",
            )
            return cpu["return_code"]
        completed_arms: list[str] = []
        failed_arms: list[str] = []
        arm_states = {arm["arm_id"]: "QUEUED" for arm in config["arms"]}
        results = []
        for index, arm in enumerate(config["arms"]):
            try:
                result = run_arm(
                    root,
                    writer,
                    config,
                    arm,
                    index,
                    completed_arms,
                    failed_arms,
                    arm_states,
                )
            except Exception as error:
                result = {
                    "schema_version": "phase17.s17_3_one_epoch_result.v1",
                    "attempt_id": f"s3_{arm['arm_id']}_001",
                    **arm,
                    "state": "FAILED",
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "test_read": False,
                    "sports_read": False,
                }
                arm_dir = resolved["output"] / "arms" / arm["arm_id"]
                arm_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(arm_dir / "result.json", result)
            results.append(result)
            if result["state"] == "COMPLETED":
                completed_arms.append(arm["arm_id"])
                arm_states[arm["arm_id"]] = "COMPLETED"
            else:
                failed_arms.append(arm["arm_id"])
                arm_states[arm["arm_id"]] = "FAILED"
            writer.transition(
                "RUNNING",
                "RUNNING_SCIENTIFIC",
                "S17_3_ARM_FINISHED_QUEUE_ADVANCING",
                workload_pid=0,
                process_alive=True,
                stage=f"arm_{index + 1}_complete_queue_advancing",
                progress={"current": index + 1, "total": len(config["arms"]), "unit": "arm"},
                current_arm=None,
                completed_arms=completed_arms,
                failed_arms=failed_arms,
                arm_states=arm_states,
            )
        summary = {
            "schema_version": "phase17.s17_3_one_epoch_portfolio_summary.v1",
            "experiment_id": EXPERIMENT_ID,
            "step_id": "S17-3",
            "substep": "one_epoch_exploration_screen",
            "queue_completed": True,
            "all_arms_passed": not failed_arms,
            "completed_arms": completed_arms,
            "failed_arms": failed_arms,
            "cpu_contracts": cpu,
            "historical_baseline": config["historical_baseline"],
            "results": results,
            "official_result_claim": False,
            "test_read": False,
            "sports_read": False,
            "completed_at": utc_now(),
        }
        atomic_json(resolved["summary"], summary)
        scientific_state = "COMPLETED" if completed_arms else "FAILED"
        writer.transition(
            scientific_state,
            "SCIENTIFIC_COMPLETED" if completed_arms else "SCIENTIFIC_FAILED",
            "S17_3_ONE_EPOCH_QUEUE_COMPLETE" if not failed_arms else "S17_3_ONE_EPOCH_QUEUE_COMPLETE_WITH_FAILURES",
            workload_pid=0,
            process_alive=False,
            stage="scientific_queue_complete",
            progress={"current": len(config["arms"]), "total": len(config["arms"]), "unit": "arm"},
            current_arm=None,
            completed_arms=completed_arms,
            failed_arms=failed_arms,
            arm_states=arm_states,
            result_selection_eligible=bool(completed_arms),
            affects_scientific_result=True,
            summary_path=str(resolved["summary"].relative_to(root)),
        )
        if completed_arms:
            source = next(arm for arm in reversed(config["arms"]) if arm["arm_id"] in completed_arms)
            runtime_loop(root, writer, config, source)
        return 0 if completed_arms else 1
    except Exception as error:
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_3_PORTFOLIO_ORCHESTRATOR_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="orchestrator_failed",
                failure_reason=f"{type(error).__name__}: {error}",
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "launch", "worker", "status"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root)
    if args.action == "launch":
        return launch(root)
    if args.action == "worker":
        worker_log = paths(root)["worker_log"]
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = worker_log.open("a", encoding="utf-8", buffering=1)
        os.dup2(log_handle.fileno(), sys.stdout.fileno())
        os.dup2(log_handle.fileno(), sys.stderr.fileno())
        return worker(root)
    print(paths(root)["status"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
