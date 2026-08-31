#!/usr/bin/env python3
"""Run the deferred paper-grade S17-3 GRAM-B0 confirmation parent."""

from __future__ import annotations

import argparse
import json
import os
import re
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
from experiment.phase17.protocol.s2_probe_runtime import parse_log  # noqa: E402


EXPERIMENT_ID = "s17_s3_gram_b0_parent_seed2023"
ATTEMPT_ID = "run-0001"
TMUX_SESSION = EXPERIMENT_ID
PHYSICAL_GPU = 1
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
GPU_WAIT_SECONDS = 30
MONITOR_SECONDS = 30
HARD_TIMEOUT_SECONDS = 40 * 60 * 60
STATIC_INPUT_FILES = (
    "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
    "item_plain_text.txt",
    "similar_item_sasrec.txt",
)


def paths(root: Path) -> dict[str, Path]:
    output = root / "artifacts/phase17/s3_formal/gram_b0_parent" / ATTEMPT_ID
    return {
        "output": output,
        "status_dir": root / "artifacts/phase17/status",
        "status": root / "artifacts/phase17/status" / f"{EXPERIMENT_ID}.status.json",
        "ledger": root / "artifacts/phase17/attempts/S17-3.attempts.jsonl",
        "summary": output / "summary.json",
        "config": output / "parent_config.json",
        "worker_log": output / "orchestrator.log",
        "run_log": output / "run.log",
        "snapshot": root / "artifacts/phase17/snapshots" / EXPERIMENT_ID / ATTEMPT_ID / "manifest.json",
        "budget": root / "experiment/phase17/config/s17_s3_confirmation_budget.json",
        "input_root": root / "artifacts/phase17/s3_formal/input_data",
        "input_dataset": root / "artifacts/phase17/s3_formal/input_data/Toys_s17_d0",
        "input_manifest": output / "preflight/input_manifest.json",
    }


def source_paths(root: Path, input_manifest: Path) -> list[Path]:
    return [
        root / "experiment/phase17/protocol/s3_formal_parent_runtime.py",
        root / "experiment/phase17/protocol/s2_probe_runtime.py",
        root / "experiment/phase17/run_stage17_s3_formal_parent_gpu1.sh",
        root / "experiment/phase17/config/s17_s3_confirmation_budget.json",
        input_manifest,
        *sorted((root / "experiment/phase17/core").glob("*.py")),
        root / "experiment/phase17/registry/module_registry.py",
        root / "GRAM/src/model/gram.py",
        root / "GRAM/src/arguments.py",
        root / "GRAM/src/main_generative_gram.py",
        root / "GRAM/src/data/multi_task_dataset_gram.py",
        root / "GRAM/src/data/test_dataset_gram.py",
        root / "GRAM/src/runner/single_runner_gram.py",
    ]


def prepare_input_view(root: Path, manifest_path: Path) -> dict:
    resolved = paths(root)
    dataset_dir = resolved["input_dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "user_sequence.txt": root / "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt",
        **{
            name: root / "artifacts/phase17/s0_audit/profile_data/Toys_s17_d0_1000" / name
            for name in STATIC_INPUT_FILES
        },
    }
    records = []
    for name, source in sources.items():
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"formal input is missing: {source}")
        target = dataset_dir / name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                raise FileExistsError(f"formal input view drift: {target}")
        else:
            target.symlink_to(source)
        records.append(
            {
                "name": name,
                "source_path": str(source.relative_to(root)),
                "view_path": str(target.relative_to(root)),
                "sha256": sha256(source),
            }
        )
    sequence = sources["user_sequence.txt"]
    user_count = sum(1 for line in sequence.open(encoding="utf-8") if line.strip())
    if user_count != 12833:
        raise ValueError(f"Toys D0 formal user count drift: {user_count}")
    shadow_manifest = json.loads(
        (root / "artifacts/phase17/s0_audit/shadow_data_manifest.json").read_text()
    )
    expected_sequence_sha = shadow_manifest["domains"]["Toys"]["folds"]["D0"][
        "output_sha256"
    ]
    if sha256(sequence) != expected_sequence_sha:
        raise ValueError("Toys D0 sequence differs from the frozen shadow manifest")
    payload = {
        "schema_version": "phase17.s17_3_formal_input.v1",
        "dataset": "Toys_s17_d0",
        "fold": "D0-discovery",
        "users": user_count,
        "files": records,
        "official_validation_position_serialized": False,
        "official_test_position_serialized": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(manifest_path, payload)
    return payload


def build_command(root: Path, output_dir: Path, epochs: int) -> list[str]:
    input_root = paths(root)["input_root"]
    return [
        str(PYTHON),
        "../src/main_generative_gram.py",
        "--data_path", str(input_root),
        "--datasets", "Toys_s17_d0",
        "--distributed", "0",
        "--gpu", "0",
        "--seed", "2023",
        "--train", "1",
        "--resource_metrics", "1",
        "--log_dir", str(output_dir / "gram_logs"),
        "--prediction_dir", str(output_dir / "predictions"),
        "--item_prompt_max_len", "128",
        "--item_prompt", "all_text",
        "--cf_model", "sasrec",
        "--id_linking", "1",
        "--max_his", "20",
        "--rec_batch_size", "16",
        "--gradient_accumulation_steps", "8",
        "--rec_lr", "1e-3",
        "--rec_epochs", str(epochs),
        "--test_epoch_rec", "0",
        "--save_rec_epochs", str(epochs),
        "--save_predictions", "0",
        "--beam_size", "50",
        "--top_k_similar_item", "5",
        "--item_id_type", "split",
        "--hierarchical_id_type", "hierarchy_v1_c32_l5_len32768_split",
        "--debug_train_100", "0",
        "--debug_test_100", "0",
        "--cf0_arm", "A",
        "--cf0_phase9", "1",
        "--hi_gram_enabled", "0",
        "--s17_modules", "",
    ]


def prepare(root: Path) -> int:
    resolved = paths(root)
    if resolved["output"].exists() or resolved["snapshot"].exists() or resolved["status"].exists():
        raise FileExistsError("S17-3 formal parent already exists; automatic retry is forbidden")
    resolved["output"].mkdir(parents=True)
    (resolved["output"] / "preflight").mkdir()
    input_manifest = prepare_input_view(root, resolved["input_manifest"])
    budget = json.loads(resolved["budget"].read_text(encoding="utf-8"))
    parent = budget["parent_stage"]
    command = build_command(root, resolved["output"], int(parent["rec_epochs"]))
    worker_command = [
        str(PYTHON),
        str(root / "experiment/phase17/protocol/s3_formal_parent_runtime.py"),
        "worker",
        "--root",
        str(root),
    ]
    config = {
        "schema_version": "phase17.s17_3_formal_parent.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "step_id": "S17-3",
        "arm": parent,
        "dataset": budget["dataset"],
        "fold": budget["fold"],
        "seed": budget["seed"],
        "command": command,
        "worker_command": worker_command,
        "allocated_physical_gpu": PHYSICAL_GPU,
        "researcher_allocation_confirmed": True,
        "minimum_free_mib_at_admission": budget["resources"]["minimum_free_mib_at_admission"],
        "usable_memory_ceiling_mib": budget["resources"]["usable_memory_mib_per_gpu"],
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "budget_path": str(resolved["budget"].relative_to(root)),
        "budget_sha256": sha256(resolved["budget"]),
        "input_manifest": input_manifest,
        "input_manifest_path": str(resolved["input_manifest"].relative_to(root)),
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(resolved["config"], config)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=worker_command,
        source_paths=source_paths(root, resolved["input_manifest"]),
        config=config,
    )
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-3",
        attempt_id=ATTEMPT_ID,
        track_id="GRAM-B0",
        canonical_result_dir=str(resolved["output"].relative_to(root)),
        log_path=str(resolved["run_log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": int(parent["rec_epochs"]), "unit": "epoch"},
            "predicted_peak_mib": 25008,
            "usable_memory_ceiling_mib": config["usable_memory_ceiling_mib"],
            "minimum_free_mib_at_admission": config["minimum_free_mib_at_admission"],
            "allocated_gpu_ids": [PHYSICAL_GPU],
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "budget_path": str(resolved["budget"].relative_to(root)),
            "input_manifest_path": str(resolved["input_manifest"].relative_to(root)),
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_3_PARENT_PREFLIGHT_COMPLETE",
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
        "S17_3_PARENT_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="cpu_and_data_contracts",
    )
    print(session)
    return 0


def fixed_gpu_record(config: dict):
    records = query_gpus()
    matches = [row for row in records if row.index == int(config["allocated_physical_gpu"])]
    if len(matches) != 1:
        raise RuntimeError("researcher-allocated physical GPU is not visible")
    return matches[0], records


def wait_for_allocated_gpu(writer: StatusWriter, config: dict, *, runtime: bool = False):
    minimum = int(config["minimum_free_mib_at_admission"])
    while True:
        selected, records = fixed_gpu_record(config)
        if selected.free_mib >= minimum:
            return selected, records
        scientific_state = "COMPLETED" if runtime else "RUNNING"
        writer.transition(
            scientific_state,
            "WAITING_FOR_GPU",
            (
                "SCIENTIFIC_COMPLETED_WAITING_FOR_RUNTIME_GPU"
                if runtime
                else "S17_3_PARENT_WAITING_ALLOCATED_GPU"
            ),
            workload_pid=0,
            process_alive=True,
            gpu_ids=[],
            gpu_snapshot={"captured_at": utc_now(), "devices": snapshot(records)},
            stage="runtime_waiting_for_gpu" if runtime else "waiting_for_allocated_gpu",
        )
        time.sleep(GPU_WAIT_SECONDS)


def run_cpu_contracts(root: Path, writer: StatusWriter) -> dict:
    resolved = paths(root)
    log_path = resolved["output"] / "preflight/cpu_contract_tests.log"
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
    writer.heartbeat(stage="cpu_and_data_contracts_complete")
    return {
        "return_code": completed.returncode,
        "log_path": str(log_path.relative_to(root)),
        "log_sha256": sha256(log_path),
    }


def latest_completed_epoch(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 512 * 1024))
        text = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r"average training loss for rec phase 1 epoch (\d+) is", text)
    return max((int(value) for value in matches), default=0)


def run_canonical_process(
    root: Path, writer: StatusWriter, config: dict, records: list, admission_free_mib: int
) -> tuple[int, int, float]:
    resolved = paths(root)
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
    started = time.monotonic()
    last_size = -1
    unchanged_checks = 0
    with resolved["run_log"].open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            config["command"],
            cwd=root / "GRAM/command",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_3_PARENT_TRAINING",
            workload_pid=process.pid,
            process_alive=True,
            gpu_ids=[PHYSICAL_GPU],
            gpu_snapshot={
                "captured_at": utc_now(),
                "devices": snapshot(records),
                "selected_gpu": PHYSICAL_GPU,
                "admission_free_mib": admission_free_mib,
                "selection_reason": "researcher-allocated fixed GPU1",
            },
            stage="training",
        )
        while True:
            try:
                return_code = process.wait(timeout=MONITOR_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                completed_epoch = latest_completed_epoch(resolved["run_log"])
                size = resolved["run_log"].stat().st_size
                unchanged_checks = unchanged_checks + 1 if size == last_size else 0
                last_size = size
                status_code = (
                    "S17_3_PARENT_STALL_SUSPECTED"
                    if unchanged_checks >= 3
                    else "S17_3_PARENT_TRAINING"
                )
                writer.transition(
                    "RUNNING",
                    "RUNNING_SCIENTIFIC",
                    status_code,
                    workload_pid=process.pid,
                    process_alive=True,
                    heartbeat_at=utc_now(),
                    stage=("stall_suspected" if unchanged_checks >= 3 else f"epoch_{completed_epoch + 1}"),
                    progress={
                        "current": completed_epoch,
                        "total": int(config["arm"]["rec_epochs"]),
                        "unit": "epoch",
                        "elapsed_seconds": round(elapsed, 1),
                    },
                )
                if elapsed > int(config["hard_timeout_seconds"]):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return_code = 124
                    break
    return return_code, process.pid, time.monotonic() - started


def find_parent_checkpoint(output: Path, epochs: int) -> Path:
    matches = list(output.glob(f"gram_logs/**/model_rec_phase_1_epoch_{epochs}.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one parent checkpoint, found {len(matches)}")
    return matches[0]


def append_attempt(root: Path, result: dict, config: dict) -> None:
    resolved = paths(root)
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": "s3_parent_b0_001",
            "step_id": "S17-3",
            "track_id": "GRAM-B0",
            "kind": "formal_parent",
            "started_at": result["started_at"],
            "ended_at": utc_now(),
            "state": result["state"],
            "config_sha256": sha256(resolved["config"]),
            "data_manifest_sha256": sha256(resolved["input_manifest"]),
            "source_sha256": sha256(resolved["snapshot"]),
            "scientific_result_eligible": result["state"] == "COMPLETED",
            "failure_reason": result.get("failure_reason"),
            "artifact_dir": str(resolved["output"].relative_to(root)),
            "physical_gpu": config["allocated_physical_gpu"],
        }
    )


def runtime_loop(root: Path, writer: StatusWriter, config: dict) -> None:
    canonical = paths(root)["output"]
    iteration = 2
    while True:
        runtime_dir = isolated_runtime_dir(root, EXPERIMENT_ID, iteration)
        assert_runtime_isolation(canonical, runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=False)
        selected, records = wait_for_allocated_gpu(writer, config, runtime=True)
        command = build_command(root, runtime_dir, int(config["arm"]["rec_epochs"]))
        atomic_json(
            runtime_dir / "runtime_config.json",
            {
                "schema_version": "phase17.runtime_cycle.v1",
                "source_arm": "gram_b0_parent",
                "iteration": iteration,
                "command": command,
                "result_selection_eligible": False,
                "affects_scientific_result": False,
                "test_read": False,
                "sports_read": False,
            },
        )
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
                env=env,
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
                            "current": latest_completed_epoch(runtime_dir / "run.log"),
                            "total": int(config["arm"]["rec_epochs"]),
                            "unit": "epoch",
                            "runtime_iteration": iteration,
                            "elapsed_seconds": round(elapsed, 1),
                        },
                    )
                    if elapsed > int(config["hard_timeout_seconds"]):
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
    started_at = utc_now()
    try:
        verify_run_snapshot(root, resolved["snapshot"])
        cpu = run_cpu_contracts(root, writer)
        if cpu["return_code"] != 0:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_3_PARENT_CPU_CONTRACTS_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="failed_cpu_contracts",
            )
            return cpu["return_code"]
        selected, records = wait_for_allocated_gpu(writer, config)
        return_code, pid, wall_seconds = run_canonical_process(
            root, writer, config, records, selected.free_mib
        )
        parsed = parse_log(resolved["run_log"])
        checkpoint = None
        checkpoint_error = None
        try:
            checkpoint = find_parent_checkpoint(
                resolved["output"], int(config["arm"]["rec_epochs"])
            )
        except FileNotFoundError as error:
            checkpoint_error = str(error)
        passed = (
            return_code == 0
            and not parsed["traceback"]
            and not parsed["forbidden_test_evidence"]
            and len(parsed["training_losses"]) == int(config["arm"]["rec_epochs"])
            and bool(parsed["validation_metrics"])
            and parsed["peak_reserved_mib"] is not None
            and parsed["peak_reserved_mib"] <= int(config["usable_memory_ceiling_mib"])
            and checkpoint is not None
        )
        result = {
            "schema_version": "phase17.s17_3_formal_parent_result.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": "s3_parent_b0_001",
            "step_id": "S17-3",
            "arm_id": "gram_b0_parent",
            "state": "COMPLETED" if passed else "FAILED",
            "started_at": started_at,
            "return_code": return_code,
            "workload_pid": pid,
            "physical_gpu": PHYSICAL_GPU,
            "admission_free_mib": selected.free_mib,
            "wall_seconds": wall_seconds,
            "cpu_contracts": cpu,
            "parsed": parsed,
            "parent_checkpoint": str(checkpoint.relative_to(root)) if checkpoint else None,
            "parent_checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
            "checkpoint_error": checkpoint_error,
            "failure_reason": None if passed else "formal parent completion contract failed",
            "test_read": False,
            "sports_read": False,
        }
        atomic_json(resolved["summary"], result)
        append_attempt(root, result, config)
        writer.transition(
            "COMPLETED" if passed else "FAILED",
            "SCIENTIFIC_COMPLETED" if passed else "SCIENTIFIC_FAILED",
            "S17_3_PARENT_COMPLETE" if passed else "S17_3_PARENT_FAILED",
            workload_pid=0,
            process_alive=False,
            stage="scientific_complete" if passed else "scientific_failed",
            progress={
                "current": len(parsed["training_losses"]),
                "total": int(config["arm"]["rec_epochs"]),
                "unit": "epoch",
            },
            result_selection_eligible=passed,
            affects_scientific_result=passed,
            summary_path=str(resolved["summary"].relative_to(root)),
            parent_checkpoint=result["parent_checkpoint"],
        )
        if passed:
            runtime_loop(root, writer, config)
        return 0 if passed else 1
    except Exception as error:
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_3_PARENT_ORCHESTRATOR_FAILED",
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
