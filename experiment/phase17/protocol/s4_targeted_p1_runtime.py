#!/usr/bin/env python3
"""Run the diagnosis-triggered S17-4 P1 screen and GPU1 runtime handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


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


EXPERIMENT_ID = "s17_s4_p1_targeted_d0_seed2023"
ATTEMPT_ID = "run-0001"
TMUX_SESSION = EXPERIMENT_ID
RUNTIME_RECOVERY_SESSION = f"{EXPERIMENT_ID}_runtime_r1"
S3_EXPERIMENT_ID = "s17_s3_one_epoch_portfolio_seed2023"
GPU1 = 1
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
GPU_WAIT_SECONDS = 30
SMOKE_TIMEOUT_SECONDS = 30 * 60


def paths(root: Path) -> dict[str, Path]:
    output = root / "artifacts/phase17/s4_p1_targeted" / ATTEMPT_ID
    return {
        "output": output,
        "status_dir": root / "artifacts/phase17/status",
        "status": root / "artifacts/phase17/status" / f"{EXPERIMENT_ID}.status.json",
        "ledger": root / "artifacts/phase17/attempts/S17-4.attempts.jsonl",
        "summary": output / "summary.json",
        "config": output / "portfolio_config.json",
        "worker_log": output / "portfolio.log",
        "snapshot": root / "artifacts/phase17/snapshots" / EXPERIMENT_ID / ATTEMPT_ID / "manifest.json",
        "budget": root / "experiment/phase17/config/s17_s4_targeted_budget.json",
        "data_root": output / "preflight/data",
        "dataset": output / "preflight/data/Toys_s17_d0_full",
        "data_manifest": output / "preflight/d0_dataset_manifest.json",
        "transition_teacher": output / "preflight/transition_teacher.json",
        "baseline": root / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt",
        "handoff": output / "preflight/gpu1_handoff.json",
        "report": root / "report/第十七阶段/Stage17_S4_P1定向迁移筛选报告.md",
        "canonical_manifest": root / "artifacts/phase17/manifests" / f"{EXPERIMENT_ID}.{ATTEMPT_ID}.canonical_results.json",
    }


def prepare_shadow_dataset(root: Path, target: Path, manifest_path: Path) -> dict[str, Any]:
    source_shadow = root / "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt"
    catalog_root = root / "GRAM/rec_datasets/Toys"
    if target.exists():
        raise FileExistsError(f"S17-4 dataset view already exists: {target}")
    target.mkdir(parents=True)
    copied = []
    sources = [source_shadow] + [
        catalog_root / name
        for name in (
            "item_plain_text.txt",
            "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
            "similar_item_sasrec.txt",
        )
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = target / source.name
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source.relative_to(root)),
                "destination": str(destination.relative_to(root)),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    shadow_manifest = json.loads(
        (root / "artifacts/phase17/s0_audit/shadow_data_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected = shadow_manifest["domains"]["Toys"]["folds"]["D0"]
    if copied[0]["sha256"] != expected["output_sha256"]:
        raise RuntimeError("copied D0 shadow sequence does not match the frozen manifest")
    payload = {
        "schema_version": "phase17.s17_4_d0_dataset.v1",
        "dataset": "Toys_s17_d0_full",
        "fold": "D0",
        "source_shadow_manifest": "artifacts/phase17/s0_audit/shadow_data_manifest.json",
        "source_shadow_manifest_sha256": sha256(
            root / "artifacts/phase17/s0_audit/shadow_data_manifest.json"
        ),
        "expected_users": int(expected["output_users"]),
        "official_validation_position_serialized": False,
        "official_test_position_serialized": False,
        "sports_read": False,
        "files": copied,
    }
    atomic_json(manifest_path, payload)
    return payload


def build_command(
    root: Path,
    *,
    arm: dict[str, Any],
    dataset: str,
    data_root: Path,
    output_dir: Path,
    epochs: int,
    save_predictions: bool,
    transition_teacher: Path,
) -> list[str]:
    command = command_template(
        root,
        track_id=arm["track_id"],
        module_id=arm["module_id"],
        dataset=dataset,
        epochs=epochs,
        output_dir=output_dir,
        transition_map=transition_teacher,
    )
    command[command.index("--data_path") + 1] = str(data_root)
    command[command.index("--save_predictions") + 1] = "1" if save_predictions else "0"
    command.extend(["--rec_model_path", str(paths(root)["baseline"])])
    return command


def source_paths(root: Path) -> list[Path]:
    resolved = paths(root)
    return [
        root / "experiment/phase17/protocol/s4_targeted_p1_runtime.py",
        root / "experiment/phase17/protocol/s2_probe_runtime.py",
        root / "experiment/phase17/run_stage17_s4_targeted_d0.sh",
        resolved["budget"],
        resolved["data_manifest"],
        root / "artifacts/phase17/s0_audit/shadow_data_manifest.json",
        *sorted((root / "experiment/phase17/core").glob("*.py")),
        root / "experiment/phase17/registry/module_registry.py",
        *sorted((root / "experiment/phase17/registry/migration_cards").glob("*.yaml")),
        root / "experiment/phase17/tests/test_p1_modules.py",
        root / "experiment/phase17/tests/test_s4_p1_contract.py",
        root / "GRAM/src/model/gram.py",
        root / "GRAM/src/arguments.py",
        root / "GRAM/src/main_generative_gram.py",
        root / "GRAM/src/data/multi_task_dataset_gram.py",
        root / "GRAM/src/data/test_dataset_gram.py",
        root / "GRAM/src/runner/single_runner_gram.py",
    ]


def _baseline_path(root: Path) -> Path:
    return root / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt"


def prepare(root: Path) -> int:
    resolved = paths(root)
    resolved["baseline"] = _baseline_path(root)
    if resolved["output"].exists() or resolved["snapshot"].exists() or resolved["status"].exists():
        raise FileExistsError("S17-4 canonical run already exists; automatic retry is forbidden")
    budget = json.loads(resolved["budget"].read_text(encoding="utf-8"))
    if [arm["arm_id"] for arm in budget["formal_arms"]] != [
        "gram_continue",
        "pawa_lite",
        "latte_sethead",
        "biflow_s2g",
    ]:
        raise ValueError("S17-4 formal arm order drifted from the frozen targeted screen")
    if budget["test_read"] or budget["sports_read"]:
        raise PermissionError("S17-4 cannot read official test or Sports")
    baseline = resolved["baseline"]
    parent = budget["parent"]
    if not baseline.is_file() or baseline.stat().st_size != int(parent["size_bytes"]):
        raise FileNotFoundError("frozen Phase2 parent checkpoint is missing or has unexpected size")
    if sha256(baseline) != parent["checkpoint_sha256"]:
        raise RuntimeError("frozen Phase2 parent checkpoint hash mismatch")

    resolved["output"].mkdir(parents=True)
    (resolved["output"] / "preflight").mkdir()
    data_manifest = prepare_shadow_dataset(
        root, resolved["dataset"], resolved["data_manifest"]
    )
    teacher = build_transition_teacher(resolved["dataset"], resolved["transition_teacher"])
    worker_command = [
        str(PYTHON),
        str(root / "experiment/phase17/protocol/s4_targeted_p1_runtime.py"),
        "worker",
        "--root",
        str(root),
    ]
    commands = {
        arm["arm_id"]: build_command(
            root,
            arm=arm,
            dataset=budget["dataset"],
            data_root=resolved["data_root"],
            output_dir=resolved["output"] / "arms" / arm["arm_id"],
            epochs=int(budget["rec_epochs"]),
            save_predictions=True,
            transition_teacher=resolved["transition_teacher"],
        )
        for arm in budget["formal_arms"]
    }
    config = {
        **budget,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "worker_command": worker_command,
        "commands": commands,
        "budget_path": str(resolved["budget"].relative_to(root)),
        "budget_sha256": sha256(resolved["budget"]),
        "data_manifest": data_manifest,
        "data_manifest_path": str(resolved["data_manifest"].relative_to(root)),
        "transition_teacher": teacher,
        "baseline_checkpoint": str(baseline.relative_to(root)),
    }
    atomic_json(resolved["config"], config)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=worker_command,
        source_paths=source_paths(root),
        config=config,
    )
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-4",
        attempt_id=ATTEMPT_ID,
        track_id="P1-TARGETED",
        canonical_result_dir=str(resolved["output"].relative_to(root)),
        log_path=str(resolved["worker_log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": len(budget["formal_arms"]), "unit": "formal_arm"},
            "arm_states": {arm["arm_id"]: "QUEUED" for arm in budget["formal_arms"]},
            "smoke_states": {arm["arm_id"]: "QUEUED" for arm in budget["smoke"]["arms"]},
            "predicted_peak_mib": budget["resources"]["expected_peak_mib"],
            "usable_memory_ceiling_mib": budget["resources"]["usable_memory_ceiling_mib"],
            "allocated_gpu_ids": [GPU1],
            "additional_idle_gpus_allowed": True,
            "post_science_gpu_ids": [GPU1],
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "budget_path": str(resolved["budget"].relative_to(root)),
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_4_TARGETED_PREFLIGHT_COMPLETE",
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
        "S17_4_TARGETED_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="cpu_contracts",
    )
    print(session)
    return 0


def environment(root: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_CACHE": str(root / ".cache/huggingface"),
            "TRANSFORMERS_CACHE": str(root / ".cache/huggingface/transformers"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def eligible_extra_gpus(config: dict[str, Any], busy: set[int]) -> tuple[list[int], list[dict]]:
    records = query_gpus()
    needed = int(config["resources"]["expected_peak_mib"]) + int(
        config["resources"]["extra_gpu_safety_margin_mib"]
    )
    eligible = [
        row
        for row in records
        if row.index != GPU1 and row.index not in busy and row.free_mib >= needed
    ]
    eligible.sort(key=lambda row: (row.utilization_percent, -row.free_mib, row.index))
    return [row.index for row in eligible], snapshot(records)


def wait_for_smoke_gpu(writer: StatusWriter, config: dict[str, Any], arm_id: str) -> tuple[int, list[dict]]:
    while True:
        eligible, records = eligible_extra_gpus(config, set())
        if eligible:
            return eligible[0], records
        writer.transition(
            "RUNNING",
            "WAITING_FOR_GPU",
            "S17_4_SMOKE_WAITING_NON_GPU1_CARD",
            stage="smoke_waiting_extra_gpu",
            process_alive=True,
            gpu_ids=[],
            gpu_snapshot={"captured_at": utc_now(), "devices": records},
            current_smoke_arm=arm_id,
        )
        time.sleep(GPU_WAIT_SECONDS)


def run_cpu_contracts(root: Path, writer: StatusWriter) -> dict[str, Any]:
    resolved = paths(root)
    log_path = resolved["output"] / "preflight/cpu_contract_tests.log"
    command = [
        str(PYTHON),
        "-m",
        "unittest",
        "-v",
        "experiment.phase17.tests.test_p1_modules",
        "experiment.phase17.tests.test_s4_p1_contract",
        "experiment.phase17.tests.test_module_registry",
        "experiment.phase17.tests.test_split_guard",
        "experiment.phase17.tests.test_trie_legality",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(root)},
            timeout=300,
            check=False,
        )
    writer.heartbeat(stage="cpu_contracts_complete")
    return {
        "return_code": completed.returncode,
        "command": command,
        "log_path": str(log_path.relative_to(root)),
        "log_sha256": sha256(log_path),
    }


def find_prediction_file(artifact_dir: Path) -> Path | None:
    matches = sorted((artifact_dir / "predictions").glob("*_pred_validation.tsv"))
    return matches[-1] if len(matches) == 1 else None


def prediction_rows(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                user = str(row["idx"])
                result[user] = {
                    "hit@5": float(row["H@5"]),
                    "hit@10": float(row["H@10"]),
                    "ndcg@5": float(row["NDCG@5"]),
                    "ndcg@10": float(row["NDCG@10"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return result


def process_checks(
    *,
    parsed: dict[str, Any],
    return_code: int,
    artifact_dir: Path,
    arm: dict[str, Any],
    memory_ceiling: int,
    predictions_required: bool,
    expected_prediction_rows: int | None,
) -> tuple[dict[str, bool], Path | None, Path | None]:
    checkpoint_matches = sorted(artifact_dir.rglob("model_rec_phase_1_epoch_1.pt"))
    prediction = find_prediction_file(artifact_dir)
    rows = prediction_rows(prediction)
    checks = {
        "exit_zero": return_code == 0,
        "no_traceback": not parsed["traceback"],
        "no_forbidden_test_evidence": not parsed["forbidden_test_evidence"],
        "one_training_epoch": len(parsed["training_losses"]) == 1,
        "validation_completed": "ndcg@10" in parsed["validation_metrics"],
        "within_memory_ceiling": parsed["peak_reserved_mib"] is not None
        and parsed["peak_reserved_mib"] <= memory_ceiling,
        "checkpoint_present": len(checkpoint_matches) == 1,
        "mechanism_metric_present": not arm["module_id"] or bool(parsed["mechanism_metric_lines"]),
        "prediction_contract": (not predictions_required)
        or (
            prediction is not None
            and expected_prediction_rows is not None
            and len(rows) == expected_prediction_rows
        ),
    }
    return (
        checks,
        checkpoint_matches[0] if len(checkpoint_matches) == 1 else None,
        prediction,
    )


def run_smokes(root: Path, writer: StatusWriter, config: dict[str, Any]) -> dict[str, Any]:
    resolved = paths(root)
    results = []
    states = {arm["arm_id"]: "QUEUED" for arm in config["smoke"]["arms"]}
    for index, arm in enumerate(config["smoke"]["arms"]):
        artifact_dir = resolved["output"] / "smoke" / arm["arm_id"]
        artifact_dir.mkdir(parents=True, exist_ok=False)
        command = build_command(
            root,
            arm=arm,
            dataset=config["smoke"]["dataset"],
            data_root=root / "artifacts/phase17/s0_audit/profile_data",
            output_dir=artifact_dir,
            epochs=int(config["smoke"]["epochs"]),
            save_predictions=False,
            transition_teacher=resolved["transition_teacher"],
        )
        arm_config = {
            "schema_version": "phase17.s17_4_smoke_arm.v1",
            **arm,
            "command": command,
            "formal_result_eligible": False,
            "test_read": False,
            "sports_read": False,
        }
        atomic_json(artifact_dir / "config.json", arm_config)
        gpu, records = wait_for_smoke_gpu(writer, config, arm["arm_id"])
        states[arm["arm_id"]] = "RUNNING"
        started_at = utc_now()
        started = time.monotonic()
        with (artifact_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=root / "GRAM/command",
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment(root, gpu),
                start_new_session=True,
            )
            writer.transition(
                "RUNNING",
                "RUNNING_SCIENTIFIC",
                "S17_4_P1_SMOKE_RUNNING",
                workload_pid=process.pid,
                process_alive=True,
                gpu_ids=[gpu],
                gpu_snapshot={"captured_at": utc_now(), "devices": records, "selected_gpu": gpu},
                stage=f"smoke_{arm['arm_id']}",
                smoke_states=states,
            )
            while True:
                try:
                    return_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    writer.heartbeat(
                        stage=f"smoke_{arm['arm_id']}",
                        progress={"current": index, "total": len(config["smoke"]["arms"]), "elapsed_seconds": round(elapsed, 1)},
                    )
                    if elapsed > SMOKE_TIMEOUT_SECONDS:
                        os.killpg(process.pid, signal.SIGTERM)
                        return_code = process.wait(timeout=60)
                        break
        parsed = parse_log(artifact_dir / "run.log")
        checks, checkpoint, _ = process_checks(
            parsed=parsed,
            return_code=return_code,
            artifact_dir=artifact_dir,
            arm=arm,
            memory_ceiling=int(config["resources"]["usable_memory_ceiling_mib"]),
            predictions_required=False,
            expected_prediction_rows=None,
        )
        passed = all(checks.values())
        states[arm["arm_id"]] = "COMPLETED" if passed else "FAILED"
        result = {
            "schema_version": "phase17.s17_4_smoke_result.v1",
            **arm,
            "state": states[arm["arm_id"]],
            "return_code": return_code,
            "physical_gpu": gpu,
            "workload_pid": process.pid,
            "wall_seconds": time.monotonic() - started,
            "checks": checks,
            "parsed": parsed,
            "checkpoint": str(checkpoint.relative_to(root)) if checkpoint else None,
            "formal_result_eligible": False,
            "test_read": False,
            "sports_read": False,
        }
        atomic_json(artifact_dir / "result.json", result)
        results.append(result)
        AttemptLedger(resolved["ledger"]).append(
            {
                "attempt_id": f"s4pf_{arm['arm_id']}_001",
                "step_id": "S17-4",
                "track_id": arm["track_id"],
                "kind": "p1_targeted_smoke",
                "started_at": started_at,
                "ended_at": utc_now(),
                "state": result["state"],
                "scientific_result_eligible": False,
                "failure_reason": None if passed else "S17-4 smoke contract failed",
                "artifact_dir": str(artifact_dir.relative_to(root)),
                "config_sha256": sha256(artifact_dir / "config.json"),
                "source_sha256": sha256(resolved["snapshot"]),
                "data_manifest_sha256": sha256(resolved["data_manifest"]),
            }
        )
        current = writer.read()
        writer.transition(
            current["scientific_state"],
            current["execution_state"],
            current["status_code"],
            stage=f"smoke_{arm['arm_id']}_complete",
            heartbeat_at=utc_now(),
            process_alive=True,
            progress={"current": index + 1, "total": len(config["smoke"]["arms"])},
            smoke_states=dict(states),
        )
        if not passed:
            break
    return {
        "passed": len(results) == len(config["smoke"]["arms"])
        and all(result["state"] == "COMPLETED" for result in results),
        "states": states,
        "results": results,
    }


def pid_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return ""
    return path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")


def handoff_gpu1(root: Path, writer: StatusWriter, config: dict[str, Any]) -> dict[str, Any]:
    """Replace only our non-scientific S3 GPU1 workload with formal S17-4."""

    resolved = paths(root)
    s3_path = resolved["status_dir"] / f"{S3_EXPERIMENT_ID}.status.json"
    before = json.loads(s3_path.read_text(encoding="utf-8"))
    if before["scientific_state"] != "COMPLETED":
        raise RuntimeError("GPU1 handoff refused because S17-3 science is not complete")
    pid = int(before.get("workload_pid") or 0)
    command = pid_command(pid) if pid else ""
    if pid and "main_generative_gram.py" not in command:
        raise RuntimeError(f"GPU1 handoff PID {pid} is not the recorded GRAM runtime workload")
    writer.transition(
        "RUNNING",
        "WAITING_FOR_GPU",
        "S17_4_GPU1_ZERO_GAP_HANDOFF_IN_PROGRESS",
        stage="gpu1_handoff",
        process_alive=True,
        gpu_ids=[GPU1],
        s3_runtime_pid=pid,
    )
    terminated_at = None
    if pid and Path(f"/proc/{pid}").exists():
        os.killpg(pid, signal.SIGTERM)
        terminated_at = utc_now()
        deadline = time.monotonic() + 120
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(1)
        if Path(f"/proc/{pid}").exists():
            raise TimeoutError("S17-3 runtime workload did not release GPU1 after SIGTERM")

    minimum = int(config["resources"]["gpu1_minimum_free_after_handoff_mib"])
    records = []
    selected = None
    while selected is None:
        rows = query_gpus()
        records = snapshot(rows)
        selected = next(
            (row for row in rows if row.index == GPU1 and row.free_mib >= minimum),
            None,
        )
        if selected is None:
            writer.heartbeat(stage="gpu1_handoff_waiting_memory_release")
            time.sleep(2)

    # The S3 writer may already have recorded the intentional non-zero runtime
    # exit.  Normalize its execution-only closeout without reopening science.
    s3_writer = StatusWriter(resolved["status_dir"], S3_EXPERIMENT_ID)
    s3_writer.transition(
        "COMPLETED",
        "SCIENTIFIC_COMPLETED",
        "S17_3_GPU1_RUNTIME_HANDED_OFF_TO_S17_4",
        workload_pid=0,
        process_alive=False,
        stage="runtime_handed_off_to_s17_4",
        occupancy_mode="handed_off_to_s17_4",
        result_selection_eligible=False,
        repeat_metrics_ignored=True,
        affects_scientific_result=False,
        handoff_successor_experiment=EXPERIMENT_ID,
    )
    record = {
        "schema_version": "phase17.s17_4_gpu1_handoff.v1",
        "source_experiment": S3_EXPERIMENT_ID,
        "successor_experiment": EXPERIMENT_ID,
        "s3_scientific_state": before["scientific_state"],
        "s3_runtime_pid": pid,
        "s3_runtime_command": command,
        "termination_signal": "SIGTERM" if pid else None,
        "terminated_at": terminated_at,
        "admitted_at": utc_now(),
        "gpu": GPU1,
        "free_mib_at_admission": selected.free_mib,
        "gpu_snapshot": records,
        "scientific_s3_result_affected": False,
    }
    atomic_json(resolved["handoff"], record)
    return record


def spawn_formal_arm(
    root: Path,
    writer: StatusWriter,
    config: dict[str, Any],
    arm: dict[str, Any],
    gpu: int,
    states: dict[str, str],
    active: dict[str, dict[str, Any]],
) -> None:
    resolved = paths(root)
    artifact_dir = resolved["output"] / "arms" / arm["arm_id"]
    artifact_dir.mkdir(parents=True, exist_ok=False)
    command = config["commands"][arm["arm_id"]]
    arm_config = {
        "schema_version": "phase17.s17_4_formal_arm.v1",
        **arm,
        "dataset": config["dataset"],
        "fold": config["fold"],
        "seed": config["seed"],
        "epochs": config["rec_epochs"],
        "command": command,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(artifact_dir / "config.json", arm_config)
    log_handle = (artifact_dir / "run.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=root / "GRAM/command",
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment(root, gpu),
        start_new_session=True,
    )
    states[arm["arm_id"]] = "RUNNING"
    active[arm["arm_id"]] = {
        "arm": arm,
        "gpu": gpu,
        "process": process,
        "log_handle": log_handle,
        "artifact_dir": artifact_dir,
        "started_monotonic": time.monotonic(),
        "started_at": utc_now(),
    }
    writer.transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_4_FORMAL_ARMS_RUNNING",
        process_alive=True,
        workload_pid=next(
            (row["process"].pid for row in active.values() if row["gpu"] == GPU1),
            process.pid,
        ),
        workload_pids={name: row["process"].pid for name, row in active.items()},
        gpu_ids=sorted({row["gpu"] for row in active.values()}),
        current_arms={name: row["gpu"] for name, row in active.items()},
        arm_states=states,
        stage="formal_science",
    )


def finalize_formal_arm(
    root: Path,
    config: dict[str, Any],
    entry: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    resolved = paths(root)
    entry["log_handle"].close()
    arm = entry["arm"]
    artifact_dir = entry["artifact_dir"]
    parsed = parse_log(artifact_dir / "run.log")
    checks, checkpoint, prediction = process_checks(
        parsed=parsed,
        return_code=return_code,
        artifact_dir=artifact_dir,
        arm=arm,
        memory_ceiling=int(config["resources"]["usable_memory_ceiling_mib"]),
        predictions_required=True,
        expected_prediction_rows=int(config["data_manifest"]["expected_users"]),
    )
    passed = all(checks.values())
    result = {
        "schema_version": "phase17.s17_4_formal_result.v1",
        **arm,
        "state": "COMPLETED" if passed else "FAILED",
        "return_code": return_code,
        "workload_pid": entry["process"].pid,
        "physical_gpu": entry["gpu"],
        "started_at": entry["started_at"],
        "ended_at": utc_now(),
        "wall_seconds": time.monotonic() - entry["started_monotonic"],
        "checks": checks,
        "parsed": parsed,
        "checkpoint": str(checkpoint.relative_to(root)) if checkpoint else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "prediction_path": str(prediction.relative_to(root)) if prediction else None,
        "prediction_sha256": sha256(prediction) if prediction else None,
        "test_read": False,
        "sports_read": False,
        "scientific_result_eligible": passed,
    }
    atomic_json(artifact_dir / "result.json", result)
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": f"s4_{arm['arm_id']}_001",
            "step_id": "S17-4",
            "track_id": arm["track_id"],
            "kind": "formal_d0_targeted_screen",
            "started_at": entry["started_at"],
            "ended_at": result["ended_at"],
            "state": result["state"],
            "scientific_result_eligible": passed,
            "failure_reason": None if passed else "formal S17-4 arm contract failed",
            "artifact_dir": str(artifact_dir.relative_to(root)),
            "config_sha256": sha256(artifact_dir / "config.json"),
            "source_sha256": sha256(resolved["snapshot"]),
            "data_manifest_sha256": sha256(resolved["data_manifest"]),
        }
    )
    return result


def run_formal(root: Path, writer: StatusWriter, config: dict[str, Any]) -> list[dict[str, Any]]:
    queue = list(config["formal_arms"])
    states = {arm["arm_id"]: "QUEUED" for arm in queue}
    active: dict[str, dict[str, Any]] = {}
    results = []
    handoff_gpu1(root, writer, config)
    first = queue.pop(0)
    spawn_formal_arm(root, writer, config, first, GPU1, states, active)

    timeout = int(config["resources"]["arm_hard_timeout_seconds"])
    monitor = int(config["resources"]["monitor_seconds"])
    while queue or active:
        busy = {entry["gpu"] for entry in active.values()}
        available = []
        if GPU1 not in busy:
            available.append(GPU1)
        extras, records = eligible_extra_gpus(config, busy)
        available.extend(extras)
        while queue and available:
            arm = queue.pop(0)
            gpu = available.pop(0)
            spawn_formal_arm(root, writer, config, arm, gpu, states, active)
        if not active:
            writer.transition(
                "RUNNING",
                "WAITING_FOR_GPU",
                "S17_4_FORMAL_WAITING_FOR_GPU",
                stage="formal_waiting_for_gpu",
                process_alive=True,
                gpu_snapshot={"captured_at": utc_now(), "devices": records},
            )
            time.sleep(GPU_WAIT_SECONDS)
            continue

        time.sleep(monitor)
        finished = []
        for arm_id, entry in active.items():
            process = entry["process"]
            return_code = process.poll()
            elapsed = time.monotonic() - entry["started_monotonic"]
            if return_code is None and elapsed > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    return_code = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    return_code = process.wait()
            if return_code is not None:
                result = finalize_formal_arm(root, config, entry, return_code)
                results.append(result)
                states[arm_id] = result["state"]
                finished.append(arm_id)
        for arm_id in finished:
            del active[arm_id]
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC" if active else "WAITING_FOR_GPU",
            "S17_4_FORMAL_ARMS_RUNNING" if active else "S17_4_FORMAL_QUEUE_ADVANCING",
            process_alive=True,
            workload_pid=next(
                (entry["process"].pid for entry in active.values() if entry["gpu"] == GPU1),
                0,
            ),
            workload_pids={name: entry["process"].pid for name, entry in active.items()},
            gpu_ids=sorted({entry["gpu"] for entry in active.values()}),
            current_arms={name: entry["gpu"] for name, entry in active.items()},
            arm_states=states,
            stage="formal_science",
            progress={
                "current": len(results),
                "total": len(config["formal_arms"]),
                "unit": "formal_arm",
            },
        )
    return sorted(results, key=lambda row: int(row["priority"]))


def bootstrap_delta(
    treatment: np.ndarray, control: np.ndarray, replicates: int, seed: int
) -> dict[str, float]:
    delta = treatment - control
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, delta.size, size=delta.size)
        samples[index] = float(delta[selected].mean())
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "paired_users": int(delta.size),
        "replicates": int(replicates),
    }


def user_subgroups(dataset_path: Path, users: set[str]) -> dict[str, dict[str, str]]:
    sequences: dict[str, tuple[list[str], str]] = {}
    train_frequency: Counter[str] = Counter()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if len(values) < 4:
                continue
            user, items = values[0], values[1:]
            train, target = items[:-2], items[-2]
            sequences[user] = (train, target)
            train_frequency.update(train)
    target_frequencies = [
        train_frequency[sequences[user][1]] for user in users if user in sequences
    ]
    q1, q2 = np.quantile(target_frequencies, [1 / 3, 2 / 3])
    result: dict[str, dict[str, str]] = {}
    for user in users:
        if user not in sequences:
            continue
        history, target = sequences[user]
        observed = min(20, len(history))
        if observed <= 3:
            length_group = "short_le3"
        elif observed <= 9:
            length_group = "medium_4_9"
        else:
            length_group = "long_ge10"
        frequency = train_frequency[target]
        frequency_group = "tail" if frequency <= q1 else "mid" if frequency <= q2 else "head"
        result[user] = {
            "history_length": length_group,
            "target_frequency": frequency_group,
        }
    return result


def analyze_results(root: Path, config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = {row["arm_id"]: row for row in results if row["state"] == "COMPLETED"}
    if "gram_continue" not in completed:
        raise RuntimeError("paired analysis requires a completed GRAM-Continue control")
    rows_by_arm = {
        arm_id: prediction_rows(root / row["prediction_path"])
        for arm_id, row in completed.items()
    }
    common = set.intersection(*(set(rows) for rows in rows_by_arm.values()))
    expected = int(config["data_manifest"]["expected_users"])
    if len(common) != expected:
        raise RuntimeError(f"paired prediction intersection is {len(common)}, expected {expected}")
    ordered_users = sorted(common, key=lambda value: (len(value), value))
    control = rows_by_arm["gram_continue"]
    replicates = int(config["paired_bootstrap_replicates"])
    comparisons = {}
    for arm_id, rows in rows_by_arm.items():
        if arm_id == "gram_continue":
            continue
        comparisons[arm_id] = {}
        for metric in ("hit@10", "ndcg@10"):
            treatment_values = np.asarray([rows[user][metric] for user in ordered_users])
            control_values = np.asarray([control[user][metric] for user in ordered_users])
            comparisons[arm_id][metric] = bootstrap_delta(
                treatment_values,
                control_values,
                replicates,
                int(config["seed"]) + sum(ord(char) for char in arm_id + metric),
            )

    groups = user_subgroups(paths(root)["dataset"] / "user_sequence.txt", common)
    subgroup_results: dict[str, Any] = {}
    for dimension in ("history_length", "target_frequency"):
        labels = sorted({value[dimension] for value in groups.values()})
        subgroup_results[dimension] = {}
        for label in labels:
            members = [user for user in ordered_users if groups[user][dimension] == label]
            subgroup_results[dimension][label] = {"users": len(members), "arms": {}}
            for arm_id, rows in rows_by_arm.items():
                metrics = {
                    metric: float(np.mean([rows[user][metric] for user in members]))
                    for metric in ("hit@10", "ndcg@10")
                }
                if arm_id != "gram_continue":
                    metrics["delta_ndcg@10_vs_control"] = (
                        metrics["ndcg@10"]
                        - subgroup_results[dimension][label]["arms"]["gram_continue"]["ndcg@10"]
                    )
                subgroup_results[dimension][label]["arms"][arm_id] = metrics

    decisions = {}
    for arm_id, comparison in comparisons.items():
        ndcg = comparison["ndcg@10"]
        if ndcg["ci95_low"] > 0:
            grade = "POSITIVE_PAIRED_CI"
        elif ndcg["mean_delta"] > 0:
            grade = "WEAK_POSITIVE_CI_CROSSES_ZERO"
        else:
            grade = "NON_POSITIVE"
        decisions[arm_id] = grade
    any_positive = any(
        comparison["ndcg@10"]["mean_delta"] > 0 for comparison in comparisons.values()
    )
    return {
        "schema_version": "phase17.s17_4_paired_analysis.v1",
        "paired_users": len(common),
        "comparisons": comparisons,
        "subgroups": subgroup_results,
        "decisions": decisions,
        "any_treatment_positive": any_positive,
        "next_action": (
            "freeze positive candidate configuration for independent S17-5 consideration"
            if any_positive
            else "stop current P1 wave; S17-5 and combinations remain locked"
        ),
    }


def canonical_manifest(root: Path, output: Path, destination: Path) -> dict[str, Any]:
    files = []
    tree = hashlib.sha256()
    total = 0
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        # The background worker keeps this orchestration log open while the
        # post-science runtime cycles run.  Arm logs and result contracts are
        # canonical; this live operational log is intentionally excluded.
        if path == output / "portfolio.log":
            continue
        relative = str(path.relative_to(root))
        digest = sha256(path)
        size = path.stat().st_size
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
        tree.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
        total += size
    payload = {
        "schema_version": "phase17.canonical_results_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "canonical_result_dir": str(output.relative_to(root)),
        "file_count": len(files),
        "total_size_bytes": total,
        "tree_sha256": tree.hexdigest(),
        "files": files,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(destination, payload)
    destination.chmod(0o444)
    return payload


def render_report(
    root: Path,
    config: dict[str, Any],
    cpu: dict[str, Any],
    smoke: dict[str, Any],
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> str:
    by_id = {row["arm_id"]: row for row in results}
    control_metrics = by_id["gram_continue"]["parsed"]["validation_metrics"]
    rows = []
    for arm in config["formal_arms"]:
        result = by_id[arm["arm_id"]]
        metrics = result["parsed"]["validation_metrics"]
        if arm["arm_id"] == "gram_continue":
            delta = "—"
            grade = "matched control"
        else:
            delta_value = analysis["comparisons"][arm["arm_id"]]["ndcg@10"]["mean_delta"]
            delta = f"{delta_value:+.6f}"
            grade = analysis["decisions"][arm["arm_id"]]
        rows.append(
            f"| `{arm['arm_id']}` | {result['state']} | {metrics.get('hit@10', float('nan')):.6f} | "
            f"{metrics.get('ndcg@10', float('nan')):.6f} | {delta} | {grade} | GPU {result['physical_gpu']} |"
        )
    smoke_rows = "\n".join(
        f"| `{row['arm_id']}` | {row['state']} | GPU {row['physical_gpu']} | {row['parsed']['peak_reserved_mib']} |"
        for row in smoke["results"]
    )
    positive = [
        arm_id
        for arm_id, comparison in analysis["comparisons"].items()
        if comparison["ndcg@10"]["mean_delta"] > 0
    ]
    conclusion = (
        "至少一个 treatment 的 paired mean NDCG@10 高于 matched control；仅按证据等级冻结候选，独立 S17-5 仍需单独执行。"
        if positive
        else "三个 treatment 均未超过 matched control；按预注册 stop rule 停止本轮，不解锁 S17-5、组合、PCRF、official test、Beauty 或 Sports。"
    )
    return f"""# Stage 17 S4：P1 定向迁移筛选报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Step：`S17-4`
- Canonical 结果：`{paths(root)['output'].relative_to(root)}`
- 数据：Toys D0 shadow fold；`test_read=false`；`sports_read=false`
- parent：`{config['baseline_checkpoint']}`
- seed / budget：{config['seed']} / 每 arm {config['rec_epochs']} epoch
- CPU 契约：return code `{cpu['return_code']}`
- P1 smoke：`{'PASS' if smoke['passed'] else 'FAIL'}`

## 1. 结论

{conclusion}

Matched `GRAM-Continue` 的 Hit@10=`{control_metrics['hit@10']:.6f}`、NDCG@10=`{control_metrics['ndcg@10']:.6f}`。正向 treatment：`{positive or 'none'}`。所有差值来自同一 D0 用户的 paired prediction；bootstrap 仅用于本折探索证据分级，不构成跨折论文级确认。

## 2. 正式结果

| Arm | 状态 | Hit@10 | NDCG@10 | ΔNDCG@10 | paired 判定 | GPU |
|---|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

详细 paired 95% bootstrap 区间与 history-length / target-frequency 分组保存在 `summary.json -> paired_analysis`。

## 3. P1 迁移与烟测

九个计划内 P1 方向均已建立独立 migration card，并通过 tiny-GRAM 接口契约；GPU smoke 只运行 S3 诊断直接触发的三条候选。

| Smoke | 状态 | GPU | peak reserved MiB |
|---|---|---:|---:|
{smoke_rows}

未进入正式屏的 LS-FiD/MHM、GraphMAE/DCRec、SPRINT 等方向均在各自卡片记录 `not_triggered` 或完整机制尚未实现的边界，不能把接口烟测写成方法有效性证据。

## 4. 资源与边界

- GPU1 仅在三条 smoke 全通过、正式命令与快照冻结后，从 S17-3 非科学重复轮直接交接给 S17-4。
- 正式科学 arm 可使用当时满足显存准入的额外空闲卡；没有抢占或终止其他用户进程。
- 正式科学结束后，额外 GPU 全部释放；仅 GPU1 进入隔离的 run-NNNN 重复轮。
- 重复轮 `result_selection_eligible=false`、`affects_scientific_result=false`，不得进入本报告数值。

## 5. 下一门槛

`{analysis['next_action']}`。无论本折结果如何，official test 与 Sports 均继续封存。
"""


def publish_closeout(
    root: Path,
    writer: StatusWriter,
    config: dict[str, Any],
    cpu: dict[str, Any],
    smoke: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved = paths(root)
    analysis = analyze_results(root, config, results)
    summary = {
        "schema_version": "phase17.s17_4_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "step_id": "S17-4",
        "cpu_contracts": cpu,
        "p1_static_cards": sorted(
            str(path.relative_to(root))
            for path in (root / "experiment/phase17/registry/migration_cards").glob("P1*.yaml")
        ),
        "smoke": smoke,
        "formal_results": results,
        "paired_analysis": analysis,
        "stop_rule_applied": not analysis["any_treatment_positive"],
        "test_read": False,
        "sports_read": False,
        "completed_at": utc_now(),
    }
    atomic_json(resolved["summary"], summary)
    report_text = render_report(root, config, cpu, smoke, results, analysis)
    resolved["report"].parent.mkdir(parents=True, exist_ok=True)
    resolved["report"].write_text(report_text, encoding="utf-8")
    manifest = canonical_manifest(
        root, resolved["output"], resolved["canonical_manifest"]
    )
    writer.transition(
        "COMPLETED",
        "SCIENTIFIC_COMPLETED",
        "S17_4_TARGETED_FORMAL_COMPLETE",
        workload_pid=0,
        workload_pids={},
        process_alive=False,
        gpu_ids=[],
        current_arms={},
        stage="scientific_complete_report_published",
        progress={"current": len(results), "total": len(config["formal_arms"]), "unit": "formal_arm"},
        result_selection_eligible=True,
        affects_scientific_result=True,
        summary_path=str(resolved["summary"].relative_to(root)),
        report_path=str(resolved["report"].relative_to(root)),
        report_sha256=sha256(resolved["report"]),
        canonical_result_manifest=str(resolved["canonical_manifest"].relative_to(root)),
        canonical_result_sha256=manifest["tree_sha256"],
        smoke_states=dict(smoke["states"]),
        test_read=False,
        sports_read=False,
    )
    return summary


def wait_for_gpu1(writer: StatusWriter, config: dict[str, Any]) -> list[dict]:
    minimum = int(config["resources"]["gpu1_minimum_free_after_handoff_mib"])
    while True:
        rows = query_gpus()
        selected = next(
            (row for row in rows if row.index == GPU1 and row.free_mib >= minimum),
            None,
        )
        if selected is not None:
            return snapshot(rows)
        writer.heartbeat(stage="post_science_gpu1_wait")
        time.sleep(GPU_WAIT_SECONDS)


def runtime_loop(
    root: Path,
    writer: StatusWriter,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    start_iteration: int = 2,
) -> None:
    resolved = paths(root)
    source_id = config["execution_contract"]["runtime_source_arm"]
    successful = {row["arm_id"] for row in results if row["state"] == "COMPLETED"}
    if source_id not in successful:
        source_id = next(iter(successful))
    arm = next(row for row in config["formal_arms"] if row["arm_id"] == source_id)
    iteration = start_iteration
    while True:
        runtime_dir = isolated_runtime_dir(root, EXPERIMENT_ID, iteration)
        assert_runtime_isolation(resolved["output"], runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=False)
        records = wait_for_gpu1(writer, config)
        command = build_command(
            root,
            arm=arm,
            dataset=config["dataset"],
            data_root=resolved["data_root"],
            output_dir=runtime_dir,
            epochs=int(config["rec_epochs"]),
            save_predictions=False,
            transition_teacher=resolved["transition_teacher"],
        )
        atomic_json(
            runtime_dir / "runtime_config.json",
            {
                "schema_version": "phase17.runtime_cycle.v1",
                "source_arm": source_id,
                "iteration": iteration,
                "physical_gpu": GPU1,
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
                env=environment(root, GPU1),
                start_new_session=True,
            )
            writer.start_runtime_cycle(
                iteration=iteration,
                runtime_result_dir=str(runtime_dir.relative_to(root)),
                workload_pid=process.pid,
            )
            writer.transition(
                "COMPLETED",
                "RUNNING_OCCUPANCY_REPEAT",
                "SCIENTIFIC_COMPLETED_REPEATING_FOR_GPU_OCCUPANCY",
                gpu_ids=[GPU1],
                gpu_snapshot={"captured_at": utc_now(), "devices": records},
                post_science_extra_gpu_policy="released",
                gpu1_state="runtime_cycle_active",
                stage="runtime_cycle_active",
            )
            started = time.monotonic()
            while True:
                try:
                    return_code = process.wait(timeout=int(config["resources"]["monitor_seconds"]))
                    break
                except subprocess.TimeoutExpired:
                    writer.heartbeat(
                        stage="runtime_cycle_active",
                        progress={
                            "current": len(config["formal_arms"]),
                            "total": len(config["formal_arms"]),
                            "unit": "formal_arm",
                            "runtime_iteration": iteration,
                            "source_arm": source_id,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        },
                    )
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


def runtime_recovery_inputs(
    root: Path,
    *,
    allow_background_launcher: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Validate a one-shot restart of the non-scientific GPU1 runtime loop."""
    resolved = paths(root)
    status = StatusWriter(resolved["status_dir"], EXPERIMENT_ID).read()
    if status["scientific_state"] != "COMPLETED":
        raise RuntimeError("S17-4 science must be complete before runtime recovery")
    launcher_only = (
        allow_background_launcher
        and status.get("stage") == "runtime_recovery_waiting_gpu1"
        and not status.get("workload_pid")
    )
    if status.get("process_alive") and not launcher_only:
        raise RuntimeError("S17-4 already records a live runtime process")
    if (
        status.get("occupancy_mode") != "stopped_after_runtime_failure"
        and not launcher_only
    ):
        raise RuntimeError("runtime recovery requires a recorded prior runtime failure")
    if not resolved["config"].is_file() or not resolved["summary"].is_file():
        raise FileNotFoundError("frozen S17-4 config or summary is missing")
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    summary = json.loads(resolved["summary"].read_text(encoding="utf-8"))
    results = summary.get("formal_results", [])
    expected = {arm["arm_id"] for arm in config["formal_arms"]}
    completed = {
        row["arm_id"] for row in results if row.get("state") == "COMPLETED"
    }
    if completed != expected:
        raise RuntimeError("frozen S17-4 formal results are incomplete")
    iteration = int(status.get("repeat_iteration") or 1) + 1
    runtime_dir = isolated_runtime_dir(root, EXPERIMENT_ID, iteration)
    if runtime_dir.exists():
        raise FileExistsError(f"runtime recovery target already exists: {runtime_dir}")
    return config, results, iteration


def launch_runtime_recovery(root: Path) -> int:
    """Launch exactly one explicitly authorized recovery of the runtime loop."""
    resolved = paths(root)
    runtime_recovery_inputs(root)
    command = [
        str(PYTHON),
        str(root / "experiment/phase17/protocol/s4_targeted_p1_runtime.py"),
        "runtime-recovery-worker",
        "--root",
        str(root),
    ]
    session = launch_background_tmux(
        experiment_id=RUNTIME_RECOVERY_SESSION,
        argv=command,
        cwd=root,
        tmux_session=RUNTIME_RECOVERY_SESSION,
    )
    StatusWriter(resolved["status_dir"], EXPERIMENT_ID).transition(
        "COMPLETED",
        "BACKGROUND_STARTED",
        "S17_4_RUNTIME_RECOVERY_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        workload_pid=0,
        process_alive=True,
        stage="runtime_recovery_waiting_gpu1",
        gpu_ids=[GPU1],
        result_selection_eligible=False,
        affects_scientific_result=False,
        repeat_metrics_ignored=True,
    )
    print(session)
    return 0


def runtime_recovery_worker(root: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    config, results, iteration = runtime_recovery_inputs(
        root, allow_background_launcher=True
    )
    attempt_id = f"{ATTEMPT_ID}-runtime-recovery-{iteration:04d}"
    started_at = utc_now()
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": attempt_id,
            "step_id": "S17-4",
            "kind": "POST_SCIENCE_RUNTIME_RECOVERY",
            "started_at": started_at,
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "affects_scientific_result": False,
            "gpu_ids": [GPU1],
            "start_iteration": iteration,
        }
    )
    runtime_loop(
        root,
        writer,
        config,
        results,
        start_iteration=iteration,
    )
    status = writer.read()
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": f"{attempt_id}-closeout",
            "closes_attempt_id": attempt_id,
            "step_id": "S17-4",
            "kind": "POST_SCIENCE_RUNTIME_RECOVERY_CLOSEOUT",
            "started_at": started_at,
            "ended_at": utc_now(),
            "state": "FAILED",
            "scientific_result_eligible": False,
            "affects_scientific_result": False,
            "runtime_return_code": status.get("runtime_return_code"),
            "repeat_iteration": status.get("repeat_iteration"),
        }
    )
    return int(status.get("runtime_return_code") or 1)


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
                "S17_4_CPU_CONTRACTS_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="failed_cpu_contracts",
            )
            return cpu["return_code"]
        smoke = run_smokes(root, writer, config)
        if not smoke["passed"]:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_4_TARGETED_SMOKE_FAILED_S3_GPU1_UNTOUCHED",
                workload_pid=0,
                process_alive=False,
                stage="smoke_failed",
                smoke_states=smoke["states"],
            )
            return 1
        results = run_formal(root, writer, config)
        if not results or any(row["state"] != "COMPLETED" for row in results):
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_4_FORMAL_COMPLETE_WITH_ARM_FAILURE",
                workload_pid=0,
                process_alive=False,
                stage="formal_failed",
                result_selection_eligible=False,
            )
            return 1
        publish_closeout(root, writer, config, cpu, smoke, results)
        runtime_loop(root, writer, config, results)
        return 0
    except Exception as error:
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_4_ORCHESTRATOR_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="orchestrator_failed",
                failure_reason=f"{type(error).__name__}: {error}",
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "launch",
            "worker",
            "status",
            "launch-runtime-recovery",
            "runtime-recovery-worker",
        ],
    )
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
    if args.action == "launch-runtime-recovery":
        return launch_runtime_recovery(root)
    if args.action == "runtime-recovery-worker":
        recovery_log = paths(root)["output"] / "runtime_recovery.log"
        recovery_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = recovery_log.open("a", encoding="utf-8", buffering=1)
        os.dup2(log_handle.fileno(), sys.stdout.fileno())
        os.dup2(log_handle.fileno(), sys.stderr.fileno())
        return runtime_recovery_worker(root)
    print(paths(root)["status"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
