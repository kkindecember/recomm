#!/usr/bin/env python3
"""Run the S17-3 A0/A1/E0 revision gate as a background smoke portfolio."""

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

from experiment.phase17.core.resource_profiler import (  # noqa: E402
    choose_idle_gpu,
    query_gpus,
    snapshot,
)
from experiment.phase17.core.run_manager import (  # noqa: E402
    freeze_run_snapshot,
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
from experiment.phase17.protocol.s2_probe_runtime import command_template, parse_log  # noqa: E402


EXPERIMENT_ID = "s17_s3_revision_gate_a1"
ATTEMPT_ID = "run-0001"
TMUX_SESSION = EXPERIMENT_ID
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
GPU_WAIT_SECONDS = 30
ARM_TIMEOUT_SECONDS = 900


def paths(root: Path) -> dict[str, Path]:
    output = root / "artifacts/phase17/s3_revision_gate" / ATTEMPT_ID
    return {
        "output": output,
        "status_dir": root / "artifacts/phase17/status",
        "status": root / "artifacts/phase17/status" / f"{EXPERIMENT_ID}.status.json",
        "ledger": root / "artifacts/phase17/attempts/S17-3.attempts.jsonl",
        "summary": output / "summary.json",
        "config": output / "portfolio_config.json",
        "worker_log": output / "portfolio.log",
        "snapshot": root / "artifacts/phase17/snapshots" / EXPERIMENT_ID / ATTEMPT_ID / "manifest.json",
        "budget": root / "experiment/phase17/config/s17_s3_revision_gate.json",
    }


def source_paths(root: Path) -> list[Path]:
    return [
        root / "experiment/phase17/protocol/s3_revision_gate_runtime.py",
        root / "experiment/phase17/protocol/s2_probe_runtime.py",
        root / "experiment/phase17/run_stage17_s3_revision_gate.sh",
        root / "experiment/phase17/config/s17_s3_revision_gate.json",
        root / "experiment/phase17/core/p0_modules.py",
        root / "experiment/phase17/core/loss_hooks.py",
        root / "experiment/phase17/core/feature_hooks.py",
        root / "experiment/phase17/core/runtime.py",
        root / "experiment/phase17/core/status_writer.py",
        root / "experiment/phase17/core/run_manager.py",
        root / "experiment/phase17/core/resource_profiler.py",
        root / "experiment/phase17/registry/module_registry.py",
        root / "experiment/phase17/registry/migration_cards/A0_bear_gram.yaml",
        root / "experiment/phase17/registry/migration_cards/E0_shortcut_fid_gram.yaml",
        root / "experiment/phase17/tests/test_p0_modules.py",
        root / "experiment/phase17/tests/test_p0_gram_integration.py",
        root / "GRAM/src/model/gram.py",
        root / "GRAM/src/arguments.py",
        root / "GRAM/src/main_generative_gram.py",
        root / "GRAM/src/data/multi_task_dataset_gram.py",
        root / "GRAM/src/data/test_dataset_gram.py",
        root / "GRAM/src/runner/single_runner_gram.py",
    ]


def prepare(root: Path) -> int:
    resolved = paths(root)
    if resolved["output"].exists() or resolved["snapshot"].exists() or resolved["status"].exists():
        raise FileExistsError("S17-3 revision gate already exists; automatic retry is forbidden")
    budget = json.loads(resolved["budget"].read_text(encoding="utf-8"))
    resolved["output"].mkdir(parents=True)
    (resolved["output"] / "preflight").mkdir()
    worker_command = [
        str(PYTHON),
        str(root / "experiment/phase17/protocol/s3_revision_gate_runtime.py"),
        "worker",
        "--root",
        str(root),
    ]
    config = {
        **budget,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "worker_command": worker_command,
        "budget_path": str(resolved["budget"].relative_to(root)),
        "budget_sha256": sha256(resolved["budget"]),
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
        step_id="S17-3",
        attempt_id=ATTEMPT_ID,
        canonical_result_dir=str(resolved["output"].relative_to(root)),
        log_path=str(resolved["worker_log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": len(budget["arms"])},
            "predicted_peak_mib": budget["resources"]["expected_peak_mib"],
            "usable_memory_ceiling_mib": budget["resources"]["usable_memory_ceiling_mib"],
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "budget_path": str(resolved["budget"].relative_to(root)),
            "result_selection_eligible": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_3_REVISION_GATE_PREFLIGHT_COMPLETE",
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
        "S17_3_REVISION_GATE_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="cpu_contracts",
    )
    print(session)
    return 0


def wait_for_gpu(writer: StatusWriter, config: dict, progress: dict) -> tuple[int, list[dict]]:
    resources = config["resources"]
    while True:
        records = query_gpus()
        selected = choose_idle_gpu(
            records,
            expected_peak_mib=int(resources["expected_peak_mib"]),
            safety_margin_mib=int(resources["safety_margin_mib"]),
        )
        if selected is not None:
            return selected.index, snapshot(records)
        writer.transition(
            "RUNNING",
            "WAITING_FOR_GPU",
            "S17_3_REVISION_GATE_WAITING_ELIGIBLE_GPU",
            workload_pid=0,
            process_alive=True,
            gpu_ids=[],
            gpu_snapshot={"captured_at": utc_now(), "devices": snapshot(records)},
            stage="waiting_for_gpu",
            progress=progress,
        )
        time.sleep(GPU_WAIT_SECONDS)


def run_process(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    writer: StatusWriter,
    arm_id: str,
    progress: dict,
) -> tuple[int, int, float]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_3_REVISION_GATE_ARM_RUNNING",
            workload_pid=process.pid,
            process_alive=True,
            stage=f"{arm_id}_running",
            progress=progress,
        )
        while True:
            try:
                return_code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                writer.heartbeat(
                    stage=f"{arm_id}_running",
                    progress={**progress, "elapsed_seconds": round(elapsed, 1)},
                )
                if elapsed > ARM_TIMEOUT_SECONDS:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return_code = 124
                    break
    return return_code, process.pid, time.monotonic() - started


def last_metric(parsed: dict, name: str) -> float | None:
    pattern = re.compile(rf"(?:^|\s){re.escape(name)}=([0-9.eE+-]+)")
    for line in reversed(parsed["mechanism_metric_lines"]):
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def evaluate_arm(arm: dict, parsed: dict, return_code: int) -> dict[str, bool]:
    checks = {
        "exit_zero": return_code == 0,
        "no_traceback": not parsed["traceback"],
        "no_forbidden_test_evidence": not parsed["forbidden_test_evidence"],
        "validation_completed": bool(parsed["validation_metrics"]),
        "mechanism_metric_present": bool(parsed["mechanism_metric_lines"]),
        "within_memory_ceiling": parsed["peak_reserved_mib"] is not None
        and parsed["peak_reserved_mib"] <= 30720,
    }
    module_id = arm["module_id"]
    if module_id == "E0_shortcut_fid":
        ratio = last_metric(parsed, f"{module_id}/selected_history_ratio")
        checks["semantic_selection_non_degenerate"] = ratio is not None and 0.0 < ratio < 1.0
    elif module_id == "E0_shortcut_fid_full_control":
        ratio = last_metric(parsed, f"{module_id}/selected_history_ratio")
        checks["full_control_selects_all"] = ratio is not None and abs(ratio - 1.0) < 1e-8
    elif module_id == "E0_shortcut_fid_random_control":
        ratio = last_metric(parsed, f"{module_id}/selected_history_ratio")
        target = last_metric(parsed, f"{module_id}/adaptive_target_ratio")
        checks["random_control_same_size"] = (
            ratio is not None and target is not None and abs(ratio - target) < 1e-8
        )
    return checks


def run_cpu_contracts(root: Path, writer: StatusWriter, total: int) -> dict:
    resolved = paths(root)
    log_path = resolved["output"] / "preflight/cpu_contract_tests.log"
    command = [
        str(PYTHON),
        "-m",
        "unittest",
        "-v",
        "experiment.phase17.tests.test_p0_modules",
        "experiment.phase17.tests.test_p0_gram_integration",
        "experiment.phase17.tests.test_module_registry",
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
            timeout=180,
            check=False,
        )
    writer.heartbeat(stage="cpu_contracts_complete", progress={"current": 0, "total": total})
    return {
        "return_code": completed.returncode,
        "log_path": str(log_path.relative_to(root)),
        "log_sha256": sha256(log_path),
    }


def run_arm(root: Path, writer: StatusWriter, config: dict, arm: dict, index: int) -> dict:
    resolved = paths(root)
    artifact_dir = resolved["output"] / "arms" / arm["arm_id"]
    if artifact_dir.exists():
        raise FileExistsError(f"revision-gate arm output exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True)
    command = command_template(
        root,
        track_id=arm["track_id"],
        module_id=arm["module_id"],
        dataset=config["dataset"],
        epochs=int(config["epochs"]),
        output_dir=artifact_dir,
        transition_map=artifact_dir / "unused_transition_teacher.json",
    )
    attempt_id = f"s3pf_{arm['arm_id']}_001"
    arm_config = {
        "schema_version": "phase17.s17_3_revision_gate_arm.v1",
        "attempt_id": attempt_id,
        **arm,
        "dataset": config["dataset"],
        "epochs": config["epochs"],
        "seed": config["seed"],
        "command": command,
        "timeout_seconds": ARM_TIMEOUT_SECONDS,
        "test_read": False,
        "sports_read": False,
    }
    config_path = artifact_dir / "config.json"
    atomic_json(config_path, arm_config)
    progress = {"current": index, "total": len(config["arms"]), "arm_id": arm["arm_id"]}
    gpu, gpu_records = wait_for_gpu(writer, config, progress)
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
    writer.transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_3_REVISION_GATE_ARM_ADMITTED",
        gpu_ids=[gpu],
        gpu_snapshot={
            "captured_at": utc_now(),
            "devices": gpu_records,
            "selected_gpu": gpu,
            "selection_reason": "lowest-utilization eligible idle card at admission",
        },
        stage=f"{arm['arm_id']}_admitted",
        progress=progress,
        process_alive=True,
    )
    started_at = utc_now()
    return_code, pid, wall_seconds = run_process(
        command=command,
        cwd=root / "GRAM/command",
        log_path=artifact_dir / "run.log",
        env=env,
        writer=writer,
        arm_id=arm["arm_id"],
        progress=progress,
    )
    parsed = parse_log(artifact_dir / "run.log")
    checks = evaluate_arm(arm, parsed, return_code)
    passed = all(checks.values())
    result = {
        "schema_version": "phase17.s17_3_revision_gate_result.v1",
        "attempt_id": attempt_id,
        **arm,
        "state": "COMPLETED" if passed else "FAILED",
        "return_code": return_code,
        "workload_pid": pid,
        "physical_gpu": gpu,
        "wall_seconds": wall_seconds,
        "checks": checks,
        "parsed": parsed,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(artifact_dir / "result.json", result)
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": attempt_id,
            "step_id": "S17-3",
            "track_id": arm["track_id"],
            "kind": "revision_gate_smoke",
            "started_at": started_at,
            "ended_at": utc_now(),
            "state": result["state"],
            "config_sha256": sha256(config_path),
            "data_manifest_sha256": sha256(
                root / "artifacts/phase17/s0_audit/shadow_data_manifest.json"
            ),
            "source_sha256": sha256(resolved["snapshot"]),
            "scientific_result_eligible": False,
            "failure_reason": None if passed else "revision gate failed; inspect checks/run.log",
            "artifact_dir": str(artifact_dir.relative_to(root)),
        }
    )
    return result


def worker(root: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    try:
        verify_run_snapshot(root, resolved["snapshot"])
        cpu = run_cpu_contracts(root, writer, len(config["arms"]))
        if cpu["return_code"] != 0:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_3_REVISION_GATE_CPU_CONTRACTS_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="failed_cpu_contracts",
            )
            return cpu["return_code"]
        results = []
        for index, arm in enumerate(config["arms"]):
            result = run_arm(root, writer, config, arm, index)
            results.append(result)
            if result["state"] != "COMPLETED":
                break
            writer.heartbeat(
                stage=f"{arm['arm_id']}_complete",
                progress={"current": index + 1, "total": len(config["arms"])},
            )
        passed = len(results) == len(config["arms"]) and all(
            result["state"] == "COMPLETED" for result in results
        )
        summary = {
            "schema_version": "phase17.s17_3_revision_gate_summary.v1",
            "experiment_id": EXPERIMENT_ID,
            "step_id": "S17-3",
            "substep": "pre_formal_revision_gate",
            "verdict": "S17_3_REVISION_GATE_PASS" if passed else "S17_3_REVISION_GATE_FAIL",
            "cpu_contracts": cpu,
            "arms": results,
            "formal_result_eligible": False,
            "official_result_claim": False,
            "test_read": False,
            "sports_read": False,
            "completed_at": utc_now(),
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED" if passed else "FAILED",
            "SCIENTIFIC_COMPLETED" if passed else "SCIENTIFIC_FAILED",
            summary["verdict"],
            workload_pid=0,
            process_alive=False,
            stage="revision_gate_complete" if passed else "revision_gate_failed",
            progress={"current": len(results), "total": len(config["arms"])},
            result_selection_eligible=False,
            affects_scientific_result=False,
            summary_path=str(resolved["summary"].relative_to(root)),
        )
        return 0 if passed else 1
    except Exception as error:
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_3_REVISION_GATE_ORCHESTRATOR_FAILED",
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
