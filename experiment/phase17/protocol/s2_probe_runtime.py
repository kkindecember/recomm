#!/usr/bin/env python3
"""Run the frozen S17-2 seven-track probe portfolio and isolated runtime cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# A pre-existing tmux server does not necessarily inherit arbitrary client
# environment variables.  Bootstrap from this file first; the environment is
# only an optional override for relocated snapshots.
ROOT_HINT = os.environ.get("S17_REPOSITORY_ROOT") or str(Path(__file__).resolve().parents[3])
if ROOT_HINT not in sys.path:
    sys.path.insert(0, ROOT_HINT)

from experiment.phase17.core.resource_profiler import (  # noqa: E402
    MAX_USABLE_MIB_PER_JOB,
    choose_idle_gpu,
    query_gpus,
    snapshot,
)
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


EXPERIMENT_ID = "s17_s2_p0_probe_matrix_r2"
ATTEMPT_ID = "run-0003"
TMUX_SESSION = "s17_s2_p0_probe_matrix_r2"
EXPECTED_PEAK_MIB = 23000
SAFETY_MARGIN_MIB = 4096
GPU_WAIT_SECONDS = 30
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
TRACKS = [
    ("A0", "A0_bear"),
    ("A1", "A1_prefixcurr"),
    ("B0", "B0_mvi"),
    ("B1", "B1_latte"),
    ("C0", "C0_biflow"),
    ("D0", "D0_ted"),
    ("E0", "E0_shortcut_fid"),
]


def paths(root: Path) -> dict[str, Path]:
    output = root / "artifacts/phase17/s2_probe" / ATTEMPT_ID
    return {
        "output": output,
        "status_dir": root / "artifacts/phase17/status",
        "status": root / "artifacts/phase17/status" / f"{EXPERIMENT_ID}.status.json",
        "ledger": root / "artifacts/phase17/attempts/S17-2.attempts.jsonl",
        "summary": output / "summary.json",
        "config": output / "portfolio_config.json",
        "worker_log": output / "portfolio.log",
        "snapshot": root / "artifacts/phase17/snapshots" / EXPERIMENT_ID / ATTEMPT_ID / "manifest.json",
        "budget": root / "experiment/phase17/config/s17_s2_probe_budget.json",
    }


def source_paths(root: Path) -> list[Path]:
    return [
        root / "experiment/phase17/protocol/s2_probe_runtime.py",
        root / "experiment/phase17/run_stage17_s2_probe_matrix.sh",
        root / "experiment/phase17/config/s17_s2_probe_budget.json",
        *sorted((root / "experiment/phase17/core").glob("*.py")),
        root / "experiment/phase17/registry/module_registry.py",
        root / "GRAM/src/model/gram.py",
        root / "GRAM/src/arguments.py",
        root / "GRAM/src/main_generative_gram.py",
        root / "GRAM/src/data/multi_task_dataset_gram.py",
        root / "GRAM/src/data/test_dataset_gram.py",
        root / "GRAM/src/runner/single_runner_gram.py",
    ]


def read_item_catalog(dataset_dir: Path) -> list[str]:
    index_path = dataset_dir / "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
    return sorted(
        line.split(" ", 1)[0]
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def build_transition_teacher(dataset_dir: Path, output: Path) -> dict:
    """Build current->next counts from seq[:-2], sealing validation/test targets."""

    items = read_item_catalog(dataset_dir)
    item2dense = {item_id: index + 1 for index, item_id in enumerate(items)}
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    transitions = 0
    users = 0
    sequence_path = dataset_dir / "user_sequence.txt"
    for line in sequence_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 2:
            continue
        users += 1
        # Leave-one-out D0: penultimate is validation, final is official test.
        train_items = values[1:-2]
        for current, following in zip(train_items, train_items[1:]):
            counts[item2dense[current]][item2dense[following]] += 1
            transitions += 1
    top_next = [0] * (len(items) + 1)
    for current, candidates in counts.items():
        top_next[current] = min(
            candidates,
            key=lambda target: (-candidates[target], target),
        )
    payload = {
        "schema_version": "phase17.fold_train_transition_teacher.v1",
        "dataset_dir": str(dataset_dir),
        "users": users,
        "catalog_items": len(items),
        "fold_train_transitions": transitions,
        "covered_source_items": sum(value != 0 for value in top_next),
        "validation_position_read": False,
        "test_position_read": False,
        "sports_read": False,
        "top_next_dense_id": top_next,
    }
    atomic_json(output, payload)
    return payload


def command_template(
    root: Path,
    *,
    track_id: str,
    module_id: str,
    dataset: str,
    epochs: int,
    output_dir: Path,
    transition_map: Path,
) -> list[str]:
    command = [
        str(PYTHON),
        "../src/main_generative_gram.py",
        "--data_path", str(root / "artifacts/phase17/s0_audit/profile_data"),
        "--datasets", dataset,
        "--distributed", "0", "--gpu", "0", "--seed", "2023", "--train", "1",
        "--resource_metrics", "1",
        "--log_dir", str(output_dir / "gram_logs"),
        "--prediction_dir", str(output_dir / "predictions"),
        "--item_prompt_max_len", "128", "--item_prompt", "all_text",
        "--cf_model", "sasrec", "--id_linking", "1", "--max_his", "20",
        "--rec_batch_size", "16", "--gradient_accumulation_steps", "8",
        "--rec_lr", "1e-3", "--rec_epochs", str(epochs), "--test_epoch_rec", "0",
        "--save_rec_epochs", str(epochs), "--save_predictions", "0", "--beam_size", "50",
        "--top_k_similar_item", "5", "--item_id_type", "split",
        "--hierarchical_id_type", "hierarchy_v1_c32_l5_len32768_split",
        "--debug_train_100", "0", "--debug_test_100", "0",
        "--cf0_arm", "A", "--cf0_phase9", "1", "--hi_gram_enabled", "0",
        "--s17_modules", module_id,
    ]
    if track_id == "D0":
        command.extend(["--s17_transition_map", str(transition_map)])
    return command


def prepare(root: Path) -> int:
    resolved = paths(root)
    if resolved["output"].exists() or resolved["snapshot"].exists() or resolved["status"].exists():
        raise FileExistsError("S17-2 run-0003 already exists; automatic retry is forbidden")
    resolved["output"].mkdir(parents=True)
    teacher_dir = resolved["output"] / "preflight"
    teacher_dir.mkdir()
    teachers = {}
    for users in (100, 1000):
        dataset = f"Toys_s17_d0_{users}"
        output = teacher_dir / f"transition_teacher_{users}.json"
        teachers[dataset] = build_transition_teacher(
            root / "artifacts/phase17/s0_audit/profile_data" / dataset,
            output,
        )
    worker_command = [
        str(PYTHON),
        str(root / "experiment/phase17/protocol/s2_probe_runtime.py"),
        "worker",
        "--root",
        str(root),
    ]
    config = {
        "schema_version": "phase17.s17_2_portfolio.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "recovery_of": "run-0002",
        "recovery_scope": "one shared validation-loader tokenizer fallback fix; mechanism and budget unchanged",
        "step_id": "S17-2",
        "tracks": [{"track_id": track, "module_id": module} for track, module in TRACKS],
        "progress_total": len(TRACKS) * 2,
        "worker_command": worker_command,
        "expected_peak_mib": EXPECTED_PEAK_MIB,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "usable_memory_ceiling_mib": MAX_USABLE_MIB_PER_JOB,
        "global_gpu_count_hard_ceiling": None,
        "planning_baseline_concurrent_gpus": 2,
        "actual_scheduler_concurrency": 1,
        "resource_policy_note": "S17-2 remains single-card serial; future large portfolios request the useful count from the researcher without a global two-GPU cap",
        "transition_teachers": teachers,
        "budget_path": str(resolved["budget"].relative_to(root)),
        "budget_sha256": sha256(resolved["budget"]),
        "test_read": False,
        "sports_read": False,
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
        step_id="S17-2",
        attempt_id=ATTEMPT_ID,
        canonical_result_dir=str(resolved["output"].relative_to(root)),
        log_path=str(resolved["worker_log"].relative_to(root)),
        extra={
            "stage": "preflight_complete",
            "progress": {"current": 0, "total": len(TRACKS) * 2},
            "predicted_peak_mib": EXPECTED_PEAK_MIB,
            "usable_memory_ceiling_mib": MAX_USABLE_MIB_PER_JOB,
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "budget_path": str(resolved["budget"].relative_to(root)),
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_2_PREFLIGHT_COMPLETE",
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
        "S17_2_BACKGROUND_PORTFOLIO_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="cpu_contracts",
    )
    print(session)
    return 0


def wait_for_gpu(writer: StatusWriter, progress: dict) -> tuple[int, list[dict]]:
    while True:
        records = query_gpus()
        selected = choose_idle_gpu(
            records,
            expected_peak_mib=EXPECTED_PEAK_MIB,
            safety_margin_mib=SAFETY_MARGIN_MIB,
        )
        if selected is not None:
            return selected.index, snapshot(records)
        writer.transition(
            "RUNNING",
            "WAITING_FOR_GPU",
            "S17_2_WAITING_ELIGIBLE_GPU",
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
    timeout_seconds: int,
    writer: StatusWriter,
    stage: str,
    progress: dict,
    gpu: int,
) -> tuple[int, int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            "S17_2_PROBE_RUNNING",
            workload_pid=process.pid,
            process_alive=True,
            gpu_ids=[gpu],
            stage=stage,
            progress=progress,
        )
        while True:
            try:
                return_code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                writer.heartbeat(
                    stage=stage,
                    progress={**progress, "elapsed_seconds": round(elapsed, 1)},
                    process_alive=True,
                )
                if elapsed > timeout_seconds:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return_code = 124
                    break
    return return_code, process.pid, time.monotonic() - started


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    loss_pattern = re.compile(r"average training loss .*? is ([0-9.eE+-]+)")
    losses = [float(value) for value in loss_pattern.findall(text)]
    metric_pattern = re.compile(r"validation (hit|ndcg)@(\d+): ([0-9.eE+-]+)")
    validation = {
        f"{name}@{cutoff}": float(value)
        for name, cutoff, value in metric_pattern.findall(text)
    }
    resource_pattern = re.compile(
        r"RESOURCE_METRIC phase=(\S+) wall_time_seconds=([0-9.]+) "
        r"peak_allocated_mib=([0-9.]+) peak_reserved_mib=([0-9.]+)"
    )
    resources = [
        {
            "phase": match.group(1),
            "wall_time_seconds": float(match.group(2)),
            "peak_allocated_mib": float(match.group(3)),
            "peak_reserved_mib": float(match.group(4)),
        }
        for match in resource_pattern.finditer(text)
    ]
    mechanism_lines = [
        line for line in text.splitlines() if "S17_MECHANISM_METRIC " in line
    ]
    forbidden = [
        token
        for token in ("automatic_last_checkpoint_test", "[test] testing", "_pred_test.tsv")
        if token in text
    ]
    return {
        "training_losses": losses,
        "validation_metrics": validation,
        "resource_metrics": resources,
        "peak_reserved_mib": max(
            (row["peak_reserved_mib"] for row in resources), default=None
        ),
        "mechanism_metric_lines": mechanism_lines[-8:],
        "forbidden_test_evidence": forbidden,
        "traceback": "Traceback (most recent call last)" in text,
        "log_sha256": sha256(log_path),
    }


def append_attempt(
    root: Path,
    *,
    attempt_id: str,
    track_id: str,
    kind: str,
    started_at: str,
    state: str,
    failure_reason: str | None,
    config_path: Path,
    artifact_dir: Path,
) -> None:
    resolved = paths(root)
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": attempt_id,
            "step_id": "S17-2",
            "track_id": track_id,
            "kind": kind,
            "started_at": started_at,
            "ended_at": utc_now(),
            "state": state,
            "config_sha256": sha256(config_path),
            "data_manifest_sha256": sha256(
                root / "artifacts/phase17/s0_audit/shadow_data_manifest.json"
            ),
            "source_sha256": sha256(resolved["snapshot"]),
            "scientific_result_eligible": state == "COMPLETED",
            "failure_reason": failure_reason,
            "artifact_dir": str(artifact_dir.relative_to(root)),
        }
    )


def run_one_probe(
    root: Path,
    writer: StatusWriter,
    *,
    track_id: str,
    module_id: str,
    kind: str,
    progress_current: int,
) -> dict:
    resolved = paths(root)
    is_overfit = kind == "overfit"
    users = 100 if is_overfit else 1000
    epochs = 4 if is_overfit else 1
    timeout_seconds = 900 if is_overfit else 1800
    dataset = f"Toys_s17_d0_{users}"
    artifact_dir = resolved["output"] / "tracks" / track_id / kind
    if artifact_dir.exists():
        raise FileExistsError(f"probe output already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True)
    transition_map = resolved["output"] / "preflight" / f"transition_teacher_{users}.json"
    command = command_template(
        root,
        track_id=track_id,
        module_id=module_id,
        dataset=dataset,
        epochs=epochs,
        output_dir=artifact_dir,
        transition_map=transition_map,
    )
    attempt_id = f"{track_id.lower()}_{kind}_r2_001"
    config_path = artifact_dir / "config.json"
    started_at = utc_now()
    config = {
        "schema_version": "phase17.s17_2_attempt.v1",
        "attempt_id": attempt_id,
        "track_id": track_id,
        "module_id": module_id,
        "kind": kind,
        "dataset": dataset,
        "epochs": epochs,
        "timeout_seconds": timeout_seconds,
        "command": command,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(config_path, config)
    progress = {
        "current": progress_current,
        "total": len(TRACKS) * 2,
        "track_id": track_id,
        "probe_kind": kind,
    }
    gpu, gpu_records = wait_for_gpu(writer, progress)
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
        "S17_2_PROBE_ADMITTED",
        gpu_ids=[gpu],
        gpu_snapshot={
            "captured_at": utc_now(),
            "devices": gpu_records,
            "selected_gpu": gpu,
            "selection_reason": "lowest-utilization eligible card at admission",
        },
        stage=f"{track_id}_{kind}_admitted",
        progress=progress,
        process_alive=True,
    )
    return_code, pid, wall_seconds = run_process(
        command=command,
        cwd=root / "GRAM/command",
        log_path=artifact_dir / "run.log",
        env=env,
        timeout_seconds=timeout_seconds,
        writer=writer,
        stage=f"{track_id}_{kind}",
        progress=progress,
        gpu=gpu,
    )
    parsed = parse_log(artifact_dir / "run.log")
    passed = (
        return_code == 0
        and not parsed["traceback"]
        and not parsed["forbidden_test_evidence"]
        and parsed["peak_reserved_mib"] is not None
        and parsed["peak_reserved_mib"] <= MAX_USABLE_MIB_PER_JOB
    )
    result = {
        "schema_version": "phase17.s17_2_probe_result.v1",
        "attempt_id": attempt_id,
        "track_id": track_id,
        "module_id": module_id,
        "kind": kind,
        "state": "COMPLETED" if passed else "FAILED",
        "return_code": return_code,
        "workload_pid": pid,
        "physical_gpu": gpu,
        "wall_seconds": wall_seconds,
        "parsed": parsed,
        "overfit_loss_decreased": (
            len(parsed["training_losses"]) >= 2
            and parsed["training_losses"][-1] < parsed["training_losses"][0]
        )
        if is_overfit
        else None,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(artifact_dir / "result.json", result)
    failure_reason = None if passed else "fixed-budget probe failed; see canonical run.log"
    append_attempt(
        root,
        attempt_id=attempt_id,
        track_id=track_id,
        kind=kind,
        started_at=started_at,
        state=result["state"],
        failure_reason=failure_reason,
        config_path=config_path,
        artifact_dir=artifact_dir,
    )
    return result


def diagnostic_labels(track: dict, baseline_ndcg10: float) -> list[str]:
    overfit = track["overfit"]
    short = track["short"]
    labels = []
    if overfit["state"] != "COMPLETED" or short["state"] != "COMPLETED":
        labels.append("INTERFACE_FAILED")
    if overfit["state"] == "COMPLETED" and not overfit["overfit_loss_decreased"]:
        labels.append("UNLEARNABLE")
    peaks = [
        result["parsed"]["peak_reserved_mib"]
        for result in (overfit, short)
        if result["parsed"]["peak_reserved_mib"] is not None
    ]
    if peaks and max(peaks) > MAX_USABLE_MIB_PER_JOB:
        labels.append("EFFECTIVE_BUT_TOO_EXPENSIVE")
    if short["state"] == "COMPLETED" and not short["parsed"]["mechanism_metric_lines"]:
        labels.append("MECHANISM_METRIC_UNCHANGED")
    ndcg10 = short["parsed"]["validation_metrics"].get("ndcg@10")
    if ndcg10 is not None and ndcg10 < baseline_ndcg10:
        labels.append("ACCURACY_NEGATIVE")
    if not any(
        label in labels
        for label in (
            "INTERFACE_FAILED",
            "UNLEARNABLE",
            "EFFECTIVE_BUT_TOO_EXPENSIVE",
            "MECHANISM_METRIC_UNCHANGED",
        )
    ):
        labels.insert(0, "PROBE_PASS")
    return labels


def run_cpu_contracts(root: Path, writer: StatusWriter) -> dict:
    resolved = paths(root)
    log_path = resolved["output"] / "preflight" / "cpu_contract_tests.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    command = [
        str(PYTHON), "-m", "unittest", "discover", "-v",
        "-s", "experiment/phase17/tests", "-p", "test_*.py",
    ]
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
    writer.heartbeat(stage="cpu_contracts_complete", progress={"current": 0, "total": 14})
    return {
        "return_code": completed.returncode,
        "log_path": str(log_path.relative_to(root)),
        "log_sha256": sha256(log_path),
    }


def worker(root: Path) -> int:
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    try:
        verify_run_snapshot(root, resolved["snapshot"])
        cpu = run_cpu_contracts(root, writer)
        if cpu["return_code"] != 0:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_2_CPU_CONTRACTS_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="failed_cpu_contracts",
            )
            return cpu["return_code"]
        progress = 0
        results = {}
        last_successful = None
        for track_id, module_id in TRACKS:
            overfit = run_one_probe(
                root,
                writer,
                track_id=track_id,
                module_id=module_id,
                kind="overfit",
                progress_current=progress,
            )
            progress += 1
            short = run_one_probe(
                root,
                writer,
                track_id=track_id,
                module_id=module_id,
                kind="short",
                progress_current=progress,
            )
            progress += 1
            results[track_id] = {"module_id": module_id, "overfit": overfit, "short": short}
            if short["state"] == "COMPLETED":
                last_successful = (track_id, module_id)
            writer.heartbeat(
                stage=f"{track_id}_complete",
                progress={"current": progress, "total": len(TRACKS) * 2},
            )
        baseline_ndcg10 = 0.045911074355879856
        for value in results.values():
            value["diagnostic_labels"] = diagnostic_labels(value, baseline_ndcg10)
        completed_probe_count = sum(
            result["state"] == "COMPLETED"
            for value in results.values()
            for result in (value["overfit"], value["short"])
        )
        portfolio_has_scientific_evidence = completed_probe_count > 0
        summary = {
            "schema_version": "phase17.s17_2_summary.v1",
            "experiment_id": EXPERIMENT_ID,
            "step_id": "S17-2",
            "verdict": (
                "S17_2_SEVEN_TRACK_PROBES_COMPLETE"
                if portfolio_has_scientific_evidence
                else "S17_2_SHARED_INTERFACE_FAILURE_ZERO_COMPLETED_PROBES"
            ),
            "completed_probe_count": completed_probe_count,
            "cpu_contracts": cpu,
            "baseline_reference": {
                "source": "S17-0 Toys_s17_d0_1000 one-epoch GRAM-B0 resource probe",
                "validation_ndcg@10": baseline_ndcg10,
                "interpretation": "diagnostic reference only",
            },
            "tracks": results,
            "test_read": False,
            "sports_read": False,
            "official_result_claim": False,
            "completed_at": utc_now(),
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED" if portfolio_has_scientific_evidence else "FAILED",
            "SCIENTIFIC_COMPLETED" if portfolio_has_scientific_evidence else "SCIENTIFIC_FAILED",
            (
                "S17_2_SEVEN_TRACK_PROBES_COMPLETE"
                if portfolio_has_scientific_evidence
                else "S17_2_SHARED_INTERFACE_FAILURE_ZERO_COMPLETED_PROBES"
            ),
            workload_pid=0,
            process_alive=True,
            stage="scientific_complete",
            progress={"current": len(TRACKS) * 2, "total": len(TRACKS) * 2},
            result_selection_eligible=portfolio_has_scientific_evidence,
            affects_scientific_result=portfolio_has_scientific_evidence,
            summary_path=str(resolved["summary"].relative_to(root)),
        )
        if portfolio_has_scientific_evidence and last_successful is not None:
            runtime_loop(root, writer, *last_successful)
        return 0
    except Exception as error:
        current = writer.read()
        if current["scientific_state"] not in {"COMPLETED", "FAILED"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_2_ORCHESTRATOR_FAILED",
                workload_pid=0,
                process_alive=False,
                stage="orchestrator_failed",
                failure_reason=f"{type(error).__name__}: {error}",
            )
        raise


def runtime_loop(root: Path, writer: StatusWriter, track_id: str, module_id: str) -> None:
    """Repeat a completed 1k workload in disjoint run-NNNN trees; never score it."""

    canonical = paths(root)["output"]
    iteration = 2
    while True:
        runtime_dir = isolated_runtime_dir(root, EXPERIMENT_ID, iteration)
        assert_runtime_isolation(canonical, runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=False)
        progress = {"current": 14, "total": 14, "runtime_iteration": iteration}
        gpu, gpu_records = wait_for_runtime_gpu(writer, progress)
        transition_map = canonical / "preflight" / "transition_teacher_1000.json"
        command = command_template(
            root,
            track_id=track_id,
            module_id=module_id,
            dataset="Toys_s17_d0_1000",
            epochs=1,
            output_dir=runtime_dir,
            transition_map=transition_map,
        )
        atomic_json(
            runtime_dir / "runtime_config.json",
            {
                "schema_version": "phase17.runtime_cycle.v1",
                "source_track": track_id,
                "module_id": module_id,
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
                "CUDA_VISIBLE_DEVICES": str(gpu),
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
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    writer.heartbeat(
                        stage="runtime_cycle_active",
                        progress=progress,
                        process_alive=True,
                    )
        iteration += 1


def wait_for_runtime_gpu(writer: StatusWriter, progress: dict) -> tuple[int, list[dict]]:
    while True:
        records = query_gpus()
        selected = choose_idle_gpu(
            records,
            expected_peak_mib=EXPECTED_PEAK_MIB,
            safety_margin_mib=SAFETY_MARGIN_MIB,
        )
        if selected is not None:
            return selected.index, snapshot(records)
        payload = writer.read()
        writer.transition(
            "COMPLETED",
            payload["execution_state"],
            payload["status_code"],
            workload_pid=0,
            process_alive=False,
            stage="runtime_waiting_for_gpu",
            progress=progress,
            gpu_snapshot={"captured_at": utc_now(), "devices": snapshot(records)},
        )
        time.sleep(GPU_WAIT_SECONDS)


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
