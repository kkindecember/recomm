#!/usr/bin/env python3
"""Authorized parallel takeover for the remaining S18-1 diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.run_manager import launch_background_tmux
from experiment.phase18.core.contracts import load_json, sha256
from experiment.phase18.protocol import s18_s1_recovery as checkpoint_recovery
from experiment.phase18.protocol import s18_s1_runtime as base


PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
EXPERIMENT_ID = "s18_s1_actionability_parallel_takeover"
ATTEMPT_ID = "run-0004"
AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_parallel_takeover_authorization.json"
OUTPUT = ROOT / "artifacts/phase18/s1_actionability/run-0004"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_parallel_takeover.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1.attempts.jsonl"
RUN2_TOYS_I0 = ROOT / "artifacts/phase18/s1_actionability/run-0002/units/toys_i0"
RUN3_TOYS_IM1 = ROOT / "artifacts/phase18/s1_actionability/run-0003/units/toys_im1"
RUN3_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_resource_recovery.status.json"
CANONICAL_SUMMARY = ROOT / "artifacts/phase18/s1_actionability/summary.json"
CANONICAL_MANIFEST = ROOT / "artifacts/phase18/s1_actionability/canonical_manifest.json"
REPORT = ROOT / "report/第十八阶段/Stage18_S1_并行资源接管报告.md"


def unit_key(domain: str, fold: str) -> str:
    return base.unit_key(domain, fold)


def unit_dir(domain: str, fold: str) -> Path:
    return OUTPUT / "units" / unit_key(domain, fold)


def update_status(*, reset: bool = False, **fields: Any) -> None:
    current = {} if reset or not STATUS.is_file() else load_json(STATUS)
    current.update(fields)
    current["updated_at"] = base.utc_now()
    current["heartbeat_at"] = base.utc_now()
    base.atomic_json(STATUS, current)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def verify_authorization() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, checkpoint_auth = checkpoint_recovery.verify_authorization()
    authorization = load_json(AUTH_PATH)
    if authorization["experiment_id"] != EXPERIMENT_ID or authorization["attempt_id"] != ATTEMPT_ID:
        raise RuntimeError("parallel takeover authorization identity mismatch")
    scope = authorization["correction_scope"]
    required_true = (
        "resource_only",
        "checkpoint_only_diagnostic_recovery",
        "parallel_takeover",
        "preserve_active_toys_worker",
        "generation_use_cache",
        "cross_attention_cache",
        "release_unused_generation_tensors",
        "release_cuda_cache_per_user",
    )
    required_false = (
        "parent_retraining",
        "item_head_retraining",
        "scientific_config_changes",
        "cohort_changes",
        "beam_changes",
        "score_changes_allowed",
        "automatic_retry",
        "automatic_s18_2",
    )
    if any(not scope[name] for name in required_true) or any(scope[name] for name in required_false):
        raise RuntimeError("parallel takeover scope is not result-preserving")
    runtime = authorization["runtime"]
    if runtime["beauty_physical_gpu"] != 0 or runtime["beauty_units"] != ["Beauty:I0", "Beauty:I-1"]:
        raise RuntimeError("parallel takeover GPU0 Beauty schedule mismatch")
    if not runtime["beauty_units_serial"] or not runtime["toys_and_beauty_parallel"]:
        raise RuntimeError("parallel lane contract mismatch")
    if runtime["required_gpu0_free_mib"] != (
        runtime["single_gpu_peak_reserved_mib"] + runtime["single_gpu_memory_buffer_mib"]
    ):
        raise RuntimeError("GPU0 memory threshold mismatch")
    for record in authorization["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen parallel takeover input mismatch: {path}")
    source = authorization["source_takeover"]
    if pid_alive(int(source["controller_pid"])):
        raise RuntimeError("run-0003 serial controller is still alive")
    source_summary = ROOT / source["active_worker_summary"]
    source_status_path = ROOT / source["active_worker_status"]
    source_status = load_json(source_status_path) if source_status_path.is_file() else {}
    matching_live_heartbeat = (
        source_status.get("pid") == int(source["active_worker_pid"])
        and source_status.get("process_alive") is True
        and source_status.get("execution_state") == "RUNNING_BOUNDED_GENERATION"
    )
    if not source_summary.is_file() and not matching_live_heartbeat:
        raise RuntimeError("preserved Toys:I-1 worker is neither alive nor complete")
    return config, checkpoint_auth, authorization


def source_manifest(checkpoint_auth: dict[str, Any]) -> dict[str, str]:
    paths = (
        AUTH_PATH,
        base.CONFIG_PATH,
        base.PREFLIGHT / "manifest.json",
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(checkpoint_recovery.__file__).resolve(),
        ROOT / "experiment/phase18/protocol/s18_s1_resource_recovery.py",
        ROOT / "experiment/phase18/protocol/s18_s1_gpu0_postrun_guard.py",
        ROOT / "experiment/phase18/core/s1_contracts.py",
        ROOT / "experiment/phase18/core/contracts.py",
        ROOT / "GRAM/src/model/gram.py",
        ROOT / "GRAM/src/model/gram_t5.py",
        ROOT / "GRAM/src/model/gram_t5_modeling.py",
    )
    records = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for pair in checkpoint_auth["checkpoints"].values():
        for record in pair.values():
            records[record["path"]] = record["sha256"]
    return records


def validate_completed_unit(source_dir: Path, domain: str, fold: str) -> Path:
    summary_path = source_dir / "summary.json"
    status_path = source_dir / "status.json"
    if not summary_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"completed source unit is incomplete: {source_dir}")
    summary = load_json(summary_path)
    status = load_json(status_path)
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("domain") != domain
        or summary.get("fold") != fold
        or summary.get("users") != 1024
        or summary.get("d1_read")
        or summary.get("d2_read")
        or summary.get("test_read")
        or summary.get("sports_read")
        or summary.get("treatment_training")
    ):
        raise RuntimeError(f"source unit scientific contract mismatch: {source_dir}")
    if status.get("execution_state") != "COMPLETED" or status.get("summary_sha256") != sha256(summary_path):
        raise RuntimeError(f"source unit terminal status mismatch: {source_dir}")
    return summary_path


def carry_forward(label: str, source_path: Path, source_attempt: str) -> None:
    domain, fold = label.split(":", 1)
    target = unit_dir(domain, fold)
    if target.exists():
        raise FileExistsError(f"run-0004 carry target exists; retry forbidden: {target}")
    target.mkdir(parents=True)
    summary = load_json(source_path)
    summary.update(
        resource_recovery_attempt=ATTEMPT_ID,
        carried_forward_from=str(source_path.relative_to(ROOT)),
        carried_forward_sha256=sha256(source_path),
        carried_forward_source_attempt=source_attempt,
    )
    base.atomic_json(target / "summary.json", summary)
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1_parallel_takeover_unit_status.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "unit": unit_key(domain, fold),
            "domain": domain,
            "fold": fold,
            "execution_state": "CARRIED_FORWARD_COMPLETED",
            "process_alive": False,
            "source_attempt": source_attempt,
            "source_summary_path": str(source_path.relative_to(ROOT)),
            "source_summary_sha256": sha256(source_path),
            "summary_path": str((target / "summary.json").relative_to(ROOT)),
            "summary_sha256": sha256(target / "summary.json"),
            "scientific_parameters_changed": False,
            "created_at": base.utc_now(),
        },
    )


def source_toys_status(authorization: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / authorization["source_takeover"]["active_worker_status"]
    return load_json(path) if path.is_file() else {"execution_state": "MISSING"}


def wait_for_gpu0(authorization: dict[str, Any], label: str) -> list[dict[str, Any]]:
    runtime = authorization["runtime"]
    required = int(runtime["required_gpu0_free_mib"])
    started = time.time()
    stable_count = 0
    history: list[dict[str, Any]] = []
    while True:
        snapshot = base.gpu_snapshot()
        history.append({"at": base.utc_now(), "gpus": snapshot})
        row = next((item for item in snapshot if item["index"] == 0), None)
        eligible = row is not None and row["free_mib"] >= required
        stable_count = stable_count + 1 if eligible else 0
        update_status(
            stage="parallel_lanes",
            execution_state="WAITING_FOR_GPU0" if stable_count < runtime["stable_snapshots_required"] else "GPU0_ADMITTED",
            scientific_state="RUNNING",
            process_alive=True,
            waiting_beauty_unit=label,
            gpu0_required_free_mib=required,
            gpu0_stable_snapshots=stable_count,
            gpu_snapshot=snapshot,
            lanes={
                "gpu1_gpu7": {"label": "Toys:I-1", "source_status": source_toys_status(authorization)},
                "gpu0": {"label": label, "state": "WAITING_FOR_ADMISSION"},
            },
        )
        if stable_count >= runtime["stable_snapshots_required"]:
            return history
        if time.time() - started > runtime["gpu_wait_hard_timeout_seconds"]:
            raise TimeoutError(f"GPU0 admission hard timeout for {label}")
        time.sleep(runtime["snapshot_interval_seconds"])


def run_beauty_unit(fold: str, physical_gpu: int) -> int:
    config, checkpoint_auth, authorization = verify_authorization()
    if physical_gpu != authorization["runtime"]["beauty_physical_gpu"]:
        raise RuntimeError("unauthorized Beauty GPU")
    base.OUTPUT = OUTPUT
    target = unit_dir("Beauty", fold)
    if target.exists():
        raise FileExistsError(f"run-0004 Beauty output exists; retry forbidden: {target}")
    target.mkdir(parents=True)
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1_parallel_takeover_unit_status.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "domain": "Beauty",
            "fold": fold,
            "unit": unit_key("Beauty", fold),
            "execution_state": "STARTING_CHECKPOINT_ONLY_DIAGNOSTIC",
            "phase": "checkpoint_load",
            "physical_gpu": physical_gpu,
            "pid": os.getpid(),
            "process_alive": True,
            "parent_retraining": False,
            "item_head_retraining": False,
            "generation_use_cache": True,
            "cross_attention_cache": True,
            "release_cuda_cache_per_user": True,
            "started_at": base.utc_now(),
            "heartbeat_at": base.utc_now(),
        },
    )
    started = time.time()
    try:
        base.set_seed(config["seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("run-0004 Beauty unit requires exactly one visible GPU")
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(config["backbone"]["snapshot"], local_files_only=True)
        parent, args, item_head, item_to_id, frequencies, sequences, provenance = checkpoint_recovery.load_frozen_models(
            config, checkpoint_auth, "Beauty", fold, device
        )
        args.tokenizer = tokenizer
        diagnostic = base.diagnose(
            config,
            "Beauty",
            fold,
            device,
            tokenizer,
            parent,
            args,
            item_head,
            item_to_id,
            frequencies,
            sequences,
            generation_use_cache=True,
            cross_attention_cache=True,
            release_cuda_cache_per_user=True,
        )
        summary = {
            **diagnostic,
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": "s18_s1_actionability/run-0001",
            "status": "COMPLETED",
            "parent_training": provenance["parent"],
            "item_head_training": provenance["item_head"],
            "physical_gpu": physical_gpu,
            "peak_by_visible_gpu": {
                "0": {
                    "allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                    "reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
                    "physical_gpu": physical_gpu,
                }
            },
            "wall_time_total_seconds": time.time() - started,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
            "treatment_training": False,
            "parent_retraining": False,
            "item_head_retraining": False,
            "scientific_parameters_changed": False,
        }
        base.atomic_json(target / "summary.json", summary)
        base.update_unit_status(
            "Beauty",
            fold,
            execution_state="COMPLETED",
            phase="complete",
            process_alive=False,
            summary_path=str((target / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(target / "summary.json"),
            elapsed_seconds=time.time() - started,
        )
        return 0
    except Exception as error:
        base.atomic_text(target / "failure.txt", f"{type(error).__name__}: {error}\n")
        base.update_unit_status(
            "Beauty",
            fold,
            execution_state="FAILED_NO_RETRY",
            phase="failed",
            process_alive=False,
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.time() - started,
        )
        raise


def execute_beauty_unit(label: str, authorization: dict[str, Any]) -> tuple[int, int, Path]:
    _, fold = label.split(":", 1)
    physical_gpu = int(authorization["runtime"]["beauty_physical_gpu"])
    log = OUTPUT / "units" / f"beauty_{fold.lower().replace('-', 'm')}.launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(physical_gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(ROOT),
    )
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "unit",
                "--fold",
                fold,
                "--physical-gpu",
                str(physical_gpu),
            ],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.time()
        timeout = authorization["runtime"]["unit_hard_timeout_seconds"]
        while process.poll() is None:
            if time.time() - started > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124, process.pid, log
            unit_status_path = unit_dir("Beauty", fold) / "status.json"
            update_status(
                stage="parallel_lanes",
                execution_state="RUNNING_PARALLEL_SCIENCE",
                scientific_state="RUNNING",
                process_alive=True,
                lanes={
                    "gpu1_gpu7": {"label": "Toys:I-1", "source_status": source_toys_status(authorization)},
                    "gpu0": {
                        "label": label,
                        "pid": process.pid,
                        "status": load_json(unit_status_path) if unit_status_path.is_file() else None,
                    },
                },
            )
            time.sleep(30)
        return int(process.returncode), process.pid, log


def wait_for_toys_im1(authorization: dict[str, Any]) -> Path:
    source = authorization["source_takeover"]
    started = time.time()
    summary_path = ROOT / source["active_worker_summary"]
    while True:
        status = source_toys_status(authorization)
        if status.get("execution_state") == "COMPLETED":
            return validate_completed_unit(RUN3_TOYS_IM1, "Toys", "I-1")
        if status.get("execution_state") == "FAILED_NO_RETRY":
            raise RuntimeError(f"preserved Toys:I-1 worker failed: {status.get('error')}")
        if not pid_alive(int(source["active_worker_pid"])) and not summary_path.is_file():
            raise RuntimeError("preserved Toys:I-1 worker exited without a summary")
        update_status(
            stage="waiting_preserved_toys_worker",
            execution_state="RUNNING_PARALLEL_SCIENCE",
            scientific_state="RUNNING",
            process_alive=True,
            lanes={
                "gpu1_gpu7": {"label": "Toys:I-1", "source_status": status},
                "gpu0": {"state": "BEAUTY_QUEUE_COMPLETED"},
            },
        )
        if time.time() - started > authorization["runtime"]["source_worker_wait_hard_timeout_seconds"]:
            raise TimeoutError("preserved Toys:I-1 worker wait hard timeout")
        time.sleep(30)


def aggregate(config: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    base.OUTPUT = OUTPUT
    summary = base.aggregate_results(config)
    summary.update(
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of="run-0002 Toys:I0 + run-0003 Toys:I-1 + run-0004 Beauty units",
        parallel_takeover_authorization_sha256=sha256(AUTH_PATH),
        checkpoint_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        scientific_parameters_changed=False,
        resource_adaptation={
            "parallel_lanes": {
                "gpu1_gpu7": "preserved run-0003 Toys:I-1 decoder model parallel worker",
                "gpu0": "run-0004 Beauty:I0 then Beauty:I-1 single-GPU exact path",
            },
            "beam_widths_changed": False,
            "score_changes_allowed": False,
        },
    )
    base.atomic_json(OUTPUT / "summary.json", summary)
    base.atomic_json(CANONICAL_SUMMARY, summary)
    base.atomic_json(
        CANONICAL_MANIFEST,
        {
            "schema_version": "phase18.s18_1_canonical_manifest.v1",
            "canonical_attempt": ATTEMPT_ID,
            "summary_path": str(CANONICAL_SUMMARY.relative_to(ROOT)),
            "summary_sha256": sha256(CANONICAL_SUMMARY),
            "source_run_summary_path": str((OUTPUT / "summary.json").relative_to(ROOT)),
            "source_run_summary_sha256": sha256(OUTPUT / "summary.json"),
            "selected_by_effect": False,
            "reason": "explicit result-preserving parallel resource takeover",
            "created_at": base.utc_now(),
        },
    )
    base.REPORT = REPORT
    base.write_report(summary)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Parallel Takeover\n\n"
            "- `Toys:I0` was carried from run-0002; `Toys:I-1` was preserved from the already-running run-0003 worker.\n"
            "- `Beauty:I0` and `Beauty:I-1` ran serially on physical GPU0 while Toys continued on GPU1+7.\n"
            "- No checkpoint, cohort, beam, score, Gate, or protected-data access changed.\n"
            "- GPU0 repeat occupancy is post-science only and scientifically ineligible.\n"
        )
    return summary


def mark_run3_takeover(authorization: dict[str, Any]) -> None:
    current = load_json(RUN3_STATUS) if RUN3_STATUS.is_file() else {}
    current.update(
        execution_state="CONTROLLER_TERMINATED_FOR_PARALLEL_TAKEOVER",
        scientific_state="PARTIAL_RUNNING_IN_AUTHORIZED_TAKEOVER",
        status_code="S18_1_RUN_0003_CONTROLLER_TERMINATED_WORKER_PRESERVED",
        process_alive=False,
        workload_pid=0,
        preserved_worker_pid=authorization["source_takeover"]["active_worker_pid"],
        takeover_attempt=ATTEMPT_ID,
        takeover_status_path=str(STATUS.relative_to(ROOT)),
        automatic_retry=False,
        automatic_s18_2=False,
        next_action="Observe run-0004 parallel takeover; do not launch another S18-1 attempt.",
        updated_at=base.utc_now(),
        heartbeat_at=base.utc_now(),
    )
    base.atomic_json(RUN3_STATUS, current)


def master() -> int:
    config, checkpoint_auth, authorization = verify_authorization()
    queue_manifest = load_json(OUTPUT / "queue_manifest.json")
    if queue_manifest["source_manifest"] != source_manifest(checkpoint_auth):
        raise RuntimeError("run-0004 source manifest changed after launch")
    try:
        toys_i0 = validate_completed_unit(RUN2_TOYS_I0, "Toys", "I0")
        carry_forward("Toys:I0", toys_i0, "s18_s1_actionability_recovery/run-0002")
        completed = 1
        run_manifest = {
            "schema_version": "phase18.s18_1_parallel_takeover_run_manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "created_at": base.utc_now(),
            "authorization_sha256": sha256(AUTH_PATH),
            "queue_manifest_sha256": sha256(OUTPUT / "queue_manifest.json"),
            "parallel_lanes": {
                "gpu1_gpu7": "preserved run-0003 Toys:I-1",
                "gpu0": authorization["runtime"]["beauty_units"],
            },
            "unit_assignments": [],
            "automatic_retry": False,
            "automatic_s18_2": False,
            "scientific_parameters_changed": False,
        }
        base.atomic_json(OUTPUT / "run_manifest.json", run_manifest)
        for label in authorization["runtime"]["beauty_units"]:
            admission_history = wait_for_gpu0(authorization, label)
            run_manifest["unit_assignments"].append(
                {"label": label, "physical_gpu": 0, "admission_history": admission_history}
            )
            base.atomic_json(OUTPUT / "run_manifest.json", run_manifest)
            return_code, pid, log = execute_beauty_unit(label, authorization)
            if return_code != 0:
                event = "parallel_takeover_unit_hard_timeout" if return_code == 124 else "parallel_takeover_unit_failed_no_retry"
                base.append_jsonl(
                    LEDGER,
                    {"event": event, "at": base.utc_now(), "attempt_id": ATTEMPT_ID, "unit": label, "physical_gpu": 0},
                )
                update_status(
                    stage="terminal_failure",
                    execution_state="FAILED_NO_RETRY",
                    scientific_state="FAILED",
                    status_code="S18_1_PARALLEL_TAKEOVER_BEAUTY_FAILURE_NO_RETRY",
                    process_alive=False,
                    workload_pid=0,
                    failed_unit=label,
                    failed_unit_pid=pid,
                    failed_unit_log=str(log.relative_to(ROOT)),
                    result_selection_eligible=False,
                )
                return 1
            completed += 1
            update_status(progress={"current": completed, "total": 4, "unit": "domain_fold_diagnostic"})
        toys_im1 = wait_for_toys_im1(authorization)
        carry_forward("Toys:I-1", toys_im1, authorization["source_takeover"]["source_attempt"])
        completed += 1
        summary = aggregate(config, authorization)
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_parallel_takeover_completed",
                "at": base.utc_now(),
                "attempt_id": ATTEMPT_ID,
                "decision": summary["decision"],
                "summary_sha256": sha256(OUTPUT / "summary.json"),
            },
        )
        update_status(
            stage="scientific_complete",
            execution_state="SCIENTIFIC_COMPLETED",
            scientific_state="COMPLETED",
            status_code=summary["decision"],
            process_alive=False,
            workload_pid=0,
            progress={"current": completed, "total": 4, "unit": "domain_fold_diagnostic"},
            summary_path=str((OUTPUT / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(OUTPUT / "summary.json"),
            canonical_summary_path=str(CANONICAL_SUMMARY.relative_to(ROOT)),
            report_path=str(REPORT.relative_to(ROOT)),
            result_selection_eligible=True,
            automatic_s18_2=False,
            next_action="Review the S18-1 Gate; do not start S18-2 automatically.",
        )
        guard = subprocess.run(
            [str(PYTHON), str(ROOT / "experiment/phase18/protocol/s18_s1_gpu0_postrun_guard.py"), "launch"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        update_status(
            postrun_occupancy={
                "launch_return_code": guard.returncode,
                "stdout": guard.stdout.strip(),
                "stderr": guard.stderr.strip(),
                "physical_gpu": 0,
                "tmux_session": authorization["postrun_occupancy"]["tmux_session"],
                "status_path": authorization["postrun_occupancy"]["status"],
                "normal_priority_preemption": True,
                "result_selection_eligible": False,
                "repeat_metrics_ignored": True,
            }
        )
        return 0
    except Exception as error:
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_parallel_takeover_failed_no_retry",
                "at": base.utc_now(),
                "attempt_id": ATTEMPT_ID,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        update_status(
            stage="terminal_failure",
            execution_state="FAILED_NO_RETRY",
            scientific_state="FAILED",
            status_code="S18_1_PARALLEL_TAKEOVER_FAILURE_NO_RETRY",
            process_alive=False,
            workload_pid=0,
            error_type=type(error).__name__,
            error=str(error),
            result_selection_eligible=False,
        )
        raise


def launch() -> int:
    _, checkpoint_auth, authorization = verify_authorization()
    if OUTPUT.exists() or STATUS.exists():
        raise FileExistsError("run-0004 artifacts already exist; automatic retry forbidden")
    OUTPUT.mkdir(parents=True)
    queue_manifest = {
        "schema_version": "phase18.s18_1_parallel_takeover_queue_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "created_at": base.utc_now(),
        "authorization_path": str(AUTH_PATH.relative_to(ROOT)),
        "authorization_sha256": sha256(AUTH_PATH),
        "source_manifest": source_manifest(checkpoint_auth),
        "preserved_worker_pid": authorization["source_takeover"]["active_worker_pid"],
        "beauty_physical_gpu": 0,
        "beauty_units": authorization["runtime"]["beauty_units"],
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(OUTPUT / "queue_manifest.json", queue_manifest)
    update_status(
        reset=True,
        schema_version="phase18.status.v1",
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        stage="background_starting",
        execution_state="RUNNING_PARALLEL_SCIENCE",
        scientific_state="RUNNING",
        status_code="S18_1_PARALLEL_TAKEOVER_STARTING",
        process_alive=True,
        workload_pid=0,
        tmux_session=authorization["runtime"]["tmux_session"],
        result_selection_eligible=False,
        affects_scientific_result=True,
        checkpoint_only=True,
        resource_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        scientific_parameters_changed=False,
        automatic_retry=False,
        automatic_s18_2=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
        started_at=base.utc_now(),
        next_action="Observe the two parallel lanes; do not launch another S18-1 attempt.",
    )
    mark_run3_takeover(authorization)
    base.append_jsonl(
        LEDGER,
        {
            "event": "authorized_parallel_takeover_queued",
            "at": base.utc_now(),
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "preserved_worker_pid": authorization["source_takeover"]["active_worker_pid"],
            "beauty_physical_gpu": 0,
            "automatic_retry": False,
        },
    )
    command = [
        "/usr/bin/env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={ROOT}",
        str(PYTHON),
        str(Path(__file__).resolve()),
        "master",
    ]
    launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=command,
        cwd=ROOT,
        tmux_session=authorization["runtime"]["tmux_session"],
        startup_log_path=OUTPUT / "master.log",
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        status = load_json(STATUS)
        if status.get("workload_pid", 0) > 0:
            print(json.dumps({"tmux_session": authorization["runtime"]["tmux_session"], "status": str(STATUS.relative_to(ROOT))}))
            return 0
        time.sleep(1)
    raise RuntimeError("run-0004 background master failed startup handshake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "launch", "master", "unit"))
    parser.add_argument("--fold", choices=("I-1", "I0"))
    parser.add_argument("--physical-gpu", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify":
        verify_authorization()
        print(json.dumps({"status": "VERIFIED", "attempt_id": ATTEMPT_ID}))
        return 0
    if args.action == "launch":
        return launch()
    if args.action == "master":
        update_status(workload_pid=os.getpid(), process_alive=True)
        return master()
    if args.action == "unit":
        if args.fold is None or args.physical_gpu is None:
            raise ValueError("unit requires fold and physical GPU")
        return run_beauty_unit(args.fold, args.physical_gpu)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
