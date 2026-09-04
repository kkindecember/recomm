#!/usr/bin/env python3
"""Explicit run-0003 resource recovery for incomplete S18-1 diagnostics."""

from __future__ import annotations

import argparse
import json
import math
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
EXPERIMENT_ID = "s18_s1_actionability_resource_recovery"
ATTEMPT_ID = "run-0003"
RECOVERY_OF = "s18_s1_actionability_recovery/run-0002"
AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_resource_recovery_authorization.json"
OUTPUT = ROOT / "artifacts/phase18/s1_actionability/run-0003"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_resource_recovery.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1.attempts.jsonl"
RUN2 = ROOT / "artifacts/phase18/s1_actionability/run-0002"
RUN2_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_recovery.status.json"
CANONICAL_SUMMARY = ROOT / "artifacts/phase18/s1_actionability/summary.json"
CANONICAL_MANIFEST = ROOT / "artifacts/phase18/s1_actionability/canonical_manifest.json"
REPORT = ROOT / "report/第十八阶段/Stage18_S1_资源适配恢复报告.md"
LABELS = ("Toys:I0", "Toys:I-1", "Beauty:I0", "Beauty:I-1")


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


def verify_authorization() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, checkpoint_auth = checkpoint_recovery.verify_authorization()
    authorization = load_json(AUTH_PATH)
    if authorization["experiment_id"] != EXPERIMENT_ID or authorization["attempt_id"] != ATTEMPT_ID:
        raise RuntimeError("resource recovery authorization identity mismatch")
    if authorization["recovery_of"] != RECOVERY_OF:
        raise RuntimeError("resource recovery source mismatch")
    scope = authorization["correction_scope"]
    required_true = (
        "resource_only",
        "checkpoint_only_diagnostic_recovery",
        "generation_use_cache",
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
        raise RuntimeError("resource recovery scope is not result-preserving")
    if scope["cross_attention_cache"] is not True or scope["decoder_model_parallel"] is not True:
        raise RuntimeError("resource recovery must retain caches and use decoder model parallelism")
    if 1 not in authorization["runtime"]["candidate_physical_gpus"]:
        raise RuntimeError("researcher-authorized GPU1 is absent")
    if not authorization["runtime"]["serial_units"]:
        raise RuntimeError("run-0003 must execute units serially")
    for record in authorization["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen resource recovery input mismatch: {path}")
    return config, checkpoint_auth, authorization


def source_manifest(checkpoint_auth: dict[str, Any]) -> dict[str, str]:
    paths = (
        AUTH_PATH,
        base.CONFIG_PATH,
        base.PREFLIGHT / "manifest.json",
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(checkpoint_recovery.__file__).resolve(),
        ROOT / "experiment/phase18/protocol/s18_s1_memory_smoke.py",
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


def run2_is_terminal() -> bool:
    if not RUN2_STATUS.is_file():
        return False
    status = load_json(RUN2_STATUS)
    return not status.get("process_alive", True) and status.get("scientific_state") in {"FAILED", "COMPLETED"}


def validate_memory_smoke(
    authorization: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, int]]:
    contract = authorization["memory_smoke"]
    path = ROOT / contract["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    smoke = load_json(path)
    parity = smoke.get("parity", {})
    if smoke.get("status") != "PASSED" or smoke.get("max_users") != contract["required_users"]:
        raise RuntimeError("required multi-user memory smoke did not pass")
    if (
        smoke.get("generation_use_cache") is not True
        or smoke.get("cross_attention_cache") is not True
        or smoke.get("release_cuda_cache_per_user") is not True
        or smoke.get("decoder_model_parallel") is not True
    ):
        raise RuntimeError("memory smoke did not use the authorized runtime mode")
    authorized_pair = authorization["runtime"]["physical_gpu_pairs"][0]
    if smoke.get("physical_gpus") != authorized_pair:
        raise RuntimeError("memory smoke physical GPU pair mismatch")
    if not parity.get("candidate_order_exact") or not parity.get("diagnostic_record_exact"):
        raise RuntimeError("memory smoke result parity failed")
    if any(float(value) > float(contract["score_tolerance"]) for value in parity["score_max_abs_deltas"].values()):
        raise RuntimeError("memory smoke score parity failed")
    required_free_mib = {
        int(row["physical_gpu"]): int(math.ceil(float(row["reserved_mib"])))
        + int(contract["memory_buffer_mib"])
        for row in smoke["peak_by_visible_gpu"].values()
    }
    return smoke, required_free_mib


def wait_for_prerequisites(
    authorization: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, int]]:
    started = time.time()
    timeout = authorization["runtime"]["prerequisite_hard_timeout_seconds"]
    while True:
        smoke_path = ROOT / authorization["memory_smoke"]["path"]
        smoke_ready = smoke_path.is_file()
        run2_terminal = run2_is_terminal()
        update_status(
            stage="waiting_prerequisites",
            execution_state="WAITING_PREREQUISITES",
            scientific_state="NOT_STARTED",
            process_alive=True,
            workload_pid=os.getpid(),
            prerequisites={"memory_smoke_passed": smoke_ready, "run_0002_terminal": run2_terminal},
            elapsed_seconds=time.time() - started,
        )
        if smoke_ready and run2_terminal:
            return validate_memory_smoke(authorization)
        smoke_status_path = ROOT / "artifacts/phase18/status/s18_s1_memory_smoke.status.json"
        if smoke_status_path.is_file():
            smoke_status = load_json(smoke_status_path)
            if (
                smoke_status.get("attempt_id") == "run-0003-cache-on-release-u32"
                and smoke_status.get("physical_gpus")
                == authorization["runtime"]["physical_gpu_pairs"][0]
                and smoke_status.get("execution_state") == "FAILED_NO_RETRY"
            ):
                raise RuntimeError(f"required memory smoke failed: {smoke_status.get('error')}")
        if time.time() - started > timeout:
            raise TimeoutError("prerequisite hard timeout")
        time.sleep(authorization["runtime"]["wait_poll_seconds"])


def completed_run2_units() -> dict[str, Path]:
    completed: dict[str, Path] = {}
    for label in LABELS:
        domain, fold = label.split(":", 1)
        source = RUN2 / "units" / unit_key(domain, fold) / "summary.json"
        status_path = RUN2 / "units" / unit_key(domain, fold) / "status.json"
        if not source.is_file() or not status_path.is_file():
            continue
        status = load_json(status_path)
        if status.get("execution_state") == "COMPLETED" and status.get("summary_sha256") == sha256(source):
            completed[label] = source
    return completed


def carry_forward(label: str, source: Path) -> None:
    domain, fold = label.split(":", 1)
    target = unit_dir(domain, fold)
    target.mkdir(parents=True)
    summary = load_json(source)
    summary["carried_forward_from"] = str(source.relative_to(ROOT))
    summary["carried_forward_sha256"] = sha256(source)
    summary["resource_recovery_attempt"] = ATTEMPT_ID
    base.atomic_json(target / "summary.json", summary)
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1_resource_recovery_unit_status.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "unit": unit_key(domain, fold),
            "domain": domain,
            "fold": fold,
            "execution_state": "CARRIED_FORWARD_COMPLETED",
            "process_alive": False,
            "source_summary_path": str(source.relative_to(ROOT)),
            "source_summary_sha256": sha256(source),
            "summary_path": str((target / "summary.json").relative_to(ROOT)),
            "summary_sha256": sha256(target / "summary.json"),
            "scientific_parameters_changed": False,
            "created_at": base.utc_now(),
        },
    )


def wait_for_gpu_pair(
    authorization: dict[str, Any],
    required_free_mib: dict[int, int],
    label: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    runtime = authorization["runtime"]
    pair = list(runtime["physical_gpu_pairs"][0])
    started = time.time()
    history: list[dict[str, Any]] = []
    stable_count = 0
    while True:
        snapshot = base.gpu_snapshot()
        history.append({"at": base.utc_now(), "gpus": snapshot})
        by_index = {row["index"]: row for row in snapshot}
        eligible = all(
            gpu in by_index and by_index[gpu]["free_mib"] >= required_free_mib[gpu]
            for gpu in pair
        )
        stable_count = stable_count + 1 if eligible else 0
        update_status(
            stage="waiting_gpu",
            execution_state="WAITING_FOR_GPU",
            scientific_state="NOT_STARTED",
            process_alive=True,
            waiting_unit=label,
            required_free_mib_by_gpu=required_free_mib,
            physical_gpu_pair=pair,
            stable_snapshots=stable_count,
            gpu_snapshot=snapshot,
        )
        if stable_count >= runtime["stable_snapshots_required"]:
            return pair, history
        if time.time() - started > runtime["gpu_wait_hard_timeout_seconds"]:
            raise TimeoutError(f"GPU-pair wait hard timeout for {label}")
        time.sleep(runtime["snapshot_interval_seconds"])


def run_unit(
    domain: str,
    fold: str,
    physical_gpu: int,
    secondary_physical_gpu: int,
) -> int:
    config, checkpoint_auth, authorization = verify_authorization()
    physical_gpus = [physical_gpu, secondary_physical_gpu]
    if physical_gpus not in authorization["runtime"]["physical_gpu_pairs"]:
        raise RuntimeError(f"unauthorized physical GPU pair: {physical_gpus}")
    base.OUTPUT = OUTPUT
    target = unit_dir(domain, fold)
    if target.exists():
        raise FileExistsError(f"run-0003 unit output exists; retry forbidden: {target}")
    target.mkdir(parents=True)
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1_resource_recovery_unit_status.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "domain": domain,
            "fold": fold,
            "unit": unit_key(domain, fold),
            "execution_state": "STARTING_CHECKPOINT_ONLY_DIAGNOSTIC",
            "phase": "checkpoint_load",
            "physical_gpus": physical_gpus,
            "pid": os.getpid(),
            "process_alive": True,
            "parent_retraining": False,
            "item_head_retraining": False,
            "generation_use_cache": True,
            "cross_attention_cache": True,
            "decoder_model_parallel": True,
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
        if torch.cuda.device_count() != 2:
            raise RuntimeError("run-0003 unit requires exactly two visible GPUs")
        for visible_gpu in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(visible_gpu)
        tokenizer = AutoTokenizer.from_pretrained(config["backbone"]["snapshot"], local_files_only=True)
        parent, args, item_head, item_to_id, frequencies, sequences, provenance = checkpoint_recovery.load_frozen_models(
            config, checkpoint_auth, domain, fold, device
        )
        decoder_device_map = base.enable_two_gpu_decoder_parallel(parent)
        args.tokenizer = tokenizer
        diagnostic = base.diagnose(
            config,
            domain,
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
        peak_by_visible_gpu = {
            str(index): {
                "allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
                "reserved_mib": torch.cuda.max_memory_reserved(index) / 1024**2,
                "physical_gpu": physical_gpus[index],
            }
            for index in range(torch.cuda.device_count())
        }
        summary = {
            **diagnostic,
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "status": "COMPLETED",
            "parent_training": provenance["parent"],
            "item_head_training": provenance["item_head"],
            "physical_gpus": physical_gpus,
            "decoder_device_map": decoder_device_map,
            "peak_by_visible_gpu": peak_by_visible_gpu,
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
            domain,
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
            domain,
            fold,
            execution_state="FAILED_NO_RETRY",
            phase="failed",
            process_alive=False,
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.time() - started,
        )
        raise


def execute_unit(
    label: str,
    physical_gpus: list[int],
    timeout: int,
) -> tuple[int, int, Path]:
    domain, fold = label.split(":", 1)
    log = OUTPUT / "units" / f"{unit_key(domain, fold)}.launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_gpus))
        process = subprocess.Popen(
            [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "unit",
                "--domain",
                domain,
                "--fold",
                fold,
                "--physical-gpu",
                str(physical_gpus[0]),
                "--secondary-physical-gpu",
                str(physical_gpus[1]),
            ],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.time()
        while process.poll() is None:
            if time.time() - started > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124, process.pid, log
            status_path = unit_dir(domain, fold) / "status.json"
            update_status(
                stage="checkpoint_only_diagnostics",
                execution_state="RUNNING_SCIENTIFIC",
                scientific_state="RUNNING",
                process_alive=True,
                current_unit=label,
                current_unit_pid=process.pid,
                current_unit_gpus=physical_gpus,
                current_unit_status=load_json(status_path) if status_path.is_file() else None,
            )
            time.sleep(30)
        return int(process.returncode), process.pid, log


def aggregate(config: dict[str, Any], authorization: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    base.OUTPUT = OUTPUT
    summary = base.aggregate_results(config)
    summary.update(
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of=RECOVERY_OF,
        resource_recovery_authorization_sha256=sha256(AUTH_PATH),
        memory_smoke_sha256=sha256(ROOT / authorization["memory_smoke"]["path"]),
        memory_smoke_peak_reserved_mib=smoke["peak_reserved_mib"],
        checkpoint_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        scientific_parameters_changed=False,
        resource_adaptation={
            "generation_use_cache": True,
            "cross_attention_cache": True,
            "release_cuda_cache_per_user": True,
            "decoder_model_parallel": True,
            "physical_gpu_pair": authorization["runtime"]["physical_gpu_pairs"][0],
            "beam_widths_changed": False,
        },
    )
    base.atomic_json(OUTPUT / "summary.json", summary)
    base.atomic_json(CANONICAL_SUMMARY, summary)
    base.atomic_json(
        CANONICAL_MANIFEST,
        {
            "schema_version": "phase18.s18_1_canonical_manifest.v1",
            "canonical_attempt": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "summary_path": str(CANONICAL_SUMMARY.relative_to(ROOT)),
            "summary_sha256": sha256(CANONICAL_SUMMARY),
            "source_run_summary_path": str((OUTPUT / "summary.json").relative_to(ROOT)),
            "source_run_summary_sha256": sha256(OUTPUT / "summary.json"),
            "selected_by_effect": False,
            "reason": "explicit resource-only recovery after concurrent-workload OOM; cache-on outputs passed exact parity",
            "created_at": base.utc_now(),
        },
    )
    base.REPORT = REPORT
    base.write_report(summary)
    return summary


def master() -> int:
    config, checkpoint_auth, authorization = verify_authorization()
    queue_manifest = load_json(OUTPUT / "queue_manifest.json")
    if queue_manifest["source_manifest"] != source_manifest(checkpoint_auth):
        raise RuntimeError("run-0003 source manifest changed after launch")
    base.append_jsonl(
        LEDGER,
        {
            "event": "authorized_resource_recovery_queued",
            "at": base.utc_now(),
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "automatic_retry": False,
        },
    )
    try:
        smoke, required_free_mib = wait_for_prerequisites(authorization)
        carried = completed_run2_units()
        for label, source in carried.items():
            carry_forward(label, source)
        missing = [label for label in LABELS if label not in carried]
        run_manifest = {
            "schema_version": "phase18.s18_1_resource_recovery_run_manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "created_at": base.utc_now(),
            "queue_manifest_sha256": sha256(OUTPUT / "queue_manifest.json"),
            "authorization_sha256": sha256(AUTH_PATH),
            "memory_smoke_path": authorization["memory_smoke"]["path"],
            "memory_smoke_sha256": sha256(ROOT / authorization["memory_smoke"]["path"]),
            "required_free_mib_by_gpu": required_free_mib,
            "candidate_physical_gpus": authorization["runtime"]["candidate_physical_gpus"],
            "physical_gpu_pairs": authorization["runtime"]["physical_gpu_pairs"],
            "carried_forward_units": {label: str(path.relative_to(ROOT)) for label, path in carried.items()},
            "units_to_execute": missing,
            "unit_assignments": [],
            "checkpoint_only": True,
            "resource_only": True,
            "scientific_parameters_changed": False,
            "automatic_retry": False,
            "automatic_s18_2": False,
        }
        base.atomic_json(OUTPUT / "run_manifest.json", run_manifest)
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_resource_recovery_started",
                "at": base.utc_now(),
                "attempt_id": ATTEMPT_ID,
                "units_to_execute": missing,
                "carried_forward_units": sorted(carried),
                "required_free_mib_by_gpu": required_free_mib,
            },
        )
        completed = len(carried)
        for label in missing:
            physical_gpus, admission_history = wait_for_gpu_pair(
                authorization, required_free_mib, label
            )
            run_manifest["unit_assignments"].append(
                {
                    "label": label,
                    "physical_gpus": physical_gpus,
                    "admission_history": admission_history,
                }
            )
            base.atomic_json(OUTPUT / "run_manifest.json", run_manifest)
            return_code, pid, log = execute_unit(
                label,
                physical_gpus,
                authorization["runtime"]["unit_hard_timeout_seconds"],
            )
            if return_code != 0:
                event = "authorized_resource_recovery_hard_timeout" if return_code == 124 else "authorized_resource_recovery_failed_no_retry"
                base.append_jsonl(
                    LEDGER,
                    {
                        "event": event,
                        "at": base.utc_now(),
                        "attempt_id": ATTEMPT_ID,
                        "unit": label,
                        "physical_gpus": physical_gpus,
                    },
                )
                update_status(
                    stage="terminal_failure",
                    execution_state="FAILED_NO_RETRY",
                    scientific_state="FAILED",
                    status_code="S18_1_RESOURCE_RECOVERY_UNIT_FAILURE_NO_RETRY",
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

        summary = aggregate(config, authorization, smoke)
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_resource_recovery_completed",
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
            progress={"current": 4, "total": 4, "unit": "domain_fold_diagnostic"},
            summary_path=str((OUTPUT / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(OUTPUT / "summary.json"),
            canonical_summary_path=str(CANONICAL_SUMMARY.relative_to(ROOT)),
            report_path=str(REPORT.relative_to(ROOT)),
            result_selection_eligible=True,
            automatic_s18_2=False,
            next_action="Review the S18-1 Gate; do not start S18-2 automatically.",
        )
        guard = subprocess.run(
            [str(PYTHON), str(ROOT / "experiment/phase18/protocol/s18_s1_postrun_guard.py"), "launch"],
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
                "physical_gpu": authorization["postrun_occupancy"]["physical_gpu"],
                "tmux_session": authorization["postrun_occupancy"]["tmux_session"],
                "result_selection_eligible": False,
                "repeat_metrics_ignored": True,
            }
        )
        return 0
    except Exception as error:
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_resource_recovery_failed_no_retry",
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
            status_code="S18_1_RESOURCE_RECOVERY_FAILURE_NO_RETRY",
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
        raise FileExistsError("run-0003 artifacts already exist; automatic retry forbidden")
    OUTPUT.mkdir(parents=True)
    queue_manifest = {
        "schema_version": "phase18.s18_1_resource_recovery_queue_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "recovery_of": RECOVERY_OF,
        "created_at": base.utc_now(),
        "authorization_path": str(AUTH_PATH.relative_to(ROOT)),
        "authorization_sha256": sha256(AUTH_PATH),
        "source_manifest": source_manifest(checkpoint_auth),
        "candidate_physical_gpus": authorization["runtime"]["candidate_physical_gpus"],
        "physical_gpu_pairs": authorization["runtime"]["physical_gpu_pairs"],
        "waits_for_memory_smoke": authorization["memory_smoke"]["path"],
        "waits_for_run_0002_terminal": str(RUN2_STATUS.relative_to(ROOT)),
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(OUTPUT / "queue_manifest.json", queue_manifest)
    session = authorization["runtime"]["tmux_session"]
    update_status(
        reset=True,
        schema_version="phase18.status.v1",
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of=RECOVERY_OF,
        stage="background_starting",
        execution_state="WAITING_PREREQUISITES",
        scientific_state="NOT_STARTED",
        status_code="S18_1_RESOURCE_RECOVERY_QUEUED",
        process_alive=True,
        workload_pid=0,
        tmux_session=session,
        candidate_physical_gpus=authorization["runtime"]["candidate_physical_gpus"],
        physical_gpu_pairs=authorization["runtime"]["physical_gpu_pairs"],
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
        next_action="Wait for the authorized memory smoke and run-0002 terminal state.",
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
        tmux_session=session,
        startup_log_path=OUTPUT / "master.log",
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        status = load_json(STATUS)
        if status.get("workload_pid", 0) > 0:
            print(json.dumps({"tmux_session": session, "status": str(STATUS.relative_to(ROOT))}))
            return 0
        time.sleep(1)
    raise RuntimeError("run-0003 background master failed startup handshake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "launch", "master", "unit"))
    parser.add_argument("--domain", choices=("Toys", "Beauty"))
    parser.add_argument("--fold", choices=("I-1", "I0"))
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--secondary-physical-gpu", type=int)
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
        return master()
    if args.action == "unit":
        if (
            args.domain is None
            or args.fold is None
            or args.physical_gpu is None
            or args.secondary_physical_gpu is None
        ):
            raise ValueError("unit requires domain, fold, and two physical GPUs")
        return run_unit(
            args.domain,
            args.fold,
            args.physical_gpu,
            args.secondary_physical_gpu,
        )
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
