#!/usr/bin/env python3
"""Beauty-only checkpoint recovery for the incomplete S18-1 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
ENTRY_PATH = Path(__file__).resolve()
EXPERIMENT_ID = "s18_s1_actionability_beauty_recovery"
ATTEMPT_ID = "run-0006"
RECOVERY_OF = "s18_s1_actionability_parallel_takeover_mp2/run-0005"
AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_beauty_recovery_authorization.json"
OUTPUT = ROOT / "artifacts/phase18/s1_actionability/run-0006"
SMOKE = ROOT / "artifacts/phase18/s1_actionability/beauty-resource-smoke-run-0006"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_beauty_recovery.status.json"
SMOKE_STATUS = ROOT / "artifacts/phase18/status/s18_s1_beauty_memory_smoke_run0006.status.json"
CANONICAL_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability.status.json"
STATUS_ARCHIVE = ROOT / "artifacts/phase18/status/history/s18_s1_actionability.run-0001.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1.attempts.jsonl"
CANONICAL_SUMMARY = ROOT / "artifacts/phase18/s1_actionability/summary.json"
CANONICAL_MANIFEST = ROOT / "artifacts/phase18/s1_actionability/canonical_manifest.json"
REPORT = ROOT / "report/第十八阶段/Stage18_S1_可作用性与first-drop诊断报告.md"
TOYS_I0 = ROOT / "artifacts/phase18/s1_actionability/run-0002/units/toys_i0"
TOYS_IM1 = ROOT / "artifacts/phase18/s1_actionability/run-0003/units/toys_im1"


def configure_attempt(
    *,
    entry_path: Path,
    experiment_id: str,
    attempt_id: str,
    recovery_of: str,
    auth_path: Path,
    output: Path,
    smoke: Path,
    status: Path,
    smoke_status: Path,
    status_archive: Path,
) -> None:
    """Bind the reusable recovery engine to one immutable attempt namespace."""

    global ENTRY_PATH, EXPERIMENT_ID, ATTEMPT_ID, RECOVERY_OF
    global AUTH_PATH, OUTPUT, SMOKE, STATUS, SMOKE_STATUS, STATUS_ARCHIVE
    ENTRY_PATH = entry_path.resolve()
    EXPERIMENT_ID = experiment_id
    ATTEMPT_ID = attempt_id
    RECOVERY_OF = recovery_of
    AUTH_PATH = auth_path
    OUTPUT = output
    SMOKE = smoke
    STATUS = status
    SMOKE_STATUS = smoke_status
    STATUS_ARCHIVE = status_archive


def status_code(suffix: str) -> str:
    attempt = ATTEMPT_ID.upper().replace("-", "_")
    return f"S18_1_{attempt}_{suffix}"


def unit_key(domain: str, fold: str) -> str:
    return base.unit_key(domain, fold)


def unit_dir(root: Path, domain: str, fold: str) -> Path:
    return root / "units" / unit_key(domain, fold)


def update_json_status(path: Path, *, reset: bool = False, **fields: Any) -> dict[str, Any]:
    current = {} if reset or not path.is_file() else load_json(path)
    current.update(fields)
    current["updated_at"] = base.utc_now()
    current["heartbeat_at"] = base.utc_now()
    base.atomic_json(path, current)
    return current


def update_status(*, reset: bool = False, **fields: Any) -> None:
    payload = update_json_status(STATUS, reset=reset, **fields)
    base.atomic_json(CANONICAL_STATUS, payload)


def verify_frozen_record(record: dict[str, str]) -> Path:
    path = ROOT / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise RuntimeError(f"frozen input mismatch: {path}")
    return path


def validate_completed_unit(
    source_dir: Path,
    domain: str,
    fold: str,
    expected_sha256: str,
) -> Path:
    summary_path = source_dir / "summary.json"
    status_path = source_dir / "status.json"
    if not summary_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"completed source unit is incomplete: {source_dir}")
    if sha256(summary_path) != expected_sha256:
        raise RuntimeError(f"completed source summary drift: {summary_path}")
    summary = load_json(summary_path)
    status = load_json(status_path)
    protected = ("d1_read", "d2_read", "test_read", "sports_read", "treatment_training")
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("domain") != domain
        or summary.get("fold") != fold
        or summary.get("users") != 1024
        or any(summary.get(name) for name in protected)
    ):
        raise RuntimeError(f"completed source unit contract mismatch: {source_dir}")
    if status.get("execution_state") != "COMPLETED" or status.get("summary_sha256") != expected_sha256:
        raise RuntimeError(f"completed source unit status mismatch: {source_dir}")
    return summary_path


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTH_PATH)
    config = load_json(base.CONFIG_PATH)
    checkpoint_auth = load_json(checkpoint_recovery.AUTH_PATH)
    if authorization.get("experiment_id") != EXPERIMENT_ID or authorization.get("attempt_id") != ATTEMPT_ID:
        raise RuntimeError(f"{ATTEMPT_ID} authorization identity mismatch")
    if authorization.get("recovery_of") != RECOVERY_OF:
        raise RuntimeError(f"{ATTEMPT_ID} recovery source mismatch")
    scope = authorization["correction_scope"]
    required_true = (
        "resource_only",
        "checkpoint_only_diagnostic_recovery",
        "beauty_only_execution",
        "carry_completed_toys",
        "generation_use_cache",
        "cross_attention_cache",
        "release_unused_generation_tensors",
        "decoder_model_parallel",
    )
    required_false = (
        "parent_retraining",
        "item_head_retraining",
        "scientific_config_changes",
        "cohort_changes",
        "beam_changes",
        "score_changes_allowed",
        "protected_data_access",
        "automatic_retry",
        "automatic_s18_2",
    )
    release_cache = bool(scope.get("release_cuda_cache_per_user"))
    retain_cache = bool(scope.get("retain_allocator_cache_between_users"))
    if (
        any(not scope.get(name) for name in required_true)
        or any(scope.get(name) for name in required_false)
        or release_cache == retain_cache
    ):
        raise RuntimeError(f"{ATTEMPT_ID} correction scope is not result-preserving")
    for record in authorization["frozen_inputs"].values():
        verify_frozen_record(record)
    terminal_key = authorization.get("terminal_status_input", "run_0005_terminal_status")
    if terminal_key not in authorization["frozen_inputs"]:
        raise RuntimeError(f"{ATTEMPT_ID} terminal status input is missing")
    source_status = load_json(ROOT / authorization["frozen_inputs"][terminal_key]["path"])
    if (
        source_status.get("scientific_state") != "FAILED"
        or source_status.get("execution_state") != "FAILED_NO_RETRY"
        or source_status.get("failed_unit") != "Beauty:I0"
        or source_status.get("d1_read")
        or source_status.get("d2_read")
        or source_status.get("test_read")
        or source_status.get("sports_read")
    ):
        raise RuntimeError(f"{RECOVERY_OF} is not the expected protected terminal failure")
    if checkpoint_auth.get("attempt_id") != "run-0002":
        raise RuntimeError("checkpoint authorization identity changed")
    for label, pair in checkpoint_auth["checkpoints"].items():
        for role, record in pair.items():
            path = ROOT / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"{label} {role} checkpoint mismatch")
    frozen = authorization["frozen_inputs"]
    validate_completed_unit(TOYS_I0, "Toys", "I0", frozen["toys_i0_summary"]["sha256"])
    validate_completed_unit(TOYS_IM1, "Toys", "I-1", frozen["toys_im1_summary"]["sha256"])
    runtime = authorization["runtime"]
    if runtime["beauty_units"] != ["Beauty:I0", "Beauty:I-1"]:
        raise RuntimeError(f"{ATTEMPT_ID} must execute exactly the two incomplete Beauty units")
    if len(runtime["beauty_physical_gpus"]) != 2 or len(set(runtime["beauty_physical_gpus"])) != 2:
        raise RuntimeError(f"{ATTEMPT_ID} requires two distinct physical GPUs")
    reservation = runtime.get("allocator_reservation_mib_by_gpu", {})
    if retain_cache:
        if not scope.get("preclaim_allocator_reservation"):
            raise RuntimeError(f"{ATTEMPT_ID} allocator preclaim is not authorized")
        expected = {str(gpu) for gpu in runtime["beauty_physical_gpus"]}
        if set(reservation) != expected or any(int(value) <= 0 for value in reservation.values()):
            raise RuntimeError(f"{ATTEMPT_ID} allocator reservation contract is invalid")
    elif reservation:
        raise RuntimeError(f"{ATTEMPT_ID} cannot reserve allocator memory while releasing its cache")
    return config, checkpoint_auth, authorization


def source_manifest(
    checkpoint_auth: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, str]:
    paths = [
        AUTH_PATH,
        base.CONFIG_PATH,
        base.PREFLIGHT / "manifest.json",
        Path(__file__).resolve(),
        ENTRY_PATH,
        Path(base.__file__).resolve(),
        Path(checkpoint_recovery.__file__).resolve(),
        ROOT / "experiment/phase18/core/contracts.py",
        ROOT / "experiment/phase18/core/s1_contracts.py",
        ROOT / "GRAM/src/model/gram.py",
        ROOT / "GRAM/src/model/gram_t5.py",
        ROOT / "GRAM/src/model/gram_t5_modeling.py",
    ]
    records = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for record in authorization["frozen_inputs"].values():
        records[record["path"]] = record["sha256"]
    for pair in checkpoint_auth["checkpoints"].values():
        for record in pair.values():
            records[record["path"]] = record["sha256"]
    return records


def first_tsv_row(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            return next(reader), next(reader)
        except StopIteration as error:
            raise RuntimeError(f"missing first data row: {path}") from error


def compare_float_lists(left: str, right: str, tolerance: float = 1e-6) -> float:
    lhs = np.asarray([float(value) for value in left.split("||")], dtype=np.float64)
    rhs = np.asarray([float(value) for value in right.split("||")], dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"parity score shape mismatch: {lhs.shape} != {rhs.shape}")
    delta = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
    if not math.isfinite(delta) or delta > tolerance:
        raise RuntimeError(f"parity score mismatch: {delta} > {tolerance}")
    return delta


def compare_first_beauty_user(candidate: Path, authorization: dict[str, Any]) -> dict[str, Any]:
    frozen = authorization["frozen_inputs"]
    maxima: dict[str, float] = {}
    for name, key in (
        ("beams_w50.tsv", "beauty_i0_partial_beam50_reference"),
        ("beams_w200.tsv", "beauty_i0_partial_beam200_reference"),
    ):
        reference_header, reference_row = first_tsv_row(ROOT / frozen[key]["path"])
        candidate_header, candidate_row = first_tsv_row(candidate / name)
        if candidate_header != reference_header or candidate_row[:3] != reference_row[:3]:
            raise RuntimeError(f"{name}: first-user identity or candidate ordering changed")
        maxima[f"{name}:normalized"] = compare_float_lists(candidate_row[3], reference_row[3])
        maxima[f"{name}:raw"] = compare_float_lists(candidate_row[4], reference_row[4])
    reference_diag = json.loads(
        (ROOT / frozen["beauty_i0_partial_diagnostic_reference"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    candidate_diag = json.loads(
        (candidate / "per_user_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    if candidate_diag != reference_diag:
        raise RuntimeError("Beauty first-user diagnostic parity changed")
    return {
        "status": "PASSED",
        "candidate_order_exact": True,
        "diagnostic_record_exact": True,
        "score_max_abs_deltas": maxima,
        "score_tolerance": 1e-6,
    }


def safe_gpu_snapshot() -> list[dict[str, Any]] | dict[str, str]:
    try:
        return base.gpu_snapshot()
    except Exception as error:
        return {"error_type": type(error).__name__, "error": str(error)}


def validate_pair_free(
    physical_gpus: list[int], required: dict[int, int]
) -> list[dict[str, Any]]:
    snapshot = base.gpu_snapshot()
    by_index = {row["index"]: row for row in snapshot}
    for gpu in physical_gpus:
        if gpu not in by_index or by_index[gpu]["free_mib"] < required[gpu]:
            observed = by_index.get(gpu, {}).get("free_mib")
            raise RuntimeError(
                f"GPU{gpu} has {observed} MiB free; {ATTEMPT_ID} requires {required[gpu]} MiB"
            )
    return snapshot


def stable_pair_admission(
    authorization: dict[str, Any], required: dict[int, int]
) -> list[dict[str, Any]]:
    runtime = authorization["runtime"]
    physical_gpus = runtime["beauty_physical_gpus"]
    history: list[dict[str, Any]] = []
    for index in range(runtime["stable_snapshots_required"]):
        snapshot = validate_pair_free(physical_gpus, required)
        history.append({"at": base.utc_now(), "gpus": snapshot})
        if index + 1 < runtime["stable_snapshots_required"]:
            time.sleep(runtime["snapshot_interval_seconds"])
    return history


def prime_allocator_reservation(
    physical_gpus: list[int], authorization: dict[str, Any]
) -> dict[str, dict[str, float | int]]:
    """Claim a reusable CUDA allocator pool before long shared-GPU generation.

    The tensors are deleted immediately but ``empty_cache`` is intentionally not
    called.  PyTorch can reuse the cached blocks while unrelated processes cannot
    consume the memory between users.
    """

    targets = {
        int(gpu): int(value)
        for gpu, value in authorization["runtime"]
        .get("allocator_reservation_mib_by_gpu", {})
        .items()
    }
    if not targets:
        return {}
    holders: list[torch.Tensor] = []
    for visible_gpu, physical_gpu in enumerate(physical_gpus):
        target_mib = targets[physical_gpu]
        with torch.cuda.device(visible_gpu):
            free_bytes, _ = torch.cuda.mem_get_info(visible_gpu)
            free_mib = free_bytes // 1024**2
            if free_mib < target_mib:
                raise RuntimeError(
                    f"GPU{physical_gpu} has {free_mib} MiB free at allocator claim; "
                    f"{ATTEMPT_ID} requires {target_mib} MiB"
                )
            holders.append(
                torch.empty(target_mib * 1024**2, dtype=torch.uint8, device=visible_gpu)
            )
    for visible_gpu in range(len(physical_gpus)):
        torch.cuda.synchronize(visible_gpu)
    holders.clear()
    reservation = {
        str(visible_gpu): {
            "physical_gpu": physical_gpu,
            "target_mib": targets[physical_gpu],
            "reserved_mib": torch.cuda.memory_reserved(visible_gpu) / 1024**2,
        }
        for visible_gpu, physical_gpu in enumerate(physical_gpus)
    }
    for row in reservation.values():
        if float(row["reserved_mib"]) + 1 < int(row["target_mib"]):
            raise RuntimeError(
                f"GPU{row['physical_gpu']} allocator retained only "
                f"{row['reserved_mib']} MiB of the {row['target_mib']} MiB claim"
            )
    return reservation


def heartbeat_loop(
    stop: threading.Event,
    path: Path,
    interval: int,
    extra: Callable[[], dict[str, Any]],
) -> None:
    while not stop.wait(interval):
        update_json_status(path, **extra())


def run_diagnostic(
    root: Path,
    fold: str,
    physical_gpus: list[int],
    *,
    max_users: int | None,
    experiment_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    config, checkpoint_auth, authorization = load_contracts()
    if physical_gpus != authorization["runtime"]["beauty_physical_gpus"]:
        raise RuntimeError(f"Beauty GPU pair differs from the authorized {ATTEMPT_ID} pair")
    scope = authorization["correction_scope"]
    release_cuda_cache_per_user = bool(scope.get("release_cuda_cache_per_user"))
    retain_allocator_cache = bool(scope.get("retain_allocator_cache_between_users"))
    base.OUTPUT = root
    target = unit_dir(root, "Beauty", fold)
    if target.exists():
        raise FileExistsError(f"{ATTEMPT_ID} output exists; automatic retry forbidden: {target}")
    target.mkdir(parents=True)
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1_beauty_recovery_unit_status.v1",
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "recovery_of": RECOVERY_OF,
            "domain": "Beauty",
            "fold": fold,
            "unit": unit_key("Beauty", fold),
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
            "release_cuda_cache_per_user": release_cuda_cache_per_user,
            "retain_allocator_cache_between_users": retain_allocator_cache,
            "max_users": max_users,
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
            raise RuntimeError(f"{ATTEMPT_ID} requires exactly two visible GPUs")
        for visible_gpu in range(2):
            torch.cuda.reset_peak_memory_stats(visible_gpu)
        allocator_reservation = prime_allocator_reservation(physical_gpus, authorization)
        base.update_unit_status(
            "Beauty",
            fold,
            allocator_reservation=allocator_reservation,
            retain_allocator_cache_between_users=retain_allocator_cache,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config["backbone"]["snapshot"], local_files_only=True
        )
        parent, args, item_head, item_to_id, frequencies, sequences, provenance = (
            checkpoint_recovery.load_frozen_models(
                config, checkpoint_auth, "Beauty", fold, device
            )
        )
        decoder_device_map = base.enable_two_gpu_decoder_parallel(parent)
        args.tokenizer = tokenizer
        base.update_unit_status(
            "Beauty",
            fold,
            execution_state="RUNNING_BOUNDED_GENERATION",
            phase="beam50_beam200_diagnostic",
            parent_retraining=False,
            item_head_retraining=False,
        )
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
            max_users=max_users,
            generation_use_cache=True,
            cross_attention_cache=True,
            release_cuda_cache_per_user=release_cuda_cache_per_user,
        )
        peak_by_visible_gpu = {
            str(index): {
                "allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
                "reserved_mib": torch.cuda.max_memory_reserved(index) / 1024**2,
                "physical_gpu": physical_gpus[index],
            }
            for index in range(2)
        }
        summary = {
            **diagnostic,
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "recovery_of": RECOVERY_OF,
            "status": "COMPLETED",
            "parent_training": provenance["parent"],
            "item_head_training": provenance["item_head"],
            "physical_gpus": physical_gpus,
            "decoder_device_map": decoder_device_map,
            "allocator_reservation": allocator_reservation,
            "release_cuda_cache_per_user": release_cuda_cache_per_user,
            "retain_allocator_cache_between_users": retain_allocator_cache,
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
            "Beauty",
            fold,
            execution_state="COMPLETED",
            phase="complete",
            process_alive=False,
            summary_path=str((target / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(target / "summary.json"),
            elapsed_seconds=time.time() - started,
        )
        return summary
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


def launch_smoke() -> int:
    _, checkpoint_auth, authorization = load_contracts()
    if SMOKE.exists() or SMOKE_STATUS.exists():
        raise FileExistsError(f"{ATTEMPT_ID} Beauty memory smoke already exists; automatic retry forbidden")
    runtime = authorization["runtime"]
    physical_gpus = runtime["beauty_physical_gpus"]
    required = {int(key): int(value) for key, value in runtime["smoke_min_free_mib_by_gpu"].items()}
    snapshot = validate_pair_free(physical_gpus, required)
    SMOKE.mkdir(parents=True)
    manifest = {
        "schema_version": "phase18.s18_1_beauty_memory_smoke_manifest.v1",
        "experiment_id": f"{EXPERIMENT_ID}_memory_smoke",
        "attempt_id": f"{ATTEMPT_ID}-smoke",
        "created_at": base.utc_now(),
        "physical_gpus": physical_gpus,
        "max_users": runtime["smoke_users"],
        "gpu_snapshot": snapshot,
        "source_manifest": source_manifest(checkpoint_auth, authorization),
        "scientific_result_eligible": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(SMOKE / "run_manifest.json", manifest)
    update_json_status(
        SMOKE_STATUS,
        reset=True,
        schema_version="phase18.s18_1_beauty_memory_smoke_status.v1",
        experiment_id=manifest["experiment_id"],
        attempt_id=manifest["attempt_id"],
        execution_state="STARTING",
        status="RUNNING",
        process_alive=True,
        workload_pid=0,
        physical_gpus=physical_gpus,
        max_users=runtime["smoke_users"],
        tmux_session=runtime["smoke_tmux_session"],
        scientific_result_eligible=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
        started_at=base.utc_now(),
    )
    command = [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={','.join(map(str, physical_gpus))}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={ROOT}",
        str(PYTHON),
        str(ENTRY_PATH),
        "smoke-worker",
    ]
    try:
        launch_background_tmux(
            experiment_id=manifest["experiment_id"],
            argv=command,
            cwd=ROOT,
            tmux_session=runtime["smoke_tmux_session"],
            startup_log_path=SMOKE / "run.log",
        )
    except Exception as error:
        update_json_status(
            SMOKE_STATUS,
            execution_state="FAILED_NO_RETRY",
            status="FAILED",
            process_alive=False,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    deadline = time.time() + 30
    while time.time() < deadline:
        status = load_json(SMOKE_STATUS)
        if status.get("workload_pid", 0) > 0:
            print(json.dumps({"status": "STARTED", "status_path": str(SMOKE_STATUS.relative_to(ROOT))}))
            return 0
        time.sleep(1)
    raise RuntimeError(f"{ATTEMPT_ID} Beauty memory smoke failed startup handshake")


def smoke_worker() -> int:
    config, checkpoint_auth, authorization = load_contracts()
    runtime = authorization["runtime"]
    manifest = load_json(SMOKE / "run_manifest.json")
    if manifest["source_manifest"] != source_manifest(checkpoint_auth, authorization):
        raise RuntimeError(f"{ATTEMPT_ID} smoke source manifest changed after launch")
    update_json_status(
        SMOKE_STATUS,
        execution_state="RUNNING_BOUNDED_GENERATION",
        status="RUNNING",
        process_alive=True,
        workload_pid=os.getpid(),
    )
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(
            stop,
            SMOKE_STATUS,
            runtime["heartbeat_seconds"],
            lambda: {
                "execution_state": "RUNNING_BOUNDED_GENERATION",
                "status": "RUNNING",
                "process_alive": True,
                "workload_pid": os.getpid(),
                "gpu_snapshot": safe_gpu_snapshot(),
            },
        ),
        daemon=True,
    )
    heartbeat.start()
    started = time.time()
    try:
        summary = run_diagnostic(
            SMOKE,
            "I0",
            runtime["beauty_physical_gpus"],
            max_users=runtime["smoke_users"],
            experiment_id=f"{EXPERIMENT_ID}_memory_smoke",
            attempt_id=f"{ATTEMPT_ID}-smoke",
        )
        parity = compare_first_beauty_user(unit_dir(SMOKE, "Beauty", "I0"), authorization)
        payload = {
            "schema_version": "phase18.s18_1_beauty_memory_smoke.v1",
            "status": "PASSED",
            "attempt_id": f"{ATTEMPT_ID}-smoke",
            "physical_gpus": runtime["beauty_physical_gpus"],
            "max_users": runtime["smoke_users"],
            "generation_use_cache": True,
            "cross_attention_cache": True,
            "decoder_model_parallel": True,
            "release_cuda_cache_per_user": summary["release_cuda_cache_per_user"],
            "retain_allocator_cache_between_users": summary[
                "retain_allocator_cache_between_users"
            ],
            "allocator_reservation": summary["allocator_reservation"],
            "peak_allocated_mib": summary["peak_allocated_mib"],
            "peak_reserved_mib": summary["peak_reserved_mib"],
            "peak_by_visible_gpu": summary["peak_by_visible_gpu"],
            "wall_time_seconds": time.time() - started,
            "parity": parity,
            "source_manifest": source_manifest(checkpoint_auth, authorization),
            "scientific_parameters_changed": False,
            "scientific_result_eligible": False,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
        }
        base.atomic_json(SMOKE / "summary.json", payload)
        update_json_status(
            SMOKE_STATUS,
            execution_state="COMPLETED",
            status="PASSED",
            process_alive=False,
            workload_pid=0,
            summary_path=str((SMOKE / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(SMOKE / "summary.json"),
            peak_by_visible_gpu=summary["peak_by_visible_gpu"],
            parity=parity,
            wall_time_seconds=time.time() - started,
        )
        print(json.dumps(payload, default=base.json_default))
        return 0
    except Exception as error:
        update_json_status(
            SMOKE_STATUS,
            execution_state="FAILED_NO_RETRY",
            status="FAILED",
            process_alive=False,
            workload_pid=0,
            error_type=type(error).__name__,
            error=str(error),
            wall_time_seconds=time.time() - started,
        )
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=2)


def carry_forward(
    label: str,
    source_path: Path,
    source_attempt: str,
) -> None:
    domain, fold = label.split(":", 1)
    target = unit_dir(OUTPUT, domain, fold)
    if target.exists():
        raise FileExistsError(f"{ATTEMPT_ID} carry target exists; retry forbidden: {target}")
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
            "schema_version": "phase18.s18_1_beauty_recovery_unit_status.v1",
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


def formal_required_free(
    smoke: dict[str, Any], authorization: dict[str, Any]
) -> dict[int, int]:
    physical_gpus = authorization["runtime"]["beauty_physical_gpus"]
    observed = {
        int(row["physical_gpu"]): int(math.ceil(float(row["reserved_mib"])))
        for row in smoke["peak_by_visible_gpu"].values()
    }
    if sorted(observed) != sorted(physical_gpus):
        raise RuntimeError(f"Beauty smoke GPU mapping differs from {ATTEMPT_ID} authorization")
    buffer_mib = int(authorization["runtime"]["formal_memory_buffer_mib"])
    return {gpu: observed[gpu] + buffer_mib for gpu in physical_gpus}


def launch() -> int:
    _, checkpoint_auth, authorization = load_contracts()
    if not (SMOKE / "summary.json").is_file():
        raise RuntimeError(f"{ATTEMPT_ID} formal launch requires the completed Beauty memory smoke")
    smoke = load_json(SMOKE / "summary.json")
    parity = smoke.get("parity", {})
    if (
        smoke.get("status") != "PASSED"
        or smoke.get("max_users") != authorization["runtime"]["smoke_users"]
        or not parity.get("candidate_order_exact")
        or not parity.get("diagnostic_record_exact")
        or any(float(value) > 1e-6 for value in parity.get("score_max_abs_deltas", {}).values())
        or smoke.get("source_manifest") != source_manifest(checkpoint_auth, authorization)
    ):
        raise RuntimeError(f"{ATTEMPT_ID} Beauty memory smoke is not exact and current")
    if OUTPUT.exists() or STATUS.exists():
        raise FileExistsError(f"{ATTEMPT_ID} formal output already exists; automatic retry forbidden")
    required = formal_required_free(smoke, authorization)
    admission_history = stable_pair_admission(authorization, required)
    OUTPUT.mkdir(parents=True)
    manifest = {
        "schema_version": "phase18.s18_1_beauty_recovery_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "recovery_of": RECOVERY_OF,
        "created_at": base.utc_now(),
        "physical_gpus": authorization["runtime"]["beauty_physical_gpus"],
        "units_to_execute": authorization["runtime"]["beauty_units"],
        "carried_forward_units": ["Toys:I0", "Toys:I-1"],
        "required_free_mib_by_gpu": required,
        "admission_history": admission_history,
        "memory_smoke_path": str((SMOKE / "summary.json").relative_to(ROOT)),
        "memory_smoke_sha256": sha256(SMOKE / "summary.json"),
        "source_manifest": source_manifest(checkpoint_auth, authorization),
        "checkpoint_only": True,
        "parent_retraining": False,
        "item_head_retraining": False,
        "scientific_parameters_changed": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(OUTPUT / "run_manifest.json", manifest)
    if not CANONICAL_STATUS.is_file():
        raise FileNotFoundError("canonical S18-1 status is missing")
    if STATUS_ARCHIVE.exists():
        raise FileExistsError(f"canonical status archive already exists: {STATUS_ARCHIVE}")
    STATUS_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(STATUS_ARCHIVE, load_json(CANONICAL_STATUS))
    update_status(
        reset=True,
        schema_version="phase18.status.v1",
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of=RECOVERY_OF,
        step_id="S18-1-INFRASTRUCTURE-CORRECTION",
        stage="background_starting",
        execution_state="STARTING_CHECKPOINT_ONLY_RECOVERY",
        scientific_state="RUNNING",
        status_code=status_code("STARTING"),
        process_alive=True,
        workload_pid=0,
        tmux_session=authorization["runtime"]["formal_tmux_session"],
        physical_gpus=authorization["runtime"]["beauty_physical_gpus"],
        units_to_execute=authorization["runtime"]["beauty_units"],
        carried_forward_units=["Toys:I0", "Toys:I-1"],
        required_free_mib_by_gpu=required,
        progress={"current": 2, "total": 4, "unit": "domain_fold_diagnostic"},
        run_manifest_path=str((OUTPUT / "run_manifest.json").relative_to(ROOT)),
        run_manifest_sha256=sha256(OUTPUT / "run_manifest.json"),
        canonical_status_archive=str(STATUS_ARCHIVE.relative_to(ROOT)),
        checkpoint_only=True,
        resource_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        scientific_parameters_changed=False,
        result_selection_eligible=False,
        automatic_retry=False,
        automatic_s18_2=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
        started_at=base.utc_now(),
        next_action="Observe this canonical status; do not launch another S18-1 attempt or S18-2.",
    )
    base.append_jsonl(
        LEDGER,
        {
            "event": "authorized_beauty_checkpoint_recovery_started",
            "at": base.utc_now(),
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "physical_gpus": authorization["runtime"]["beauty_physical_gpus"],
            "run_manifest_sha256": sha256(OUTPUT / "run_manifest.json"),
            "automatic_retry": False,
            "automatic_s18_2": False,
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
        str(ENTRY_PATH),
        "master",
    ]
    try:
        launch_background_tmux(
            experiment_id=EXPERIMENT_ID,
            argv=command,
            cwd=ROOT,
            tmux_session=authorization["runtime"]["formal_tmux_session"],
            startup_log_path=OUTPUT / "master.log",
        )
    except Exception as error:
        update_status(
            stage="terminal_failure",
            execution_state="FAILED_NO_RETRY",
            scientific_state="FAILED",
            status_code=status_code("TMUX_START_FAILED_NO_RETRY"),
            process_alive=False,
            workload_pid=0,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    deadline = time.time() + 30
    while time.time() < deadline:
        status = load_json(STATUS)
        if status.get("workload_pid", 0) > 0:
            print(
                json.dumps(
                    {
                        "status": "STARTED",
                        "status_path": str(CANONICAL_STATUS.relative_to(ROOT)),
                        "attempt_status_path": str(STATUS.relative_to(ROOT)),
                        "physical_gpus": authorization["runtime"]["beauty_physical_gpus"],
                    }
                )
            )
            return 0
        time.sleep(1)
    raise RuntimeError(f"{ATTEMPT_ID} formal recovery failed startup handshake")


def execute_unit(
    label: str, authorization: dict[str, Any]
) -> tuple[int, int, Path]:
    _, fold = label.split(":", 1)
    physical_gpus = authorization["runtime"]["beauty_physical_gpus"]
    required = formal_required_free(load_json(SMOKE / "summary.json"), authorization)
    unit_admission = validate_pair_free(physical_gpus, required)
    update_status(
        stage="unit_resource_admission",
        current_unit=label,
        unit_required_free_mib_by_gpu=required,
        unit_admission_snapshot=unit_admission,
    )
    log = OUTPUT / "units" / f"beauty_{fold.lower().replace('-', 'm')}.launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, physical_gpus)),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(ROOT),
    )
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(PYTHON), str(ENTRY_PATH), "unit", "--fold", fold],
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
            unit_status_path = unit_dir(OUTPUT, "Beauty", fold) / "status.json"
            update_status(
                stage="beauty_checkpoint_recovery",
                execution_state="RUNNING_BOUNDED_GENERATION",
                scientific_state="RUNNING",
                status_code=status_code("RUNNING"),
                process_alive=True,
                workload_pid=os.getpid(),
                current_unit=label,
                current_unit_pid=process.pid,
                current_unit_status=(
                    load_json(unit_status_path) if unit_status_path.is_file() else None
                ),
                gpu_snapshot=safe_gpu_snapshot(),
            )
            time.sleep(authorization["runtime"]["heartbeat_seconds"])
        return int(process.returncode), process.pid, log


def fail_unit(label: str, pid: int, log: Path, return_code: int) -> int:
    base.append_jsonl(
        LEDGER,
        {
            "event": "beauty_checkpoint_recovery_unit_failed_no_retry",
            "at": base.utc_now(),
            "attempt_id": ATTEMPT_ID,
            "unit": label,
            "return_code": return_code,
        },
    )
    update_status(
        stage="terminal_failure",
        execution_state="FAILED_NO_RETRY",
        scientific_state="FAILED",
        status_code=status_code("BEAUTY_FAILURE_NO_RETRY"),
        process_alive=False,
        workload_pid=0,
        failed_unit=label,
        failed_unit_pid=pid,
        failed_unit_log=str(log.relative_to(ROOT)),
        result_selection_eligible=False,
        next_action="Inspect the named failure; do not retry or start S18-2 without researcher direction.",
    )
    return 1


def aggregate(config: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    base.OUTPUT = OUTPUT
    summary = base.aggregate_results(config)
    scope = authorization["correction_scope"]
    summary.update(
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of=(
            "run-0002 Toys:I0 + run-0003 Toys:I-1 + "
            f"{ATTEMPT_ID} Beauty:I0 and Beauty:I-1"
        ),
        beauty_recovery_authorization_sha256=sha256(AUTH_PATH),
        checkpoint_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        scientific_parameters_changed=False,
        resource_adaptation={
            "physical_gpus": authorization["runtime"]["beauty_physical_gpus"],
            "decoder_model_parallel": True,
            "generation_use_cache": True,
            "cross_attention_cache": True,
            "release_cuda_cache_per_user": bool(
                scope.get("release_cuda_cache_per_user")
            ),
            "retain_allocator_cache_between_users": bool(
                scope.get("retain_allocator_cache_between_users")
            ),
            "allocator_reservation_mib_by_gpu": authorization["runtime"].get(
                "allocator_reservation_mib_by_gpu", {}
            ),
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
            "reason": "named Beauty-only checkpoint recovery after shared-GPU OOM",
            "created_at": base.utc_now(),
        },
    )
    base.REPORT = REPORT
    base.write_report(summary)
    report = REPORT.read_text(encoding="utf-8")
    report += (
        "\n## Infrastructure Recovery\n\n"
        "- Toys:I0 was carried from run-0002 and Toys:I-1 from run-0003.\n"
        "- Beauty:I0 and Beauty:I-1 were regenerated from the frozen run-0001 epoch-10 checkpoints.\n"
        "- No parent or item-head retraining occurred; cohort, beam widths, scores, and Gates were unchanged.\n"
        "- D1, D2, official test, and Sports remained unread.\n"
        "- S18-2 was not started automatically.\n"
    )
    base.atomic_text(REPORT, report)
    return summary


def master() -> int:
    config, checkpoint_auth, authorization = load_contracts()
    manifest = load_json(OUTPUT / "run_manifest.json")
    if manifest["source_manifest"] != source_manifest(checkpoint_auth, authorization):
        raise RuntimeError(f"{ATTEMPT_ID} source manifest changed after launch")
    frozen = authorization["frozen_inputs"]
    update_status(
        stage="carrying_completed_toys",
        execution_state="RUNNING_CHECKPOINT_ONLY_RECOVERY",
        scientific_state="RUNNING",
        status_code=status_code("RUNNING"),
        process_alive=True,
        workload_pid=os.getpid(),
    )
    try:
        carry_forward(
            "Toys:I0",
            validate_completed_unit(
                TOYS_I0,
                "Toys",
                "I0",
                frozen["toys_i0_summary"]["sha256"],
            ),
            "s18_s1_actionability_recovery/run-0002",
        )
        carry_forward(
            "Toys:I-1",
            validate_completed_unit(
                TOYS_IM1,
                "Toys",
                "I-1",
                frozen["toys_im1_summary"]["sha256"],
            ),
            "s18_s1_actionability_resource_recovery/run-0003",
        )
        completed = 2
        update_status(progress={"current": completed, "total": 4, "unit": "domain_fold_diagnostic"})
        for label in authorization["runtime"]["beauty_units"]:
            return_code, pid, log = execute_unit(label, authorization)
            if return_code != 0:
                return fail_unit(label, pid, log, return_code)
            completed += 1
            update_status(
                progress={"current": completed, "total": 4, "unit": "domain_fold_diagnostic"}
            )
        summary = aggregate(config, authorization)
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_beauty_checkpoint_recovery_completed",
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
            canonical_manifest_path=str(CANONICAL_MANIFEST.relative_to(ROOT)),
            report_path=str(REPORT.relative_to(ROOT)),
            result_selection_eligible=True,
            automatic_s18_2=False,
            next_action="Review the S18-1 Gate; S18-2 remains unstarted and requires separate direction.",
        )
        return 0
    except Exception as error:
        base.append_jsonl(
            LEDGER,
            {
                "event": "authorized_beauty_checkpoint_recovery_failed_no_retry",
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
            status_code=status_code("FAILURE_NO_RETRY"),
            process_alive=False,
            workload_pid=0,
            error_type=type(error).__name__,
            error=str(error),
            result_selection_eligible=False,
            next_action="Inspect the named failure; do not retry or start S18-2 without researcher direction.",
        )
        raise


def run_unit(fold: str) -> int:
    _, _, authorization = load_contracts()
    run_diagnostic(
        OUTPUT,
        fold,
        authorization["runtime"]["beauty_physical_gpus"],
        max_users=None,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("verify", "launch-smoke", "smoke-worker", "launch", "master", "unit")
    )
    parser.add_argument("--fold", choices=("I-1", "I0"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify":
        _, checkpoint_auth, authorization = load_contracts()
        print(
            json.dumps(
                {
                    "status": "VERIFIED",
                    "attempt_id": ATTEMPT_ID,
                    "physical_gpus": authorization["runtime"]["beauty_physical_gpus"],
                    "source_manifest": source_manifest(checkpoint_auth, authorization),
                }
            )
        )
        return 0
    if args.action == "launch-smoke":
        return launch_smoke()
    if args.action == "smoke-worker":
        return smoke_worker()
    if args.action == "launch":
        return launch()
    if args.action == "master":
        return master()
    if args.action == "unit":
        if args.fold is None:
            raise ValueError("unit requires --fold")
        return run_unit(args.fold)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
