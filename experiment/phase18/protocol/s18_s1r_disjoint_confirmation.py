#!/usr/bin/env python3
"""Checkpoint-only S18-1R-B confirmation on a frozen disjoint cohort."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from experiment.phase17.core.run_manager import launch_background_tmux
from experiment.phase18.core.contracts import ROOT, load_json, normalize_repo_relative, sha256
from experiment.phase18.core.s1_contracts import capped_recall_metrics, evaluate_capacity_gate
from experiment.phase18.protocol import s18_s1_recovery as checkpoint_recovery
from experiment.phase18.protocol import s18_s1_runtime as base


PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
ENTRY = Path(__file__).resolve()
AUTH = ROOT / "experiment/phase18/config/s18_s1r_disjoint_resource_authorization.json"
PREFLIGHT = ROOT / "artifacts/phase18/s1r_disjoint_confirmation/preflight"
OUTPUT = ROOT / "artifacts/phase18/s1r_disjoint_confirmation/run-0001"
SMOKE = ROOT / "artifacts/phase18/s1r_disjoint_confirmation/smoke-run-0001"
STATUS = ROOT / "artifacts/phase18/status/s18_s1r_disjoint_confirmation.status.json"
SMOKE_STATUS = ROOT / "artifacts/phase18/status/s18_s1r_disjoint_smoke.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1R.attempts.jsonl"
REPORT = ROOT / "report/第十八阶段/Stage18_S18-1R_未见Cohort确认报告.md"


def verified_path(record: dict[str, str]) -> Path:
    path = ROOT / normalize_repo_relative(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise RuntimeError(f"frozen input mismatch: {path}")
    return path


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTH)
    if (
        authorization.get("experiment_id") != "s18_s1r_disjoint_confirmation"
        or authorization.get("attempt_id") != "run-0001"
    ):
        raise RuntimeError("S18-1R-B resource authorization identity mismatch")
    paths = {name: verified_path(record) for name, record in authorization["frozen_inputs"].items()}
    confirmation = load_json(paths["confirmation_config"])
    preflight = load_json(paths["confirmation_preflight"])
    original = load_json(paths["s18_s1_config"])
    checkpoint_auth = load_json(paths["checkpoint_authorization"])
    retrospective = load_json(paths["retrospective_audit"])
    scope = authorization["scope"]
    required_true = (
        "checkpoint_only_diagnostic",
        "disjoint_confirmation",
        "two_gpu_decoder_parallel",
        "generation_use_cache",
        "cross_attention_cache",
        "retain_allocator_cache_between_users",
    )
    required_false = (
        "parent_retraining",
        "item_head_retraining",
        "treatment_training",
        "cohort_changes",
        "beam_changes",
        "score_changes",
        "protected_data_access",
        "automatic_retry",
        "automatic_s18_2",
    )
    if any(not scope.get(name) for name in required_true) or any(
        scope.get(name) for name in required_false
    ):
        raise RuntimeError("S18-1R-B resource scope is not fail-closed")
    if retrospective.get("decision") != "S18_1R_RETROSPECTIVE_CAPACITY_PASS_REQUIRES_DISJOINT_CONFIRMATION":
        raise RuntimeError("retrospective capacity audit prerequisite changed")
    if (
        preflight.get("checks_passed") != preflight.get("checks_total")
        or preflight.get("scientific_attempt_started")
        or any(preflight["protected_data"].values())
    ):
        raise RuntimeError("S18-1R-B preflight is not clean and complete")
    if confirmation["checkpoints"] != checkpoint_auth["checkpoints"]:
        raise RuntimeError("checkpoint provenance changed")
    for domain in ("Toys", "Beauty"):
        cohort_path = PREFLIGHT / f"cohort_{domain}.txt"
        expected = confirmation["cohort"]["domains"][domain]["confirmation_sha256"]
        if not cohort_path.is_file() or sha256(cohort_path) != expected:
            raise RuntimeError(f"{domain}: disjoint cohort mismatch")
        if len(cohort_path.read_text(encoding="utf-8").splitlines()) != 1024:
            raise RuntimeError(f"{domain}: disjoint cohort size mismatch")
    runtime = authorization["runtime"]
    if len(runtime["physical_gpus"]) != 2 or len(set(runtime["physical_gpus"])) != 2:
        raise RuntimeError("S18-1R-B requires exactly two physical GPUs")
    expected_units = ["Toys:I-1", "Toys:I0", "Beauty:I-1", "Beauty:I0"]
    if runtime["serial_units"] != expected_units:
        raise RuntimeError("S18-1R-B must execute all four units in the frozen order")
    return original, confirmation, checkpoint_auth, authorization


def source_manifest() -> dict[str, str]:
    _, confirmation, checkpoint_auth, authorization = load_contracts()
    paths = [
        AUTH,
        ENTRY,
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
    for domain in ("Toys", "Beauty"):
        path = PREFLIGHT / f"cohort_{domain}.txt"
        records[str(path.relative_to(ROOT))] = sha256(path)
    for pair in checkpoint_auth["checkpoints"].values():
        for record in pair.values():
            records[record["path"]] = record["sha256"]
    for domain in confirmation["cohort"]["domains"].values():
        records[domain["shadow_path"]] = domain["shadow_sha256"]
    return records


def update_status(path: Path, *, reset: bool = False, **fields: Any) -> None:
    current = {} if reset or not path.is_file() else load_json(path)
    current.update(fields)
    current["updated_at"] = base.utc_now()
    current["heartbeat_at"] = base.utc_now()
    base.atomic_json(path, current)


def stable_admission(authorization: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = authorization["runtime"]
    required = {int(key): int(value) for key, value in runtime["minimum_free_mib_by_gpu"].items()}
    history = []
    for ordinal in range(runtime["stable_snapshots_required"]):
        snapshot = base.gpu_snapshot()
        by_gpu = {row["index"]: row for row in snapshot}
        for gpu in runtime["physical_gpus"]:
            free = by_gpu.get(gpu, {}).get("free_mib")
            if free is None or free < required[gpu]:
                raise RuntimeError(f"GPU{gpu} has {free} MiB free; requires {required[gpu]} MiB")
        history.append({"at": base.utc_now(), "gpus": snapshot})
        if ordinal + 1 < runtime["stable_snapshots_required"]:
            time.sleep(runtime["snapshot_interval_seconds"])
    return history


def reserve_allocator(physical_gpus: list[int], authorization: dict[str, Any]) -> dict[str, Any]:
    targets = {
        int(key): int(value)
        for key, value in authorization["runtime"]["allocator_reservation_mib_by_gpu"].items()
    }
    holders = []
    for visible, physical in enumerate(physical_gpus):
        with torch.cuda.device(visible):
            free_mib = torch.cuda.mem_get_info(visible)[0] // 1024**2
            if free_mib < targets[physical]:
                raise RuntimeError(
                    f"GPU{physical} has {free_mib} MiB at allocator claim; requires {targets[physical]} MiB"
                )
            holders.append(
                torch.empty(targets[physical] * 1024**2, dtype=torch.uint8, device=visible)
            )
    for visible in range(len(physical_gpus)):
        torch.cuda.synchronize(visible)
    holders.clear()
    result = {
        str(visible): {
            "physical_gpu": physical,
            "target_mib": targets[physical],
            "reserved_mib": torch.cuda.memory_reserved(visible) / 1024**2,
        }
        for visible, physical in enumerate(physical_gpus)
    }
    if any(float(row["reserved_mib"]) + 1 < int(row["target_mib"]) for row in result.values()):
        raise RuntimeError("allocator reservation was not retained")
    return result


def unit_dir(root: Path, domain: str, fold: str) -> Path:
    return root / "units" / base.unit_key(domain, fold)


def run_unit(root: Path, domain: str, fold: str, max_users: int | None) -> int:
    config, _, checkpoint_auth, authorization = load_contracts()
    physical_gpus = authorization["runtime"]["physical_gpus"]
    if torch.cuda.device_count() != 2:
        raise RuntimeError("unit requires exactly two visible CUDA devices")
    base.OUTPUT = root
    target = unit_dir(root, domain, fold)
    if target.exists():
        raise FileExistsError(f"unit output exists; automatic retry forbidden: {target}")
    target.mkdir(parents=True)
    started = time.time()
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1r_disjoint_unit_status.v1",
            "experiment_id": "s18_s1r_disjoint_confirmation",
            "attempt_id": "run-0001",
            "domain": domain,
            "fold": fold,
            "execution_state": "STARTING_CHECKPOINT_ONLY_DIAGNOSTIC",
            "physical_gpus": physical_gpus,
            "pid": os.getpid(),
            "process_alive": True,
            "max_users": max_users,
            "parent_retraining": False,
            "item_head_retraining": False,
            "treatment_training": False,
            "started_at": base.utc_now(),
            "heartbeat_at": base.utc_now(),
        },
    )
    try:
        base.set_seed(config["seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        for visible in range(2):
            torch.cuda.reset_peak_memory_stats(visible)
        reservation = reserve_allocator(physical_gpus, authorization)
        tokenizer = AutoTokenizer.from_pretrained(
            config["backbone"]["snapshot"], local_files_only=True
        )
        parent, args, item_head, item_to_id, frequencies, sequences, provenance = (
            checkpoint_recovery.load_frozen_models(
                config, checkpoint_auth, domain, fold, device
            )
        )
        decoder_map = base.enable_two_gpu_decoder_parallel(parent)
        args.tokenizer = tokenizer
        base.update_unit_status(
            domain,
            fold,
            execution_state="RUNNING_BOUNDED_GENERATION",
            phase="beam50_beam200_disjoint_diagnostic",
            allocator_reservation=reservation,
            decoder_device_map=decoder_map,
        )
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
            max_users=max_users,
            generation_use_cache=True,
            cross_attention_cache=True,
            release_cuda_cache_per_user=False,
            cohort_path=PREFLIGHT / f"cohort_{domain}.txt",
        )
        peak = {
            str(visible): {
                "physical_gpu": physical,
                "allocated_mib": torch.cuda.max_memory_allocated(visible) / 1024**2,
                "reserved_mib": torch.cuda.max_memory_reserved(visible) / 1024**2,
            }
            for visible, physical in enumerate(physical_gpus)
        }
        summary = {
            **diagnostic,
            "experiment_id": "s18_s1r_disjoint_confirmation",
            "attempt_id": "run-0001",
            "status": "COMPLETED",
            "cohort_path": str((PREFLIGHT / f"cohort_{domain}.txt").relative_to(ROOT)),
            "cohort_sha256": sha256(PREFLIGHT / f"cohort_{domain}.txt"),
            "parent_training": provenance["parent"],
            "item_head_training": provenance["item_head"],
            "physical_gpus": physical_gpus,
            "decoder_device_map": decoder_map,
            "allocator_reservation": reservation,
            "peak_by_visible_gpu": peak,
            "wall_time_total_seconds": time.time() - started,
            "checkpoint_only": True,
            "parent_retraining": False,
            "item_head_retraining": False,
            "treatment_training": False,
            "i1_i2_read": False,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
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


def spawn_unit(root: Path, label: str, max_users: int | None) -> tuple[subprocess.Popen, Path]:
    domain, fold = label.split(":", 1)
    _, _, _, authorization = load_contracts()
    physical_gpus = authorization["runtime"]["physical_gpus"]
    log = root / "units" / f"{base.unit_key(domain, fold)}.launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(ENTRY),
        "unit",
        "--root",
        str(root),
        "--domain",
        domain,
        "--fold",
        fold,
    ]
    if max_users is not None:
        command.extend(["--max-users", str(max_users)])
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, physical_gpus)),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(ROOT),
    )
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process, log


def wait_unit(
    process: subprocess.Popen,
    log: Path,
    root: Path,
    label: str,
    status_path: Path,
    completed: int,
    total: int,
) -> int:
    _, _, _, authorization = load_contracts()
    timeout = authorization["runtime"]["unit_hard_timeout_seconds"]
    started = time.time()
    domain, fold = label.split(":", 1)
    while process.poll() is None:
        if time.time() - started > timeout:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124
        unit_status = unit_dir(root, domain, fold) / "status.json"
        update_status(
            status_path,
            execution_state="RUNNING_BOUNDED_GENERATION",
            scientific_state="RUNNING" if root == OUTPUT else "SMOKE_RUNNING",
            process_alive=True,
            workload_pid=os.getpid(),
            current_unit=label,
            current_unit_pid=process.pid,
            current_unit_status=load_json(unit_status) if unit_status.is_file() else None,
            current_unit_log=str(log.relative_to(ROOT)),
            progress={"current": completed, "total": total, "unit": "domain_fold_diagnostic"},
        )
        time.sleep(authorization["runtime"]["heartbeat_seconds"])
    return int(process.returncode)


def terminate_failure(status_path: Path, label: str, code: int, log: Path) -> int:
    update_status(
        status_path,
        execution_state="FAILED_NO_RETRY",
        scientific_state="FAILED",
        status_code="S18_1R_DISJOINT_UNIT_FAILED_NO_RETRY",
        process_alive=False,
        workload_pid=0,
        failed_unit=label,
        return_code=code,
        failed_unit_log=str(log.relative_to(ROOT)),
        result_selection_eligible=False,
        next_action="Inspect the named failure; do not retry or start S18-2 automatically.",
    )
    base.append_jsonl(
        LEDGER,
        {
            "at": base.utc_now(),
            "attempt_id": "run-0001",
            "event": "disjoint_confirmation_unit_failed_no_retry",
            "unit": label,
            "return_code": code,
        },
    )
    return 1


def load_capacity_events(path: Path, k: int) -> list[dict[str, int]]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            actual = set(row["actual_pruner_items"])
            selected = set(row["hard_negative_items"])
            intersection = len(actual & selected)
            if len(selected) > k:
                raise RuntimeError(f"more than K negatives at {path}:{line_number}")
            if (
                len(actual) != int(row["hard_negative_actual_denominator"])
                or intersection != int(row["hard_negative_intersection"])
            ):
                raise RuntimeError(f"capacity count mismatch at {path}:{line_number}")
            if actual:
                events.append(
                    {
                        "intersection": intersection,
                        "actual": len(actual),
                        "depth": int(row["first_drop_depth"]),
                    }
                )
    return events


def capacity_summary(events: list[dict[str, int]], k: int) -> dict[str, Any]:
    metrics = capped_recall_metrics(
        ((row["intersection"], row["actual"]) for row in events), k
    )
    by_depth: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in events:
        by_depth[row["depth"]].append((row["intersection"], row["actual"]))
    metrics["by_first_drop_depth"] = {
        str(depth): capped_recall_metrics(rows, k)
        for depth, rows in sorted(by_depth.items())
    }
    return metrics


def aggregate() -> dict[str, Any]:
    original, confirmation, _, _ = load_contracts()
    units = {
        f"{domain}:{fold}": load_json(unit_dir(OUTPUT, domain, fold) / "summary.json")
        for domain in ("Toys", "Beauty")
        for fold in ("I-1", "I0")
    }
    domains = {}
    k = int(confirmation["gate"]["hard_negative_k"])
    for domain in ("Toys", "Beauty"):
        rows = [units[f"{domain}:{fold}"] for fold in ("I-1", "I0")]
        users = sum(int(row["users"]) for row in rows)
        first_drop = sum(int(row["first_drop_events"]) for row in rows)
        nonempty = sum(int(row["nonempty_actual_pruner_events"]) for row in rows)
        events_by_fold = {
            fold: load_capacity_events(
                unit_dir(OUTPUT, domain, fold) / "per_user_diagnostics.jsonl", k
            )
            for fold in ("I-1", "I0")
        }
        events = [event for fold in ("I-1", "I0") for event in events_by_fold[fold]]
        capacity = capacity_summary(events, k)
        raw_from_units = sum(int(row["hard_negative_intersection_total"]) for row in rows) / max(
            1, sum(int(row["hard_negative_actual_denominator_total"]) for row in rows)
        )
        if abs(raw_from_units - float(capacity["raw_micro_recall"])) > 1e-12:
            raise RuntimeError(f"{domain}: aggregate raw recall cross-check failed")
        metrics = {
            "pooled_headroom": (
                sum(int(row["beam200_hits"]) for row in rows)
                - sum(int(row["beam50_hits"]) for row in rows)
            )
            / users,
            "beam200_only_events": sum(int(row["beam200_only_events"]) for row in rows),
            "nonempty_actual_pruner_fraction": nonempty / max(1, first_drop),
            "finite_and_trie_legal": all(row["finite_and_trie_legal"] for row in rows),
            "cf_target_z_mean_drift": units[f"{domain}:I0"]["target_cf_z"]["mean"]
            - units[f"{domain}:I-1"]["target_cf_z"]["mean"],
            "capacity": capacity,
        }
        gates = confirmation["gate"]
        inherited = {
            "gate1_headroom": metrics["pooled_headroom"] >= gates["per_domain_headroom_min"],
            "gate2_beam200_only_events": metrics["beam200_only_events"]
            >= gates["per_domain_beam200_only_events_min"],
            "gate3_nonempty_actual_pruner": metrics["nonempty_actual_pruner_fraction"]
            >= gates["per_domain_nonempty_actual_pruner_fraction_min"],
            "gate5_finite_and_trie_legal": bool(metrics["finite_and_trie_legal"]),
            "gate6_cf_target_z_drift": abs(metrics["cf_target_z_mean_drift"])
            < gates["absolute_cf_target_z_mean_drift_max_exclusive"],
        }
        repaired = evaluate_capacity_gate(capacity, gates)
        passed = all(inherited.values()) and repaired["decision"] == "CAPACITY_ADJUSTED_ACTIONABILITY_PASS"
        domains[domain] = {
            "metrics": metrics,
            "inherited_checks": inherited,
            "capacity_gate": repaired,
            "decision": "PASS" if passed else "FAIL",
            "folds": {fold: units[f"{domain}:{fold}"] for fold in ("I-1", "I0")},
        }
    passed = all(row["decision"] == "PASS" for row in domains.values())
    return {
        "schema_version": "phase18.s18_1r_disjoint_confirmation_summary.v1",
        "experiment_id": "s18_s1r_disjoint_confirmation",
        "attempt_id": "run-0001",
        "status": "COMPLETED",
        "decision": (
            "S18_1R_ACTIONABILITY_REPAIRED_PASS"
            if passed
            else "S18_1R_DISJOINT_CONFIRMATION_FAILED"
        ),
        "domains": domains,
        "confirmation_config_sha256": sha256(
            ROOT / "experiment/phase18/config/s18_s1r_disjoint_confirmation.json"
        ),
        "run_manifest_sha256": sha256(OUTPUT / "run_manifest.json"),
        "checkpoint_only": True,
        "parent_retraining": False,
        "item_head_retraining": False,
        "treatment_training": False,
        "i1_i2_read": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
        "completed_at": base.utc_now(),
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Stage18 S18-1R 未见 Cohort 确认报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill：`academic-research-suite / experiment-agent`",
        "- Origin Mode：`validate + run`",
        f"- Verification Status：`{summary['status']}`",
        f"- Decision：`{summary['decision']}`",
        "- Cohort：每域冻结哈希排序 `[1024,2048)`，与原 cohort 零重叠",
        "- Execution：仅复用四组 epoch-10 checkpoint；未重训，未读保护数据",
        "",
        "## Domain Gate",
        "",
        "| Domain | headroom | beam200-only | nonempty pruner | raw micro@8 | capacity-normalized@8 | any-hit@8 | CF z drift | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for domain in ("Toys", "Beauty"):
        row = summary["domains"][domain]
        metrics = row["metrics"]
        capacity = metrics["capacity"]
        lines.append(
            f"| {domain} | {metrics['pooled_headroom']:.6f} | {metrics['beam200_only_events']} | "
            f"{metrics['nonempty_actual_pruner_fraction']:.6f} | {capacity['raw_micro_recall']:.6f} | "
            f"{capacity['capacity_normalized_recall']:.6f} | {capacity['event_any_hit_rate']:.6f} | "
            f"{metrics['cf_target_z_mean_drift']:.6f} | `{row['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "本结果不自动启动 S18-2。I1/I2、D1/D2、official validation/test 与 Sports 均未读取。",
            "",
        ]
    )
    base.atomic_text(REPORT, "\n".join(lines))


def smoke_master() -> int:
    _, _, _, authorization = load_contracts()
    manifest = load_json(SMOKE / "run_manifest.json")
    if manifest["source_manifest"] != source_manifest():
        raise RuntimeError("smoke source manifest changed after launch")
    labels = ["Toys:I0", "Beauty:I0"]
    update_status(
        SMOKE_STATUS,
        execution_state="SMOKE_RUNNING",
        scientific_state="NON_SCIENTIFIC_SMOKE",
        process_alive=True,
        workload_pid=os.getpid(),
    )
    for completed, label in enumerate(labels):
        process, log = spawn_unit(
            SMOKE, label, authorization["runtime"]["smoke_users_per_domain"]
        )
        code = wait_unit(process, log, SMOKE, label, SMOKE_STATUS, completed, len(labels))
        if code != 0:
            return terminate_failure(SMOKE_STATUS, label, code, log)
    for label in labels:
        domain, fold = label.split(":", 1)
        summary = load_json(unit_dir(SMOKE, domain, fold) / "summary.json")
        rows = (unit_dir(SMOKE, domain, fold) / "per_user_diagnostics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        expected_user = (PREFLIGHT / f"cohort_{domain}.txt").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        if (
            summary.get("users") != authorization["runtime"]["smoke_users_per_domain"]
            or not summary.get("finite_and_trie_legal")
            or len(rows) != authorization["runtime"]["smoke_users_per_domain"]
            or json.loads(rows[0])["user"] != expected_user
        ):
            raise RuntimeError(f"{label}: smoke result is not valid")
    payload = {
        "schema_version": "phase18.s18_1r_disjoint_smoke.v1",
        "status": "PASSED",
        "attempt_id": "run-0001-smoke",
        "units": labels,
        "users_per_domain": authorization["runtime"]["smoke_users_per_domain"],
        "source_manifest": source_manifest(),
        "scientific_result_eligible": False,
        "parent_retraining": False,
        "item_head_retraining": False,
        "treatment_training": False,
        "protected_data_read": False,
        "automatic_s18_2": False,
        "completed_at": base.utc_now(),
    }
    base.atomic_json(SMOKE / "summary.json", payload)
    update_status(
        SMOKE_STATUS,
        execution_state="COMPLETED",
        scientific_state="NON_SCIENTIFIC_SMOKE_COMPLETED",
        status_code="S18_1R_DISJOINT_SMOKE_PASS",
        process_alive=False,
        workload_pid=0,
        progress={"current": 2, "total": 2, "unit": "domain_smoke"},
        summary_path=str((SMOKE / "summary.json").relative_to(ROOT)),
        summary_sha256=sha256(SMOKE / "summary.json"),
        next_action="Launch the separately authorized formal disjoint confirmation.",
    )
    return 0


def formal_master() -> int:
    _, _, _, authorization = load_contracts()
    manifest = load_json(OUTPUT / "run_manifest.json")
    if manifest["source_manifest"] != source_manifest():
        raise RuntimeError("formal source manifest changed after launch")
    labels = authorization["runtime"]["serial_units"]
    update_status(
        STATUS,
        execution_state="RUNNING_BOUNDED_GENERATION",
        scientific_state="RUNNING",
        status_code="S18_1R_DISJOINT_RUNNING",
        process_alive=True,
        workload_pid=os.getpid(),
    )
    for completed, label in enumerate(labels):
        process, log = spawn_unit(OUTPUT, label, None)
        code = wait_unit(process, log, OUTPUT, label, STATUS, completed, len(labels))
        if code != 0:
            return terminate_failure(STATUS, label, code, log)
        update_status(
            STATUS,
            progress={"current": completed + 1, "total": len(labels), "unit": "domain_fold_diagnostic"},
        )
    summary = aggregate()
    base.atomic_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    base.append_jsonl(
        LEDGER,
        {
            "at": summary["completed_at"],
            "attempt_id": "run-0001",
            "event": "disjoint_confirmation_completed",
            "decision": summary["decision"],
            "summary_sha256": sha256(OUTPUT / "summary.json"),
            "automatic_s18_2": False,
        },
    )
    update_status(
        STATUS,
        execution_state="SCIENTIFIC_COMPLETED",
        scientific_state="COMPLETED",
        status_code=summary["decision"],
        process_alive=False,
        workload_pid=0,
        current_unit=None,
        current_unit_pid=0,
        progress={"current": 4, "total": 4, "unit": "domain_fold_diagnostic"},
        summary_path=str((OUTPUT / "summary.json").relative_to(ROOT)),
        summary_sha256=sha256(OUTPUT / "summary.json"),
        report_path=str(REPORT.relative_to(ROOT)),
        result_selection_eligible=True,
        automatic_s18_2=False,
        next_action="Review this repaired Gate; S18-2 remains unstarted and requires separate authorization.",
    )
    return 0


def launch_smoke() -> int:
    _, _, _, authorization = load_contracts()
    if SMOKE.exists() or SMOKE_STATUS.exists():
        raise FileExistsError("S18-1R-B smoke exists; automatic rerun forbidden")
    admission = stable_admission(authorization)
    SMOKE.mkdir(parents=True)
    manifest = {
        "schema_version": "phase18.s18_1r_disjoint_smoke_manifest.v1",
        "experiment_id": "s18_s1r_disjoint_smoke",
        "attempt_id": "run-0001-smoke",
        "created_at": base.utc_now(),
        "physical_gpus": authorization["runtime"]["physical_gpus"],
        "admission_history": admission,
        "source_manifest": source_manifest(),
        "scientific_result_eligible": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(SMOKE / "run_manifest.json", manifest)
    update_status(
        SMOKE_STATUS,
        reset=True,
        schema_version="phase18.status.v1",
        experiment_id="s18_s1r_disjoint_smoke",
        attempt_id="run-0001-smoke",
        execution_state="STARTING",
        scientific_state="NON_SCIENTIFIC_SMOKE",
        process_alive=True,
        workload_pid=0,
        tmux_session=authorization["runtime"]["smoke_tmux_session"],
        physical_gpus=authorization["runtime"]["physical_gpus"],
        result_selection_eligible=False,
        automatic_s18_2=False,
    )
    command = [
        "/usr/bin/env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={ROOT}",
        str(PYTHON),
        str(ENTRY),
        "smoke-master",
    ]
    launch_background_tmux(
        experiment_id="s18_s1r_disjoint_smoke",
        argv=command,
        cwd=ROOT,
        tmux_session=authorization["runtime"]["smoke_tmux_session"],
        startup_log_path=SMOKE / "master.log",
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if load_json(SMOKE_STATUS).get("workload_pid", 0) > 0:
            print(json.dumps({"status": "STARTED", "status_path": str(SMOKE_STATUS.relative_to(ROOT))}))
            return 0
        time.sleep(1)
    raise RuntimeError("S18-1R-B smoke failed startup handshake")


def launch_formal() -> int:
    _, _, _, authorization = load_contracts()
    smoke_path = SMOKE / "summary.json"
    if not smoke_path.is_file():
        raise RuntimeError("formal confirmation requires a completed smoke")
    smoke = load_json(smoke_path)
    if smoke.get("status") != "PASSED" or smoke.get("source_manifest") != source_manifest():
        raise RuntimeError("formal confirmation smoke is not passed and current")
    if OUTPUT.exists():
        raise FileExistsError("S18-1R-B formal output exists; automatic retry forbidden")
    admission = stable_admission(authorization)
    OUTPUT.mkdir(parents=True)
    manifest = {
        "schema_version": "phase18.s18_1r_disjoint_run_manifest.v1",
        "experiment_id": "s18_s1r_disjoint_confirmation",
        "attempt_id": "run-0001",
        "created_at": base.utc_now(),
        "physical_gpus": authorization["runtime"]["physical_gpus"],
        "serial_units": authorization["runtime"]["serial_units"],
        "admission_history": admission,
        "smoke_path": str(smoke_path.relative_to(ROOT)),
        "smoke_sha256": sha256(smoke_path),
        "source_manifest": source_manifest(),
        "checkpoint_only": True,
        "parent_retraining": False,
        "item_head_retraining": False,
        "treatment_training": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(OUTPUT / "run_manifest.json", manifest)
    update_status(
        STATUS,
        reset=True,
        schema_version="phase18.status.v1",
        experiment_id="s18_s1r_disjoint_confirmation",
        attempt_id="run-0001",
        step_id="S18-1R-B",
        execution_state="STARTING",
        scientific_state="RUNNING",
        status_code="S18_1R_DISJOINT_STARTING",
        process_alive=True,
        workload_pid=0,
        tmux_session=authorization["runtime"]["formal_tmux_session"],
        physical_gpus=authorization["runtime"]["physical_gpus"],
        progress={"current": 0, "total": 4, "unit": "domain_fold_diagnostic"},
        run_manifest_path=str((OUTPUT / "run_manifest.json").relative_to(ROOT)),
        run_manifest_sha256=sha256(OUTPUT / "run_manifest.json"),
        checkpoint_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        treatment_training=False,
        result_selection_eligible=False,
        automatic_retry=False,
        automatic_s18_2=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
    )
    base.append_jsonl(
        LEDGER,
        {
            "at": base.utc_now(),
            "attempt_id": "run-0001",
            "event": "disjoint_confirmation_started",
            "physical_gpus": authorization["runtime"]["physical_gpus"],
            "run_manifest_sha256": sha256(OUTPUT / "run_manifest.json"),
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
        str(ENTRY),
        "formal-master",
    ]
    launch_background_tmux(
        experiment_id="s18_s1r_disjoint_confirmation",
        argv=command,
        cwd=ROOT,
        tmux_session=authorization["runtime"]["formal_tmux_session"],
        startup_log_path=OUTPUT / "master.log",
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if load_json(STATUS).get("workload_pid", 0) > 0:
            print(json.dumps({"status": "STARTED", "status_path": str(STATUS.relative_to(ROOT))}))
            return 0
        time.sleep(1)
    raise RuntimeError("S18-1R-B formal confirmation failed startup handshake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("verify", "launch-smoke", "smoke-master", "launch", "formal-master", "unit"),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--domain", choices=("Toys", "Beauty"))
    parser.add_argument("--fold", choices=("I-1", "I0"))
    parser.add_argument("--max-users", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify":
        _, _, _, authorization = load_contracts()
        print(
            json.dumps(
                {
                    "status": "VERIFIED",
                    "physical_gpus": authorization["runtime"]["physical_gpus"],
                    "source_manifest": source_manifest(),
                }
            )
        )
        return 0
    if args.action == "launch-smoke":
        return launch_smoke()
    if args.action == "smoke-master":
        return smoke_master()
    if args.action == "launch":
        return launch_formal()
    if args.action == "formal-master":
        return formal_master()
    if args.action == "unit":
        if args.root is None or args.domain is None or args.fold is None:
            raise ValueError("unit requires --root, --domain and --fold")
        root = args.root.resolve()
        if root not in {OUTPUT, SMOKE}:
            raise PermissionError(f"unauthorized unit root: {root}")
        return run_unit(root, args.domain, args.fold, args.max_users)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
