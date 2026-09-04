#!/usr/bin/env python3
"""Immutable background runtime for the FP3 train-only SASRec tokenizer."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiment.phase17.core.full_setrec_backend import load_setrec_catalog
from experiment.phase17.core.full_setrec_cf_tokenizer import (
    SetRecCFSpec,
    train_setrec_cf_tokenizer,
)
from experiment.phase17.core.fullport_data import (
    build_train_and_internal_dev_examples,
    read_train_prefix_users,
    select_internal_dev_users,
)
from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    verify_run_snapshot,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import (
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)


ROOT = Path(__file__).resolve().parents[3]
NATIVE_PYTHON_SUFFIX = Path(
    "artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python"
)
DATA_SUFFIX = Path("artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt")
FP3_CONFIG_SUFFIX = Path("experiment/phase17/config/s17_fp3_setrec.json")
SEMANTIC_MANIFEST_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/attempt_001/tokenizer/manifest.json"
)
SEMANTIC_FEATURE_SUFFIX = SEMANTIC_MANIFEST_SUFFIX.parent / "sentence_embeddings.npy"
EXPERIMENT_ID = "s17_fp3_setrec_cf_tokenizer"
STEP_ID = "S17-FP3-TOKENIZER"
ATTEMPT_ID = "attempt_001"
TMUX_SESSION = EXPERIMENT_ID
TARGET_GPU_ID = 7
MINIMUM_FREE_MIB = 4096
EXPECTED_USERS = 12833
EXPECTED_TRAIN_EXAMPLES = 56421
EXPECTED_DEV_EXAMPLES = 1283
EXPECTED_CATALOG_ITEMS = 11924


def paths(root: Path) -> dict[str, Path]:
    result = root / f"artifacts/phase17/fullport/fp3_setrec/tokenizer/{ATTEMPT_ID}"
    snapshot = root / f"artifacts/phase17/snapshots/{EXPERIMENT_ID}/{ATTEMPT_ID}/manifest.json"
    return {
        "result": result,
        "output": result / "sasrec_item_embeddings.pt",
        "config": result / "config.json",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "authorization": root
        / f"artifacts/phase17/authorizations/{EXPERIMENT_ID}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / "artifacts/phase17/attempts/S17-FP3.attempts.jsonl",
        "snapshot": snapshot,
        "snapshot_worker": snapshot.parent
        / "src/000_s17_fp3_setrec_tokenizer_runtime.py",
        "native_python": root / NATIVE_PYTHON_SUFFIX,
        "data": root / DATA_SUFFIX,
        "fp3_config": root / FP3_CONFIG_SUFFIX,
        "semantic_manifest": root / SEMANTIC_MANIFEST_SUFFIX,
        "semantic_feature": root / SEMANTIC_FEATURE_SUFFIX,
    }


def _gpu_state() -> dict[str, Any]:
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    devices: list[dict[str, Any]] = []
    uuid_to_index: dict[str, int] = {}
    for row in csv.reader(io.StringIO(raw)):
        index = int(row[0].strip())
        uuid_to_index[row[1].strip()] = index
        devices.append(
            {
                "index": index,
                "uuid": row[1].strip(),
                "total_mib": int(row[2].strip()),
                "used_mib": int(row[3].strip()),
                "free_mib": int(row[4].strip()),
                "utilization_percent": int(row[5].strip()),
            }
        )
    process_raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    processes: dict[int, list[dict[str, Any]]] = {
        row["index"]: [] for row in devices
    }
    for row in csv.reader(io.StringIO(process_raw)):
        if len(row) != 4 or row[0].strip() not in uuid_to_index:
            continue
        processes[uuid_to_index[row[0].strip()]].append(
            {
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "used_memory_mib": int(row[3].strip()),
            }
        )
    return {"captured_at": utc_now(), "devices": devices, "compute_processes": processes}


def _selected(snapshot: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in snapshot["devices"] if row["index"] == TARGET_GPU_ID]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU{TARGET_GPU_ID} is not uniquely visible")
    return matches[0]


def two_snapshot_admission() -> dict[str, Any]:
    first = _gpu_state()
    time.sleep(5)
    second = _gpu_state()
    for name, state in (("first", first), ("second", second)):
        free = _selected(state)["free_mib"]
        if free < MINIMUM_FREE_MIB:
            raise RuntimeError(
                f"GPU{TARGET_GPU_ID} {name} free {free} MiB is below {MINIMUM_FREE_MIB} MiB"
            )
    return {
        "selected_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "first": first,
        "second": second,
        "preexisting_compute_processes": second["compute_processes"].get(
            TARGET_GPU_ID, []
        ),
        "automatic_process_termination": False,
    }


def frozen_config(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    users = read_train_prefix_users(resolved["data"], root=root)
    dev_ids = select_internal_dev_users(
        users, count=EXPECTED_DEV_EXAMPLES, seed=2023
    )
    train, dev = build_train_and_internal_dev_examples(
        users, dev_ids, max_history_items=20
    )
    catalog = load_setrec_catalog(root, require_cf=False)
    observed = (len(users), len(train), len(dev), len(catalog.ordered_items))
    expected = (
        EXPECTED_USERS,
        EXPECTED_TRAIN_EXAMPLES,
        EXPECTED_DEV_EXAMPLES,
        EXPECTED_CATALOG_ITEMS,
    )
    if observed != expected:
        raise RuntimeError(f"FP3 tokenizer data contract drift: {observed} != {expected}")
    return {
        "schema_version": "phase17.s17_fp3_setrec_cf_tokenizer_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "step_id": STEP_ID,
        "attempt_id": ATTEMPT_ID,
        "prepared_at": utc_now(),
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "spec": asdict(SetRecCFSpec()),
        "users": len(users),
        "train_examples": len(train),
        "internal_dev_examples": len(dev),
        "catalog_items": len(catalog.ordered_items),
        "internal_dev_user_ids": list(dev_ids),
        "inputs": {
            "data_path": str(resolved["data"].relative_to(root)),
            "data_sha256": sha256(resolved["data"]),
            "fp3_config_path": str(resolved["fp3_config"].relative_to(root)),
            "fp3_config_sha256": sha256(resolved["fp3_config"]),
            "semantic_manifest_sha256": sha256(resolved["semantic_manifest"]),
            "semantic_feature_sha256": sha256(resolved["semantic_feature"]),
        },
        "checkpoint_rule": "maximum internal-dev NDCG@10; ties choose earlier epoch",
        "full_catalog_internal_dev": True,
        "background_required": True,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


def worker_command(root: Path, resolved: dict[str, Path]) -> list[str]:
    return [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={TARGET_GPU_ID}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(resolved["native_python"]),
        str(resolved["snapshot_worker"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(resolved["snapshot"]),
    ]


def prepare(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    if resolved["result"].exists() or resolved["snapshot"].exists():
        raise FileExistsError("FP3 tokenizer attempt_001 already exists")
    if not resolved["native_python"].is_file():
        raise FileNotFoundError(resolved["native_python"])
    config = frozen_config(root)
    resolved["result"].mkdir(parents=True, exist_ok=False)
    atomic_json(resolved["config"], config)
    manifest = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=worker_command(root, resolved),
        source_paths=[
            Path(__file__),
            root / "experiment/phase17/core/full_setrec_cf_tokenizer.py",
            root / "experiment/phase17/core/full_setrec_backend.py",
            root / "experiment/phase17/core/full_setrec_contracts.py",
            root / "experiment/phase17/core/fullport_data.py",
            root / "experiment/phase17/core/run_manager.py",
            root / "experiment/phase17/core/status_writer.py",
        ],
        config=config,
    )
    AttemptLedger(resolved["ledger"]).append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": STEP_ID,
            "kind": "train_prefix_only_sasrec_cf_tokenizer",
            "started_at": utc_now(),
            "state": "PREFLIGHT_READY_EXACT_COMMAND_CONFIRMATION_REQUIRED",
            "scientific_result_eligible": False,
            "automatic_retry": False,
            "gpu_ids": [],
            "snapshot_manifest": str(manifest.relative_to(root)),
        }
    )
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    writer.initialize(
        step_id=STEP_ID,
        attempt_id=ATTEMPT_ID,
        track_id="FP3-TOKENIZER",
        canonical_result_dir=str(resolved["result"].relative_to(root)),
        log_path=str(resolved["log"].relative_to(root)),
        extra={
            "stage": "preflight_complete_waiting_exact_command_confirmation",
            "progress": {"current": 0, "total": 10, "unit": "epoch"},
            "run_snapshot_manifest": str(manifest.relative_to(root)),
            "gpu_ids": [],
            "target_gpu_id": TARGET_GPU_ID,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "launch_authorized": False,
            "automatic_process_termination": False,
            "automatic_retry": False,
            "external_target_materialized": False,
            "d1_read": False,
            "d2_read": False,
            "result_selection_eligible": False,
            "affects_scientific_result": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP3_SETREC_TOKENIZER_READY_EXACT_COMMAND_CONFIRMATION_REQUIRED",
        process_alive=False,
    )
    print(manifest)
    return 0


def authorize(root: Path, researcher_direction: str) -> int:
    root = root.resolve()
    resolved = paths(root)
    if not researcher_direction.strip():
        raise ValueError("authorize requires the researcher's exact-command confirmation")
    if resolved["authorization"].exists():
        raise FileExistsError(resolved["authorization"])
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"FP3 tokenizer is not authorizable: {status['scientific_state']}")
    snapshot = _gpu_state()
    if _selected(snapshot)["free_mib"] < MINIMUM_FREE_MIB:
        raise RuntimeError(f"GPU{TARGET_GPU_ID} lacks tokenizer headroom")
    payload = {
        "schema_version": "phase17.s17_fp3_setrec_tokenizer_authorization.v1",
        "authorized_at": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "authorized": True,
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "researcher_direction": researcher_direction,
        "confirmed_exact_command": (
            "bash experiment/phase17/run_stage17_fp3_setrec_tokenizer.sh launch"
        ),
        "tokenizer_only": True,
        "effect_experiment_authorized": False,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "gpu_snapshot": snapshot,
    }
    atomic_json(resolved["authorization"], payload)
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP3_SETREC_TOKENIZER_AUTHORIZED_READY_TO_LAUNCH",
        launch_authorized=True,
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        gpu_snapshot=snapshot,
    )
    print(resolved["authorization"])
    return 0


def verify_authorization(root: Path) -> dict[str, Any]:
    resolved = paths(root)
    payload = json.loads(resolved["authorization"].read_text(encoding="utf-8"))
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "authorized": True,
        "physical_gpu": TARGET_GPU_ID,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "tokenizer_only": True,
        "effect_experiment_authorized": False,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"invalid FP3 tokenizer authorization field: {key}")
    if not payload.get("researcher_direction"):
        raise PermissionError("FP3 tokenizer authorization lacks researcher direction")
    return payload


def launch(root: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"FP3 tokenizer is not launchable: {status['scientific_state']}")
    verify_authorization(root)
    admission = two_snapshot_admission()
    session = launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=worker_command(root, resolved),
        cwd=root,
        tmux_session=TMUX_SESSION,
        startup_log_path=resolved["log"],
    )
    writer.transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_FP3_SETREC_TOKENIZER_BACKGROUND_STARTED",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
        stage="background_started",
        gpu_ids=[TARGET_GPU_ID],
        gpu_snapshot=admission,
        launch_authorized=True,
    )
    if not wait_for_tmux_startup(session):
        latest = writer.read()
        if latest["scientific_state"] == "RUNNING":
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP3_SETREC_TOKENIZER_STARTUP_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
            )
        raise RuntimeError("FP3 tokenizer exited during startup handshake")
    print(session)
    return 0


def worker(root: Path, manifest_path: Path) -> int:
    root = root.resolve()
    resolved = paths(root)
    writer = StatusWriter(resolved["status_dir"], EXPERIMENT_ID)
    started = time.monotonic()
    try:
        verify_run_snapshot(root, manifest_path)
        authorization = verify_authorization(root)
        admission = two_snapshot_admission()
        config = json.loads(resolved["config"].read_text(encoding="utf-8"))
        for path_key, hash_key in (
            ("data", "data_sha256"),
            ("fp3_config", "fp3_config_sha256"),
            ("semantic_manifest", "semantic_manifest_sha256"),
            ("semantic_feature", "semantic_feature_sha256"),
        ):
            if sha256(resolved[path_key]) != config["inputs"][hash_key]:
                raise RuntimeError(f"FP3 tokenizer input drift: {path_key}")
        import torch

        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        users = read_train_prefix_users(resolved["data"], root=root)
        dev_ids = tuple(config["internal_dev_user_ids"])
        train, dev = build_train_and_internal_dev_examples(
            users, dev_ids, max_history_items=20
        )
        catalog = load_setrec_catalog(root, require_cf=False)
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "S17_FP3_SETREC_TOKENIZER_RUNNING",
            workload_pid=os.getpid(),
            process_alive=True,
            stage="sasrec_training",
            gpu_ids=[TARGET_GPU_ID],
            gpu_snapshot=admission,
        )

        def heartbeat(epoch: int, record: dict[str, Any]) -> None:
            writer.heartbeat(
                stage=f"sasrec_epoch_{epoch:02d}",
                progress={
                    "current": epoch,
                    "total": SetRecCFSpec().epochs,
                    "unit": "epoch",
                    "latest": record,
                },
            )

        payload = train_setrec_cf_tokenizer(
            ordered_items=catalog.ordered_items,
            train_examples=train,
            dev_examples=dev,
            output_path=resolved["output"],
            device=torch.device("cuda"),
            spec=SetRecCFSpec(),
            heartbeat=heartbeat,
        )
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / 1048576
        peak_reserved = torch.cuda.max_memory_reserved() / 1048576
        summary = {
            "schema_version": "phase17.s17_fp3_setrec_cf_tokenizer_summary.v1",
            "verdict": "PASS_S17_FP3_SETREC_CF_TOKENIZER",
            "completed_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "physical_gpu": TARGET_GPU_ID,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "artifact_path": str(resolved["output"].relative_to(root)),
            "artifact_sha256": sha256(resolved["output"]),
            "catalog_items": len(payload["ordered_items"]),
            "train_examples": payload["train_examples"],
            "internal_dev_examples": payload["internal_dev_examples"],
            "train_fit_item_count": payload["train_fit_item_count"],
            "best_epoch": payload["best_epoch"],
            "best_internal_dev_ndcg@10": payload["best_internal_dev_ndcg@10"],
            "learning_curve": payload["learning_curve"],
            "authorization_sha256": sha256(resolved["authorization"]),
            "researcher_direction": authorization["researcher_direction"],
            "automatic_process_termination": False,
            "automatic_retry": False,
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
            "next_gate": "S17_FP3_FOUR_ARM_RESOURCE_PROFILE",
        }
        atomic_json(resolved["summary"], summary)
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "PASS_S17_FP3_SETREC_CF_TOKENIZER",
            process_alive=False,
            workload_pid=0,
            stage="sasrec_tokenizer_complete",
            progress={"current": 10, "total": 10, "unit": "epoch"},
            gpu_ids=[],
            summary_path=str(resolved["summary"].relative_to(root)),
            summary_sha256=sha256(resolved["summary"]),
            artifact_path=summary["artifact_path"],
            artifact_sha256=summary["artifact_sha256"],
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
            external_target_materialized=False,
            result_selection_eligible=False,
            affects_scientific_result=False,
        )
        return 0
    except BaseException as error:
        resolved["result"].mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "phase17.failure.v1",
            "failed_at": utc_now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "automatic_process_termination": False,
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        atomic_json(resolved["failure"], failure)
        with resolved["log"].open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] terminal_error={error!r}\n")
            handle.write(failure["traceback"])
        current = writer.read()
        if current["scientific_state"] not in {
            "COMPLETED",
            "FAILED",
            "STOPPED",
            "BLOCKED",
        }:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "S17_FP3_SETREC_TOKENIZER_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                stage="terminal_failure_no_retry",
                gpu_ids=[],
                failure_path=str(resolved["failure"].relative_to(root)),
                failure_sha256=sha256(resolved["failure"]),
                terminal_error=repr(error),
                automatic_retry=False,
                automatic_process_termination=False,
                external_target_materialized=False,
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
        return 1


def inspect(root: Path) -> int:
    resolved = paths(root.resolve())
    payload: dict[str, Any] = {}
    status_path = resolved["status_dir"] / f"{EXPERIMENT_ID}.status.json"
    if status_path.is_file():
        payload["status"] = json.loads(status_path.read_text(encoding="utf-8"))
    if resolved["summary"].is_file():
        payload["summary"] = json.loads(resolved["summary"].read_text(encoding="utf-8"))
    if resolved["failure"].is_file():
        payload["failure"] = json.loads(resolved["failure"].read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "authorize", "launch", "worker", "inspect")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--researcher-direction", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root)
    if args.action == "authorize":
        return authorize(root, args.researcher_direction)
    if args.action == "launch":
        return launch(root)
    if args.action == "inspect":
        return inspect(root)
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
