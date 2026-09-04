#!/usr/bin/env python3
"""Named checkpoint-only correction for the S18-1 JSON serialization failure."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from experiment.phase18.protocol import s18_s1_runtime as base


PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
EXPERIMENT_ID = "s18_s1_actionability_recovery"
ATTEMPT_ID = "run-0002"
RECOVERY_OF = "s18_s1_actionability/run-0001"
AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_recovery_authorization.json"
OUTPUT = ROOT / "artifacts/phase18/s1_actionability/run-0002"
SMOKE = ROOT / "artifacts/phase18/s1_actionability/recovery-smoke-run-0002"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_recovery.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1.attempts.jsonl"
REPORT = ROOT / "report/第十八阶段/Stage18_S1_基础设施修复恢复报告.md"
CANONICAL_SUMMARY = ROOT / "artifacts/phase18/s1_actionability/summary.json"
CANONICAL_MANIFEST = ROOT / "artifacts/phase18/s1_actionability/canonical_manifest.json"


def unit_key(domain: str, fold: str) -> str:
    return base.unit_key(domain, fold)


def unit_dir(domain: str, fold: str) -> Path:
    return OUTPUT / "units" / unit_key(domain, fold)


def update_status(**fields: Any) -> None:
    current = load_json(STATUS) if STATUS.is_file() else {}
    current.update(fields)
    current["updated_at"] = base.utc_now()
    current["heartbeat_at"] = base.utc_now()
    base.atomic_json(STATUS, current)


def resolve_frozen_input(record: dict[str, str]) -> Path:
    """Resolve a frozen input, including an exact-SHA immutable status archive.

    Early S18-1 authorizations referenced the mutable canonical status path.
    Later named attempts correctly archived that payload before replacing the
    canonical file.  The fallback is deliberately limited to status/history and
    still requires the authorization's exact SHA-256.
    """

    path = ROOT / record["path"]
    if path.is_file() and sha256(path) == record["sha256"]:
        return path
    canonical_status = ROOT / "artifacts/phase18/status/s18_s1_actionability.status.json"
    if path != canonical_status:
        raise RuntimeError(f"frozen recovery input mismatch: {path}")
    history = ROOT / "artifacts/phase18/status/history"
    matches = [
        candidate
        for candidate in sorted(history.glob("s18_s1_actionability.*.status.json"))
        if candidate.is_file() and sha256(candidate) == record["sha256"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"frozen recovery input mismatch: {path}")
    return matches[0]


def verify_authorization(*, verify_checkpoint_hashes: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    auth = load_json(AUTH_PATH)
    config = load_json(base.CONFIG_PATH)
    if auth["experiment_id"] != EXPERIMENT_ID or auth["attempt_id"] != ATTEMPT_ID:
        raise RuntimeError("recovery authorization identity mismatch")
    if auth["recovery_of"] != RECOVERY_OF:
        raise RuntimeError("unexpected recovery source")
    scope = auth["correction_scope"]
    required_false = (
        "parent_retraining",
        "item_head_retraining",
        "scientific_config_changes",
        "effect_results_read_before_correction",
        "automatic_retry",
        "automatic_s18_2",
    )
    if not scope["checkpoint_only_diagnostic_recovery"] or any(scope[name] for name in required_false):
        raise RuntimeError("recovery scope is not checkpoint-only")
    frozen_paths = {
        name: resolve_frozen_input(record)
        for name, record in auth["frozen_inputs"].items()
    }
    failed = load_json(frozen_paths["failed_status"])
    if failed.get("scientific_state") != "FAILED" or failed.get("status_code") != "S18_1_UNIT_FAILURE_NO_RETRY":
        raise RuntimeError("source attempt is not the expected terminal infrastructure failure")
    if failed.get("d1_read") or failed.get("d2_read") or failed.get("test_read") or failed.get("sports_read"):
        raise RuntimeError("protected-data flags changed in source attempt")
    errors = {
        row.get("unit_status", {}).get("error")
        for row in failed.get("units", {}).values()
    }
    if errors != {"Object of type bool_ is not JSON serializable"}:
        raise RuntimeError(f"source attempt failure is not homogeneous: {errors}")
    if verify_checkpoint_hashes:
        for label, pair in auth["checkpoints"].items():
            for role, record in pair.items():
                path = ROOT / record["path"]
                if not path.is_file() or sha256(path) != record["sha256"]:
                    raise RuntimeError(f"{label} {role} checkpoint mismatch")
    return config, auth


def checkpoint_record(auth: dict[str, Any], domain: str, fold: str, role: str) -> dict[str, str]:
    return auth["checkpoints"][f"{domain}:{fold}"][role]


def load_frozen_models(
    config: dict[str, Any], auth: dict[str, Any], domain: str, fold: str, device: torch.device
):
    parent_record = checkpoint_record(auth, domain, fold, "parent")
    parent_path = ROOT / parent_record["path"]
    parent_state = torch.load(parent_path, map_location="cpu")
    if (
        parent_state.get("schema_version") != "phase18.s18_1_parent.v1"
        or parent_state.get("domain") != domain
        or parent_state.get("fold") != fold
        or parent_state.get("seed") != config["seed"]
        or parent_state.get("epochs") != 10
        or parent_state.get("target_based_selection") is not False
    ):
        raise RuntimeError(f"{domain}/{fold}: parent checkpoint metadata mismatch")
    parent = base.initialize_parent(config, device)
    parent.load_state_dict(parent_state["model_state_dict"], strict=True)
    parent.eval()
    parent_history = parent_state["history"]
    del parent_state

    domain_config = config["domains"][domain]
    index_name = f"item_generative_indexing_{domain_config['hierarchy']}.txt"
    dataset_dir = base.PREFLIGHT / "data" / base.dataset_name_from_manifest(domain, fold)
    item_to_id, samples, frequencies, sequences = base.read_numeric_fold_data(dataset_dir, index_name)
    head_record = checkpoint_record(auth, domain, fold, "item_head")
    head_path = ROOT / head_record["path"]
    head_state = torch.load(head_path, map_location="cpu")
    if (
        head_state.get("schema_version") != "phase18.s18_1_item_head.v1"
        or head_state.get("domain") != domain
        or head_state.get("fold") != fold
        or head_state.get("seed") != config["seed"]
        or head_state.get("target_based_selection") is not False
    ):
        raise RuntimeError(f"{domain}/{fold}: item-head checkpoint metadata mismatch")
    model_config = head_state["model_config"]
    if model_config["num_items"] != len(item_to_id):
        raise RuntimeError(f"{domain}/{fold}: item-head catalog size mismatch")
    item_head = base.CF0B2ItemHead(
        num_items=model_config["num_items"],
        max_history=model_config["max_history"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        temperature=model_config["temperature_initial"],
    ).to(device)
    item_head.load_state_dict(head_state["model_state_dict"], strict=True)
    item_head.eval()
    head_history = head_state["history"]
    del head_state

    args = base.gram_args(config, domain, fold)
    return parent, args, item_head, item_to_id, frequencies, sequences, {
        "parent": {
            "checkpoint": parent_record["path"],
            "checkpoint_sha256": parent_record["sha256"],
            "history": parent_history,
            "reused_without_training": True,
        },
        "item_head": {
            "checkpoint": head_record["path"],
            "checkpoint_sha256": head_record["sha256"],
            "history": head_history,
            "samples": len(samples),
            "reused_without_training": True,
        },
    }


def configure_base_output(path: Path) -> None:
    base.OUTPUT = path


def smoke(physical_gpu: int) -> int:
    config, auth = verify_authorization()
    if SMOKE.exists() and any(SMOKE.iterdir()):
        raise FileExistsError(f"recovery smoke already exists: {SMOKE}")
    SMOKE.mkdir(parents=True, exist_ok=True)
    configure_base_output(SMOKE)
    base.unit_dir("Toys", "I0").mkdir(parents=True)
    base.set_seed(config["seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"]["snapshot"], local_files_only=True)
    parent, args, item_head, item_to_id, frequencies, sequences, provenance = load_frozen_models(
        config, auth, "Toys", "I0", device
    )
    args.tokenizer = tokenizer
    started = time.time()
    diagnostic = base.diagnose(
        config,
        "Toys",
        "I0",
        device,
        tokenizer,
        parent,
        args,
        item_head,
        item_to_id,
        frequencies,
        sequences,
        max_users=1,
    )
    payload = {
        "schema_version": "phase18.s18_1_recovery_smoke.v1",
        "status": "PASSED",
        "physical_gpu": physical_gpu,
        "peak_allocated_mib": diagnostic["peak_allocated_mib"],
        "peak_reserved_mib": diagnostic["peak_reserved_mib"],
        "wall_time_seconds": time.time() - started,
        "one_user_diagnostic": diagnostic,
        "checkpoint_provenance": provenance,
        "scientific_result_eligible": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    base.atomic_json(SMOKE / "summary.json", payload)
    print(json.dumps(payload, default=base.json_default))
    return 0


def run_unit(domain: str, fold: str, physical_gpu: int) -> int:
    config, auth = verify_authorization()
    configure_base_output(OUTPUT)
    target = unit_dir(domain, fold)
    if target.exists():
        raise FileExistsError(f"recovery unit output exists; automatic retry forbidden: {target}")
    target.mkdir(parents=True)
    base.atomic_json(
        target / "status.json",
        {
            "schema_version": "phase18.s18_1_recovery_unit_status.v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "domain": domain,
            "fold": fold,
            "unit": unit_key(domain, fold),
            "scientific_attempt": ATTEMPT_ID,
            "execution_state": "STARTING_CHECKPOINT_ONLY_DIAGNOSTIC",
            "phase": "checkpoint_load",
            "physical_gpu": physical_gpu,
            "pid": os.getpid(),
            "process_alive": True,
            "parent_retraining": False,
            "item_head_retraining": False,
            "started_at": base.utc_now(),
            "heartbeat_at": base.utc_now(),
        },
    )
    started = time.time()
    try:
        base.set_seed(config["seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(config["backbone"]["snapshot"], local_files_only=True)
        parent, args, item_head, item_to_id, frequencies, sequences, provenance = load_frozen_models(
            config, auth, domain, fold, device
        )
        args.tokenizer = tokenizer
        base.update_unit_status(
            domain,
            fold,
            execution_state="RUNNING_CHECKPOINT_ONLY_DIAGNOSTIC",
            phase="beam50_beam200_diagnostic",
            parent_retraining=False,
            item_head_retraining=False,
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
        )
        summary = {
            **diagnostic,
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "status": "COMPLETED",
            "parent_training": provenance["parent"],
            "item_head_training": provenance["item_head"],
            "physical_gpu": physical_gpu,
            "wall_time_total_seconds": time.time() - started,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
            "treatment_training": False,
            "parent_retraining": False,
            "item_head_retraining": False,
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


def eligible_gpu_waves(auth: dict[str, Any], smoke_summary: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], int]:
    snapshot = base.gpu_snapshot()
    required = int(math.ceil(float(smoke_summary["peak_reserved_mib"]))) + int(
        auth["runtime"]["memory_buffer_mib_above_smoke_peak"]
    )
    candidates = set(auth["runtime"]["candidate_physical_gpus"])
    eligible = sorted(
        (row for row in snapshot if row["index"] in candidates and row["free_mib"] >= required),
        key=lambda row: (-row["free_mib"], row["index"]),
    )
    if len(eligible) < auth["runtime"]["minimum_parallel_gpus"]:
        raise RuntimeError(f"no authorized GPU has the required {required} MiB free")
    labels = ["Toys:I0", "Toys:I-1", "Beauty:I0", "Beauty:I-1"]
    waves: list[list[dict[str, Any]]] = []
    width = min(len(eligible), len(labels))
    for offset in range(0, len(labels), width):
        waves.append(
            [
                {"label": label, "physical_gpu": eligible[index]["index"]}
                for index, label in enumerate(labels[offset : offset + width])
            ]
        )
    return waves, snapshot, required


def source_manifest(auth: dict[str, Any]) -> dict[str, str]:
    paths = [
        AUTH_PATH,
        base.CONFIG_PATH,
        base.PREFLIGHT / "manifest.json",
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        ROOT / "experiment/phase18/protocol/s18_s1_postrun_guard.py",
        ROOT / "experiment/phase18/core/s1_contracts.py",
        ROOT / "experiment/phase18/core/contracts.py",
    ]
    records = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for pair in auth["checkpoints"].values():
        for record in pair.values():
            records[record["path"]] = record["sha256"]
    return records


def launch() -> int:
    config, auth = verify_authorization()
    smoke_summary = load_json(SMOKE / "summary.json")
    if smoke_summary.get("status") != "PASSED":
        raise RuntimeError("recovery launch requires a passed checkpoint-only smoke")
    if OUTPUT.exists() or STATUS.exists():
        raise FileExistsError("run-0002 recovery artifacts already exist; automatic retry forbidden")
    waves, snapshot, required = eligible_gpu_waves(auth, smoke_summary)
    OUTPUT.mkdir(parents=True)
    manifest = {
        "schema_version": "phase18.s18_1_recovery_run_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "recovery_of": RECOVERY_OF,
        "created_at": base.utc_now(),
        "scientific_config_sha256": sha256(base.CONFIG_PATH),
        "recovery_authorization_sha256": sha256(AUTH_PATH),
        "source_manifest": source_manifest(auth),
        "gpu_snapshot": snapshot,
        "required_free_mib": required,
        "waves": waves,
        "checkpoint_only": True,
        "parent_retraining": False,
        "item_head_retraining": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
    }
    base.atomic_json(OUTPUT / "run_manifest.json", manifest)
    session = auth["runtime"]["tmux_session"]
    update_status(
        schema_version="phase18.status.v1",
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of=RECOVERY_OF,
        step_id="S18-1-INFRASTRUCTURE-CORRECTION",
        stage="background_starting",
        execution_state="RUNNING_SCIENTIFIC",
        scientific_state="RUNNING",
        status_code="S18_1_RECOVERY_STARTING",
        process_alive=True,
        workload_pid=0,
        tmux_session=session,
        gpu_ids=sorted({entry["physical_gpu"] for wave in waves for entry in wave}),
        gpu_schedule=waves,
        progress={"current": 0, "total": 4, "unit": "domain_fold_diagnostic"},
        run_manifest_path=str((OUTPUT / "run_manifest.json").relative_to(ROOT)),
        run_manifest_sha256=sha256(OUTPUT / "run_manifest.json"),
        result_selection_eligible=True,
        affects_scientific_result=True,
        checkpoint_only=True,
        parent_retraining=False,
        item_head_retraining=False,
        automatic_retry=False,
        automatic_s18_2=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
        started_at=base.utc_now(),
        next_action="Observe this recovery status only; do not start S18-2 automatically.",
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
    try:
        launch_background_tmux(
            experiment_id=EXPERIMENT_ID,
            argv=command,
            cwd=ROOT,
            tmux_session=session,
            startup_log_path=OUTPUT / "master.log",
        )
    except Exception as error:
        update_status(
            execution_state="FAILED_NO_RETRY",
            scientific_state="FAILED",
            status_code="S18_1_RECOVERY_TMUX_START_FAILED_NO_RETRY",
            process_alive=False,
            error=str(error),
        )
        raise
    deadline = time.time() + 30
    while time.time() < deadline:
        status = load_json(STATUS)
        if status.get("workload_pid", 0) > 0 and status.get("status_code") == "S18_1_RECOVERY_RUNNING":
            print(json.dumps({"tmux_session": session, "status": str(STATUS.relative_to(ROOT)), "waves": waves}))
            return 0
        time.sleep(1)
    raise RuntimeError("recovery background master failed startup handshake")


def run_wave(wave: list[dict[str, Any]], auth: dict[str, Any], completed: int, states: dict[str, Any]) -> tuple[bool, int]:
    processes: dict[str, dict[str, Any]] = {}
    for entry in wave:
        label = entry["label"]
        physical_gpu = entry["physical_gpu"]
        domain, fold = label.split(":", 1)
        log = OUTPUT / "units" / f"{unit_key(domain, fold)}.launcher.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w", encoding="utf-8")
        command = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "unit",
            "--domain",
            domain,
            "--fold",
            fold,
            "--physical-gpu",
            str(physical_gpu),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes[label] = {"process": process, "handle": handle, "started": time.time(), "gpu": physical_gpu, "log": str(log.relative_to(ROOT))}
    timeout_seconds = auth["runtime"]["unit_hard_timeout_seconds"]
    while True:
        wave_finished = 0
        for label, record in processes.items():
            process = record["process"]
            return_code = process.poll()
            state = "RUNNING"
            if return_code is not None:
                wave_finished += 1
                state = "COMPLETED" if return_code == 0 else "FAILED_NO_RETRY"
            elif time.time() - record["started"] > timeout_seconds:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                wave_finished += 1
                state = "HARD_TIMEOUT_NO_RETRY"
            domain, fold = label.split(":", 1)
            status_path = unit_dir(domain, fold) / "status.json"
            states[label] = {
                "state": state,
                "pid": process.pid,
                "gpu": record["gpu"],
                "log": record["log"],
                "unit_status": load_json(status_path) if status_path.is_file() else None,
            }
        update_status(
            stage="checkpoint_only_diagnostics",
            execution_state="RUNNING_SCIENTIFIC",
            scientific_state="RUNNING",
            status_code="S18_1_RECOVERY_RUNNING",
            process_alive=True,
            workload_pid=os.getpid(),
            progress={"current": completed + wave_finished, "total": 4, "unit": "domain_fold_diagnostic"},
            units=states,
        )
        if wave_finished == len(processes):
            break
        time.sleep(auth["runtime"]["heartbeat_seconds"])
    for record in processes.values():
        record["handle"].close()
    failures = [label for label, record in processes.items() if record["process"].returncode != 0]
    return not failures, completed + len(processes)


def master() -> int:
    config, auth = verify_authorization()
    manifest = load_json(OUTPUT / "run_manifest.json")
    if manifest["source_manifest"] != source_manifest(auth):
        raise RuntimeError("recovery source manifest changed after launch")
    update_status(
        stage="checkpoint_only_diagnostics",
        execution_state="RUNNING_SCIENTIFIC",
        scientific_state="RUNNING",
        status_code="S18_1_RECOVERY_RUNNING",
        process_alive=True,
        workload_pid=os.getpid(),
    )
    base.append_jsonl(
        LEDGER,
        {
            "event": "named_infrastructure_correction_started",
            "at": base.utc_now(),
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "recovery_of": RECOVERY_OF,
            "run_manifest_sha256": sha256(OUTPUT / "run_manifest.json"),
            "parent_retraining": False,
            "item_head_retraining": False,
        },
    )
    completed = 0
    states: dict[str, Any] = {}
    for wave in manifest["waves"]:
        ok, completed = run_wave(wave, auth, completed, states)
        if not ok:
            base.append_jsonl(LEDGER, {"event": "named_infrastructure_correction_failed_no_retry", "at": base.utc_now(), "attempt_id": ATTEMPT_ID})
            update_status(
                stage="terminal_failure",
                execution_state="FAILED_NO_RETRY",
                scientific_state="FAILED",
                status_code="S18_1_RECOVERY_UNIT_FAILURE_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                result_selection_eligible=False,
            )
            return 1
    configure_base_output(OUTPUT)
    summary = base.aggregate_results(config)
    summary.update(
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        recovery_of=RECOVERY_OF,
        recovery_authorization_sha256=sha256(AUTH_PATH),
        original_resource_authorization_sha256=summary.pop("resource_authorization_sha256"),
        checkpoint_only=True,
        parent_retraining=False,
        item_head_retraining=False,
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
            "reason": "single named infrastructure correction after pre-effect JSON serialization failure",
            "created_at": base.utc_now(),
        },
    )
    base.REPORT = REPORT
    base.write_report(summary)
    base.append_jsonl(
        LEDGER,
        {
            "event": "named_infrastructure_correction_completed",
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
        [
            str(PYTHON),
            str(ROOT / "experiment/phase18/protocol/s18_s1_postrun_guard.py"),
            "launch",
        ],
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
            "tmux_session": auth["postrun_occupancy"]["tmux_session"],
            "status_path": auth["postrun_occupancy"]["status"],
            "physical_gpu": auth["postrun_occupancy"]["physical_gpu"],
            "result_selection_eligible": False,
            "repeat_metrics_ignored": True,
        }
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "smoke", "launch", "master", "unit"))
    parser.add_argument("--domain", choices=("Toys", "Beauty"))
    parser.add_argument("--fold", choices=("I-1", "I0"))
    parser.add_argument("--physical-gpu", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify":
        verify_authorization()
        print(json.dumps({"status": "VERIFIED", "attempt_id": ATTEMPT_ID}))
        return 0
    if args.action == "smoke":
        if args.physical_gpu is None:
            raise ValueError("smoke requires --physical-gpu")
        return smoke(args.physical_gpu)
    if args.action == "launch":
        return launch()
    if args.action == "master":
        return master()
    if args.action == "unit":
        if args.domain is None or args.fold is None or args.physical_gpu is None:
            raise ValueError("unit requires domain, fold, and physical GPU")
        return run_unit(args.domain, args.fold, args.physical_gpu)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
