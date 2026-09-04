#!/usr/bin/env python3
"""Fresh-process GPU4 occupancy guard after the corrected S18-1 result is complete."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.run_manager import launch_background_tmux, tmux_session_exists
from experiment.phase18.core.contracts import load_json, sha256
from experiment.phase18.protocol import s18_s1_runtime as base


EXPERIMENT_ID = "s18_s1_postrun_guard"
AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_recovery_authorization.json"
RECOVERY_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_recovery.status.json"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_postrun_guard.status.json"
RUNTIME_ROOT = ROOT / "artifacts/phase18/runtime/s18_s1_postrun_guard"
SOURCE_CHECKPOINT = ROOT / "artifacts/phase18/s1_actionability/run-0001/units/beauty_i0/parent_epoch10.pt"
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
POLL_SECONDS = 60
INTER_CYCLE_SECONDS = 10


def update_status(**fields: Any) -> None:
    current = load_json(STATUS) if STATUS.is_file() else {}
    current.update(fields)
    current["updated_at"] = base.utc_now()
    current["heartbeat_at"] = base.utc_now()
    base.atomic_json(STATUS, current)


def occupancy_config() -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTH_PATH)
    config = load_json(base.CONFIG_PATH)
    occupancy = authorization["postrun_occupancy"]
    if not occupancy["authorized"] or occupancy["physical_gpu"] != 4:
        raise RuntimeError("GPU4 post-run occupancy is not authorized")
    if not occupancy["fresh_cuda_process_per_cycle"]:
        raise RuntimeError("fresh CUDA process per cycle is mandatory")
    checkpoint_record = authorization["checkpoints"]["Beauty:I0"]["parent"]
    if ROOT / checkpoint_record["path"] != SOURCE_CHECKPOINT:
        raise RuntimeError("occupancy source checkpoint path mismatch")
    if not SOURCE_CHECKPOINT.is_file() or sha256(SOURCE_CHECKPOINT) != checkpoint_record["sha256"]:
        raise RuntimeError("occupancy source checkpoint hash mismatch")
    return config, authorization


def gpu4_snapshot() -> dict[str, int | str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    for line in completed.stdout.splitlines():
        index, uuid, used, free, utilization = [field.strip() for field in line.split(",")]
        if int(index) == 4:
            return {
                "index": 4,
                "uuid": uuid,
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_gpu_percent": int(utilization),
            }
    raise RuntimeError("physical GPU4 not found")


def cycle_dir(iteration: int) -> Path:
    if iteration < 1:
        raise ValueError("occupancy iteration must be positive")
    path = RUNTIME_ROOT / f"run-{iteration:04d}"
    if path.parent.resolve() != RUNTIME_ROOT.resolve():
        raise PermissionError("occupancy cycle escaped runtime root")
    return path


def cycle_command(iteration: int) -> list[str]:
    directory = cycle_dir(iteration)
    return [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "cycle-worker",
        "--iteration",
        str(iteration),
        "--cycle-dir",
        str(directory),
    ]


def cycle_worker(iteration: int, directory: Path) -> int:
    config, authorization = occupancy_config()
    expected = cycle_dir(iteration)
    if directory.resolve() != expected.resolve() or not directory.is_dir():
        raise RuntimeError("occupancy cycle directory mismatch")
    base.set_seed(config["seed"] + iteration)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"]["snapshot"], local_files_only=True)
    args = base.gram_args(config, "Beauty", "I0")
    args.tokenizer = tokenizer
    dataset = base.MultiTaskDatasetGRAM(
        args,
        base.dataset_name_from_manifest("Beauty", "I0"),
        "train",
        None,
        tokenizer,
        phase=0,
        regenerate=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["parent_training"]["rec_batch_size"],
        shuffle=False,
        collate_fn=base.CollatorGRAM(tokenizer=tokenizer, args=args, mode="train"),
        num_workers=0,
    )
    model = base.load_parent(config, SOURCE_CHECKPOINT, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    started = time.time()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            history_item_ids=batch["history_item_ids"].to(device),
            history_item_mask=batch["history_item_mask"].to(device),
            target_item_ids=batch["target_item_ids"].to(device),
            labels=batch["target_ids"].to(device),
            return_dict=False,
        )[0]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite ignored occupancy loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if batch_index % 50 == 0:
            update_status(
                execution_state="RUNNING_OCCUPANCY_REPEAT",
                scientific_state="COMPLETED",
                status_code="S18_1_SCIENTIFIC_COMPLETED_OCCUPANCY_RUNNING",
                stage="fresh_cuda_cycle",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[4],
                physical_gpu=4,
                repeat_iteration=iteration,
                repeat_result_dir=str(directory.relative_to(ROOT)),
                progress={"current": batch_index, "total": len(loader), "unit": "batch"},
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
    payload = {
        "schema_version": "phase18.s18_1_occupancy_cycle.v1",
        "iteration": iteration,
        "status": "COMPLETED_IGNORED",
        "batches": len(loader),
        "ignored_mean_loss": float(np.mean(losses)),
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": authorization["checkpoints"]["Beauty:I0"]["parent"]["sha256"],
        "result_selection_eligible": False,
        "repeat_metrics_ignored": True,
        "affects_scientific_result": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "completed_at": base.utc_now(),
    }
    base.atomic_json(directory / "cycle.json", payload)
    return 0


def worker() -> int:
    _, authorization = occupancy_config()
    recovery = load_json(RECOVERY_STATUS)
    if recovery.get("scientific_state") != "COMPLETED" or recovery.get("process_alive") is not False:
        raise RuntimeError("corrected S18-1 result is not terminal-complete")
    minimum_free = int(authorization["postrun_occupancy"]["minimum_free_mib"])
    iteration = 1
    while True:
        try:
            snapshot = gpu4_snapshot()
        except Exception as error:
            update_status(
                execution_state="WAITING_FOR_GPU",
                scientific_state="COMPLETED",
                status_code="S18_1_OCCUPANCY_RESOURCE_PROBE_WAITING",
                stage="resource_probe_transient_failure",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[],
                last_resource_probe_error=repr(error),
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(POLL_SECONDS)
            continue
        if snapshot["free_mib"] < minimum_free:
            update_status(
                execution_state="WAITING_FOR_GPU",
                scientific_state="COMPLETED",
                status_code="S18_1_OCCUPANCY_WAITING_FOR_GPU4_MEMORY",
                stage="waiting_for_gpu4_memory",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[],
                physical_gpu=4,
                minimum_free_mib=minimum_free,
                gpu_snapshot=snapshot,
                repeat_iteration=iteration,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(POLL_SECONDS)
            continue
        directory = cycle_dir(iteration)
        directory.mkdir(parents=True, exist_ok=False)
        log_path = directory / "cycle.log"
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="4",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=str(ROOT),
        )
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(cycle_command(iteration), cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            update_status(
                execution_state="RUNNING_OCCUPANCY_REPEAT",
                scientific_state="COMPLETED",
                status_code="S18_1_SCIENTIFIC_COMPLETED_OCCUPANCY_RUNNING",
                stage="fresh_cuda_cycle_starting",
                process_alive=True,
                workload_pid=process.pid,
                gpu_ids=[4],
                physical_gpu=4,
                minimum_free_mib=minimum_free,
                gpu_snapshot=snapshot,
                repeat_iteration=iteration,
                repeat_result_dir=str(directory.relative_to(ROOT)),
                progress={"current": 0, "total": 1, "unit": "cycle"},
                fresh_cuda_process_per_cycle=True,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            return_code = process.wait()
        if return_code != 0 or not (directory / "cycle.json").is_file():
            update_status(
                execution_state="FAILED_NO_RETRY",
                scientific_state="COMPLETED",
                status_code="S18_1_OCCUPANCY_CYCLE_FAILED_NO_RETRY",
                stage="terminal_occupancy_failure",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                failed_iteration=iteration,
                cycle_return_code=return_code,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            return 1
        update_status(
            execution_state="WAITING_FOR_GPU",
            scientific_state="COMPLETED",
            status_code="S18_1_OCCUPANCY_CYCLE_COMPLETE_CUDA_RELEASED",
            stage="cycle_child_exited_cuda_released",
            process_alive=True,
            workload_pid=os.getpid(),
            gpu_ids=[],
            completed_iteration=iteration,
            repeat_iteration=iteration + 1,
            progress={"current": iteration, "total": 0, "unit": "cycle"},
            result_selection_eligible=False,
            repeat_metrics_ignored=True,
            affects_scientific_result=False,
        )
        iteration += 1
        time.sleep(INTER_CYCLE_SECONDS)


def launch() -> int:
    _, authorization = occupancy_config()
    recovery = load_json(RECOVERY_STATUS)
    if recovery.get("scientific_state") != "COMPLETED":
        raise RuntimeError("cannot launch occupancy before corrected science completes")
    if RUNTIME_ROOT.exists() or STATUS.exists():
        raise FileExistsError("post-run occupancy guard already has artifacts")
    RUNTIME_ROOT.mkdir(parents=True)
    session = authorization["postrun_occupancy"]["tmux_session"]
    base.atomic_json(
        STATUS,
        {
            "schema_version": "phase18.status.v1",
            "experiment_id": EXPERIMENT_ID,
            "scientific_source": "s18_s1_actionability_recovery/run-0002",
            "scientific_state": "COMPLETED",
            "execution_state": "WAITING_FOR_GPU",
            "status_code": "S18_1_OCCUPANCY_STARTING",
            "stage": "background_starting",
            "process_alive": True,
            "launcher_pid": os.getpid(),
            "workload_pid": 0,
            "tmux_session": session,
            "physical_gpu": 4,
            "gpu_ids": [],
            "minimum_free_mib": authorization["postrun_occupancy"]["minimum_free_mib"],
            "fresh_cuda_process_per_cycle": True,
            "repeat_iteration": 1,
            "result_selection_eligible": False,
            "repeat_metrics_ignored": True,
            "affects_scientific_result": False,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
            "started_at": base.utc_now(),
            "updated_at": base.utc_now(),
            "heartbeat_at": base.utc_now(),
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
        "worker",
    ]
    launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=command,
        cwd=ROOT,
        tmux_session=session,
        startup_log_path=RUNTIME_ROOT / "guard.log",
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if tmux_session_exists(session):
            status = load_json(STATUS)
            if status.get("workload_pid", 0) > 0:
                print(json.dumps({"tmux_session": session, "status": str(STATUS.relative_to(ROOT))}))
                return 0
        time.sleep(1)
    raise RuntimeError("post-run occupancy guard failed startup handshake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "worker", "cycle-worker"))
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--cycle-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "launch":
        return launch()
    if args.action == "worker":
        return worker()
    if args.action == "cycle-worker":
        if args.iteration is None or args.cycle_dir is None:
            raise ValueError("cycle-worker requires iteration and cycle directory")
        return cycle_worker(args.iteration, args.cycle_dir)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
