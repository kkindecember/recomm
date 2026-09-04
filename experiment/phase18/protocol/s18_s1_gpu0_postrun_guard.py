#!/usr/bin/env python3
"""Scientifically isolated GPU0 repeat guard with normal-work preemption."""

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


PROFILE = os.environ.get("S18_OCCUPANCY_PROFILE", "run-0004")
if PROFILE == "run-0005":
    EXPERIMENT_ID = "s18_s1_gpu0_postrun_guard_early"
    AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_parallel_takeover_mp2_authorization.json"
    SCIENCE_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_parallel_takeover_mp2.status.json"
    STATUS = ROOT / "artifacts/phase18/status/s18_s1_gpu0_postrun_guard_early.status.json"
    RUNTIME_ROOT = ROOT / "artifacts/phase18/runtime/s18_s1_gpu0_postrun_guard_early"
elif PROFILE == "run-0004":
    EXPERIMENT_ID = "s18_s1_gpu0_postrun_guard"
    AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_parallel_takeover_authorization.json"
    SCIENCE_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability_parallel_takeover.status.json"
    STATUS = ROOT / "artifacts/phase18/status/s18_s1_gpu0_postrun_guard.status.json"
    RUNTIME_ROOT = ROOT / "artifacts/phase18/runtime/s18_s1_gpu0_postrun_guard"
else:
    raise RuntimeError(f"unknown S18 occupancy profile: {PROFILE}")
SOURCE_CHECKPOINT = ROOT / "artifacts/phase18/s1_actionability/run-0001/units/beauty_i0/parent_epoch10.pt"
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
PREEMPT_EXIT_CODE = 75


def update_status(**fields: Any) -> None:
    current = load_json(STATUS) if STATUS.is_file() else {}
    current.update(fields)
    current["updated_at"] = base.utc_now()
    current["heartbeat_at"] = base.utc_now()
    base.atomic_json(STATUS, current)


def occupancy_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTH_PATH)
    config = load_json(base.CONFIG_PATH)
    occupancy = authorization["postrun_occupancy"]
    if not occupancy["authorized"] or occupancy["physical_gpu"] != 0:
        raise RuntimeError("GPU0 post-run occupancy is not authorized")
    if (
        not occupancy["fresh_cuda_process_per_cycle"]
        or not occupancy["normal_priority_preemption"]
        or occupancy["result_selection_eligible"]
        or not occupancy["repeat_metrics_ignored"]
        or occupancy["affects_scientific_result"]
    ):
        raise RuntimeError("post-run occupancy isolation contract mismatch")
    checkpoint_authorization = load_json(
        ROOT / authorization["frozen_inputs"]["checkpoint_authorization"]["path"]
    )
    checkpoint_record = checkpoint_authorization["checkpoints"]["Beauty:I0"]["parent"]
    if ROOT / checkpoint_record["path"] != SOURCE_CHECKPOINT:
        raise RuntimeError("occupancy checkpoint path mismatch")
    if not SOURCE_CHECKPOINT.is_file() or sha256(SOURCE_CHECKPOINT) != checkpoint_record["sha256"]:
        raise RuntimeError("occupancy checkpoint hash mismatch")
    return config, authorization, checkpoint_authorization


def science_launch_gate_satisfied(science: dict[str, Any], authorization: dict[str, Any]) -> bool:
    occupancy = authorization["postrun_occupancy"]
    if occupancy.get("launch_after_beauty_lane"):
        return (
            science.get("beauty_lane_state") == "COMPLETED"
            and science.get("beauty_lane_process_alive") is False
        )
    return science.get("scientific_state") == "COMPLETED" and science.get("process_alive") is False


def preempt_path(authorization: dict[str, Any]) -> Path:
    return ROOT / authorization["postrun_occupancy"]["normal_priority_request_path"]


def gpu0_snapshot() -> dict[str, Any]:
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
        if int(index) == 0:
            return {
                "index": 0,
                "uuid": uuid,
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_gpu_percent": int(utilization),
            }
    raise RuntimeError("physical GPU0 not found")


def current_user_gpu0_pids(exclude_pid: int | None = None) -> list[int]:
    snapshot = gpu0_snapshot()
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        uuid, raw_pid = [field.strip() for field in line.split(",")]
        if uuid != snapshot["uuid"]:
            continue
        pid = int(raw_pid)
        if pid == exclude_pid:
            continue
        try:
            if Path(f"/proc/{pid}").stat().st_uid == os.getuid():
                pids.append(pid)
        except FileNotFoundError:
            continue
    return sorted(pids)


def active_normal_science_statuses() -> list[str]:
    active: list[str] = []
    for path in sorted((ROOT / "artifacts/phase18/status").glob("*.status.json")):
        if path in {STATUS, SCIENCE_STATUS}:
            continue
        try:
            payload = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            payload.get("affects_scientific_result") is True
            and payload.get("process_alive") is True
            and payload.get("scientific_state") in {"RUNNING", "NOT_STARTED"}
        ):
            active.append(str(path.relative_to(ROOT)))
    return active


def normal_priority_reason(authorization: dict[str, Any], exclude_pid: int | None = None) -> dict[str, Any] | None:
    request = preempt_path(authorization)
    if request.is_file():
        return {"kind": "explicit_preempt_request", "path": str(request.relative_to(ROOT))}
    active_statuses = active_normal_science_statuses()
    if active_statuses:
        return {"kind": "active_normal_science", "statuses": active_statuses}
    gpu_pids = current_user_gpu0_pids(exclude_pid=exclude_pid)
    if gpu_pids:
        return {"kind": "current_user_gpu0_process", "pids": gpu_pids}
    return None


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
        "nice",
        "-n",
        "19",
        str(PYTHON),
        str(Path(__file__).resolve()),
        "cycle-worker",
        "--iteration",
        str(iteration),
        "--cycle-dir",
        str(directory),
    ]


def cycle_worker(iteration: int, directory: Path) -> int:
    config, authorization, checkpoint_authorization = occupancy_config()
    expected = cycle_dir(iteration)
    if directory.resolve() != expected.resolve() or not directory.is_dir():
        raise RuntimeError("occupancy cycle directory mismatch")
    reason = normal_priority_reason(authorization, exclude_pid=os.getpid())
    if reason is not None:
        base.atomic_json(
            directory / "cycle.json",
            {
                "schema_version": "phase18.s18_1_gpu0_occupancy_cycle.v1",
                "iteration": iteration,
                "status": "PREEMPTED_NORMAL_PRIORITY",
                "reason": reason,
                "result_selection_eligible": False,
                "repeat_metrics_ignored": True,
                "affects_scientific_result": False,
                "completed_at": base.utc_now(),
            },
        )
        return PREEMPT_EXIT_CODE
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
    max_batches = int(authorization["postrun_occupancy"]["max_batches_per_cycle"])
    preempted_reason: dict[str, Any] | None = None
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > max_batches:
            break
        preempted_reason = normal_priority_reason(authorization, exclude_pid=os.getpid())
        if preempted_reason is not None:
            break
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
        if batch_index % 25 == 0:
            update_status(
                execution_state="RUNNING_OCCUPANCY_REPEAT",
                scientific_state="COMPLETED",
                status_code="S18_1_SCIENTIFIC_COMPLETED_GPU0_OCCUPANCY_RUNNING",
                stage="fresh_cuda_cycle",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[0],
                physical_gpu=0,
                repeat_iteration=iteration,
                repeat_result_dir=str(directory.relative_to(ROOT)),
                progress={"current": batch_index, "total": min(len(loader), max_batches), "unit": "batch"},
                normal_priority_preemption=True,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
    status = "PREEMPTED_NORMAL_PRIORITY" if preempted_reason is not None else "COMPLETED_IGNORED"
    payload = {
        "schema_version": "phase18.s18_1_gpu0_occupancy_cycle.v1",
        "iteration": iteration,
        "status": status,
        "preemption_reason": preempted_reason,
        "batches": len(losses),
        "ignored_mean_loss": float(np.mean(losses)) if losses else None,
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": checkpoint_authorization["checkpoints"]["Beauty:I0"]["parent"]["sha256"],
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
    return PREEMPT_EXIT_CODE if preempted_reason is not None else 0


def mark_preempted(reason: dict[str, Any]) -> None:
    update_status(
        execution_state="PREEMPTED_FOR_NORMAL_EXPERIMENT",
        scientific_state="COMPLETED",
        status_code="S18_1_GPU0_OCCUPANCY_PREEMPTED_NORMAL_PRIORITY",
        stage="terminal_normal_priority_handoff",
        process_alive=False,
        workload_pid=0,
        gpu_ids=[],
        preemption_reason=reason,
        result_selection_eligible=False,
        repeat_metrics_ignored=True,
        affects_scientific_result=False,
    )


def worker() -> int:
    _, authorization, _ = occupancy_config()
    science = load_json(SCIENCE_STATUS)
    if not science_launch_gate_satisfied(science, authorization):
        raise RuntimeError("GPU0 repeat occupancy launch gate is not satisfied")
    occupancy = authorization["postrun_occupancy"]
    minimum_free = int(occupancy["minimum_free_mib"])
    iteration = 1
    while True:
        try:
            reason = normal_priority_reason(authorization)
            if reason is not None:
                mark_preempted(reason)
                return 0
            snapshot = gpu0_snapshot()
        except Exception as error:
            update_status(
                execution_state="WAITING_FOR_GPU",
                scientific_state="COMPLETED",
                status_code="S18_1_GPU0_OCCUPANCY_RESOURCE_PROBE_WAITING",
                stage="resource_probe_transient_failure",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[],
                last_resource_probe_error=repr(error),
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(15)
            continue
        if snapshot["free_mib"] < minimum_free:
            update_status(
                execution_state="WAITING_FOR_GPU",
                scientific_state="COMPLETED",
                status_code="S18_1_GPU0_OCCUPANCY_WAITING_FOR_MEMORY",
                stage="waiting_for_gpu0_memory",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_ids=[],
                physical_gpu=0,
                minimum_free_mib=minimum_free,
                gpu_snapshot=snapshot,
                repeat_iteration=iteration,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(15)
            continue
        directory = cycle_dir(iteration)
        directory.mkdir(parents=True, exist_ok=False)
        log_path = directory / "cycle.log"
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=str(ROOT),
        )
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                cycle_command(iteration),
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            update_status(
                execution_state="RUNNING_OCCUPANCY_REPEAT",
                scientific_state="COMPLETED",
                status_code="S18_1_SCIENTIFIC_COMPLETED_GPU0_OCCUPANCY_RUNNING",
                stage="fresh_cuda_cycle_starting",
                process_alive=True,
                workload_pid=process.pid,
                gpu_ids=[0],
                physical_gpu=0,
                minimum_free_mib=minimum_free,
                gpu_snapshot=snapshot,
                repeat_iteration=iteration,
                repeat_result_dir=str(directory.relative_to(ROOT)),
                progress={"current": 0, "total": occupancy["max_batches_per_cycle"], "unit": "batch"},
                fresh_cuda_process_per_cycle=True,
                normal_priority_preemption=True,
                normal_priority_request_path=occupancy["normal_priority_request_path"],
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            return_code = process.wait()
        cycle = load_json(directory / "cycle.json") if (directory / "cycle.json").is_file() else {}
        if return_code == PREEMPT_EXIT_CODE and cycle.get("status") == "PREEMPTED_NORMAL_PRIORITY":
            mark_preempted(cycle.get("preemption_reason") or cycle.get("reason") or {"kind": "cycle_preempted"})
            return 0
        if return_code != 0 or cycle.get("status") != "COMPLETED_IGNORED":
            update_status(
                execution_state="FAILED_NO_RETRY",
                scientific_state="COMPLETED",
                status_code="S18_1_GPU0_OCCUPANCY_CYCLE_FAILED_NO_RETRY",
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
            status_code="S18_1_GPU0_OCCUPANCY_CYCLE_COMPLETE_CUDA_RELEASED",
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
        time.sleep(10)


def launch() -> int:
    _, authorization, _ = occupancy_config()
    science = load_json(SCIENCE_STATUS)
    if not science_launch_gate_satisfied(science, authorization):
        raise RuntimeError("cannot launch GPU0 occupancy before its authorized science gate")
    if RUNTIME_ROOT.exists() or STATUS.exists():
        raise FileExistsError("GPU0 post-run occupancy guard already has artifacts")
    RUNTIME_ROOT.mkdir(parents=True)
    occupancy = authorization["postrun_occupancy"]
    base.atomic_json(
        STATUS,
        {
            "schema_version": "phase18.status.v1",
            "experiment_id": EXPERIMENT_ID,
            "scientific_source": f"{authorization['experiment_id']}/{authorization['attempt_id']}",
            "scientific_state": (
                "BEAUTY_LANE_COMPLETED_TOYS_RUNNING"
                if authorization["postrun_occupancy"].get("launch_after_beauty_lane")
                else "COMPLETED"
            ),
            "execution_state": "WAITING_FOR_GPU",
            "status_code": "S18_1_GPU0_OCCUPANCY_STARTING",
            "stage": "background_starting",
            "process_alive": True,
            "launcher_pid": os.getpid(),
            "workload_pid": 0,
            "tmux_session": occupancy["tmux_session"],
            "physical_gpu": 0,
            "gpu_ids": [],
            "minimum_free_mib": occupancy["minimum_free_mib"],
            "fresh_cuda_process_per_cycle": True,
            "normal_priority_preemption": True,
            "normal_priority_request_path": occupancy["normal_priority_request_path"],
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
        tmux_session=occupancy["tmux_session"],
        startup_log_path=RUNTIME_ROOT / "guard.log",
    )
    deadline = time.time() + 15
    while time.time() < deadline:
        if tmux_session_exists(occupancy["tmux_session"]):
            status = load_json(STATUS)
            if status.get("workload_pid", 0) > 0:
                print(json.dumps({"tmux_session": occupancy["tmux_session"], "status": str(STATUS.relative_to(ROOT))}))
                return 0
        time.sleep(1)
    raise RuntimeError("GPU0 post-run occupancy guard failed startup handshake")


def request_preempt() -> int:
    _, authorization, _ = occupancy_config()
    request = preempt_path(authorization)
    request.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_text(request, f"normal-priority GPU0 requested at {base.utc_now()}\n")
    if STATUS.is_file():
        update_status(preemption_requested=True, preemption_request_path=str(request.relative_to(ROOT)))
    print(json.dumps({"status": "PREEMPT_REQUESTED", "path": str(request.relative_to(ROOT))}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "worker", "cycle-worker", "preempt"))
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
    if args.action == "preempt":
        return request_preempt()
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
