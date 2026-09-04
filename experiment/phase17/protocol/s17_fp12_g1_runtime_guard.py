#!/usr/bin/env python3
"""Keep the G1 GPU occupied after its canonical external evaluation completes."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import (
    launch_background_tmux,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_external_d0_g1_parallel_runtime as g1
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as profile_base


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "s17_fp12_external_d0_g1_guard"
ATTEMPT_ID = "attempt_001"
RESULT_SUFFIX = Path("artifacts/phase17/runtime/s17_fp12_external_d0_g1_guard/attempt_001")
STATUS_DIR_SUFFIX = Path("artifacts/phase17/status")
PHYSICAL_GPU = g1.PHYSICAL_GPU
MINIMUM_FREE_MIB = g1.MINIMUM_FREE_MIB


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _profile_spec():
    return profile_base.ProfileSpec(
        g1.ARM_ID,
        "S17-FP12-EXTERNAL-D0-G1-GUARD",
        PHYSICAL_GPU,
        15892,
        MINIMUM_FREE_MIB,
        2,
        1,
        "gram",
    )


def _admitted() -> tuple[bool, dict[str, Any]]:
    first = profile_base.gpu_snapshot_once(_profile_spec())
    time.sleep(5)
    second = profile_base.gpu_snapshot_once(_profile_spec())
    admitted = all(
        row["selected"]["free_mib"] >= MINIMUM_FREE_MIB
        for row in (first, second)
    )
    return admitted, {
        "first": first,
        "second": second,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "utilization_recorded_only": True,
        "preexisting_processes_preserved": True,
    }


def worker(root: Path, manifest: Path) -> int:
    root = root.resolve()
    resolved = g1.paths(root)
    g1.verify(root, manifest)
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    iteration = 1
    while True:
        g1_status = _read(root / STATUS_DIR_SUFFIX / f"{g1.EXPERIMENT_ID}.status.json")
        if g1_status["scientific_state"] in {"FAILED", "BLOCKED", "STOPPED"}:
            writer.transition(
                "BLOCKED",
                "BLOCKED",
                "S17_FP12_G1_GUARD_BLOCKED_BY_G1_TERMINAL_STATE",
                process_alive=False,
                workload_pid=0,
                terminal_g1_state=g1_status["scientific_state"],
            )
            return 2
        if g1_status["scientific_state"] != "COMPLETED":
            writer.heartbeat(
                stage="waiting_for_g1_scientific_completion",
                progress=g1_status.get(
                    "progress", {"current": 0, "total": 12833, "unit": "external_user"}
                ),
            )
            time.sleep(30)
            continue
        if writer.read()["scientific_state"] == "PREFLIGHT":
            writer.transition(
                "RUNNING",
                "RUNNING_SCIENTIFIC",
                "S17_FP12_G1_GUARD_ACTIVATING",
                process_alive=True,
                stage="g1_complete_waiting_gpu_admission",
            )
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "PASS_S17_FP12_G1_GUARD_READY",
                process_alive=True,
                stage="g1_complete_waiting_gpu_admission",
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
        admitted, snapshots = _admitted()
        if not admitted:
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "S17_FP12_G1_GUARD_WAITING_FOR_MEMORY",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_snapshot=snapshots,
                stage="waiting_for_gpu_memory",
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(30)
            continue
        iteration += 1
        cycle_dir = root / RESULT_SUFFIX / f"run-{iteration:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=False)
        writer.start_runtime_cycle(
            iteration=iteration,
            runtime_result_dir=str(cycle_dir.relative_to(root)),
            workload_pid=os.getpid(),
        )
        writer.heartbeat(
            stage="isolated_g1_resource_cycle",
            progress={"current": iteration - 1, "total": 0, "unit": "cycle"},
        )
        try:
            from experiment.phase17.core.full_latte_profile_executor import (
                run_resource_profile,
            )

            measurements = run_resource_profile(
                root,
                g1.ARM_ID,
                train_batch_size=2,
                eval_batch_size=1,
                heartbeat=None,
            )
            atomic_json(
                cycle_dir / "cycle.json",
                {
                    "schema_version": "phase17.g1_runtime_cycle.v1",
                    "completed_at": utc_now(),
                    "iteration": iteration,
                    "measurements": measurements,
                    "result_selection_eligible": False,
                    "repeat_metrics_ignored": True,
                    "affects_scientific_result": False,
                    "external_target_materialized": False,
                    "raw_external_projection_reopened": False,
                    "test_read": False,
                    "sports_read": False,
                    "d1_read": False,
                    "d2_read": False,
                },
            )
        except BaseException as error:
            atomic_json(
                cycle_dir / "failure.json",
                {
                    "failed_at": utc_now(),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "automatic_retry": False,
                    "result_selection_eligible": False,
                    "affects_scientific_result": False,
                },
            )
            writer.transition(
                "COMPLETED",
                "SCIENTIFIC_COMPLETED",
                "S17_FP12_G1_GUARD_CYCLE_FAILED_NO_RETRY",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                terminal_error=repr(error),
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            return 1


def launch(root: Path, manifest: Path) -> int:
    root = root.resolve()
    resolved = g1.paths(root)
    g1.verify(root, manifest)
    status_path = root / STATUS_DIR_SUFFIX / f"{EXPERIMENT_ID}.status.json"
    result = root / RESULT_SUFFIX
    if status_path.exists() or result.exists():
        raise FileExistsError("G1 post-completion guard already exists")
    result.mkdir(parents=True, exist_ok=False)
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-FP12-EXTERNAL-D0-G1-GUARD",
        attempt_id=ATTEMPT_ID,
        track_id="G1_RUNTIME_GUARD",
        canonical_result_dir=str(RESULT_SUFFIX),
        log_path=str(RESULT_SUFFIX / "guard.log"),
        extra={
            "stage": "waiting_for_g1_scientific_completion",
            "progress": {"current": 0, "total": 12833, "unit": "external_user"},
            "target_gpu_id": PHYSICAL_GPU,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "process_alive": True,
            "result_selection_eligible": False,
            "repeat_metrics_ignored": True,
            "affects_scientific_result": False,
            "external_target_materialized": False,
            "raw_external_projection_reopened": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "WAITING_FOR_GPU",
        "S17_FP12_G1_GUARD_WAITING_FOR_G1_COMPLETION",
        process_alive=True,
    )
    command = [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(profile_base.GRAM_PYTHON),
        str(resolved["snapshot_guard"]),
        "worker",
        "--root",
        str(root),
        "--manifest",
        str(manifest),
    ]
    launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=command,
        cwd=root,
        tmux_session=EXPERIMENT_ID,
        startup_log_path=result / "guard.log",
    )
    writer.transition(
        "PREFLIGHT",
        "WAITING_FOR_GPU",
        "S17_FP12_G1_GUARD_WAITING_FOR_G1_COMPLETION",
        tmux_session=EXPERIMENT_ID,
        launcher_pid=os.getpid(),
        process_alive=True,
    )
    if not wait_for_tmux_startup(EXPERIMENT_ID):
        writer.transition(
            "BLOCKED",
            "BLOCKED",
            "S17_FP12_G1_GUARD_STARTUP_FAILED",
            process_alive=False,
        )
        raise RuntimeError("G1 guard exited during startup handshake")
    print(EXPERIMENT_ID)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "worker"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "launch":
        return launch(args.root, args.manifest)
    return worker(args.root, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
