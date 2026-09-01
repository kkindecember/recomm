#!/usr/bin/env python3
"""Wait for formal G2, then keep GPU1 occupied with isolated resource cycles."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.run_manager import launch_background_tmux, sha256, wait_for_tmux_startup
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as profile_base


EXPERIMENT_ID = "s17_fp12_gpu1_runtime_guard"
ATTEMPT_ID = "attempt_004"
FORMAL_STATUS = Path(
    "artifacts/phase17/status/"
    "s17_fp12_formal_g2_gram_latte_full_seed2023.status.json"
)
RESULT_ROOT = Path("artifacts/phase17/fullport/runtime_guard/gpu1/attempt_004")
STATUS_DIR = Path("artifacts/phase17/status")
MINIMUM_FREE_MIB = 19992
PHYSICAL_GPU = 1


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_spec():
    return profile_base.ProfileSpec(
        "G2_GRAM_LATTE_FULL",
        "S17-FP2",
        PHYSICAL_GPU,
        15896,
        MINIMUM_FREE_MIB,
        2,
        1,
        "gram",
    )


def _snapshot() -> dict[str, Any]:
    return profile_base.gpu_snapshot_once(_profile_spec())


def _admitted() -> tuple[bool, dict[str, Any]]:
    first = _snapshot()
    time.sleep(5)
    second = _snapshot()
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


def worker(root: Path) -> int:
    root = root.resolve()
    writer = StatusWriter(root / STATUS_DIR, EXPERIMENT_ID)
    iteration = 0
    while True:
        formal = _read(root / FORMAL_STATUS)
        if formal["scientific_state"] not in {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}:
            writer.heartbeat(
                stage="waiting_for_formal_g2_terminal_state",
                progress=formal.get("progress", {"current": 0, "total": 50, "unit": "epoch"}),
            )
            time.sleep(30)
            continue
        current = writer.read()
        if current["scientific_state"] == "PREFLIGHT":
            writer.transition(
                "RUNNING",
                "RUNNING_SCIENTIFIC",
                "S17_FP12_GPU1_RUNTIME_GUARD_ACTIVATING",
                stage="formal_g2_terminal_preparing_isolated_cycle",
                formal_g2_terminal_state=formal["scientific_state"],
                process_alive=True,
            )
            writer.transition(
                "COMPLETED",
                "SCIENTIFIC_COMPLETED",
                "PASS_S17_FP12_GPU1_RUNTIME_GUARD_READY",
                stage="runtime_guard_ready",
                process_alive=True,
                result_selection_eligible=False,
                affects_scientific_result=False,
            )
        admitted, snapshots = _admitted()
        if not admitted:
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "S17_FP12_GPU1_RUNTIME_GUARD_WAITING_FOR_MEMORY",
                stage="waiting_for_gpu1_memory",
                gpu_snapshot=snapshots,
                process_alive=True,
                workload_pid=os.getpid(),
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(30)
            continue
        iteration += 1
        cycle_dir = root / RESULT_ROOT / f"run-{iteration:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=False)
        writer.start_runtime_cycle(
            iteration=iteration,
            runtime_result_dir=str(cycle_dir.relative_to(root)),
            workload_pid=os.getpid(),
        )
        writer.heartbeat(
            stage="isolated_gpu1_resource_cycle",
            progress={"current": iteration, "total": 0, "unit": "cycle"},
        )
        try:
            from experiment.phase17.core.full_latte_profile_executor import run_resource_profile

            measurements = run_resource_profile(
                root,
                "G2_GRAM_LATTE_FULL",
                train_batch_size=2,
                eval_batch_size=1,
                heartbeat=None,
            )
            atomic_json(
                cycle_dir / "cycle.json",
                {
                    "schema_version": "phase17.gpu1_runtime_cycle.v1",
                    "completed_at": utc_now(),
                    "iteration": iteration,
                    "measurements": measurements,
                    "result_selection_eligible": False,
                    "repeat_metrics_ignored": True,
                    "affects_scientific_result": False,
                    "external_target_materialized": False,
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
                "S17_FP12_GPU1_RUNTIME_CYCLE_FAILED_NO_RETRY",
                stage="isolated_cycle_failure_no_retry",
                process_alive=False,
                workload_pid=0,
                gpu_ids=[],
                terminal_error=repr(error),
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            return 1


def launch(root: Path) -> int:
    root = root.resolve()
    status_path = root / STATUS_DIR / f"{EXPERIMENT_ID}.status.json"
    if status_path.exists():
        previous = _read(status_path)
        if previous.get("scientific_state") not in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError("previous GPU1 runtime guard status is not terminal")
        if previous.get("attempt_id") == ATTEMPT_ID:
            raise FileExistsError(f"GPU1 runtime guard already exists: {status_path}")
    result = root / RESULT_ROOT
    result.mkdir(parents=True, exist_ok=False)
    writer = StatusWriter(root / STATUS_DIR, EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-FP2",
        attempt_id=ATTEMPT_ID,
        track_id="GPU1_RUNTIME_GUARD",
        canonical_result_dir=str(RESULT_ROOT),
        log_path=str(RESULT_ROOT / "guard.log"),
        extra={
            "stage": "waiting_for_formal_g2_terminal_state",
            "progress": {"current": 0, "total": 50, "unit": "epoch"},
            "gpu_ids": [],
            "target_gpu_id": PHYSICAL_GPU,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "formal_status_path": str(FORMAL_STATUS),
            "formal_status_sha256_at_guard_launch": sha256(root / FORMAL_STATUS),
            "process_alive": True,
            "result_selection_eligible": False,
            "repeat_metrics_ignored": True,
            "affects_scientific_result": False,
            "external_target_materialized": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "WAITING_FOR_GPU",
        "S17_FP12_GPU1_RUNTIME_GUARD_WAITING_FOR_FORMAL_G2",
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
        str(Path(__file__).resolve()),
        "worker",
        "--root",
        str(root),
    ]
    session = launch_background_tmux(
        experiment_id=EXPERIMENT_ID,
        argv=command,
        cwd=root,
        tmux_session=EXPERIMENT_ID,
        startup_log_path=root / RESULT_ROOT / "guard.log",
    )
    writer.transition(
        "PREFLIGHT",
        "WAITING_FOR_GPU",
        "S17_FP12_GPU1_RUNTIME_GUARD_WAITING_FOR_FORMAL_G2",
        tmux_session=session,
        launcher_pid=os.getpid(),
        process_alive=True,
    )
    if not wait_for_tmux_startup(session):
        raise RuntimeError("GPU1 runtime guard exited during startup")
    print(session)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "worker"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "launch":
        return launch(args.root)
    return worker(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
