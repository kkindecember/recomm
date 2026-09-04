#!/usr/bin/env python3
"""Resilient non-scientific G1 occupancy guard with fail-soft GPU polling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import (
    freeze_run_snapshot,
    launch_background_tmux,
    sha256,
    verify_run_snapshot,
    wait_for_tmux_startup,
)
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_external_d0_g1_parallel_runtime as g1
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as profile_base


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "s17_fp12_external_d0_g1_guard_v2"
ATTEMPT_ID = "attempt_001"
RESULT_SUFFIX = Path("artifacts/phase17/runtime/s17_fp12_external_d0_g1_guard_v2/attempt_001")
STATUS_DIR_SUFFIX = Path("artifacts/phase17/status")
SNAPSHOT_SUFFIX = Path(
    "artifacts/phase17/snapshots/s17_fp12_external_d0_g1_guard_v2/attempt_001/manifest.json"
)
PHYSICAL_GPU = g1.PHYSICAL_GPU
MINIMUM_FREE_MIB = g1.MINIMUM_FREE_MIB
RESOURCE_PROBE_ERRORS = (
    subprocess.SubprocessError,
    OSError,
    RuntimeError,
    ValueError,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _profile_spec():
    return profile_base.ProfileSpec(
        g1.ARM_ID,
        "S17-FP12-EXTERNAL-D0-G1-GUARD-V2",
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


def _probe_admission_fail_soft() -> tuple[
    bool | None, dict[str, Any] | None, str | None
]:
    """Treat resource-observation failures as wait signals, never cycle retries."""

    try:
        admitted, snapshots = _admitted()
    except RESOURCE_PROBE_ERRORS as error:
        return None, None, repr(error)
    return admitted, snapshots, None


def _verify_self(root: Path, manifest: Path) -> None:
    verify_run_snapshot(root, manifest)
    record = _read(manifest)["files"][0]
    if sha256(root / record["source_path"]) != record["sha256"]:
        raise RuntimeError("G1 guard v2 live source drifted")


def prepare(root: Path) -> int:
    root = root.resolve()
    result = root / RESULT_SUFFIX
    status_path = root / STATUS_DIR_SUFFIX / f"{EXPERIMENT_ID}.status.json"
    manifest = root / SNAPSHOT_SUFFIX
    if result.exists() or status_path.exists() or manifest.exists():
        raise FileExistsError("G1 guard v2 already exists")
    g1.verify(root)
    config = {
        "schema_version": "phase17.s17_fp12_g1_guard_v2_config.v1",
        "attempt_id": ATTEMPT_ID,
        "arm_id": g1.ARM_ID,
        "physical_gpu": PHYSICAL_GPU,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "g1_authorization_path": str(g1.paths(root)["authorization"].relative_to(root)),
        "g1_authorization_sha256": sha256(g1.paths(root)["authorization"]),
        "g1_snapshot_manifest": str(g1.paths(root)["snapshot"].relative_to(root)),
        "resource_probe_timeout_is_fail_soft": True,
        "scientific_cycle_retry": False,
        "result_selection_eligible": False,
        "repeat_metrics_ignored": True,
        "affects_scientific_result": False,
        "raw_external_projection_reopened": False,
    }
    result.mkdir(parents=True, exist_ok=False)
    atomic_json(result / "config.json", config)
    frozen = freeze_run_snapshot(
        root=root,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=["wait_for_g1_then_hold_gpu4_with_non_scientific_cycles"],
        source_paths=[Path(__file__).resolve()],
        config=config,
    )
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-FP12-EXTERNAL-D0-G1-GUARD",
        attempt_id=ATTEMPT_ID,
        track_id="G1_RUNTIME_GUARD_V2",
        canonical_result_dir=str(RESULT_SUFFIX),
        log_path=str(RESULT_SUFFIX / "guard.log"),
        extra={
            "stage": "waiting_for_g1_scientific_completion",
            "run_snapshot_manifest": str(frozen.relative_to(root)),
            "target_gpu_id": PHYSICAL_GPU,
            "minimum_free_mib": MINIMUM_FREE_MIB,
            "process_alive": False,
            "result_selection_eligible": False,
            "repeat_metrics_ignored": True,
            "affects_scientific_result": False,
            "external_target_materialized": False,
            "raw_external_projection_reopened": False,
            "resource_probe_timeout_is_fail_soft": True,
            "automatic_retry": False,
            "automatic_process_termination": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_G1_GUARD_V2_PREFLIGHT_COMPLETE",
        process_alive=False,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def worker(root: Path, manifest: Path) -> int:
    root = root.resolve()
    _verify_self(root, manifest)
    g1.verify(root, g1.paths(root)["snapshot"])
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    iteration = 1
    resource_probe_failures = 0
    while True:
        g1_status = _read(root / STATUS_DIR_SUFFIX / f"{g1.EXPERIMENT_ID}.status.json")
        if g1_status["scientific_state"] in {"FAILED", "BLOCKED", "STOPPED"}:
            writer.transition(
                "BLOCKED",
                "BLOCKED",
                "S17_FP12_G1_GUARD_V2_BLOCKED_BY_G1_TERMINAL_STATE",
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
                "S17_FP12_G1_GUARD_V2_ACTIVATING",
                process_alive=True,
            )
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "PASS_S17_FP12_G1_GUARD_V2_READY",
                process_alive=True,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
        admitted, snapshots, probe_error = _probe_admission_fail_soft()
        if admitted is None:
            resource_probe_failures += 1
            writer.heartbeat(
                stage="resource_probe_transient_failure_waiting",
                progress={"current": iteration - 1, "total": 0, "unit": "cycle"},
            )
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "S17_FP12_G1_GUARD_V2_RESOURCE_PROBE_WAITING",
                process_alive=True,
                workload_pid=os.getpid(),
                resource_probe_failures=resource_probe_failures,
                last_resource_probe_error=probe_error,
                result_selection_eligible=False,
                repeat_metrics_ignored=True,
                affects_scientific_result=False,
            )
            time.sleep(30)
            continue
        assert snapshots is not None
        if not admitted:
            writer.transition(
                "COMPLETED",
                "WAITING_FOR_GPU",
                "S17_FP12_G1_GUARD_V2_WAITING_FOR_MEMORY",
                process_alive=True,
                workload_pid=os.getpid(),
                gpu_snapshot=snapshots,
                resource_probe_failures=resource_probe_failures,
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
                    "schema_version": "phase17.g1_runtime_cycle.v2",
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
                "S17_FP12_G1_GUARD_V2_CYCLE_FAILED_NO_RETRY",
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
    manifest = root / SNAPSHOT_SUFFIX
    _verify_self(root, manifest)
    g1.verify(root, g1.paths(root)["snapshot"])
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, EXPERIMENT_ID)
    if writer.read()["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("G1 guard v2 is not in PREFLIGHT")
    snapshot_worker = manifest.parent / f"src/000_{Path(__file__).name}"
    command = [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={root}",
        str(profile_base.GRAM_PYTHON),
        str(snapshot_worker),
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
        startup_log_path=root / RESULT_SUFFIX / "guard.log",
    )
    writer.transition(
        "PREFLIGHT",
        "WAITING_FOR_GPU",
        "S17_FP12_G1_GUARD_V2_WAITING_FOR_G1_COMPLETION",
        tmux_session=EXPERIMENT_ID,
        launcher_pid=os.getpid(),
        process_alive=True,
    )
    if not wait_for_tmux_startup(EXPERIMENT_ID):
        writer.transition(
            "BLOCKED",
            "BLOCKED",
            "S17_FP12_G1_GUARD_V2_STARTUP_FAILED",
            process_alive=False,
        )
        raise RuntimeError("G1 guard v2 exited during startup handshake")
    print(EXPERIMENT_ID)
    return 0


def supersede_old(root: Path) -> int:
    root = root.resolve()
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, "s17_fp12_external_d0_g1_guard")
    current = writer.read()
    if current["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("old G1 guard is not supersedable PREFLIGHT")
    writer.transition(
        "STOPPED",
        "STOPPED",
        "S17_FP12_G1_GUARD_SUPERSEDED_BY_FAIL_SOFT_V2",
        process_alive=False,
        workload_pid=0,
        tmux_session=None,
        superseded_by_experiment_id=EXPERIMENT_ID,
        result_selection_eligible=False,
        repeat_metrics_ignored=True,
        affects_scientific_result=False,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "worker", "supersede-old"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.root)
    if args.action == "launch":
        return launch(args.root)
    if args.action == "supersede-old":
        return supersede_old(args.root)
    if args.manifest is None:
        parser.error("worker requires --manifest")
    return worker(args.root, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
