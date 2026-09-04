#!/usr/bin/env python3
"""One-pass operational controller for the authorized long recovery run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_external_d0_g1_parallel_runtime as g1_parallel
from experiment.phase17.protocol import s17_fp12_external_d0_recovery_runtime as recovery
from experiment.phase17.protocol import s17_fp12_external_d0_runtime as external


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
POLL_SECONDS = 60
HARD_TIMEOUT_SECONDS = 48 * 60 * 60


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _status(root: Path, experiment_id: str) -> dict[str, Any] | None:
    path = external.paths(root)["status_dir"] / f"{experiment_id}.status.json"
    return _read(path) if path.is_file() else None


def _snapshot(root: Path) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for arm_id in ("N0_NATIVE_PSID", "N1_NATIVE_LATTE"):
        statuses[arm_id] = _status(root, external.arm_experiment_id(arm_id))
    for arm_id in ("G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL"):
        statuses[arm_id] = _status(root, recovery.arm_experiment_id(arm_id))
    statuses["G1_GRAM_PSID_FULL"] = _status(root, g1_parallel.EXPERIMENT_ID)
    return statuses


def _run(command: list[str], root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
        "completed_at": utc_now(),
    }


def control(root: Path) -> int:
    root = root.resolve()
    resolved = recovery.paths(root)
    state_path = resolved["result"] / "controller_state.json"
    log_path = resolved["result"] / "controller.log"
    if state_path.is_file() and _read(state_path).get("terminal") is True:
        raise RuntimeError("recovery controller already reached a terminal state")
    started = time.monotonic()
    state: dict[str, Any] = {
        "schema_version": "phase17.s17_fp12_recovery_controller.v1",
        "started_at": utc_now(),
        "terminal": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
        "raw_external_projection_reopened": False,
        "g1_schedule": "attempt_003_parallel_gpu4_launched_externally",
        "analysis_attempted": False,
        "report_attempted": False,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
    }
    atomic_json(state_path, state)
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            statuses = _snapshot(root)
            state.update(updated_at=utc_now(), statuses=statuses)
            failed = {
                arm: status
                for arm, status in statuses.items()
                if status is not None
                and status.get("scientific_state") in {"FAILED", "BLOCKED", "STOPPED"}
                and not (
                    arm in {"G0_GRAM_B0_FRESH", "G2_GRAM_LATTE_FULL"}
                    and status.get("attempt_id") == external.ATTEMPT_ID
                )
            }
            if failed:
                state.update(
                    terminal=True,
                    state="RECOVERY_ARM_TERMINAL_FAILURE_NO_AUTOMATIC_RETRY",
                    failed_arms=sorted(failed),
                )
                atomic_json(state_path, state)
                log.write(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n")
                log.flush()
                return 1

            completed = all(
                status is not None and status.get("scientific_state") == "COMPLETED"
                for status in statuses.values()
            )
            if completed:
                state["analysis_attempted"] = True
                atomic_json(state_path, state)
                analysis = _run(
                    [
                        str(PYTHON),
                        str(g1_parallel.paths(root)["snapshot_worker"]),
                        "analyze",
                        "--root",
                        str(root),
                    ],
                    root,
                )
                state["analysis"] = analysis
                log.write(json.dumps(analysis, ensure_ascii=False, sort_keys=True) + "\n")
                log.flush()
                if analysis["returncode"] != 0:
                    state.update(terminal=True, state="ANALYSIS_FAILED")
                    atomic_json(state_path, state)
                    return 3
                state["report_attempted"] = True
                report = _run(
                    [
                        str(PYTHON),
                        str(
                            root
                            / "experiment/phase17/protocol/s17_fp12_external_d0_report.py"
                        ),
                        "--root",
                        str(root),
                    ],
                    root,
                )
                state["report"] = report
                if report["returncode"] != 0:
                    state.update(terminal=True, state="REPORT_RENDER_FAILED")
                    atomic_json(state_path, state)
                    return 4
                report_path = root / external.REPORT_SUFFIX if hasattr(external, "REPORT_SUFFIX") else root / "report/第十七阶段/Stage17_FP12_ExternalD0评测准备报告.md"
                state.update(
                    terminal=True,
                    state="CONTROLLED_RECOVERY_ANALYSIS_AND_REPORT_COMPLETED",
                    analysis_path=str(resolved["analysis"].relative_to(root)),
                    analysis_sha256=sha256(resolved["analysis"]),
                    report_path=str(report_path.relative_to(root)),
                    report_sha256=sha256(report_path),
                    completed_at=utc_now(),
                )
                atomic_json(state_path, state)
                log.write(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n")
                log.flush()
                return 0

            if time.monotonic() - started >= HARD_TIMEOUT_SECONDS:
                state.update(
                    terminal=True,
                    state="HARD_TIMEOUT_NO_PROCESS_TERMINATION_RESEARCHER_REVIEW_REQUIRED",
                    timed_out_at=utc_now(),
                )
                atomic_json(state_path, state)
                return 5
            atomic_json(state_path, state)
            time.sleep(POLL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    return control(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
