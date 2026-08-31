#!/usr/bin/env python3
"""Close the three stale FP0 attempt_001 statuses after verified bootstrap failure."""

from __future__ import annotations

import json
from pathlib import Path

from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import AttemptLedger, StatusWriter, atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
STATUS_DIR = ROOT / "artifacts/phase17/status"
ROOT_CAUSE = "ModuleNotFoundError: No module named 'experiment'"
TASKS = (
    {
        "experiment_id": "s17_fp0_native_env_setup",
        "result_dir": ROOT / "artifacts/phase17/fullport/fp0/native_env_setup/attempt_001",
        "ledger": ROOT / "artifacts/phase17/attempts/S17-FP0-NATIVE-ENV.attempts.jsonl",
        "status_code": "FAILED_S17_FP0_NATIVE_ENV_ATTEMPT_001_BOOTSTRAP_IMPORT",
    },
    {
        "experiment_id": "s17_fp0_sentence_t5_cache",
        "result_dir": ROOT / "artifacts/phase17/fullport/fp0/sentence_t5_cache/attempt_001",
        "ledger": ROOT / "artifacts/phase17/attempts/S17-FP0-SENTENCE-T5-CACHE.attempts.jsonl",
        "status_code": "FAILED_S17_FP0_SENTENCE_T5_ATTEMPT_001_BOOTSTRAP_IMPORT",
    },
    {
        "experiment_id": "s17_fp0_tokenizer_bounded_profile",
        "result_dir": ROOT / "artifacts/phase17/fullport/fp0/tokenizer_profile/attempt_001",
        "ledger": ROOT / "artifacts/phase17/attempts/S17-FP0-TOKENIZER-PROFILE.attempts.jsonl",
        "status_code": "FAILED_S17_FP0_TOKENIZER_PROFILE_ATTEMPT_001_BOOTSTRAP_IMPORT",
    },
)


def main() -> int:
    closed = []
    for task in TASKS:
        writer = StatusWriter(STATUS_DIR, task["experiment_id"])
        status = writer.read()
        if status["attempt_id"] != "attempt_001":
            raise RuntimeError(f"unexpected attempt for {task['experiment_id']}")
        if status["scientific_state"] != "RUNNING":
            raise RuntimeError(f"attempt_001 is not stale RUNNING: {task['experiment_id']}")
        failure_path = task["result_dir"] / "failure.json"
        if failure_path.exists():
            raise FileExistsError(failure_path)
        payload = {
            "schema_version": "phase17.s17_fp0_bootstrap_failure.v1",
            "experiment_id": task["experiment_id"],
            "attempt_id": "attempt_001",
            "closed_at": utc_now(),
            "failure_class": "ENGINEERING_BOOTSTRAP_FAILURE",
            "root_cause": ROOT_CAUSE,
            "root_cause_certainty": "CONFIRMED",
            "diagnosis": (
                "The immutable snapshot was executed by absolute file path without the "
                "repository root on PYTHONPATH. Import failed before the worker try/except, "
                "heartbeat, and run-log initialization."
            ),
            "evidence": {
                "tmux_session_present_at_diagnosis": False,
                "run_log_present_at_diagnosis": False,
                "workload_pid_recorded": 0,
                "manual_read_only_reproduction": (
                    "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python "
                    "artifacts/phase17/snapshots/s17_fp0_native_env_setup/attempt_001/"
                    "src/000_s17_fp0_native_env_runtime.py --help"
                ),
            },
            "scientific_result_eligible": False,
            "effect_experiment_started": False,
            "gpu_used": False,
            "gpu_ids": [],
            "gpu1_used": False,
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
            "automatic_retry": False,
            "recovery_authorized_by_user": True,
            "recovery_attempt_id": "attempt_002",
        }
        atomic_json(failure_path, payload)
        AttemptLedger(task["ledger"]).append(
            {
                "attempt_id": "attempt_001_startup_closeout",
                "step_id": status["step_id"],
                "kind": "startup_failure_closeout",
                "started_at": payload["closed_at"],
                "ended_at": payload["closed_at"],
                "state": "FAILED",
                "scientific_result_eligible": False,
                "closes_attempt_id": "attempt_001",
                "failure_path": str(failure_path.relative_to(ROOT)),
                "failure_sha256": sha256(failure_path),
                "automatic_retry": False,
            }
        )
        writer.transition(
            "FAILED",
            "SCIENTIFIC_FAILED",
            task["status_code"],
            process_alive=False,
            workload_pid=0,
            tmux_session=None,
            stage="bootstrap_import_failure_closed",
            terminal_error=ROOT_CAUSE,
            failure_path=str(failure_path.relative_to(ROOT)),
            failure_sha256=sha256(failure_path),
            automatic_retry=False,
            result_selection_eligible=False,
            affects_scientific_result=False,
            gpu_ids=[],
            gpu1_handoff_used=False,
            next_attempt_id="attempt_002",
            recovery_authorized_by_user=True,
        )
        closed.append({"experiment_id": task["experiment_id"], "failure": str(failure_path)})
    print(json.dumps(closed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
