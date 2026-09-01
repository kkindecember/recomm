#!/usr/bin/env python3
"""Record a researcher-directed supersession of a running FP1/FP2 attempt.

This helper never sends a signal.  It only records the stop decision before an
operator targets the exact tmux session/PID, then records process termination.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now


ARMS = {
    "G0_GRAM_B0_FRESH": "g0_gram_b0_fresh",
    "G1_GRAM_PSID_FULL": "g1_gram_psid_full",
    "G2_GRAM_LATTE_FULL": "g2_gram_latte_full",
}
SEED = 2023
STOP_CODE = "S17_FP12_FORMAL_STOPPED_FOR_MICROBATCH_UPSCALE"


def experiment_id(arm_id: str) -> str:
    return f"s17_fp12_formal_{ARMS[arm_id]}_seed{SEED}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mark(
    root: Path, arm_id: str, stopped_attempt_id: str, replacement_attempt_id: str
) -> int:
    root = root.resolve()
    exp_id = experiment_id(arm_id)
    writer = StatusWriter(root / "artifacts/phase17/status", exp_id)
    status = writer.read()
    if status.get("attempt_id") != stopped_attempt_id:
        raise RuntimeError("formal status does not match the requested stopped attempt")
    if status.get("scientific_state") != "RUNNING":
        raise RuntimeError(f"formal arm is not running: {status.get('scientific_state')}")
    result_dir = root / status["canonical_result_dir"]
    atomic_json(result_dir / "status_before_researcher_stop.json", status)
    decision = {
        "schema_version": "phase17.formal_supersession.v1",
        "recorded_at": utc_now(),
        "experiment_id": exp_id,
        "arm_id": arm_id,
        "stopped_attempt_id": stopped_attempt_id,
        "replacement_attempt_id": replacement_attempt_id,
        "reason": "researcher_directed_microbatch_upscale_to_reduce_wall_time",
        "external_target_materialized": False,
        "checkpoint_frozen": False,
        "result_selection_eligible": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
    }
    atomic_json(result_dir / "supersession.json", decision)
    writer.transition(
        "STOPPED",
        "STOPPED",
        STOP_CODE,
        stage="researcher_stop_recorded_waiting_exact_process_signal",
        process_alive=True,
        stop_requested_at=decision["recorded_at"],
        stop_reason=decision["reason"],
        superseded_by_attempt_id=replacement_attempt_id,
        result_selection_eligible=False,
        affects_scientific_result=False,
        checkpoint_frozen=False,
        external_target_materialized=False,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def finalize(root: Path, arm_id: str, stopped_attempt_id: str) -> int:
    root = root.resolve()
    exp_id = experiment_id(arm_id)
    writer = StatusWriter(root / "artifacts/phase17/status", exp_id)
    status = writer.read()
    if status.get("scientific_state") != "STOPPED":
        raise RuntimeError("stop must be recorded before finalization")
    final = writer.transition(
        "STOPPED",
        "STOPPED",
        STOP_CODE,
        stage="superseded_attempt_process_terminated",
        process_alive=False,
        workload_pid=0,
        gpu_ids=[],
        stopped_at=utc_now(),
        result_selection_eligible=False,
        affects_scientific_result=False,
    )
    result_dir = root / final["canonical_result_dir"]
    atomic_json(result_dir / "final_status.json", final)
    archive = (
        root
        / "artifacts/phase17/status"
        / f"{exp_id}.{stopped_attempt_id}.stopped.status.json"
    )
    atomic_json(archive, final)
    print(json.dumps({"archive": str(archive), "status": STOP_CODE}, indent=2))
    return 0


def mark_guard(
    root: Path, stopped_attempt_id: str, replacement_attempt_id: str
) -> int:
    root = root.resolve()
    exp_id = "s17_fp12_gpu1_runtime_guard"
    writer = StatusWriter(root / "artifacts/phase17/status", exp_id)
    status = writer.read()
    if status.get("attempt_id") != stopped_attempt_id:
        raise RuntimeError("unexpected GPU1 guard attempt")
    if status.get("scientific_state") not in {"PENDING", "PREFLIGHT", "RUNNING"}:
        raise RuntimeError("GPU1 guard is not stoppable")
    result_dir = root / status["canonical_result_dir"]
    atomic_json(result_dir / "status_before_researcher_stop.json", status)
    stopped = writer.transition(
        "STOPPED",
        "STOPPED",
        "S17_FP12_GPU1_RUNTIME_GUARD_STOPPED_FOR_FORMAL_UPSCALE",
        stage="guard_stop_recorded_waiting_exact_process_signal",
        process_alive=True,
        stop_requested_at=utc_now(),
        stop_reason=f"formal_g2_{stopped_attempt_id}_superseded_by_microbatch_upscale",
        superseded_by_attempt_id=replacement_attempt_id,
        result_selection_eligible=False,
        affects_scientific_result=False,
    )
    atomic_json(result_dir / "supersession.json", stopped)
    return 0


def finalize_guard(root: Path, stopped_attempt_id: str) -> int:
    root = root.resolve()
    exp_id = "s17_fp12_gpu1_runtime_guard"
    writer = StatusWriter(root / "artifacts/phase17/status", exp_id)
    status = writer.read()
    if status.get("scientific_state") != "STOPPED":
        raise RuntimeError("GPU1 guard stop was not recorded")
    final = writer.transition(
        "STOPPED",
        "STOPPED",
        "S17_FP12_GPU1_RUNTIME_GUARD_STOPPED_FOR_FORMAL_UPSCALE",
        stage="superseded_guard_process_terminated",
        process_alive=False,
        workload_pid=0,
        gpu_ids=[],
        stopped_at=utc_now(),
        result_selection_eligible=False,
        affects_scientific_result=False,
    )
    result_dir = root / final["canonical_result_dir"]
    atomic_json(result_dir / "final_status.json", final)
    atomic_json(
        root
        / "artifacts/phase17/status"
        / f"{exp_id}.{stopped_attempt_id}.stopped.status.json",
        final,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("mark", "finalize", "mark-guard", "finalize-guard")
    )
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--stopped-attempt-id", default="attempt_001")
    parser.add_argument("--replacement-attempt-id", default="attempt_002")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "mark-guard":
        return mark_guard(
            args.root, args.stopped_attempt_id, args.replacement_attempt_id
        )
    if args.action == "finalize-guard":
        return finalize_guard(args.root, args.stopped_attempt_id)
    if args.arm is None:
        parser.error("--arm is required for formal-arm actions")
    if args.action == "mark":
        return mark(
            args.root,
            args.arm,
            args.stopped_attempt_id,
            args.replacement_attempt_id,
        )
    return finalize(args.root, args.arm, args.stopped_attempt_id)


if __name__ == "__main__":
    raise SystemExit(main())
