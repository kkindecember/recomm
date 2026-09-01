#!/usr/bin/env python3
"""Append-only reconciliation for terminal Stage17 FP0 infrastructure attempts.

Historical attempt rows are immutable.  This command only appends a uniquely
named closeout row for the *current* terminal status and writes a separate
audit artifact.  It deliberately does not infer terminal states for superseded
attempts whose final status snapshots were not retained.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import AttemptLedger, atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = Path("artifacts/phase17/fullport/fp0/audit_reconciliation_001.json")


@dataclass(frozen=True)
class Binding:
    status: str
    ledger: str


BINDINGS = (
    Binding(
        "artifacts/phase17/status/s17_fp0_native_data_adapter_audit.status.json",
        "artifacts/phase17/attempts/S17-FP0-NATIVE-DATA-ADAPTER.attempts.jsonl",
    ),
    Binding(
        "artifacts/phase17/status/s17_fp0_native_env_setup.status.json",
        "artifacts/phase17/attempts/S17-FP0-NATIVE-ENV.attempts.jsonl",
    ),
    Binding(
        "artifacts/phase17/status/s17_fp0_sentence_t5_cache.status.json",
        "artifacts/phase17/attempts/S17-FP0-SENTENCE-T5-CACHE.attempts.jsonl",
    ),
    Binding(
        "artifacts/phase17/status/s17_fp0_cuda_compat_env.status.json",
        "artifacts/phase17/attempts/S17-FP0-CUDA-COMPAT-ENV.attempts.jsonl",
    ),
    Binding(
        "artifacts/phase17/status/s17_fp0_tokenizer_bounded_profile.status.json",
        "artifacts/phase17/attempts/S17-FP0-TOKENIZER-PROFILE.attempts.jsonl",
    ),
    Binding(
        "artifacts/phase17/status/s17_fp0_full_data_tokenizer.status.json",
        "artifacts/phase17/attempts/S17-FP0-FULL-DATA-TOKENIZER.attempts.jsonl",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def inspect_binding(root: Path, binding: Binding) -> dict[str, Any]:
    status_path = root / binding.status
    ledger_path = root / binding.ledger
    status = _load_json(status_path)
    if status["scientific_state"] not in {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}:
        raise RuntimeError(f"status is not terminal: {binding.status}")
    rows = _ledger_rows(ledger_path)
    base = [row for row in rows if row["attempt_id"] == status["attempt_id"]]
    if len(base) != 1:
        raise RuntimeError(
            f"expected one base ledger row for {status['attempt_id']} in {binding.ledger}"
        )
    closeout_id = f"{status['attempt_id']}_terminal_reconciliation"
    closeouts = [row for row in rows if row["attempt_id"] == closeout_id]
    if len(closeouts) > 1:
        raise RuntimeError(f"duplicate closeout rows in {binding.ledger}: {closeout_id}")

    summary_path = status.get("summary_path")
    summary_check: dict[str, Any] | None = None
    if summary_path:
        resolved_summary = root / summary_path
        actual = sha256(resolved_summary)
        expected = status.get("summary_sha256")
        summary_check = {
            "path": summary_path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches_status": expected == actual,
        }
        if expected and expected != actual:
            raise RuntimeError(f"terminal summary hash drift: {summary_path}")

    return {
        "status_path": binding.status,
        "status_sha256": sha256(status_path),
        "ledger_path": binding.ledger,
        "attempt_id": status["attempt_id"],
        "step_id": status["step_id"],
        "scientific_state": status["scientific_state"],
        "execution_state": status["execution_state"],
        "status_code": status["status_code"],
        "updated_at": status["updated_at"],
        "summary": summary_check,
        "closeout_id": closeout_id,
        "closeout_present": len(closeouts) == 1,
    }


def _report_hash_observation(root: Path) -> dict[str, Any]:
    foundation = root / "artifacts/phase17/fullport/fp0/attempt_001/summary.json"
    summary = _load_json(foundation)
    report_path = root / summary["report_path"]
    frozen_hash = summary["report_sha256"]
    current_hash = sha256(report_path)
    return {
        "summary_path": str(foundation.relative_to(root)),
        "report_path": summary["report_path"],
        "hash_at_foundation_freeze": frozen_hash,
        "current_report_sha256": current_hash,
        "changed_after_foundation_freeze": frozen_hash != current_hash,
        "interpretation": (
            "the report received append-only operational updates after the FP0 foundation "
            "freeze; the original summary remains an immutable historical pointer"
        ),
    }


def reconcile(root: Path, *, apply: bool) -> dict[str, Any]:
    root = root.resolve()
    inspections = [inspect_binding(root, binding) for binding in BINDINGS]
    if apply:
        for row in inspections:
            if row["closeout_present"]:
                continue
            AttemptLedger(root / row["ledger_path"]).append(
                {
                    "attempt_id": row["closeout_id"],
                    "closes_attempt_id": row["attempt_id"],
                    "step_id": row["step_id"],
                    "kind": "terminal_status_reconciliation",
                    "started_at": utc_now(),
                    "ended_at": row["updated_at"],
                    "state": row["scientific_state"],
                    "scientific_result_eligible": False,
                    "status_code": row["status_code"],
                    "status_path": row["status_path"],
                    "status_sha256": row["status_sha256"],
                    "summary": row["summary"],
                    "automatic_retry": False,
                }
            )
            row["closeout_present"] = True

    unresolved = []
    for binding in BINDINGS:
        rows = _ledger_rows(root / binding.ledger)
        closed = {row.get("closes_attempt_id") for row in rows}
        for row in rows:
            attempt_id = row["attempt_id"]
            if "_closeout" in attempt_id or "_terminal_reconciliation" in attempt_id:
                continue
            if row["state"] not in {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}:
                if attempt_id not in closed:
                    unresolved.append(
                        {
                            "ledger_path": binding.ledger,
                            "attempt_id": attempt_id,
                            "last_recorded_state": row["state"],
                            "reason": "no retained terminal status snapshot; not inferred",
                        }
                    )

    payload = {
        "schema_version": "phase17.s17_fp0_audit_reconciliation.v1",
        "created_at": utc_now(),
        "mode": "apply" if apply else "inspect",
        "append_only": True,
        "current_terminal_attempts": inspections,
        "historical_attempts_without_machine_terminal_closeout": unresolved,
        "report_hash_observation": _report_hash_observation(root),
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "effect_experiment_started": False,
        "gpu_used": False,
    }
    if apply:
        atomic_json(root / AUDIT_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = reconcile(args.root, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
