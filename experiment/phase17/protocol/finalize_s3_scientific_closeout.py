#!/usr/bin/env python3
"""Freeze and verify the completed S17-3 canonical scientific result tree.

This closeout utility never reads or writes the isolated runtime-cycle tree and
never signals a process.  It creates one immutable manifest for the canonical
run, verifies the exactly-one report contract, and annotates the live status
without changing its current scientific or execution state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_ID = "s17_s3_one_epoch_portfolio_seed2023"
CANONICAL = ROOT / "artifacts/phase17/s3_exploration/run-0001"
SUMMARY = CANONICAL / "summary.json"
REPORT = ROOT / "report/第十七阶段/Stage17_S3_P0独立正式筛选报告.md"
REPORT_DIR = REPORT.parent
MANIFEST = (
    ROOT
    / "artifacts/phase17/manifests"
    / "s17_s3_one_epoch_portfolio_seed2023.run-0001.canonical_results.json"
)
SNAPSHOT = (
    ROOT
    / "artifacts/phase17/snapshots"
    / EXPERIMENT_ID
    / "run-0001/manifest.json"
)
BUDGET = ROOT / "experiment/phase17/config/s17_s3_formal_budget.json"
LEDGER = ROOT / "artifacts/phase17/attempts/S17-3.attempts.jsonl"
PARENT_CHECKPOINT = (
    ROOT
    / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30"
    / "model_rec_phase_1_epoch_30.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_summary() -> dict[str, Any]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected S17-3 experiment id")
    if not summary.get("queue_completed") or not summary.get("all_arms_passed"):
        raise ValueError("S17-3 canonical queue is not completely successful")
    if summary.get("failed_arms"):
        raise ValueError("S17-3 canonical summary contains failed arms")
    if len(summary.get("completed_arms", [])) != 10 or len(summary.get("results", [])) != 10:
        raise ValueError("S17-3 canonical closeout requires exactly ten completed arms")
    if summary.get("official_result_claim"):
        raise ValueError("exploration-only S17-3 output cannot be an official result claim")
    if summary.get("test_read") or summary.get("sports_read"):
        raise PermissionError("S17-3 closeout detected a forbidden data read")
    for result in summary["results"]:
        if result.get("state") != "COMPLETED" or result.get("return_code") != 0:
            raise ValueError(f"incomplete canonical arm: {result.get('arm_id')}")
        if not all(result.get("checks", {}).values()):
            raise ValueError(f"failed canonical checks: {result.get('arm_id')}")
        if result.get("test_read") or result.get("sports_read"):
            raise PermissionError(f"forbidden data read in arm: {result.get('arm_id')}")
    return summary


def canonical_files() -> tuple[list[dict[str, Any]], int, str]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(CANONICAL.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"canonical result tree contains a symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": size,
                "sha256": sha256(path),
            }
        )
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    tree_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return records, total_bytes, tree_sha256


def build_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    from experiment.phase17.core.report_contract import enforce_one_report
    from experiment.phase17.core.status_writer import utc_now

    enforce_one_report(REPORT_DIR, "S17-3", "COMPLETED")
    files, total_bytes, tree_sha256 = canonical_files()
    return {
        "schema_version": "phase17.canonical_result_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": "run-0001",
        "step_id": "S17-3",
        "scientific_completed": True,
        "scientific_completed_at": summary["completed_at"],
        "manifest_created_at": utc_now(),
        "canonical_result_dir": str(CANONICAL.relative_to(ROOT)),
        "canonical_tree_file_count": len(files),
        "canonical_tree_size_bytes": total_bytes,
        "canonical_tree_sha256": tree_sha256,
        "files": files,
        "external_evidence": {
            "report": {
                "path": str(REPORT.relative_to(ROOT)),
                "sha256": sha256(REPORT),
            },
            "run_snapshot": {
                "path": str(SNAPSHOT.relative_to(ROOT)),
                "sha256": sha256(SNAPSHOT),
            },
            "budget": {
                "path": str(BUDGET.relative_to(ROOT)),
                "sha256": sha256(BUDGET),
            },
            "attempt_ledger": {
                "path": str(LEDGER.relative_to(ROOT)),
                "sha256": sha256(LEDGER),
            },
            "parent_checkpoint": {
                "path": str(PARENT_CHECKPOINT.relative_to(ROOT)),
                "sha256": sha256(PARENT_CHECKPOINT),
            },
        },
        "result_selection_eligible": True,
        "official_result_claim": False,
        "test_read": False,
        "sports_read": False,
    }


def verify_manifest(payload: dict[str, Any]) -> None:
    current, total_bytes, tree_sha256 = canonical_files()
    if current != payload.get("files"):
        raise RuntimeError("canonical S17-3 result files no longer match the frozen manifest")
    if total_bytes != payload.get("canonical_tree_size_bytes"):
        raise RuntimeError("canonical S17-3 byte count changed")
    if tree_sha256 != payload.get("canonical_tree_sha256"):
        raise RuntimeError("canonical S17-3 tree digest changed")
    for record in payload.get("external_evidence", {}).values():
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"S17-3 external evidence changed: {record['path']}")


def annotate_status(payload: dict[str, Any]) -> None:
    from experiment.phase17.core.status_writer import StatusWriter, sha256_bytes

    writer = StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID)
    current = writer.read()
    manifest_bytes = MANIFEST.read_bytes()
    updates: dict[str, Any] = {
        "scientific_completed_at": payload["scientific_completed_at"],
        "scientific_closeout_state": "ARTIFACTS_FROZEN_REPORT_PUBLISHED",
        "canonical_result_manifest": str(MANIFEST.relative_to(ROOT)),
        "canonical_result_manifest_sha256": sha256_bytes(manifest_bytes),
        "canonical_result_sha256": payload["canonical_tree_sha256"],
        "report_path": str(REPORT.relative_to(ROOT)),
        "report_sha256": payload["external_evidence"]["report"]["sha256"],
    }
    if current.get("execution_state") == "RUNNING_OCCUPANCY_REPEAT":
        updates["gpu1_state"] = "runtime_cycle_active"
        runtime_dir = current.get("repeat_result_dir")
        if runtime_dir:
            updates["current_arm_log"] = f"{runtime_dir}/run.log"
    writer.transition(
        current["scientific_state"],
        current["execution_state"],
        current["status_code"],
        **updates,
    )


def finalize() -> None:
    from experiment.phase17.core.status_writer import atomic_json

    summary = validate_summary()
    if MANIFEST.exists():
        raise FileExistsError(f"canonical manifest already exists: {MANIFEST}")
    payload = build_manifest(summary)
    atomic_json(MANIFEST, payload)
    MANIFEST.chmod(0o444)
    verify_manifest(payload)
    annotate_status(payload)


def verify() -> None:
    validate_summary()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify_manifest(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["finalize", "verify"])
    args = parser.parse_args()
    if args.action == "finalize":
        finalize()
    else:
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
