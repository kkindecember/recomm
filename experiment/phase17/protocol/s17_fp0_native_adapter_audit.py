#!/usr/bin/env python3
"""One-shot CPU audit of the Stage17 full-data native LATTE adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path

from experiment.phase17.core.full_latte_native_adapter import build_latte_native_bundle
from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import AttemptLedger, StatusWriter, atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "s17_fp0_native_data_adapter_audit"
ATTEMPT_ID = "attempt_001"
RESULT_DIR = ROOT / "artifacts/phase17/fullport/fp0/native_data_adapter/attempt_001"
SUMMARY_PATH = RESULT_DIR / "summary.json"
STATUS_DIR = ROOT / "artifacts/phase17/status"
LEDGER_PATH = ROOT / "artifacts/phase17/attempts/S17-FP0-NATIVE-DATA-ADAPTER.attempts.jsonl"
DATA_MANIFEST = ROOT / "artifacts/phase17/fullport/manifests/data_manifest.json"


def main() -> int:
    status_path = STATUS_DIR / f"{EXPERIMENT_ID}.status.json"
    if status_path.exists() or RESULT_DIR.exists():
        raise FileExistsError("native adapter audit attempt_001 already exists; no implicit retry")
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    AttemptLedger(LEDGER_PATH).append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": "S17-FP0-NATIVE-DATA-ADAPTER",
            "kind": "cpu_contract_audit",
            "started_at": started_at,
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "automatic_retry": False,
            "gpu_ids": [],
        }
    )
    writer = StatusWriter(STATUS_DIR, EXPERIMENT_ID)
    writer.initialize(
        step_id="S17-FP0-NATIVE-DATA-ADAPTER",
        attempt_id=ATTEMPT_ID,
        track_id="FP0-INFRASTRUCTURE",
        canonical_result_dir=str(RESULT_DIR.relative_to(ROOT)),
        log_path=None,
        extra={
            "affects_scientific_result": False,
            "result_selection_eligible": False,
            "automatic_retry": False,
            "gpu_ids": [],
            "gpu1_handoff_used": False,
            "effect_experiment_started": False,
            "external_target_materialized": False,
            "d1_read": False,
            "d2_read": False,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP0_NATIVE_DATA_ADAPTER_PREFLIGHT",
        process_alive=True,
        stage="build_train_prefix_bundle",
    )
    writer.transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_FP0_NATIVE_DATA_ADAPTER_AUDITING",
        process_alive=True,
    )
    bundle = build_latte_native_bundle(root=ROOT)
    expected = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    actual = {
        "users": len(bundle.train_sequences),
        "internal_dev_users": len(bundle.internal_dev_sequences),
        "rolling_train_examples": bundle.rolling_train_examples,
        "train_catalog_items": bundle.train_catalog_items,
        "item_catalog_items": bundle.catalog_items,
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError(f"native adapter count drift for {key}: {value} != {expected[key]}")
    summary = {
        "schema_version": "phase17.s17_fp0_native_data_adapter_summary.v1",
        "verdict": "PASS_S17_FP0_NATIVE_DATA_ADAPTER",
        "completed_at": utc_now(),
        "counts": actual,
        "tokenizer_fit_catalog_items": bundle.tokenizer_fit_catalog_items,
        "tokenizer_fit_scope": "supervised_train_sequences_after_internal_dev_position_holdout",
        "data_manifest_path": str(DATA_MANIFEST.relative_to(ROOT)),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "train_split_role": "train_prefix_with_internal_dev_position_removed",
        "validation_split_role": "train_prefix_position_held_out_internal_dev",
        "official_pipeline_test_role": "non_efficacy_internal_dev_alias",
        "external_evaluation_path": "separate_authorized_one_shot_runner_not_opened",
        "external_target_materialized": False,
        "effect_experiment_started": False,
        "gpu_used": False,
        "gpu_ids": [],
        "gpu1_handoff_used": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "automatic_retry": False,
        "next_gate": "S17-FP0-NATIVE-TOKENIZER-INTEGRATION",
    }
    atomic_json(SUMMARY_PATH, summary)
    writer.transition(
        "COMPLETED",
        "SCIENTIFIC_COMPLETED",
        "PASS_S17_FP0_NATIVE_DATA_ADAPTER",
        process_alive=False,
        stage="native_data_adapter_complete",
        progress={"current": 1, "total": 1, "unit": "adapter_audit"},
        summary_path=str(SUMMARY_PATH.relative_to(ROOT)),
        summary_sha256=sha256(SUMMARY_PATH),
        next_gate=summary["next_gate"],
        result_selection_eligible=False,
        affects_scientific_result=False,
        gpu_ids=[],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
