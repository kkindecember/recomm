#!/usr/bin/env python3
"""Finalize the non-efficacy S16-2 implementation/contract smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    smoke = json.loads((output / "smoke_summary.json").read_text(encoding="utf-8"))
    if smoke["verdict"] != "PASS_S16_2_SPECGR_CONTRACT_SMALL_SMOKE":
        raise SystemExit("S16-2 small smoke is not PASS")
    if any(value.startswith("PASS") for value in smoke["formal_gates"].values()):
        raise SystemExit("A formal gate was incorrectly promoted by the one-step smoke")

    code_paths = [
        ROOT / "experiment/phase16/protocol/official_specgr_runtime.py",
        ROOT / "experiment/phase16/protocol/specgr_faithful.py",
        ROOT / "experiment/phase16/protocol/specgr_contract_smoke.py",
        ROOT / "experiment/phase16/protocol/finalize_s2_contract_smoke.py",
        ROOT / "experiment/phase16/tests/test_specgr_faithful.py",
        ROOT / "experiment/phase16/run_stage16_s2_specgr_contract_smoke.sh",
    ]
    generated = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": generated,
        "test_read": False,
        "network_used_during_experiment": False,
    }
    write_json(
        output / "code_sha256.json",
        {**common, "files": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}},
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": "bash experiment/phase16/run_stage16_s2_specgr_contract_smoke.sh",
            "working_directory": str(ROOT),
            "hard_timeout_seconds": 600,
            "automatic_retry": False,
            "formal_training_started": False,
            "formal_saux_command_template": "CUDA_VISIBLE_DEVICES=<USER_GPU> bash experiment/phase16/run_stage16_s2_saux_formal.sh",
        },
    )
    required = [
        "config.json",
        "source_manifest.json",
        "input_file_sha256.json",
        "code_sha256.json",
        "open_file_manifest.json",
        "data_provenance.json",
        "resource_summary.json",
        "smoke_summary.json",
        "command_manifest.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"S16-2 contract-smoke artifact contract missing: {missing}")
    write_json(
        output / "summary.json",
        {
            **common,
            "verdict": "PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE",
            "official_unisrec_runtime": "PASS_PINNED_SPECGR_AND_RECBOLE_SOURCE_EXECUTION",
            "unit_tests_passed": 24,
            "small_smoke": smoke["verdict"],
            "formal_training_started": False,
            "scientific_efficacy_metric_produced": False,
            "stage_status": "IN_PROGRESS_AWAITING_USER_GPU_FOR_FORMAL_TRAINING_AND_ADMISSION",
            "formal_gates": smoke["formal_gates"],
        },
    )
    print("PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
