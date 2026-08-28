#!/usr/bin/env python3
"""Finalize paired S-PLUS/S-PLUS-CTRL formal reproducibility contracts."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    for relative, expected in config["code_freeze"].items():
        if sha256(ROOT / relative) != expected:
            raise SystemExit(f"Formal code SHA drift at finalization: {relative}")
    summaries = {
        arm: json.loads((output / "arms" / arm / "summary.json").read_text(encoding="utf-8"))
        for arm in config["arms"]
    }
    plus, control = summaries["S-PLUS"], summaries["S-PLUS-CTRL"]
    if plus["verdict"] != "PASS_S16_2_S_PLUS_FORMAL_EXECUTION":
        raise SystemExit("S-PLUS formal arm is not PASS")
    if control["verdict"] != "PASS_S16_2_S_PLUS_CTRL_FORMAL_EXECUTION":
        raise SystemExit("S-PLUS-CTRL formal arm is not PASS")
    budget_fields = {}
    for stage in ("pretrain", "finetune"):
        left = plus["budget_audit"][stage]["budget"]
        right = control["budget_audit"][stage]["budget"]
        if left != right:
            raise SystemExit(f"Paired formal budget mismatch at {stage}")
        budget_fields[stage] = {"matched": True, "budget": left}
    pseudo = plus["pseudo_cold_full_catalog_admission"]
    if not pseudo["all_finite"] or pseudo["events"] != config["formal_budget"]["pseudo_cold_events"]:
        raise SystemExit("S-PLUS pseudo-cold admission is incomplete or non-finite")
    if pseudo["candidate_items"] != config["formal_budget"]["full_catalog_items"]:
        raise SystemExit("S-PLUS full catalog index is incomplete")
    for summary in summaries.values():
        admission = summary["internal_dev_generation_admission"]
        if not admission["all_finite"] or admission["events"] != config["formal_budget"]["internal_dev_transitions"]:
            raise SystemExit("Internal-dev generation admission is incomplete or non-finite")
        if not summary["base_checkpoint_unchanged"] or summary["test_read"] or summary["validation_used"]:
            raise SystemExit("Checkpoint or sealed-data contract failed")
        if summary["peak_cuda_reserved_mib"] > config["admission"]["maximum_eligible_peak_reserved_mib"]:
            raise SystemExit("Formal peak reserved memory exceeded admission ceiling")
    if plus["base_checkpoint_sha256_before"] != control["base_checkpoint_sha256_before"]:
        raise SystemExit("Paired arms did not start from the same GRAM checkpoint")

    runner = ROOT / config["execution"]["runner_path"]
    code_paths = [
        ROOT / "experiment/phase16/protocol/official_specgr_runtime.py",
        ROOT / "experiment/phase16/protocol/resource_probe.py",
        ROOT / "experiment/phase16/protocol/specgr_contract_smoke.py",
        ROOT / "experiment/phase16/protocol/specgr_faithful.py",
        ROOT / "experiment/phase16/protocol/splus_formal_train.py",
        ROOT / "experiment/phase16/protocol/finalize_splus_formal.py",
        ROOT / "experiment/phase16/tests/test_splus_formal.py",
        runner,
    ]
    code_paths = list({path.resolve(): path for path in code_paths + [ROOT / relative for relative in config["code_freeze"]]}.values())
    common = {
        "schema_version": config["schema_version"], "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"], "config_sha256": sha256(config_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False, "validation_used": False,
    }
    write_json(output / "code_sha256.json", {**common, "files": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths}})
    write_json(
        output / "command_manifest.json",
        {
            **common, "working_directory": str(ROOT),
            "exact_start_command": config["execution"]["exact_start_command"],
            "background_start_command": config["execution"]["background_start_command"],
            "physical_gpu": config["resources"]["physical_gpu"], "visible_gpu": 0,
            "per_arm_hard_timeout_seconds": config["resources"]["per_arm_hard_timeout_seconds"],
            "automatic_retry": False,
        },
    )
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_inputs": [spec["path"] for spec in config["inputs"].values()],
            "source_validation_opened": False, "test_opened": False,
        },
    )
    checkpoint_paths = []
    for arm in config["arms"]:
        for stage in ("pretrain", "finetune"):
            for name in ("last_state.pt", "final_model.pt"):
                checkpoint_paths.append(output / "arms" / arm / "checkpoints" / stage / name)
    missing_checkpoints = [str(path.relative_to(output)) for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        raise SystemExit(f"Formal recovery checkpoints missing: {missing_checkpoints}")
    write_json(
        output / "recovery_manifest.json",
        {
            **common, "automatic_resume": False, "user_authorization_required": True,
            "checkpoints": {str(path.relative_to(output)): sha256(path) for path in checkpoint_paths},
            "resume_command_template": "CUDA_VISIBLE_DEVICES=5 python experiment/phase16/protocol/splus_formal_train.py --config experiment/phase16/configs/stage16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32.json --arm <ARM> --resume",
        },
    )
    maximum_peak = max(summary["peak_cuda_reserved_mib"] for summary in summaries.values())
    pair_summary = {
        **common, "status": "completed",
        "verdict": "PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION",
        "control_execution_verdict": "PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION",
        "arms": summaries, "paired_budget_audit": budget_fields,
        "same_start_checkpoint": True,
        "maximum_peak_reserved_mib": maximum_peak,
        "holder_released": False,
        "scientific_scope": "train-only formal contract/admission; no source validation or test",
        "scientific_efficacy_metric_produced": False,
        "formal_training_completed": True,
    }
    write_json(output / "summary.json", pair_summary)
    required = [
        "summary.json", "status.json", "config.json", "source_manifest.json",
        "input_file_sha256.json", "code_sha256.json", "command_manifest.json",
        "open_file_manifest.json", "data_provenance.json", "recovery_manifest.json",
        "gpu_telemetry.csv", "run.log", "arms/S-PLUS/summary.json",
        "arms/S-PLUS/metrics.jsonl", "arms/S-PLUS-CTRL/summary.json",
        "arms/S-PLUS-CTRL/metrics.jsonl",
    ] + [str(path.relative_to(output)) for path in checkpoint_paths]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"Formal S-PLUS artifact contract missing: {missing}")
    write_json(
        output / "artifact_contract.json",
        {**common, "verdict": "PASS_SPLUS_CTRL_FORMAL_ARTIFACT_CONTRACT", "required": required},
    )
    print("PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
