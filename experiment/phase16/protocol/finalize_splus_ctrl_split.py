#!/usr/bin/env python3
"""Validate an isolated CTRL arm and pair it with the frozen GPU5 S-PLUS arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STAGES = ("pretrain", "finetune")
SCIENTIFIC_CONFIG_FIELDS = (
    "seed",
    "domain",
    "inputs",
    "model",
    "formal_budget",
    "admission",
    "resource_evidence",
    "batching_adaptation",
    "compatibility_patch",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def scientific_core(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in SCIENTIFIC_CONFIG_FIELDS}


def verify_code_freeze(config: dict[str, Any]) -> None:
    for relative, expected in config["code_freeze"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Code SHA drift: {relative}")


def arm_checkpoints(output: Path, arm: str) -> list[Path]:
    return [
        output / "arms" / arm / "checkpoints" / stage / name
        for stage in STAGES
        for name in ("last_state.pt", "final_model.pt")
    ]


def validate_arm_summary(summary: dict[str, Any], config: dict[str, Any], arm: str) -> None:
    expected_verdict = f"PASS_S16_2_{arm.replace('-', '_')}_FORMAL_EXECUTION"
    if summary.get("verdict") != expected_verdict or summary.get("arm") != arm:
        raise ValueError(f"{arm} formal arm is not PASS")
    expected_steps = sum(config["formal_budget"][stage]["optimizer_steps"] for stage in STAGES)
    if summary.get("arm_optimizer_steps") != expected_steps:
        raise ValueError(f"{arm} optimizer-step count drift")
    admission = summary["internal_dev_generation_admission"]
    if not admission["all_finite"] or admission["events"] != config["formal_budget"]["internal_dev_transitions"]:
        raise ValueError(f"{arm} internal-dev admission is incomplete or non-finite")
    if not summary["base_checkpoint_unchanged"] or summary["test_read"] or summary["validation_used"]:
        raise ValueError(f"{arm} checkpoint or sealed-data contract failed")
    if summary["peak_cuda_reserved_mib"] > config["admission"]["maximum_eligible_peak_reserved_mib"]:
        raise ValueError(f"{arm} peak reserved memory exceeded admission ceiling")


def validate_pair(
    plus: dict[str, Any],
    control: dict[str, Any],
    plus_config: dict[str, Any],
    control_config: dict[str, Any],
) -> dict[str, Any]:
    if scientific_core(plus_config) != scientific_core(control_config):
        raise ValueError("Split arms do not share an identical frozen scientific configuration")
    validate_arm_summary(plus, plus_config, "S-PLUS")
    validate_arm_summary(control, control_config, "S-PLUS-CTRL")
    pseudo = plus["pseudo_cold_full_catalog_admission"]
    if (
        not pseudo["all_finite"]
        or pseudo["events"] != plus_config["formal_budget"]["pseudo_cold_events"]
        or pseudo["candidate_items"] != plus_config["formal_budget"]["full_catalog_items"]
    ):
        raise ValueError("S-PLUS pseudo-cold full-catalog admission failed")
    if control["pseudo_cold_full_catalog_admission"] is not None:
        raise ValueError("S-PLUS-CTRL unexpectedly produced a pseudo-cold efficacy admission")
    budget_fields: dict[str, Any] = {}
    for stage in STAGES:
        left = plus["budget_audit"][stage]["budget"]
        right = control["budget_audit"][stage]["budget"]
        if left != right:
            raise ValueError(f"Split paired budget mismatch at {stage}")
        budget_fields[stage] = {"matched": True, "budget": left}
    if plus["base_checkpoint_sha256_before"] != control["base_checkpoint_sha256_before"]:
        raise ValueError("Split arms did not start from the same GRAM checkpoint")
    return budget_fields


def common_payload(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "config_sha256": sha256(config_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False,
        "validation_used": False,
    }


def finalize_control_arm(config_path: Path) -> int:
    config = read_json(config_path)
    verify_code_freeze(config)
    output = ROOT / config["output_dir"]
    control_path = output / "arms" / "S-PLUS-CTRL" / "summary.json"
    control = read_json(control_path)
    validate_arm_summary(control, config, "S-PLUS-CTRL")
    if control["pseudo_cold_full_catalog_admission"] is not None:
        raise ValueError("Split CTRL arm unexpectedly produced pseudo-cold efficacy output")
    checkpoints = arm_checkpoints(output, "S-PLUS-CTRL")
    missing = [str(path.relative_to(output)) for path in checkpoints if not path.is_file()]
    if missing:
        raise ValueError(f"Split CTRL recovery checkpoints missing: {missing}")

    common = common_payload(config, config_path)
    code_paths = [ROOT / relative for relative in config["code_freeze"]]
    write_json(
        output / "code_sha256.json",
        {**common, "files": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}},
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "working_directory": str(ROOT),
            "exact_start_command": config["execution"]["exact_start_command"],
            "background_start_command": config["execution"]["background_start_command"],
            "physical_gpu": config["resources"]["physical_gpu"],
            "visible_gpu": 0,
            "hard_timeout_seconds": config["resources"]["per_arm_hard_timeout_seconds"],
            "automatic_retry": False,
            "parent_a3_modified": False,
        },
    )
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_inputs": [spec["path"] for spec in config["inputs"].values()],
            "source_validation_opened": False,
            "test_opened": False,
        },
    )
    write_json(
        output / "recovery_manifest.json",
        {
            **common,
            "automatic_resume": False,
            "user_authorization_required": True,
            "checkpoints": {str(path.relative_to(output)): sha256(path) for path in checkpoints},
            "resume_command_template": config["execution"]["resume_command_template"],
        },
    )
    split_summary = {
        **common,
        "status": "completed",
        "verdict": "PASS_S16_2_S_PLUS_CTRL_SPLIT_FORMAL_EXECUTION",
        "paired_gate": "PENDING_CROSS_ATTEMPT_FINALIZATION",
        "arm": control,
        "parent_splus_attempt": config["parallel_split"]["plus_source_attempt_id"],
        "same_frozen_scientific_config_as_parent": True,
        "formal_training_completed": True,
        "scientific_efficacy_metric_produced": False,
        "scientific_scope": "train-only CTRL formal execution; no source validation or test",
    }
    write_json(output / "summary.json", split_summary)
    required = [
        "summary.json",
        "status.json",
        "resolved_config.json",
        "config.json",
        "source_manifest.json",
        "input_file_sha256.json",
        "code_sha256.json",
        "command_manifest.json",
        "open_file_manifest.json",
        "data_provenance.json",
        "recovery_manifest.json",
        "gpu_telemetry.csv",
        "run.log",
        "arms/S-PLUS-CTRL/summary.json",
        "arms/S-PLUS-CTRL/metrics.jsonl",
    ] + [str(path.relative_to(output)) for path in checkpoints]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"Split CTRL artifact contract missing: {missing}")
    write_json(
        output / "artifact_contract.json",
        {**common, "verdict": "PASS_SPLUS_CTRL_SPLIT_ARM_ARTIFACT_CONTRACT", "required": required},
    )
    print("PASS_S16_2_S_PLUS_CTRL_SPLIT_FORMAL_EXECUTION", flush=True)
    return 0


def load_source(pair_config: dict[str, Any], arm: str) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    source = pair_config["sources"][arm]
    output = ROOT / source["output_dir"]
    config_path = ROOT / source["resolved_config_path"]
    config = read_json(config_path)
    if config["attempt_id"] != source["attempt_id"] or config["output_dir"] != source["output_dir"]:
        raise ValueError(f"{arm} source identity mismatch")
    if config["resources"]["physical_gpu"] != source["physical_gpu"]:
        raise ValueError(f"{arm} source physical-GPU identity mismatch")
    summary = read_json(output / "arms" / arm / "summary.json")
    return output, config_path, config, summary


def finalize_pair(pair_config_path: Path) -> int:
    pair_config = read_json(pair_config_path)
    verify_code_freeze(pair_config)
    plus_output, plus_config_path, plus_config, plus = load_source(pair_config, "S-PLUS")
    control_output, control_config_path, control_config, control = load_source(pair_config, "S-PLUS-CTRL")
    if plus_output.resolve() == control_output.resolve():
        raise ValueError("Split pair source outputs must be disjoint")
    verify_code_freeze(plus_config)
    verify_code_freeze(control_config)
    control_attempt = read_json(control_output / "summary.json")
    if control_attempt.get("verdict") != "PASS_S16_2_S_PLUS_CTRL_SPLIT_FORMAL_EXECUTION":
        raise ValueError("Isolated CTRL attempt has not passed its own artifact finalization")
    budget_fields = validate_pair(plus, control, plus_config, control_config)

    source_checkpoints = {
        "S-PLUS": arm_checkpoints(plus_output, "S-PLUS"),
        "S-PLUS-CTRL": arm_checkpoints(control_output, "S-PLUS-CTRL"),
    }
    missing = [
        str(path)
        for paths in source_checkpoints.values()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"Split pair source checkpoints missing: {missing}")

    output = ROOT / pair_config["output_dir"]
    common = common_payload(pair_config, pair_config_path)
    write_json(output / "config.json", pair_config)
    source_files = {
        "S-PLUS": [plus_config_path, plus_output / "arms/S-PLUS/summary.json", plus_output / "arms/S-PLUS/metrics.jsonl"],
        "S-PLUS-CTRL": [
            control_config_path,
            control_output / "arms/S-PLUS-CTRL/summary.json",
            control_output / "arms/S-PLUS-CTRL/metrics.jsonl",
        ],
    }
    write_json(
        output / "source_attempt_manifest.json",
        {
            **common,
            "sources": {
                arm: {str(path.relative_to(ROOT)): sha256(path) for path in paths}
                for arm, paths in source_files.items()
            },
            "source_artifacts_modified": False,
        },
    )
    write_json(
        output / "code_sha256.json",
        {
            **common,
            "files": {
                relative: sha256(ROOT / relative)
                for relative in pair_config["code_freeze"]
            },
        },
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "working_directory": str(ROOT),
            "exact_start_command": pair_config["execution"]["exact_start_command"],
            "source_physical_gpus": {"S-PLUS": 5, "S-PLUS-CTRL": 7},
            "automatic_retry": False,
        },
    )
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_inputs": [str(path.relative_to(ROOT)) for paths in source_files.values() for path in paths],
            "source_validation_opened": False,
            "test_opened": False,
        },
    )
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "scientific_config_identity": scientific_core(plus_config),
            "same_seed_data_epochs_batches_optimizer_timeout": True,
            "only_allowed_execution_difference": "physical GPU and isolated artifact root",
            "source_artifacts_modified": False,
        },
    )
    recovery = {
        arm: {str(path.relative_to(ROOT)): sha256(path) for path in paths}
        for arm, paths in source_checkpoints.items()
    }
    write_json(
        output / "recovery_manifest.json",
        {**common, "automatic_resume": False, "source_checkpoints": recovery},
    )
    pair_summary = {
        **common,
        "status": "completed",
        "verdict": "PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION",
        "control_execution_verdict": "PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION",
        "execution_layout": "parallel split across identical RTX A6000 GPUs 5 and 7",
        "arms": {"S-PLUS": plus, "S-PLUS-CTRL": control},
        "paired_budget_audit": budget_fields,
        "same_frozen_scientific_config": True,
        "same_start_checkpoint": True,
        "source_artifacts_modified": False,
        "maximum_peak_reserved_mib": max(plus["peak_cuda_reserved_mib"], control["peak_cuda_reserved_mib"]),
        "scientific_scope": "train-only formal contract/admission; no source validation or test",
        "scientific_efficacy_metric_produced": False,
        "formal_training_completed": True,
    }
    write_json(output / "summary.json", pair_summary)
    required = [
        "summary.json",
        "status.json",
        "config.json",
        "source_attempt_manifest.json",
        "code_sha256.json",
        "command_manifest.json",
        "open_file_manifest.json",
        "data_provenance.json",
        "recovery_manifest.json",
        "run.log",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"Split pair artifact contract missing: {missing}")
    write_json(
        output / "artifact_contract.json",
        {**common, "verdict": "PASS_SPLUS_CTRL_SPLIT_PAIR_ARTIFACT_CONTRACT", "required": required},
    )
    print("PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION", flush=True)
    return 0


def preflight_pair(control_config_path: Path, plus_config_path: Path) -> int:
    control = read_json(control_config_path)
    plus = read_json(plus_config_path)
    if scientific_core(control) != scientific_core(plus):
        raise ValueError("Prepared CTRL config is not scientifically identical to the running S-PLUS config")
    if control["output_dir"] == plus["output_dir"]:
        raise ValueError("Prepared CTRL output collides with the running S-PLUS output")
    if control["resources"]["physical_gpu"] != 7 or plus["resources"]["physical_gpu"] != 5:
        raise ValueError("Prepared split physical-GPU mapping drift")
    verify_code_freeze(control)
    print("PASS_S16_2_SPLUS_CTRL_SPLIT_PREFLIGHT", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "arm", "pair"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plus-config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == "preflight":
        if args.plus_config is None:
            raise SystemExit("--plus-config is required for preflight")
        return preflight_pair(config_path, args.plus_config.resolve())
    if args.mode == "arm":
        return finalize_control_arm(config_path)
    return finalize_pair(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
