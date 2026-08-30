#!/usr/bin/env python3
"""Fail-closed artifact finalizer for Stage16 formal G-RIDGE admission."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiment.phase16.protocol.gridge_formal_admission import (
    FORMAL_CODE_PATHS,
    ROOT,
    sha256,
    utc_now,
    write_json,
)
from experiment.phase16.protocol.genrecedit_inspired import (
    GRIDGE_METHOD_NAME,
    GRIDGE_RIDGE_RULE,
    GRIDGE_SOLVE_VARIANT,
    validate_gridge_method_config,
)


REQUIRED_CONTRACT_CHECKS = {
    "resource_parent_pass",
    "full_request_universe_exact",
    "full_covariance_universe_exact",
    "covariance_convergence_complete",
    "all_position_z_counts_complete",
    "all_position_ridge_solves_pass",
    "aggregate_covers_four_edited_layers",
    "held_ground_truth_opened_after_state_freeze",
    "item_disjoint_admission_exact_and_finite",
    "warm_preservation_exact_and_finite",
    "every_trigger_position_exercised",
    "base_parameter_parity",
    "base_checkpoint_unchanged",
    "fixed_gpu_contract",
    "peak_within_admitted_free_memory",
    "validation_and_test_sealed",
    "automatic_retry_false",
    "config_and_code_identity_unchanged",
}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_position_rows(raw: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rows = raw.get("position_diagnostics")
    if not isinstance(rows, dict) or set(rows) != {str(position) for position in range(6)}:
        raise ValueError("Formal position diagnostics must cover positions 0--5")
    expected = {
        str(key): int(value)
        for key, value in config["frozen_workload"]["request_counts_by_position"].items()
    }
    compressed: dict[str, Any] = {}
    for position, row in rows.items():
        count = expected[position]
        probabilities = row.get("full_vocabulary_target_probabilities")
        legal_ranks = row.get("legal_target_ranks")
        vocabulary_ranks = row.get("full_vocabulary_target_ranks")
        if (
            row.get("request_count") != count
            or row.get("valid_z_count", -1) + row.get("failed_z_count", -1) != count
            or not isinstance(probabilities, list)
            or not isinstance(legal_ranks, list)
            or not isinstance(vocabulary_ranks, list)
            or len(probabilities) != count
            or len(legal_ranks) != count
            or len(vocabulary_ranks) != count
            or any(not finite_number(value) or not 0.0 <= float(value) <= 1.0 for value in probabilities)
            or any(not isinstance(value, int) or value < 1 for value in legal_ranks)
            or any(not isinstance(value, int) or value < 1 for value in vocabulary_ranks)
        ):
            raise ValueError(f"Formal position {position} request diagnostics are incomplete")
        if (
            row.get("method_name") != GRIDGE_METHOD_NAME
            or row.get("solve_variant") != GRIDGE_SOLVE_VARIANT
            or row.get("ridge_rule") != GRIDGE_RIDGE_RULE
            or row.get("faithful_reproduction") is not False
            or row.get("ridge_added") is not True
            or row.get("pseudoinverse_used") is not False
            or row.get("jitter_fallback_used") is not False
            or row.get("outcome_resampling_used") is not False
            or row.get("regularized_rank") != config["frozen_workload"]["linear_system_width"]
            or row.get("regularized_nullity") != 0
            or row.get("regularized_system_cholesky_info") != 0
            or not finite_number(row.get("solve_relative_residual"))
            or row["solve_relative_residual"]
            > config["frozen_workload"]["maximum_solve_relative_residual"]
        ):
            raise ValueError(f"Formal position {position} G-RIDGE solve contract failed")
        compressed[position] = {
            "request_count": count,
            "valid_z_count": row["valid_z_count"],
            "failed_z_count": row["failed_z_count"],
            "full_vocabulary_probability_mean": sum(probabilities) / count,
            "legal_rank_one_fraction": sum(value == 1 for value in legal_ranks) / count,
            "full_vocabulary_rank_one_fraction": sum(value == 1 for value in vocabulary_ranks) / count,
            "delta_norm": row["delta_norm"],
            "delta_rank": row["delta_rank"],
            "unregularized_rank": row["unregularized_rank"],
            "unregularized_nullity": row["unregularized_nullity"],
            "regularized_rank": row["regularized_rank"],
            "regularized_condition": row["regularized_condition"],
            "ridge_value": row["ridge_value"],
            "ridge_relative_to_spectral_scale": row["ridge_relative_to_spectral_scale"],
            "solve_relative_residual": row["solve_relative_residual"],
        }
    return compressed


def validate_checkpoints(raw: Mapping[str, Any], config_sha: str) -> dict[str, Any]:
    contract = raw.get("checkpoint_manifest", {})
    path = ROOT / str(contract.get("path", ""))
    if not path.is_file() or path.is_symlink() or sha256(path) != contract.get("sha256"):
        raise ValueError("Formal checkpoint manifest is missing or drifted")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("config_sha256") != config_sha
        or manifest.get("automatic_resume") is not False
        or manifest.get("manual_resume_requires_explicit_user_confirmation") is not True
    ):
        raise ValueError("Formal checkpoint resume contract drift")
    expected = {"full_covariance", "aggregate_deltas"} | {
        f"position_{position}_delta" for position in range(6)
    }
    if set(manifest.get("artifacts", {})) != expected:
        raise ValueError("Formal checkpoint artifact set is incomplete")
    for label, spec in manifest["artifacts"].items():
        artifact = ROOT / spec["path"]
        if not artifact.is_file() or artifact.is_symlink() or sha256(artifact) != spec["sha256"]:
            raise ValueError(f"Formal checkpoint drift: {label}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "artifact_count": len(expected),
        "automatic_resume": False,
        "manual_resume_requires_explicit_user_confirmation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_gridge_method_config(config)
    output = ROOT / config["output_dir"]
    raw_path = output / "formal_admission_summary.json"
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise SystemExit("Refusing to overwrite completed formal G-RIDGE summary")
    if not raw_path.is_file() or raw_path.is_symlink():
        raise ValueError("Formal G-RIDGE raw admission is missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    config_sha = sha256(config_path)
    method = raw.get("method", {})
    if (
        raw.get("experiment_id") != config["experiment_id"]
        or raw.get("attempt_id") != config["attempt_id"]
        or config.get("run_role") != "authoritative_formal"
        or raw.get("run_role") != "authoritative_formal"
        or raw.get("verdict") != "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION_RAW"
        or raw.get("formal_gate") != "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION"
        or method != config["method"]
        or raw.get("validation_used") is not False
        or raw.get("test_read") is not False
        or raw.get("automatic_retry") is not False
        or raw.get("scientific_efficacy_metric_produced") is not False
    ):
        raise ValueError("Formal G-RIDGE raw top-level contract failed")
    checks = raw.get("contract_checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CONTRACT_CHECKS or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("Formal G-RIDGE contract checks are not exact PASS")
    identity = raw.get("execution_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("config_sha256") != config_sha
        or identity.get("config_path") != str(config_path.relative_to(ROOT))
        or identity.get("code_sha256")
        != {path: sha256(ROOT / path) for path in FORMAL_CODE_PATHS}
    ):
        raise ValueError("Formal G-RIDGE execution identity drift")
    identity_path = output / "execution_identity.json"
    if sha256(identity_path) != raw.get("execution_identity_sha256"):
        raise ValueError("Formal G-RIDGE identity artifact drift")
    for label, spec in config["inputs"].items():
        if "sha256" not in spec:
            continue
        path = ROOT / spec["path"]
        if not path.is_file() or path.is_symlink() or sha256(path) != spec["sha256"]:
            raise ValueError(f"Formal frozen input changed after execution: {label}")
    source = ROOT / config["inputs"]["official_genrecedit"]["path"]
    head = __import__("subprocess").check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = __import__("subprocess").check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if head != config["inputs"]["official_genrecedit"]["commit"] or dirty:
        raise ValueError("Formal pinned GenRecEdit source drifted after execution")
    positions = validate_position_rows(raw, config)
    checkpoints = validate_checkpoints(raw, config_sha)
    pseudo = raw["item_disjoint_admission_non_promotional"]
    warm = raw["warm_preservation_non_promotional"]
    if (
        pseudo.get("events") != config["admission"]["item_disjoint_events"]
        or pseudo.get("beam_size") != config["admission"]["beam_size"]
        or pseudo.get("all_finite") is not True
        or warm.get("events") != config["admission"]["warm_preservation_events"]
        or warm.get("base_all_finite") is not True
        or warm.get("edited_all_finite") is not True
        or raw.get("deferred_input_policy", {}).get("opened_after_state_frozen") is not True
        or raw.get("deferred_input_policy", {}).get("used_for_state_selection") is not False
        or raw.get("deferred_input_policy", {}).get("used_for_ridge_selection") is not False
    ):
        raise ValueError("Formal admission/deferred-input contract failed")
    numeric_metrics = [
        pseudo["hit_at_50_non_promotional"],
        pseudo["mrr_non_promotional"],
        warm["exact_top50_fraction"],
        warm["mean_top50_set_overlap"],
        warm["base_hit_at_50_non_promotional"],
        warm["edited_hit_at_50_non_promotional"],
        warm["base_mrr_non_promotional"],
        warm["edited_mrr_non_promotional"],
    ]
    if any(not finite_number(value) for value in numeric_metrics):
        raise ValueError("Formal admission aggregate metrics are non-finite")
    expected_opened = {
        spec["path"]
        for spec in config["inputs"].values()
        if "sha256" in spec
    }
    opened = set(raw.get("opened_files", []))
    if not expected_opened.issubset(opened):
        raise ValueError("Formal open-file manifest omits a declared input")
    forbidden_opened = [
        path
        for path in opened
        if "validation" in Path(path).name.lower() or "test" in Path(path).name.lower()
    ]
    if forbidden_opened:
        raise ValueError(f"Formal validation/test input was opened: {forbidden_opened}")

    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": utc_now(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha,
        "test_read": False,
        "validation_used": False,
        "automatic_retry": False,
    }
    summary = {
        **common,
        "status": "COMPLETED",
        "verdict": "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION",
        "scientific_scope": "train-only contract/admission; metrics are non-promotional",
        "scientific_efficacy_metric_produced": False,
        "method": config["method"],
        "resource_parent": raw["resource_parent"],
        "full_universe": raw["full_universe"],
        "position_diagnostics": positions,
        "aggregated_parameters": raw["aggregated_parameters"],
        "item_disjoint_admission_non_promotional": pseudo,
        "warm_preservation_non_promotional": warm,
        "trigger_evidence": raw["trigger_evidence"],
        "base_parameter_parity": raw["base_parameter_parity"],
        "contract_checks": checks,
        "checkpoint_contract": checkpoints,
        "resource_summary": raw["resource_summary"],
        "elapsed_seconds": raw["elapsed_seconds"],
        "next_action": "S16-3 is complete; S16-4 G-RIDGE efficacy remains sealed and requires separate authorization.",
    }
    write_json(output / "config.json", config)
    write_json(summary_path, summary)
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "edit_state": "5963 real-cold targets x 10 S16-1 train-only pseudo contexts",
            "covariance": "all S16-1 retained-warm train transitions by legal lexical position",
            "ridge_selection": "train-only system spectrum; no validation/test/outcome selection",
            "item_disjoint_admission": "7435 S16-1 train-derived pseudo-cold held events opened only after edit-state freeze",
            "warm_preservation": "512 deterministic S16-1 train transitions",
            "metrics_promotion_eligible": False,
        },
    )
    write_json(
        output / "input_file_sha256.json",
        {
            **common,
            "files": {spec["path"]: spec["sha256"] for spec in config["inputs"].values() if "sha256" in spec},
            "official_genrecedit": config["inputs"]["official_genrecedit"],
        },
    )
    write_json(
        output / "code_sha256.json",
        {**common, "files": identity["code_sha256"]},
    )
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_files": sorted(opened),
            "deferred_input_policy": raw["deferred_input_policy"],
            "declared_external_input_scope": "explicit frozen data/config/source/tokenizer identities and frozen request shards",
        },
    )
    write_json(output / "resource_summary.json", {**common, **raw["resource_summary"]})
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": config["exact_start_command"],
            "physical_gpu": config["resources"]["fixed_physical_gpu"],
            "visible_gpu": 0,
            "automatic_retry": False,
        },
    )
    write_json(
        output / "authoritative_completion.json",
        {
            **common,
            "authoritative_stage_status": "COMPLETED",
            "authoritative_status_code": "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION",
            "summary_path": str(summary_path.relative_to(ROOT)),
            "summary_sha256": sha256(summary_path),
            "full_requests_completed": 302400,
            "ridge_positions_completed": 6,
            "item_disjoint_events_completed": 7435,
            "warm_preservation_pairs_completed": 512,
            "validation_used": False,
            "test_read": False,
            "scientific_efficacy_metric_produced": False,
        },
    )
    print("PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
