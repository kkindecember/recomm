#!/usr/bin/env python3
"""Mechanically finalize S16-3B without promoting the faithful S16-3 Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiment.phase16.protocol.genrecedit_rank_sufficiency import (
    RANK_TOLERANCE_RULE,
    STRUCTURAL_BLOCKED,
    classify_all_request_upper_bound,
    effective_checkpoints,
)
from experiment.phase16.protocol.gfull_objective_resource_sweep import (
    ROOT,
    sha256,
    utc_now,
    write_json,
)
from experiment.phase16.protocol.gfull_rank_sufficiency_diagnostic import (
    EXECUTED_CODE_PATHS,
    execution_identity_payload,
)


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0


def finalize(config_path: Path) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    output = ROOT / config["output_dir"]
    raw_path = output / "rank_diagnostic_raw.json"
    summary_path = output / "summary.json"
    identity_path = output / "execution_identity.json"
    if summary_path.exists():
        raise ValueError("Refusing to overwrite an existing S16-3B summary")
    if not raw_path.is_file() or raw_path.is_symlink():
        raise ValueError("Missing regular S16-3B raw diagnostic")
    if not identity_path.is_file() or identity_path.is_symlink():
        raise ValueError("Missing regular S16-3B execution identity")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    current_identity = execution_identity_payload(config_path, config_sha)
    identity_unchanged = all(
        identity.get(key) == current_identity[key]
        for key in ("config_path", "config_sha256", "code_sha256")
    ) and isinstance(identity.get("captured_at_utc"), str)

    width = int(config["diagnostic"]["linear_system_width"])
    expected_requests = {
        int(position): int(value)
        for position, value in config["diagnostic"][
            "full_request_counts_by_position"
        ].items()
    }
    expected_covariance = {
        int(position): int(value)
        for position, value in config["diagnostic"][
            "full_covariance_rows_by_position"
        ].items()
    }
    positions = raw.get("position_diagnostics", {})
    position_contract = set(positions) == {str(position) for position in range(6)}
    if position_contract:
        for position in range(6):
            row = positions[str(position)]
            checkpoints = effective_checkpoints(
                config["diagnostic"]["request_key_checkpoints"],
                total=expected_requests[position],
            )
            curve = row.get("rank_curve", [])
            final = curve[-1] if curve else {}
            position_contract = position_contract and (
                row.get("position") == position
                and row.get("layer") == position % 4
                and row.get("request_count") == expected_requests[position]
                and row.get("covariance_rows") == expected_covariance[position]
                and row.get("effective_checkpoints") == list(checkpoints)
                and [entry.get("request_count") for entry in curve]
                == list(checkpoints)
                and row.get("full_covariance_universe_processed") is True
                and row.get("full_request_key_universe_processed") is True
                and row.get("all_request_key_superset") is True
                and row.get("valid_z_filter_applied") is False
                and row.get("z_optimization_run") is False
                and row.get("weight_delta_solve_run") is False
                and row.get("ridge_added") is False
                and row.get("pseudoinverse_used") is False
                and row.get("jitter_fallback_used") is False
                and row.get("outcome_resampling_used") is False
                and isinstance(row.get("request_order_sha256"), str)
                and len(row["request_order_sha256"]) == 64
                and final.get("system", {}).get("rank")
                == row.get("final_system_rank")
                and final.get("system", {}).get("nullity")
                == row.get("final_system_nullity")
                and final.get("key_gram", {}).get("rank")
                == row.get("final_key_rank")
                and int(row.get("final_system_rank", -1))
                + int(row.get("final_system_nullity", -1))
                == width
                and _finite_nonnegative(row.get("covariance_elapsed_seconds"))
                and _finite_nonnegative(row.get("key_and_rank_elapsed_seconds"))
            )
            for entry in curve:
                for label in ("key_gram", "system"):
                    diagnostic = entry.get(label, {})
                    position_contract = position_contract and (
                        diagnostic.get("width") == width
                        and 0 <= int(diagnostic.get("rank", -1)) <= width
                        and diagnostic.get("nullity")
                        == width - int(diagnostic.get("rank", -1))
                        and diagnostic.get("tolerance_rule") == RANK_TOLERANCE_RULE
                        and diagnostic.get("significant_negative_eigenvalues") == 0
                    )
            covariance_row = row.get("covariance", {})
            position_contract = position_contract and (
                covariance_row.get("width") == width
                and covariance_row.get("tolerance_rule") == RANK_TOLERANCE_RULE
                and covariance_row.get("significant_negative_eigenvalues") == 0
            )

    classification = (
        classify_all_request_upper_bound(positions, width=width)
        if position_contract
        else None
    )
    parent_hashes_current = {
        label: sha256(ROOT / config["inputs"][label]["path"])
        for label in (
            "parent_a4_raw",
            "parent_a4_status",
            "parent_request_manifest",
            "parent_request_checkpoint",
        )
    }
    final_checks = {
        "raw_verdict_pass": raw.get("verdict")
        == "PASS_S16_3B_RANK_DIAGNOSTIC_RAW",
        "raw_contract_checks_pass": bool(raw.get("contract_checks"))
        and all(raw["contract_checks"].values()),
        "execution_identity_unchanged": identity_unchanged,
        "executed_code_paths_exact": set(identity.get("code_sha256", {}))
        == set(EXECUTED_CODE_PATHS),
        "position_full_universe_contract_pass": position_contract,
        "classification_recomputed_exact": classification is not None
        and classification == raw.get("classification"),
        "parent_artifacts_unchanged": raw.get("parent_hashes_before")
        == raw.get("parent_hashes_after")
        == parent_hashes_current,
        "parent_a4_verdict_preserved": raw.get("parent_a4_baseline", {}).get(
            "verdict"
        )
        == "RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM",
        "train_only_scope": raw.get("parent_request_dataset", {}).get("train_only")
        is True
        and raw.get("validation_used") is False
        and raw.get("test_read") is False,
        "no_scientific_or_faithful_gate_promotion": raw.get(
            "scientific_efficacy_metric_produced"
        )
        is False
        and raw.get("faithful_gate_promoted") is False
        and classification is not None
        and classification["faithful_gate_promoted"] is False,
        "no_retry_or_resume": raw.get("automatic_retry") is False
        and raw.get("automatic_resume") is False,
        "base_checkpoint_unchanged": raw.get("base_checkpoint_unchanged") is True,
        "resource_contract_pass": raw.get("physical_gpu")
        == config["resources"]["fixed_physical_gpu"]
        and raw.get("worker_hard_timeout_seconds")
        == config["resources"]["hard_timeout_seconds"]
        and raw.get("expected_peak_mib") == config["resources"]["expected_peak_mib"]
        and float(raw.get("maximum_peak_reserved_mib", math.inf))
        <= float(config["resources"]["expected_peak_mib"]),
    }
    if not all(final_checks.values()):
        failed = [name for name, passed in final_checks.items() if not passed]
        raise ValueError(f"S16-3B artifact contract failed: {failed}")

    diagnostic_classification = classification["classification"]
    if diagnostic_classification == STRUCTURAL_BLOCKED:
        next_action = (
            "Close faithful no-ridge G-FULL for this GRAM representation. Any ridge, "
            "pseudoinverse, or representation change is a separately named modified method."
        )
    else:
        next_action = (
            "Design a separately approved S16-3C progressive valid-z key diagnostic; "
            "the faithful S16-3 Gate remains closed until an actual valid-z system solves."
        )
    summary = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "verdict": "PASS_S16_3B_RANK_DIAGNOSTIC_COMPLETE",
        "generated_at_utc": utc_now(),
        "diagnostic_classification": diagnostic_classification,
        "classification": classification,
        "s16_3_faithful_gate": "NOT_PASSED_UNCHANGED",
        "s16_4_gfull_unlocked": False,
        "position_final_system_rank": classification[
            "final_system_rank_by_position"
        ],
        "structurally_blocked_positions": classification[
            "structurally_blocked_positions"
        ],
        "next_action": next_action,
        "final_contract_checks": final_checks,
        "raw_artifact": {
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": sha256(raw_path),
        },
        "execution_identity_artifact": {
            "path": str(identity_path.relative_to(ROOT)),
            "sha256": sha256(identity_path),
        },
        "maximum_peak_allocated_mib": raw["maximum_peak_allocated_mib"],
        "maximum_peak_reserved_mib": raw["maximum_peak_reserved_mib"],
        "elapsed_seconds": raw["elapsed_seconds"],
        "scientific_efficacy_metric_produced": False,
        "faithful_gate_promoted": False,
        "validation_used": False,
        "test_read": False,
        "automatic_retry": False,
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = finalize(args.config)
    print(summary["verdict"])
    print(summary["diagnostic_classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
