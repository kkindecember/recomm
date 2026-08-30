#!/usr/bin/env python3
"""CPU-only adjudication of the immutable failed S16-3B b1 artifact.

This recovery does not recompute covariance, keys, eigenvalues, or ranks.  It
mechanically applies a narrower proof-eligibility rule to the frozen b1 raw
diagnostics while preserving the source artifact's FAILED terminal status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
RANK_TOLERANCE_RULE = "max(matrix_shape)*float64_eps*max_abs_eigenvalue"
STRUCTURAL_BLOCKED = "PROVEN_STRUCTURAL_RANK_BLOCKED"
VALID_Z_REQUIRED = "ALL_REQUEST_UPPER_BOUND_FULL_RANK_VALID_Z_DIAGNOSTIC_REQUIRED"
INCONCLUSIVE_PSD = "INCONCLUSIVE_NUMERICAL_PSD_EVIDENCE"
RECOVERY_VERDICT = "PASS_S16_3B_RECOVERY_ADJUDICATION_COMPLETE"
EXECUTED_CODE_PATHS = (
    "experiment/phase16/protocol/finalize_s3b_rank_sufficiency_recovery.py",
    "experiment/phase16/run_stage16_s3b_rank_sufficiency_recovery_c1_cpu.sh",
    "experiment/phase16/tests/test_gfull_rank_sufficiency_recovery.py",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _rank_diagnostic(
    diagnostic: Mapping[str, Any], *, width: int, label: str
) -> dict[str, Any]:
    rank = _exact_nonnegative_int(diagnostic.get("rank"), label=f"{label}.rank")
    nullity = _exact_nonnegative_int(
        diagnostic.get("nullity"), label=f"{label}.nullity"
    )
    negative = _exact_nonnegative_int(
        diagnostic.get("significant_negative_eigenvalues"),
        label=f"{label}.significant_negative_eigenvalues",
    )
    tolerance = diagnostic.get("tolerance")
    minimum = diagnostic.get("min_eigenvalue")
    maximum = diagnostic.get("max_abs_eigenvalue")
    if (
        diagnostic.get("width") != width
        or rank > width
        or rank + nullity != width
        or diagnostic.get("tolerance_rule") != RANK_TOLERANCE_RULE
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
    ):
        raise ValueError(f"Malformed frozen rank diagnostic: {label}")
    return {
        "rank": rank,
        "nullity": nullity,
        "tolerance": float(tolerance),
        "min_eigenvalue": float(minimum),
        "max_abs_eigenvalue": float(maximum),
        "significant_negative_eigenvalues": negative,
        "numerical_psd": negative == 0,
    }


def _effective_checkpoints(configured: Sequence[int | str], total: int) -> list[int]:
    if total < 1 or not configured:
        raise ValueError("Recovery checkpoints require a positive full universe")
    values: list[int] = []
    for value in configured:
        if value == "full":
            values.append(total)
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Recovery checkpoints must be positive integers or 'full'")
        else:
            values.append(min(value, total))
    return sorted(set(values + [total]))


def adjudicate_positions(
    positions: Mapping[int | str, Mapping[str, Any]],
    *,
    width: int,
    expected_request_counts: Mapping[int, int],
    expected_covariance_rows: Mapping[int, int],
    configured_checkpoints: Sequence[int | str],
) -> dict[str, Any]:
    """Apply the frozen final-system-only numerical-PSD proof rule."""

    normalized = {int(position): row for position, row in positions.items()}
    expected_positions = set(range(6))
    if (
        width < 1
        or set(normalized) != expected_positions
        or set(expected_request_counts) != expected_positions
        or set(expected_covariance_rows) != expected_positions
    ):
        raise ValueError("Recovery adjudication requires complete positions 0--5")

    position_results: dict[str, Any] = {}
    eligible: list[int] = []
    ineligible: list[int] = []
    blocked: list[int] = []
    final_ranks: dict[str, int] = {}

    for position in range(6):
        row = normalized[position]
        expected_requests = expected_request_counts[position]
        expected_covariance = expected_covariance_rows[position]
        expected_curve = _effective_checkpoints(
            configured_checkpoints, expected_requests
        )
        curve = row.get("rank_curve")
        if not isinstance(curve, list) or not curve:
            raise ValueError(f"Position {position} has no frozen rank curve")
        observed_curve = [entry.get("request_count") for entry in curve]
        if (
            row.get("position") != position
            or row.get("layer") != position % 4
            or row.get("request_count") != expected_requests
            or row.get("covariance_rows") != expected_covariance
            or row.get("effective_checkpoints") != expected_curve
            or observed_curve != expected_curve
            or row.get("full_covariance_universe_processed") is not True
            or row.get("full_request_key_universe_processed") is not True
            or row.get("all_request_key_superset") is not True
            or row.get("valid_z_filter_applied") is not False
            or row.get("z_optimization_run") is not False
            or row.get("weight_delta_solve_run") is not False
            or row.get("ridge_added") is not False
            or row.get("pseudoinverse_used") is not False
            or row.get("jitter_fallback_used") is not False
            or row.get("outcome_resampling_used") is not False
        ):
            raise ValueError(f"Position {position} violates full-universe recovery contract")

        covariance = _rank_diagnostic(
            row.get("covariance", {}), width=width, label=f"p{position}.covariance"
        )
        prefix_system_negatives: list[dict[str, int]] = []
        for entry in curve:
            _rank_diagnostic(
                entry.get("key_gram", {}),
                width=width,
                label=f"p{position}.key@{entry.get('request_count')}",
            )
            system_entry = _rank_diagnostic(
                entry.get("system", {}),
                width=width,
                label=f"p{position}.system@{entry.get('request_count')}",
            )
            if system_entry["significant_negative_eigenvalues"]:
                prefix_system_negatives.append(
                    {
                        "request_count": int(entry["request_count"]),
                        "significant_negative_eigenvalues": system_entry[
                            "significant_negative_eigenvalues"
                        ],
                    }
                )

        final = curve[-1]
        final_key = _rank_diagnostic(
            final.get("key_gram", {}), width=width, label=f"p{position}.final_key"
        )
        final_system = _rank_diagnostic(
            final.get("system", {}), width=width, label=f"p{position}.final_system"
        )
        if (
            final.get("request_count") != expected_requests
            or row.get("final_key_rank") != final_key["rank"]
            or row.get("final_system_rank") != final_system["rank"]
            or row.get("final_system_nullity") != final_system["nullity"]
        ):
            raise ValueError(f"Position {position} final diagnostic linkage failed")

        reasons: list[str] = []
        if not covariance["numerical_psd"]:
            reasons.append("covariance_significant_negative_eigenvalues")
        if not final_key["numerical_psd"]:
            reasons.append("final_key_gram_significant_negative_eigenvalues")
        if not final_system["numerical_psd"]:
            reasons.append("final_system_significant_negative_eigenvalues")
        proof_eligible = not reasons
        final_rank = final_system["rank"]
        final_ranks[str(position)] = final_rank
        if proof_eligible:
            eligible.append(position)
            if final_rank < width:
                blocked.append(position)
        else:
            ineligible.append(position)
        position_results[str(position)] = {
            "position": position,
            "proof_eligible": proof_eligible,
            "proof_ineligibility_reasons": reasons,
            "covariance": covariance,
            "final_key_gram": final_key,
            "final_system": final_system,
            "final_system_rank_deficient": final_rank < width,
            "intermediate_prefix_system_negatives_diagnostic_only": [
                entry
                for entry in prefix_system_negatives
                if entry["request_count"] != expected_requests
            ],
        }

    if blocked:
        classification = STRUCTURAL_BLOCKED
    elif ineligible:
        classification = INCONCLUSIVE_PSD
    else:
        classification = VALID_Z_REQUIRED
    return {
        "classification": classification,
        "linear_system_width": width,
        "proof_eligible_positions": eligible,
        "proof_ineligible_positions": ineligible,
        "structurally_blocked_positions": blocked,
        "final_system_rank_by_position": final_ranks,
        "position_adjudications": position_results,
        "faithful_gate_promoted": False,
        "logical_basis": (
            "A position is proof-eligible only when its complete covariance, final "
            "all-request key Gram, and final full system have zero significant "
            "negative eigenvalues under the frozen tolerance. Intermediate prefix "
            "systems are diagnostic only. One proof-eligible deficient all-request "
            "system is sufficient because every faithful valid-z key set is a subset."
        ),
    }


def _load_frozen_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    observed: dict[str, str] = {}
    for label, spec in config["frozen_inputs"].items():
        path = ROOT / spec["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Frozen recovery input is not a regular file: {label}")
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(f"Frozen recovery input SHA mismatch: {label}")
        paths[label] = path
        observed[label] = digest
    return paths, observed


def finalize(config_path: Path) -> dict[str, Any]:
    started_at = utc_now()
    absolute_config = config_path if config_path.is_absolute() else ROOT / config_path
    if not absolute_config.is_file() or absolute_config.is_symlink():
        raise ValueError("Recovery config must be a regular file")
    config_bytes = absolute_config.read_bytes()
    config = json.loads(config_bytes)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    output = ROOT / config["output_dir"]
    if output.exists():
        raise ValueError("Refusing to overwrite an existing recovery attempt root")
    if (
        config["resources"].get("cpu_only") is not True
        or config["resources"].get("gpu_count") != 0
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("Recovery must execute CPU-only with CUDA_VISIBLE_DEVICES empty")

    paths, input_hashes_before = _load_frozen_inputs(config)
    raw = json.loads(paths["source_b1_raw"].read_text(encoding="utf-8"))
    status = json.loads(paths["source_b1_status"].read_text(encoding="utf-8"))
    checkpoint = json.loads(
        paths["source_b1_checkpoint"].read_text(encoding="utf-8")
    )
    source_identity = json.loads(
        paths["source_b1_identity"].read_text(encoding="utf-8")
    )
    source = config["source_b1_contract"]
    source_summary = ROOT / source["summary_path"]

    failed_raw_checks = sorted(
        name for name, passed in raw.get("contract_checks", {}).items() if not passed
    )
    non_psd_checks_pass = bool(raw.get("contract_checks")) and all(
        passed
        for name, passed in raw["contract_checks"].items()
        if name != "positive_semidefinite_evidence"
    )
    expected_requests = {
        int(position): int(value)
        for position, value in config["adjudication"][
            "full_request_counts_by_position"
        ].items()
    }
    expected_covariance = {
        int(position): int(value)
        for position, value in config["adjudication"][
            "full_covariance_rows_by_position"
        ].items()
    }
    adjudication = adjudicate_positions(
        raw.get("position_diagnostics", {}),
        width=int(config["adjudication"]["linear_system_width"]),
        expected_request_counts=expected_requests,
        expected_covariance_rows=expected_covariance,
        configured_checkpoints=config["adjudication"]["request_key_checkpoints"],
    )

    code_hashes_before = {
        relative: sha256(ROOT / relative) for relative in EXECUTED_CODE_PATHS
    }
    source_identity_sha = input_hashes_before["source_b1_identity"]
    final_checks = {
        "source_b1_raw_sha_frozen": input_hashes_before["source_b1_raw"]
        == config["frozen_inputs"]["source_b1_raw"]["sha256"],
        "source_b1_terminal_failed_preserved": status.get("status")
        == source["source_status_must_remain"]
        and status.get("status_code") == source["source_status_must_remain"]
        and status.get("stage") == "finished"
        and status.get("exit_code") == 3
        and status.get("process_alive") is False,
        "source_b1_full_compute_complete": status.get("progress_current")
        == status.get("progress_total")
        == source["expected_request_total"]
        and status.get("progress_unit") == "train_only_request_keys"
        and checkpoint.get("completed_positions") == 6,
        "source_b1_summary_absent": not source_summary.exists(),
        "source_b1_raw_verdict_failed_preserved": raw.get("verdict")
        == source["raw_verdict"],
        "only_frozen_psd_contract_failed": failed_raw_checks
        == source["expected_failed_raw_checks"]
        and non_psd_checks_pass,
        "rank_rule_frozen": config["adjudication"]["rank_tolerance_rule"]
        == RANK_TOLERANCE_RULE,
        "source_identity_link_exact": raw.get("execution_identity")
        == source_identity
        and raw.get("execution_identity_artifact", {}).get("sha256")
        == source_identity_sha
        and checkpoint.get("execution_identity_sha256") == source_identity_sha,
        "source_attempt_link_exact": raw.get("attempt_id") == source["attempt_id"]
        and status.get("attempt_id") == source["attempt_id"]
        and checkpoint.get("attempt_id") == source["attempt_id"],
        "train_only_and_sealed_scope_preserved": raw.get("parent_request_dataset", {}).get(
            "train_only"
        )
        is True
        and raw.get("validation_used") is False
        and raw.get("test_read") is False
        and status.get("validation_used") is False
        and status.get("test_read") is False,
        "no_retry_resume_or_numerical_fallback": raw.get("automatic_retry") is False
        and raw.get("automatic_resume") is False
        and status.get("automatic_retry") is False
        and status.get("automatic_resume") is False,
        "recovery_proof_exists": adjudication["classification"]
        == STRUCTURAL_BLOCKED
        and bool(adjudication["structurally_blocked_positions"]),
        "faithful_gate_remains_closed": adjudication["faithful_gate_promoted"]
        is False
        and raw.get("faithful_gate_promoted") is False,
        "cpu_only_execution": config["resources"]["gpu_count"] == 0
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    if not all(final_checks.values()):
        failed = [name for name, passed in final_checks.items() if not passed]
        raise ValueError(f"S16-3B recovery contract failed: {failed}")

    _, input_hashes_after = _load_frozen_inputs(config)
    code_hashes_after = {
        relative: sha256(ROOT / relative) for relative in EXECUTED_CODE_PATHS
    }
    if input_hashes_after != input_hashes_before:
        raise ValueError("Frozen b1 inputs changed during recovery adjudication")
    if code_hashes_after != code_hashes_before:
        raise ValueError("Recovery code changed during adjudication")

    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "captured_at_utc": utc_now(),
        "config_path": str(absolute_config.relative_to(ROOT)),
        "config_sha256": config_sha,
        "code_sha256": code_hashes_before,
        "source_input_sha256": input_hashes_before,
        "cpu_only": True,
        "cuda_visible_devices": "",
    }
    identity_path = output / "execution_identity.json"
    write_json(identity_path, identity)
    result = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "verdict": RECOVERY_VERDICT,
        "generated_at_utc": utc_now(),
        "source_b1_terminal_status": "FAILED_PRESERVED",
        "source_b1_raw_verdict": raw["verdict"],
        "source_b1_raw_classification": raw.get("classification"),
        "diagnostic_classification": adjudication["classification"],
        "adjudication": adjudication,
        "final_contract_checks": final_checks,
        "source_input_sha256_before": input_hashes_before,
        "source_input_sha256_after": input_hashes_after,
        "execution_identity_artifact": {
            "path": str(identity_path.relative_to(ROOT)),
            "sha256": sha256(identity_path),
        },
        "s16_3_faithful_gate": "NOT_PASSED_UNCHANGED",
        "s16_4_gfull_unlocked": False,
        "scientific_efficacy_metric_produced": False,
        "faithful_gate_promoted": False,
        "validation_used": False,
        "test_read": False,
        "automatic_retry": False,
        "automatic_resume": False,
        "gpu_used": False,
        "next_action": (
            "Close faithful no-ridge G-FULL for this frozen GRAM representation. "
            "Any ridge, pseudoinverse, representation change, or modified solve must "
            "be a separately named method and cannot inherit the faithful Gate."
        ),
    }
    result_path = output / "adjudication.json"
    write_json(result_path, result)
    status_payload = {
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "COMPLETED",
        "status_code": RECOVERY_VERDICT,
        "stage": "finished",
        "reason": (
            "CPU-only recovery adjudication completed from immutable b1 evidence; "
            "the source b1 artifact remains FAILED."
        ),
        "started_at": started_at,
        "updated_at": utc_now(),
        "process_alive": False,
        "gpu_count": 0,
        "gpu_used": False,
        "exit_code": 0,
        "source_b1_status": "FAILED_PRESERVED",
        "diagnostic_classification": adjudication["classification"],
        "faithful_gate_promoted": False,
        "validation_used": False,
        "test_read": False,
        "automatic_retry": False,
        "automatic_resume": False,
        "exact_start_command": config["exact_start_command"],
        "output_dir": config["output_dir"],
        "adjudication_path": str(result_path.relative_to(ROOT)),
    }
    write_json(output / "status.json", status_payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.config)
    print(result["verdict"])
    print(result["diagnostic_classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
