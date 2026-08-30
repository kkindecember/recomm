#!/usr/bin/env python3
"""Finalize the bounded S16-3 G-FULL resource sweep artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

EXECUTED_CODE_PATHS = (
    "experiment/phase16/protocol/genrecedit_faithful.py",
    "experiment/phase16/protocol/genrecedit_inspired.py",
    "experiment/phase16/protocol/genrecedit_data.py",
    "experiment/phase16/protocol/gfull_objective_resource_sweep.py",
    "experiment/phase16/protocol/finalize_s3_gfull_resource_sweep.py",
    "experiment/phase16/protocol/resource_probe.py",
    "experiment/phase16/protocol/specgr_contract_smoke.py",
    "experiment/phase16/protocol/official_specgr_runtime.py",
    "experiment/phase16/protocol/specgr_faithful.py",
    "experiment/phase16/tests/test_genrecedit_faithful.py",
    "experiment/phase16/tests/test_genrecedit_inspired.py",
    "experiment/phase16/tests/test_genrecedit_data.py",
    "experiment/phase16/tests/test_gfull_resource_contract.py",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a2.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a3.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu4.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu4_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve_inner.sh",
    "experiment/phase15/protocol/genrecedit_gram_adapter.py",
    "GRAM/src/model/__init__.py",
    "GRAM/src/model/gram.py",
    "GRAM/src/model/gram_t5.py",
    "GRAM/src/model/gram_t5_config.py",
    "GRAM/src/model/gram_t5_modeling.py",
    "GRAM/src/model/gram_t5_outputs.py",
)

REQUIRED_CONTRACT_KEYS = {
    "full_universe_counts_match",
    "train_only_zero_leakage",
    "all_candidate_semantics_pass",
    "independent_full_lifecycle_probe_pass",
    "candidate_workload_identical",
    "all_positions_exercised",
    "valid_failed_counts_complete",
    "position_contract_workload_exact",
    "covariance_position_coverage_exact",
    "long_position_resource_rows_present",
    "covariance_resource_allocation_exact",
    "covariance_resource_rank_rule_exact",
    "covariance_convergence_report_complete",
    "formal_cache_empty",
    "key_extraction_engineering_contract_exact",
    "isolated_cache_probe_pass",
    "all_position_trigger_parity_contract_pass",
    "strict_generation_resource_path_pass",
    "valid_z_filter_complete",
    "solve_aggregate_trigger_exercised_if_valid",
    "faithful_solve_completed_for_every_valid_position",
    "base_parameter_parity_after_trigger",
    "base_checkpoint_unchanged",
    "peak_within_resource_attempt_cap",
    "fixed_gpu_resource_contract_exact",
    "formal_projection_objective_complete",
}

GRIDGE_REQUIRED_CONTRACT_KEYS = (
    REQUIRED_CONTRACT_KEYS - {"faithful_solve_completed_for_every_valid_position"}
) | {"inspired_ridge_solve_completed_for_every_valid_position"}

REQUIRED_CANDIDATE_SEMANTIC_KEYS = {
    "formal_cache_empty",
    "identical_fixed_request_count",
    "full_30_step_budget_configured",
    "official_lifecycle_prefix",
    "first_ten_outcome_independent_objective_timing",
    "scheduler_finite",
    "valid_failed_cover_fixed_subset",
    "checkpoint_file_unchanged",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def candidate_throughput_from_batches(row: dict[str, Any]) -> float:
    seconds = 0.0
    request_steps = 0
    observed_by_position: dict[str, int] = {}
    observed_batch_keys: set[tuple[int, int]] = set()
    records = row.get("batch_records", [])
    if not records:
        raise ValueError("Candidate has no raw batch timing records")
    for record in records:
        timings = record.get("first_ten_objective_step_seconds", [])
        request_count = int(record.get("request_count", 0))
        position = int(record.get("position", -1))
        batch_index = int(record.get("batch_index", -1))
        if (
            len(timings) != 10
            or request_count <= 0
            or position < 0
            or batch_index < 0
            or (position, batch_index) in observed_batch_keys
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in timings
            )
        ):
            raise ValueError("Candidate raw first-ten timing record is invalid")
        seconds += sum(float(value) for value in timings)
        request_steps += request_count * 10
        observed_batch_keys.add((position, batch_index))
        observed_by_position[str(position)] = (
            observed_by_position.get(str(position), 0) + request_count
        )
    declared_by_position = {
        str(position): int(count)
        for position, count in row.get("candidate_requests_by_position", {}).items()
    }
    if (
        observed_by_position != declared_by_position
        or sum(observed_by_position.values())
        != int(row.get("candidate_request_count", -1))
    ):
        raise ValueError("Candidate raw batches do not cover its declared request workload")
    if seconds <= 0.0 or request_steps <= 0:
        raise ValueError("Candidate raw throughput basis is empty")
    return request_steps / seconds


def expected_selected_microbatch(
    candidates: list[dict[str, Any]], maximum_peak_mib: float
) -> int | None:
    eligible = []
    for row in candidates:
        semantics = row.get("semantic_checks", {})
        try:
            recomputed_throughput = candidate_throughput_from_batches(row)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        recomputed = (
            set(semantics) == REQUIRED_CANDIDATE_SEMANTIC_KEYS
            and all(semantics.values())
            and float(row.get("peak_reserved_mib", 1e30)) <= maximum_peak_mib
            and math.isclose(
                float(row.get("steady_request_steps_per_second", math.inf)),
                recomputed_throughput,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        if not isinstance(row.get("eligible"), bool) or bool(row["eligible"]) != recomputed:
            return None
        if recomputed:
            eligible.append((row, recomputed_throughput))
    if not eligible:
        return None
    best = max(eligible, key=lambda pair: pair[1])
    near = [
        pair
        for pair in eligible
        if pair[1] >= 0.98 * best[1]
    ]
    return int(
        min(
            near,
            key=lambda pair: (
                int(pair[0]["microbatch"]),
                float(pair[0]["peak_reserved_mib"]),
            ),
        )[0]["microbatch"]
    )


def observed_full_30_step_path(probe: dict[str, Any]) -> bool:
    expected_trace = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    return (
        probe.get("scope")
        == "synthetic_failure_row_not_used_for_candidate_selection_or_runtime"
        and probe.get("lifecycle_check_steps") == expected_trace
        and int(probe.get("forward_calls", -1)) == 30
        and int(probe.get("scheduler_step_count", -1)) == 30
        and int(probe.get("failed_z_count", -1)) == 1
        and probe.get("pass") is True
    )


def convergence_row_equivalents(
    checkpoints_by_position: dict[int, tuple[int | str, ...]],
    available_rows_by_position: dict[int, int],
) -> int:
    total = 0
    for position, checkpoints in checkpoints_by_position.items():
        available = int(available_rows_by_position[position])
        effective = {
            available if checkpoint == "full" else min(int(checkpoint), available)
            for checkpoint in checkpoints
        }
        total += available + sum(count for count in effective if count < available)
    return total


def recompute_projection(
    config: dict[str, Any], raw: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Rebuild every formal runtime component from raw resource measurements."""

    positions = raw["position_diagnostics"]
    full_requests_by_position = {
        int(position): int(value)
        for position, value in raw["full_universe"][
            "request_counts_by_position"
        ].items()
    }
    if set(full_requests_by_position) != set(range(6)):
        raise ValueError("Projection basis lacks full request counts for all positions")
    full_requests = sum(full_requests_by_position.values())
    resource_requests = sum(int(row["request_count"]) for row in positions.values())
    resource_valid = sum(int(row["valid_z_count"]) for row in positions.values())
    solved_positions = sum(row.get("solve_completed") is True for row in positions.values())
    if resource_requests <= 0 or resource_valid <= 0 or solved_positions <= 0:
        raise ValueError("Projection basis lacks measured requests, valid z, or solves")
    selected_rows = [
        row
        for row in raw["candidates"]
        if int(row["microbatch"]) == int(raw["selected_request_microbatch"])
    ]
    if len(selected_rows) != 1:
        raise ValueError("Projection basis does not identify one selected candidate")
    throughput = candidate_throughput_from_batches(selected_rows[0])
    if not math.isclose(
        throughput,
        float(selected_rows[0]["steady_request_steps_per_second"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Selected candidate throughput is not raw-record reproducible")
    if throughput <= 0:
        raise ValueError("Projection basis has non-positive z throughput")

    measurements = raw["projection_measurements"]
    repeated_z = sum(float(row["z_objective_step_seconds"]) for row in positions.values())
    final_z = sum(float(row["final_z_reprobe_seconds"]) for row in positions.values())
    post_z = sum(
        float(row["post_z_filter_rank_diagnostics_seconds"])
        for row in positions.values()
    )
    key_seconds = sum(float(row.get("key_extraction_seconds", 0.0)) for row in positions.values())
    fixed_system_seconds = sum(
        float(row.get("system_fixed_setup_seconds", 0.0))
        for row in positions.values()
    )
    matrix_products_seconds = sum(
        float(row.get("valid_z_matrix_products_seconds", 0.0))
        for row in positions.values()
    )
    formation_seconds = sum(
        float(row.get("system_formation_seconds", 0.0)) for row in positions.values()
    )
    factorization_seconds = sum(
        float(row.get("solve_factorization_diagnostics_seconds", 0.0))
        for row in positions.values()
    )
    solve_seconds = sum(float(row.get("solve_diagnostic_seconds", 0.0)) for row in positions.values())
    for key, value in {
        "repeated_z_step_seconds": repeated_z,
        "final_z_probe_seconds": final_z,
        "post_z_filter_rank_diagnostics_seconds": post_z,
        "key_extraction_seconds": key_seconds,
        "system_fixed_setup_seconds": fixed_system_seconds,
        "valid_z_matrix_products_seconds": matrix_products_seconds,
        "system_formation_seconds": formation_seconds,
        "solve_factorization_diagnostics_seconds": factorization_seconds,
        "solve_diagnostic_seconds": solve_seconds,
    }.items():
        if not math.isclose(
            float(measurements[key]), value, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(f"Projection measurement disagrees with position rows: {key}")
    if not math.isclose(
        formation_seconds,
        fixed_system_seconds + matrix_products_seconds,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("System formation does not partition into fixed and row-linear work")
    if not math.isclose(
        solve_seconds,
        formation_seconds + factorization_seconds,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("Solve timing does not partition into formation and factorization")

    projected_valid_by_position = {
        position: (
            full_requests_by_position[position]
            * int(positions[str(position)]["valid_z_count"])
            / int(positions[str(position)]["request_count"])
        )
        for position in range(6)
    }
    projected_valid = sum(projected_valid_by_position.values())
    covariance_resource = raw["covariance_resource"]
    covariance_seconds_by_position = {
        int(position): float(seconds)
        for position, seconds in covariance_resource["elapsed_seconds_by_position"].items()
    }
    covariance_resource_counts = {
        int(position): int(count)
        for position, count in covariance_resource["rows_by_position"].items()
    }
    covariance_formal_counts = {
        int(position): int(count)
        for position, count in raw["full_universe"]["covariance_counts_by_position"].items()
    }
    if set(covariance_seconds_by_position) != set(range(6)):
        raise ValueError("Projection basis lacks position-specific covariance timing")
    if not math.isclose(
        sum(covariance_seconds_by_position.values()),
        float(covariance_resource["elapsed_seconds"]),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("Position covariance timing does not sum to its total")
    resource_checkpoints = {
        int(position): tuple(int(value) for value in checkpoints)
        for position, checkpoints in config["sweep"][
            "resource_covariance_convergence_checkpoints_by_position"
        ].items()
    }
    formal_checkpoints = {
        position: tuple(config["sweep"]["formal_covariance_convergence_checkpoints"])
        for position in range(6)
    }
    resource_equivalents = convergence_row_equivalents(
        resource_checkpoints, covariance_resource_counts
    )
    formal_equivalents = convergence_row_equivalents(
        formal_checkpoints, covariance_formal_counts
    )
    if (
        resource_equivalents
        != int(covariance_resource["resource_convergence_row_equivalents"])
        or formal_equivalents
        != int(covariance_resource["formal_convergence_row_equivalents"])
    ):
        raise ValueError("Covariance convergence row-equivalent declaration drifted")
    generation = raw["generation_resource_probe"]
    generation_events = int(generation["events"])
    if generation_events != int(config["sweep"]["generation_resource_events"]):
        raise ValueError("Generation projection basis does not cover the frozen event count")
    base_per_event = float(generation["base_elapsed_seconds"]) / generation_events
    edited_per_event = float(generation["edited_elapsed_seconds"]) / generation_events
    base_plus_edited_per_event = (
        float(generation["base_elapsed_seconds"])
        + float(generation["edited_elapsed_seconds"])
    ) / generation_events
    for key, value in {
        "base_seconds_per_event": base_per_event,
        "edited_seconds_per_event": edited_per_event,
        "base_plus_edited_seconds_per_event": base_plus_edited_per_event,
    }.items():
        if not math.isclose(
            float(generation[key]), value, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(f"Generation derived timing drifted: {key}")
    fixed_trigger = float(measurements["trigger_contract_seconds"]) + max(
        0.0,
        float(raw["position_contract_seconds"])
        - repeated_z
        - final_z
        - post_z
        - key_seconds
        - solve_seconds,
    )
    components = {
        "full_context_and_request_manifest": float(measurements["context_build_seconds"]),
        "full_z_optimization": (
            full_requests * int(config["frozen_workload"]["z_steps"]) / throughput
        ),
        "full_position_covariance": sum(
            covariance_seconds_by_position[position]
            * covariance_formal_counts[position]
            / covariance_resource_counts[position]
            for position in range(6)
        ),
        "formal_covariance_convergence_diagnostics": (
            float(covariance_resource["convergence_elapsed_seconds"])
            * formal_equivalents
            / resource_equivalents
        ),
        "full_final_z_reprobe_diagnostics": sum(
            float(positions[str(position)]["final_z_reprobe_seconds"])
            * full_requests_by_position[position]
            / int(positions[str(position)]["request_count"])
            for position in range(6)
        ),
        "full_post_z_filter_and_rank_diagnostics": sum(
            float(
                positions[str(position)][
                    "post_z_filter_rank_diagnostics_seconds"
                ]
            )
            * full_requests_by_position[position]
            / int(positions[str(position)]["request_count"])
            for position in range(6)
        ),
        "full_request_key_extraction": sum(
            float(positions[str(position)].get("key_extraction_seconds", 0.0))
            * full_requests_by_position[position]
            / int(positions[str(position)]["request_count"])
            for position in range(6)
        ),
        "projected_valid_z_matrix_products": sum(
            float(
                positions[str(position)].get(
                    "valid_z_matrix_products_seconds", 0.0
                )
            )
            * projected_valid_by_position[position]
            / int(positions[str(position)]["valid_z_count"])
            for position in range(6)
        ),
        "six_position_system_fixed_setup": (
            fixed_system_seconds * 6 / sum(
                int(row["valid_z_count"]) > 0 for row in positions.values()
            )
        ),
        "six_position_solve_factorization_and_diagnostics": (
            factorization_seconds * 6 / solved_positions
        ),
        "aggregation_and_trigger_contract": fixed_trigger,
        "fixed_7435_event_item_disjoint_admission": (
            edited_per_event
            * int(config["sweep"]["formal_item_disjoint_admission_events"])
        ),
        "fixed_512_event_warm_preservation_base_plus_edit": (
            base_plus_edited_per_event
            * int(config["sweep"]["formal_warm_preservation_events"])
        ),
    }
    basis = {
        "resource_valid_z_count": float(resource_valid),
        "resource_request_count": float(resource_requests),
        "projected_valid_z_count": float(projected_valid),
        "projected_valid_z_count_by_position": {
            str(position): float(value)
            for position, value in projected_valid_by_position.items()
        },
    }
    return components, basis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from experiment.phase16.protocol.genrecedit_inspired import (
        GRIDGE_METHOD_NAME,
        GRIDGE_RIDGE_RULE,
        GRIDGE_SOLVE_VARIANT,
        condition_targeted_ridge_value,
        validate_gridge_method_config,
    )

    gridge_method = (
        validate_gridge_method_config(config) if "method" in config else None
    )
    ridge_enabled = gridge_method is not None
    required_contract_keys = (
        GRIDGE_REQUIRED_CONTRACT_KEYS if ridge_enabled else REQUIRED_CONTRACT_KEYS
    )
    expected_raw_verdict = (
        "PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP_RAW"
        if ridge_enabled
        else "PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP_RAW"
    )
    output = ROOT / config["output_dir"]
    raw_path = output / "resource_sweep_summary.json"
    if not raw_path.is_file():
        raise SystemExit("Missing S16-3 raw resource sweep summary")
    if (output / "summary.json").exists():
        raise SystemExit("Refusing to overwrite finalized S16-3 resource artifacts")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    expected = config["frozen_workload"]
    counts = raw.get("full_universe", {})
    candidates = raw.get("candidates", [])
    candidate_microbatches = [row.get("microbatch") for row in candidates]
    candidate_hashes = {row.get("candidate_subset_sha256") for row in candidates}
    contract_checks = raw.get("contract_checks", {})
    maximum_peak = config["sweep"].get(
        "maximum_candidate_peak_reserved_mib",
        config["sweep"]["maximum_eligible_peak_reserved_mib"],
    )
    recomputed_microbatch = expected_selected_microbatch(candidates, maximum_peak)
    expected_opened = {
        spec["path"]
        for spec in config["inputs"].values()
        if isinstance(spec, dict) and "sha256" in spec
    } | {
        config["inputs"]["official_genrecedit"]["path"],
        str(config_path.relative_to(ROOT)),
        "experiment/phase16/configs/stage16_s1_data_resource_preflight.json",
        f"hf://{config['tokenizer']['name']}@{config['tokenizer']['revision']}",
    }
    opened = raw.get("opened_files", [])
    forbidden_fragments = ("validation", "internal_dev", "held_ground_truth", "test_events")
    forbidden_opened = sorted(
        path for path in opened if any(fragment in path.lower() for fragment in forbidden_fragments)
    )
    request_manifest = raw.get("candidate_request_manifest", [])
    expected_candidate_requests_by_position = {
        str(position): int(config["sweep"]["candidate_requests_per_position"])
        for position in config["sweep"]["candidate_positions"]
    }
    expected_position_contract_requests = {
        str(position): int(count)
        for position, count in config["sweep"].get(
            "position_contract_requests_by_position",
            {
                str(position): config["sweep"]["candidate_requests_per_position"]
                for position in range(6)
            },
        ).items()
    }
    manifest_payload = [
        {key: value for key, value in row.items() if key != "row_id"}
        for row in request_manifest
        if isinstance(row, dict)
    ]
    manifest_sha = hashlib.sha256(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    row_ids_valid = len(manifest_payload) == len(request_manifest) and all(
        row.get("row_id")
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in row.items() if key != "row_id"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for row in request_manifest
    )
    position_request_manifest = raw.get("position_contract_request_manifest", [])
    position_manifest_payload = [
        {key: value for key, value in row.items() if key != "row_id"}
        for row in position_request_manifest
        if isinstance(row, dict)
    ]
    position_manifest_sha = hashlib.sha256(
        json.dumps(
            position_manifest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    position_row_ids_valid = len(position_manifest_payload) == len(
        position_request_manifest
    ) and all(
        row.get("row_id")
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in row.items() if key != "row_id"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for row in position_request_manifest
    )
    covariance_resource = raw.get("covariance_resource", {})
    expected_rank_capacity = {
        position: int(config["sweep"]["covariance_rows_by_position"][position])
        + expected_position_contract_requests[position]
        for position in expected_position_contract_requests
    }
    convergence = covariance_resource.get("convergence", {})
    convergence_exact = (
        covariance_resource.get("rows_by_position")
        == config["sweep"]["covariance_rows_by_position"]
        and covariance_resource.get("formal_convergence_checkpoints")
        == config["sweep"]["formal_covariance_convergence_checkpoints"]
        and set(convergence)
        == set(config["sweep"]["resource_covariance_convergence_checkpoints_by_position"])
    )
    if convergence_exact:
        for position, expected_checkpoints in config["sweep"][
            "resource_covariance_convergence_checkpoints_by_position"
        ].items():
            rows = convergence[position]
            values = [
                row.get("relative_frobenius_drift_to_largest_resource_checkpoint")
                for row in rows
            ]
            convergence_exact = convergence_exact and (
                [row.get("rows") for row in rows] == expected_checkpoints
                and all(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in values
                )
                and float(values[-1]) == 0.0
            )
    projection = raw.get("formal_projection", {})
    projection_components = projection.get("component_seconds", {})
    expected_projection_components = {
        "full_context_and_request_manifest",
        "full_z_optimization",
        "full_position_covariance",
        "formal_covariance_convergence_diagnostics",
        "full_final_z_reprobe_diagnostics",
        "full_post_z_filter_and_rank_diagnostics",
        "full_request_key_extraction",
        "projected_valid_z_matrix_products",
        "six_position_system_fixed_setup",
        "six_position_solve_factorization_and_diagnostics",
        "aggregation_and_trigger_contract",
        "fixed_7435_event_item_disjoint_admission",
        "fixed_512_event_warm_preservation_base_plus_edit",
    }
    projection_values = list(projection_components.values())
    try:
        recomputed_components, recomputed_basis = recompute_projection(config, raw)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        recomputed_components, recomputed_basis = {}, {}
    component_formulas_exact = (
        set(recomputed_components) == expected_projection_components
        and set(projection_components) == expected_projection_components
        and all(
            math.isclose(
                float(projection_components[key]),
                float(recomputed_components[key]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            for key in expected_projection_components
        )
        and all(
            math.isclose(
                float(projection.get(key, math.inf)),
                value,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            for key, value in recomputed_basis.items()
            if key != "projected_valid_z_count_by_position"
        )
        and set(projection.get("projected_valid_z_count_by_position", {}))
        == set(recomputed_basis.get("projected_valid_z_count_by_position", {}))
        and all(
            math.isclose(
                float(projection["projected_valid_z_count_by_position"][position]),
                float(
                    recomputed_basis["projected_valid_z_count_by_position"][position]
                ),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            for position in recomputed_basis.get(
                "projected_valid_z_count_by_position", {}
            )
        )
    )
    key_extraction_contract_exact = (
        projection.get("key_extraction_batch_policy")
        == config["sweep"].get("key_extraction_batch_policy")
        == "selected_z_microbatch"
        and projection.get("key_extraction_layer_policy")
        == config["sweep"].get("key_extraction_layer_policy")
        == "position_selected_layer_only_output_equivalent_to_unused_official_key_bank_elision"
        and projection.get("key_extraction_batch_size")
        == raw.get("selected_request_microbatch")
        and all(
            row.get("key_extraction_batch_size")
            == raw.get("selected_request_microbatch")
            and row.get("key_extraction_layer")
            == int(position) % 4
            for position, row in raw.get("position_diagnostics", {}).items()
        )
    )
    projection_exact = (
        projection.get("projection_objective_complete") is True
        and set(projection_components) == expected_projection_components
        and component_formulas_exact
        and key_extraction_contract_exact
        and all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in projection_values
        )
        and math.isclose(
            sum(map(float, projection_values)),
            float(projection.get("measured_core_seconds", math.inf)),
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and math.isclose(
            float(projection.get("lower_wall_seconds", math.inf)),
            float(projection["measured_core_seconds"])
            * config["sweep"]["runtime_projection_lower_multiplier"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and math.isclose(
            float(projection.get("upper_wall_seconds", math.inf)),
            float(projection["measured_core_seconds"])
            * config["sweep"]["runtime_projection_upper_multiplier"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and raw.get("generation_resource_probe", {}).get("timer_scope")
        == "tokenization_context_transfer_and_generation"
        and math.isclose(
            float(projection.get("lower_gpu_hours", math.inf)),
            float(projection["lower_wall_seconds"]) / 3600.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(projection.get("upper_gpu_hours", math.inf)),
            float(projection["upper_wall_seconds"]) / 3600.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and int(projection.get("minimum_free_mib_per_gpu", -1))
        == max(
            8192,
            int(
                math.ceil(
                    (
                        float(raw["maximum_peak_reserved_mib"])
                        + max(
                            4096.0,
                            0.5 * float(raw["maximum_peak_reserved_mib"]),
                        )
                    )
                    / 1024.0
                )
                * 1024
            ),
        )
        and math.isclose(
            float(projection.get("expected_peak_reserved_mib_per_gpu", math.inf)),
            float(raw["maximum_peak_reserved_mib"]),
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and projection.get("gpu_count") == 1
        and isinstance(projection.get("cpu_ram_peak_mib"), (int, float))
        and math.isfinite(float(projection["cpu_ram_peak_mib"]))
        and float(projection["cpu_ram_peak_mib"]) > 0.0
        and projection.get("disk_reservation_mib") == 32768
        and projection.get("hard_timeout_seconds") == 604800
    )
    dataset_artifact = raw.get("request_dataset_artifact", {})
    dataset_manifest_path = ROOT / dataset_artifact.get("manifest_path", "__missing__")
    dataset_checkpoint_path = ROOT / dataset_artifact.get("checkpoint_path", "__missing__")
    dataset_artifact_exact = (
        dataset_manifest_path.is_file()
        and dataset_checkpoint_path.is_file()
        and sha256(dataset_manifest_path) == dataset_artifact.get("manifest_sha256")
        and sha256(dataset_checkpoint_path) == dataset_artifact.get("checkpoint_sha256")
        and dataset_artifact.get("completed_shards", 0) > 0
    )
    execution_identity = raw.get("execution_identity", {})
    execution_artifact = raw.get("execution_identity_artifact", {})
    execution_identity_path = ROOT / execution_artifact.get("path", "__missing__")
    current_code_hashes = {
        path: sha256(ROOT / path) for path in EXECUTED_CODE_PATHS
    }
    expected_config_relative = str(config_path.relative_to(ROOT))
    execution_identity_exact = (
        set(execution_identity)
        == {"captured_at_utc", "config_path", "config_sha256", "code_sha256"}
        and isinstance(execution_identity.get("captured_at_utc"), str)
        and execution_identity.get("config_path") == expected_config_relative
        and execution_identity.get("config_sha256") == sha256(config_path)
        and execution_identity.get("code_sha256") == current_code_hashes
        and execution_identity_path.is_file()
        and not execution_identity_path.is_symlink()
        and sha256(execution_identity_path) == execution_artifact.get("sha256")
        and json.loads(execution_identity_path.read_text(encoding="utf-8"))
        == execution_identity
    )
    frozen_inputs_unchanged = all(
        (ROOT / spec["path"]).is_file()
        and not (ROOT / spec["path"]).is_symlink()
        and sha256(ROOT / spec["path"]) == spec["sha256"]
        for spec in config["inputs"].values()
        if isinstance(spec, dict) and "sha256" in spec
    )
    resolved_labels = (
        "train_sequences",
        "split_manifest",
        "retained_warm_items",
        "pseudo_cold_items",
        "cold_items",
        "lexical_paths",
        "metadata",
        "content_embeddings",
    )
    try:
        s1_config = json.loads(
            (ROOT / config["inputs"]["s1_preflight_config"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        expected_s1_contract = {
            "preflight_config": config["inputs"]["s1_preflight_config"],
            "files": {label: config["inputs"][label] for label in resolved_labels},
            "counts": {
                "targets": int(config["frozen_workload"]["edit_targets"]),
                "contexts": int(config["frozen_workload"]["contexts"]),
                "requests": int(
                    config["frozen_workload"]["prefix_next_token_requests"]
                ),
            },
            "maximum_history_items": int(
                s1_config["split_policy"]["maximum_history_items"]
            ),
            "pass": True,
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        expected_s1_contract = None
    s1_resolution_exact = (
        expected_s1_contract is not None
        and raw.get("s1_resolved_input_contract") == expected_s1_contract
    )
    position_diagnostics = raw.get("position_diagnostics", {})
    solve_diagnostics_exact = set(position_diagnostics) == {str(i) for i in range(6)}
    if solve_diagnostics_exact:
        if ridge_enabled:
            for row in position_diagnostics.values():
                try:
                    expected_ridge = condition_targeted_ridge_value(
                        min_eigenvalue=float(row["unregularized_min_eigenvalue"]),
                        max_eigenvalue=float(row["unregularized_max_eigenvalue"]),
                        max_abs_eigenvalue=float(
                            row["unregularized_max_abs_eigenvalue"]
                        ),
                        target_condition=float(
                            gridge_method["target_condition_number"]
                        ),
                        safety_margin=float(gridge_method["ridge_safety_margin"]),
                    )
                except (KeyError, TypeError, ValueError):
                    solve_diagnostics_exact = False
                    break
                solve_diagnostics_exact = solve_diagnostics_exact and (
                    row.get("valid_z_count", 0) > 0
                    and row.get("solve_completed") is True
                    and row.get("method_name") == GRIDGE_METHOD_NAME
                    and row.get("method_family") == "GenRecEdit-inspired"
                    and row.get("faithful_reproduction") is False
                    and row.get("solve_variant") == GRIDGE_SOLVE_VARIANT
                    and row.get("ridge_added") is True
                    and row.get("ridge_rule") == GRIDGE_RIDGE_RULE
                    and math.isclose(
                        float(row.get("ridge_value", math.inf)),
                        expected_ridge,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and row.get("target_condition")
                    == gridge_method["target_condition_number"]
                    and row.get("safety_margin")
                    == gridge_method["ridge_safety_margin"]
                    and row.get("regularized_rank")
                    == config["sweep"]["linear_system_width"]
                    and row.get("regularized_nullity") == 0
                    and row.get("system_rank")
                    == config["sweep"]["linear_system_width"]
                    and row.get("regularized_system_cholesky_info") == 0
                    and isinstance(row.get("regularized_condition"), (int, float))
                    and math.isfinite(float(row["regularized_condition"]))
                    and float(row["regularized_condition"])
                    <= float(gridge_method["target_condition_number"])
                    * (1.0 + 1e-9)
                    and isinstance(row.get("solve_relative_residual"), (int, float))
                    and math.isfinite(float(row["solve_relative_residual"]))
                    and float(row["solve_relative_residual"])
                    <= config["sweep"]["maximum_solve_relative_residual"]
                    and row.get("pseudoinverse_used") is False
                    and row.get("jitter_fallback_used") is False
                    and row.get("outcome_resampling_used") is False
                    and "solve_error" not in row
                )
        else:
            solve_diagnostics_exact = all(
                row.get("valid_z_count", 0) > 0
                and row.get("solve_completed") is True
                and row.get("system_rank") == config["sweep"]["linear_system_width"]
                and int(row.get("covariance_rank", 0))
                + int(row.get("valid_key_rank", 0))
                >= config["sweep"]["linear_system_width"]
                and row.get("valid_key_rank_method")
                == "symmetric_key_gram_eigenvalue_tolerance"
                and isinstance(row.get("valid_key_rank_tolerance"), (int, float))
                and math.isfinite(float(row["valid_key_rank_tolerance"]))
                and float(row["valid_key_rank_tolerance"]) >= 0.0
                and row.get("rank_tolerance_rule")
                == config["sweep"]["rank_tolerance_rule"]
                and isinstance(row.get("system_condition"), (int, float))
                and math.isfinite(float(row["system_condition"]))
                and float(row.get("system_min_abs_eigenvalue", 0.0)) > 0.0
                and isinstance(row.get("solve_relative_residual"), (int, float))
                and math.isfinite(float(row["solve_relative_residual"]))
                and float(row["solve_relative_residual"])
                <= config["sweep"]["maximum_solve_relative_residual"]
                and row.get("ridge_added") is False
                and row.get("pseudoinverse_used") is False
                and row.get("jitter_fallback_used") is False
                and row.get("outcome_resampling_used") is False
                and "solve_error" not in row
                for row in position_diagnostics.values()
            )
    covariance_counts = {
        position: int(count)
        for position, count in counts.get("covariance_counts_by_position", {}).items()
    }
    expected_covariance_by_rank_rule = {
        position: min(count, 2 * int(config["sweep"]["linear_system_width"]))
        for position, count in covariance_counts.items()
    }
    checks = {
        "raw_verdict_pass": raw.get("verdict") == expected_raw_verdict,
        "all_core_contract_checks_pass": set(contract_checks) == required_contract_keys
        and all(contract_checks.values()),
        "edit_target_count_frozen": counts.get("edit_targets") == expected["edit_targets"],
        "context_count_frozen": counts.get("contexts") == expected["contexts"],
        "request_count_frozen": counts.get("prefix_next_token_requests")
        == expected["prefix_next_token_requests"],
        "covariance_row_count_frozen": counts.get("covariance_rows")
        == expected["covariance_rows"],
        "covariance_convergence_mechanically_verified": convergence_exact,
        "position_contract_requests_mechanically_verified": {
            position: int(row.get("request_count", -1))
            for position, row in raw.get("position_diagnostics", {}).items()
        }
        == expected_position_contract_requests,
        "linear_system_algebraic_capacity_verified": covariance_resource.get(
            "linear_system_width"
        )
        == config["sweep"]["linear_system_width"]
        and covariance_resource.get("algebraic_rank_capacity_by_position")
        == expected_rank_capacity
        and all(
            capacity >= config["sweep"]["linear_system_width"]
            for capacity in expected_rank_capacity.values()
        ),
        (
            "inspired_ridge_solve_spectral_diagnostics_verified"
            if ridge_enabled
            else "faithful_solve_spectral_diagnostics_verified"
        ): solve_diagnostics_exact,
        "covariance_resource_rank_rule_verified": expected_covariance_by_rank_rule
        == config["sweep"]["covariance_rows_by_position"]
        and covariance_resource.get("row_selection_rule")
        == config["sweep"]["covariance_resource_rule"],
        "formal_projection_mechanically_verified": projection_exact,
        "request_dataset_artifact_verified": dataset_artifact_exact,
        "execution_time_config_and_code_identity_verified": execution_identity_exact,
        "frozen_inputs_unchanged_after_execution": frozen_inputs_unchanged,
        "s1_resolved_inputs_match_s3_frozen_inputs": s1_resolution_exact,
        "peak_within_resource_attempt_cap": float(
            raw.get("maximum_peak_reserved_mib", 1e30)
        )
        <= config["sweep"].get(
            "maximum_resource_peak_reserved_mib",
            config["sweep"]["maximum_eligible_peak_reserved_mib"],
        ),
        "fixed_gpu_resource_admission_verified": (
            config["resources"].get("fixed_physical_gpu") is None
            or int(raw.get("physical_gpu", -1))
            == int(config["resources"]["fixed_physical_gpu"])
        )
        and int(raw.get("admission_free_mib", -1))
        >= int(config["resources"]["minimum_free_mib"])
        and int(raw.get("worker_readmission_free_mib", -1))
        >= int(config["resources"]["minimum_free_mib"])
        and int(raw.get("resource_attempt_hard_timeout_seconds", -1))
        == int(config["resources"]["hard_timeout_seconds"])
        and int(raw.get("resource_attempt_expected_peak_mib", -1))
        == int(config["resources"].get("expected_peak_mib", 8192))
        == int(
            config["sweep"].get(
                "maximum_resource_peak_reserved_mib",
                config["sweep"]["maximum_eligible_peak_reserved_mib"],
            )
        ),
        "candidate_microbatches_exact": candidate_microbatches
        == config["sweep"]["candidate_request_microbatches"],
        "candidate_fixed_workload_exact": len(candidate_hashes) == 1
        and all(
            isinstance(value, str) and len(value) == 64 for value in candidate_hashes
        )
        and all(
            row.get("candidate_request_count")
            == config["sweep"]["candidate_total_cache_miss_requests"]
            for row in candidates
        )
        and all(
            row.get("candidate_requests_by_position")
            == expected_candidate_requests_by_position
            for row in candidates
        ),
        "candidate_request_manifest_exact": len(request_manifest)
        == config["sweep"]["candidate_total_cache_miss_requests"]
        and row_ids_valid
        and manifest_sha == raw.get("selected_candidate_subset_sha256")
        and all(
            len(
                {
                    row["cold_item"]
                    for row in request_manifest
                    if row["position"] == position
                }
            )
            == config["sweep"]["candidate_requests_per_position"]
            for position in config["sweep"]["candidate_positions"]
        ),
        "position_contract_request_manifest_exact": len(position_request_manifest)
        == sum(expected_position_contract_requests.values())
        and position_row_ids_valid
        and position_manifest_sha == raw.get("position_contract_subset_sha256")
        and all(
            len(
                {
                    row["cold_item"]
                    for row in position_request_manifest
                    if str(row["position"]) == position
                }
            )
            == expected_count
            for position, expected_count in expected_position_contract_requests.items()
        ),
        "selection_rule_recomputed": recomputed_microbatch is not None
        and raw.get("selected_request_microbatch") == recomputed_microbatch,
        "full_30_step_path_exercised": raw.get("z_steps_per_candidate")
        == expected["z_steps"]
        and observed_full_30_step_path(raw.get("independent_full_lifecycle_probe", {})),
        "declared_external_inputs_exact_allowlist": isinstance(opened, list)
        and set(opened) == expected_opened
        and len(opened) == len(set(opened)),
        "declared_external_input_scope_honest": raw.get("declared_external_input_scope")
        == (
            "explicit frozen data/config/source/tokenizer identities only; generated "
            "request shards are output artifacts covered by their manifest SHA; this is "
            "not an OS-level syscall open audit"
        ),
        "tokenizer_provenance_frozen": raw.get("tokenizer_provenance", {}).get(
            "revision"
        )
        == config["tokenizer"]["revision"]
        and raw.get("tokenizer_provenance", {}).get("vocabulary_sha256")
        == config["tokenizer"]["vocabulary_sha256"]
        and raw.get("tokenizer_provenance", {}).get("sentencepiece_sha256")
        == config["tokenizer"]["sentencepiece_sha256"],
        "forbidden_files_absent": not forbidden_opened,
        "test_read_false": raw.get("test_read") is False,
        "validation_used_false": raw.get("validation_used") is False,
        "efficacy_metric_absent": raw.get("scientific_efficacy_metric_produced") is False,
        "base_checkpoint_unchanged": raw.get("base_checkpoint_unchanged") is True,
        "automatic_retry_false": config.get("automatic_retry") is False
        and raw.get("automatic_retry") is False,
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise SystemExit(f"S16-3 resource artifact contract failed: {failed}")

    generated = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": generated,
        "test_read": False,
        "validation_used": False,
        "automatic_retry": False,
        "method": raw.get("method"),
    }
    input_files = {
        spec["path"]: spec["sha256"]
        for spec in config["inputs"].values()
        if isinstance(spec, dict) and "sha256" in spec
    }
    code_hashes = execution_identity["code_sha256"]
    write_json(output / "config.json", config)
    write_json(
        output / "input_file_sha256.json",
        {
            **common,
            "files": input_files,
            "source_commits": {
                config["inputs"]["official_genrecedit"]["path"]: config["inputs"]
                ["official_genrecedit"]["commit"]
            },
        },
    )
    write_json(
        output / "code_sha256.json",
        {
            **common,
            "config_path": execution_identity["config_path"],
            "config_sha256": execution_identity["config_sha256"],
            "captured_at_utc": execution_identity["captured_at_utc"],
            "files": code_hashes,
        },
    )
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_files": opened,
            "opened_files_allowlist": sorted(expected_opened),
            "forbidden_files_opened": forbidden_opened,
            "audit_scope": raw["declared_external_input_scope"],
            "test_read": False,
        },
    )
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "domain": config["domain"],
            "context_source": "S16-1 student-readable interaction-train sequences only",
            "context_neighbor_universe": "S16-1 retained-warm only",
            "edit_target_source": "frozen real-cold catalog; no validation/test occurrence selection",
            "item_disjoint_admission_source": "S16-1 train-derived pseudo-cold protocol only",
            "stage15_context_artifact_reused": False,
            "request_dataset_artifact": dataset_artifact,
        },
    )
    write_json(
        output / "resource_summary.json",
        {
            **common,
            "physical_gpu": raw["physical_gpu"],
            "visible_gpu": 0,
            "gpu_count": 1,
            "admission_free_mib": raw["admission_free_mib"],
            "maximum_peak_allocated_mib": raw["maximum_peak_allocated_mib"],
            "maximum_peak_reserved_mib": raw["maximum_peak_reserved_mib"],
            "elapsed_seconds": raw["elapsed_seconds"],
            "formal_projection": raw["formal_projection"],
            "tokenizer_provenance": raw["tokenizer_provenance"],
            "runtime_provenance": raw["runtime_provenance"],
        },
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": config["exact_start_command"],
            "formal_command_template": None,
            "formal_runner_status": "PENDING_RESOURCE_EVIDENCE_AND_USER_GPU_SELECTION",
            "formal_launch_authorized": False,
        },
    )
    summary = {
        **common,
        "verdict": (
            "PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP"
            if ridge_enabled
            else "PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP"
        ),
        "formal_gate": (
            "PENDING_PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION"
            if ridge_enabled
            else "PENDING_PASS_S16_3_GFULL_FAITHFUL_CONTRACT_ADMISSION"
        ),
        "scientific_efficacy_metric_produced": False,
        "checks": checks,
        "full_universe": counts,
            "selected_request_microbatch": raw["selected_request_microbatch"],
        "formal_projection": raw["formal_projection"],
        "next_action": (
            "Disclose measured resources and obtain explicit user GPU authorization "
            "before formal G-RIDGE editing/admission."
            if ridge_enabled
            else "Disclose measured resources and obtain explicit user GPU authorization before formal G-FULL editing/admission."
        ),
    }
    write_json(output / "summary.json", summary)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
