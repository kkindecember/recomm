#!/usr/bin/env python3
"""Finalize the bounded S16-3 G-FULL resource sweep artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

REQUIRED_CONTRACT_KEYS = {
    "full_universe_counts_match",
    "train_only_zero_leakage",
    "all_candidate_semantics_pass",
    "candidate_workload_identical",
    "all_positions_exercised",
    "valid_failed_counts_complete",
    "covariance_position_coverage_exact",
    "long_position_resource_rows_present",
    "covariance_resource_allocation_exact",
    "covariance_convergence_report_complete",
    "formal_cache_empty",
    "isolated_cache_probe_pass",
    "valid_z_filter_complete",
    "solve_aggregate_trigger_exercised_if_valid",
    "base_parameter_parity_after_trigger",
    "base_checkpoint_unchanged",
    "peak_within_small_experiment_cap",
}

REQUIRED_CANDIDATE_SEMANTIC_KEYS = {
    "formal_cache_empty",
    "identical_fixed_request_count",
    "full_30_step_budget_configured",
    "full_30_step_path_observed",
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


def expected_selected_microbatch(
    candidates: list[dict[str, Any]], maximum_peak_mib: float
) -> int | None:
    eligible = []
    for row in candidates:
        semantics = row.get("semantic_checks", {})
        recomputed = (
            set(semantics) == REQUIRED_CANDIDATE_SEMANTIC_KEYS
            and all(semantics.values())
            and float(row.get("peak_reserved_mib", 1e30)) <= maximum_peak_mib
        )
        if not isinstance(row.get("eligible"), bool) or bool(row["eligible"]) != recomputed:
            return None
        if recomputed:
            eligible.append(row)
    if not eligible:
        return None
    best = max(eligible, key=lambda row: float(row["steady_request_steps_per_second"]))
    near = [
        row
        for row in eligible
        if float(row["steady_request_steps_per_second"])
        >= 0.98 * float(best["steady_request_steps_per_second"])
    ]
    return int(
        min(near, key=lambda row: (int(row["microbatch"]), float(row["peak_reserved_mib"])))
        ["microbatch"]
    )


def observed_full_30_step_path(candidates: list[dict[str, Any]]) -> bool:
    expected_trace = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    return bool(candidates) and all(
        bool(row.get("batch_records"))
        and all(
            batch.get("lifecycle_check_steps") == expected_trace
            and batch.get("observed_step_29") is True
            and int(batch.get("forward_calls", -1)) == 30
            for batch in row["batch_records"]
        )
        for row in candidates
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    maximum_peak = config["sweep"]["maximum_eligible_peak_reserved_mib"]
    recomputed_microbatch = expected_selected_microbatch(candidates, maximum_peak)
    expected_opened = {
        spec["path"]
        for spec in config["inputs"].values()
        if isinstance(spec, dict) and "sha256" in spec
    } | {
        config["inputs"]["official_genrecedit"]["path"],
        "experiment/phase16/configs/stage16_s1_data_resource_preflight.json",
    }
    opened = raw.get("opened_files", [])
    forbidden_fragments = ("validation", "internal_dev", "held_ground_truth", "test_events")
    forbidden_opened = sorted(
        path for path in opened if any(fragment in path.lower() for fragment in forbidden_fragments)
    )
    checks = {
        "raw_verdict_pass": raw.get("verdict")
        == "PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP_RAW",
        "all_core_contract_checks_pass": set(contract_checks) == REQUIRED_CONTRACT_KEYS
        and all(contract_checks.values()),
        "edit_target_count_frozen": counts.get("edit_targets") == expected["edit_targets"],
        "context_count_frozen": counts.get("contexts") == expected["contexts"],
        "request_count_frozen": counts.get("prefix_next_token_requests")
        == expected["prefix_next_token_requests"],
        "covariance_row_count_frozen": counts.get("covariance_rows")
        == expected["covariance_rows"],
        "peak_within_small_experiment_cap": float(raw.get("maximum_peak_reserved_mib", 1e30))
        <= config["sweep"]["maximum_eligible_peak_reserved_mib"],
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
        ),
        "selection_rule_recomputed": recomputed_microbatch is not None
        and raw.get("selected_request_microbatch") == recomputed_microbatch,
        "full_30_step_path_exercised": raw.get("z_steps_per_candidate")
        == expected["z_steps"]
        and observed_full_30_step_path(candidates),
        "opened_files_exact_allowlist": isinstance(opened, list)
        and set(opened) == expected_opened
        and len(opened) == len(set(opened)),
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
    }
    input_files = {
        spec["path"]: spec["sha256"]
        for spec in config["inputs"].values()
        if isinstance(spec, dict) and "sha256" in spec
    }
    code_paths = [
        Path("experiment/phase16/configs/stage16_s3_gfull_objective_resource_sweep.json"),
        Path("experiment/phase16/protocol/genrecedit_faithful.py"),
        Path("experiment/phase16/protocol/genrecedit_data.py"),
        Path("experiment/phase16/protocol/gfull_objective_resource_sweep.py"),
        Path("experiment/phase16/protocol/finalize_s3_gfull_resource_sweep.py"),
        Path("experiment/phase16/tests/test_genrecedit_faithful.py"),
        Path("experiment/phase16/tests/test_genrecedit_data.py"),
        Path("experiment/phase16/tests/test_gfull_resource_contract.py"),
        Path("experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh"),
        Path("experiment/phase15/protocol/genrecedit_gram_adapter.py"),
    ]
    code_hashes = {str(path): sha256(ROOT / path) for path in code_paths}
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
    write_json(output / "code_sha256.json", {**common, "files": code_hashes})
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_files": opened,
            "opened_files_allowlist": sorted(expected_opened),
            "forbidden_files_opened": forbidden_opened,
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
        },
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": "bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh",
            "formal_command_template": "bash experiment/phase16/run_stage16_s3_gfull_formal.sh <USER_GPU>",
            "formal_launch_authorized": False,
        },
    )
    summary = {
        **common,
        "verdict": "PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP",
        "formal_gate": "PENDING_PASS_S16_3_GFULL_FAITHFUL_CONTRACT_ADMISSION",
        "scientific_efficacy_metric_produced": False,
        "checks": checks,
        "full_universe": counts,
            "selected_request_microbatch": raw["selected_request_microbatch"],
        "formal_projection": raw["formal_projection"],
        "next_action": "Disclose measured resources and obtain explicit user GPU authorization before formal G-FULL editing/admission.",
    }
    write_json(output / "summary.json", summary)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
