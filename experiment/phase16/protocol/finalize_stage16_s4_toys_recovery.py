#!/usr/bin/env python3
"""CPU-only recovery finalization of immutable S16-4 GPU4 a7 predictions.

The failed a7 attempt remains untouched.  This module verifies every frozen a7
input, reuses its completed rankings, applies the preregistered paired bootstrap
and Holm familywise correction, and writes a separate write-once a8 artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiment.phase15.protocol.common_adapter import (
    read_projected_sequences,
    read_validation_predictions,
)
from experiment.phase16.protocol.finalize_stage16_s4_toys import (
    exact_paired_binary_greater,
    holm_adjust,
    paired_bootstrap,
    read_arm_predictions,
    strictly_dominates,
    summarize,
    verify_formal_runtime_identity,
)
from experiment.phase16.protocol.stage16_s4_toys_validation import (
    EXPECTED_CONTROLS,
    FORMAL_ARMS,
    ROOT,
    SCIENTIFIC_ARMS,
    ranking_metrics,
    sha256_file,
    verify_regular,
)


RECOVERY_VERDICT = "PASS_S16_4_TOYS_CPU_RECOVERY_FINALIZATION"
FORMAL_VERDICT = "COMPLETED_S16_4_TOYS_STANDALONE_FROZEN_VALIDATION"
EXECUTED_CODE_PATHS = (
    "experiment/phase16/protocol/finalize_stage16_s4_toys.py",
    "experiment/phase16/protocol/finalize_stage16_s4_toys_recovery.py",
    "experiment/phase16/run_stage16_s4_toys_recovery_gpu4_a7_cpu_a8.sh",
    "experiment/phase16/tests/test_stage16_s4_toys_recovery.py",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_frozen_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    observed: dict[str, str] = {}
    for label, declaration in config["frozen_inputs"].items():
        path = ROOT / declaration["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/non-regular S16-4 recovery input: {label}")
        digest = sha256_file(path)
        if digest != declaration["sha256"]:
            raise ValueError(f"Frozen S16-4 recovery input SHA drift: {label}")
        paths[label] = path
        observed[label] = digest
    return paths, observed


def _verify_source_attempt(
    recovery: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    source_contract = recovery["source_attempt"]
    source_root = ROOT / source_contract["output_dir"]
    source_config = load_json(paths["source_config"])
    source_status = load_json(paths["source_status"])
    runtime_manifest = load_json(paths["source_runtime_manifest"])
    run_log = paths["source_run_log"].read_text(encoding="utf-8")
    missing_results = [
        name
        for name in ("summary.json", "event_metrics.jsonl", "artifact_contract.json")
        if not (source_root / name).exists()
    ]
    expected_absent = {
        "summary.json": source_contract["source_summary_must_be_absent"],
        "event_metrics.jsonl": source_contract["source_event_metrics_must_be_absent"],
        "artifact_contract.json": source_contract["source_artifact_contract_must_be_absent"],
    }
    if any(expected_absent.values()) and set(missing_results) != {
        name for name, required_absent in expected_absent.items() if required_absent
    }:
        raise ValueError("Immutable a7 result-absence contract drift")
    if (
        source_config.get("attempt_id") != source_contract["attempt_id"]
        or source_config.get("output_dir") != source_contract["output_dir"]
        or source_status.get("attempt_id") != source_contract["attempt_id"]
        or source_status.get("status") != source_contract["terminal_status"]
        or source_status.get("status_code") != source_contract["terminal_status_code"]
        or source_status.get("exit_code") != source_contract["terminal_exit_code"]
        or source_status.get("progress_current") != source_contract["progress_current"]
        or source_status.get("progress_total") != source_contract["progress_total"]
        or source_status.get("process_alive") is not False
        or source_status.get("repeat_started") is not False
        or source_status.get("test_read") is not False
        or source_status.get("automatic_retry") is not False
        or runtime_manifest.get("schema_version")
        != source_contract["expected_runtime_schema"]
        or "Formal-root isolated-runtime identity drift" not in run_log
    ):
        raise ValueError("Immutable a7 failure-lineage contract drift")
    for arm in FORMAL_ARMS:
        marker = f"[s16-s4-{arm}] events=8789/8789"
        if marker not in run_log:
            raise ValueError(f"Immutable a7 completion marker missing: {arm}")
    verify_formal_runtime_identity(paths["source_config"], source_config, source_root)
    return source_config, source_status, source_root


def _build_events(
    source_config: Mapping[str, Any],
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    preflight_path = verify_regular(
        ROOT, source_config["preflight"]["config"], "preflight_config"
    )
    preflight = load_json(preflight_path)
    projected_path = verify_regular(
        ROOT,
        preflight["inputs"]["projected_train_validation_sequences"],
        "projected_sequences",
    )
    frozen_path = verify_regular(
        ROOT,
        preflight["inputs"]["frozen_f0_r2_validation_predictions"],
        "frozen_f0_r2",
    )
    projected = read_projected_sequences(projected_path)
    frozen = read_validation_predictions(frozen_path)
    users = set(projected)
    if users != set(frozen) or len(users) != source_config["validation_events"]:
        raise ValueError("S16-4 recovery validation universe drift")

    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    arm_sha: dict[str, Any] = {}
    for arm in FORMAL_ARMS:
        arm_root = source_root / "arms" / arm
        prediction_path = arm_root / "predictions_validation.jsonl"
        summary_path = arm_root / "summary.json"
        if prediction_path.is_symlink() or summary_path.is_symlink():
            raise ValueError(f"S16-4 recovery source arm is symlinked: {arm}")
        summary = load_json(summary_path)
        expected_verdict = f"COMPLETED_S16_4_TOYS_{arm.replace('-', '_')}_FROZEN_VALIDATION"
        if (
            summary.get("attempt_id") != source_config["attempt_id"]
            or summary.get("verdict") != expected_verdict
            or summary.get("events") != source_config["validation_events"]
            or summary.get("promotion_eligible") is not True
            or summary.get("scientific_efficacy_metric_produced") is not True
            or summary.get("test_read") is not False
        ):
            raise ValueError(f"S16-4 recovery arm summary contract failed: {arm}")
        arm_rows[arm] = read_arm_predictions(prediction_path, arm, users)
        arm_summaries[arm] = summary
        arm_sha[arm] = {
            "summary": sha256_file(summary_path),
            "predictions": sha256_file(prediction_path),
        }

    events: list[dict[str, Any]] = []
    for event_index, (user, sequence) in enumerate(projected.items(), 1):
        target = sequence[-1]
        source = frozen[user]
        if str(source.get("target")) != target:
            raise ValueError(f"Frozen target linkage drift: {user}")
        rankings = {
            "F0": [str(item) for item in source["v0_top50"]],
            "R2": [str(item) for item in source["r2_top50"]],
            **{arm: arm_rows[arm][user]["top50"] for arm in FORMAL_ARMS},
        }
        events.append(
            {
                "event_index": event_index,
                "user_id": user,
                "target_item": target,
                "is_cold": bool(source["is_cold"]),
                "metrics": {
                    arm: ranking_metrics(rankings[arm], target)
                    for arm in SCIENTIFIC_ARMS
                },
            }
        )
    return events, arm_summaries, arm_sha


def _pair_hit_overlap(
    events: Sequence[Mapping[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    cold = [row for row in events if row["is_cold"]]
    treatment_only = sum(
        row["metrics"][treatment]["hit@50"] == 1
        and row["metrics"][control]["hit@50"] == 0
        for row in cold
    )
    control_only = sum(
        row["metrics"][treatment]["hit@50"] == 0
        and row["metrics"][control]["hit@50"] == 1
        for row in cold
    )
    both = sum(
        row["metrics"][treatment]["hit@50"] == 1
        and row["metrics"][control]["hit@50"] == 1
        for row in cold
    )
    return {
        "events": len(cold),
        "treatment_only_hits": int(treatment_only),
        "control_only_hits": int(control_only),
        "both_hit": int(both),
        "neither_hit": int(len(cold) - treatment_only - control_only - both),
        "oracle_union_hit@50": float((treatment_only + control_only + both) / len(cold)),
        "gate_role": "diagnostic_only_not_used_for_S16_4_promotion",
    }


def _calculate_results(
    recovery_config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    events, arm_summaries, arm_sha = _build_events(source_config, source_root)
    metrics = {
        arm: {
            subset: summarize(events, arm, subset)
            for subset in ("overall", "cold", "warm")
        }
        for arm in SCIENTIFIC_ARMS
    }

    pairs = [
        ("R2", "F0"),
        ("S-AUX", "F0"),
        ("S-AUX", "R2"),
        ("S-PLUS-CTRL", "F0"),
        ("S-PLUS", "S-PLUS-CTRL"),
        ("S-PLUS", "R2"),
        ("G-RIDGE", "F0"),
        ("G-RIDGE", "R2"),
    ]
    statistics = recovery_config["statistics"]
    if (
        statistics["paired_bootstrap_resamples"]
        != source_config["statistics"]["paired_bootstrap_resamples"]
        or statistics["paired_bootstrap_seed"]
        != source_config["statistics"]["paired_bootstrap_seed"]
    ):
        raise ValueError("Recovery bootstrap contract differs from frozen a7")
    comparisons: dict[str, Any] = {}
    seed_offset = 0
    for treatment, control in pairs:
        key = f"{treatment}_vs_{control}"
        comparisons[key] = {}
        for label, metric, subset in (
            ("cold_hit@50", "hit@50", "cold"),
            ("cold_ndcg@10", "ndcg@10", "cold"),
            ("warm_ndcg@10", "ndcg@10", "warm"),
            ("overall_ndcg@10", "ndcg@10", "overall"),
        ):
            comparisons[key][label] = paired_bootstrap(
                events,
                treatment,
                control,
                metric,
                subset,
                resamples=statistics["paired_bootstrap_resamples"],
                seed=statistics["paired_bootstrap_seed"] + seed_offset,
            )
            seed_offset += 1

    alpha = float(statistics["familywise_alpha"])
    primary_tests = {
        arm: exact_paired_binary_greater(events, arm, EXPECTED_CONTROLS[arm])
        for arm in FORMAL_ARMS
    }
    adjusted = holm_adjust(
        {arm: result["raw_p_value"] for arm, result in primary_tests.items()},
        alpha=alpha,
    )
    for arm in FORMAL_ARMS:
        primary_tests[arm].update(adjusted[arm])
    multiplicity = {
        "method": "Holm",
        "family": statistics["primary_family"],
        "family_size": len(FORMAL_ARMS),
        "alpha": alpha,
        "test": statistics["multiplicity_test"],
        "raw_bootstrap_ci_reported_separately": True,
        "primary_tests": primary_tests,
    }

    preflight = load_json(
        verify_regular(ROOT, source_config["preflight"]["config"], "preflight_config")
    )
    source_summaries = {
        "S-AUX": load_json(
            verify_regular(ROOT, preflight["inputs"]["saux_summary"], "saux_summary")
        ),
        "S-PLUS": load_json(
            verify_regular(ROOT, preflight["inputs"]["splus_summary"], "splus_summary")
        ),
        "S-PLUS-CTRL": load_json(
            verify_regular(
                ROOT, preflight["inputs"]["splus_ctrl_summary"], "splus_ctrl_summary"
            )
        ),
        "G-RIDGE": load_json(
            verify_regular(
                ROOT, preflight["inputs"]["gridge_formal_summary"], "gridge_summary"
            )
        ),
    }
    control_cost = source_config["control_cost"]
    costs: dict[str, dict[str, float]] = {
        "F0": {
            "update_seconds": 0.0,
            "inference_seconds": float(control_cost["shared_f0_r2_source_runtime_seconds"]),
            "extra_state_bytes": 0.0,
        },
        "R2": {
            "update_seconds": float(control_cost["shared_f0_r2_source_runtime_seconds"]),
            "inference_seconds": float(control_cost["shared_f0_r2_source_runtime_seconds"]),
            "extra_state_bytes": float(control_cost["r2_extra_state_bytes"]),
        },
    }
    for arm in FORMAL_ARMS:
        update_seconds = (
            source_summaries[arm].get("elapsed_seconds")
            if arm == "G-RIDGE"
            else source_summaries[arm].get("runtime_seconds")
        )
        costs[arm] = {
            "update_seconds": float(update_seconds),
            "inference_seconds": float(arm_summaries[arm]["inference_seconds"]),
            "extra_state_bytes": float(arm_summaries[arm]["extra_state_bytes"]),
        }

    vectors = {
        arm: {
            "cold_hit@50": metrics[arm]["cold"]["hit@50"],
            "warm_ndcg@10": metrics[arm]["warm"]["ndcg@10"],
            **costs[arm],
        }
        for arm in SCIENTIFIC_ARMS
    }
    gates: dict[str, Any] = {}
    for arm in FORMAL_ARMS:
        control = EXPECTED_CONTROLS[arm]
        interval = comparisons[f"{arm}_vs_{control}"]["cold_hit@50"]
        corrected = primary_tests[arm]
        cold_signal = interval["ci_low"] > 0 and corrected["reject_at_alpha"]
        dominators = [
            comparator
            for comparator in dict.fromkeys((control, "R2"))
            if comparator != arm and strictly_dominates(vectors[comparator], vectors[arm])
        ]
        if not cold_signal:
            label = "FAIL_STANDALONE"
        elif not dominators:
            label = "PASS_STANDALONE_PARETO"
        else:
            label = "PASS_STANDALONE_COLD_SIGNAL"
        gates[arm] = {
            "label": label,
            "correct_control": control,
            "cold_signal": cold_signal,
            "cold_gain_interval": interval,
            "multiplicity_corrected_test": corrected,
            "strict_dominators": dominators,
            "vector": vectors[arm],
        }

    complementarity = {
        f"{left}_vs_{right}": _pair_hit_overlap(events, left, right)
        for left, right in (
            ("S-AUX", "F0"),
            ("S-AUX", "R2"),
            ("S-PLUS", "R2"),
            ("G-RIDGE", "F0"),
            ("G-RIDGE", "R2"),
            ("G-RIDGE", "S-AUX"),
        )
    }
    mechanisms = {
        arm: {
            "mechanism_totals": arm_summaries[arm].get("mechanism_totals", {}),
            "runtime_seconds": arm_summaries[arm]["runtime_seconds"],
            "peak_cuda_reserved_mib": arm_summaries[arm]["peak_cuda_reserved_mib"],
        }
        for arm in FORMAL_ARMS
    }
    return {
        "events": events,
        "metrics": metrics,
        "paired_bootstrap": comparisons,
        "multiplicity": multiplicity,
        "standalone_gates": gates,
        "costs": costs,
        "mechanisms": mechanisms,
        "cold_hit_complementarity_diagnostic": complementarity,
        "arm_artifact_sha256": arm_sha,
        "cost_comparability_note": control_cost["comparability_note"],
    }


def finalize(config_path: Path) -> dict[str, Any]:
    started_at = utc_now()
    absolute_config = config_path if config_path.is_absolute() else ROOT / config_path
    if absolute_config.is_symlink() or not absolute_config.is_file():
        raise ValueError("S16-4 recovery config must be a regular file")
    recovery_config = load_json(absolute_config)
    if recovery_config.get("schema_version") != "stage16_s4_toys_recovery_v1":
        raise ValueError("Unexpected S16-4 recovery schema")
    output = ROOT / recovery_config["output_dir"]
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite an existing S16-4 recovery attempt root")
    if (
        recovery_config["resources"].get("cpu_only") is not True
        or recovery_config["resources"].get("gpu_count") != 0
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("S16-4 recovery must be CPU-only with CUDA_VISIBLE_DEVICES empty")

    paths, input_hashes_before = _load_frozen_inputs(recovery_config)
    source_config, source_status, source_root = _verify_source_attempt(
        recovery_config, paths
    )
    code_hashes_before = {
        relative: sha256_file(ROOT / relative) for relative in EXECUTED_CODE_PATHS
    }
    calculation = _calculate_results(recovery_config, source_config, source_root)
    _, input_hashes_after = _load_frozen_inputs(recovery_config)
    code_hashes_after = {
        relative: sha256_file(ROOT / relative) for relative in EXECUTED_CODE_PATHS
    }
    if input_hashes_after != input_hashes_before:
        raise ValueError("Frozen a7 inputs changed during CPU recovery")
    if code_hashes_after != code_hashes_before:
        raise ValueError("S16-4 recovery code changed during finalization")

    output.mkdir(parents=True, exist_ok=False)
    config_copy = output / "config.json"
    with config_copy.open("xb") as handle:
        handle.write(absolute_config.read_bytes())
    identity = {
        "captured_at_utc": utc_now(),
        "config_path": str(absolute_config.relative_to(ROOT)),
        "config_sha256": sha256_file(absolute_config),
        "code_sha256": code_hashes_before,
        "source_input_sha256_before": input_hashes_before,
        "source_input_sha256_after": input_hashes_after,
        "cpu_only": True,
        "cuda_visible_devices": "",
    }
    identity_path = output / "execution_identity.json"
    write_json(identity_path, identity)
    recovery_manifest = {
        "verdict": RECOVERY_VERDICT,
        "source_attempt_id": source_config["attempt_id"],
        "source_terminal_status": source_status["status"],
        "source_terminal_status_code": source_status["status_code"],
        "source_attempt_preserved": True,
        "gpu_scientific_inference_recompute": False,
        "derived_statistical_finalization_from_frozen_predictions": True,
        "failure_repaired": recovery_config["source_attempt"]["expected_failure"],
        "multiplicity_gap_repaired": True,
        "test_read": False,
        "automatic_retry": False,
    }
    write_json(output / "recovery_manifest.json", recovery_manifest)
    write_json(
        output / "command_manifest.json",
        {"exact_start_command": recovery_config["exact_start_command"]},
    )
    write_json(output / "input_file_sha256.json", input_hashes_before)

    event_metrics_path = output / "event_metrics.jsonl"
    with event_metrics_path.open("x", encoding="utf-8") as handle:
        for row in calculation.pop("events"):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema_version": recovery_config["schema_version"],
        "experiment_id": recovery_config["experiment_id"],
        "attempt_id": recovery_config["attempt_id"],
        "status": "COMPLETED",
        "verdict": RECOVERY_VERDICT,
        "formal_verdict": FORMAL_VERDICT,
        "generated_at_utc": utc_now(),
        "source_attempt_id": source_config["attempt_id"],
        "source_attempt_terminal_status": "FAILED_PRESERVED",
        "events": source_config["validation_events"],
        **calculation,
        "event_metrics_sha256": sha256_file(event_metrics_path),
        "source_config_sha256": input_hashes_before["source_config"],
        "source_runtime_manifest_sha256": input_hashes_before["source_runtime_manifest"],
        "execution_identity_artifact": {
            "path": str(identity_path.relative_to(ROOT)),
            "sha256": sha256_file(identity_path),
        },
        "gpu_used": False,
        "gpu_scientific_inference_recompute": False,
        "derived_statistical_finalization_from_frozen_predictions": True,
        "validation_used_for_tuning_or_state_selection": False,
        "scientific_efficacy_metric_produced": True,
        "test_read": False,
        "automatic_retry": False,
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    status = {
        "experiment_id": recovery_config["experiment_id"],
        "attempt_id": recovery_config["attempt_id"],
        "status": "COMPLETED",
        "status_code": RECOVERY_VERDICT,
        "stage": "finished",
        "reason": (
            "CPU-only finalization completed from immutable GPU4 a7 predictions; "
            "a7 remains FAILED and no GPU inference was rerun."
        ),
        "started_at": started_at,
        "updated_at": utc_now(),
        "process_alive": False,
        "gpu_count": 0,
        "gpu_used": False,
        "progress_current": 1,
        "progress_total": 1,
        "progress_unit": "recovery_finalization",
        "exit_code": 0,
        "source_attempt_status": "FAILED_PRESERVED",
        "scientific_efficacy_metric_produced": True,
        "test_read": False,
        "automatic_retry": False,
        "automatic_resume": False,
        "exact_start_command": recovery_config["exact_start_command"],
        "output_dir": recovery_config["output_dir"],
        "summary_path": str(summary_path.relative_to(ROOT)),
    }
    status_path = output / "status.json"
    write_json(status_path, status)
    artifact_contract = {
        "verdict": "PASS_S16_4_TOYS_RECOVERY_ARTIFACT_CONTRACT",
        "required_local": [
            "status.json",
            "config.json",
            "execution_identity.json",
            "recovery_manifest.json",
            "command_manifest.json",
            "input_file_sha256.json",
            "summary.json",
            "event_metrics.jsonl",
        ],
        "source_attempt_id": source_config["attempt_id"],
        "source_artifact_sha256": input_hashes_before,
        "source_attempt_preserved": True,
        "local_sha256": {
            "status.json": sha256_file(status_path),
            "config.json": sha256_file(config_copy),
            "execution_identity.json": sha256_file(identity_path),
            "recovery_manifest.json": sha256_file(output / "recovery_manifest.json"),
            "command_manifest.json": sha256_file(output / "command_manifest.json"),
            "input_file_sha256.json": sha256_file(output / "input_file_sha256.json"),
            "summary.json": sha256_file(summary_path),
            "event_metrics.jsonl": sha256_file(event_metrics_path),
        },
    }
    write_json(output / "artifact_contract.json", artifact_contract)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = finalize(args.config)
    print(result["verdict"])
    for arm, gate in result["standalone_gates"].items():
        print(f"{arm}: {gate['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
