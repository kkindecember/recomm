#!/usr/bin/env python3
"""CPU-only correction of the misidentified S16-4 R2 comparator.

The completed GPU4 a7 predictions and CPU a8 recovery remain immutable.  This
module replaces only the a8 ``R2`` event metrics with the preregistered
Phase-13 unconditional ``portfolio@2`` ranking, verifies the reconstruction
against the frozen Phase-13 P6 summary, and writes an independent a9 artifact.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiment.phase13.protocol.b1_portfolio_confirmation import (
    portfolio_ranking,
    unique_in_order,
)
from experiment.phase16.protocol.finalize_stage16_s4_toys import (
    exact_paired_binary_greater,
    holm_adjust,
    paired_bootstrap,
    strictly_dominates,
    summarize,
)
from experiment.phase16.protocol.finalize_stage16_s4_toys_recovery import (
    load_json,
    write_json,
)
from experiment.phase16.protocol.stage16_s4_toys_validation import (
    EXPECTED_CONTROLS,
    FORMAL_ARMS,
    ROOT,
    SCIENTIFIC_ARMS,
    ranking_metrics,
    sha256_file,
)


CORRECTION_VERDICT = "PASS_S16_4_TOYS_PORTFOLIO2_COMPARATOR_CORRECTION"
CORRECT_COMPARATOR = "stage13_v1_r2_unconditional_portfolio2"
INCORRECT_COMPARATOR = "stage13_v1_r2_toys_p0_r2_top50"
EXPECTED_EVENTS = 8789
EXPECTED_COLD_EVENTS = 4367
EXPECTED_WARM_EVENTS = 4422
EXECUTED_CODE_PATHS = (
    "experiment/phase13/protocol/b1_portfolio_confirmation.py",
    "experiment/phase15/protocol/common_adapter.py",
    "experiment/phase16/protocol/stage16_s4_toys_validation.py",
    "experiment/phase16/protocol/finalize_stage16_s4_toys.py",
    "experiment/phase16/protocol/finalize_stage16_s4_toys_recovery.py",
    "experiment/phase16/protocol/finalize_stage16_s4_portfolio2_correction.py",
    "experiment/phase16/run_stage16_s4_portfolio2_correction_a9_cpu.sh",
    "experiment/phase16/tests/test_stage16_s4_portfolio2_correction.py",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_frozen_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    observed: dict[str, str] = {}
    for label, declaration in config["frozen_inputs"].items():
        path = ROOT / declaration["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/non-regular portfolio correction input: {label}")
        digest = sha256_file(path)
        if digest != declaration["sha256"]:
            raise ValueError(f"Portfolio correction input SHA drift: {label}")
        paths[label] = path
        observed[label] = digest
    return paths, observed


def _read_jsonl_by_user(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row.get("user_id"))
            if user in rows or user == "None":
                raise ValueError(f"Duplicate/invalid {label} user at line {line_number}")
            rows[user] = row
    if len(rows) != EXPECTED_EVENTS:
        raise ValueError(f"Unexpected {label} event count: {len(rows)}")
    return rows


def _read_a8_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    users: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row.get("user_id"))
            if (
                row.get("event_index") != line_number
                or user in users
                or set(row.get("metrics", {})) != set(SCIENTIFIC_ARMS)
            ):
                raise ValueError(f"Invalid a8 event identity at line {line_number}")
            users.add(user)
            rows.append(row)
    if (
        len(rows) != EXPECTED_EVENTS
        or sum(bool(row["is_cold"]) for row in rows) != EXPECTED_COLD_EVENTS
    ):
        raise ValueError("Unexpected a8 validation universe")
    return rows


def _verify_a8_lineage(paths: Mapping[str, Path]) -> dict[str, Any]:
    summary = load_json(paths["a8_summary"])
    status = load_json(paths["a8_status"])
    contract = load_json(paths["a8_artifact_contract"])
    config = load_json(paths["a8_config"])
    if (
        summary.get("verdict") != "PASS_S16_4_TOYS_CPU_RECOVERY_FINALIZATION"
        or summary.get("status") != "COMPLETED"
        or summary.get("events") != EXPECTED_EVENTS
        or summary.get("gpu_used") is not False
        or summary.get("test_read") is not False
        or status.get("status") != "COMPLETED"
        or status.get("status_code")
        != "PASS_S16_4_TOYS_CPU_RECOVERY_FINALIZATION"
        or status.get("process_alive") is not False
        or contract.get("verdict")
        != "PASS_S16_4_TOYS_RECOVERY_ARTIFACT_CONTRACT"
        or contract.get("source_attempt_preserved") is not True
        or config.get("schema_version") != "stage16_s4_toys_recovery_v1"
    ):
        raise ValueError("Authoritative a8 recovery lineage contract drift")
    return summary


def _metric_close(left: Any, right: Any, *, tolerance: float = 1e-15) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def _verify_aggregate_against_phase13(
    events: Sequence[Mapping[str, Any]],
    p6_summary: Mapping[str, Any],
) -> dict[str, Any]:
    expected = p6_summary["metrics"]["unconditional_portfolio2"]
    observed = {
        subset: summarize(events, "R2", subset)
        for subset in ("overall", "cold", "warm")
    }
    max_abs_error = 0.0
    for subset, phase13_subset in (("overall", "all"), ("cold", "cold"), ("warm", "warm")):
        if observed[subset]["events"] != expected[phase13_subset]["n"]:
            raise ValueError(f"portfolio@2 {subset} event count differs from Phase13")
        for metric in ("hit@50", "ndcg@10"):
            left = observed[subset][metric]
            right = expected[phase13_subset][metric]
            max_abs_error = max(max_abs_error, abs(float(left) - float(right)))
            if not _metric_close(left, right):
                raise ValueError(f"portfolio@2 {subset}/{metric} differs from Phase13")
    return {
        "verdict": "PASS_EXACT_PHASE13_PORTFOLIO2_METRIC_RECONSTRUCTION",
        "max_abs_error": max_abs_error,
        "observed": observed,
    }


def _build_corrected_events(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    a8_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events = _read_a8_events(paths["a8_event_metrics"])
    p0_rows = _read_jsonl_by_user(paths["phase13_p0_predictions"], "Phase13 P0")
    p6_rows = _read_jsonl_by_user(paths["phase13_p6_predictions"], "Phase13 P6")
    cold_items = {
        line.strip()
        for line in paths["cold_items"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(cold_items) != 5963:
        raise ValueError("Frozen Toys cold catalog identity drift")

    contract = config["portfolio2_contract"]
    if (
        contract["candidate_exclusion_prefix"] != 7
        or contract["ranking_anchor_count"] != 8
        or contract["candidate_count"] != 3
        or contract["portfolio_size"] != 2
        or contract["ranking_size"] != 50
    ):
        raise ValueError("Unexpected portfolio@2 correction contract")

    corrected: list[dict[str, Any]] = []
    portfolio_predictions: list[dict[str, Any]] = []
    candidate_mismatches = 0
    f0_metric_mismatches = 0
    for source_event in events:
        event = copy.deepcopy(source_event)
        user = str(event["user_id"])
        p0 = p0_rows.get(user)
        p6 = p6_rows.get(user)
        if p0 is None or p6 is None:
            raise ValueError(f"portfolio@2 source missing user: {user}")
        target = str(event["target_item"])
        if (
            str(p0.get("target")) != target
            or str(p6.get("target")) != target
            or bool(p0.get("is_cold")) != bool(event["is_cold"])
            or bool(p6.get("is_cold")) != bool(event["is_cold"])
        ):
            raise ValueError(f"portfolio@2 target/subset identity drift: {user}")

        gram = unique_in_order([str(item) for item in p0["v0_top50"]])
        resolver = unique_in_order([str(item) for item in p0["resolver_top50"]])
        if len(gram) != 50 or len(resolver) != 50:
            raise ValueError(f"portfolio@2 source ranking width drift: {user}")
        protected = set(gram[: contract["candidate_exclusion_prefix"]])
        candidates = [
            item for item in resolver if item in cold_items and item not in protected
        ][: contract["candidate_count"]]
        stored_candidates = unique_in_order(
            [str(item) for item in p6["portfolio_candidates"]]
        )[: contract["candidate_count"]]
        if candidates != stored_candidates:
            candidate_mismatches += 1
            raise ValueError(f"Phase13 portfolio candidate mismatch: {user}")
        ranking = portfolio_ranking(
            gram,
            resolver,
            candidates,
            contract["portfolio_size"],
        )[: contract["ranking_size"]]
        if len(ranking) != 50 or len(set(ranking)) != 50:
            raise ValueError(f"Invalid reconstructed portfolio@2 ranking: {user}")

        f0_metrics = ranking_metrics(gram, target)
        if f0_metrics != event["metrics"]["F0"]:
            f0_metric_mismatches += 1
            raise ValueError(f"a8 F0 event metric differs from Phase13 P0: {user}")
        event["metrics"]["R2"] = ranking_metrics(ranking, target)
        corrected.append(event)
        portfolio_predictions.append(
            {
                "event_index": event["event_index"],
                "user_id": user,
                "target_item": target,
                "is_cold": bool(event["is_cold"]),
                "arm": "R2",
                "method": CORRECT_COMPARATOR,
                "portfolio_candidates": candidates,
                "top50": ranking,
            }
        )

    p0_summary = load_json(paths["phase13_p0_summary"])
    old_r2_metrics = {
        subset: a8_summary["metrics"]["R2"][subset]
        for subset in ("overall", "cold", "warm")
    }
    for subset, phase13_subset in (("overall", "all"), ("cold", "cold"), ("warm", "warm")):
        for metric in ("hit@50", "ndcg@10"):
            if not _metric_close(
                old_r2_metrics[subset][metric],
                p0_summary["metrics"]["r2"][phase13_subset][metric],
            ):
                raise ValueError("a8 actual R2 does not match declared P0 mismatch")

    reconstruction = _verify_aggregate_against_phase13(
        corrected, load_json(paths["phase13_p6_summary"])
    )
    reconstruction.update(
        {
            "events": len(corrected),
            "candidate_mismatches": candidate_mismatches,
            "f0_metric_mismatches": f0_metric_mismatches,
            "old_a8_r2_exactly_matches_phase13_p0_r2": True,
        }
    )
    return corrected, portfolio_predictions, reconstruction


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


def _calculate_corrected_results(
    config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    a8_summary: Mapping[str, Any],
) -> dict[str, Any]:
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
    statistics = config["statistics"]
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
    if multiplicity != a8_summary["multiplicity"]:
        raise ValueError("Comparator correction unexpectedly changed the primary Holm family")
    for key in (
        "S-AUX_vs_F0",
        "S-PLUS-CTRL_vs_F0",
        "S-PLUS_vs_S-PLUS-CTRL",
        "G-RIDGE_vs_F0",
    ):
        if comparisons[key] != a8_summary["paired_bootstrap"][key]:
            raise ValueError(f"Comparator correction changed non-R2 comparison: {key}")

    costs = copy.deepcopy(a8_summary["costs"])
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
        corrected_test = primary_tests[arm]
        cold_signal = interval["ci_low"] > 0 and corrected_test["reject_at_alpha"]
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
            "multiplicity_corrected_test": corrected_test,
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
    changed_labels = {
        arm: {
            "a8": a8_summary["standalone_gates"][arm]["label"],
            "a9": gates[arm]["label"],
        }
        for arm in FORMAL_ARMS
        if a8_summary["standalone_gates"][arm]["label"] != gates[arm]["label"]
    }
    return {
        "metrics": metrics,
        "paired_bootstrap": comparisons,
        "multiplicity": multiplicity,
        "standalone_gates": gates,
        "costs": costs,
        "mechanisms": copy.deepcopy(a8_summary["mechanisms"]),
        "cold_hit_complementarity_diagnostic": complementarity,
        "arm_artifact_sha256": copy.deepcopy(a8_summary["arm_artifact_sha256"]),
        "cost_comparability_note": (
            f"{a8_summary['cost_comparability_note']} R2 now denotes the frozen "
            "unconditional portfolio@2 rule over the same Phase13 resolver; timing "
            "remains historical and is not hardware-normalized."
        ),
        "correction_impact": {
            "incorrect_a8_comparator": INCORRECT_COMPARATOR,
            "correct_a9_comparator": CORRECT_COMPARATOR,
            "changed_standalone_labels": changed_labels,
            "primary_holm_family_exactly_unchanged": True,
            "non_r2_primary_comparisons_exactly_unchanged": True,
            "r2_dependent_metrics_comparisons_dominance_and_complementarity_recomputed": True,
        },
    }


def finalize(config_path: Path) -> dict[str, Any]:
    started_at = utc_now()
    absolute_config = config_path if config_path.is_absolute() else ROOT / config_path
    if absolute_config.is_symlink() or not absolute_config.is_file():
        raise ValueError("S16-4 portfolio correction config must be a regular file")
    config = load_json(absolute_config)
    if config.get("schema_version") != "stage16_s4_toys_portfolio2_correction_v1":
        raise ValueError("Unexpected S16-4 portfolio correction schema")
    output = ROOT / config["output_dir"]
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite an existing S16-4 correction root")
    if (
        config["resources"].get("cpu_only") is not True
        or config["resources"].get("gpu_count") != 0
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("S16-4 portfolio correction must be CPU-only")

    paths, input_hashes_before = _load_frozen_inputs(config)
    a8_summary = _verify_a8_lineage(paths)
    code_hashes_before = {
        relative: sha256_file(ROOT / relative) for relative in EXECUTED_CODE_PATHS
    }
    events, portfolio_predictions, reconstruction = _build_corrected_events(
        config, paths, a8_summary
    )
    calculation = _calculate_corrected_results(config, events, a8_summary)
    _, input_hashes_after = _load_frozen_inputs(config)
    code_hashes_after = {
        relative: sha256_file(ROOT / relative) for relative in EXECUTED_CODE_PATHS
    }
    if input_hashes_before != input_hashes_after:
        raise ValueError("Frozen inputs changed during S16-4 portfolio correction")
    if code_hashes_before != code_hashes_after:
        raise ValueError("Code changed during S16-4 portfolio correction")

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
    write_json(
        output / "correction_manifest.json",
        {
            "verdict": CORRECTION_VERDICT,
            "source_a8_attempt_id": a8_summary["attempt_id"],
            "source_a8_preserved": True,
            "source_a7_preserved_via_a8_contract": True,
            "mismatch": {
                "plan_expected": CORRECT_COMPARATOR,
                "a8_actual": INCORRECT_COMPARATOR,
            },
            "corrected_fields": [
                "R2 event rankings and metrics",
                "paired comparisons involving R2",
                "Pareto dominance and standalone labels",
                "cold-hit complementarity involving R2",
            ],
            "unchanged_fields": [
                "GPU arm predictions",
                "S-AUX vs F0 primary comparison",
                "S-PLUS vs matched control primary comparison",
                "G-RIDGE vs F0 primary comparison",
                "four-comparison Holm family",
            ],
            "gpu_scientific_inference_recompute": False,
            "validation_used_for_tuning_or_state_selection": False,
            "test_read": False,
            "automatic_retry": False,
        },
    )
    write_json(
        output / "command_manifest.json",
        {"exact_start_command": config["exact_start_command"]},
    )
    write_json(output / "input_file_sha256.json", input_hashes_before)

    portfolio_path = output / "portfolio2_predictions_validation.jsonl"
    with portfolio_path.open("x", encoding="utf-8") as handle:
        for row in portfolio_predictions:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    event_metrics_path = output / "event_metrics.jsonl"
    with event_metrics_path.open("x", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "COMPLETED",
        "verdict": CORRECTION_VERDICT,
        "generated_at_utc": utc_now(),
        "source_a8_attempt_id": a8_summary["attempt_id"],
        "source_a8_status": "COMPLETED_PRESERVED",
        "events": len(events),
        "corrected_arm_id": "R2",
        "corrected_comparator_identity": CORRECT_COMPARATOR,
        "supersedes_a8_for_r2_dependent_conclusions": True,
        "portfolio2_reconstruction": reconstruction,
        **calculation,
        "portfolio2_predictions_sha256": sha256_file(portfolio_path),
        "event_metrics_sha256": sha256_file(event_metrics_path),
        "execution_identity_artifact": {
            "path": str(identity_path.relative_to(ROOT)),
            "sha256": sha256_file(identity_path),
        },
        "gpu_used": False,
        "gpu_scientific_inference_recompute": False,
        "validation_used_for_tuning_or_state_selection": False,
        "scientific_efficacy_metric_produced": True,
        "test_read": False,
        "automatic_retry": False,
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    status = {
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "COMPLETED",
        "status_code": CORRECTION_VERDICT,
        "stage": "finished",
        "reason": (
            "CPU-only correction replaced the misidentified P0 R2 comparator with "
            "the preregistered unconditional portfolio@2; a7/a8 remain preserved."
        ),
        "started_at": started_at,
        "updated_at": utc_now(),
        "process_alive": False,
        "gpu_count": 0,
        "gpu_used": False,
        "progress_current": 1,
        "progress_total": 1,
        "progress_unit": "comparator_correction",
        "exit_code": 0,
        "source_a8_status": "COMPLETED_PRESERVED",
        "scientific_efficacy_metric_produced": True,
        "test_read": False,
        "automatic_retry": False,
        "automatic_resume": False,
        "exact_start_command": config["exact_start_command"],
        "output_dir": config["output_dir"],
        "summary_path": str(summary_path.relative_to(ROOT)),
    }
    status_path = output / "status.json"
    write_json(status_path, status)
    artifact_contract = {
        "verdict": "PASS_S16_4_TOYS_PORTFOLIO2_CORRECTION_ARTIFACT_CONTRACT",
        "required_local": [
            "status.json",
            "config.json",
            "execution_identity.json",
            "correction_manifest.json",
            "command_manifest.json",
            "input_file_sha256.json",
            "portfolio2_predictions_validation.jsonl",
            "event_metrics.jsonl",
            "summary.json",
        ],
        "source_a8_attempt_id": a8_summary["attempt_id"],
        "source_a8_preserved": True,
        "source_artifact_sha256": input_hashes_before,
        "local_sha256": {
            "status.json": sha256_file(status_path),
            "config.json": sha256_file(config_copy),
            "execution_identity.json": sha256_file(identity_path),
            "correction_manifest.json": sha256_file(output / "correction_manifest.json"),
            "command_manifest.json": sha256_file(output / "command_manifest.json"),
            "input_file_sha256.json": sha256_file(output / "input_file_sha256.json"),
            "portfolio2_predictions_validation.jsonl": sha256_file(portfolio_path),
            "event_metrics.jsonl": sha256_file(event_metrics_path),
            "summary.json": sha256_file(summary_path),
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
