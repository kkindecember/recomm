#!/usr/bin/env python3
"""Finalize all frozen S16-4 Toys arms with paired-bootstrap Gates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiment.phase15.protocol.common_adapter import (
    read_projected_sequences,
    read_validation_predictions,
)
from experiment.phase16.protocol.stage16_s4_toys_validation import (
    EXPECTED_CONTROLS,
    FORMAL_ARMS,
    ROOT,
    SCIENTIFIC_ARMS,
    atomic_json,
    load_json,
    ranking_metrics,
    sha256_file,
    verify_regular,
)


def read_arm_predictions(path: Path, arm: str, expected_users: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row.get("user_id"))
            ranking = row.get("top50")
            if row.get("arm") != arm or user in rows or user not in expected_users:
                raise ValueError(f"Invalid {arm} prediction identity at line {line_number}")
            if not isinstance(ranking, list) or len(ranking) != 50 or len(set(ranking)) != 50:
                raise ValueError(f"Invalid {arm} top-50 at line {line_number}")
            rows[user] = row
    if set(rows) != expected_users:
        raise ValueError(f"{arm} prediction users differ from frozen validation")
    return rows


def summarize(events: Sequence[Mapping[str, Any]], arm: str, subset: str) -> dict[str, Any]:
    selected = [
        row for row in events
        if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
    ]
    if not selected:
        raise ValueError(f"Empty finalizer subset: {subset}")
    return {
        "events": len(selected),
        "hit@50": float(np.mean([row["metrics"][arm]["hit@50"] for row in selected])),
        "ndcg@10": float(np.mean([row["metrics"][arm]["ndcg@10"] for row in selected])),
    }


def paired_bootstrap(
    events: Sequence[Mapping[str, Any]],
    treatment: str,
    control: str,
    metric: str,
    subset: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    selected = [
        row for row in events
        if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
    ]
    delta = np.asarray(
        [row["metrics"][treatment][metric] - row["metrics"][control][metric] for row in selected],
        dtype=np.float64,
    )
    if not len(delta):
        raise ValueError("Paired bootstrap received an empty event subset")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 250):
        count = min(250, resamples - start)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "events": len(delta),
        "observed": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "verdict": "PASS" if low > 0 else ("NEGATIVE" if high < 0 else "INCONCLUSIVE"),
    }


def strictly_dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    benefits = ("cold_hit@50", "warm_ndcg@10")
    costs = ("update_seconds", "inference_seconds", "extra_state_bytes")
    weak = all(float(left[key]) >= float(right[key]) for key in benefits) and all(
        float(left[key]) <= float(right[key]) for key in costs
    )
    strict = any(float(left[key]) > float(right[key]) for key in benefits) or any(
        float(left[key]) < float(right[key]) for key in costs
    )
    return weak and strict


def run(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema_version") != "stage16_s4_toys_standalone_v1":
        raise ValueError("Unexpected S16-4 finalizer schema")
    output = ROOT / config["output_dir"]
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError("Refusing to overwrite completed S16-4 summary")
    frozen_config_path = output / "config.json"
    runtime_manifest_path = output / "runtime_snapshot_manifest.json"
    if frozen_config_path.is_symlink() or not frozen_config_path.is_file():
        raise ValueError("Missing formal-root frozen config")
    if runtime_manifest_path.is_symlink() or not runtime_manifest_path.is_file():
        raise ValueError("Missing formal-root isolated-runtime manifest")
    if sha256_file(frozen_config_path) != sha256_file(config_path):
        raise ValueError("Formal-root config differs from the executed config")
    runtime_manifest = load_json(runtime_manifest_path)
    if (
        runtime_manifest.get("schema_version")
        != "stage16_s4_toys_gpu0_a3_isolated_runtime_v1"
        or runtime_manifest.get("config_sha256") != sha256_file(config_path)
    ):
        raise ValueError("Formal-root isolated-runtime identity drift")

    parent_path = verify_regular(ROOT, config["preflight"]["config"], "preflight_config")
    parent = load_json(parent_path)
    projected_path = verify_regular(
        ROOT, parent["inputs"]["projected_train_validation_sequences"], "projected_sequences"
    )
    frozen_path = verify_regular(
        ROOT, parent["inputs"]["frozen_f0_r2_validation_predictions"], "frozen_f0_r2"
    )
    projected = read_projected_sequences(projected_path)
    frozen = read_validation_predictions(frozen_path)
    users = set(projected)
    if users != set(frozen) or len(users) != config["validation_events"]:
        raise ValueError("Finalizer frozen validation universe drift")

    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    arm_sha: dict[str, Any] = {}
    for arm in FORMAL_ARMS:
        arm_root = output / "arms" / arm
        prediction_path = arm_root / "predictions_validation.jsonl"
        arm_summary_path = arm_root / "summary.json"
        if prediction_path.is_symlink() or not prediction_path.is_file():
            raise ValueError(f"Missing formal predictions: {arm}")
        if arm_summary_path.is_symlink() or not arm_summary_path.is_file():
            raise ValueError(f"Missing formal summary: {arm}")
        arm_summary = load_json(arm_summary_path)
        expected_verdict = f"COMPLETED_S16_4_TOYS_{arm.replace('-', '_')}_FROZEN_VALIDATION"
        if (
            arm_summary.get("verdict") != expected_verdict
            or arm_summary.get("events") != config["validation_events"]
            or arm_summary.get("promotion_eligible") is not True
            or arm_summary.get("test_read") is not False
        ):
            raise ValueError(f"Formal arm summary contract failed: {arm}")
        arm_rows[arm] = read_arm_predictions(prediction_path, arm, users)
        arm_summaries[arm] = arm_summary
        arm_sha[arm] = {
            "summary": sha256_file(arm_summary_path),
            "predictions": sha256_file(prediction_path),
        }

    events: list[dict[str, Any]] = []
    metrics_path = output / "event_metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8") as handle:
        for event_index, (user, sequence) in enumerate(projected.items(), 1):
            target = sequence[-1]
            source = frozen[user]
            rankings = {
                "F0": [str(item) for item in source["v0_top50"]],
                "R2": [str(item) for item in source["r2_top50"]],
                **{arm: arm_rows[arm][user]["top50"] for arm in FORMAL_ARMS},
            }
            row = {
                "event_index": event_index,
                "user_id": user,
                "target_item": target,
                "is_cold": bool(source["is_cold"]),
                "metrics": {arm: ranking_metrics(rankings[arm], target) for arm in SCIENTIFIC_ARMS},
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            events.append(row)

    metrics = {
        arm: {subset: summarize(events, arm, subset) for subset in ("overall", "cold", "warm")}
        for arm in SCIENTIFIC_ARMS
    }
    comparisons: dict[str, Any] = {}
    seed_offset = 0
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
                resamples=config["statistics"]["paired_bootstrap_resamples"],
                seed=config["statistics"]["paired_bootstrap_seed"] + seed_offset,
            )
            seed_offset += 1

    source_summaries = {
        "S-AUX": load_json(verify_regular(ROOT, parent["inputs"]["saux_summary"], "saux_summary")),
        "S-PLUS": load_json(verify_regular(ROOT, parent["inputs"]["splus_summary"], "splus_summary")),
        "S-PLUS-CTRL": load_json(
            verify_regular(ROOT, parent["inputs"]["splus_ctrl_summary"], "splus_ctrl_summary")
        ),
        "G-RIDGE": load_json(
            verify_regular(ROOT, parent["inputs"]["gridge_formal_summary"], "gridge_formal_summary")
        ),
    }
    control_cost = config["control_cost"]
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
        cold_interval = comparisons[f"{arm}_vs_{control}"]["cold_hit@50"]
        cold_signal = cold_interval["ci_low"] > 0
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
            "cold_gain_interval": cold_interval,
            "strict_dominators": dominators,
            "vector": vectors[arm],
        }

    summary = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "completed",
        "verdict": "COMPLETED_S16_4_TOYS_STANDALONE_FROZEN_VALIDATION",
        "events": len(events),
        "metrics": metrics,
        "paired_bootstrap": comparisons,
        "standalone_gates": gates,
        "costs": costs,
        "cost_comparability_note": control_cost["comparability_note"],
        "arm_artifact_sha256": arm_sha,
        "event_metrics_sha256": sha256_file(metrics_path),
        "validation_used_for_tuning_or_state_selection": False,
        "scientific_efficacy_metric_produced": True,
        "test_read": False,
        "automatic_retry": False,
        "config_sha256": sha256_file(config_path),
        "runtime_snapshot_manifest_sha256": sha256_file(runtime_manifest_path),
        "isolated_code_files": len(runtime_manifest["code_sha256"]),
    }
    atomic_json(summary_path, summary)
    atomic_json(
        output / "artifact_contract.json",
        {
            "verdict": "PASS_S16_4_TOYS_FORMAL_ARTIFACT_CONTRACT",
            "required": [
                "status.json",
                "config.json",
                "runtime_snapshot_manifest.json",
                "summary.json",
                "event_metrics.jsonl",
                *[f"arms/{arm}/summary.json" for arm in FORMAL_ARMS],
                *[f"arms/{arm}/predictions_validation.jsonl" for arm in FORMAL_ARMS],
            ],
        },
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args.config.resolve())
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
