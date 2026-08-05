#!/usr/bin/env python3
"""Exploratory post-P2 diagnosis of frozen-gate target selection shift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE11 = REPO_ROOT / "experiment/phase11"
if str(PHASE11) not in sys.path:
    sys.path.insert(0, str(PHASE11))

from eval_bw3_p2_one_shot_validation import (  # noqa: E402
    DATASETS,
    FEATURES,
    prepare_domain,
    validate_gate,
)
from train_bw3_listwise_admission import prepare_events  # noqa: E402


EXPERIMENT_ID = "GRAM_PHASE11_BW3_P2_EXPLORATORY_SELECTION_SHIFT_DIAGNOSTIC_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q10": None, "median": None, "mean": None, "q90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def candidate_logits(event: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in event["expansion"]:
        feature = np.asarray(candidate["features"], dtype=np.float64)
        z = (feature - model["mean"]) / model["std"]
        contributions = z * model["weight"]
        logit = float(contributions.sum() + model["bias"])
        if not math.isfinite(logit):
            raise ValueError("non-finite diagnostic logit")
        rows.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "logit": logit,
                "z": z,
                "contributions": contributions,
            }
        )
    return sorted(rows, key=lambda row: (-row["logit"], row["candidate_id"]))


def analyze_split(
    dataset: str,
    split: str,
    events: list[dict[str, Any]],
    model: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    target_rows: list[dict[str, Any]] = []
    target_z: list[np.ndarray] = []
    target_contributions: list[np.ndarray] = []
    all_target_logits: list[float] = []
    all_nontarget_logits: list[float] = []
    any_admission_users = 0
    total_passing = 0
    for event in events:
        scored = candidate_logits(event, model)
        passing = [row for row in scored if row["logit"] >= model["margin"]]
        any_admission_users += bool(passing)
        total_passing += min(len(passing), 3)
        target = next((row for row in scored if row["candidate_id"] == event["target"]), None)
        for row in scored:
            if row is not target:
                all_nontarget_logits.append(row["logit"])
        if target is None:
            continue
        rank = scored.index(target) + 1
        passes = target["logit"] >= model["margin"]
        selected = passes and rank <= 3
        all_target_logits.append(target["logit"])
        target_z.append(target["z"])
        target_contributions.append(target["contributions"])
        target_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "user": event["user"],
                "target_item_id": event["target"],
                "target_logit": target["logit"],
                "target_rank_in_expansion": rank,
                "target_passes_margin": int(passes),
                "target_selected_top3": int(selected),
                "passing_candidates": len(passing),
                "max_nontarget_logit": max(
                    (row["logit"] for row in scored if row["candidate_id"] != event["target"]),
                    default=float("-inf"),
                ),
            }
        )
    target_count = len(target_rows)
    target_passes = sum(row["target_passes_margin"] for row in target_rows)
    target_selected = sum(row["target_selected_top3"] for row in target_rows)
    summary = {
        "dataset": dataset,
        "split": split,
        "users": len(events),
        "expansion_target_users": target_count,
        "target_passes_margin": target_passes,
        "target_pass_rate": target_passes / target_count if target_count else None,
        "target_selected_top3": target_selected,
        "target_selection_rate": target_selected / target_count if target_count else None,
        "target_passes_but_competition_rejects": sum(
            row["target_passes_margin"] and not row["target_selected_top3"] for row in target_rows
        ),
        "target_below_margin": target_count - target_passes,
        "users_with_any_admission": any_admission_users,
        "total_admissions_after_cap": total_passing,
        "target_logit": quantiles(all_target_logits),
        "nontarget_logit": quantiles(all_nontarget_logits),
        "target_rank": quantiles([float(row["target_rank_in_expansion"]) for row in target_rows]),
        "exploratory_only": True,
    }
    z_matrix = np.stack(target_z) if target_z else np.empty((0, len(FEATURES)))
    contribution_matrix = (
        np.stack(target_contributions) if target_contributions else np.empty((0, len(FEATURES)))
    )
    return summary, target_rows, z_matrix, contribution_matrix


def feature_shift(
    calibration_z: np.ndarray,
    validation_z: np.ndarray,
    calibration_contrib: np.ndarray,
    validation_contrib: np.ndarray,
) -> list[dict[str, Any]]:
    if not len(calibration_z) or not len(validation_z):
        raise ValueError("feature shift requires target examples in both splits")
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(FEATURES):
        calibration_z_mean = float(calibration_z[:, index].mean())
        validation_z_mean = float(validation_z[:, index].mean())
        calibration_contrib_mean = float(calibration_contrib[:, index].mean())
        validation_contrib_mean = float(validation_contrib[:, index].mean())
        rows.append(
            {
                "feature": feature,
                "calibration_target_z_mean": calibration_z_mean,
                "validation_target_z_mean": validation_z_mean,
                "target_z_shift": validation_z_mean - calibration_z_mean,
                "calibration_logit_contribution_mean": calibration_contrib_mean,
                "validation_logit_contribution_mean": validation_contrib_mean,
                "logit_contribution_shift": validation_contrib_mean - calibration_contrib_mean,
            }
        )
    return sorted(rows, key=lambda row: row["logit_contribution_shift"])


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") or torch.cuda.is_available():
        raise RuntimeError("exploratory diagnostic is CPU-only")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "GRAM_PHASE11_BW3_P2_LISTWISE_ADMISSION_ONE_SHOT_VALIDATION_V1":
        raise ValueError("unexpected P2 config")
    p2_summary = json.loads(
        (REPO_ROOT / "artifacts/phase11/bw3_p2_one_shot_validation/scientific/summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not (p2_summary.get("validation_consumed") and p2_summary.get("results_revealed")):
        raise PermissionError("diagnostic requires an already consumed and revealed P2")

    args.output_dir.mkdir(parents=True)
    domains: list[dict[str, Any]] = []
    for dataset in ("Toys", "Beauty"):
        gate_path = REPO_ROOT / config["domain_inputs"][dataset]["gate"]
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        model = validate_gate(gate, dataset)
        calibration_events, _, _ = prepare_events(
            dataset,
            3,
            REPO_ROOT / "artifacts/phase11/bw3_p1_admission_recovery" / dataset / "calibration",
        )
        validation_events, _ = prepare_domain(dataset, config)
        calibration, calibration_rows, calibration_z, calibration_contrib = analyze_split(
            dataset, "calibration_t_minus_3", calibration_events, model
        )
        validation, validation_rows, validation_z, validation_contrib = analyze_split(
            dataset, "validation_t_minus_2", validation_events, model
        )
        shifts = feature_shift(calibration_z, validation_z, calibration_contrib, validation_contrib)
        domain_dir = args.output_dir / dataset
        domain_dir.mkdir()
        write_tsv(domain_dir / "target_candidates.tsv", calibration_rows + validation_rows)
        write_tsv(domain_dir / "feature_contribution_shift.tsv", shifts)
        domain = {
            "dataset": dataset,
            "calibration": calibration,
            "validation": validation,
            "selection_rate_delta": validation["target_selection_rate"] - calibration["target_selection_rate"],
            "target_logit_mean_delta": validation["target_logit"]["mean"] - calibration["target_logit"]["mean"],
            "feature_contribution_shifts": shifts,
            "gate_sha256": p2_summary["datasets"][0 if dataset == "Toys" else 1]["gate_sha256"],
            "confirmatory_p2_status_unchanged": "failed",
        }
        (domain_dir / "summary.json").write_text(
            json.dumps(domain, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        domains.append(domain)
    aggregate = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_exploratory_diagnostic",
        "scope": "post_hoc_exploratory_analysis_of_already_revealed_p2",
        "confirmatory_p2_status_unchanged": "failed",
        "margin_or_gate_modified": False,
        "test_read": False,
        "sports_read": False,
        "datasets": domains,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": aggregate["status"], "datasets": domains}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
