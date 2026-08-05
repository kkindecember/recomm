#!/usr/bin/env python3
"""Validation-only recovery for the GACR-v8 integrity-gate implementation bug.

The completed calibration run is immutable input.  This program recomputes the
pre-registered qualification predicate with typed integrity checks, verifies all
reused checkpoint hashes, and evaluates only qualified arms on the frozen fresh
cohort.  It never trains or overwrites a residual checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gacr_p0 import relative_gain  # noqa: E402
from experiment.phase4.gcdh_p0 import ROOT, sha256, write_json  # noqa: E402
from experiment.phase6.gacr_v2 import (  # noqa: E402
    paired_bootstrap_candidate,
    serializable_rows,
    validate_checkpoint_lineage,
)
from experiment.phase6.gacr_v3 import evaluate_budget  # noqa: E402
from experiment.phase6.gacr_v8 import build_records, evaluate  # noqa: E402


METRICS = ("Recall@10", "NDCG@10", "Recall@50")


def completed_run_integrity_passes(integrity: dict) -> bool:
    """Interpret typed integrity evidence rather than applying truthiness to it."""
    return (
        integrity.get("all_fit_records_used") is True
        and integrity.get("fit_calibration_user_disjoint") is True
        and integrity.get("parent_checkpoint_sha_unchanged_during_training") is True
        and integrity.get("backbone_optimizer_steps") == 0
        and integrity.get("test_data_read") is False
        and integrity.get("sports_data_read") is False
    )


def recompute_qualified_arms(summary: dict, config: dict) -> dict[str, bool]:
    """Recompute D/E qualification from every frozen domain-seed calibration cell."""
    integrity_ok = completed_run_integrity_passes(summary.get("integrity", {}))
    result = {}
    for arm in ("D", "E"):
        cells = []
        for dataset in config["datasets"]:
            arm_results = summary["training"][dataset]["arms"][arm]
            for seed in config["training_seeds"]:
                cell = arm_results[str(seed)]
                cells.append(
                    cell.get("finite_checkpoint") is True
                    and cell["calibration_noninferiority"].get("eligible") is True
                )
        result[arm] = integrity_ok and bool(cells) and all(cells)
    return result


def verified_state(path: Path, expected_sha256: str) -> dict:
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA mismatch: path={path}:expected={expected_sha256}:actual={actual}"
        )
    return torch.load(path, map_location="cpu")


def incumbent_rows(
    records: list[dict], state: dict, bound: float, budget: float, device: torch.device
) -> tuple[dict, list[dict]]:
    adapted = [dict(record, features=record["features6"]) for record in records]
    return evaluate_budget(adapted, state, bound, budget, device)


def direct_comparison_rows(b_rows: list[dict], e_rows: list[dict]) -> list[dict]:
    if [row["sample_key"] for row in b_rows] != [row["sample_key"] for row in e_rows]:
        raise ValueError("B/E fresh-validation rows are not aligned")
    output = []
    for b_row, e_row in zip(b_rows, e_rows):
        row = {
            "sample_key": b_row["sample_key"],
            "target_group": b_row["target_group"],
            "baseline_rank": b_row["candidate_rank"],
            "candidate_rank": e_row["candidate_rank"],
            "union_covered": b_row["union_covered"],
            "changed": int(b_row["candidate_rank"] != e_row["candidate_rank"]),
        }
        for metric in METRICS:
            row[f"baseline_{metric}"] = b_row[f"candidate_{metric}"]
            row[f"candidate_{metric}"] = e_row[f"candidate_{metric}"]
        row["broad_harm"] = int(
            row["baseline_Recall@10"] == 1.0 and row["candidate_Recall@10"] == 0.0
        )
        output.append(row)
    return output


def summarize_rows(rows: list[dict]) -> dict:
    result = {}
    for group in ("overall", "head", "tail"):
        selected = rows if group == "overall" else [
            row for row in rows if row["target_group"] == group
        ]
        entry = {
            "n": len(selected),
            "changed_user_coverage": float(np.mean([row["changed"] for row in selected])),
            "broad_harm_rate": float(np.mean([row["broad_harm"] for row in selected])),
        }
        for metric in METRICS:
            baseline = float(np.mean([row[f"baseline_{metric}"] for row in selected]))
            candidate = float(np.mean([row[f"candidate_{metric}"] for row in selected]))
            entry[metric] = {
                "baseline": baseline,
                "candidate": candidate,
                "absolute_delta": candidate - baseline,
                "relative_gain": relative_gain(baseline, candidate),
            }
        result[group] = entry
    return result


def aggregate_seed_rows(seed_rows: dict[str, list[dict]]) -> list[dict]:
    """Average each user's metrics over seeds before bootstrap, as preregistered."""
    ordered = [seed_rows[key] for key in sorted(seed_rows, key=int)]
    keys = [[row["sample_key"] for row in rows] for rows in ordered]
    if any(value != keys[0] for value in keys[1:]):
        raise ValueError("cross-seed fresh-validation rows are not aligned")
    output = []
    for index, sample_key in enumerate(keys[0]):
        source = [rows[index] for rows in ordered]
        row = {
            "sample_key": sample_key,
            "target_group": source[0]["target_group"],
            "baseline_rank": None,
            "candidate_rank": None,
            "union_covered": source[0]["union_covered"],
            "changed": float(np.mean([item["changed"] for item in source])),
            "broad_harm": float(np.mean([item["broad_harm"] for item in source])),
        }
        for metric in METRICS:
            for side in ("baseline", "candidate"):
                row[f"{side}_{metric}"] = float(
                    np.mean([item[f"{side}_{metric}"] for item in source])
                )
        output.append(row)
    return output


def bootstrap_report(rows: list[dict], seed: int) -> dict:
    tail = [row for row in rows if row["target_group"] == "tail"]
    return {
        "unit": "fresh user after within-user mean over three training seeds",
        "samples": 10000,
        "overall_ndcg10_absolute_delta_ci95": paired_bootstrap_candidate(
            rows, "NDCG@10", False, seed + 1
        ),
        "overall_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(
            rows, "NDCG@10", True, seed + 2
        ),
        "overall_recall10_absolute_delta_ci95": paired_bootstrap_candidate(
            rows, "Recall@10", False, seed + 3
        ),
        "overall_recall50_absolute_delta_ci95": paired_bootstrap_candidate(
            rows, "Recall@50", False, seed + 4
        ),
        "tail_ndcg10_absolute_delta_ci95": paired_bootstrap_candidate(
            tail, "NDCG@10", False, seed + 5
        ),
        "tail_recall50_absolute_delta_ci95": paired_bootstrap_candidate(
            tail, "Recall@50", False, seed + 6
        ),
    }


def stratified_macro_bootstrap(
    domain_rows: dict[str, list[dict]], field: str, relative: bool, seed: int, samples: int = 10000
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    arrays = {}
    for dataset, rows in domain_rows.items():
        arrays[dataset] = (
            np.asarray([row[f"baseline_{field}"] for row in rows], dtype=float),
            np.asarray([row[f"candidate_{field}"] for row in rows], dtype=float),
        )
    for _ in range(samples):
        domain_values = []
        for baseline, candidate in arrays.values():
            indices = rng.integers(0, len(baseline), size=len(baseline))
            b_mean = float(baseline[indices].mean())
            c_mean = float(candidate[indices].mean())
            domain_values.append(relative_gain(b_mean, c_mean) if relative else c_mean - b_mean)
        values.append(float(np.mean(domain_values)))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def write_rows(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = serializable_rows(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)
    return sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v8 validation recovery requires CUDA")

    recovery = json.loads(args.recovery_config.read_text())
    config_path = ROOT / recovery["original_preregistered_config"]
    summary_path = ROOT / recovery["completed_calibration_summary"]
    if sha256(config_path) != recovery["original_preregistered_config_sha256"]:
        raise RuntimeError("original preregistered config SHA mismatch")
    if sha256(summary_path) != recovery["completed_calibration_summary_sha256"]:
        raise RuntimeError("completed calibration summary SHA mismatch")
    config = json.loads(config_path.read_text())
    completed = json.loads(summary_path.read_text())
    validate_checkpoint_lineage(config)

    qualified = recompute_qualified_arms(completed, config)
    if qualified != recovery["expected_recomputed_qualified_arms"]:
        raise RuntimeError(f"unexpected recomputed qualification: {qualified}")
    if [arm for arm, allowed in qualified.items() if allowed] != ["E"]:
        raise RuntimeError("validation-only recovery is locked to qualified arm E")

    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")
    validation = {}
    aggregated = {}
    reused = {"B": {}, "E": {}}
    positive_cells = 0

    for dataset_index, dataset in enumerate(config["datasets"]):
        metadata, records = build_records(dataset, config, p0_config, device, validation=True)
        seeds = {}
        direct_by_seed = {}
        reused["B"][dataset] = {}
        reused["E"][dataset] = {}
        for seed in config["training_seeds"]:
            seed_key = str(seed)
            b_path = ROOT / recovery["incumbent"]["checkpoint_root"] / dataset / f"residual_seed{seed}.pt"
            e_path = ROOT / recovery["qualified_checkpoint_root"] / dataset / f"E_seed{seed}.pt"
            b_state = verified_state(
                b_path, recovery["incumbent"]["expected_checkpoint_sha256"][dataset][seed_key]
            )
            e_state = verified_state(
                e_path, recovery["expected_qualified_checkpoint_sha256"][dataset][seed_key]
            )
            reused["B"][dataset][seed_key] = sha256(b_path)
            reused["E"][dataset][seed_key] = sha256(e_path)

            b_groups, b_rows = incumbent_rows(
                records,
                b_state,
                float(recovery["incumbent"]["bound"]),
                float(recovery["incumbent"]["selected_domain_budgets"][dataset]),
                device,
            )
            e_groups, e_rows = evaluate(records, e_state, "E", device)
            direct = direct_comparison_rows(b_rows, e_rows)
            direct_by_seed[seed_key] = direct
            direct_summary = summarize_rows(direct)
            positive_cells += int(direct_summary["overall"]["NDCG@10"]["absolute_delta"] > 0.0)

            output_dir = args.output_root / dataset
            seeds[seed_key] = {
                "A_vs_B": {
                    "groups": b_groups,
                    "per_user_sha256": write_rows(output_dir / f"B_seed{seed}_per_user.csv", b_rows),
                },
                "A_vs_E": {
                    "groups": e_groups,
                    "per_user_sha256": write_rows(output_dir / f"E_seed{seed}_per_user.csv", e_rows),
                },
                "E_vs_B": {
                    "groups": direct_summary,
                    "per_user_sha256": write_rows(
                        output_dir / f"E_vs_B_seed{seed}_per_user.csv", direct
                    ),
                },
            }

        averaged = aggregate_seed_rows(direct_by_seed)
        averaged_path = args.output_root / dataset / "E_vs_B_three_seed_mean_per_user.csv"
        validation[dataset] = metadata | {
            "seeds": seeds,
            "E_vs_B_three_seed_user_mean": {
                "groups": summarize_rows(averaged),
                "bootstrap": bootstrap_report(averaged, 8100 + dataset_index * 100),
                "per_user_sha256": write_rows(averaged_path, averaged),
            },
        }
        aggregated[dataset] = averaged
        del records
        torch.cuda.empty_cache()

    domain_absolute = [
        validation[dataset]["E_vs_B_three_seed_user_mean"]["groups"]["overall"]["NDCG@10"][
            "absolute_delta"
        ]
        for dataset in config["datasets"]
    ]
    domain_relative = [
        validation[dataset]["E_vs_B_three_seed_user_mean"]["groups"]["overall"]["NDCG@10"][
            "relative_gain"
        ]
        for dataset in config["datasets"]
    ]
    macro = {
        "comparison": "E_vs_B",
        "domain_equal_weight": True,
        "overall_ndcg10_absolute_delta": float(np.mean(domain_absolute)),
        "overall_ndcg10_relative_gain": float(np.mean(domain_relative)),
        "positive_domain_seed_cells": positive_cells,
        "total_domain_seed_cells": len(config["datasets"]) * len(config["training_seeds"]),
        "overall_ndcg10_absolute_delta_ci95": stratified_macro_bootstrap(
            aggregated, "NDCG@10", False, 8301
        ),
        "overall_ndcg10_relative_gain_ci95": stratified_macro_bootstrap(
            aggregated, "NDCG@10", True, 8302
        ),
    }

    integrity = {
        "recovery_mode": "validation_only_reuse_completed_v8_E_and_frozen_v3_B",
        "typed_completed_run_integrity_passed": completed_run_integrity_passes(
            completed["integrity"]
        ),
        "recomputed_qualified_arms": qualified,
        "retrained_residuals": False,
        "reused_checkpoint_sha256": reused,
        "parent_checkpoint_sha_unchanged": all(
            validation[dataset]["parent_checkpoint_sha256_before"]
            == validation[dataset]["parent_checkpoint_sha256_after"]
            for dataset in config["datasets"]
        ),
        "fresh_validation_zero_overlap": all(
            validation[dataset]["gcdh_or_training_overlap"] == 0
            and validation[dataset]["prior_gacr_p0_overlap"] == 0
            for dataset in config["datasets"]
        ),
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    if not (
        integrity["typed_completed_run_integrity_passed"]
        and integrity["parent_checkpoint_sha_unchanged"]
        and integrity["fresh_validation_zero_overlap"]
    ):
        raise RuntimeError(f"validation recovery integrity failure: {integrity}")

    write_json(
        args.output_root / "summary.json",
        {
            "experiment_id": config["experiment_id"],
            "recovery_id": recovery["recovery_id"],
            "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
            "recovery": recovery,
            "qualified_arms": qualified,
            "validation": validation,
            "macro": macro,
            "integrity": integrity,
        },
    )
    print("GACR_V8_VALIDATION_RECOVERY_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
