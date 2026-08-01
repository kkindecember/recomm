#!/usr/bin/env python3
"""Phase 6 GACR-v5: target-free soft benefit weighting."""

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
from experiment.phase4.gacr_s0 import BoundedResidualRanker  # noqa: E402
from experiment.phase4.gcdh_p0 import ROOT, sha256, write_json  # noqa: E402
from experiment.phase6.gacr_v2 import (  # noqa: E402
    build_training_records,
    build_validation_records,
    paired_bootstrap_candidate,
    rank_metrics,
    serializable_rows,
    summarize_seed_stability,
    validate_checkpoint_lineage,
)
from experiment.phase6.gacr_v3 import v3_method_result  # noqa: E402
from experiment.phase6.gacr_v4 import (  # noqa: E402
    gate_probability,
    rank_from_scores,
    target_free_gate_features,
)


def soft_multiplier(probability: float, alpha: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("gate probability must be in [0, 1]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return float(alpha + (1.0 - alpha) * probability)


@torch.no_grad()
def evaluate_soft_weight(
    records: list[dict],
    residual_state: dict,
    gate_state: dict,
    bound: float,
    alpha: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    ranker = BoundedResidualRanker(6, 16, bound).to(device)
    ranker.load_state_dict(residual_state, strict=True)
    ranker.eval()
    rows = []
    for record in records:
        residual = ranker(record["features"].to(device))
        feature = target_free_gate_features(record, residual)
        probability = gate_probability(feature, gate_state)
        multiplier = soft_multiplier(probability, alpha)
        baseline_rank = record["gram_rank"]
        candidate_rank = rank_from_scores(
            record,
            record["base"].to(device) + multiplier * residual,
        )
        b_r10, b_ndcg, b_r50 = rank_metrics(baseline_rank)
        c_r10, c_ndcg, c_r50 = rank_metrics(candidate_rank)
        rows.append(
            {
                "sample_key": record["sample_key"],
                "target_group": record["target_group"],
                "baseline_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "union_covered": int(record["target_index"] is not None),
                "baseline_Recall@10": b_r10,
                "baseline_NDCG@10": b_ndcg,
                "baseline_Recall@50": b_r50,
                "candidate_Recall@10": c_r10,
                "candidate_NDCG@10": c_ndcg,
                "candidate_Recall@50": c_r50,
                "changed": int(candidate_rank != baseline_rank),
                "broad_harm": int(b_r10 == 1.0 and c_r10 == 0.0),
                "gate_probability": probability,
                "soft_multiplier": multiplier,
                "residual_spread": float(residual.max() - residual.min()),
            }
        )

    def summarize(selected: list[dict]) -> dict:
        covered = [row for row in selected if row["union_covered"]]
        multipliers = np.asarray(
            [row["soft_multiplier"] for row in selected], dtype=float
        )
        return {
            "n": len(selected),
            "baseline_Recall@10": float(np.mean([r["baseline_Recall@10"] for r in selected])),
            "baseline_NDCG@10": float(np.mean([r["baseline_NDCG@10"] for r in selected])),
            "baseline_Recall@50": float(np.mean([r["baseline_Recall@50"] for r in selected])),
            "candidate_Recall@10": float(np.mean([r["candidate_Recall@10"] for r in selected])),
            "candidate_NDCG@10": float(np.mean([r["candidate_NDCG@10"] for r in selected])),
            "candidate_Recall@50": float(np.mean([r["candidate_Recall@50"] for r in selected])),
            "union_coverage": float(np.mean([r["union_covered"] for r in selected])),
            "changed_user_coverage": float(np.mean([r["changed"] for r in selected])),
            "changed_covered_user_coverage": (
                float(np.mean([r["changed"] for r in covered])) if covered else 0.0
            ),
            "broad_harm_rate": float(np.mean([r["broad_harm"] for r in selected])),
            "mean_gate_probability": float(
                np.mean([r["gate_probability"] for r in selected])
            ),
            "mean_soft_multiplier": float(multipliers.mean()),
            "soft_multiplier_quantiles": [
                float(value) for value in np.quantile(multipliers, [0.0, 0.25, 0.5, 0.75, 1.0])
            ],
        }

    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize([r for r in rows if r["target_group"] == group])
    return groups, rows


def select_domain_alphas(training: dict, config: dict) -> tuple[dict, dict]:
    selected, audit = {}, {}
    raw_candidates = config["soft_weight"]["alpha_candidates"]
    if 1.0 not in [float(value) for value in raw_candidates]:
        raise ValueError("alpha=1 identity control is required")
    for dataset in config["datasets"]:
        candidates = []
        for raw_alpha in raw_candidates:
            alpha = float(raw_alpha)
            cells = [
                training[dataset]["seeds"][str(seed)]["calibration"][str(raw_alpha)]
                for seed in config["training_seeds"]
            ]
            safety_eligible = all(
                cell["overall"]["candidate_Recall@10"]
                >= cell["overall"]["baseline_Recall@10"]
                and cell["overall"]["candidate_Recall@50"]
                >= cell["overall"]["baseline_Recall@50"]
                and cell["tail"]["candidate_NDCG@10"]
                >= cell["tail"]["baseline_NDCG@10"]
                and cell["tail"]["candidate_Recall@50"]
                >= cell["tail"]["baseline_Recall@50"]
                and cell["overall"]["broad_harm_rate"]
                <= float(config["calibration_safety"]["broad_harm_max"])
                for cell in cells
            )
            candidates.append(
                {
                    "alpha": alpha,
                    "identity_control": alpha == 1.0,
                    "eligible": safety_eligible,
                    "mean_overall_ndcg10": float(
                        np.mean([c["overall"]["candidate_NDCG@10"] for c in cells])
                    ),
                    "mean_tail_ndcg10": float(
                        np.mean([c["tail"]["candidate_NDCG@10"] for c in cells])
                    ),
                    "mean_overall_recall50": float(
                        np.mean([c["overall"]["candidate_Recall@50"] for c in cells])
                    ),
                    "mean_soft_multiplier": float(
                        np.mean([c["overall"]["mean_soft_multiplier"] for c in cells])
                    ),
                    "maximum_broad_harm_rate": float(
                        np.max([c["overall"]["broad_harm_rate"] for c in cells])
                    ),
                }
            )
        identity = next(row for row in candidates if row["identity_control"])
        eligible_non_identity = [
            row
            for row in candidates
            if not row["identity_control"]
            and row["eligible"]
            and row["mean_overall_ndcg10"] > identity["mean_overall_ndcg10"]
        ]
        if eligible_non_identity:
            chosen = sorted(
                eligible_non_identity,
                key=lambda row: (
                    -row["mean_overall_ndcg10"],
                    -row["mean_tail_ndcg10"],
                    -row["mean_overall_recall50"],
                    row["maximum_broad_harm_rate"],
                    -row["alpha"],
                ),
            )[0]
        else:
            chosen = identity
        selected[dataset] = float(chosen["alpha"])
        audit[dataset] = candidates
    return selected, audit


def method_result(
    records: list[dict],
    residual_state: dict,
    gate_state: dict,
    config: dict,
    alpha: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    groups, rows = evaluate_soft_weight(
        records,
        residual_state,
        gate_state,
        float(config["residual"]["bound"]),
        alpha,
        device,
    )
    overall = groups["overall"]
    tail = groups["tail"]
    tail_rows = [row for row in rows if row["target_group"] == "tail"]
    return {
        "alpha": alpha,
        "groups": groups,
        "gains": {
            "overall_ndcg10_relative_gain": relative_gain(
                overall["baseline_NDCG@10"], overall["candidate_NDCG@10"]
            ),
            "overall_recall10_absolute_gain": (
                overall["candidate_Recall@10"] - overall["baseline_Recall@10"]
            ),
            "overall_recall50_absolute_gain": (
                overall["candidate_Recall@50"] - overall["baseline_Recall@50"]
            ),
            "tail_ndcg10_relative_gain": relative_gain(
                tail["baseline_NDCG@10"], tail["candidate_NDCG@10"]
            ),
            "tail_recall50_absolute_gain": (
                tail["candidate_Recall@50"] - tail["baseline_Recall@50"]
            ),
        },
        "bootstrap": {
            "overall_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(
                rows, "NDCG@10", True, seed + 11
            ),
            "overall_recall10_absolute_gain_ci95": paired_bootstrap_candidate(
                rows, "Recall@10", False, seed + 21
            ),
            "tail_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(
                tail_rows, "NDCG@10", True, seed + 31
            ),
        },
    }, rows


def load_locked_state(
    root_key: str,
    sha_key: str,
    filename: str,
    dataset: str,
    seed: int,
    config: dict,
) -> tuple[dict, str]:
    path = ROOT / config["inputs"][root_key] / dataset / filename.format(seed=seed)
    expected = config["inputs"][sha_key][dataset][str(seed)]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"locked state SHA mismatch {dataset}/{seed}: expected={expected} actual={actual}"
        )
    return torch.load(path, map_location="cpu"), actual


def validate_implementation(config: dict) -> None:
    expected = config["integrity"]["code_sha256"]
    actual = sha256(Path(__file__))
    if expected != actual:
        raise RuntimeError(
            f"implementation SHA mismatch: expected={expected} actual={actual}"
        )


def rows_identical(first: list[dict], second: list[dict]) -> bool:
    fields = (
        "sample_key",
        "baseline_rank",
        "candidate_rank",
        "baseline_Recall@10",
        "baseline_NDCG@10",
        "baseline_Recall@50",
        "candidate_Recall@10",
        "candidate_NDCG@10",
        "candidate_Recall@50",
        "changed",
        "broad_harm",
    )
    return len(first) == len(second) and all(
        all(left[field] == right[field] for field in fields)
        for left, right in zip(first, second)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v5 requires CUDA")
    config = json.loads(args.config.read_text())
    validate_implementation(config)
    validate_checkpoint_lineage(config)
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")

    training = {}
    for dataset in config["datasets"]:
        metadata, fit_records, calibration_records = build_training_records(
            dataset, config, p0_config, device
        )
        seeds = {}
        for seed in config["training_seeds"]:
            residual_state, residual_sha = load_locked_state(
                "v3_residual_root",
                "expected_residual_sha256",
                "residual_seed{seed}.pt",
                dataset,
                int(seed),
                config,
            )
            gate_state, gate_sha = load_locked_state(
                "v4_gate_root",
                "expected_gate_sha256",
                "gate_seed{seed}.pt",
                dataset,
                int(seed),
                config,
            )
            calibration = {}
            for alpha in config["soft_weight"]["alpha_candidates"]:
                groups, _ = evaluate_soft_weight(
                    calibration_records,
                    residual_state,
                    gate_state,
                    float(config["residual"]["bound"]),
                    float(alpha),
                    device,
                )
                calibration[str(alpha)] = groups
            seeds[str(seed)] = {
                "calibration": calibration,
                "residual_sha256": residual_sha,
                "gate_sha256": gate_sha,
            }
            print(
                f"GACR_V5_CALIBRATION dataset={dataset} seed={seed}",
                flush=True,
            )
        training[dataset] = metadata | {"seeds": seeds}
        del fit_records, calibration_records
        torch.cuda.empty_cache()

    selected_alphas, alpha_selection = select_domain_alphas(training, config)
    validation = {}
    identity_checks = []
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(dataset, config, p0_config, device)
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = {}
        for seed in config["training_seeds"]:
            residual_state, _ = load_locked_state(
                "v3_residual_root",
                "expected_residual_sha256",
                "residual_seed{seed}.pt",
                dataset,
                int(seed),
                config,
            )
            gate_state, _ = load_locked_state(
                "v4_gate_root",
                "expected_gate_sha256",
                "gate_seed{seed}.pt",
                dataset,
                int(seed),
                config,
            )
            v3_result, v3_rows = v3_method_result(
                records,
                residual_state,
                config,
                float(config["v3_identity_budget"]),
                int(seed) + 1000,
                device,
            )
            identity_result, identity_rows = method_result(
                records,
                residual_state,
                gate_state,
                config,
                1.0,
                int(seed) + 1500,
                device,
            )
            identity = rows_identical(v3_rows, identity_rows)
            identity_checks.append(identity)
            if not identity:
                raise RuntimeError(f"alpha=1 identity failed: {dataset}/{seed}")
            v5_result, v5_rows = method_result(
                records,
                residual_state,
                gate_state,
                config,
                selected_alphas[dataset],
                int(seed) + 2000,
                device,
            )
            seeds[str(seed)] = {
                "gacr_v3": v3_result,
                "alpha1_identity": identity_result | {"matches_v3": identity},
                "gacr_v5": v5_result,
            }
            for method, rows in (("gacr_v3", v3_rows), ("gacr_v5", v5_rows)):
                path = output_dir / f"{method}_seed{seed}_per_user.csv"
                with path.open("w", newline="") as handle:
                    serial = serializable_rows(rows)
                    writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
                    writer.writeheader()
                    writer.writerows(serial)
                seeds[str(seed)][method]["per_user_sha256"] = sha256(path)
        validation[dataset] = metadata | {"seeds": seeds}
        del records
        torch.cuda.empty_cache()

    integrity = {
        "fit_calibration_user_disjoint": all(
            training[d]["fit_calibration_user_overlap"] == 0 for d in config["datasets"]
        ),
        "parent_checkpoint_sha_unchanged": all(
            training[d]["parent_checkpoint_sha256_before"]
            == training[d]["parent_checkpoint_sha256_after"]
            == validation[d]["parent_checkpoint_sha256_before"]
            == validation[d]["parent_checkpoint_sha256_after"]
            for d in config["datasets"]
        ),
        "fresh_validation_zero_overlap": all(
            validation[d]["gcdh_or_training_overlap"] == 0
            and validation[d]["prior_gacr_p0_overlap"] == 0
            for d in config["datasets"]
        ),
        "alpha1_exact_v3_identity": all(identity_checks),
        "frozen_v3_residuals_only": True,
        "frozen_v4_gates_only": True,
        "target_free_soft_weight": True,
        "gate_optimizer_steps": 0,
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "single_changed_factor": "target_free_continuous_residual_multiplier",
        "selected_domain_alphas": selected_alphas,
        "alpha_selection": alpha_selection,
        "training": training,
        "validation": validation,
        "seed_stability": {
            method: summarize_seed_stability(validation, method)
            for method in ("gacr_v3", "gacr_v5")
        },
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

