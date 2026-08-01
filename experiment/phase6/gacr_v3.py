#!/usr/bin/env python3
"""Phase 6 GACR-v3: target-free per-user residual-spread safety attenuation."""

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
from experiment.phase4.gacr_s0 import BoundedResidualRanker, stable_ranking  # noqa: E402
from experiment.phase4.gcdh_p0 import ROOT, sha256, write_json  # noqa: E402
from experiment.phase6.gacr_v2 import (  # noqa: E402
    build_training_records,
    build_validation_records,
    method_result,
    paired_bootstrap_candidate,
    rank_metrics,
    serializable_rows,
    summarize_seed_stability,
    train_seed,
    validate_checkpoint_lineage,
)


def residual_safety_multiplier(residual: torch.Tensor, budget: float) -> float:
    """Cap pairwise residual spread without using the target or any label."""
    if budget <= 0.0:
        return 0.0
    spread = float(residual.max() - residual.min())
    if spread <= 0.0:
        return 1.0
    return min(1.0, float(budget) / spread)


@torch.no_grad()
def evaluate_budget(
    records: list[dict],
    state: dict,
    bound: float,
    budget: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    ranker = BoundedResidualRanker(6, 16, bound).to(device)
    ranker.load_state_dict(state, strict=True)
    ranker.eval()
    rows = []
    for record in records:
        residual = ranker(record["features"].to(device))
        spread = float(residual.max() - residual.min())
        multiplier = residual_safety_multiplier(residual, budget)
        baseline_rank = record["gram_rank"]
        if record["target_index"] is None:
            candidate_rank = None
        else:
            scores = record["base"].to(device) + multiplier * residual
            candidate_rank = (
                stable_ranking(scores).index(int(record["target_index"])) + 1
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
                "residual_spread": spread,
                "safety_multiplier": multiplier,
                "attenuated": int(multiplier < 1.0),
            }
        )

    def summarize(selected: list[dict]) -> dict:
        covered = [row for row in selected if row["union_covered"]]
        return {
            "n": len(selected),
            "baseline_Recall@10": float(
                np.mean([row["baseline_Recall@10"] for row in selected])
            ),
            "baseline_NDCG@10": float(
                np.mean([row["baseline_NDCG@10"] for row in selected])
            ),
            "baseline_Recall@50": float(
                np.mean([row["baseline_Recall@50"] for row in selected])
            ),
            "candidate_Recall@10": float(
                np.mean([row["candidate_Recall@10"] for row in selected])
            ),
            "candidate_NDCG@10": float(
                np.mean([row["candidate_NDCG@10"] for row in selected])
            ),
            "candidate_Recall@50": float(
                np.mean([row["candidate_Recall@50"] for row in selected])
            ),
            "union_coverage": float(
                np.mean([row["union_covered"] for row in selected])
            ),
            "changed_user_coverage": float(
                np.mean([row["changed"] for row in selected])
            ),
            "changed_covered_user_coverage": float(
                np.mean([row["changed"] for row in covered])
            )
            if covered
            else 0.0,
            "broad_harm_rate": float(
                np.mean([row["broad_harm"] for row in selected])
            ),
            "attenuation_rate": float(
                np.mean([row["attenuated"] for row in selected])
            ),
            "mean_safety_multiplier": float(
                np.mean([row["safety_multiplier"] for row in selected])
            ),
            "mean_residual_spread": float(
                np.mean([row["residual_spread"] for row in selected])
            ),
        }

    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize(
            [row for row in rows if row["target_group"] == group]
        )
    return groups, rows


def select_domain_budgets(training: dict, config: dict) -> tuple[dict, dict]:
    selected, audit = {}, {}
    tolerance = 1e-12
    for dataset in config["datasets"]:
        rows = []
        for raw_budget in config["safety_budget_candidates"]:
            cells = []
            for seed in config["training_seeds"]:
                groups = training[dataset]["seeds"][str(seed)][
                    "budget_calibration"
                ][str(raw_budget)]
                overall, tail = groups["overall"], groups["tail"]
                cells.append(
                    {
                        "ndcg_gain": relative_gain(
                            overall["baseline_NDCG@10"],
                            overall["candidate_NDCG@10"],
                        ),
                        "recall_gain": overall["candidate_Recall@10"]
                        - overall["baseline_Recall@10"],
                        "tail_gain": relative_gain(
                            tail["baseline_NDCG@10"],
                            tail["candidate_NDCG@10"],
                        ),
                        "harm": overall["broad_harm_rate"],
                        "changed": overall["changed_user_coverage"],
                        "multiplier": overall["mean_safety_multiplier"],
                    }
                )
            eligible = all(
                cell["recall_gain"] >= -tolerance
                and cell["tail_gain"] >= -tolerance
                and cell["harm"]
                <= float(config["calibration_safety"]["broad_harm_max"])
                for cell in cells
            )
            rows.append(
                {
                    "budget": float(raw_budget),
                    "eligible": eligible,
                    "mean_relative_ndcg10_gain": float(
                        np.mean([cell["ndcg_gain"] for cell in cells])
                    ),
                    "mean_changed_user_coverage": float(
                        np.mean([cell["changed"] for cell in cells])
                    ),
                    "mean_safety_multiplier": float(
                        np.mean([cell["multiplier"] for cell in cells])
                    ),
                    "maximum_broad_harm_rate": float(
                        np.max([cell["harm"] for cell in cells])
                    ),
                }
            )
        eligible_rows = [row for row in rows if row["eligible"]]
        if not eligible_rows:
            raise RuntimeError(f"identity safety candidate unexpectedly failed: {dataset}")
        chosen = sorted(
            eligible_rows,
            key=lambda row: (
                -row["mean_relative_ndcg10_gain"],
                row["maximum_broad_harm_rate"],
                -row["mean_changed_user_coverage"],
                row["budget"],
            ),
        )[0]
        selected[dataset] = float(chosen["budget"])
        audit[dataset] = rows
    return selected, audit


def v3_method_result(
    records: list[dict],
    state: dict,
    config: dict,
    budget: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    groups, rows = evaluate_budget(
        records, state, float(config["residual"]["bound"]), budget, device
    )
    overall = groups["overall"]
    tail_rows = [row for row in rows if row["target_group"] == "tail"]
    result = {
        "safety_budget": budget,
        "groups": groups,
        "gains": {
            "overall_ndcg10_relative_gain": relative_gain(
                overall["baseline_NDCG@10"], overall["candidate_NDCG@10"]
            ),
            "overall_recall10_absolute_gain": overall["candidate_Recall@10"]
            - overall["baseline_Recall@10"],
            "tail_ndcg10_relative_gain": relative_gain(
                groups["tail"]["baseline_NDCG@10"],
                groups["tail"]["candidate_NDCG@10"],
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
    }
    return result, rows


def state_matches_v2(state: dict, path: Path) -> bool:
    prior = torch.load(path, map_location="cpu")
    return set(state) == set(prior) and all(
        torch.equal(state[key], prior[key]) for key in state
    )


def summarize_incremental(validation: dict) -> dict:
    result = {}
    for dataset, dataset_result in validation.items():
        ndcg, recall, tail = [], [], []
        for seed_result in dataset_result["seeds"].values():
            v2 = seed_result["gacr_v2_unclipped"]["groups"]
            v3 = seed_result["gacr_v3"]["groups"]
            ndcg.append(
                v3["overall"]["candidate_NDCG@10"]
                - v2["overall"]["candidate_NDCG@10"]
            )
            recall.append(
                v3["overall"]["candidate_Recall@10"]
                - v2["overall"]["candidate_Recall@10"]
            )
            tail.append(
                v3["tail"]["candidate_NDCG@10"]
                - v2["tail"]["candidate_NDCG@10"]
            )
        result[dataset] = {
            "mean_ndcg10_absolute_delta_vs_v2": float(np.mean(ndcg)),
            "mean_recall10_absolute_delta_vs_v2": float(np.mean(recall)),
            "mean_tail_ndcg10_absolute_delta_vs_v2": float(np.mean(tail)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v3 requires CUDA")
    config = json.loads(args.config.read_text())
    validate_checkpoint_lineage(config)
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")

    training = {}
    for dataset in config["datasets"]:
        metadata, fit_records, calibration_records = build_training_records(
            dataset, config, p0_config, device
        )
        seed_results = {}
        for seed in config["training_seeds"]:
            result = train_seed(
                fit_records, calibration_records, config, int(seed), device
            )
            budget_calibration = {}
            for budget in config["safety_budget_candidates"]:
                groups, _ = evaluate_budget(
                    calibration_records,
                    result["state"],
                    float(config["residual"]["bound"]),
                    float(budget),
                    device,
                )
                budget_calibration[str(budget)] = groups
            result["budget_calibration"] = budget_calibration
            prior_path = (
                ROOT
                / config["inputs"]["v2_residual_root"]
                / dataset
                / f"residual_seed{seed}.pt"
            )
            result["state_matches_v2"] = state_matches_v2(
                result["state"], prior_path
            )
            seed_results[str(seed)] = result
            print(
                f"GACR_V3_TRAIN dataset={dataset} seed={seed} "
                f"loss={result['last_loss']:.6f} "
                f"state_matches_v2={result['state_matches_v2']}",
                flush=True,
            )
        training[dataset] = metadata | {"seeds": seed_results}
        del fit_records, calibration_records
        torch.cuda.empty_cache()

    selected_budgets, budget_selection = select_domain_budgets(training, config)
    validation = {}
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(
            dataset, config, p0_config, device
        )
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_results = {}
        for seed in config["training_seeds"]:
            state = training[dataset]["seeds"][str(seed)]["state"]
            p0_result, p0_rows = method_result(
                records, state, config, 1.0, int(seed), device
            )
            v3_result, v3_rows = v3_method_result(
                records,
                state,
                config,
                selected_budgets[dataset],
                int(seed) + 2000,
                device,
            )
            seed_results[str(seed)] = {
                "matched_p0": p0_result,
                "gacr_v2_unclipped": dict(p0_result),
                "gacr_v3": v3_result,
                "residual_state_matches_v2": training[dataset]["seeds"][str(seed)][
                    "state_matches_v2"
                ],
            }
            for method, rows in (
                ("matched_p0", p0_rows),
                ("gacr_v2_unclipped", p0_rows),
                ("gacr_v3", v3_rows),
            ):
                path = output_dir / f"{method}_seed{seed}_per_user.csv"
                with path.open("w", newline="") as handle:
                    serial = serializable_rows(rows)
                    writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
                    writer.writeheader()
                    writer.writerows(serial)
                seed_results[str(seed)][method]["per_user_sha256"] = sha256(path)
        validation[dataset] = metadata | {"seeds": seed_results}
        del records
        torch.cuda.empty_cache()

    training_summary = {}
    for dataset, result in training.items():
        training_summary[dataset] = {
            key: value for key, value in result.items() if key != "seeds"
        }
        training_summary[dataset]["seeds"] = {
            seed: {key: value for key, value in item.items() if key != "state"}
            for seed, item in result["seeds"].items()
        }
    integrity = {
        "fit_calibration_user_disjoint": all(
            training[d]["fit_calibration_user_overlap"] == 0
            for d in config["datasets"]
        ),
        "parent_checkpoint_sha_unchanged": all(
            training[d]["parent_checkpoint_sha256_before"]
            == training[d]["parent_checkpoint_sha256_after"]
            == validation[d]["parent_checkpoint_sha256_before"]
            == validation[d]["parent_checkpoint_sha256_after"]
            for d in config["datasets"]
        ),
        "residual_states_match_v2": all(
            training[d]["seeds"][str(seed)]["state_matches_v2"]
            for d in config["datasets"]
            for seed in config["training_seeds"]
        ),
        "fresh_validation_zero_overlap": all(
            validation[d]["gcdh_or_training_overlap"] == 0
            and validation[d]["prior_gacr_p0_overlap"] == 0
            for d in config["datasets"]
        ),
        "safety_gate_target_free": True,
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "single_changed_factor": "target_free_per_user_residual_spread_budget",
        "selected_domain_budgets": selected_budgets,
        "budget_selection": budget_selection,
        "training": training_summary,
        "validation": validation,
        "seed_stability": {
            method: summarize_seed_stability(validation, method)
            for method in ("matched_p0", "gacr_v2_unclipped", "gacr_v3")
        },
        "incremental_vs_v2": summarize_incremental(validation),
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
