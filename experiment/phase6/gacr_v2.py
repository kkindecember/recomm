#!/usr/bin/env python3
"""Phase 6 GACR-v2: calibrated bounded residual-strength growth pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gacr_p0 import (  # noqa: E402
    balanced_training_loss,
    relative_gain,
    select_fresh_validation_users,
    split_training_users,
    to_cpu_record,
)
from experiment.phase4.gacr_s0 import (  # noqa: E402
    BoundedResidualRanker,
    build_candidate_record,
    select_stratified_samples,
    stable_ranking,
)
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    build_validation_samples,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)


def rank_metrics(rank: int | None) -> tuple[float, float, float]:
    recall10 = float(rank is not None and rank <= 10)
    ndcg10 = 1.0 / math.log2(rank + 1) if recall10 else 0.0
    recall50 = float(rank is not None and rank <= 50)
    return recall10, ndcg10, recall50


def paired_bootstrap_candidate(
    rows: list[dict],
    field: str,
    relative: bool,
    seed: int,
    samples: int = 10000,
) -> list[float]:
    baseline = np.asarray([row[f"baseline_{field}"] for row in rows], dtype=float)
    candidate = np.asarray([row[f"candidate_{field}"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        baseline_mean = baseline[indices].mean()
        candidate_mean = candidate[indices].mean()
        values.append(
            relative_gain(baseline_mean, candidate_mean)
            if relative
            else candidate_mean - baseline_mean
        )
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def scaled_rank(
    record: dict,
    ranker: BoundedResidualRanker,
    scale: float,
    device: torch.device,
) -> int | None:
    if record["target_index"] is None:
        return None
    scores = record["base"].to(device) + float(scale) * ranker(
        record["features"].to(device)
    )
    return stable_ranking(scores).index(int(record["target_index"])) + 1


@torch.no_grad()
def evaluate_scale(
    records: list[dict],
    state: dict,
    bound: float,
    scale: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    ranker = BoundedResidualRanker(6, 16, bound).to(device)
    ranker.load_state_dict(state, strict=True)
    ranker.eval()
    rows = []
    for record in records:
        baseline_rank = record["gram_rank"]
        candidate_rank = scaled_rank(record, ranker, scale, device)
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
        }

    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize(
            [row for row in rows if row["target_group"] == group]
        )
    return groups, rows


def build_training_records(
    dataset: str,
    config: dict,
    p0_config: dict,
    device: torch.device,
) -> tuple[dict, list[dict], list[dict]]:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1" / "model.pt"
    )
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    fit_users, calibration_users = split_training_users(
        train_users, int(config["cohort_seed"]), dataset
    )
    fit_pool = build_train_samples(
        prepared["sequences"],
        fit_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    calibration_pool = build_train_samples(
        prepared["sequences"],
        calibration_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    fit_samples = select_stratified_samples(
        fit_pool,
        prepared["heads"],
        int(config["cohort_seed"]),
        f"{dataset}|gacr-p0-fit",
        int(config["fit_samples_per_group"]),
        int(config["fit_samples_per_group"]),
    )
    calibration_samples = select_stratified_samples(
        calibration_pool,
        prepared["heads"],
        int(config["cohort_seed"]),
        f"{dataset}|gacr-p0-calibration",
        int(config["calibration_samples_per_group"]),
        int(config["calibration_samples_per_group"]),
    )
    fit_records, calibration_records = [], []
    phase = {row["sample_key"]: "fit" for row in fit_samples}
    for row in calibration_samples:
        phase[row["sample_key"]] = "calibration"
    all_samples = fit_samples + calibration_samples
    for index, sample in enumerate(all_samples, 1):
        record = build_candidate_record(sample, prepared, config, device)
        if phase[sample["sample_key"]] == "fit":
            fit_records.append(record)
        else:
            calibration_records.append(record)
        if index % 64 == 0:
            print(
                f"GACR_V2_TRAIN_CANDIDATES dataset={dataset} "
                f"samples={index}/{len(all_samples)}",
                flush=True,
            )
    metadata = {
        "fit_samples": len(fit_records),
        "fit_covered": sum(
            record["target_index"] is not None for record in fit_records
        ),
        "calibration_samples": len(calibration_records),
        "calibration_covered": sum(
            record["target_index"] is not None for record in calibration_records
        ),
        "fit_user_sha256": stable_sha(fit_users),
        "calibration_user_sha256": stable_sha(calibration_users),
        "fit_calibration_user_overlap": len(fit_users & calibration_users),
        "parent_checkpoint_sha256_before": checkpoint_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
    }
    return metadata, fit_records, calibration_records


def train_seed(
    fit_records: list[dict],
    calibration_records: list[dict],
    config: dict,
    seed: int,
    device: torch.device,
) -> dict:
    residual = config["residual"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ranker = BoundedResidualRanker(
        6, int(residual["hidden_dim"]), float(residual["bound"])
    ).to(device)
    with torch.no_grad():
        zero_identity = float(
            np.mean(
                [
                    stable_ranking(record["base"])
                    == stable_ranking(record["base"] + ranker(record["features"]))
                    for record in fit_records + calibration_records
                ]
            )
        )
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=float(residual["learning_rate"]),
        weight_decay=float(residual["weight_decay"]),
    )
    first_loss = None
    last_loss = None
    for step in range(1, int(residual["fixed_training_step"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_training_loss(
            fit_records, ranker, float(residual["margin"]), device
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite GACR-v2 loss for seed={seed}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise ValueError(f"non-finite GACR-v2 gradient for seed={seed}")
        optimizer.step()
        first_loss = float(loss) if first_loss is None else first_loss
        last_loss = float(loss)
    state = {
        key: value.detach().cpu().clone()
        for key, value in ranker.state_dict().items()
    }
    calibration = {}
    for scale in config["deployment_scale_candidates"]:
        groups, _ = evaluate_scale(
            calibration_records,
            state,
            float(residual["bound"]),
            float(scale),
            device,
        )
        calibration[str(scale)] = groups
    return {
        "state": state,
        "zero_residual_identity_rate": zero_identity,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "calibration": calibration,
    }


def select_shared_scale(training: dict, config: dict) -> tuple[float, list[dict]]:
    rows = []
    for raw_scale in config["deployment_scale_candidates"]:
        scale = float(raw_scale)
        cells = []
        for dataset in config["datasets"]:
            for seed in config["training_seeds"]:
                groups = training[dataset]["seeds"][str(seed)]["calibration"][
                    str(raw_scale)
                ]
                overall = groups["overall"]
                tail = groups["tail"]
                cells.append(
                    {
                        "ndcg_gain": relative_gain(
                            overall["baseline_NDCG@10"],
                            overall["candidate_NDCG@10"],
                        ),
                        "recall_gain": (
                            overall["candidate_Recall@10"]
                            - overall["baseline_Recall@10"]
                        ),
                        "tail_gain": relative_gain(
                            tail["baseline_NDCG@10"],
                            tail["candidate_NDCG@10"],
                        ),
                        "harm": overall["broad_harm_rate"],
                        "changed": overall["changed_user_coverage"],
                    }
                )
        eligible = all(
            cell["recall_gain"] >= 0
            and cell["tail_gain"] >= 0
            and cell["harm"] <= float(config["calibration_safety"]["broad_harm_max"])
            for cell in cells
        )
        rows.append(
            {
                "scale": scale,
                "eligible": eligible,
                "mean_relative_ndcg10_gain": float(
                    np.mean([cell["ndcg_gain"] for cell in cells])
                ),
                "mean_changed_user_coverage": float(
                    np.mean([cell["changed"] for cell in cells])
                ),
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    pool = eligible_rows if eligible_rows else rows
    selected = sorted(
        pool,
        key=lambda row: (
            -row["mean_relative_ndcg10_gain"],
            -row["mean_changed_user_coverage"],
            abs(row["scale"] - 1.0),
        ),
    )[0]["scale"]
    return float(selected), rows


def build_validation_records(
    dataset: str,
    config: dict,
    p0_config: dict,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1" / "model.pt"
    )
    before_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    gcdh_validation = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "validation_users.txt"
    )
    all_users = set(prepared["sequences"])
    # Do not use ``dict.get(key, config[...])`` here: Python evaluates the
    # default expression eagerly, including for newer configs that already
    # provide ``prior_validation_salts``.
    if "prior_validation_salts" in config:
        prior_salts = list(config["prior_validation_salts"])
    else:
        prior_salts = [config["prior_gacr_p0_validation_salt"]]
    exclusions = train_users | gcdh_validation
    prior_validation_users: set[str] = set()
    for prior_salt in prior_salts:
        prior_users = set(
            select_fresh_validation_users(
                all_users,
                exclusions,
                dataset,
                prior_salt,
                int(config["validation_users_per_dataset"]),
            )
        )
        prior_validation_users |= prior_users
        exclusions |= prior_users
    validation_users = select_fresh_validation_users(
        all_users,
        exclusions,
        dataset,
        config["validation_salt"],
        int(config["validation_users_per_dataset"]),
    )
    samples = build_validation_samples(
        prepared["sequences"],
        set(validation_users),
        prepared["item2input"],
        prepared["item2lexid"],
    )
    records = []
    for index, sample in enumerate(samples, 1):
        records.append(
            to_cpu_record(build_candidate_record(sample, prepared, config, device))
        )
        if index % 64 == 0:
            print(
                f"GACR_V2_VALIDATION dataset={dataset} "
                f"users={index}/{len(samples)}",
                flush=True,
            )
    metadata = {
        "users": len(records),
        "validation_user_sha256": stable_sha(set(validation_users)),
        "gcdh_or_training_overlap": len(set(validation_users) & (train_users | gcdh_validation)),
        "prior_gacr_p0_overlap": len(set(validation_users) & prior_validation_users),
        "prior_validation_cohorts_excluded": len(prior_salts),
        "parent_checkpoint_sha256_before": before_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
    }
    del prepared
    torch.cuda.empty_cache()
    return metadata, records


def serializable_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        copied = dict(row)
        copied["baseline_rank"] = (
            "" if copied["baseline_rank"] is None else copied["baseline_rank"]
        )
        copied["candidate_rank"] = (
            "" if copied["candidate_rank"] is None else copied["candidate_rank"]
        )
        result.append(copied)
    return result


def method_result(
    records: list[dict],
    state: dict,
    config: dict,
    scale: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    groups, rows = evaluate_scale(
        records,
        state,
        float(config["residual"]["bound"]),
        scale,
        device,
    )
    overall = groups["overall"]
    tail_rows = [row for row in rows if row["target_group"] == "tail"]
    result = {
        "scale": scale,
        "groups": groups,
        "gains": {
            "overall_ndcg10_relative_gain": relative_gain(
                overall["baseline_NDCG@10"],
                overall["candidate_NDCG@10"],
            ),
            "overall_recall10_absolute_gain": (
                overall["candidate_Recall@10"]
                - overall["baseline_Recall@10"]
            ),
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


def summarize_seed_stability(validation: dict, method: str) -> dict:
    values = [
        validation[dataset]["seeds"][str(seed)][method]["gains"][
            "overall_ndcg10_relative_gain"
        ]
        for dataset in validation
        for seed in validation[dataset]["seeds"]
    ]
    return {
        "domain_seed_cells": len(values),
        "mean_overall_ndcg10_relative_gain": float(np.mean(values)),
        "minimum_overall_ndcg10_relative_gain": float(np.min(values)),
        "maximum_overall_ndcg10_relative_gain": float(np.max(values)),
        "positive_cell_fraction": float(np.mean(np.asarray(values) > 0)),
    }


def validate_checkpoint_lineage(config: dict) -> None:
    checkpoint_root = ROOT / config["inputs"]["checkpoint_root"]
    expected = config["inputs"]["expected_c1_checkpoint_sha256"]
    problems = []
    for dataset in config["datasets"]:
        checkpoint = checkpoint_root / dataset / "C1" / "model.pt"
        if not checkpoint.is_file():
            problems.append(f"{dataset}:missing:{checkpoint}")
            continue
        actual = sha256(checkpoint)
        if actual != expected[dataset]:
            problems.append(
                f"{dataset}:sha256_mismatch:expected={expected[dataset]}:actual={actual}"
            )
    if problems:
        raise RuntimeError("checkpoint lineage gate failed: " + "; ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v2 requires CUDA")
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
                fit_records,
                calibration_records,
                config,
                int(seed),
                device,
            )
            seed_results[str(seed)] = result
            print(
                f"GACR_V2_TRAIN dataset={dataset} seed={seed} "
                f"loss={result['last_loss']:.6f}",
                flush=True,
            )
        training[dataset] = metadata | {"seeds": seed_results}
        del fit_records, calibration_records
        torch.cuda.empty_cache()

    selected_scale, scale_selection = select_shared_scale(training, config)
    validation = {}
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(
            dataset, config, p0_config, device
        )
        seed_results = {}
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for seed in config["training_seeds"]:
            state = training[dataset]["seeds"][str(seed)]["state"]
            p0_result, p0_rows = method_result(
                records,
                state,
                config,
                float(config["matched_p0_scale"]),
                int(seed),
                device,
            )
            v2_result, v2_rows = method_result(
                records,
                state,
                config,
                selected_scale,
                int(seed) + 1000,
                device,
            )
            seed_results[str(seed)] = {
                "matched_p0": p0_result,
                "gacr_v2": v2_result,
            }
            for method, rows in (("matched_p0", p0_rows), ("gacr_v2", v2_rows)):
                path = output_dir / f"{method}_seed{seed}_per_user.csv"
                with path.open("w", newline="") as handle:
                    serial = serializable_rows(rows)
                    writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
                    writer.writeheader()
                    writer.writerows(serial)
                seed_results[str(seed)][method]["per_user_sha256"] = sha256(path)
            checkpoint_path = output_dir / f"residual_seed{seed}.pt"
            torch.save(state, checkpoint_path)
            seed_results[str(seed)]["residual_checkpoint_sha256"] = sha256(
                checkpoint_path
            )
        validation[dataset] = metadata | {"seeds": seed_results}
        del records
        torch.cuda.empty_cache()

    integrity = {
        "fit_calibration_user_disjoint": all(
            training[dataset]["fit_calibration_user_overlap"] == 0
            for dataset in config["datasets"]
        ),
        "parent_checkpoint_sha_unchanged": all(
            training[dataset]["parent_checkpoint_sha256_before"]
            == training[dataset]["parent_checkpoint_sha256_after"]
            == validation[dataset]["parent_checkpoint_sha256_before"]
            == validation[dataset]["parent_checkpoint_sha256_after"]
            for dataset in config["datasets"]
        ),
        "zero_residual_identity": all(
            training[dataset]["seeds"][str(seed)]["zero_residual_identity_rate"]
            == 1.0
            for dataset in config["datasets"]
            for seed in config["training_seeds"]
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
    training_summary = {}
    for dataset, result in training.items():
        training_summary[dataset] = {
            key: value for key, value in result.items() if key != "seeds"
        }
        training_summary[dataset]["seeds"] = {
            seed: {
                key: value
                for key, value in seed_result.items()
                if key != "state"
            }
            for seed, seed_result in result["seeds"].items()
        }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "matched_p0_scale": float(config["matched_p0_scale"]),
        "selected_v2_scale": selected_scale,
        "scale_selection": scale_selection,
        "training": training_summary,
        "validation": validation,
        "seed_stability": {
            "matched_p0": summarize_seed_stability(validation, "matched_p0"),
            "gacr_v2": summarize_seed_stability(validation, "gacr_v2"),
        },
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
