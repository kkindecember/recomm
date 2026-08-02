#!/usr/bin/env python3
"""Phase 6 GACR-v7: metric-aligned full-fit residual ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gacr_p0 import relative_gain  # noqa: E402
from experiment.phase4.gacr_s0 import BoundedResidualRanker, stable_ranking  # noqa: E402
from experiment.phase4.gcdh_p0 import ROOT, sha256, write_json  # noqa: E402
from experiment.phase6.gacr_v2 import (  # noqa: E402
    build_validation_records,
    evaluate_scale,
    method_result,
    serializable_rows,
    validate_checkpoint_lineage,
)
from experiment.phase6.gacr_v6 import (  # noqa: E402
    add_standard_metrics,
    build_full_training_records,
)


def cutoff_discount(rank: int) -> float:
    """Frozen NDCG@10 discount used by the v7 loss."""
    return 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0


def metric_pair_weights(base: torch.Tensor, target_index: int) -> torch.Tensor:
    """Return the preregistered NDCG@10/Recall@50 pair weights."""
    ranking = stable_ranking(base.detach().cpu())
    ranks = {candidate: rank for rank, candidate in enumerate(ranking, 1)}
    target_rank = ranks[target_index]
    weights = []
    for candidate in range(base.numel()):
        candidate_rank = ranks[candidate]
        if candidate == target_index:
            weight = 0.0
        else:
            weight = abs(cutoff_discount(target_rank) - cutoff_discount(candidate_rank))
            weight += 0.25 * abs(
                float(target_rank <= 50) - float(candidate_rank <= 50)
            )
        weights.append(weight)
    return torch.tensor(weights, dtype=base.dtype, device=base.device)


def metric_aligned_pairwise_loss(
    base: torch.Tensor,
    residual: torch.Tensor,
    target_index: int,
) -> torch.Tensor | None:
    """Weighted pairwise logistic loss for one covered record."""
    weights = metric_pair_weights(base, target_index)
    denominator = weights.sum()
    if float(denominator.detach()) == 0.0:
        return None
    scores = base + residual
    pair_losses = F.softplus(scores - scores[target_index])
    return torch.sum(weights * pair_losses) / denominator


def train_metric_aligned_seed(
    fit_records: list[dict],
    calibration_records: list[dict],
    config: dict,
    seed: int,
    device: torch.device,
    chunk_size: int = 128,
) -> dict:
    """Train v7 with exact full-batch, equal-weight head/tail gradients."""
    residual_config = config["residual"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ranker = BoundedResidualRanker(
        6,
        int(residual_config["hidden_dim"]),
        float(residual_config["bound"]),
    ).to(device)
    with torch.no_grad():
        identity_checks = []
        for record in fit_records + calibration_records:
            base = record["base"].to(device)
            features = record["features"].to(device)
            identity_checks.append(
                stable_ranking(base) == stable_ranking(base + ranker(features))
            )
        zero_identity = float(np.mean(identity_checks))

    grouped: dict[str, list[dict]] = {}
    zero_weight_records = {"head": 0, "tail": 0}
    for group in ("head", "tail"):
        effective = []
        for record in fit_records:
            if record["target_index"] is None or record["target_group"] != group:
                continue
            weights = metric_pair_weights(
                record["base"], int(record["target_index"])
            )
            if float(weights.sum()) == 0.0:
                zero_weight_records[group] += 1
            else:
                effective.append(record)
        grouped[group] = effective
    if any(not records for records in grouped.values()):
        raise ValueError("v7 training has an empty effective head/tail group")

    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=float(residual_config["learning_rate"]),
        weight_decay=float(residual_config["weight_decay"]),
    )
    first_loss = last_loss = last_gradient_norm = None
    steps = int(residual_config["fixed_training_step"])
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        scalar_loss = 0.0
        for group in ("head", "tail"):
            records = grouped[group]
            denominator = 2.0 * len(records)
            for start in range(0, len(records), chunk_size):
                losses = []
                for record in records[start : start + chunk_size]:
                    value = metric_aligned_pairwise_loss(
                        record["base"].to(device),
                        ranker(record["features"].to(device)),
                        int(record["target_index"]),
                    )
                    if value is None:
                        raise AssertionError("zero-weight record escaped preprocessing")
                    losses.append(value)
                chunk_loss = torch.stack(losses).sum() / denominator
                if not torch.isfinite(chunk_loss):
                    raise ValueError(f"non-finite GACR-v7 loss seed={seed}")
                chunk_loss.backward()
                scalar_loss += float(chunk_loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise ValueError(f"non-finite GACR-v7 gradient seed={seed}")
        optimizer.step()
        first_loss = scalar_loss if first_loss is None else first_loss
        last_loss = scalar_loss
        last_gradient_norm = float(norm)
        print(
            f"GACR_V7_TRAIN seed={seed} step={step}/{steps} loss={scalar_loss:.6f}",
            flush=True,
        )

    state = {
        key: value.detach().cpu().clone()
        for key, value in ranker.state_dict().items()
    }
    calibration, _ = evaluate_scale(
        calibration_records,
        state,
        float(residual_config["bound"]),
        float(config["deployment_scale"]),
        device,
    )
    finite_state = all(torch.isfinite(value).all().item() for value in state.values())
    return {
        "state": state,
        "optimizer_steps": steps,
        "effective_head_records": len(grouped["head"]),
        "effective_tail_records": len(grouped["tail"]),
        "zero_weight_head_records": zero_weight_records["head"],
        "zero_weight_tail_records": zero_weight_records["tail"],
        "zero_residual_identity_rate": zero_identity,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "last_gradient_norm": last_gradient_norm,
        "finite_checkpoint": finite_state,
        "calibration": calibration,
    }


def assess_calibration_noninferiority(groups: dict, config: dict) -> dict:
    """Apply the frozen pre-validation safety gate to one domain/seed cell."""
    gate = config["calibration_noninferiority"]
    overall = groups["overall"]
    tail = groups["tail"]
    observed = {
        "broad_harm_rate": overall["broad_harm_rate"],
        "overall_recall10_absolute_delta": (
            overall["candidate_Recall@10"] - overall["baseline_Recall@10"]
        ),
        "overall_recall50_absolute_delta": (
            overall["candidate_Recall@50"] - overall["baseline_Recall@50"]
        ),
        "tail_recall50_absolute_delta": (
            tail["candidate_Recall@50"] - tail["baseline_Recall@50"]
        ),
        "tail_ndcg10_absolute_delta": (
            tail["candidate_NDCG@10"] - tail["baseline_NDCG@10"]
        ),
    }
    tolerance = 1e-12
    checks = {
        "broad_harm": observed["broad_harm_rate"]
        <= float(gate["broad_harm_max"]) + tolerance,
        "overall_recall10": observed["overall_recall10_absolute_delta"]
        >= float(gate["overall_recall10_absolute_delta_min"]) - tolerance,
        "overall_recall50": observed["overall_recall50_absolute_delta"]
        >= float(gate["overall_recall50_absolute_delta_min"]) - tolerance,
        "tail_recall50": observed["tail_recall50_absolute_delta"]
        >= float(gate["tail_recall50_absolute_delta_min"]) - tolerance,
        "tail_ndcg10": observed["tail_ndcg10_absolute_delta"]
        >= float(gate["tail_ndcg10_absolute_delta_min"]) - tolerance,
    }
    return {"eligible": all(checks.values()), "observed": observed, "checks": checks}


def compare_aligned_methods(
    left_name: str,
    left_rows: list[dict],
    right_name: str,
    right_rows: list[dict],
) -> dict:
    if [row["sample_key"] for row in left_rows] != [
        row["sample_key"] for row in right_rows
    ]:
        raise ValueError(f"{left_name}/{right_name} per-user rows are not aligned")
    output = {}
    for group in ("overall", "head", "tail"):
        left_group = left_rows if group == "overall" else [
            row for row in left_rows if row["target_group"] == group
        ]
        right_group = right_rows if group == "overall" else [
            row for row in right_rows if row["target_group"] == group
        ]
        output[group] = {}
        for cutoff in (5, 10):
            for metric in ("Recall", "NDCG"):
                key = f"candidate_{metric}@{cutoff}"
                left_value = float(np.mean([row[key] for row in left_group]))
                right_value = float(np.mean([row[key] for row in right_group]))
                output[group][f"{metric}@{cutoff}"] = {
                    left_name: left_value,
                    right_name: right_value,
                    "absolute_delta": right_value - left_value,
                    "relative_gain": relative_gain(left_value, right_value),
                }
    return output


def load_locked_state(config: dict, version: str, dataset: str, seed: int) -> dict:
    root = ROOT / config["inputs"][f"{version}_residual_root"]
    path = root / dataset / f"residual_seed{seed}.pt"
    expected = config["inputs"][f"expected_{version}_residual_sha256"][dataset][
        str(seed)
    ]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{version} residual SHA mismatch {dataset}/{seed}: "
            f"expected={expected} actual={actual}"
        )
    return torch.load(path, map_location="cpu")


def training_summary_without_states(training: dict) -> dict:
    output = {}
    for dataset, result in training.items():
        output[dataset] = {key: value for key, value in result.items() if key != "seeds"}
        output[dataset]["seeds"] = {
            seed: {key: value for key, value in cell.items() if key != "state"}
            for seed, cell in result["seeds"].items()
        }
    return output


def write_rows(path: Path, rows: list[dict]) -> str:
    serial = serializable_rows(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)
    return sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v7 requires CUDA")
    config = json.loads(args.config.read_text())
    validate_checkpoint_lineage(config)
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")

    training = {}
    calibration_cells = []
    for dataset in config["datasets"]:
        metadata, fit_records, calibration_records = build_full_training_records(
            dataset, config, p0_config, device
        )
        seeds = {}
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for seed in config["training_seeds"]:
            trained = train_metric_aligned_seed(
                fit_records, calibration_records, config, int(seed), device
            )
            checkpoint = output_dir / f"residual_seed{seed}.pt"
            torch.save(trained["state"], checkpoint)
            trained["residual_checkpoint_sha256"] = sha256(checkpoint)
            trained["calibration_noninferiority"] = assess_calibration_noninferiority(
                trained["calibration"], config
            )
            calibration_cells.append(
                trained["finite_checkpoint"]
                and math.isfinite(float(trained["first_loss"]))
                and math.isfinite(float(trained["last_loss"]))
                and math.isfinite(float(trained["last_gradient_norm"]))
                and trained["calibration_noninferiority"]["eligible"]
            )
            seeds[str(seed)] = trained
        training[dataset] = metadata | {"seeds": seeds}
        del fit_records, calibration_records
        torch.cuda.empty_cache()

    training_summary = training_summary_without_states(training)
    prevalidation_integrity = {
        "all_fit_records_used": True,
        "fit_calibration_user_disjoint": all(
            training[d]["fit_calibration_user_overlap"] == 0
            for d in config["datasets"]
        ),
        "parent_checkpoint_sha_unchanged_during_training": all(
            training[d]["parent_checkpoint_sha256_before"]
            == training[d]["parent_checkpoint_sha256_after"]
            for d in config["datasets"]
        ),
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    calibration_eligible = all(calibration_cells) and all(
        prevalidation_integrity.values()
    )
    if not calibration_eligible:
        summary = {
            "experiment_id": config["experiment_id"],
            "result_status": "STOPPED_BEFORE_FRESH_VALIDATION_CALIBRATION_GATE_FAILED",
            "single_changed_factor": config["single_changed_factor"],
            "calibration_eligible": False,
            "training": training_summary,
            "validation": {},
            "integrity": prevalidation_integrity,
        }
        write_json(args.output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    validation = {}
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(dataset, config, p0_config, device)
        output_dir = args.output_root / dataset
        seeds = {}
        for seed in config["training_seeds"]:
            states = {
                "gacr_v3": load_locked_state(config, "v3", dataset, int(seed)),
                "gacr_v6": load_locked_state(config, "v6", dataset, int(seed)),
                "gacr_v7": training[dataset]["seeds"][str(seed)]["state"],
            }
            results = {}
            enriched_rows = {}
            for offset, method in enumerate(("gacr_v3", "gacr_v6", "gacr_v7")):
                result, rows = method_result(
                    records,
                    states[method],
                    config,
                    float(config["deployment_scale"]),
                    int(seed) + 1000 * offset,
                    device,
                )
                result, rows = add_standard_metrics(result, rows)
                result["per_user_sha256"] = write_rows(
                    output_dir / f"{method}_seed{seed}_per_user.csv", rows
                )
                results[method] = result
                enriched_rows[method] = rows
            seeds[str(seed)] = results | {
                "incremental_v7_vs_v3": compare_aligned_methods(
                    "v3", enriched_rows["gacr_v3"], "v7", enriched_rows["gacr_v7"]
                ),
                "incremental_v7_vs_v6": compare_aligned_methods(
                    "v6", enriched_rows["gacr_v6"], "v7", enriched_rows["gacr_v7"]
                ),
            }
        validation[dataset] = metadata | {"seeds": seeds}
        del records
        torch.cuda.empty_cache()

    integrity = prevalidation_integrity | {
        "calibration_noninferiority_passed_all_cells": True,
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
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "single_changed_factor": config["single_changed_factor"],
        "calibration_eligible": True,
        "training": training_summary,
        "validation": validation,
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
