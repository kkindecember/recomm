#!/usr/bin/env python3
"""GACR P0: frozen-head residual-ranking effect pilot on a fresh cohort."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gacr_s0 import (  # noqa: E402
    BoundedResidualRanker,
    build_candidate_record,
    hinge_loss,
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


def split_training_samples(
    samples: list[dict],
    head_items: set[str],
    seed: int,
    dataset: str,
    fit_per_group: int,
    calibration_per_group: int,
) -> tuple[list[dict], list[dict]]:
    fit, calibration = [], []
    for group, is_head in (("head", True), ("tail", False)):
        selected = [row for row in samples if (row["positive_item"] in head_items) == is_head]
        selected.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{dataset}|p0-split|{group}|{row['sample_key']}".encode()
            ).hexdigest()
        )
        required = fit_per_group + calibration_per_group
        if len(selected) != required:
            raise ValueError(f"unexpected {dataset}/{group} sample count")
        fit.extend(selected[:fit_per_group])
        calibration.extend(selected[fit_per_group:])
    return sorted(fit, key=lambda row: row["sample_key"]), sorted(
        calibration, key=lambda row: row["sample_key"]
    )


def split_training_users(
    users: set[str],
    seed: int,
    dataset: str,
    fit_fraction: float = 0.8,
) -> tuple[set[str], set[str]]:
    ordered = sorted(
        users,
        key=lambda user: hashlib.sha256(
            f"{seed}|{dataset}|p0-user-split|{user}".encode()
        ).hexdigest(),
    )
    cut = int(round(len(ordered) * fit_fraction))
    return set(ordered[:cut]), set(ordered[cut:])


def select_fresh_validation_users(
    all_users: set[str],
    excluded: set[str],
    dataset: str,
    salt: str,
    count: int,
) -> list[str]:
    candidates = sorted(
        all_users - excluded,
        key=lambda user: hashlib.sha256(
            f"{salt}|{dataset}|{user}".encode()
        ).hexdigest(),
    )
    if len(candidates) < count:
        raise ValueError(f"insufficient fresh validation users for {dataset}")
    return candidates[:count]


def to_cpu_record(record: dict) -> dict:
    copied = dict(record)
    copied["base"] = record["base"].cpu()
    copied["features"] = record["features"].cpu()
    return copied


def balanced_training_loss(
    records: list[dict],
    ranker: BoundedResidualRanker,
    margin: float,
    device: torch.device,
) -> torch.Tensor:
    group_losses = []
    for group in ("head", "tail"):
        selected = [
            record
            for record in records
            if record["target_index"] is not None and record["target_group"] == group
        ]
        if not selected:
            raise ValueError(f"no covered {group} training pairs")
        losses = [
            hinge_loss(
                record["base"].to(device),
                ranker(record["features"].to(device)),
                int(record["target_index"]),
                margin,
            )
            for record in selected
        ]
        group_losses.append(torch.stack(losses).mean())
    return torch.stack(group_losses).mean()


def rank_metrics(rank: int | None) -> tuple[float, float, float]:
    recall10 = float(rank is not None and rank <= 10)
    ndcg10 = 1.0 / math.log2(rank + 1) if recall10 else 0.0
    recall50 = float(rank is not None and rank <= 50)
    return recall10, ndcg10, recall50


@torch.no_grad()
def evaluate_records(
    records: list[dict],
    state: dict,
    bound: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    ranker = BoundedResidualRanker(6, 16, bound).to(device)
    ranker.load_state_dict(state, strict=True)
    ranker.eval()
    rows = []
    for record in records:
        baseline_rank = record["gram_rank"]
        if record["target_index"] is None:
            residual_rank = None
        else:
            scores = record["base"].to(device) + ranker(
                record["features"].to(device)
            )
            order = stable_ranking(scores)
            residual_rank = order.index(int(record["target_index"])) + 1
        b_r10, b_ndcg, b_r50 = rank_metrics(baseline_rank)
        r_r10, r_ndcg, r_r50 = rank_metrics(residual_rank)
        rows.append(
            {
                "sample_key": record["sample_key"],
                "target_group": record["target_group"],
                "baseline_rank": "" if baseline_rank is None else baseline_rank,
                "residual_rank": "" if residual_rank is None else residual_rank,
                "union_covered": int(record["target_index"] is not None),
                "baseline_Recall@10": b_r10,
                "baseline_NDCG@10": b_ndcg,
                "baseline_Recall@50": b_r50,
                "residual_Recall@10": r_r10,
                "residual_NDCG@10": r_ndcg,
                "residual_Recall@50": r_r50,
                "broad_harm": int(b_r10 == 1.0 and r_r10 == 0.0),
            }
        )

    def summarize(selected: list[dict]) -> dict:
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
            "residual_Recall@10": float(
                np.mean([row["residual_Recall@10"] for row in selected])
            ),
            "residual_NDCG@10": float(
                np.mean([row["residual_NDCG@10"] for row in selected])
            ),
            "residual_Recall@50": float(
                np.mean([row["residual_Recall@50"] for row in selected])
            ),
            "union_coverage": float(
                np.mean([row["union_covered"] for row in selected])
            ),
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


def relative_gain(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else float("inf")
    return candidate / baseline - 1.0


def paired_bootstrap(
    rows: list[dict],
    field: str,
    relative: bool,
    seed: int,
    samples: int = 10000,
) -> list[float]:
    baseline = np.asarray([row[f"baseline_{field}"] for row in rows], dtype=float)
    candidate = np.asarray([row[f"residual_{field}"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        b_mean = baseline[indices].mean()
        c_mean = candidate[indices].mean()
        values.append(relative_gain(b_mean, c_mean) if relative else c_mean - b_mean)
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def generate_training_phase(
    dataset: str,
    config: dict,
    p0_config: dict,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1" / "model.pt"
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    fit_users, calibration_users = split_training_users(
        train_users, int(config["seed"]), dataset
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
        int(config["seed"]),
        f"{dataset}|gacr-p0-fit",
        int(config["fit_samples_per_group"]),
        int(config["fit_samples_per_group"]),
    )
    calibration_samples = select_stratified_samples(
        calibration_pool,
        prepared["heads"],
        int(config["seed"]),
        f"{dataset}|gacr-p0-calibration",
        int(config["calibration_samples_per_group"]),
        int(config["calibration_samples_per_group"]),
    )
    phase_by_key = {
        sample["sample_key"]: "fit" for sample in fit_samples
    } | {
        sample["sample_key"]: "calibration" for sample in calibration_samples
    }
    fit_records, calibration_records = [], []
    for index, sample in enumerate(fit_samples + calibration_samples, 1):
        record = build_candidate_record(sample, prepared, config, device)
        if phase_by_key[sample["sample_key"]] == "fit":
            fit_records.append(record)
        else:
            calibration_records.append(record)
        if index % 64 == 0:
            print(
                f"GACR_P0_TRAIN_CANDIDATES dataset={dataset} "
                f"samples={index}/{len(fit_samples) + len(calibration_samples)}",
                flush=True,
            )
    torch.manual_seed(int(config["seed"]))
    ranker = BoundedResidualRanker(
        6,
        int(config["residual"]["hidden_dim"]),
        float(config["residual"]["bound"]),
    ).to(device)
    zero_identity = []
    with torch.no_grad():
        for record in fit_records + calibration_records:
            residual = ranker(record["features"])
            zero_identity.append(
                stable_ranking(record["base"])
                == stable_ranking(record["base"] + residual)
            )
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=float(config["residual"]["learning_rate"]),
        weight_decay=float(config["residual"]["weight_decay"]),
    )
    checkpoint_steps = set(config["residual"]["checkpoint_steps"])
    states, calibration = {}, {}
    for step in range(1, int(config["residual"]["training_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_training_loss(
            fit_records,
            ranker,
            float(config["residual"]["margin"]),
            device,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite P0 loss for {dataset}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise ValueError(f"non-finite P0 gradient for {dataset}")
        optimizer.step()
        if step in checkpoint_steps:
            state = {
                key: value.detach().cpu().clone()
                for key, value in ranker.state_dict().items()
            }
            states[step] = state
            groups, _ = evaluate_records(
                calibration_records,
                state,
                float(config["residual"]["bound"]),
                device,
            )
            calibration[step] = groups
            print(
                f"GACR_P0_TRAIN dataset={dataset} step={step} "
                f"loss={float(loss):.6f}",
                flush=True,
            )
    result = {
        "states": states,
        "calibration": calibration,
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
        "zero_residual_identity_rate": sum(zero_identity) / len(zero_identity),
        "parent_checkpoint_sha256_before": checkpoint_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
    }
    del prepared, fit_records, calibration_records, ranker
    torch.cuda.empty_cache()
    return result


def select_shared_step(training: dict, config: dict) -> tuple[int, list[dict]]:
    rows = []
    for step in config["residual"]["checkpoint_steps"]:
        dataset_checks = []
        gains = []
        for dataset in config["datasets"]:
            groups = training[dataset]["calibration"][step]
            overall = groups["overall"]
            tail = groups["tail"]
            ndcg_gain = relative_gain(
                overall["baseline_NDCG@10"], overall["residual_NDCG@10"]
            )
            recall_gain = (
                overall["residual_Recall@10"] - overall["baseline_Recall@10"]
            )
            tail_gain = relative_gain(
                tail["baseline_NDCG@10"], tail["residual_NDCG@10"]
            )
            dataset_checks.append(recall_gain >= 0 and tail_gain >= 0)
            gains.append(ndcg_gain)
        rows.append(
            {
                "step": step,
                "eligible": all(dataset_checks),
                "mean_relative_ndcg10_gain": float(np.mean(gains)),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible if eligible else rows
    selected = sorted(
        pool,
        key=lambda row: (-row["mean_relative_ndcg10_gain"], row["step"]),
    )[0]["step"]
    return int(selected), rows


def validation_phase(
    dataset: str,
    selected_state: dict,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1" / "model.pt"
    before_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    old_validation_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "validation_users.txt"
    )
    validation_users = select_fresh_validation_users(
        set(prepared["sequences"]),
        train_users | old_validation_users,
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
                f"GACR_P0_VALIDATION dataset={dataset} "
                f"users={index}/{len(samples)}",
                flush=True,
            )
    groups, rows = evaluate_records(
        records,
        selected_state,
        float(config["residual"]["bound"]),
        device,
    )
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = output_dir / "validation_per_user.csv"
    with per_user_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    residual_path = output_dir / "residual.pt"
    torch.save(selected_state, residual_path)
    result = {
        "users": len(rows),
        "validation_user_sha256": stable_sha(set(validation_users)),
        "prior_cohort_overlap": len(
            set(validation_users) & (train_users | old_validation_users)
        ),
        "groups": groups,
        "per_user_sha256": sha256(per_user_path),
        "residual_checkpoint_sha256": sha256(residual_path),
        "parent_checkpoint_sha256_before": before_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
        "parent_checkpoint_sha_unchanged": before_sha == sha256(checkpoint),
        "target_free_candidate_construction": True,
        "backbone_optimizer_steps": 0,
        "finite_rate": 1.0,
        "test_data_read": False,
        "bootstrap": {
            "overall_ndcg10_relative_gain_ci95": paired_bootstrap(
                rows, "NDCG@10", True, int(config["seed"]) + 11
            ),
            "overall_recall10_absolute_gain_ci95": paired_bootstrap(
                rows, "Recall@10", False, int(config["seed"]) + 21
            ),
            "tail_ndcg10_relative_gain_ci95": paired_bootstrap(
                [row for row in rows if row["target_group"] == "tail"],
                "NDCG@10",
                True,
                int(config["seed"]) + 31,
            ),
        },
    }
    del prepared, records
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR P0 requires CUDA")
    config = json.loads(args.config.read_text())
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")

    training = {
        dataset: generate_training_phase(dataset, config, p0_config, device)
        for dataset in config["datasets"]
    }
    selected_step, selection_rows = select_shared_step(training, config)
    validation = {
        dataset: validation_phase(
            dataset,
            training[dataset]["states"][selected_step],
            config,
            p0_config,
            args.output_root,
            device,
        )
        for dataset in config["datasets"]
    }
    gate_rows = []
    integrity_valid = all(
        training[dataset]["fit_calibration_user_overlap"] == 0
        and training[dataset]["zero_residual_identity_rate"] == 1.0
        and training[dataset]["parent_checkpoint_sha256_before"]
        == training[dataset]["parent_checkpoint_sha256_after"]
        for dataset in config["datasets"]
    )
    for dataset in config["datasets"]:
        result = validation[dataset]
        overall = result["groups"]["overall"]
        tail = result["groups"]["tail"]
        gains = {
            "overall_ndcg10_relative_gain": relative_gain(
                overall["baseline_NDCG@10"], overall["residual_NDCG@10"]
            ),
            "overall_recall10_absolute_gain": (
                overall["residual_Recall@10"] - overall["baseline_Recall@10"]
            ),
            "tail_ndcg10_relative_gain": relative_gain(
                tail["baseline_NDCG@10"], tail["residual_NDCG@10"]
            ),
            "broad_harm_rate": overall["broad_harm_rate"],
        }
        gates = config["validation_gates"]
        checks = {
            "overall_ndcg10": gains["overall_ndcg10_relative_gain"]
            >= gates["overall_ndcg10_relative_gain_min"],
            "overall_recall10": gains["overall_recall10_absolute_gain"]
            >= gates["overall_recall10_absolute_gain_min"],
            "tail_ndcg10": gains["tail_ndcg10_relative_gain"]
            >= gates["tail_ndcg10_relative_gain_min"],
            "broad_harm": gains["broad_harm_rate"]
            <= gates["broad_harm_rate_max"],
        }
        gate_rows.append(
            {
                "dataset": dataset,
                "gains": gains,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
        integrity_valid &= (
            result["prior_cohort_overlap"] == 0
            and result["parent_checkpoint_sha_unchanged"]
            and result["backbone_optimizer_steps"] == 0
            and result["finite_rate"] == 1.0
            and not result["test_data_read"]
        )
    effect_pass = all(row["pass"] for row in gate_rows)
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "GACR_P1_JOINT_TRAINING_DESIGN_ALLOWED"
        if effect_pass
        else "STOP_GACR_NO_RESIDUAL_RANK_EFFECT"
    )
    training_summary = {
        dataset: {
            key: value
            for key, value in result.items()
            if key not in ("states", "calibration")
        }
        | {"calibration": result["calibration"]}
        for dataset, result in training.items()
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "selected_shared_step": selected_step,
        "selection_rows": selection_rows,
        "training": training_summary,
        "validation": validation,
        "gate_rows": gate_rows,
        "integrity_valid": integrity_valid,
        "test_data_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
