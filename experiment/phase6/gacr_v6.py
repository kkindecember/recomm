#!/usr/bin/env python3
"""Phase 6 GACR-v6: full-fit scaling of the frozen-GRAM residual ranker."""

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
    relative_gain,
    split_training_users,
    to_cpu_record,
)
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
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)
from experiment.phase6.gacr_v2 import (  # noqa: E402
    build_validation_records,
    evaluate_scale,
    method_result,
    serializable_rows,
    validate_checkpoint_lineage,
)


def build_full_training_records(
    dataset: str,
    config: dict,
    p0_config: dict,
    device: torch.device,
) -> tuple[dict, list[dict], list[dict]]:
    """Build every eligible record from the locked fit-user split."""
    prepared = prepare(dataset, p0_config, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    fit_users, calibration_users = split_training_users(
        train_users, int(config["cohort_seed"]), dataset
    )
    fit_samples = build_train_samples(
        prepared["sequences"], fit_users, prepared["item2input"], prepared["item2lexid"]
    )
    calibration_pool = build_train_samples(
        prepared["sequences"], calibration_users, prepared["item2input"], prepared["item2lexid"]
    )
    calibration_samples = select_stratified_samples(
        calibration_pool,
        prepared["heads"],
        int(config["cohort_seed"]),
        f"{dataset}|gacr-p0-calibration",
        int(config["calibration_samples_per_group"]),
        int(config["calibration_samples_per_group"]),
    )
    fit_records: list[dict] = []
    calibration_records: list[dict] = []
    total = len(fit_samples) + len(calibration_samples)
    for index, sample in enumerate(fit_samples + calibration_samples, 1):
        record = to_cpu_record(build_candidate_record(sample, prepared, config, device))
        if index <= len(fit_samples):
            fit_records.append(record)
        else:
            calibration_records.append(record)
        if index % 64 == 0 or index == total:
            print(
                f"GACR_V6_FULL_CANDIDATES dataset={dataset} samples={index}/{total}",
                flush=True,
            )
    metadata = {
        "train_users": len(train_users),
        "fit_users": len(fit_users),
        "calibration_users": len(calibration_users),
        "fit_samples": len(fit_records),
        "fit_covered": sum(row["target_index"] is not None for row in fit_records),
        "calibration_samples": len(calibration_records),
        "calibration_covered": sum(
            row["target_index"] is not None for row in calibration_records
        ),
        "fit_sample_sha256": stable_sha({row["sample_key"] for row in fit_samples}),
        "fit_user_sha256": stable_sha(fit_users),
        "calibration_user_sha256": stable_sha(calibration_users),
        "fit_calibration_user_overlap": len(fit_users & calibration_users),
        "parent_checkpoint_sha256_before": checkpoint_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
    }
    del prepared
    torch.cuda.empty_cache()
    return metadata, fit_records, calibration_records


def train_full_batch_seed(
    fit_records: list[dict],
    calibration_records: list[dict],
    config: dict,
    seed: int,
    device: torch.device,
    chunk_size: int = 128,
) -> dict:
    """Exact full-batch gradient, backpropagated in normalized chunks."""
    residual = config["residual"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ranker = BoundedResidualRanker(
        6, int(residual["hidden_dim"]), float(residual["bound"])
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
    grouped = {
        group: [
            row
            for row in fit_records
            if row["target_index"] is not None and row["target_group"] == group
        ]
        for group in ("head", "tail")
    }
    if any(not rows for rows in grouped.values()):
        raise ValueError("full-fit training has an empty covered head/tail group")
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=float(residual["learning_rate"]),
        weight_decay=float(residual["weight_decay"]),
    )
    first_loss = last_loss = last_gradient_norm = None
    steps = int(residual["fixed_training_step"])
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        scalar_loss = 0.0
        for group in ("head", "tail"):
            rows = grouped[group]
            denominator = 2.0 * len(rows)
            for start in range(0, len(rows), chunk_size):
                losses = []
                for record in rows[start : start + chunk_size]:
                    losses.append(
                        hinge_loss(
                            record["base"].to(device),
                            ranker(record["features"].to(device)),
                            int(record["target_index"]),
                            float(residual["margin"]),
                        )
                    )
                chunk_loss = torch.stack(losses).sum() / denominator
                if not torch.isfinite(chunk_loss):
                    raise ValueError(f"non-finite GACR-v6 loss seed={seed}")
                chunk_loss.backward()
                scalar_loss += float(chunk_loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise ValueError(f"non-finite GACR-v6 gradient seed={seed}")
        optimizer.step()
        first_loss = scalar_loss if first_loss is None else first_loss
        last_loss = scalar_loss
        last_gradient_norm = float(norm)
        print(
            f"GACR_V6_TRAIN seed={seed} step={step}/{steps} loss={scalar_loss:.6f}",
            flush=True,
        )
    state = {key: value.detach().cpu().clone() for key, value in ranker.state_dict().items()}
    calibration, _ = evaluate_scale(
        calibration_records,
        state,
        float(residual["bound"]),
        float(config["deployment_scale"]),
        device,
    )
    return {
        "state": state,
        "optimizer_steps": steps,
        "covered_head_records": len(grouped["head"]),
        "covered_tail_records": len(grouped["tail"]),
        "zero_residual_identity_rate": zero_identity,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "last_gradient_norm": last_gradient_norm,
        "calibration": calibration,
    }


def rank_metric(rank: int | None, cutoff: int, ndcg: bool) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1) if ndcg else 1.0


def add_standard_metrics(result: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    enriched = []
    for source in rows:
        row = dict(source)
        for cutoff in (5, 10):
            for metric, ndcg in (("Recall", False), ("NDCG", True)):
                row[f"baseline_{metric}@{cutoff}"] = rank_metric(
                    row["baseline_rank"], cutoff, ndcg
                )
                row[f"candidate_{metric}@{cutoff}"] = rank_metric(
                    row["candidate_rank"], cutoff, ndcg
                )
        enriched.append(row)
    standard = {}
    for group in ("overall", "head", "tail"):
        selected = enriched if group == "overall" else [
            row for row in enriched if row["target_group"] == group
        ]
        standard[group] = {}
        for cutoff in (5, 10):
            for metric in ("Recall", "NDCG"):
                key = f"{metric}@{cutoff}"
                baseline = float(np.mean([row[f"baseline_{key}"] for row in selected]))
                candidate = float(np.mean([row[f"candidate_{key}"] for row in selected]))
                standard[group][key] = {
                    "baseline": baseline,
                    "candidate": candidate,
                    "absolute_delta": candidate - baseline,
                    "relative_gain": relative_gain(baseline, candidate),
                }
    result = dict(result)
    result["standard_metrics"] = standard
    return result, enriched


def compare_methods(v3_rows: list[dict], v6_rows: list[dict]) -> dict:
    if [row["sample_key"] for row in v3_rows] != [row["sample_key"] for row in v6_rows]:
        raise ValueError("v3/v6 per-user rows are not aligned")
    output = {}
    for cutoff in (5, 10):
        for metric in ("Recall", "NDCG"):
            key = f"candidate_{metric}@{cutoff}"
            v3 = float(np.mean([row[key] for row in v3_rows]))
            v6 = float(np.mean([row[key] for row in v6_rows]))
            output[f"{metric}@{cutoff}"] = {
                "v3": v3,
                "v6": v6,
                "absolute_delta": v6 - v3,
                "relative_gain": relative_gain(v3, v6),
            }
    return output


def load_v3_state(config: dict, dataset: str, seed: int) -> dict:
    path = ROOT / config["inputs"]["v3_residual_root"] / dataset / f"residual_seed{seed}.pt"
    expected = config["inputs"]["expected_v3_residual_sha256"][dataset][str(seed)]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"v3 residual SHA mismatch {dataset}/{seed}: {actual}")
    return torch.load(path, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v6 requires CUDA")
    config = json.loads(args.config.read_text())
    validate_checkpoint_lineage(config)
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")
    training = {}
    for dataset in config["datasets"]:
        metadata, fit_records, calibration_records = build_full_training_records(
            dataset, config, p0_config, device
        )
        seeds = {}
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for seed in config["training_seeds"]:
            trained = train_full_batch_seed(
                fit_records, calibration_records, config, int(seed), device
            )
            checkpoint = output_dir / f"residual_seed{seed}.pt"
            torch.save(trained["state"], checkpoint)
            trained["residual_checkpoint_sha256"] = sha256(checkpoint)
            seeds[str(seed)] = trained
        training[dataset] = metadata | {"seeds": seeds}
        del fit_records, calibration_records
        torch.cuda.empty_cache()

    validation = {}
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(dataset, config, p0_config, device)
        output_dir = args.output_root / dataset
        seeds = {}
        for seed in config["training_seeds"]:
            v3_state = load_v3_state(config, dataset, int(seed))
            v6_state = training[dataset]["seeds"][str(seed)]["state"]
            v3_result, v3_rows = method_result(
                records, v3_state, config, 1.0, int(seed), device
            )
            v6_result, v6_rows = method_result(
                records, v6_state, config, 1.0, int(seed) + 1000, device
            )
            v3_result, v3_rows = add_standard_metrics(v3_result, v3_rows)
            v6_result, v6_rows = add_standard_metrics(v6_result, v6_rows)
            seeds[str(seed)] = {
                "gacr_v3": v3_result,
                "gacr_v6": v6_result,
                "incremental_v6_vs_v3": compare_methods(v3_rows, v6_rows),
            }
            for method, rows in (("gacr_v3", v3_rows), ("gacr_v6", v6_rows)):
                path = output_dir / f"{method}_seed{seed}_per_user.csv"
                serial = serializable_rows(rows)
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
                    writer.writeheader()
                    writer.writerows(serial)
                seeds[str(seed)][method]["per_user_sha256"] = sha256(path)
        validation[dataset] = metadata | {"seeds": seeds}
        del records
        torch.cuda.empty_cache()

    integrity = {
        "all_fit_records_used": True,
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
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    training_summary = {}
    for dataset, result in training.items():
        training_summary[dataset] = {key: value for key, value in result.items() if key != "seeds"}
        training_summary[dataset]["seeds"] = {
            seed: {key: value for key, value in value.items() if key != "state"}
            for seed, value in result["seeds"].items()
        }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "single_changed_factor": config["single_changed_factor"],
        "training": training_summary,
        "validation": validation,
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
