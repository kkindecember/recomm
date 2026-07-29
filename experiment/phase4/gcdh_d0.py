#!/usr/bin/env python3
"""GCDH D0: no-update diagnosis of pooled user states and catalog readout."""

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

from experiment.phase4.gcdh_p0 import (
    ROOT,
    build_validation_samples,
    collate,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)


def exact_rank(logits: torch.Tensor, target_index: int) -> int:
    """One-based descending rank with catalog-index tie breaking."""
    target = logits[target_index]
    indices = torch.arange(logits.numel(), device=logits.device)
    return int(
        1
        + (logits > target).sum().item()
        + ((logits == target) & (indices < target_index)).sum().item()
    )


def normalized_entropy(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits.double(), dim=0)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-300))).sum()
    return float(entropy / math.log(logits.numel()))


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(x, y) / denominator)


def state_statistics(states: np.ndarray, sample_users: int) -> dict:
    selected = torch.from_numpy(states[:sample_users]).double()
    feature_std = selected.std(dim=0, unbiased=False)
    centered = selected - selected.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square()
    effective_rank = float(
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-30)
    )
    normalized = torch.nn.functional.normalize(selected, dim=1)
    cosine = normalized @ normalized.T
    upper = torch.triu_indices(len(selected), len(selected), offset=1)
    cosine_distance = 1.0 - cosine[upper[0], upper[1]]
    norms = torch.linalg.vector_norm(selected, dim=1)
    return {
        "sample_users": len(selected),
        "pooled_rms_feature_std": float(torch.sqrt(feature_std.square().mean())),
        "median_pairwise_cosine_distance": float(cosine_distance.median()),
        "effective_rank": effective_rank,
        "mean_state_norm": float(norms.mean()),
        "state_norm_cv": float(norms.std(unbiased=False) / norms.mean().clamp_min(1e-30)),
    }


@torch.no_grad()
def diagnose_control(
    dataset: str,
    control: str,
    config: dict,
    p0_config: dict,
    device: torch.device,
) -> tuple[dict, dict[str, dict]]:
    prepared = prepare(dataset, p0_config, device)
    checkpoint_dir = ROOT / config["inputs"]["checkpoint_root"] / dataset / control
    checkpoint = checkpoint_dir / "model.pt"
    training_summary = json.loads(
        (checkpoint_dir / "training_summary.json").read_text()
    )
    before_sha = sha256(checkpoint)
    if before_sha != training_summary["checkpoint_sha256"]:
        raise ValueError(f"checkpoint SHA mismatch for {dataset}/{control}")
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    model = prepared["model"]
    model.eval()
    users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "validation_users.txt"
    )
    samples = build_validation_samples(
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    if len(samples) != int(config["validation_users"]):
        raise ValueError(f"unexpected validation size for {dataset}: {len(samples)}")

    all_states = []
    rows: dict[str, dict] = {}
    repeat_difference = None
    batch_size = int(config["batch_size"])
    popularity = torch.tensor(
        [
            math.log1p(prepared["popularity"].get(item, 0))
            for item in prepared["catalog"]
        ],
        dtype=torch.float64,
        device=device,
    )
    for start in range(0, len(samples), batch_size):
        selected = samples[start : start + batch_size]
        batch = collate(prepared["collator"], selected)
        input_ids = batch["item_text_ids"].to(device)
        attention = batch["item_text_masks"].to(device)
        model.backbone.encoder.n_passages = input_ids.size(1)
        flat_ids = input_ids.view(input_ids.size(0), -1)
        flat_attention = attention.view(attention.size(0), -1)
        hidden = model.backbone.encoder(
            input_ids=flat_ids,
            attention_mask=flat_attention,
            return_dict=True,
        )[0]
        pooled = model.pool_coarse(hidden, attention, input_ids.shape[-1])
        logits_batch = model.catalog_head(pooled)
        if start == 0:
            hidden_repeat = model.backbone.encoder(
                input_ids=flat_ids,
                attention_mask=flat_attention,
                return_dict=True,
            )[0]
            pooled_repeat = model.pool_coarse(
                hidden_repeat, attention, input_ids.shape[-1]
            )
            repeat_difference = float((pooled - pooled_repeat).abs().max().cpu())
        if not torch.isfinite(pooled).all() or not torch.isfinite(logits_batch).all():
            raise ValueError(f"non-finite D0 output for {dataset}/{control}")
        all_states.append(pooled.float().cpu().numpy())
        for offset, sample in enumerate(selected):
            logits = logits_batch[offset].clone()
            for item in sample["history_items"]:
                item_index = model.item_to_index.get(item)
                if item_index is not None:
                    logits[item_index] = -torch.inf
            finite = torch.isfinite(logits)
            target_index = model.item_to_index[sample["positive_item"]]
            target_rank = exact_rank(logits, target_index)
            top_indices = torch.topk(
                logits, k=int(config["catalog_top_k"])
            ).indices.tolist()
            rows[sample["user_id"]] = {
                "user_id": sample["user_id"],
                "target_item": sample["positive_item"],
                "target_group": (
                    "head"
                    if sample["positive_item"] in prepared["heads"]
                    else "tail"
                ),
                "target_index": target_index,
                "target_rank": target_rank,
                "recall10": float(target_rank <= 10),
                "recall50": float(target_rank <= 50),
                "reciprocal_rank": 1.0 / target_rank,
                "normalized_entropy": normalized_entropy(logits[finite]),
                "popularity_logit_pearson": pearson(
                    logits[finite], popularity[finite]
                ),
                "top50_indices": top_indices,
            }
        print(
            f"D0_PROGRESS dataset={dataset} control={control} "
            f"users={min(start + batch_size, len(samples))}/{len(samples)}",
            flush=True,
        )

    states = np.concatenate(all_states, axis=0)
    ranks = np.asarray([rows[user]["target_rank"] for user in sorted(rows)])
    reciprocal = np.asarray(
        [rows[user]["reciprocal_rank"] for user in sorted(rows)]
    )
    result = {
        "dataset": dataset,
        "control": control,
        "users": len(rows),
        "validation_user_sha256": stable_sha(users),
        "checkpoint_sha256_before": before_sha,
        "checkpoint_sha256_after": sha256(checkpoint),
        "checkpoint_sha_unchanged": before_sha == sha256(checkpoint),
        "optimizer_steps": 0,
        "finite_rate": 1.0,
        "target_mapping_rate": 1.0,
        "repeat_max_abs_difference": repeat_difference,
        "catalog": {
            "mean_target_rank": float(ranks.mean()),
            "median_target_rank": float(np.median(ranks)),
            "mrr": float(reciprocal.mean()),
            "recall10": float(np.mean(ranks <= 10)),
            "recall50": float(np.mean(ranks <= 50)),
            "mean_normalized_entropy": float(
                np.mean([row["normalized_entropy"] for row in rows.values()])
            ),
            "mean_popularity_logit_pearson": float(
                np.mean(
                    [row["popularity_logit_pearson"] for row in rows.values()]
                )
            ),
        },
        "state": state_statistics(states, int(config["state_sample_users"])),
        "test_data_read": False,
    }
    del prepared, model, states
    torch.cuda.empty_cache()
    return result, rows


def source_attribution(path: Path) -> dict:
    counts = {"generator_only": 0, "catalog_only": 0, "both": 0, "neither": 0}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        gram = float(row["gram_Recall@50"]) == 1.0
        catalog = float(row["catalog_Recall@50"]) == 1.0
        key = (
            "both"
            if gram and catalog
            else "generator_only"
            if gram
            else "catalog_only"
            if catalog
            else "neither"
        )
        counts[key] += 1
    return {
        "users": len(rows),
        "counts": counts,
        "rates": {key: value / len(rows) for key, value in counts.items()},
        "union_oracle_recall50": 1.0 - counts["neither"] / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GCDH D0 requires CUDA")
    config = json.loads(args.config.read_text())
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {}
    row_cache = {}
    integrity_valid = True
    representation_healthy = True
    incremental_signal = True
    for dataset in config["datasets"]:
        results[dataset] = {}
        row_cache[dataset] = {}
        for control in config["controls"]:
            result, rows = diagnose_control(
                dataset, control, config, p0_config, device
            )
            results[dataset][control] = result
            row_cache[dataset][control] = rows
            state = result["state"]
            gates = config["representation_noncollapse_gates"]
            representation_healthy &= (
                state["pooled_rms_feature_std"]
                >= gates["pooled_rms_feature_std_min"]
                and state["median_pairwise_cosine_distance"]
                >= gates["median_pairwise_cosine_distance_min"]
                and state["effective_rank"] >= gates["effective_rank_min"]
            )
            integrity_valid &= (
                result["finite_rate"] == 1.0
                and result["target_mapping_rate"] == 1.0
                and result["repeat_max_abs_difference"] == 0.0
                and result["checkpoint_sha_unchanged"]
                and result["optimizer_steps"] == 0
                and not result["test_data_read"]
            )
        c0 = results[dataset]["C0"]["catalog"]
        c1 = results[dataset]["C1"]["catalog"]
        incremental_signal &= c1["mrr"] > c0["mrr"] and c1["recall50"] >= c0["recall50"]
        users = sorted(row_cache[dataset]["C0"])
        overlap = [
            len(
                set(row_cache[dataset]["C0"][user]["top50_indices"])
                & set(row_cache[dataset]["C1"][user]["top50_indices"])
            )
            / int(config["catalog_top_k"])
            for user in users
        ]
        results[dataset]["c0_c1_catalog_top50_overlap_mean"] = float(np.mean(overlap))
        results[dataset]["source_attribution_c1"] = source_attribution(
            ROOT
            / config["inputs"]["checkpoint_root"]
            / dataset
            / "C1"
            / "validation_per_user.csv"
        )

    if not integrity_valid:
        decision = "EXECUTION_INVALID"
    elif not representation_healthy:
        decision = "GCDH_D0_USER_STATE_COLLAPSE"
    elif incremental_signal:
        decision = "GCDH_D0_READOUT_RANKING_MISMATCH"
    else:
        decision = "GCDH_D0_NO_CROSS_DOMAIN_INCREMENTAL_CATALOG_SIGNAL"
    output = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "integrity_valid": integrity_valid,
        "representation_healthy": representation_healthy,
        "cross_domain_incremental_catalog_signal": incremental_signal,
        "results": results,
        "test_data_read": False,
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
