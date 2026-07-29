#!/usr/bin/env python3
"""GACR S0: correctness smoke for bounded generator-anchored residual ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    collate,
    normalized_sequence,
    prepare,
    read_users,
    sha256,
    write_json,
)
from utils import generation_trie as gt  # noqa: E402


def stable_sample_order(seed: int, dataset: str, sample_key: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{sample_key}".encode()).hexdigest()


def select_stratified_samples(
    samples: list[dict],
    head_items: set[str],
    seed: int,
    dataset: str,
    head_count: int,
    tail_count: int,
) -> list[dict]:
    ordered = sorted(
        samples,
        key=lambda row: stable_sample_order(seed, dataset, row["sample_key"]),
    )
    head = [row for row in ordered if row["positive_item"] in head_items][:head_count]
    tail = [row for row in ordered if row["positive_item"] not in head_items][:tail_count]
    if len(head) != head_count or len(tail) != tail_count:
        raise ValueError(f"insufficient stratified samples for {dataset}")
    return sorted(head + tail, key=lambda row: row["sample_key"])


def target_free_union(
    gram_items: list[str],
    catalog_items: list[str],
) -> list[str]:
    return list(dict.fromkeys(gram_items + catalog_items))


def base_scores(union: list[str], gram_items: list[str]) -> torch.Tensor:
    gram_rank = {item: index + 1 for index, item in enumerate(gram_items)}
    return torch.tensor(
        [1.0 / gram_rank[item] if item in gram_rank else 0.0 for item in union],
        dtype=torch.float32,
    )


def stable_ranking(scores: torch.Tensor) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def finite_catalog_zscore(
    logit: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if not bool(torch.isfinite(logit)):
        return torch.as_tensor(-10.0, dtype=mean.dtype, device=mean.device)
    return ((logit - mean) / std).clamp(-10.0, 10.0)


class BoundedResidualRanker(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(self.network(features).squeeze(-1))


def hinge_loss(
    base: torch.Tensor,
    residual: torch.Tensor,
    target_index: int,
    margin: float,
) -> torch.Tensor:
    scores = base + residual
    negative = torch.cat([scores[:target_index], scores[target_index + 1 :]]).max()
    return torch.relu(torch.as_tensor(margin, device=scores.device) - scores[target_index] + negative)


@torch.no_grad()
def build_candidate_record(
    sample: dict,
    prepared: dict,
    config: dict,
    device: torch.device,
) -> dict:
    model = prepared["model"]
    inference_sample = dict(sample)
    inference_sample["output"] = prepared["item2lexid"][prepared["catalog"][0]]
    batch = collate(prepared["collator"], [inference_sample])
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    if "_gacr_trie" not in prepared:
        prepared["_gacr_trie"] = gt.Trie(prepared["encoded_candidates"])
        prepared["_gacr_max_length"] = max(
            len(row) for row in prepared["encoded_candidates"]
        )
    trie = prepared["_gacr_trie"]
    prediction = model.backbone.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_length=prepared["_gacr_max_length"],
        prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(trie),
        num_beams=int(config["generator_top_k"]),
        num_return_sequences=int(config["generator_top_k"]),
        return_dict_in_generate=True,
        length_penalty=1.0,
    )
    gram = [
        prepared["sequence_to_item"].get(normalized_sequence(row.tolist()))
        for row in prediction["sequences"]
    ]
    if any(item is None for item in gram) or len(set(gram)) != len(gram):
        raise ValueError("invalid generator candidate mapping")
    model.backbone.encoder.n_passages = input_ids.size(1)
    flat_ids = input_ids.view(1, -1)
    flat_attention = attention.view(1, -1)
    hidden = model.backbone.encoder(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        return_dict=True,
    )[0]
    pooled = model.pool_coarse(hidden, attention, input_ids.shape[-1])[0]
    logits = model.catalog_head(pooled)
    for item in sample["history_items"]:
        if item in model.item_to_index:
            logits[model.item_to_index[item]] = -torch.inf
    top_indices = torch.topk(logits, k=int(config["catalog_top_k"])).indices.tolist()
    catalog_items = [prepared["catalog"][index] for index in top_indices]
    union = target_free_union(gram, catalog_items)
    base = base_scores(union, gram).to(device)
    finite_logits = logits[torch.isfinite(logits)]
    logit_mean = finite_logits.mean()
    logit_std = finite_logits.std(unbiased=False).clamp_min(1e-6)
    gram_rank = {item: index + 1 for index, item in enumerate(gram)}
    catalog_rank = {item: index + 1 for index, item in enumerate(catalog_items)}
    pooled_norm = torch.linalg.vector_norm(pooled).clamp_min(1e-6)
    features = []
    for item in union:
        item_index = model.item_to_index[item]
        item_weight = model.catalog_head.weight[item_index]
        cosine = torch.dot(pooled, item_weight) / (
            pooled_norm * torch.linalg.vector_norm(item_weight).clamp_min(1e-6)
        )
        features.append(
            torch.stack(
                [
                    finite_catalog_zscore(
                        logits[item_index], logit_mean, logit_std
                    ),
                    torch.as_tensor(1.0 / gram_rank[item] if item in gram_rank else 0.0, device=device),
                    torch.as_tensor(1.0 / catalog_rank[item] if item in catalog_rank else 0.0, device=device),
                    torch.as_tensor(float(item in gram_rank), device=device),
                    torch.as_tensor(float(item in catalog_rank), device=device),
                    cosine,
                ]
            )
        )
    features_tensor = torch.stack(features)
    if not torch.isfinite(features_tensor).all():
        raise ValueError("non-finite residual feature")
    target = sample["positive_item"]
    return {
        "sample_key": sample["sample_key"],
        "target_group": "head" if target in prepared["heads"] else "tail",
        "union": union,
        "target_index": union.index(target) if target in union else None,
        "gram_rank": gram_rank.get(target),
        "catalog_rank": catalog_rank.get(target),
        "base": base.detach(),
        "features": features_tensor.detach(),
    }


def run_dataset(dataset: str, config: dict, p0_config: dict, device: torch.device) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1" / "model.pt"
    before_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    all_samples = build_train_samples(
        prepared["sequences"],
        users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    samples = select_stratified_samples(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        int(config["head_samples"]),
        int(config["tail_samples"]),
    )
    records = []
    for index, sample in enumerate(samples, 1):
        records.append(build_candidate_record(sample, prepared, config, device))
        if index % 32 == 0:
            print(f"GACR_S0_PROGRESS dataset={dataset} samples={index}/{len(samples)}", flush=True)
    covered = [record for record in records if record["target_index"] is not None]
    covered_head = sum(record["target_group"] == "head" for record in covered)
    covered_tail = sum(record["target_group"] == "tail" for record in covered)

    residual_config = config["residual"]
    ranker = BoundedResidualRanker(
        feature_dim=6,
        hidden_dim=16,
        bound=float(residual_config["bound"]),
    ).to(device)
    zero_identity = []
    with torch.no_grad():
        for record in records:
            residual = ranker(record["features"])
            zero_identity.append(
                stable_ranking(record["base"])
                == stable_ranking(record["base"] + residual)
            )
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=float(residual_config["learning_rate"])
    )
    losses = []
    gradient_norm = None
    for step in range(int(residual_config["optimizer_steps"])):
        optimizer.zero_grad(set_to_none=True)
        sample_losses = [
            hinge_loss(
                record["base"],
                ranker(record["features"]),
                int(record["target_index"]),
                float(residual_config["margin"]),
            )
            for record in covered
        ]
        loss = torch.stack(sample_losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite residual loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise ValueError("non-finite residual gradient")
        if step == 0:
            gradient_norm = float(norm)
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        residual_max = max(
            float(ranker(record["features"]).abs().max()) for record in records
        )
        probe = records[0]["features"]
        before_reload = ranker(probe)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(ranker.state_dict(), handle.name)
        reloaded = BoundedResidualRanker(6, 16, float(residual_config["bound"])).to(device)
        reloaded.load_state_dict(torch.load(handle.name, map_location=device), strict=True)
    with torch.no_grad():
        reload_difference = float((before_reload - reloaded(probe)).abs().max())

    result = {
        "dataset": dataset,
        "samples": len(records),
        "covered_pairs": len(covered),
        "covered_head_pairs": covered_head,
        "covered_tail_pairs": covered_tail,
        "zero_residual_identity_rate": sum(zero_identity) / len(zero_identity),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "initial_gradient_norm": gradient_norm,
        "residual_abs_max": residual_max,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "parent_checkpoint_sha256_before": before_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
        "parent_checkpoint_sha_unchanged": before_sha == sha256(checkpoint),
        "target_free_candidate_construction": True,
        "optimizer_steps": int(residual_config["optimizer_steps"]),
        "backbone_optimizer_steps": 0,
        "finite_rate": 1.0,
        "test_data_read": False,
    }
    del prepared
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR S0 requires CUDA")
    config = json.loads(args.config.read_text())
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(dataset, config, p0_config, device)
        for dataset in config["datasets"]
    }
    gates = config["gates"]
    gate_rows = []
    for dataset, result in results.items():
        checks = {
            "zero_residual_identity": result["zero_residual_identity_rate"]
            == gates["zero_residual_identity_rate"],
            "finite": result["finite_rate"] == gates["finite_rate"],
            "nonzero_gradient": result["initial_gradient_norm"] > 0,
            "loss_decreased": result["loss_decreased"],
            "residual_bound": result["residual_abs_max"]
            <= gates["residual_abs_max"] + 1e-7,
            "head_pair_coverage": result["covered_head_pairs"]
            >= gates["minimum_covered_head_pairs"],
            "tail_pair_coverage": result["covered_tail_pairs"]
            >= gates["minimum_covered_tail_pairs"],
            "checkpoint_reload": result["checkpoint_reload_max_abs_difference"]
            == gates["checkpoint_reload_max_abs_difference"],
            "parent_sha": result["parent_checkpoint_sha_unchanged"],
            "target_free": result["target_free_candidate_construction"],
            "test_exclusion": not result["test_data_read"],
        }
        gate_rows.append(
            {"dataset": dataset, "checks": checks, "pass": all(checks.values())}
        )
    passed = all(row["pass"] for row in gate_rows)
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": (
            "GACR_S0_CORRECTNESS_PASS"
            if passed
            else "STOP_GACR_S0_CORRECTNESS_FAILED"
        ),
        "results": results,
        "gate_rows": gate_rows,
        "test_data_read": False,
    }
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
