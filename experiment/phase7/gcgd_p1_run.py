#!/usr/bin/env python3
"""Frozen single-seed development pilot for Graph-Conditioned GRAM Decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gacr_p0 import relative_gain, split_training_users  # noqa: E402
from experiment.phase4.gacr_s0 import select_stratified_samples  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    build_validation_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from experiment.phase6.gacr_v2 import paired_bootstrap_candidate  # noqa: E402
from experiment.phase7.gcgd_p0 import read_train_sequences  # noqa: E402
from experiment.phase7.gcgd_p1 import (  # noqa: E402
    AdaptiveGraphPrefixLogitsProcessor,
    adapter_training_loss,
    arm_metric_row,
    build_indexed_graph,
    generate_arm_items,
    graph_logits_for_user,
    graph_prefix_inputs,
    matched_gacr_v3_rank,
    metrics_for_rank,
    select_development_users,
    train_lightgcn,
)
from experiment.phase7.gcgd_v1 import (  # noqa: E402
    GraphReliabilityAdapter,
    aggregate_graph_prefix_logits,
    reliability_features,
)


def stable_order(seed: int, dataset: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|gcgd-p1|{key}".encode()).hexdigest()


def prefix_feature(
    gram_logits: torch.Tensor,
    graph_log_probabilities: torch.Tensor,
    *,
    compatible_leaf_fraction: float,
    depth: int,
    maximum_depth: int,
) -> tuple[float, ...]:
    probability = graph_log_probabilities.exp()
    if probability.numel() == 1:
        entropy, margin = 0.0, 1.0
    else:
        entropy = float(
            -(probability * graph_log_probabilities).sum()
            / math.log(probability.numel())
        )
        ordered = probability.sort(descending=True).values
        margin = float(ordered[0] - ordered[1])
    return reliability_features(
        graph_coverage=1.0,
        normalized_entropy=max(0.0, min(1.0, entropy)),
        top_margin=max(0.0, min(1.0, margin)),
        compatible_leaf_fraction=compatible_leaf_fraction,
        gram_graph_agreement=float(int(gram_logits.argmax()) == int(graph_log_probabilities.argmax())),
        normalized_depth=min(1.0, max(0.0, (depth - 1) / max(1, maximum_depth))),
    )


@torch.no_grad()
def build_adapter_record(
    sample: dict,
    prepared: dict,
    item_paths: Mapping[str, Sequence[int]],
    item_logits: Mapping[str, float],
    leaf_fractions: Mapping[tuple[int, ...], float],
    device: torch.device,
    maximum_depth: int,
    seed: int,
    dataset: str,
) -> dict:
    target_path = tuple(item_paths[sample["positive_item"]])
    depths = list(range(1, len(target_path)))
    selected = int(stable_order(seed, dataset, sample["sample_key"]), 16) % len(depths)
    depth = depths[selected]
    prefix = target_path[:depth]
    target_token = target_path[depth]
    graph = aggregate_graph_prefix_logits(item_paths, item_logits, prefix)
    if target_token not in graph:
        raise ValueError("adapter target token lacks positive graph mass")
    legal_tokens = sorted(graph)
    batch = collate(prepared["collator"], [sample])
    output = prepared["model"].backbone(
        input_ids=batch["item_text_ids"].to(device),
        attention_mask=batch["item_text_masks"].to(device),
        decoder_input_ids=torch.tensor([prefix], dtype=torch.long, device=device),
        return_dict=True,
    )
    gram_legal = output.logits[0, -1, legal_tokens].float()
    graph_legal = torch.tensor(
        [graph[token] for token in legal_tokens], dtype=torch.float32, device=device
    )
    target_position = legal_tokens.index(target_token)
    feature = prefix_feature(
        gram_legal,
        graph_legal,
        compatible_leaf_fraction=float(leaf_fractions[prefix]),
        depth=depth,
        maximum_depth=maximum_depth,
    )
    return {
        "feature": torch.tensor(feature, dtype=torch.float32),
        "gram": gram_legal.cpu(),
        "graph": graph_legal.cpu(),
        "target": target_position,
        "reliability": float(int(graph_legal.argmax()) == target_position),
    }


def train_adapter(
    records: list[dict], config: dict, device: torch.device
) -> tuple[GraphReliabilityAdapter, list[dict]]:
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    adapter = GraphReliabilityAdapter().to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(config["adapter"]["learning_rate"])
    )
    batch_size = int(config["adapter"]["optimizer_batch_prefixes"])
    steps = int(config["adapter"]["fixed_steps"])
    weights = config["adapter"]["loss_weights"]
    history = []
    for step in range(steps):
        selected = [records[(step * batch_size + index) % len(records)] for index in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        next_token_losses = []
        reliability_losses = []
        for record in selected:
            loss, parts = adapter_training_loss(
                adapter,
                record["feature"].to(device).unsqueeze(0),
                record["gram"].to(device).unsqueeze(0),
                record["graph"].to(device).unsqueeze(0),
                torch.tensor([record["target"]], device=device),
                torch.tensor([record["reliability"]], device=device),
                alpha=float(config["decoding"]["C_alpha"]),
                next_token_weight=float(weights["next_token_ce"]),
                reliability_weight=float(weights["gate_reliability_bce"]),
            )
            losses.append(loss)
            next_token_losses.append(parts["next_token_ce"])
            reliability_losses.append(parts["gate_reliability_bce"])
        total = torch.stack(losses).mean()
        if not torch.isfinite(total):
            raise ValueError(f"non-finite adapter loss at step {step + 1}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 10.0)
        if not torch.isfinite(gradient_norm):
            raise ValueError(f"non-finite adapter gradient at step {step + 1}")
        optimizer.step()
        history.append({
            "step": step + 1,
            "loss": float(total.detach()),
            "next_token_ce": float(torch.stack(next_token_losses).mean().detach()),
            "gate_reliability_bce": float(torch.stack(reliability_losses).mean().detach()),
            "gradient_norm": float(gradient_norm.detach()),
        })
    adapter.eval()
    return adapter, history


@torch.no_grad()
def adapter_diagnostics(adapter: GraphReliabilityAdapter, records: list[dict], device: torch.device) -> dict:
    features = torch.stack([record["feature"] for record in records]).to(device)
    labels = torch.tensor([record["reliability"] for record in records], device=device)
    gates, temperature = adapter(features)
    return {
        "records": len(records),
        "mean_gate": float(gates.mean()),
        "gate_accuracy_at_0_5": float(((gates >= 0.5) == (labels >= 0.5)).float().mean()),
        "temperature": float(temperature),
    }


def summarize_rows(rows: list[dict]) -> dict:
    metric_keys = ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "Recall@50", "MRR")

    def group(selected: list[dict]) -> dict:
        if not selected:
            return {"n": 0, "available": False}
        result: dict[str, object] = {"n": len(selected), "available": True}
        for key in metric_keys:
            baseline = float(np.mean([row[f"baseline_{key}"] for row in selected]))
            candidate = float(np.mean([row[f"candidate_{key}"] for row in selected]))
            result[f"baseline_{key}"] = baseline
            result[f"candidate_{key}"] = candidate
            result[f"absolute_delta_{key}"] = candidate - baseline
            result[f"relative_gain_{key}"] = relative_gain(baseline, candidate)
        for key in (
            "target_in_baseline_beam50",
            "target_in_candidate_beam50",
            "new_hit_at10_outside_A_beam",
            "changed",
            "broad_harm",
        ):
            result[f"mean_{key}"] = float(np.mean([row[key] for row in selected]))
        return result

    result = {"overall": group(rows)}
    result["head"] = group([row for row in rows if row["target_group"] == "head"])
    result["tail"] = group([row for row in rows if row["target_group"] == "tail"])
    result["graph_covered"] = group([row for row in rows if row["graph_covered"]])
    result["graph_uncovered"] = group([row for row in rows if not row["graph_covered"]])
    return result


def bootstrap_rows(rows: list[dict], seed: int) -> dict:
    tail = [row for row in rows if row["target_group"] == "tail"]
    return {
        "overall_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(rows, "NDCG@10", True, seed + 11),
        "overall_recall10_absolute_gain_ci95": paired_bootstrap_candidate(rows, "Recall@10", False, seed + 21),
        "tail_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(tail, "NDCG@10", True, seed + 31),
    }


def gacr_row(sample: dict, baseline_items: Sequence[str], gram_rank: int | None, candidate_rank: int | None, target_group: str) -> dict:
    target = sample["positive_item"]
    expected = baseline_items.index(target) + 1 if target in baseline_items else None
    if gram_rank != expected:
        raise ValueError("matched GACR-v3 regenerated GRAM rank differs from A arm")
    baseline = metrics_for_rank(gram_rank)
    candidate = metrics_for_rank(candidate_rank)
    return {
        "sample_key": sample["sample_key"],
        "target_group": target_group,
        "graph_covered": 1,
        "baseline_rank": gram_rank,
        "candidate_rank": candidate_rank,
        **{f"baseline_{key}": value for key, value in baseline.items()},
        **{f"candidate_{key}": value for key, value in candidate.items()},
        "target_in_baseline_beam50": int(gram_rank is not None),
        "target_in_candidate_beam50": int(candidate_rank is not None),
        "new_hit_at10_outside_A_beam": int(gram_rank is None and candidate_rank is not None and candidate_rank <= 10),
        "changed": int(candidate_rank != gram_rank),
        "broad_harm": int(gram_rank is not None and gram_rank <= 10 and (candidate_rank is None or candidate_rank > 10)),
    }


def write_rows(path: Path, arm_rows: Mapping[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"arm": arm, **row} for arm, values in arm_rows.items() for row in values]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_domain(dataset: str, config: dict, output_root: Path) -> dict:
    device = torch.device("cuda:0")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    parent = json.loads((ROOT / config["inputs"]["phase4_parent_config"]).read_text())
    prepared = prepare(dataset, parent, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"
    expected_checkpoint = config["inputs"]["expected_parent_checkpoint_sha256"][dataset]
    if sha256(checkpoint) != expected_checkpoint:
        raise ValueError(f"{dataset} parent checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()

    train_sequences = read_train_sequences(ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt", 2)
    graph = build_indexed_graph(train_sequences, prepared["catalog"])
    graph_model, graph_training = train_lightgcn(graph, config["graph"], device)
    graph_model.eval()
    with torch.no_grad():
        propagated = graph_model.propagate(graph.edges.to(device))

    split_root = ROOT / config["inputs"]["split_root"] / dataset
    phase4_train = read_users(split_root / "train_users.txt")
    phase4_validation = read_users(split_root / "validation_users.txt")
    development_users, cohort = select_development_users(
        set(prepared["sequences"]),
        phase4_train,
        phase4_validation,
        dataset,
        config["prior_validation_salts"],
        config["development_salt"],
        int(config["development_users_per_dataset"]),
    )
    expected_cohort = config["expected_development_cohort"][dataset]
    if cohort["user_sha256"] != expected_cohort["user_sha256"] or cohort["users"] != expected_cohort["users"]:
        raise ValueError(f"{dataset} development cohort lineage mismatch")

    item_paths = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    _, leaf_fractions = graph_prefix_inputs(
        item_paths, {item: 0.0 for item in prepared["catalog"]}
    )
    maximum_depth = max(len(path) for path in item_paths.values()) - 1
    fit_users, calibration_users = split_training_users(phase4_train, int(config["seed"]), dataset)
    fit_pool = build_train_samples(prepared["sequences"], fit_users, prepared["item2input"], prepared["item2lexid"])
    calibration_pool = build_train_samples(prepared["sequences"], calibration_users, prepared["item2input"], prepared["item2lexid"])
    fit_count = int(config["adapter"]["fit_samples_per_dataset"]) // 2
    calibration_count = int(config["adapter"]["calibration_samples_per_dataset"]) // 2
    fit_samples = select_stratified_samples(fit_pool, prepared["heads"], int(config["seed"]), f"{dataset}|gcgd-p1-adapter-fit", fit_count, fit_count)
    calibration_samples = select_stratified_samples(calibration_pool, prepared["heads"], int(config["seed"]), f"{dataset}|gcgd-p1-adapter-calibration", calibration_count, calibration_count)

    def adapter_records(samples: list[dict], label: str) -> list[dict]:
        records = []
        for index, sample in enumerate(samples, 1):
            logits = graph_logits_for_user(
                graph_model, graph, sample["user_id"],
                visible_history_items=sample["history_items"],
                propagated_embeddings=propagated,
            )
            records.append(build_adapter_record(sample, prepared, item_paths, logits, leaf_fractions, device, maximum_depth, int(config["seed"]), dataset))
            if index % 128 == 0:
                print(f"GCGD_P1_ADAPTER_RECORDS dataset={dataset} split={label} records={index}/{len(samples)}", flush=True)
        return records

    fit_records = adapter_records(fit_samples, "fit")
    calibration_records = adapter_records(calibration_samples, "calibration")
    adapter, adapter_training = train_adapter(fit_records, config, device)
    calibration_diagnostics = adapter_diagnostics(adapter, calibration_records, device)

    development_samples = build_validation_samples(
        prepared["sequences"], set(development_users), prepared["item2input"], prepared["item2lexid"]
    )
    residual_path = ROOT / config["inputs"]["gacr_v3_residual_root"] / dataset / "residual_seed2023.pt"
    if sha256(residual_path) != config["inputs"]["expected_gacr_v3_seed2023_residual_sha256"][dataset]:
        raise ValueError(f"{dataset} GACR-v3 residual SHA mismatch")
    residual_state = torch.load(residual_path, map_location=device)
    comparator_config = {
        "generator_top_k": int(config["decoding"]["generator_top_k"]),
        "catalog_top_k": int(config["decoding"]["catalog_top_k_for_gacr_v3"]),
    }
    arm_rows: dict[str, list[dict]] = {"B": [], "C": [], "GACR_v3": []}
    identity_exact = False
    torch.cuda.reset_peak_memory_stats(device)
    for index, sample in enumerate(development_samples, 1):
        logits = graph_logits_for_user(
            graph_model, graph, sample["user_id"],
            visible_history_items=sample["history_items"],
            propagated_embeddings=propagated,
        )
        prefix_scores, leaf_fractions = graph_prefix_inputs(item_paths, logits)
        baseline, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=None)
        if index == 1:
            identity_processor = AdaptiveGraphPrefixLogitsProcessor(prefix_scores, leaf_fractions, alpha=0.0, maximum_depth=maximum_depth, adapter=adapter)
            identity, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=identity_processor)
            identity_exact = baseline == identity
            if not identity_exact:
                raise ValueError("alpha=0 exact identity failed")
        b_processor = AdaptiveGraphPrefixLogitsProcessor(prefix_scores, leaf_fractions, alpha=float(config["decoding"]["B_alpha"]), maximum_depth=maximum_depth, adapter=None)
        c_processor = AdaptiveGraphPrefixLogitsProcessor(prefix_scores, leaf_fractions, alpha=float(config["decoding"]["C_alpha"]), maximum_depth=maximum_depth, adapter=adapter)
        b_items, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=b_processor)
        c_items, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=c_processor)
        target_group = "head" if sample["positive_item"] in prepared["heads"] else "tail"
        graph_covered = bool(prefix_scores.get((0,)))
        arm_rows["B"].append(arm_metric_row(sample_key=sample["sample_key"], target=sample["positive_item"], baseline_items=baseline, candidate_items=b_items, target_group=target_group, graph_covered=graph_covered))
        arm_rows["C"].append(arm_metric_row(sample_key=sample["sample_key"], target=sample["positive_item"], baseline_items=baseline, candidate_items=c_items, target_group=target_group, graph_covered=graph_covered))
        gram_rank, gacr_rank = matched_gacr_v3_rank(sample, prepared, comparator_config, residual_state, budget=float(config["inputs"]["gacr_v3_domain_budget"][dataset]), device=device)
        arm_rows["GACR_v3"].append(gacr_row(sample, baseline, gram_rank, gacr_rank, target_group))
        if index % 16 == 0:
            print(f"GCGD_P1_VALIDATION dataset={dataset} users={index}/{len(development_samples)}", flush=True)

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph_model.state_dict(), output_dir / "lightgcn.pt")
    torch.save(adapter.state_dict(), output_dir / "adapter.pt")
    write_rows(output_dir / "per_user.csv", arm_rows)
    methods = {
        arm: {
            "groups": summarize_rows(rows),
            "bootstrap": bootstrap_rows(rows, int(config["seed"]) + offset),
        }
        for offset, (arm, rows) in enumerate(arm_rows.items(), 1)
    }
    result = {
        "dataset": dataset,
        "status": "PASS",
        "cohort": cohort,
        "graph_training": graph_training,
        "adapter": {
            "fit_records": len(fit_records),
            "calibration_records": len(calibration_records),
            "fit_calibration_user_overlap": len(fit_users & calibration_users),
            "training": adapter_training,
            "calibration_diagnostics": calibration_diagnostics,
        },
        "methods": methods,
        "alpha_zero_identity_exact": identity_exact,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "parent_checkpoint_sha256_before": expected_checkpoint,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
        "test_read": False,
        "sports_read": False,
        "effect_decision_automatic": False,
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps({"dataset": dataset, "status": "PASS", "methods": methods}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("execution_enabled") is not True or config.get("decision_status") != "PREREGISTERED_FROZEN_READY_TO_RUN":
        raise ValueError("formal P1 config is not frozen and enabled")
    if not torch.cuda.is_available():
        raise RuntimeError("formal P1 requires CUDA")
    run_domain(args.dataset, config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
