#!/usr/bin/env python3
"""Reusable P1 components for Graph-Conditioned GRAM Decoding.

The executable entry point remains deliberately absent until matched decoding,
GACR-v3 comparison, and domain-specific GPU lease orchestration are complete.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = ROOT / "GRAM/src"
for path in (ROOT, GRAM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment.phase4.gacr_p0 import select_fresh_validation_users
from experiment.phase4.gacr_s0 import BoundedResidualRanker, build_candidate_record, stable_ranking
from experiment.phase4.gcdh_p0 import collate, normalized_sequence
from experiment.phase6.gacr_v3 import residual_safety_multiplier
from experiment.phase7.gcgd_p0_gpu_smoke import prefix_log_probabilities
from experiment.phase7.gcgd_v1 import GraphReliabilityAdapter, LightGCN, reliability_features
from utils import generation_trie as gt


@dataclass(frozen=True)
class IndexedBipartiteGraph:
    users: tuple[str, ...]
    items: tuple[str, ...]
    edges: torch.LongTensor
    user_history: tuple[frozenset[int], ...]


def graph_prefix_inputs(
    item_paths: Mapping[str, Sequence[int]],
    item_logits: Mapping[str, float],
) -> tuple[dict[tuple[int, ...], dict[int, float]], dict[tuple[int, ...], float]]:
    if set(item_paths) != set(item_logits):
        raise ValueError("item paths and graph logits must cover the same catalog")
    log_probabilities = prefix_log_probabilities(item_paths, item_logits)
    leaf_counts: dict[tuple[int, ...], int] = {}
    for raw_path in item_paths.values():
        path = tuple(raw_path)
        for depth in range(1, len(path)):
            prefix = path[:depth]
            leaf_counts[prefix] = leaf_counts.get(prefix, 0) + 1
    total = len(item_paths)
    fractions = {prefix: count / total for prefix, count in leaf_counts.items()}
    return log_probabilities, fractions


class AdaptiveGraphPrefixLogitsProcessor(LogitsProcessor):
    """Inject fixed-B or adaptive-C graph evidence without adding illegal tokens."""

    def __init__(
        self,
        prefix_scores: Mapping[tuple[int, ...], Mapping[int, float]],
        compatible_leaf_fraction: Mapping[tuple[int, ...], float],
        *,
        alpha: float,
        maximum_depth: int,
        adapter: GraphReliabilityAdapter | None,
    ) -> None:
        self.prefix_scores = prefix_scores
        self.compatible_leaf_fraction = compatible_leaf_fraction
        self.alpha = float(alpha)
        self.maximum_depth = max(1, int(maximum_depth))
        self.adapter = adapter
        self.calls = 0
        self.applied_rows = 0
        self.gates: list[float] = []

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.calls += 1
        if self.alpha == 0.0:
            return scores
        output = scores.clone()
        for row, token_ids in enumerate(input_ids.tolist()):
            prefix = tuple(token_ids)
            graph = self.prefix_scores.get(prefix)
            if not graph:
                continue
            tokens = list(graph)
            graph_probability = [math.exp(graph[token]) for token in tokens]
            if len(tokens) == 1:
                entropy = 0.0
                margin = 1.0
            else:
                entropy = -sum(value * math.log(max(value, 1e-30)) for value in graph_probability)
                entropy /= math.log(len(tokens))
                ordered = sorted(graph_probability, reverse=True)
                margin = ordered[0] - ordered[1]
            gram_top = max(tokens, key=lambda token: (float(scores[row, token]), -token))
            graph_top = max(tokens, key=lambda token: (graph[token], -token))
            if self.adapter is None:
                gate_value = torch.tensor(1.0, device=scores.device)
                temperature = torch.tensor(1.0, device=scores.device)
            else:
                feature = torch.tensor(
                    [reliability_features(
                        graph_coverage=1.0,
                        normalized_entropy=float(entropy),
                        top_margin=float(margin),
                        compatible_leaf_fraction=float(self.compatible_leaf_fraction[prefix]),
                        gram_graph_agreement=float(gram_top == graph_top),
                        normalized_depth=min(1.0, (len(prefix) - 1) / self.maximum_depth),
                    )],
                    dtype=scores.dtype,
                    device=scores.device,
                )
                gate_value, temperature = self.adapter(feature)
                gate_value = gate_value[0]
            for token, value in graph.items():
                output[row, token] += self.alpha * gate_value * temperature * float(value)
            self.applied_rows += 1
            self.gates.append(float(gate_value))
        return output


def stable_sha(values: Sequence[str] | set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def build_indexed_graph(
    train_sequences: Mapping[str, Sequence[str]], catalog: Sequence[str]
) -> IndexedBipartiteGraph:
    users = tuple(sorted(train_sequences))
    items = tuple(catalog)
    if len(items) != len(set(items)):
        raise ValueError("catalog contains duplicate items")
    item_to_index = {item: index for index, item in enumerate(items)}
    edge_users: list[int] = []
    edge_items: list[int] = []
    histories: list[frozenset[int]] = []
    for user_index, user in enumerate(users):
        history: set[int] = set()
        for item in train_sequences[user]:
            if item not in item_to_index:
                raise ValueError(f"train item outside catalog: {item}")
            history.add(item_to_index[item])
        if not history:
            raise ValueError(f"empty train history: {user}")
        histories.append(frozenset(history))
        for item_index in sorted(history):
            edge_users.append(user_index)
            edge_items.append(item_index)
    edges = torch.tensor([edge_users, edge_items], dtype=torch.long)
    return IndexedBipartiteGraph(users, items, edges, tuple(histories))


def seeded_negative_items(
    graph: IndexedBipartiteGraph,
    edge_users: torch.LongTensor,
    *,
    seed: int,
) -> torch.LongTensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = torch.randint(len(graph.items), edge_users.shape, generator=generator)
    for position, user_index in enumerate(edge_users.tolist()):
        history = graph.user_history[user_index]
        if len(history) >= len(graph.items):
            raise ValueError("user history covers the entire catalog")
        while int(result[position]) in history:
            result[position] = torch.randint(len(graph.items), (), generator=generator)
    return result


def lightgcn_epoch_loss(
    model: LightGCN,
    edges: torch.LongTensor,
    negatives: torch.LongTensor,
    *,
    batch_size: int,
    l2: float,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    users, positives = edges
    if negatives.shape != positives.shape:
        raise ValueError("negative item shape mismatch")
    user_embedding, item_embedding = model.propagate(edges)
    losses = []
    weights = []
    for start in range(0, users.numel(), batch_size):
        selected = slice(start, start + batch_size)
        u = user_embedding[users[selected]]
        positive = item_embedding[positives[selected]]
        negative = item_embedding[negatives[selected]]
        losses.append(-torch.nn.functional.logsigmoid(
            (u * positive).sum(dim=-1) - (u * negative).sum(dim=-1)
        ).mean())
        weights.append(users[selected].numel())
    ranking = sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights)
    regularizer = model.embedding.weight.square().mean()
    return ranking + float(l2) * regularizer


def train_lightgcn(
    graph: IndexedBipartiteGraph,
    config: Mapping[str, object],
    device: torch.device,
) -> tuple[LightGCN, list[dict[str, float | int]]]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LightGCN(
        len(graph.users),
        len(graph.items),
        int(config["embedding_dim"]),
        int(config["layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=0.0,
    )
    edges = graph.edges.to(device)
    objective_weight = float(config.get("objective_weight", 1.0))
    if not math.isfinite(objective_weight) or objective_weight <= 0.0:
        raise ValueError("LightGCN objective_weight must be finite and positive")
    records = []
    for epoch in range(1, int(config["epochs"]) + 1):
        negatives = seeded_negative_items(
            graph, graph.edges[0], seed=seed + epoch
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        raw_loss = lightgcn_epoch_loss(
            model,
            edges,
            negatives,
            batch_size=int(config["batch_size"]),
            l2=float(config["l2"]),
        )
        loss = objective_weight * raw_loss
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite LightGCN loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(gradient_norm):
            raise ValueError(f"non-finite LightGCN gradient at epoch {epoch}")
        optimizer.step()
        records.append({
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "raw_bpr_l2_loss": float(raw_loss.detach().cpu()),
            "objective_weight": objective_weight,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        })
    return model, records


@torch.no_grad()
def graph_logits_for_user(
    model: LightGCN,
    graph: IndexedBipartiteGraph,
    user: str,
    *,
    visible_history_items: Sequence[str] | None = None,
    propagated_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    seen_item_sentinel: float = -10000.0,
) -> dict[str, float]:
    try:
        user_index = graph.users.index(user)
    except ValueError as error:
        raise ValueError(f"user absent from train graph: {user}") from error
    if propagated_embeddings is None:
        edges = graph.edges.to(model.embedding.weight.device)
        user_embedding, item_embedding = model.propagate(edges)
    else:
        user_embedding, item_embedding = propagated_embeddings
        if user_embedding.shape[0] != len(graph.users) or item_embedding.shape[0] != len(graph.items):
            raise ValueError("propagated LightGCN embedding shape mismatch")
    scores = item_embedding @ user_embedding[user_index]
    scores = scores.detach().cpu().tolist()
    if visible_history_items is None:
        seen_item_indices = graph.user_history[user_index]
    else:
        item_to_index = {item: index for index, item in enumerate(graph.items)}
        unknown = sorted(set(visible_history_items) - set(item_to_index))
        if unknown:
            raise ValueError(f"visible history contains item outside catalog: {unknown[0]}")
        seen_item_indices = {item_to_index[item] for item in visible_history_items}
    for item_index in seen_item_indices:
        scores[item_index] = float(seen_item_sentinel)
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("non-finite LightGCN item score")
    return dict(zip(graph.items, map(float, scores)))


def adapter_training_loss(
    adapter: GraphReliabilityAdapter,
    features: torch.Tensor,
    gram_legal_logits: torch.Tensor,
    graph_legal_log_probabilities: torch.Tensor,
    target_positions: torch.LongTensor,
    reliability_labels: torch.Tensor,
    *,
    alpha: float,
    next_token_weight: float,
    reliability_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gate, temperature = adapter(features)
    fused = gram_legal_logits + float(alpha) * gate[:, None] * temperature * graph_legal_log_probabilities
    next_token = F.cross_entropy(fused, target_positions)
    reliability = F.binary_cross_entropy(gate, reliability_labels.to(gate.dtype))
    total = float(next_token_weight) * next_token + float(reliability_weight) * reliability
    return total, {
        "next_token_ce": next_token,
        "gate_reliability_bce": reliability,
        "temperature": temperature,
    }


@torch.no_grad()
def generate_arm_items(
    sample: Mapping[str, object],
    prepared: dict,
    *,
    beam_size: int,
    length_penalty: float,
    device: torch.device,
    processor: LogitsProcessor | None,
) -> tuple[list[str], dict[str, int | float]]:
    inference_sample = dict(sample)
    inference_sample["output"] = prepared["item2lexid"][prepared["catalog"][0]]
    batch = collate(prepared["collator"], [inference_sample])
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    if "_gcgd_trie" not in prepared:
        prepared["_gcgd_trie"] = gt.Trie(prepared["encoded_candidates"])
        prepared["_gcgd_max_length"] = max(len(row) for row in prepared["encoded_candidates"])
    kwargs = {}
    if processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    prediction = prepared["model"].backbone.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_length=prepared["_gcgd_max_length"],
        prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(prepared["_gcgd_trie"]),
        num_beams=int(beam_size),
        num_return_sequences=int(beam_size),
        return_dict_in_generate=True,
        length_penalty=float(length_penalty),
        **kwargs,
    )["sequences"]
    items = [
        prepared["sequence_to_item"].get(normalized_sequence(row.tolist()))
        for row in prediction
    ]
    if any(item is None for item in items) or len(set(items)) != len(items):
        raise ValueError("invalid or duplicate constrained-beam item mapping")
    diagnostics = {
        "beam_items": len(items),
        "processor_calls": int(getattr(processor, "calls", 0)),
        "processor_applied_rows": int(getattr(processor, "applied_rows", 0)),
        "mean_gate": float(np.mean(getattr(processor, "gates", [0.0]))),
    }
    return items, diagnostics


@torch.no_grad()
def matched_gacr_v3_rank(
    sample: dict,
    prepared: dict,
    comparator_config: Mapping[str, object],
    residual_state: Mapping[str, torch.Tensor],
    *,
    budget: float,
    device: torch.device,
) -> tuple[int | None, int | None]:
    record = build_candidate_record(sample, prepared, dict(comparator_config), device)
    ranker = BoundedResidualRanker(6, 16, 0.2).to(device)
    ranker.load_state_dict(residual_state, strict=True)
    ranker.eval()
    if record["target_index"] is None:
        return record["gram_rank"], None
    residual = ranker(record["features"].to(device))
    multiplier = residual_safety_multiplier(residual, float(budget))
    ranking = stable_ranking(record["base"].to(device) + multiplier * residual)
    return record["gram_rank"], ranking.index(int(record["target_index"])) + 1


def arm_metric_row(
    *,
    sample_key: str,
    target: str,
    baseline_items: Sequence[str],
    candidate_items: Sequence[str],
    target_group: str,
    graph_covered: bool,
) -> dict[str, object]:
    baseline_rank = baseline_items.index(target) + 1 if target in baseline_items else None
    candidate_rank = candidate_items.index(target) + 1 if target in candidate_items else None
    baseline = metrics_for_rank(baseline_rank)
    candidate = metrics_for_rank(candidate_rank)
    return {
        "sample_key": sample_key,
        "target_group": target_group,
        "graph_covered": int(graph_covered),
        "baseline_rank": baseline_rank,
        "candidate_rank": candidate_rank,
        **{f"baseline_{key}": value for key, value in baseline.items()},
        **{f"candidate_{key}": value for key, value in candidate.items()},
        "target_in_baseline_beam50": int(baseline_rank is not None),
        "target_in_candidate_beam50": int(candidate_rank is not None),
        "new_hit_at10_outside_A_beam": int(baseline_rank is None and candidate_rank is not None and candidate_rank <= 10),
        "changed": int(tuple(baseline_items) != tuple(candidate_items)),
        "broad_harm": int(baseline_rank is not None and baseline_rank <= 10 and (candidate_rank is None or candidate_rank > 10)),
    }


def select_development_users(
    all_users: set[str],
    phase4_train_users: set[str],
    phase4_validation_users: set[str],
    dataset: str,
    prior_salts: Sequence[str],
    development_salt: str,
    development_users: int,
    historical_users_per_salt: int = 1024,
) -> tuple[list[str], dict[str, object]]:
    exclusions = set(phase4_train_users) | set(phase4_validation_users)
    prior_users: set[str] = set()
    for salt in prior_salts:
        cohort = set(select_fresh_validation_users(
            all_users,
            exclusions,
            dataset,
            salt,
            historical_users_per_salt,
        ))
        prior_users |= cohort
        exclusions |= cohort
    selected = select_fresh_validation_users(
        all_users,
        exclusions,
        dataset,
        development_salt,
        development_users,
    )
    overlap = set(selected) & exclusions
    if overlap:
        raise ValueError("P1 development cohort overlaps an excluded cohort")
    return selected, {
        "users": len(selected),
        "user_sha256": stable_sha(selected),
        "excluded_phase4_users": len(phase4_train_users | phase4_validation_users),
        "excluded_prior_validation_users": len(prior_users),
        "prior_salts": list(prior_salts),
        "overlap": 0,
    }


def metrics_for_rank(rank: int | None) -> dict[str, float]:
    return {
        "Recall@5": float(rank is not None and rank <= 5),
        "NDCG@5": 1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0,
        "Recall@10": float(rank is not None and rank <= 10),
        "NDCG@10": 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0,
        "Recall@50": float(rank is not None and rank <= 50),
        "MRR": 1.0 / rank if rank is not None else 0.0,
    }


def summarize_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize empty metric rows")
    keys = ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "Recall@50", "MRR")
    return {"n": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in keys}}
