#!/usr/bin/env python3
"""Design utilities for phase-7 Graph-Conditioned GRAM Decoding.

This module intentionally contains no training entry point yet.  It freezes the
mathematical prefix projection and validates resource/integrity declarations
before the P0 implementation is allowed to grow into a GPU experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


Token = Hashable
ItemPath = Sequence[Token]


class LightGCN(nn.Module):
    """Minimal full-batch LightGCN used by the frozen P1 graph branch."""

    def __init__(self, users: int, items: int, embedding_dim: int, layers: int) -> None:
        super().__init__()
        if users <= 0 or items <= 0 or embedding_dim <= 0 or layers < 0:
            raise ValueError("invalid LightGCN dimensions")
        self.users = users
        self.items = items
        self.layers = layers
        self.embedding = nn.Embedding(users + items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self, user_item_edges: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        if user_item_edges.ndim != 2 or user_item_edges.shape[0] != 2:
            raise ValueError("user_item_edges must have shape [2, E]")
        users, items = user_item_edges
        if users.numel() == 0:
            raise ValueError("LightGCN graph must contain edges")
        if int(users.min()) < 0 or int(users.max()) >= self.users:
            raise ValueError("user index outside LightGCN range")
        if int(items.min()) < 0 or int(items.max()) >= self.items:
            raise ValueError("item index outside LightGCN range")
        item_nodes = items + self.users
        source = torch.cat((users, item_nodes))
        target = torch.cat((item_nodes, users))
        nodes = self.users + self.items
        degree = torch.zeros(nodes, dtype=self.embedding.weight.dtype, device=source.device)
        degree.index_add_(0, source, torch.ones_like(source, dtype=degree.dtype))
        norm = degree[source].clamp_min(1).rsqrt() * degree[target].clamp_min(1).rsqrt()
        current = self.embedding.weight
        layers = [current]
        for _ in range(self.layers):
            following = torch.zeros_like(current)
            following.index_add_(0, target, current[source] * norm[:, None])
            current = following
            layers.append(current)
        final = torch.stack(layers).mean(dim=0)
        return final[: self.users], final[self.users :]

    def bpr_loss(
        self,
        user_item_edges: torch.LongTensor,
        users: torch.LongTensor,
        positives: torch.LongTensor,
        negatives: torch.LongTensor,
        l2: float,
    ) -> torch.Tensor:
        user_embedding, item_embedding = self.propagate(user_item_edges)
        positive_score = (user_embedding[users] * item_embedding[positives]).sum(dim=-1)
        negative_score = (user_embedding[users] * item_embedding[negatives]).sum(dim=-1)
        ranking = -F.logsigmoid(positive_score - negative_score).mean()
        regularizer = (
            self.embedding(users).square().sum()
            + self.embedding(positives + self.users).square().sum()
            + self.embedding(negatives + self.users).square().sum()
        ) / max(1, users.numel())
        return ranking + float(l2) * regularizer


class GraphReliabilityAdapter(nn.Module):
    """Frozen P1 C-arm scalar adapter and target-free reliability gate."""

    def __init__(self, feature_dim: int = 6, hidden_dim: int = 16) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.gate[0].in_features:
            raise ValueError("reliability feature width mismatch")
        gate = torch.sigmoid(self.gate(features)).squeeze(-1)
        temperature = 0.5 + 1.5 * torch.sigmoid(self.raw_temperature)
        return gate, temperature


def reliability_features(
    *,
    graph_coverage: float,
    normalized_entropy: float,
    top_margin: float,
    compatible_leaf_fraction: float,
    gram_graph_agreement: float,
    normalized_depth: float,
) -> tuple[float, ...]:
    values = (
        graph_coverage,
        normalized_entropy,
        top_margin,
        compatible_leaf_fraction,
        gram_graph_agreement,
        normalized_depth,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("all reliability features must be finite values in [0, 1]")
    return values


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def normalize_item_logits(item_logits: Mapping[str, float]) -> dict[str, float]:
    """Return catalog-normalized log probabilities for graph item logits."""
    if not item_logits:
        raise ValueError("item_logits must not be empty")
    if not all(math.isfinite(float(value)) for value in item_logits.values()):
        raise ValueError("item_logits must all be finite")
    normalizer = _logsumexp([float(value) for value in item_logits.values()])
    return {item: float(value) - normalizer for item, value in item_logits.items()}


def aggregate_graph_prefix_logits(
    item_paths: Mapping[str, ItemPath],
    item_logits: Mapping[str, float],
    prefix: Sequence[Token],
) -> dict[Token, float]:
    """Marginalize normalized graph item mass onto the next legal Trie token.

    Items absent from either mapping are ignored.  Returned values are log
    probabilities whose exponentiated mass sums to one over the compatible
    continuation items.  An empty mapping means that the graph branch must
    abstain and the decoder must fall back to GRAM.
    """
    log_probabilities = normalize_item_logits(item_logits)
    depth = len(prefix)
    grouped: dict[Token, list[float]] = defaultdict(list)
    for item, path in item_paths.items():
        if item not in log_probabilities or len(path) <= depth:
            continue
        if tuple(path[:depth]) != tuple(prefix):
            continue
        grouped[path[depth]].append(log_probabilities[item])
    if not grouped:
        return {}
    continuation_log_mass = _logsumexp([value for values in grouped.values() for value in values])
    return {
        token: _logsumexp(values) - continuation_log_mass
        for token, values in grouped.items()
    }


def fuse_token_logits(
    gram_logits: Mapping[Token, float],
    graph_prefix_logits: Mapping[Token, float],
    *,
    alpha: float,
    gate: float,
) -> dict[Token, float]:
    """Fuse graph evidence only into legal GRAM tokens; gate=0 is identity."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
        raise ValueError("gate must be in [0, 1]")
    if not graph_prefix_logits or gate == 0.0 or alpha == 0.0:
        return dict(gram_logits)
    illegal = set(graph_prefix_logits) - set(gram_logits)
    if illegal:
        raise ValueError(f"graph logits include tokens outside the legal Trie set: {illegal!r}")
    return {
        token: float(logit) + alpha * gate * float(graph_prefix_logits.get(token, 0.0))
        for token, logit in gram_logits.items()
    }


def validate_design_config(config: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    execution = config.get("execution")
    integrity = config.get("integrity")
    if not isinstance(execution, Mapping):
        return ["execution must be an object"]
    if not isinstance(integrity, Mapping):
        return ["integrity must be an object"]
    if execution.get("physical_gpu") != 0 or execution.get("cuda_visible_devices") != "0":
        errors.append("phase7 must use physical GPU0 only")
    total = execution.get("total_gpu_lease_mib")
    peak = execution.get("expected_workload_peak_mib")
    sidecar = execution.get("sidecar_reservation_mib")
    if not isinstance(total, int) or total != 30720:
        errors.append("total GPU lease must be exactly 30720 MiB")
    domain_leases = execution.get("domain_gpu_lease_mib")
    if isinstance(domain_leases, Mapping):
        for domain in ("Toys", "Beauty"):
            lease = domain_leases.get(domain)
            if not isinstance(lease, Mapping):
                errors.append(f"execution.domain_gpu_lease_mib.{domain} must be an object")
                continue
            domain_peak = lease.get("expected_workload_peak_mib")
            domain_sidecar = lease.get("sidecar_reservation_mib")
            if not all(isinstance(value, int) for value in (domain_peak, domain_sidecar)):
                errors.append(f"{domain} peak and sidecar reservations must be integers")
            elif domain_peak <= 0 or domain_sidecar < 0 or domain_peak + domain_sidecar != total:
                errors.append(f"{domain} workload peak plus sidecar must equal total GPU lease")
    elif not all(isinstance(value, int) for value in (total, peak, sidecar)):
        errors.append("peak and sidecar reservations must be integers")
    elif peak < 0 or sidecar < 0 or peak + sidecar != total:
        errors.append("workload peak plus sidecar must equal total GPU lease")
    for key in (
        "background_tmux_required",
        "codellama_must_be_running_before_start",
        "stop_codellama_before_workload",
        "restore_codellama_after_every_exit",
        "no_automatic_retry",
    ):
        if execution.get(key) is not True:
            errors.append(f"execution.{key} must be true")
    for key in ("test_predictions_forbidden", "sports_forbidden"):
        if integrity.get(key) is not True:
            errors.append(f"integrity.{key} must be true")
    if config.get("execution_enabled") is not False:
        errors.append("design config must remain execution_enabled=false before preregistration")
    if config.get("scientific_workload_implemented") is not False:
        errors.append("design config must remain scientific_workload_implemented=false")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-check", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.design_check.read_text())
    errors = validate_design_config(config)
    result = {
        "experiment_id": config.get("experiment_id"),
        "design_check": "passed" if not errors else "failed",
        "execution_enabled": config.get("execution_enabled"),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
