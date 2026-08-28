#!/usr/bin/env python3
"""Clean-room SpecGR-to-GRAM adapters frozen by the Stage16 fidelity matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from official_specgr_runtime import load_official_unisrec_class, official_unisrec_config


def constrained_draft(logits: torch.Tensor, size: int) -> torch.Tensor:
    if logits.ndim != 2 or size < 1 or size > logits.shape[1]:
        raise ValueError("Invalid constrained-draft shape or size")
    indices = torch.topk(logits, size, dim=1).indices
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    logits[rows, indices] = float("-inf")
    return indices


def target_aware_score(
    logits: torch.Tensor, candidates: torch.Tensor, prefix_lengths: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 3 or candidates.shape != logits.shape[:2]:
        raise ValueError("Logits/candidate shapes do not align")
    if prefix_lengths.shape != (logits.shape[0],) or torch.any(prefix_lengths < 1):
        raise ValueError("Each candidate requires a positive identifiable prefix")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), candidates.reshape(-1), reduction="none"
    ).reshape_as(candidates)
    positions = torch.arange(candidates.shape[1], device=logits.device).unsqueeze(0)
    mask = positions < prefix_lengths.unsqueeze(1)
    return -(losses * mask).sum(dim=1) / prefix_lengths


def strict_accept(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return scores > threshold


def guided_prefix_mask(
    item_paths: Sequence[Sequence[int]], live_beam_prefixes: Sequence[Sequence[int]]
) -> torch.Tensor:
    prefixes = {tuple(int(token) for token in row) for row in live_beam_prefixes}
    if not prefixes:
        return torch.zeros(len(item_paths), dtype=torch.bool)
    lengths = {len(prefix) for prefix in prefixes}
    if len(lengths) != 1:
        raise ValueError("Live beam prefixes must have one shared depth")
    depth = next(iter(lengths))
    return torch.tensor(
        [len(path) >= depth and tuple(int(token) for token in path[:depth]) in prefixes for path in item_paths],
        dtype=torch.bool,
    )


def adaptive_exit(accepted_count: int, top_k: int, round_depth: int, maximum_depth: int) -> bool:
    if min(accepted_count, top_k, round_depth, maximum_depth) < 0:
        raise ValueError("Adaptive-exit counters must be non-negative")
    return accepted_count >= top_k or round_depth >= maximum_depth


def finalize_recommendations(
    accepted: Sequence[tuple[str, float]],
    rejected: Sequence[tuple[str, float]],
    verifier_fallback: Sequence[tuple[str, float]],
    top_k: int,
) -> list[tuple[str, float]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected: dict[str, float] = {}
    for item, score in accepted:
        selected.setdefault(item, float(score))
    if len(selected) < top_k:
        pool: dict[str, float] = {}
        for item, score in (*rejected, *verifier_fallback):
            if item not in selected:
                pool[item] = max(pool.get(item, float("-inf")), float(score))
        for item, score in sorted(pool.items(), key=lambda row: (-row[1], row[0])):
            selected[item] = score
            if len(selected) >= top_k:
                break
    if len(selected) < top_k:
        raise ValueError("Insufficient unique items for recommendation fallback")
    return sorted(selected.items(), key=lambda row: (-row[1], row[0]))[:top_k]


class OfficialUniSRecDrafterGRAM(nn.Module):
    """Pinned official UniSRec with GRAM's frozen content-vector width."""

    def __init__(self, train_item_embeddings_with_padding: torch.Tensor):
        super().__init__()
        if train_item_embeddings_with_padding.ndim != 2:
            raise ValueError("UniSRec content embeddings must be a matrix")
        if torch.count_nonzero(train_item_embeddings_with_padding[0]).item() != 0:
            raise ValueError("Row zero must be the padding embedding")
        official_class = load_official_unisrec_class()
        config = official_unisrec_config(train_item_embeddings_with_padding.shape[1])
        self.model = official_class(config, train_item_embeddings_with_padding)
        self.input_dimension = train_item_embeddings_with_padding.shape[1]

    def calculate_loss(
        self, item_sequence: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model.calculate_loss(item_sequence, lengths, labels)

    def inductive_scores(
        self,
        item_sequence: torch.Tensor,
        lengths: torch.Tensor,
        history_content_embeddings: torch.Tensor,
        candidate_content_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.predict(
            item_sequence,
            lengths,
            history_content_embeddings,
            candidate_content_embeddings,
        )


def sequence_item_contrastive_loss(
    sequence_embeddings: torch.Tensor,
    positive_item_embeddings: torch.Tensor,
    item_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if sequence_embeddings.shape != positive_item_embeddings.shape:
        raise ValueError("Sequence and positive-item embeddings must align")
    logits = sequence_embeddings @ positive_item_embeddings.T / temperature
    duplicate = item_ids.unsqueeze(1).eq(item_ids.unsqueeze(0))
    duplicate.fill_diagonal_(False)
    # Preserve the official implementation exactly: same-positive off-diagonal
    # logits are replaced by zero before exponentiation (thus contribute 1).
    logits = logits.masked_fill(duplicate, 0.0)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def splus_pretrain_loss(
    contrastive_loss: torch.Tensor,
    generative_loss: torch.Tensor,
    lambda_embedding: float = 6.0,
    lambda_generation: float = 1.0,
) -> torch.Tensor:
    return lambda_embedding * contrastive_loss + lambda_generation * generative_loss


def splus_finetune_loss(
    ranking_loss: torch.Tensor,
    generative_loss: torch.Tensor,
    lambda_embedding: float = 6.0,
    lambda_generation: float = 1.0,
) -> torch.Tensor:
    return lambda_embedding * ranking_loss + lambda_generation * generative_loss


class GRAMSelfDrafter(nn.Module):
    """Official SpecGR++ normalized projection/index algebra over GRAM states."""

    def __init__(self, gram_hidden_size: int = 512, projection_size: int = 64):
        super().__init__()
        self.projection = nn.Linear(gram_hidden_size, projection_size)

    def pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape[:2] != attention_mask.shape:
            raise ValueError("Hidden states and attention mask do not align")
        denominator = attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden_states.dtype)
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / denominator
        return F.normalize(self.projection(pooled), dim=-1)

    @staticmethod
    def draft_logits(sequence_embeddings: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        return F.normalize(sequence_embeddings, dim=-1) @ F.normalize(item_index, dim=-1).T

    @staticmethod
    def ranking_loss(
        sequence_embeddings: torch.Tensor,
        frozen_train_item_index: torch.Tensor,
        target_indices: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        logits = GRAMSelfDrafter.draft_logits(sequence_embeddings, frozen_train_item_index)
        return F.cross_entropy(logits / temperature, target_indices)


@dataclass(frozen=True)
class TrainingBudget:
    dataset_manifest_sha256: str
    transitions: int
    epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    physical_microbatch: int
    gradient_accumulation: int
    optimizer_steps: int
    gpu_count: int
    timeout_seconds: int


def assert_splus_control_budget_match(splus: TrainingBudget, control: TrainingBudget) -> dict:
    left, right = asdict(splus), asdict(control)
    differences = {key: (left[key], right[key]) for key in left if left[key] != right[key]}
    if differences:
        raise ValueError(f"S-PLUS/CTRL budget mismatch: {differences}")
    return {"matched": True, "fields": sorted(left), "budget": left}


def validate_cold_content_only(
    train_labels: Iterable[str], cold_items: Iterable[str], candidate_items: Iterable[str]
) -> dict:
    labels, cold, candidates = set(train_labels), set(cold_items), set(candidate_items)
    leaked = labels & cold
    missing = cold - candidates
    if leaked or missing:
        raise ValueError(
            f"Cold content-only contract failed: label_leaks={len(leaked)}, missing_candidates={len(missing)}"
        )
    return {
        "cold_interaction_label_count": 0,
        "cold_candidate_content_count": len(cold),
        "cold_candidates_complete": True,
    }
