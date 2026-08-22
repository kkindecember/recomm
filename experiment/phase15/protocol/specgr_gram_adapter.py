"""Clean-room SpecGR-to-GRAM contract primitives for Stage15 B2.

This module deliberately contains no training or evaluation entry point.  It
defines the auditable semantics that the later GPU smoke must satisfy for
variable-length GRAM lexical paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def lexical_tokens(lexical_id: str) -> tuple[str, ...]:
    tokens = tuple(token for token in lexical_id.split("|") if token)
    if not tokens:
        raise ValueError(f"Empty lexical path: {lexical_id!r}")
    return tokens


@dataclass(frozen=True)
class PathCatalog:
    paths: Mapping[str, tuple[str, ...]]
    warm_items: frozenset[str]
    cold_items: frozenset[str]

    @classmethod
    def build(
        cls,
        item_to_lexical: Mapping[str, str],
        warm_items: Iterable[str],
        cold_items: Iterable[str],
    ) -> "PathCatalog":
        paths = {item: lexical_tokens(path) for item, path in item_to_lexical.items()}
        warm = frozenset(warm_items)
        cold = frozenset(cold_items)
        if warm & cold:
            raise ValueError("Warm and cold item sets overlap")
        if warm | cold != set(paths):
            raise ValueError("Warm/cold sets must exactly partition the catalog")
        reverse: dict[tuple[str, ...], list[str]] = {}
        for item, path in paths.items():
            reverse.setdefault(path, []).append(item)
        collisions = [items for items in reverse.values() if len(items) != 1]
        if collisions:
            raise ValueError(f"Catalog contains {len(collisions)} path collisions")
        return cls(paths=paths, warm_items=warm, cold_items=cold)

    @property
    def max_depth(self) -> int:
        return max(map(len, self.paths.values()))

    def score_length(self, item: str, minimum_cold_prefix: int = 2) -> int:
        """Return the token count used by target-aware verification.

        Warm paths use their complete lexical path.  Cold paths use their
        longest prefix observed among warm catalog paths, matching SpecGR's
        exclusion of cold-only identification suffixes.  A two-token floor is
        frozen for the port and must not exceed the path length.
        """

        if item not in self.paths:
            raise KeyError(f"Unknown catalog item: {item}")
        path = self.paths[item]
        if item in self.warm_items:
            return len(path)
        if len(path) < minimum_cold_prefix:
            raise ValueError(f"Cold path too short for verifier contract: {item}")
        warm_prefixes = {
            warm_path[:depth]
            for warm_item in self.warm_items
            for warm_path in (self.paths[warm_item],)
            for depth in range(1, len(warm_path) + 1)
        }
        longest = max(
            (depth for depth in range(1, len(path) + 1) if path[:depth] in warm_prefixes),
            default=0,
        )
        return min(len(path), max(minimum_cold_prefix, longest))


class AuxiliaryContentDrafter(nn.Module):
    """Train-only inductive content drafter for the SpecGR-GRAM port.

    The model follows the auxiliary-drafter mechanism: frozen catalog content
    vectors are projected into a shared space, a causal Transformer summarizes
    the item history, and normalized dot products rank both warm and unseen
    cold items.  Only warm train transitions may be used as labels; cold items
    enter through their frozen content vectors at retrieval time.
    """

    def __init__(
        self,
        *,
        item_content_embeddings: torch.Tensor,
        hidden_size: int,
        max_history: int,
        transformer_layers: int,
        attention_heads: int,
        feedforward_size: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        if item_content_embeddings.ndim != 2 or item_content_embeddings.size(0) < 2:
            raise ValueError("Content embeddings must be a non-trivial matrix")
        if hidden_size < 1 or max_history < 1 or transformer_layers < 1:
            raise ValueError("Invalid drafter dimensions")
        if attention_heads < 1 or hidden_size % attention_heads:
            raise ValueError("attention_heads must divide hidden_size")
        if feedforward_size < 1:
            raise ValueError("feedforward_size must be positive")
        if not 0.0 <= dropout < 1.0 or temperature <= 0:
            raise ValueError("Invalid drafter dropout or temperature")
        content = item_content_embeddings.detach().float().contiguous()
        if not bool(torch.isfinite(content).all()):
            raise ValueError("Content embeddings must be finite")
        self.register_buffer("item_content_embeddings", content, persistent=False)
        self.hidden_size = int(hidden_size)
        self.max_history = int(max_history)
        self.temperature = float(temperature)
        self.content_projection = nn.Linear(content.size(1), hidden_size, bias=False)
        self.position_embedding = nn.Embedding(max_history, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def catalog_size(self) -> int:
        return int(self.item_content_embeddings.size(0))

    def _validate_histories(
        self, history_indices: torch.Tensor, history_lengths: torch.Tensor
    ) -> None:
        if history_indices.ndim != 2 or history_lengths.ndim != 1:
            raise ValueError("Drafter histories must be a matrix and length vector")
        if history_indices.size(0) != history_lengths.numel():
            raise ValueError("Drafter history batch and lengths do not align")
        if history_indices.size(1) != self.max_history:
            raise ValueError("Drafter histories must use the frozen max_history")
        if bool(((history_lengths < 1) | (history_lengths > self.max_history)).any()):
            raise ValueError("Drafter history length is outside the frozen range")
        active = torch.arange(self.max_history, device=history_indices.device)[None]
        active = active < history_lengths[:, None]
        if bool(((history_indices < 0) & active).any()):
            raise ValueError("Active drafter history positions cannot be padding")
        if bool(((history_indices >= self.catalog_size) & active).any()):
            raise ValueError("Drafter history contains an unknown catalog index")
        if bool((history_indices[~active] != -1).any()):
            raise ValueError("Inactive drafter history positions must equal -1")

    def project_catalog(self) -> torch.Tensor:
        return F.normalize(self.content_projection(self.item_content_embeddings), dim=-1)

    def encode_histories(
        self, history_indices: torch.Tensor, history_lengths: torch.Tensor
    ) -> torch.Tensor:
        self._validate_histories(history_indices, history_lengths)
        safe_indices = history_indices.clamp_min(0)
        projected = self.content_projection(self.item_content_embeddings[safe_indices])
        positions = torch.arange(self.max_history, device=history_indices.device)
        hidden = self.input_norm(projected + self.position_embedding(positions)[None])
        hidden = self.dropout(hidden).transpose(0, 1)
        causal_mask = torch.full(
            (self.max_history, self.max_history),
            float("-inf"),
            device=history_indices.device,
        ).triu(1)
        padding_mask = history_indices.eq(-1)
        encoded = self.history_encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        ).transpose(0, 1)
        row = torch.arange(history_indices.size(0), device=history_indices.device)
        last = encoded[row, history_lengths - 1]
        return F.normalize(last, dim=-1)

    def forward(
        self, history_indices: torch.Tensor, history_lengths: torch.Tensor
    ) -> torch.Tensor:
        histories = self.encode_histories(history_indices, history_lengths)
        return histories @ self.project_catalog().T / self.temperature


def drafter_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    warm_catalog_indices: Iterable[int],
) -> torch.Tensor:
    """Cross entropy that hard-fails if cold items enter drafter supervision."""

    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.numel():
        raise ValueError("Drafter logits and labels do not align")
    warm = {int(index) for index in warm_catalog_indices}
    observed = {int(index) for index in labels.detach().cpu().tolist()}
    if not observed.issubset(warm):
        raise ValueError("Drafter labels must be warm train-only catalog items")
    if any(index < 0 or index >= logits.size(1) for index in warm):
        raise ValueError("Warm catalog index is outside drafter logits")
    return F.cross_entropy(logits, labels)


def rank_drafter_items(
    scores: torch.Tensor,
    item_ids: Sequence[str],
    *,
    exclude_items: Iterable[str] = (),
) -> list[str]:
    """Full catalog ranking with finite-score and item-id tie contracts."""

    if scores.ndim != 1 or scores.numel() != len(item_ids):
        raise ValueError("Drafter scores and catalog IDs do not align")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Drafter catalog contains duplicate item IDs")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Drafter scores must be finite")
    excluded = set(exclude_items)
    unknown = excluded - set(item_ids)
    if unknown:
        raise ValueError("Drafter exclusion set contains unknown items")
    values = [float(value) for value in scores.detach().cpu()]
    return [
        item_ids[index]
        for index in sorted(
            (index for index, item in enumerate(item_ids) if item not in excluded),
            key=lambda index: (-values[index], item_ids[index]),
        )
    ]


def target_aware_score(token_log_probabilities: Sequence[float], score_length: int) -> float:
    """Length-normalized GRAM verifier score for one proposed item."""

    if score_length < 1 or score_length > len(token_log_probabilities):
        raise ValueError("score_length is outside the candidate path")
    selected = [float(value) for value in token_log_probabilities[:score_length]]
    if not all(math.isfinite(value) for value in selected):
        raise ValueError("Verifier log probabilities must be finite")
    return sum(selected) / score_length


def padded_candidate_labels(
    paths: Sequence[Sequence[int]], *, device: torch.device
) -> torch.Tensor:
    """Pad non-empty, EOS-free lexical paths for GRAM teacher forcing."""

    if not paths or any(not path for path in paths):
        raise ValueError("Candidate token paths must be non-empty")
    labels = torch.full(
        (len(paths), max(map(len, paths))),
        -100,
        dtype=torch.long,
        device=device,
    )
    for row, path in enumerate(paths):
        labels[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
    return labels


def candidate_token_log_probabilities(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather raw-vocabulary log probabilities without scoring padding."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Verifier logits and labels do not align")
    mask = labels.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every verifier candidate must contain at least one token")
    safe_labels = labels.masked_fill(~mask, 0)
    token_logp = F.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logp = token_logp.masked_fill(~mask, 0.0)
    if not bool(torch.isfinite(token_logp[mask]).all()):
        raise ValueError("Verifier produced non-finite token log probabilities")
    return token_logp, mask


def target_aware_scores_tensor(
    token_logp: torch.Tensor,
    token_mask: torch.Tensor,
    score_lengths: Sequence[int],
) -> torch.Tensor:
    """Vectorized target-aware mean score with per-candidate path lengths."""

    if token_logp.ndim != 2 or token_mask.shape != token_logp.shape:
        raise ValueError("Token scores and mask do not align")
    if len(score_lengths) != token_logp.size(0):
        raise ValueError("score_lengths does not align with candidates")
    lengths = torch.tensor(score_lengths, dtype=torch.long, device=token_logp.device)
    available = token_mask.sum(dim=1)
    if bool(((lengths < 1) | (lengths > available)).any()):
        raise ValueError("A target-aware score length is outside its candidate path")
    positions = torch.arange(token_logp.size(1), device=token_logp.device)[None]
    selected = positions < lengths[:, None]
    return (token_logp * selected).sum(dim=1) / lengths


def score_candidate_paths_with_frozen_gram(
    *,
    model,
    batch: Mapping[str, torch.Tensor],
    candidate_token_ids: Sequence[Sequence[Sequence[int]]],
    score_lengths: Sequence[Sequence[int]],
    candidate_chunk_size: int,
) -> dict[str, object]:
    """Encode each history once and score variable-length candidate paths.

    This is the Stage15 B2 GPU hook.  It performs no optimization and does not
    inspect recommendation targets.  Candidates are decoded by teacher forcing
    in bounded chunks, while the frozen GRAM encoder state is reused.
    """

    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")
    required = {"item_text_ids", "item_text_masks"}
    if not required.issubset(batch):
        raise ValueError("GRAM verifier batch is missing encoder inputs")
    batch_size = int(batch["item_text_ids"].size(0))
    if len(candidate_token_ids) != batch_size or len(score_lengths) != batch_size:
        raise ValueError("Candidate rows do not align with the GRAM batch")
    counts = {len(row) for row in candidate_token_ids}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise ValueError("Every history must have the same positive candidate budget")
    candidate_count = next(iter(counts))
    if any(len(row) != candidate_count for row in score_lengths):
        raise ValueError("Score-length rows do not align with the candidate budget")

    passages = int(batch["item_text_ids"].size(1))
    flat_input = batch["item_text_ids"].reshape(batch_size, -1)
    flat_mask = batch["item_text_masks"].reshape(batch_size, -1)
    model.encoder.n_passages = passages
    encoder_hidden = model.encoder(
        input_ids=flat_input,
        attention_mask=flat_mask,
        return_dict=True,
    )[0]

    score_chunks: list[torch.Tensor] = []
    token_rows: list[list[list[float]]] = [[] for _ in range(batch_size)]
    for start in range(0, candidate_count, candidate_chunk_size):
        stop = min(start + candidate_chunk_size, candidate_count)
        chunk_size = stop - start
        flat_paths = [
            path
            for candidates in candidate_token_ids
            for path in candidates[start:stop]
        ]
        flat_lengths = [
            length
            for lengths in score_lengths
            for length in lengths[start:stop]
        ]
        labels = padded_candidate_labels(flat_paths, device=flat_input.device)
        repeated_hidden = encoder_hidden.repeat_interleave(chunk_size, dim=0)
        repeated_mask = flat_mask.repeat_interleave(chunk_size, dim=0)
        outputs = model(
            encoder_outputs=(repeated_hidden,),
            attention_mask=repeated_mask,
            labels=labels,
        )
        token_logp, token_mask = candidate_token_log_probabilities(outputs.logits, labels)
        chunk_scores = target_aware_scores_tensor(token_logp, token_mask, flat_lengths)
        score_chunks.append(chunk_scores.view(batch_size, chunk_size))
        cursor = 0
        for batch_index in range(batch_size):
            for _candidate_index in range(chunk_size):
                active = token_mask[cursor]
                token_rows[batch_index].append(
                    [float(value) for value in token_logp[cursor][active].detach().cpu()]
                )
                cursor += 1

    scores = torch.cat(score_chunks, dim=1)
    if scores.shape != (batch_size, candidate_count):
        raise RuntimeError("Verifier hook produced an unexpected score matrix")
    return {
        "scores": scores,
        "token_log_probabilities": token_rows,
        "encoder_forward_histories": batch_size,
        "verifier_forward_candidates": batch_size * candidate_count,
        "candidate_chunks": math.ceil(candidate_count / candidate_chunk_size),
    }


def guided_redraft(
    ranked_drafter_items: Sequence[str],
    *,
    catalog: PathCatalog,
    verifier_prefixes: Iterable[Sequence[str]],
    prefix_depth: int,
    already_drafted: Iterable[str],
    draft_size: int,
) -> list[str]:
    """Select the next unique drafter batch under verifier-prefix guidance."""

    if draft_size < 1:
        raise ValueError("draft_size must be positive")
    if len(set(ranked_drafter_items)) != len(ranked_drafter_items):
        raise ValueError("Drafter ranking contains duplicate items")
    unknown = set(ranked_drafter_items) - set(catalog.paths)
    if unknown:
        raise ValueError(f"Drafter ranking contains unknown items: {len(unknown)}")
    if prefix_depth < 0 or prefix_depth > catalog.max_depth:
        raise ValueError("prefix_depth outside the catalog")
    prefixes = {tuple(prefix) for prefix in verifier_prefixes}
    if prefix_depth and any(len(prefix) != prefix_depth for prefix in prefixes):
        raise ValueError("Verifier prefix length mismatch")
    excluded = set(already_drafted)
    selected: list[str] = []
    for item in ranked_drafter_items:
        if item in excluded:
            continue
        if prefix_depth and catalog.paths[item][:prefix_depth] not in prefixes:
            continue
        selected.append(item)
        if len(selected) == draft_size:
            break
    return selected


@dataclass(frozen=True)
class VerifiedCandidate:
    item_id: str
    score: float
    accepted: bool


def finalize_recommendations(
    *,
    verified: Sequence[VerifiedCandidate],
    beam_fallback: Sequence[tuple[str, float]],
    catalog: PathCatalog,
    k: int,
) -> list[tuple[str, float, str]]:
    """Rank accepted drafts by verifier score, then fill from GRAM beam."""

    if k < 1:
        raise ValueError("k must be positive")
    verified_ids = [row.item_id for row in verified]
    if len(set(verified_ids)) != len(verified_ids):
        raise ValueError("A candidate was verified more than once")
    beam_ids = [item for item, _score in beam_fallback]
    if len(set(beam_ids)) != len(beam_ids):
        raise ValueError("Beam fallback contains duplicate items")
    unknown = (set(verified_ids) | set(beam_ids)) - set(catalog.paths)
    if unknown:
        raise ValueError(f"Unknown recommendation items: {len(unknown)}")
    accepted = sorted(
        (row for row in verified if row.accepted),
        key=lambda row: (-float(row.score), row.item_id),
    )
    output: list[tuple[str, float, str]] = []
    used: set[str] = set()
    for row in accepted:
        if not math.isfinite(float(row.score)):
            raise ValueError("Accepted verifier score must be finite")
        output.append((row.item_id, float(row.score), "accepted_draft"))
        used.add(row.item_id)
        if len(output) == k:
            return output
    for item, score in sorted(beam_fallback, key=lambda row: (-float(row[1]), row[0])):
        if item in used:
            continue
        if not math.isfinite(float(score)):
            raise ValueError("Beam score must be finite")
        output.append((item, float(score), "gram_beam_fallback"))
        used.add(item)
        if len(output) == k:
            return output
    raise ValueError(f"Only {len(output)} unique catalog items available for top-{k}")


def validate_specgr_budget_trace(
    *,
    drafted_by_round: Sequence[Sequence[str]],
    draft_size: int,
    max_path_depth: int,
    verifier_forward_candidates: int,
) -> dict[str, int]:
    if len(drafted_by_round) > max_path_depth:
        raise ValueError("SpecGR exceeded the maximum path-depth round budget")
    flattened = [item for batch in drafted_by_round for item in batch]
    if any(len(batch) > draft_size for batch in drafted_by_round):
        raise ValueError("A SpecGR round exceeded draft_size")
    if len(flattened) != len(set(flattened)):
        raise ValueError("SpecGR re-verified a drafted item")
    if verifier_forward_candidates != len(flattened):
        raise ValueError("Verifier forward accounting does not match drafted candidates")
    return {
        "rounds": len(drafted_by_round),
        "drafted_candidates": len(flattened),
        "verifier_forward_candidates": verifier_forward_candidates,
    }
