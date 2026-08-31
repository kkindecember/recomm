"""Clean-room SETRec contracts derived from the paper and public interfaces.

No SETRec source code is copied: the repository has no standard open-source
license.  The contracts explicitly distinguish the paper's sparse history mask
from the public T5 repository's shared-within-item position-id implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def item_group_position_ids(
    *,
    n_items: int,
    n_tokens_per_item: int,
    prefix_tokens: int = 0,
    suffix_tokens: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Public-repository parity: tokens in one item share a position id."""

    if n_items <= 0 or n_tokens_per_item <= 0:
        raise ValueError("item/token counts must be positive")
    if prefix_tokens < 0 or suffix_tokens < 0:
        raise ValueError("prefix/suffix counts cannot be negative")
    prefix = torch.arange(prefix_tokens, dtype=torch.long, device=device)
    middle = torch.arange(n_items, dtype=torch.long, device=device).repeat_interleave(
        n_tokens_per_item
    ) + prefix_tokens
    suffix = torch.arange(
        prefix_tokens + n_items,
        prefix_tokens + n_items + suffix_tokens,
        dtype=torch.long,
        device=device,
    )
    return torch.cat((prefix, middle, suffix))


def paper_sparse_history_mask(
    *,
    n_items: int,
    n_tokens_per_item: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Paper-faithful visibility for a flattened set-identifier history.

    A token sees itself and every token belonging to prior items, but it cannot
    see another information dimension in its own item identifier or a future
    item.  ``True`` means visible.
    """

    if n_items <= 0 or n_tokens_per_item <= 0:
        raise ValueError("item/token counts must be positive")
    total = n_items * n_tokens_per_item
    positions = torch.arange(total, device=device)
    query_item = torch.div(positions[:, None], n_tokens_per_item, rounding_mode="floor")
    key_item = torch.div(positions[None, :], n_tokens_per_item, rounding_mode="floor")
    self_visibility = positions[:, None] == positions[None, :]
    return (key_item < query_item) | self_visibility


def independent_query_mask(
    n_query: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Each generated dimension sees itself, not the other query vectors."""

    if n_query <= 0:
        raise ValueError("n_query must be positive")
    return torch.eye(n_query, dtype=torch.bool, device=device)


class SemanticSetAutoencoder(nn.Module):
    """One AE that emits N continuous semantic tokens and reconstructs input."""

    def __init__(
        self,
        *,
        semantic_dim: int,
        model_dim: int,
        n_semantic_tokens: int = 4,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if semantic_dim <= 0 or model_dim <= 0 or n_semantic_tokens <= 0:
            raise ValueError("semantic/model/token dimensions must be positive")
        bottleneck = model_dim * n_semantic_tokens
        encoder_dims = [semantic_dim, *map(int, hidden_dims), bottleneck]
        if any(dim <= 0 for dim in encoder_dims):
            raise ValueError("AE hidden dimensions must be positive")
        self.semantic_dim = semantic_dim
        self.model_dim = model_dim
        self.n_semantic_tokens = n_semantic_tokens
        self.encoder = self._mlp(encoder_dims, dropout=dropout, final_activation=False)
        self.decoder = self._mlp(
            list(reversed(encoder_dims)), dropout=dropout, final_activation=False
        )

    @staticmethod
    def _mlp(
        dimensions: Sequence[int], *, dropout: float, final_activation: bool
    ) -> nn.Sequential:
        modules: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            modules.append(nn.Linear(in_features, out_features))
            is_final = index == len(dimensions) - 2
            if not is_final or final_activation:
                modules.append(nn.ReLU())
                if dropout:
                    modules.append(nn.Dropout(dropout))
        return nn.Sequential(*modules)

    def forward(self, semantic_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic_features.shape[-1] != self.semantic_dim:
            raise ValueError("semantic feature dimension mismatch")
        encoded = self.encoder(semantic_features)
        reconstruction = self.decoder(encoded)
        tokens = encoded.view(
            *encoded.shape[:-1], self.n_semantic_tokens, self.model_dim
        )
        return tokens, reconstruction


@dataclass(frozen=True)
class SetRecGroundingOutput:
    per_dimension_scores: torch.Tensor
    item_scores: torch.Tensor


def ground_continuous_queries(
    query_outputs: torch.Tensor,
    token_corpus: torch.Tensor,
    *,
    beta: float,
) -> SetRecGroundingOutput:
    """Ground every query against its full-catalog, per-dimension corpus."""

    if query_outputs.ndim != 3 or token_corpus.ndim != 3:
        raise ValueError("expected query [B,Q,D] and corpus [Q,I,D]")
    if query_outputs.shape[1] != token_corpus.shape[0]:
        raise ValueError("query dimension and token corpus dimension differ")
    if query_outputs.shape[2] != token_corpus.shape[2]:
        raise ValueError("query and corpus embedding dimensions differ")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    per_dimension = torch.einsum("bqd,qid->qbi", query_outputs, token_corpus)
    weights = torch.full(
        (token_corpus.shape[0],),
        float(beta),
        dtype=per_dimension.dtype,
        device=per_dimension.device,
    )
    weights[0] = 1.0 - float(beta)
    item_scores = (per_dimension * weights[:, None, None]).sum(dim=0)
    return SetRecGroundingOutput(
        per_dimension_scores=per_dimension, item_scores=item_scores
    )


@dataclass(frozen=True)
class SetRecLossOutput:
    loss: torch.Tensor
    generation_loss: torch.Tensor
    reconstruction_loss: torch.Tensor


def setrec_joint_loss(
    per_dimension_scores: torch.Tensor,
    target_item_indices: torch.Tensor,
    *,
    semantic_features: torch.Tensor,
    semantic_reconstruction: torch.Tensor,
    alpha: float = 0.7,
) -> SetRecLossOutput:
    """Dimension-wise full-catalog CE plus the unified AE reconstruction loss."""

    if per_dimension_scores.ndim != 3:
        raise ValueError("per_dimension_scores must be [Q,B,I]")
    n_query, batch_size, catalog_size = per_dimension_scores.shape
    labels = target_item_indices.reshape(-1)
    if labels.shape[0] != batch_size:
        raise ValueError("target batch size mismatch")
    if labels.numel() and (labels.min() < 0 or labels.max() >= catalog_size):
        raise ValueError("target item index is outside the catalog")
    repeated_labels = labels.repeat(n_query)
    generation = F.cross_entropy(
        per_dimension_scores.reshape(n_query * batch_size, catalog_size),
        repeated_labels,
    )
    reconstruction = F.mse_loss(semantic_reconstruction, semantic_features)
    total = generation + float(alpha) * reconstruction
    return SetRecLossOutput(
        loss=total,
        generation_loss=generation,
        reconstruction_loss=reconstruction,
    )


def full_set_recovery(
    per_dimension_scores: torch.Tensor, target_item_indices: torch.Tensor
) -> torch.Tensor:
    """Return whether every information dimension grounds to the target item."""

    if per_dimension_scores.ndim != 3:
        raise ValueError("per_dimension_scores must be [Q,B,I]")
    predictions = per_dimension_scores.argmax(dim=-1)
    targets = target_item_indices.reshape(1, -1).expand_as(predictions)
    return (predictions == targets).all(dim=0)
