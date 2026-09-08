"""DIFF sequence mechanisms adapted to fixed-width GRAM histories.

Reference: Kim et al., SIGIR 2025, arXiv:2505.13974; author implementation
HyeYoung1218/DIFF at ef6283a3bf9ed242a9b867fcf63368f0fb36eb33.
This independent implementation preserves frequency filtering, early and
intermediate fusion, shared feed-forward layers, and bidirectional alignment.
Explicit padding masks and the GRAM memory interface are task adaptations.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffConfig:
    hidden_size: int = 256
    inner_size: int = 256
    heads: int = 2
    layers: int = 2
    history_length: int = 20
    dropout: float = 0.5
    attention_dropout: Optional[float] = None
    cutoff: int = 9
    alpha: float = 0.5
    alignment_weight: float = 100.0
    auxiliary_weight: float = 0.1


def valid_only(value, valid):
    return value * valid.unsqueeze(-1).to(value.dtype)


class FrequencyFilter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bins = config.cutoff // 2 + 1
        self.beta = nn.Parameter(torch.randn(()))
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, value, valid):
        value = valid_only(value, valid)
        spectrum = torch.fft.rfft(value.float(), dim=1, norm="ortho")
        keep = torch.arange(spectrum.size(1), device=value.device) < self.bins
        low = torch.fft.irfft(spectrum * keep[None, :, None], n=value.size(1),
                             dim=1, norm="ortho").to(value.dtype)
        filtered = low + self.beta.square() * (value - low)
        return valid_only(self.norm(value + self.dropout(filtered)), valid)


class SequenceAttention(nn.Module):
    """Separate Q/K per stream; intermediate fusion aggregates item values."""

    def __init__(self, config, streams=1):
        super().__init__()
        self.heads = config.heads
        self.width = config.hidden_size // config.heads
        if config.hidden_size % config.heads:
            raise ValueError("Hidden size must be divisible by heads")
        self.queries = nn.ModuleList(nn.Linear(config.hidden_size, config.hidden_size)
                                     for _ in range(streams))
        self.keys = nn.ModuleList(nn.Linear(config.hidden_size, config.hidden_size)
                                  for _ in range(streams))
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.score_fusion = (nn.Linear(config.history_length * streams, config.history_length)
                             if streams > 1 else None)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_dropout = nn.Dropout(config.dropout if config.attention_dropout is None else config.attention_dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def split_heads(self, value):
        return value.reshape(value.size(0), value.size(1), self.heads, self.width).transpose(1, 2)

    def forward(self, streams, item_values, valid):
        scores = [self.split_heads(q(x)) @ self.split_heads(k(x)).transpose(-1, -2)
                  for x, q, k in zip(streams, self.queries, self.keys)]
        scores = self.score_fusion(torch.cat(scores, dim=-1)) if self.score_fusion else scores[0]
        length = valid.size(1)
        causal = torch.ones(length, length, device=valid.device, dtype=torch.bool).tril()
        allowed = valid[:, None, None, :] & causal[None, None, :, :]
        scores = (scores / math.sqrt(self.width)).masked_fill(~allowed, -1e4)
        probabilities = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = probabilities @ self.split_heads(self.value(item_values))
        attended = attended.transpose(1, 2).contiguous().reshape_as(item_values)
        return valid_only(self.norm(item_values + self.dropout(self.output(attended))), valid)


class SharedFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.inner_size)
        self.down = nn.Linear(config.inner_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, value, valid):
        result = self.down(F.gelu(self.up(value)))
        return valid_only(self.norm(value + self.dropout(result)), valid)


class DualFusionLayer(nn.Module):
    def __init__(self, config, attributes):
        super().__init__()
        self.filters = nn.ModuleList(FrequencyFilter(config) for _ in range(attributes + 2))
        self.intermediate = SequenceAttention(config, streams=attributes + 2)
        self.early = SequenceAttention(config)
        self.feed_forward = SharedFeedForward(config)

    def forward(self, item, attributes, fused, position, valid):
        filtered_fused = self.filters[0](fused, valid)
        filtered_item = self.filters[1](item, valid)
        filtered_attributes = [layer(value, valid)
                               for layer, value in zip(self.filters[2:], attributes)]
        # Author concat ordering: attributes, item, position.
        item = self.intermediate(filtered_attributes + [filtered_item, position],
                                 filtered_item, valid)
        fused = self.early([filtered_fused], filtered_fused, valid)
        return self.feed_forward(item, valid), self.feed_forward(fused, valid)


def alignment_loss(item, attributes, valid, temperature=0.1):
    """Original symmetric item/attribute distribution alignment, pad-safe."""
    item = F.normalize(item.float(), dim=-1)
    attribute = F.normalize(torch.stack(attributes).sum(0).float(), dim=-1)
    left = item @ attribute.transpose(-1, -2) / temperature
    right = attribute @ item.transpose(-1, -2) / temperature
    left = F.log_softmax(left.masked_fill(~valid[:, None, :], -1e4), dim=-1)
    right = F.log_softmax(right.masked_fill(~valid[:, None, :], -1e4), dim=-1)
    positive = ((attribute @ attribute.transpose(-1, -2) - 1).abs() < 1e-6)
    pairs = positive & valid[:, :, None] & valid[:, None, :]
    lengths = valid.sum(-1).clamp_min(1)
    return (-((left + right) * pairs).sum(-1) / lengths[:, None]).mean()


class DiffSequence(nn.Module):
    def __init__(self, num_items, attribute_tables, attribute_sizes, config=DiffConfig()):
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(num_items + 1, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config.history_length, config.hidden_size)
        self.attribute_embeddings = nn.ModuleList(
            nn.Embedding(size, config.hidden_size, padding_idx=0) for size in attribute_sizes)
        if len(attribute_tables) != len(attribute_sizes) or not attribute_tables:
            raise ValueError("Attribute tables/sizes must match and be nonempty")
        for index, (table, size) in enumerate(zip(attribute_tables, attribute_sizes)):
            table = torch.as_tensor(table, dtype=torch.long)
            if table.ndim != 2 or table.size(0) != num_items + 1 or table.min() < 0 or table.max() >= size:
                raise ValueError("Invalid item-to-attribute table")
            if table[0].any():
                raise ValueError("Padding item must have padding attributes")
            self.register_buffer(f"attribute_table_{index}", table)
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.input_dropout = nn.Dropout(config.dropout)
        self.early_fusion = nn.Linear(config.hidden_size * (len(attribute_sizes) + 2), config.hidden_size)
        self.layers = nn.ModuleList(DualFusionLayer(config, len(attribute_sizes)) for _ in range(config.layers))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def chronological_ids(self, recent_first_ids):
        if recent_first_ids.ndim != 2 or recent_first_ids.size(1) > self.config.history_length:
            raise ValueError("History must be a recent-first, right-padded matrix")
        if recent_first_ids.size(1) == 0:
            raise ValueError("Empty history is not supported")
        valid = recent_first_ids.ne(0)
        lengths = valid.sum(1)
        expected = torch.arange(valid.size(1), device=valid.device)[None] < lengths[:, None]
        if not torch.equal(valid, expected) or (lengths == 0).any():
            raise ValueError("History needs a nonempty contiguous valid prefix")
        positions = torch.arange(valid.size(1), device=valid.device)[None].expand_as(recent_first_ids)
        indices = torch.where(valid, lengths[:, None] - 1 - positions, positions)
        ordered = recent_first_ids.gather(1, indices)
        return F.pad(ordered, (0, self.config.history_length - ordered.size(1)))

    def forward(self, recent_first_ids, compute_alignment=True):
        ids = self.chronological_ids(recent_first_ids)
        valid = ids.ne(0)
        raw_item = self.item_embedding(ids)
        item = valid_only(self.input_dropout(self.input_norm(raw_item)), valid)
        positions = torch.arange(ids.size(1), device=ids.device)
        position = valid_only(self.position_embedding(positions)[None].expand_as(item), valid)
        attributes = [valid_only(embedding(getattr(self, f"attribute_table_{index}")[ids]).sum(-2), valid)
                      for index, embedding in enumerate(self.attribute_embeddings)]
        fused = valid_only(self.early_fusion(torch.cat([item] + attributes + [position], dim=-1)), valid)
        for layer in self.layers:
            item, fused = layer(item, attributes, fused, position, valid)
        memory = self.config.alpha * item + (1 - self.config.alpha) * fused
        user = memory[torch.arange(ids.size(0), device=ids.device), valid.sum(1) - 1]
        align = alignment_loss(raw_item, attributes, valid) if compute_alignment else None
        return memory, valid, user, align

    def item_loss(self, user, targets):
        if (targets <= 0).any() or (targets >= self.item_embedding.num_embeddings).any():
            raise ValueError("Target outside real item catalog")
        logits = F.linear(user, self.item_embedding.weight)
        logits = logits.masked_fill(torch.arange(logits.size(1), device=logits.device)[None] == 0, -1e4)
        return F.cross_entropy(logits, targets)
