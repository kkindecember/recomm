"""RaSeRec's two-channel reader, with an explicit GRAM decoder-memory bridge.

The author neural layers are preserved in raserec_author.py. Retrieval uses an
exact cosine search instead of Faiss IVF; all memories come from training only.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput

from .raserec_author import CrossMultiHeadAttention, FeedForward


@dataclass
class ReaderConfig:
    hidden_size: int = 64
    inner_size: int = 256
    heads: int = 2
    dropout: float = 0.5
    alpha: float = 0.5
    beta: float = 1.0
    topk: int = 20
    attn_tau: float = 1.0
    auxiliary_weight: float = 0.1


class RaSeRecReader(nn.Module):
    """Author seq_augmented equations, including the shared FFN in its code."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        for name in ('seq_tar_ram', 'seq_tar_ram_1', 'tar_seq_ram', 'tar_seq_ram_1'):
            setattr(self, name, CrossMultiHeadAttention(
                config.heads, config.hidden_size, config.dropout, config.dropout,
                1e-12, attn_tau=config.attn_tau))
        self.seq_tar_ram_fnn = FeedForward(config.hidden_size, config.inner_size,
                                          config.dropout, 'gelu', 1e-12)
        # Retained for compatibility: upstream declares this FFN but calls the
        # seq_tar FFN in both channels (raserec.py:379-384).
        self.tar_seq_ram_fnn = FeedForward(config.hidden_size, config.inner_size,
                                          config.dropout, 'gelu', 1e-12)

    def forward(self, query, histories, outcomes):
        b, d = query.shape
        if histories.shape != outcomes.shape or histories.shape != (b, self.config.topk, d):
            raise ValueError('Expected matched B x K x D history/outcome memories')
        # Upstream attention calls squeeze(); reshape preserves a singleton batch.
        h = self.seq_tar_ram(query[:, None], histories, outcomes).reshape(b, d)
        h = self.seq_tar_ram_fnn(h)
        h = self.seq_tar_ram_1(h[:, None], histories, outcomes).reshape(b, d)
        v = self.tar_seq_ram(query[:, None], outcomes, histories).reshape(b, d)
        v = self.seq_tar_ram_fnn(v)
        v = self.tar_seq_ram_1(v[:, None], outcomes, histories).reshape(b, d)
        cfg = self.config
        return cfg.alpha * query + (1 - cfg.alpha) * (cfg.beta * h + (1 - cfg.beta) * v)


@torch.no_grad()
def exact_case_neighbors(queries, keys, query_users, key_users, topk, chunk_size=256):
    """Cosine top-k; never admit a case owned by the current query user."""
    if len(keys) < topk or len(queries) != len(query_users) or len(keys) != len(key_users):
        raise ValueError('Invalid memory/search dimensions')
    keys = F.normalize(keys, dim=-1)
    results = []
    for start in range(0, len(queries), chunk_size):
        q = F.normalize(queries[start:start + chunk_size], dim=-1)
        scores = q @ keys.T
        same_user = query_users[start:start + chunk_size, None].eq(key_users[None])
        scores.masked_fill_(same_user, -torch.inf)
        values, indices = scores.topk(topk, dim=-1)
        if not torch.isfinite(values).all():
            raise ValueError('Not enough other-user cases for every query')
        results.append(indices.cpu())
    return torch.cat(results)


class RaSeRecGram(nn.Module):
    """Keep native GRAM context and append learned RaSeRec evidence tokens.

    The bridge exposes the original query, the complete RaSeRec output, and the
    retrieved history/outcome pairs. These tokens supplement native FiD memory;
    native lexical IDs, item passages, and the decoder remain intact.
    """
    def __init__(self, gram, reader, item_embeddings):
        super().__init__()
        self.gram, self.reader = gram, reader
        d = gram.config.d_model
        self.memory_projection = nn.Linear(reader.config.hidden_size, d)
        self.memory_norm = nn.LayerNorm(d)
        self.memory_type = nn.Embedding(4, d)
        self.memory_rank = nn.Embedding(reader.config.topk + 1, d)
        nn.init.zeros_(self.memory_type.weight)
        nn.init.zeros_(self.memory_rank.weight)
        self.register_buffer('item_embeddings', item_embeddings.detach().clone(), persistent=False)
        self.backbone_frozen = False
        self.memory_enabled = True
        self.last_loss_components = None

    def set_backbone_frozen(self, frozen):
        self.backbone_frozen = bool(frozen)
        for p in self.gram.parameters():
            p.requires_grad_(not frozen)
        self.gram.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.gram.eval()
        return self

    def encode(self, input_ids, attention_mask, history_item_ids,
               memory_query, memory_keys, memory_values):
        if input_ids.ndim != 3 or input_ids.shape != attention_mask.shape:
            raise ValueError('GRAM expects B x passages x tokens')
        valid = attention_mask[:, 1:].bool().any(-1)
        if history_item_ids.shape != valid.shape or not torch.equal(history_item_ids.ne(0), valid):
            raise ValueError('History IDs must align with native history passages')
        self.gram.encoder.n_passages = input_ids.size(1)
        self.gram.encoder.set_migration_context(None, None)
        native = self.gram.encoder(input_ids=input_ids.flatten(1),
                                   attention_mask=attention_mask.flatten(1), return_dict=True)[0]
        mask = attention_mask.flatten(1).bool()
        if not self.memory_enabled:
            return BaseModelOutput(last_hidden_state=native), mask, memory_query
        augmented = self.reader(memory_query, memory_keys, memory_values)
        evidence = torch.cat([memory_query[:, None], augmented[:, None], memory_keys, memory_values], 1)
        k = self.reader.config.topk
        types = torch.tensor([0, 1] + [2] * k + [3] * k, device=native.device)
        ranks = torch.tensor([0, 0] + list(range(1, k + 1)) * 2, device=native.device)
        evidence = self.memory_norm(self.memory_projection(evidence))
        evidence = evidence + self.memory_type(types)[None] + self.memory_rank(ranks)[None]
        mask = torch.cat([mask, torch.ones(evidence.shape[:2], dtype=torch.bool, device=mask.device)], 1)
        return BaseModelOutput(last_hidden_state=torch.cat([native, evidence], 1)), mask, augmented

    def forward(self, input_ids, attention_mask, history_item_ids,
                memory_query, memory_keys, memory_values, labels, target_item_ids):
        encoded, mask, augmented = self.encode(input_ids, attention_mask, history_item_ids,
                                                memory_query, memory_keys, memory_values)
        result = self.gram(encoder_outputs=encoded, attention_mask=mask, labels=labels,
                           use_cache=False, return_dict=True)
        generation = result.loss
        item = F.cross_entropy(augmented @ self.item_embeddings.T, target_item_ids)
        result.loss = generation + (self.reader.config.auxiliary_weight * item if self.memory_enabled else 0)
        self.last_loss_components = {'generation': float(generation.detach()), 'item': float(item.detach()),
                                     'total': float(result.loss.detach())}
        return result

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, history_item_ids,
                 memory_query, memory_keys, memory_values, **kwargs):
        from model.gram_t5 import T5ForConditionalGeneration_GRAM
        encoded, mask, _ = self.encode(input_ids, attention_mask, history_item_ids,
                                       memory_query, memory_keys, memory_values)
        return T5ForConditionalGeneration_GRAM.generate(
            self.gram, input_ids=input_ids.flatten(1), attention_mask=mask,
            encoder_outputs=encoded, **kwargs)
