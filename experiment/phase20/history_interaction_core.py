"""Frozen-item, candidate-conditioned ranking; feature API has no target input."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def align_text(catalog_items, item_ids, embeddings):
    if len(item_ids) != len(set(item_ids)):
        raise ValueError('Duplicate embedding item IDs')
    if catalog_items[0] != '' or len(catalog_items) != len(set(catalog_items)):
        raise ValueError('Invalid catalog/padding')
    if set(catalog_items[1:]) != set(item_ids):
        raise ValueError('Embedding/catalog item mismatch')
    lookup = {item: i for i, item in enumerate(item_ids)}
    out = torch.zeros((len(catalog_items), embeddings.shape[1]), dtype=torch.float32)
    out[1:] = embeddings[[lookup[item] for item in catalog_items[1:]]]
    if not torch.isfinite(out).all():
        raise ValueError('Nonfinite text embeddings')
    return out


def history_inputs(histories, frequencies, max_history=20):
    """Position means recency within visible history; never use a sample key."""
    ids = np.zeros((len(histories), max_history), dtype=np.int64)
    attributes = np.zeros((len(histories), max_history, 2), dtype=np.float32)
    divisor = max(float(np.log1p(frequencies).max()), 1.)
    for row, h in enumerate(histories):
        if not 0 < len(h) <= max_history or min(h) <= 0 or max(h) >= len(frequencies):
            raise ValueError('Invalid visible history')
        ids[row, :len(h)] = h
        attributes[row, :len(h), 0] = np.arange(len(h)-1, -1, -1) / max_history
        attributes[row, :len(h), 1] = np.log1p(frequencies[h]) / divisor
    return ids, attributes


def masked_attention(logits, mask):
    if not mask.any(-1).all():
        raise ValueError('Empty history')
    return torch.softmax(logits.masked_fill(~mask[:, None, :], -torch.inf), dim=-1)


class HistoryRanker(nn.Module):
    """Ordinary independent attention vs extra same-event joint attention.

    Item tables are external frozen assets, not nn.Parameters. The ordinary
    arm has a slightly larger post-pooling MLP than the joint attention MLP.
    Both arms have the same shared modules and residual-head dimensions.
    """
    def __init__(self, arm, text_dim=1024, cf_dim=512, dim=32, bound=.5):
        super().__init__()
        if arm not in ('B_scores', 'C_attention', 'D_joint'):
            raise ValueError(arm)
        self.arm, self.dim, self.bound = arm, dim, bound
        if arm == 'B_scores':
            self.head = nn.Sequential(nn.Linear(6, 384), nn.GELU(),
                                      nn.Linear(384, 368), nn.GELU(),
                                      nn.Linear(368, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            self.sq = nn.Linear(text_dim, dim, bias=False)
            self.sk = nn.Linear(text_dim, dim, bias=False)
            self.sv = nn.Linear(text_dim, dim, bias=False)
            self.cq = nn.Linear(cf_dim, dim, bias=False)
            self.ck = nn.Linear(cf_dim, dim, bias=False)
            self.cv = nn.Linear(cf_dim, dim, bias=False)
            self.position = nn.Linear(2, 2)
            # Build the common head before the arm-specific module for equal init.
            self.head = nn.Sequential(nn.Linear(6 + 6 * dim, 64), nn.GELU(),
                                      nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
            if arm == 'C_attention':
                self.pooled_control = nn.Sequential(nn.Linear(2*dim, 32), nn.GELU(),
                                                     nn.Linear(32, 2*dim))
            else:
                self.joint = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, candidates, histories, attributes, text_table, cf_table,
                return_details=False):
        if self.arm == 'B_scores':
            return self.bound * torch.tanh(self.head(features).squeeze(-1))
        mask = histories.ne(0)
        ts, tc = text_table[candidates], cf_table[candidates]
        hs, hc = text_table[histories], cf_table[histories]
        qs, qc = self.sq(ts), self.cq(tc)
        ks, kc, vs, vc = self.sk(hs), self.ck(hc), self.sv(hs), self.cv(hc)
        # Fixed normalization of inputs is done once in the prepared asset table.
        bias = self.position(attributes)
        ls = torch.matmul(qs, ks.transpose(1, 2)) / math.sqrt(self.dim) + bias[:, None, :, 0]
        lc = torch.matmul(qc, kc.transpose(1, 2)) / math.sqrt(self.dim) + bias[:, None, :, 1]
        ws, wc = masked_attention(ls, mask), masked_attention(lc, mask)
        zs, zc = torch.matmul(ws, vs), torch.matmul(wc, vc)
        separate = torch.cat([zs, zc], dim=-1)
        details = {'semantic_weights': ws, 'cf_weights': wc, 'separate': separate}
        if self.arm == 'C_attention':
            extra = self.pooled_control(separate)
        else:
            attr = attributes[:, None].expand(-1, candidates.shape[1], -1, -1)
            relations = torch.cat([torch.stack([ls, lc, ls-lc, ls*lc], -1), attr], -1)
            wj = masked_attention(self.joint(relations).squeeze(-1), mask)
            extra = torch.cat([torch.matmul(wj, vs), torch.matmul(wj, vc)], -1)
            details['joint_weights'] = wj
        joined = torch.cat([features, qs, qc, separate, extra], -1)
        out = self.bound * torch.tanh(self.head(joined).squeeze(-1))
        return (out, details) if return_details else out


def build_model(config, arm):
    torch.manual_seed(config['seed'])
    return HistoryRanker(arm, config['text_dim'], config['cf_dim'],
                         config['projection_dim'], config['correction_bound'])


def global_group_weights(head):
    head = np.asarray(head, dtype=bool)
    if not head.any() or head.all():
        raise ValueError('Both training frequency groups are required')
    return np.where(head, len(head)/(2*head.sum()), len(head)/(2*(~head).sum()))


def dense_pairwise_losses(scores, positions):
    """Same all49 objective without PyTorch1.11 CUDA gather backward scatter."""
    assert scores.shape[1] == 50 and torch.all((positions >= 0) & (positions < 50))
    target_mask = torch.arange(50, device=scores.device)[None] == positions[:, None]
    positive = scores.masked_fill(~target_mask, 0).sum(-1, keepdim=True)
    losses = F.softplus(scores-positive).masked_fill(target_mask, 0)
    return losses.sum(-1)/49
