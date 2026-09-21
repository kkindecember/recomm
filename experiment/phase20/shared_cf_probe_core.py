"""Frozen GRAM representation -> mean calibrated CF contribution.

The branch has no access to CF representations or target labels at inference.
Its output is centered, not standardized: zero initialization must remain zero.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SharedCFHead(nn.Module):
    def __init__(self, dimension, projection=128, hidden=64):
        super().__init__()
        self.user = nn.Linear(dimension, projection)
        self.item = nn.Linear(dimension, projection)
        self.scorer = nn.Sequential(nn.Linear(4 * projection + 6, hidden),
                                    nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, users, items, features):
        # Non-affine normalization belongs only to this new branch.
        u = torch.tanh(self.user(F.layer_norm(users, (users.shape[-1],))))[:, None]
        v = torch.tanh(self.item(F.layer_norm(items, (items.shape[-1],))))
        u = u.expand_as(v)
        score = self.scorer(torch.cat((u, v, u * v, (u - v).abs(), features), -1)).squeeze(-1)
        return score - score.mean(-1, keepdim=True)


def fuse(base, component, reliability):
    return base + reliability[:, None] * component


def teaching_loss(prediction, target):
    return (prediction.double() - target.double()).square().mean(-1)


def next_item_loss(score, positions):
    # Dense target reduction also supports deterministic backward on torch 1.11.
    mask = torch.arange(score.shape[1], device=score.device)[None] == positions[:, None]
    return torch.logsumexp(score, -1) - (score * mask).sum(-1)


def agreement(student, teacher, base):
    order = lambda a: np.argsort(-a, axis=1, kind='stable')
    so, to, bo = order(student), order(teacher), order(base)
    inverse = np.argsort(so, axis=1, kind='stable')
    overlaps, relations = [], []
    for j in range(len(student)):
        ss, tt, bb = set(so[j, :10]), set(to[j, :10]), set(bo[j, :10])
        overlaps.append(len(ss & tt) / 10)
        up, down = list(tt - bb), list(bb - tt)
        if up and down:
            relations.append(float((inverse[j, up][:, None] < inverse[j, down][None]).mean()))
    return {'teacher_top10_overlap': float(np.mean(overlaps)),
            'changed_queries': len(relations),
            'up_down_satisfaction': float(np.mean(relations)) if relations else None}


def checks():
    torch.manual_seed(31)
    model = SharedCFHead(8, 4, 5).double()
    u, v, x = torch.randn(7, 8).double(), torch.randn(7, 50, 8).double(), torch.randn(7, 50, 6).double()
    b, r = torch.randn(7, 50).double(), torch.rand(7).double()
    assert torch.equal(fuse(b, model(u, v, x), r), b)
    with torch.no_grad(): model.scorer[-1].weight.normal_()
    out = model(u, v, x)
    perm = torch.randperm(50)
    torch.testing.assert_close(model(u, v[:, perm], x[:, perm]), out[:, perm], atol=1e-12, rtol=0)
    torch.testing.assert_close(out.mean(-1), torch.zeros(7).double(), atol=1e-12, rtol=0)
    assert torch.equal(fuse(b, out, torch.zeros_like(r)), b)
    pos = torch.tensor([0, 3, 9, 15, 22, 35, 49])
    torch.testing.assert_close(next_item_loss(b, pos), F.cross_entropy(b, pos, reduction='none'))
    # Objective/gradient parity for uneven minibatch accumulation.
    target = torch.randn(7, 50).double()
    gradients = []
    for chunk in (7, 3):
        model.zero_grad(set_to_none=True)
        for start in range(0, 7, chunk):
            sl = slice(start, start + chunk)
            (teaching_loss(model(u[sl], v[sl], x[sl]), target[sl]).sum() / 7).backward()
        gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(*gradients, atol=1e-12, rtol=1e-10)
    toy = np.tile(np.arange(50, dtype=float), (2, 1))
    identity = agreement(toy, toy, -toy)
    assert identity['teacher_top10_overlap'] == identity['up_down_satisfaction'] == 1
    return dict(zero_branch_exact_identity=True, candidate_permutation_equivariance=True,
                reliability_zero_identity=True, centered_output=True, dense_ce_parity=True,
                uneven_chunk_gradient_parity=True, rank_agreement_positive_control=True)
