"""Freeze the CE expert mixture and learn only a user guidance strength.

Uses the existing LIME-derived expert inputs and AGRec-inspired decoder rule.
The ranking objective compares completed paths; it does not simulate beam
pruning and is still a surrogate for fresh generation NDCG.
"""
import numpy as np
import torch
from experiment.phase21 import joint_fusion_core as J


class AlphaOnly(J.JointFusion):
    def __init__(self, parent):
        super().__init__('user', parent)
        self.input.requires_grad_(False)
        self.beta.requires_grad_(False)


def sequence_scores(native, legal, q, alpha, valid):
    a = alpha[:, None, None]
    mixed = torch.logaddexp(native + torch.log1p(-a), legal + q + a.log())
    return (mixed * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def objective(scores, candidates, kind):
    if kind == 'nll':
        return -scores[:, 0].mean()
    assert kind == 'rank'
    losses = torch.nn.functional.softplus(scores[:, 1:] - scores[:, :1])
    mask = candidates[:, 1:]
    return ((losses * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def pool_logs(tree, scores, items):
    """Exact catalog-to-prefix aggregation, vectorized for several paths/user."""
    index = tree.index
    node = torch.as_tensor(index.node_items, device=scores.device, dtype=torch.long)
    item = torch.as_tensor(index.item_refs - 1, device=scores.device, dtype=torch.long)
    weights = (20 * (scores - scores.max(-1, keepdim=True).values)).exp()
    mass = torch.zeros(len(scores), len(index.nodes), device=scores.device, dtype=scores.dtype)
    mass.scatter_add_(1, node[None, :].expand(len(scores), -1), weights[:, item])
    ancestors = tree.ancestors[:, items].permute(1, 2, 0)
    values = mass.gather(1, ancestors.reshape(len(scores), -1)).reshape(ancestors.shape).log()
    q = values[..., 1:] - values[..., :-1]
    valid = tree.valid[items]
    return q.masked_fill(~valid, 0), valid


def tests():
    from experiment.phase21 import single_fusion_core as C
    torch.manual_seed(1701)
    parent = C.Fusion()
    torch.nn.init.normal_(parent.output.weight, std=.1)
    model = AlphaOnly(parent.state_dict())
    u = torch.randn(4, 10)
    beta, alpha = model(u)
    torch.testing.assert_close(beta, parent(u, None, 'user')[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(alpha, torch.full_like(alpha, .3))
    assert sum(p.numel() for p in model.parameters()) == 451
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
    paths = [[0,2,4,1],[0,2,5,1],[0,3,1],[0,3,7,8,1]]
    tree = J.PathMass(paths, 'cpu')
    scores = torch.rand(4, 4, dtype=torch.double)
    items = torch.tensor([[0,2,3],[1,0,3],[2,1,0],[3,2,1]])
    q, valid = pool_logs(tree, scores, items)
    for col in range(3):
        ref = tree.logs(scores, items[:, col])
        torch.testing.assert_close(q[:, col][valid[:, col]], ref[valid[:, col]], atol=1e-12, rtol=1e-12)
    native, legal = torch.full_like(q, -3.), torch.full_like(q, -.3)
    z = sequence_scores(native, legal, q, alpha.double(), valid)
    ref = J.path_nll(native[:, 0], legal[:, 0], q[:, 0], alpha.double(), valid[:, 0])
    torch.testing.assert_close(-z[:, 0], ref)
    zero = sequence_scores(native, legal, q, torch.zeros(4, dtype=torch.double), valid)
    torch.testing.assert_close(zero, torch.full_like(zero, -3.))
    objective(z, torch.ones(4, 3, dtype=torch.bool), 'rank').backward()
    assert model.strength.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('strength.'))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Increasing a competitor score must worsen the ranking loss.
    x = torch.tensor([[-1., -2., -3.]], requires_grad=True)
    objective(x, torch.tensor([[True, True, False]]), 'rank').backward()
    assert x.grad[0, 0] < 0 and x.grad[0, 1] > 0 and x.grad[0, 2] == 0
    return dict(frozen_beta_exact=True, initial_alpha_point3=True, total_parameters=451,
        adapted_parameters=33, variable_length_pool_logs_match_original=True,
        target_nll_matches_original=True, alpha_zero_exact=True, ranking_gradient_verified=True)
