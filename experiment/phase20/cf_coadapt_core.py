"""Low-rank adaptation inside the final CF Transformer feed-forward block."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))


def install(model, rank):
    model.eval().requires_grad_(False)
    model.transformer.layers[-1].linear2 = ResidualLinear(model.transformer.layers[-1].linear2, rank)
    return model


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def adapter_state(model):
    layer = model.transformer.layers[-1].linear2
    return {k: getattr(layer, k).weight.detach().cpu().clone() for k in ('down', 'up')}


def restore(model, state):
    layer = model.transformer.layers[-1].linear2
    with torch.no_grad():
        for k, value in state.items(): getattr(layer, k).weight.copy_(value)


def standardize(x):
    x = x.double()
    centered = x - x.mean(-1, keepdim=True)
    return centered / centered.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def component(cf_scores, pop_z, reliability):
    # cf_scores: seed x query x candidate. Same deployment calibration as baseline.
    adjusted = standardize(standardize(cf_scores) - .5 * pop_z[None])
    return reliability[:, None] * adjusted.mean(0)


def dense_ce(score, positions):
    mask = torch.arange(score.shape[-1], device=score.device)[None] == positions[:, None]
    return torch.logsumexp(score, -1) - (score * mask).sum(-1)


def objective(full_logits, cf_scores, base, pop_z, reliability, positions, targets, arm):
    full = torch.stack([F.cross_entropy(x, targets - 1, reduction='none') for x in full_logits]).mean(0).double()
    if arm == 'full_catalog': return full
    c = component(cf_scores, pop_z, reliability)
    if arm == 'coadapt': c = c + base
    elif arm != 'candidate_cf': raise ValueError(arm)
    candidate = dense_ce(c, positions.clamp_min(0))
    return torch.where(positions >= 0, candidate, full)


def checks():
    from experiment.phase20.future_fusion_alignment import CF0B2ItemHead
    torch.manual_seed(910)
    model = CF0B2ItemHead(80, 6, 16, 2, 4, 0., .07).eval()
    history = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
    before = model.score(history, history.ne(0)).detach()
    install(model, 4)
    assert torch.equal(before, model.score(history, history.ne(0)))
    frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    parameters = trainable(model); assert len(parameters) == 2
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    F.cross_entropy(model.score(history, history.ne(0)), torch.tensor([4, 7, 20])).backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert model.transformer.layers[-1].linear2.up.weight.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(before, model.score(history, history.ne(0)))
    assert all(torch.equal(p, frozen[k]) for k, p in model.named_parameters() if not p.requires_grad)
    rng = np.random.RandomState(5)
    cf = torch.tensor(rng.normal(size=(3, 4, 50)), requires_grad=True)
    pop = standardize(torch.tensor(rng.normal(size=(4, 50))))
    reliability = torch.tensor([0., .3, .7, 1.]); base = torch.tensor(rng.normal(size=(4, 50)))
    positions = torch.tensor([0, 3, 9, 49])
    z = lambda a: (a - a.mean(-1, keepdims=True)) / np.maximum(a.std(-1, keepdims=True), 1e-6)
    manual = reliability.numpy()[:, None] * z(z(cf.detach().numpy()) - .5 * pop.numpy()[None]).mean(0)
    np.testing.assert_allclose(component(cf, pop, reliability).detach(), manual, atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda a: dense_ce(base + component(a, pop, reliability), positions).sum(), (cf,))
    final = base + component(cf, pop, reliability)
    torch.testing.assert_close(dense_ce(final, positions), F.cross_entropy(final, positions, reduction='none'))
    perm = torch.randperm(50)
    torch.testing.assert_close(component(cf[:, :, perm], pop[:, perm], reliability), component(cf, pop, reliability)[:, perm])
    # The matched candidate control and coadaptation differ only by frozen base.
    g0 = torch.autograd.grad(dense_ce(component(cf, pop, reliability), positions).sum(), cf, retain_graph=True)[0]
    g1 = torch.autograd.grad(dense_ce(final, positions).sum(), cf)[0]
    assert not torch.allclose(g0, g1) and torch.equal(g1[:, 0], torch.zeros_like(g1[:, 0]))
    outside = torch.tensor([-1, 3, -1, 49]); labels = torch.tensor([1, 20, 40, 60])
    full = [torch.randn(4, 80, dtype=torch.float64) for _ in range(3)]
    losses = [objective(full, cf, base, pop, reliability, outside, labels, arm) for arm in ('full_catalog', 'candidate_cf', 'coadapt')]
    for loss in losses[1:]: assert torch.equal(loss[outside < 0], losses[0][outside < 0])
    return dict(zero_adapter_exact_identity=True, only_adapter_receives_gradient=True,
                optimizer_changes_output=True, pretrained_parameters_unchanged=True,
                deployment_numpy_parity=True, fused_loss_finite_difference=True,
                dense_ce_parity=True, candidate_permutation=True,
                generator_changes_cf_gradient=True, reliability_zero_blocks_candidate_gradient=True,
                uncovered_targets_use_identical_full_catalog_loss=True)
