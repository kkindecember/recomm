"""Compact user fusion and differentiable existing probability guidance.

Expert scoring derives from the project's LIME-Rec-based pipeline. The decoder
mixture is the existing AGRec-inspired rule, preserving native legal mass.
This module learns its coefficients; it does not claim new expert architectures.
"""
import math
import numpy as np
import torch
from experiment.phase20.lime_decode_screen_core import ExpertTrie


class JointFusion(torch.nn.Module):
    def __init__(self, mode, parent):
        super().__init__()
        assert mode in ('calibrate', 'global', 'user')
        self.mode = mode
        self.input = torch.nn.Linear(10, 32)
        self.beta = torch.nn.Linear(32, 2)
        with torch.no_grad():
            self.input.weight.copy_(parent['input.weight'][:, :10])
            self.input.bias.copy_(parent['input.bias'])
            self.beta.weight.copy_(parent['output.weight'])
            self.beta.bias.copy_(parent['output.bias'])
        if mode == 'user':
            self.strength = torch.nn.Linear(32, 1)
            torch.nn.init.zeros_(self.strength.weight)
            torch.nn.init.constant_(self.strength.bias, math.log(.3/.7))
        else:
            self.strength = torch.nn.Parameter(torch.tensor(math.log(.3/.7)))
        if mode == 'calibrate':
            self.input.requires_grad_(False)
            self.beta.requires_grad_(False)

    def forward(self, user):
        h = torch.relu(self.input(user))
        beta = self.beta(h).softmax(-1)
        logit = self.strength(h).squeeze(-1) if self.mode == 'user' else self.strength.expand(len(user))
        # Numerical endpoint protection, not a tuned strength range.
        alpha = logit.sigmoid().clamp(1e-6, 1.-1e-6)
        return beta, alpha


class PathMass:
    def __init__(self, paths, device):
        self.index = ExpertTrie(paths, range(1, len(paths)+1))
        depth = max(map(len, paths)) - 1
        ancestors = np.empty((depth+1, len(paths)), dtype=np.int64)
        valid = np.zeros((len(paths), depth), dtype=bool)
        for item, path in enumerate(paths):
            for d in range(depth+1):
                ancestors[d, item] = self.index.lookup[tuple(path[:min(d+1, len(path))])]
            valid[item, :len(path)-1] = True
        self.ancestors = torch.from_numpy(ancestors).to(device)
        self.valid = torch.from_numpy(valid).to(device)

    def logs(self, scores, targets, scale=20.):
        z = scale*scores
        previous = torch.logsumexp(z, -1)
        logs = []
        for d in range(1, len(self.ancestors)):
            nodes = self.ancestors[d]
            mask = nodes[None, :] == nodes[targets, None]
            current = torch.logsumexp(z.masked_fill(~mask, -torch.inf), -1)
            logs.append(current-previous)
            previous = current
        return torch.stack(logs, -1)


def path_nll(native, legal_mass, q, alpha, valid):
    a = alpha[:, None]
    mixed = torch.logaddexp(native + torch.log1p(-a), legal_mass + q + a.log())
    return -(mixed*valid).sum(-1)/valid.sum(-1)


def normalize(user, mean, std):
    return ((user-mean[:10])/std[:10]).clamp(-10, 10)


def tests():
    from experiment.phase21 import single_fusion_core as C
    from experiment.phase20.lime_decode_screen_core import mixed_log_values
    torch.set_num_threads(2)
    torch.manual_seed(0)
    parent = C.Fusion()
    torch.nn.init.normal_(parent.output.weight, std=.1)
    user = torch.randn(3, 10)
    counts = {}
    for mode in ('calibrate', 'global', 'user'):
        model = JointFusion(mode, parent.state_dict())
        beta, alpha = model(user)
        torch.testing.assert_close(beta, parent(user, torch.zeros(3, 4, 6), 'user')[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(alpha, torch.full((3,), .3))
        counts[mode] = dict(total=sum(p.numel() for p in model.parameters()),
                           adapted=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert counts == {'calibrate': {'total': 419, 'adapted': 1},
                      'global': {'total': 419, 'adapted': 419}, 'user': {'total': 451, 'adapted': 451}}
    paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 3, 1], [0, 3, 7, 8, 1]]
    tree = PathMass(paths, 'cpu')
    scores = torch.tensor([[.2,.5,.7,.1], [.8,.6,.2,.3], [.1,.3,.7,.8]], dtype=torch.double, requires_grad=True)
    targets = torch.tensor([0, 2, 3])
    q = tree.logs(scores, targets)
    for j, target in enumerate(targets.tolist()):
        reference = tree.index.conditional_logs(scores[j].detach().numpy())
        path = paths[target]
        expected = [reference[tuple(path[:d])][path[d]] for d in range(1, len(path))]
        np.testing.assert_allclose(q[j, :len(expected)].detach().numpy(), expected, atol=1e-12)
    native = torch.full_like(q, -3.)
    mass = torch.full_like(q, -.4)
    alpha = torch.tensor([.01, .3, .9], dtype=torch.double, requires_grad=True)
    valid = tree.valid[targets]
    actual = path_nll(native, mass, q, alpha, valid)
    expected = [-np.mean(mixed_log_values(-3., -.4, q[j, valid[j]].detach().numpy(), float(alpha[j]))) for j in range(3)]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    assert torch.autograd.gradcheck(lambda s,a: path_nll(native, mass, tree.logs(s, targets), a, valid),
                                    (scores, alpha), eps=1e-6, atol=1e-5)
    zero = path_nll(native, mass, q, torch.zeros(3, dtype=torch.double), valid)
    torch.testing.assert_close(zero, torch.full((3,), 3., dtype=torch.double))
    return dict(parameters=counts, parent_beta_exact=True, initial_strength_point3=True,
                variable_length_tree_matches_numpy=True, decoder_mixture_matches_numpy=True,
                finite_difference_beta_score_and_alpha_gradients=True, alpha_zero_exact=True)
