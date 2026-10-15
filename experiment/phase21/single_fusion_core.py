"""One full-catalog SASRec/BGE fusion network; no prior gate or residual.

Expert caches are from the project's LIME-Rec-derived expert pipeline.
Item features are computed without a target argument or candidate shortlist.
"""
import numpy as np
import torch

DU, DC, WIDTH = 10, 6, 16
USER_NAMES = ['log_history_length'] + [f'{s}_{e}' for s in
    ('confidence', 'margin', 'std', 'history_in_top5') for e in ('sas', 'text')] + ['history_ratio']
ITEM_NAMES = ['score_sas', 'score_text', 'reciprocal_rank_sas', 'reciprocal_rank_text',
              'log_training_frequency', 'in_history']


def retained(raw):
    """Select expert channels before constructing any feature."""
    assert raw.ndim == 2 and raw.shape[0] == 3
    return np.asarray(raw[[0, 2]], dtype=np.float32)


def features(experts, history, counts):
    e = np.asarray(experts, dtype=np.float32)
    assert e.ndim == 2 and e.shape[0] == 2 and np.isfinite(e).all()
    v = e.shape[1]
    assert len(counts) == v + 1 and v >= 5
    mask = np.zeros(v, dtype=bool)
    mask[np.asarray(history, dtype=np.int64) - 1] = True
    orders = [np.lexsort((np.arange(v), -s)) for s in e]
    ranks = np.empty((2, v), dtype=np.int32)
    for j, order in enumerate(orders):
        ranks[j, order] = np.arange(1, v + 1)
    top = np.asarray([e[j, order[:2]] for j, order in enumerate(orders)])
    u = np.concatenate(([np.log1p(len(history))/5], top[:, 0] - e.mean(1), top[:, 0] - top[:, 1],
        e.std(1, ddof=1), [mask[order[:5]].mean() for order in orders], [mask.mean()])).astype(np.float32)
    c = np.column_stack((e.T, 1./ranks.T, np.log1p(counts[1:]), mask)).astype(np.float32)
    assert u.shape == (DU,) and c.shape == (v, DC)
    return u, c


class Fusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(WIDTH, 32)
        self.output = torch.nn.Linear(32, 2)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, user, item, arm):
        # Algebraically identical to concatenating user/item features. The
        # factorization avoids repeating a user projection for every item.
        h = torch.nn.functional.linear(user, self.input.weight[:, :DU], self.input.bias)[:, None, :]
        if arm == 'candidate':
            h = h + torch.nn.functional.linear(item, self.input.weight[:, DU:])
        else:
            assert arm == 'user'
        return self.output(torch.relu(h)).softmax(-1)


def normalized(user, item, mean, std):
    return ((user - mean[:DU]) / std[:DU]).clamp(-10, 10), ((item - mean[DU:]) / std[DU:]).clamp(-10, 10)


def score(model, user, item, mean, std, arm):
    u, c = normalized(user, item, mean, std)
    weights = model(u, c, arm)
    return (weights * item[..., :2]).sum(-1), weights


def numpy_score(state, user, item, mean, std, arm):
    """Independent concatenated dense MLP, including every catalog item."""
    u = np.clip((user - mean[:DU]) / std[:DU], -10, 10)
    c = np.clip((item - mean[DU:]) / std[DU:], -10, 10)
    if arm == 'user':
        c = np.zeros_like(c)
    x = np.concatenate((np.broadcast_to(u[None, :], (len(item), DU)), c), axis=1)
    sd = {name: value.detach().cpu().numpy() for name, value in state.items()}
    z = np.maximum(x @ sd['input.weight'].T + sd['input.bias'], 0)
    z = z @ sd['output.weight'].T + sd['output.bias']
    w = np.exp(z - z.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w * item[:, :2]).sum(-1), w


def target_rank(scores, target, history):
    # Original item IDs are 1-based; missing/filtered targets stay in denominator.
    s = np.asarray(scores).copy()
    s[np.asarray(history, dtype=np.int64) - 1] = -np.inf
    y = target - 1
    if not np.isfinite(s[y]):
        return len(s) + 1
    return int(1 + (s > s[y]).sum() + ((s == s[y]) & (np.arange(len(s)) < y)).sum())


def tests():
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    raw = rng.random((3, 231)).astype(np.float32)
    history = [1, 3, 7]
    u, c = features(retained(raw), history, np.ones(232))
    raw[1] = rng.normal(100, 20, 231)
    uu, cc = features(retained(raw), history, np.ones(232))
    np.testing.assert_array_equal(u, uu); np.testing.assert_array_equal(c, cc)
    ut = torch.from_numpy(np.stack((u, u + .2)))
    ct = torch.from_numpy(np.stack((c, c)))
    mean, std = torch.zeros(WIDTH), torch.ones(WIDTH)
    torch.manual_seed(0); model = Fusion()
    for arm in ('user', 'candidate'):
        f, w = score(model, ut, ct, mean, std, arm)
        torch.testing.assert_close(f, ct[..., :2].mean(-1), rtol=0, atol=0)
        torch.testing.assert_close(w, torch.full_like(w, .5), rtol=0, atol=0)
    torch.nn.init.normal_(model.output.weight, std=.2)
    for arm in ('user', 'candidate'):
        f, w = score(model, ut, ct, mean, std, arm)
        assert f.shape == (2, 231)
        torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]))
        assert ((f >= ct[..., :2].min(-1).values - 1e-6) & (f <= ct[..., :2].max(-1).values + 1e-6)).all()
        if arm == 'user':
            assert w.shape == (2, 1, 2)
        else:
            assert float(w[0, :, 0].std()) > 0 and float((f[0, 150:] - ct[0, 150:, :2].mean(-1)).abs().max()) > 0
        for j in range(2):
            a, _ = numpy_score(model.state_dict(), ut[j].numpy(), ct[j].numpy(), mean.numpy(), std.numpy(), arm)
            np.testing.assert_allclose(a, f[j].detach().numpy(), atol=2e-6, rtol=1e-5)
            single, _ = score(model, ut[j:j+1], ct[j:j+1], mean, std, arm)
            torch.testing.assert_close(single[0], f[j])
        # Compare the exact training CE to a separate stable NumPy expression.
        target = torch.tensor([0, 230])
        loss = torch.nn.functional.cross_entropy(20*f, target)
        z = (20*f).detach().double().numpy(); mx = z.max(-1)
        expected = np.mean(mx + np.log(np.exp(z-mx[:, None]).sum(-1)) - z[np.arange(2), target.numpy()])
        assert abs(float(loss) - expected) < 2e-6
        model.zero_grad(); loss.backward()
        assert torch.isfinite(model.output.weight.grad).all() and model.output.weight.grad.abs().sum() > 0
    assert target_rank(np.ones(231), 3, [2]) == 2
    assert target_rank(np.ones(231), 3, [3]) == 232
    return dict(removed_expert_invariant=True, uniform_initialization=True, simplex_and_convex_bounds=True,
        user_shared_weights=True, full_catalog_candidate_weights=True, numpy_forward_and_dense_CE=True,
        batch_invariance=True, history_and_tie_rank=True, parameters=sum(p.numel() for p in model.parameters()))
