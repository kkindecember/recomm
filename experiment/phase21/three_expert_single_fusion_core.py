"""Three-channel extension of the current user-level single fusion network.

Same feature definitions, hidden width, score normalization and loss interface.
The frozen two-channel model is used only as a reference, never as a residual.
Expert cache order is SASRec, ItemCF, BGE. The user gate has no item dependence.
"""
import numpy as np
import torch
from experiment.phase21 import single_fusion_core as TWO

DU, DC, WIDTH = 14, 8, 22
EXPERTS = ('sas', 'itemcf', 'text')
USER_NAMES = ['log_history_length']+[f'{s}_{e}' for s in
    ('confidence', 'margin', 'std', 'history_in_top5') for e in EXPERTS]+['history_ratio']
ITEM_NAMES = [f'score_{e}' for e in EXPERTS]+[f'reciprocal_rank_{e}' for e in EXPERTS]+['log_training_frequency', 'in_history']
TWO_USER_COLUMNS = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
TWO_ITEM_COLUMNS = [0, 2, 3, 5, 6, 7]
TWO_INPUT_COLUMNS = TWO_USER_COLUMNS+[DU+i for i in TWO_ITEM_COLUMNS]


def retained(raw):
    assert raw.ndim == 2 and raw.shape[0] == 3
    return np.asarray(raw, dtype=np.float32)


def features(experts, history, counts):
    e = np.asarray(experts, dtype=np.float32)
    assert e.ndim == 2 and e.shape[0] == 3 and np.isfinite(e).all()
    v = e.shape[1]
    assert len(counts) == v+1 and v >= 5
    mask = np.zeros(v, dtype=bool)
    mask[np.asarray(history, dtype=np.int64)-1] = True
    orders = [np.lexsort((np.arange(v), -s)) for s in e]
    ranks = np.empty((3, v), dtype=np.int32)
    for j, order in enumerate(orders):
        ranks[j, order] = np.arange(1, v+1)
    top = np.asarray([e[j, order[:2]] for j, order in enumerate(orders)])
    u = np.concatenate(([np.log1p(len(history))/5], top[:, 0]-e.mean(1), top[:, 0]-top[:, 1],
        e.std(1, ddof=1), [mask[order[:5]].mean() for order in orders], [mask.mean()])).astype(np.float32)
    c = np.column_stack((e.T, 1./ranks.T, np.log1p(counts[1:]), mask)).astype(np.float32)
    assert u.shape == (DU,) and c.shape == (v, DC)
    return u, c


class Fusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Recreate the original RANDOM initialization, not trained weights.
        # This preserves every shared input weight/bias under the same seed.
        original = TWO.Fusion()
        self.input = torch.nn.Linear(WIDTH, 32)
        self.output = torch.nn.Linear(32, 3)
        with torch.no_grad():
            self.input.weight[:, TWO_INPUT_COLUMNS] = original.input.weight
            self.input.bias.copy_(original.input.bias)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, user, item, arm):
        assert arm == 'user'
        h = torch.nn.functional.linear(user, self.input.weight[:, :DU], self.input.bias)[:, None, :]
        return self.output(torch.relu(h)).softmax(-1)


def normalized(user, item, mean, std):
    return ((user-mean[:DU])/std[:DU]).clamp(-10, 10), ((item-mean[DU:])/std[DU:]).clamp(-10, 10)


def score(model, user, item, mean, std, arm):
    u, c = normalized(user, item, mean, std)
    weights = model(u, c, arm)
    return (weights*item[..., :3]).sum(-1), weights


def numpy_score(state, user, item, mean, std, arm):
    assert arm == 'user'
    u = np.clip((user-mean[:DU])/std[:DU], -10, 10)
    x = np.concatenate((np.broadcast_to(u[None], (len(item), DU)), np.zeros((len(item), DC), dtype=np.float32)), axis=1)
    sd = {k: v.detach().cpu().numpy() for k, v in state.items()}
    z = np.maximum(x@sd['input.weight'].T+sd['input.bias'], 0)
    z = z@sd['output.weight'].T+sd['output.bias']
    w = np.exp(z-z.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w*item[:, :3]).sum(-1), w


target_rank = TWO.target_rank


def tests():
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261015)
    raw = rng.random((3, 231)).astype(np.float32)
    history, counts = [1, 7, 7, 10], np.arange(232)
    u, c = features(raw, history, counts)
    old_u, old_c = TWO.features(raw[[0, 2]], history, counts)
    np.testing.assert_array_equal(u[TWO_USER_COLUMNS], old_u)
    np.testing.assert_array_equal(c[:, TWO_ITEM_COLUMNS], old_c)
    changed = raw.copy(); changed[1] = rng.random(231)
    changed_u, changed_c = features(changed, history, counts)
    np.testing.assert_array_equal(changed_u[TWO_USER_COLUMNS], old_u)
    np.testing.assert_array_equal(changed_c[:, TWO_ITEM_COLUMNS], old_c)
    assert not np.array_equal(changed_c[:, 1], c[:, 1])
    torch.manual_seed(0); original = TWO.Fusion()
    torch.manual_seed(0); model = Fusion()
    torch.testing.assert_close(model.input.weight[:, TWO_INPUT_COLUMNS], original.input.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.input.bias, original.input.bias, rtol=0, atol=0)
    user = torch.from_numpy(np.stack((u, changed_u)))
    item = torch.from_numpy(np.stack((c, changed_c)))
    norm = torch.zeros(WIDTH), torch.ones(WIDTH)
    value, weights = score(model, user, item, *norm, 'user')
    torch.testing.assert_close(value, item[..., :3].mean(-1), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(weights, torch.full((2, 1, 3), 1/3), rtol=0, atol=0)
    torch.nn.init.normal_(model.output.weight, std=.2)
    value, weights = score(model, user, item, *norm, 'user')
    assert weights.shape == (2, 1, 3)
    torch.testing.assert_close(weights.sum(-1), torch.ones((2, 1)))
    assert ((value >= item[..., :3].min(-1).values-2e-6) & (value <= item[..., :3].max(-1).values+2e-6)).all()
    for j in range(2):
        actual, w = numpy_score(model.state_dict(), user[j].numpy(), item[j].numpy(), norm[0].numpy(), norm[1].numpy(), 'user')
        np.testing.assert_allclose(actual, value[j].detach().numpy(), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(w, np.broadcast_to(w[:1], w.shape), atol=1e-7)
        single, _ = score(model, user[j:j+1], item[j:j+1], *norm, 'user')
        torch.testing.assert_close(single[0], value[j])
    target = torch.tensor([0, 230])
    loss = torch.nn.functional.cross_entropy(20*value, target)
    z = (20*value).detach().double().numpy(); mx = z.max(-1)
    expected = np.mean(mx+np.log(np.exp(z-mx[:, None]).sum(-1))-z[np.arange(2), target.numpy()])
    assert abs(float(loss)-expected) < 2e-6
    model.zero_grad(); loss.backward()
    assert torch.isfinite(model.output.weight.grad).all() and model.output.weight.grad.abs().sum() > 0
    # Exact two-expert reduction: map an arbitrary trained-like 2-way head,
    # remove CF features and mask its output. This is a unit check, not a run arm.
    torch.nn.init.normal_(original.output.weight, std=.2)
    with torch.no_grad():
        model.input.weight.zero_()
        model.input.weight[:, TWO_INPUT_COLUMNS] = original.input.weight
        model.input.bias.copy_(original.input.bias)
        model.output.weight[[0, 2]] = original.output.weight
        model.output.bias[[0, 2]] = original.output.bias
    h = torch.relu(torch.nn.functional.linear(user, model.input.weight[:, :DU], model.input.bias))
    logits = model.output(h); logits[:, 1] = -torch.inf
    recovered = (logits.softmax(-1)[:, None]*item[..., :3]).sum(-1)
    expected, _ = TWO.score(original, user[:, TWO_USER_COLUMNS], item[:, :, TWO_ITEM_COLUMNS],
                            torch.zeros(TWO.WIDTH), torch.ones(TWO.WIDTH), 'user')
    torch.testing.assert_close(recovered, expected, atol=2e-6, rtol=1e-5)
    return dict(shared_features_identical=True, itemcf_cannot_change_other_features=True,
        shared_random_initialization_identical=True, uniform_three_initialization=True,
        user_shared_weights=True, convex_bounds=True, numpy_forward_and_CE=True,
        batch_invariance=True, masked_three_reduces_to_two=True,
        parameters_three=sum(p.numel() for p in model.parameters()),
        parameters_two=sum(p.numel() for p in original.parameters()))
