"""Remove only history_ratio from the existing ten-feature joint user head."""
import math
import numpy as np
import torch
from experiment.phase21 import decision_features_core as ORIGINAL
from experiment.phase21.single_fusion_core import USER_NAMES

NAMES = USER_NAMES[:9]
assert USER_NAMES[9] == 'history_ratio'


def select(user, dimension):
    user = np.asarray(user)
    assert dimension in (9, 10) and user.ndim == 2 and user.shape[1] == 10
    assert np.isfinite(user).all()
    return np.ascontiguousarray(user[:, :dimension])


class Fusion(torch.nn.Module):
    def __init__(self, dimension, parent=None):
        super().__init__()
        assert dimension in (9, 10)
        prototype = ORIGINAL.Fusion(10)
        self.input = torch.nn.Linear(dimension, 32)
        self.beta = torch.nn.Linear(32, 2)
        self.strength = torch.nn.Linear(32, 1)
        state = prototype.state_dict() if parent is None else parent
        with torch.no_grad():
            self.input.weight.copy_(state['input.weight'][:, :dimension])
            self.input.bias.copy_(state['input.bias'])
            key = 'beta' if 'beta.weight' in state else 'output'
            self.beta.weight.copy_(state[key+'.weight'])
            self.beta.bias.copy_(state[key+'.bias'])
        torch.nn.init.zeros_(self.strength.weight)
        torch.nn.init.constant_(self.strength.bias, math.log(.3/.7))

    def forward(self, user):
        h = torch.relu(self.input(user))
        return self.beta(h).softmax(-1), self.strength(h).squeeze(-1).sigmoid().clamp(1e-6, 1.-1e-6)


normalize, numpy_forward = ORIGINAL.normalize, ORIGINAL.numpy_forward


def tests():
    rng = np.random.default_rng(46)
    user = rng.normal(size=(8, 10)).astype(np.float32)
    altered = user.copy(); altered[:, 9] += 100
    np.testing.assert_array_equal(select(user, 9), select(altered, 9))
    np.testing.assert_array_equal(select(user, 9), user[:, :9])
    torch.manual_seed(0); original = ORIGINAL.Fusion(10)
    torch.manual_seed(0); actual = Fusion(10)
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, actual.state_dict()[k], atol=0, rtol=0)
    x = torch.from_numpy(user)
    # Match the old head under nonuniform beta and nonconstant alpha, including
    # a real CE optimizer update, rather than merely checking initial constants.
    with torch.no_grad():
        original.beta.weight.normal_(std=.1)
        original.strength.weight.normal_(std=.1)
    actual.load_state_dict(original.state_dict())
    for a, b in zip(actual(x), original(x)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    item = torch.from_numpy(rng.random((8, 17, 2)).astype(np.float32))
    target = torch.arange(8)
    for model in (original, actual):
        opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        opt.zero_grad(); beta, alpha = model(x)
        loss = torch.nn.functional.cross_entropy(20*(beta[:, None, :]*item).sum(-1), target)
        loss = loss - alpha.log().mean()
        loss.backward(); opt.step()
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, actual.state_dict()[k], atol=0, rtol=0)
    torch.manual_seed(0); nine = Fusion(9)
    torch.manual_seed(0); ten = Fusion(10)
    torch.testing.assert_close(nine.input.weight, ten.input.weight[:, :9], atol=0, rtol=0)
    torch.testing.assert_close(nine.input.bias, ten.input.bias, atol=0, rtol=0)
    counts = {9: sum(p.numel() for p in nine.parameters()), 10: sum(p.numel() for p in ten.parameters())}
    assert counts == {9: 419, 10: 451}
    with torch.no_grad():
        nine.beta.weight.normal_(std=.1)
        nine.strength.weight.normal_(std=.1)
    inputs = select(user, 9)
    a, b = nine(torch.from_numpy(inputs))
    na, nb = numpy_forward(nine.state_dict(), inputs, np.zeros(9, np.float32), np.ones(9, np.float32))
    np.testing.assert_allclose(a.detach().numpy(), na, atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(b.detach().numpy(), nb, atol=2e-7, rtol=1e-6)
    # CE-parent transfer preserves beta while deliberately resetting alpha.
    restored = Fusion(9, nine.state_dict())
    rbeta, ralpha = restored(torch.from_numpy(inputs))
    torch.testing.assert_close(rbeta, a, atol=0, rtol=0)
    torch.testing.assert_close(ralpha, torch.full((8,), .3))
    return dict(deleted_feature_invariant=True, first_nine_columns_exact=True,
                original10_initialization_forward_optimizer_exact=True,
                common_nine_initial_columns_exact=True, parameters=counts,
                nine_numpy_forward_verified=True, parent_transfer_beta_preserved=True,
                alpha_reset_point3=True)
