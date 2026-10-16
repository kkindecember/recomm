"""Add three observable GRAM statistics only to the alpha output branch.

The existing nine-feature shared trunk and beta branch are unchanged. Extra
features enter alpha's logit through three zero-initialized linear weights.
Joint training can still change the shared trunk and hence fitted beta values.
"""
import numpy as np
import torch
from experiment.phase21 import reduced_user_features_core as R

EXTRA_NAMES = ['entropy_gram', 'js_behavior_gram', 'js_text_gram']
NAMES = R.NAMES + EXTRA_NAMES
normalize = R.normalize


def root_features(probabilities):
    """Input is [user, GRAM/behavior/text, same legal first-token support]."""
    p = np.asarray(probabilities, dtype=np.float64)
    assert p.ndim == 3 and p.shape[1] == 3 and p.shape[2] > 1
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(-1), 1., atol=2e-6, rtol=0)
    p = p/p.sum(-1, keepdims=True)
    def entropy(x):
        return -(x*np.log(np.maximum(x, 1e-300))).sum(-1)
    ent = entropy(p)
    js = [(entropy((p[:, i]+p[:, 0])/2)-(ent[:, i]+ent[:, 0])/2)/np.log(2)
          for i in (1, 2)]
    return np.column_stack((ent[:, 0]/np.log(p.shape[2]),
                            np.clip(js[0], 0., 1.), np.clip(js[1], 0., 1.))).astype(np.float32)


class Fusion(R.Fusion):
    def __init__(self, dimension, parent=None):
        assert dimension in (9, 12)
        super().__init__(9, parent)
        self.dimension = dimension
        if dimension == 12:
            self.gram_strength = torch.nn.Linear(3, 1, bias=False)
            torch.nn.init.zeros_(self.gram_strength.weight)

    def forward(self, user):
        assert user.shape[-1] == self.dimension
        h = torch.relu(self.input(user[:, :9]))
        logit = self.strength(h).squeeze(-1)
        if self.dimension == 12:
            logit = logit+self.gram_strength(user[:, 9:]).squeeze(-1)
        return self.beta(h).softmax(-1), logit.sigmoid().clamp(1e-6, 1.-1e-6)


def numpy_forward(state, user, mean, std):
    sd = {k: v.detach().cpu().numpy() for k, v in state.items()}
    u = np.clip((user-mean)/std, -10, 10)
    h = np.maximum(u[:, :9]@sd['input.weight'].T+sd['input.bias'], 0)
    z = h@sd['beta.weight'].T+sd['beta.bias']
    beta = np.exp(z-z.max(-1, keepdims=True)); beta /= beta.sum(-1, keepdims=True)
    logit = (h@sd['strength.weight'].T+sd['strength.bias']).ravel()
    if 'gram_strength.weight' in sd:
        logit += (u[:, 9:]@sd['gram_strength.weight'].T).ravel()
    return beta, np.clip(1/(1+np.exp(-logit)), 1e-6, 1-1e-6)


def tests():
    torch.set_num_threads(2)
    rng = np.random.default_rng(46)
    u = torch.from_numpy(rng.normal(size=(8, 12)).astype(np.float32))
    torch.manual_seed(0); prior = R.Fusion(9)
    torch.manual_seed(0); base = Fusion(9)
    torch.manual_seed(0); extra = Fusion(12)
    for key, value in prior.state_dict().items():
        torch.testing.assert_close(value, base.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(value, extra.state_dict()[key], atol=0, rtol=0)
    for actual in (base(u[:, :9]), extra(u)):
        for a, b in zip(actual, prior(u[:, :9])):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Baseline forward and a real optimizer update must retain the old protocol.
    with torch.no_grad():
        prior.beta.weight.normal_(std=.1)
        prior.strength.weight.normal_(std=.1)
    base.load_state_dict(prior.state_dict())
    for model in (prior, base):
        opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        beta, alpha = model(u[:, :9])
        loss = -beta[:, 0].log().mean()-alpha.log().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for key, value in prior.state_dict().items():
        torch.testing.assert_close(value, base.state_dict()[key], atol=0, rtol=0)
    with torch.no_grad():
        for key, value in base.state_dict().items():
            extra.state_dict()[key].copy_(value)
    for a, b in zip(base(u[:, :9]), extra(u)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with torch.no_grad():
        extra.gram_strength.weight.copy_(torch.tensor([[.2, -.1, .3]]))
    b, a = extra(u)
    for coordinate in (9, 10, 11):
        changed = u.clone(); changed[:, coordinate] += .5
        bb, aa = extra(changed)
        torch.testing.assert_close(b, bb, atol=0, rtol=0)
        assert (a-aa).abs().min() > 1e-5
    x = u.clone().requires_grad_(True)
    bx, ax = extra(x)
    assert torch.autograd.grad(bx[:, 0].sum(), x, retain_graph=True)[0][:, 9:].abs().sum() == 0
    assert torch.autograd.grad(ax.sum(), x)[0][:, 9:].abs().sum() > 0
    nb, na = numpy_forward(extra.state_dict(), u.numpy(), np.zeros(12), np.ones(12))
    np.testing.assert_allclose(b.detach().numpy(), nb, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(a.detach().numpy(), na, atol=2e-7, rtol=2e-6)
    transferred = Fusion(12, base.state_dict())
    tb, ta = transferred(u)
    torch.testing.assert_close(tb, base(u[:, :9])[0], atol=0, rtol=0)
    torch.testing.assert_close(ta, torch.full((8,), .3))
    counts = {9: sum(p.numel() for p in base.parameters()),
              12: sum(p.numel() for p in extra.parameters())}
    assert counts == {9: 419, 12: 422}
    uniform = np.full((1, 3, 4), .25)
    np.testing.assert_allclose(root_features(uniform), [[1, 0, 0]], atol=1e-7)
    p = np.asarray([[[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]])
    np.testing.assert_allclose(root_features(p), [[0, 0, 1]], atol=1e-7)
    np.testing.assert_allclose(root_features(p[:, :, [2, 0, 3, 1]]), root_features(p), atol=0)
    return dict(baseline9_initialization_and_optimizer_exact=True,
                common_initial_outputs_exact=True, zero_extra_recovers_trained_base=True,
                each_extra_feature_changes_only_alpha_at_fixed_weights=True,
                beta_extra_input_gradient_zero=True, alpha_extra_input_gradient_nonzero=True,
                numpy_forward_verified=True, ce_parent_beta_preserved_alpha_reset=True,
                parameters=counts, root_entropy_js_extremes_and_permutation=True)
