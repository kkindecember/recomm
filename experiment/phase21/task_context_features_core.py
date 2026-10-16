"""Task-context features for the existing joint beta/alpha user fuser.

Keep seven LIME-derived history/expert statistics. Replace history_ratio and
the two score standard deviations with observable initial GRAM entropy and
two expert/GRAM divergences. The added inputs enter the shared nonlinear
trunk, unlike the earlier alpha-only additive experiment.
"""
import math
import numpy as np
import torch
from experiment.phase21 import decision_features_core as O
from experiment.phase21.alpha_gram_features_core import root_features
from experiment.phase21.single_fusion_core import USER_NAMES

KEEP = [0, 1, 2, 3, 4, 7, 8]
REPLACED = [5, 6, 9]
EXTRA_NAMES = ['entropy_gram', 'js_behavior_gram', 'js_text_gram']
NAMES = dict(old10=list(USER_NAMES), pruned7=[USER_NAMES[j] for j in KEEP],
             task10=[USER_NAMES[j] for j in KEEP] + EXTRA_NAMES)
ARMS = tuple(NAMES)
normalize, numpy_forward = O.normalize, O.numpy_forward


def features(original, probabilities, arm):
    original = np.asarray(original)
    assert original.ndim == 2 and original.shape[1] == 10
    assert np.isfinite(original).all() and arm in ARMS
    if arm == 'old10':
        return np.ascontiguousarray(original)
    retained = original[:, KEEP]
    if arm == 'pruned7':
        return np.ascontiguousarray(retained)
    extra = root_features(probabilities)
    assert len(extra) == len(original)
    return np.ascontiguousarray(np.concatenate((retained, extra), axis=1))


class Fusion(torch.nn.Module):
    def __init__(self, arm, parent=None):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        dimension = len(NAMES[arm])
        prototype = O.Fusion(10)
        self.input = torch.nn.Linear(dimension, 32)
        self.beta = torch.nn.Linear(32, 2)
        self.strength = torch.nn.Linear(32, 1)
        state = prototype.state_dict() if parent is None else parent
        with torch.no_grad():
            if parent is None:
                columns = list(range(10)) if arm == 'old10' else KEEP + (REPLACED if arm == 'task10' else [])
                self.input.weight.copy_(state['input.weight'][:, columns])
            else:
                assert state['input.weight'].shape[1] >= dimension
                self.input.weight.copy_(state['input.weight'][:, :dimension])
            self.input.bias.copy_(state['input.bias'])
            key = 'beta' if 'beta.weight' in state else 'output'
            self.beta.weight.copy_(state[key+'.weight'])
            self.beta.bias.copy_(state[key+'.bias'])
        torch.nn.init.zeros_(self.strength.weight)
        torch.nn.init.constant_(self.strength.bias, math.log(.3/.7))

    def forward(self, user):
        assert user.shape[-1] == len(NAMES[self.arm])
        h = torch.relu(self.input(user))
        return self.beta(h).softmax(-1), self.strength(h).squeeze(-1).sigmoid().clamp(1e-6, 1.-1e-6)


def tests():
    torch.set_num_threads(2)
    rng = np.random.default_rng(61)
    old = rng.normal(size=(8, 10)).astype(np.float32)
    p = rng.random((8, 3, 6)); p /= p.sum(-1, keepdims=True)
    values = {a: features(old, p, a) for a in ARMS}
    np.testing.assert_array_equal(values['task10'][:, :7], values['pruned7'])
    altered = old.copy(); altered[:, REPLACED] += 100
    for arm in ('pruned7', 'task10'):
        np.testing.assert_array_equal(features(altered, p, arm), values[arm])
    q = root_features(p)
    entropy = -(p[:, 0]*np.log(p[:, 0])).sum(-1)/np.log(p.shape[-1])
    expected = [entropy]
    for j in (1, 2):
        m = (p[:, 0]+p[:, j])/2
        expected.append((.5*(p[:, 0]*np.log(p[:, 0]/m)).sum(-1)
                         +.5*(p[:, j]*np.log(p[:, j]/m)).sum(-1))/np.log(2))
    np.testing.assert_allclose(q, np.stack(expected, axis=1), atol=1e-7, rtol=1e-6)
    swapped = features(old[:, [0,2,1,4,3,6,5,8,7,9]], p[:, [0,2,1]], 'task10')
    np.testing.assert_array_equal(swapped, values['task10'][:, [0,2,1,4,3,6,5,7,9,8]])
    models = {}
    for arm in ARMS:
        torch.manual_seed(0); model = Fusion(arm); models[arm] = model
        beta, alpha = model(torch.from_numpy(values[arm]))
        torch.testing.assert_close(beta, torch.full((8, 2), .5), atol=0, rtol=0)
        torch.testing.assert_close(alpha, torch.full((8,), .3))
    torch.manual_seed(0); original = O.Fusion(10)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, models['old10'].state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(models['pruned7'].input.weight,
                               models['old10'].input.weight[:, KEEP], atol=0, rtol=0)
    torch.testing.assert_close(models['task10'].input.weight[:, :7],
                               models['pruned7'].input.weight, atol=0, rtol=0)
    with torch.no_grad():
        original.beta.weight.normal_(std=.1); original.strength.weight.normal_(std=.1)
    models['old10'].load_state_dict(original.state_dict())
    x = torch.from_numpy(old)
    for model in (original, models['old10']):
        opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        beta, alpha = model(x)
        loss = -beta[:, 0].log().mean()-alpha.log().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, models['old10'].state_dict()[key], atol=0, rtol=0)
    task = models['task10']
    with torch.no_grad():
        task.input.weight.zero_(); task.input.bias.fill_(2.)
        task.input.weight[:3, 7:] = torch.eye(3)
        task.beta.weight.zero_(); task.beta.weight[0, :3] = .1; task.beta.weight[1, :3] = -.1
        task.strength.weight.zero_(); task.strength.weight[0, :3] = .2
    x = torch.from_numpy(values['task10'])
    beta, alpha = task(x)
    for j in range(7, 10):
        changed = x.clone(); changed[:, j] += .5
        b, a = task(changed)
        assert (b-beta).abs().min() > 1e-5 and (a-alpha).abs().min() > 1e-5
    for arm, model in models.items():
        v = values[arm]
        beta, alpha = model(torch.from_numpy(v))
        nb, na = numpy_forward(model.state_dict(), v, np.zeros(v.shape[1]), np.ones(v.shape[1]))
        np.testing.assert_allclose(beta.detach().numpy(), nb, atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(alpha.detach().numpy(), na, atol=3e-7, rtol=2e-6)
        restored = Fusion(arm, model.state_dict())
        b, a = restored(torch.from_numpy(v))
        torch.testing.assert_close(b, beta, atol=0, rtol=0)
        torch.testing.assert_close(a, torch.full((len(v),), .3))
    counts = {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}
    assert counts == dict(old10=451, pruned7=355, task10=451)
    return dict(retained_features_exact=True, removed_feature_invariance=True,
                gram_features_independent_kl_verified=True, expert_swap_equivariant=True,
                old10_initialization_and_optimizer_exact=True, common_feature_initialization_exact=True,
                extra_features_can_change_both_beta_and_alpha=True, numpy_forward_all_arms=True,
                ce_transfer_preserves_beta_and_resets_alpha=True, parameters=counts)
