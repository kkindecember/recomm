"""Refine the accepted seven expert/history inputs without GRAM-state inputs.

Factorial screen: replace Top5/full-history overlap by per-expert probability
mass on the items actually filtered, and retain/drop the two top1-top2 gaps.
Probability mass uses each expert independently under exp(20 * score), not
the eventual beta-weighted mixed distribution.
"""
import math
import numpy as np
import torch
from experiment.phase21 import task_context_features_core as PRIOR

SCALE = 20.
KEEP_WITHOUT_GAP = [0, 1, 2, 5, 6]
BASE_NAMES = list(PRIOR.NAMES['pruned7'])
MASS_NAMES = BASE_NAMES[:5] + ['filtered_mass_sas', 'filtered_mass_text']
NAMES = dict(base7=BASE_NAMES, mass7=MASS_NAMES,
             no_gap5=[BASE_NAMES[i] for i in KEEP_WITHOUT_GAP],
             mass_no_gap5=[MASS_NAMES[i] for i in KEEP_WITHOUT_GAP])
ARMS = tuple(NAMES)
normalize, numpy_forward = PRIOR.normalize, PRIOR.numpy_forward


def filtered_mass(experts, filtered_history, scale=SCALE):
    e = np.asarray(experts, dtype=np.float64)
    assert e.ndim == 2 and e.shape[0] == 2 and e.shape[1] > 0 and np.isfinite(e).all()
    ids = np.unique(np.asarray(filtered_history, dtype=np.int64))
    assert ids.ndim == 1 and ((ids >= 1) & (ids <= e.shape[1])).all()
    weights = np.exp(scale*(e-e.max(1, keepdims=True)))
    mass = weights[:, ids-1].sum(1)/weights.sum(1)
    assert np.isfinite(mass).all() and ((mass >= 0) & (mass <= 1+1e-12)).all()
    return np.clip(mass, 0., 1.).astype(np.float32)


def features(original, experts, filtered_history, arm):
    original = np.asarray(original, dtype=np.float32)
    assert original.shape == (10,) and np.isfinite(original).all() and arm in ARMS
    base = original[PRIOR.KEEP].copy()
    if arm in ('mass7', 'mass_no_gap5'):
        base[5:] = filtered_mass(experts, filtered_history)
    if arm in ('no_gap5', 'mass_no_gap5'):
        base = base[KEEP_WITHOUT_GAP]
    assert base.shape == (len(NAMES[arm]),)
    return base


class Fusion(torch.nn.Module):
    def __init__(self, arm, parent=None):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        dimension = len(NAMES[arm])
        prototype = PRIOR.Fusion('pruned7')
        self.input = torch.nn.Linear(dimension, 32)
        self.beta = torch.nn.Linear(32, 2)
        self.strength = torch.nn.Linear(32, 1)
        state = prototype.state_dict() if parent is None else parent
        with torch.no_grad():
            if parent is None:
                columns = list(range(7)) if dimension == 7 else KEEP_WITHOUT_GAP
                self.input.weight.copy_(state['input.weight'][:, columns])
            else:
                assert state['input.weight'].shape[1] == dimension
                self.input.weight.copy_(state['input.weight'])
            self.input.bias.copy_(state['input.bias'])
            self.beta.weight.copy_(state['beta.weight'])
            self.beta.bias.copy_(state['beta.bias'])
        torch.nn.init.zeros_(self.strength.weight)
        torch.nn.init.constant_(self.strength.bias, math.log(.3/.7))

    def forward(self, user):
        assert user.shape[-1] == len(NAMES[self.arm])
        h = torch.relu(self.input(user))
        return self.beta(h).softmax(-1), self.strength(h).squeeze(-1).sigmoid().clamp(1e-6, 1.-1e-6)


def tests():
    from experiment.phase20.lime_decode_screen_core import ExpertTrie
    torch.set_num_threads(2)
    rng = np.random.default_rng(62)
    e = rng.random((2, 13)).astype(np.float32)
    history = [1, 4, 9]
    p = np.exp(SCALE*e.astype(np.float64)); p /= p.sum(1, keepdims=True)
    actual = filtered_mass(e, history)
    np.testing.assert_allclose(actual, p[:, np.asarray(history)-1].sum(1), atol=3e-8, rtol=1e-6)
    np.testing.assert_allclose(filtered_mass(e+4., history), actual, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(filtered_mass(e, history+history), actual)
    np.testing.assert_array_equal(filtered_mass(e, []), [0., 0.])
    np.testing.assert_allclose(filtered_mass(np.ones((2, 13)), history), [3/13, 3/13], atol=3e-8)
    paths = [[0, i+2, 1] for i in range(13)]
    trie = ExpertTrie(paths, range(1, 14))
    for j in range(2):
        logs = trie.conditional_logs(e[j], temperature_scale=SCALE)[(0,)]
        expected = sum(math.exp(logs[i+1]) for i in history)
        assert abs(float(actual[j])-expected) < 3e-8
    old = rng.random(10).astype(np.float32)
    values = {a: features(old, e, history, a) for a in ARMS}
    np.testing.assert_array_equal(values['base7'], old[PRIOR.KEEP])
    np.testing.assert_array_equal(values['mass7'][:5], values['base7'][:5])
    np.testing.assert_array_equal(values['mass7'][5:], actual)
    for a, full in (('no_gap5', 'base7'), ('mass_no_gap5', 'mass7')):
        np.testing.assert_array_equal(values[a], values[full][KEEP_WITHOUT_GAP])
    changed = old.copy(); changed[[3,4]] += 20
    for a in ('no_gap5', 'mass_no_gap5'):
        np.testing.assert_array_equal(features(changed, e, history, a), values[a])
    changed = old.copy(); changed[[7,8]] += 20
    for a in ('mass7', 'mass_no_gap5'):
        np.testing.assert_array_equal(features(changed, e, history, a), values[a])
    models = {}
    for a in ARMS:
        torch.manual_seed(0); models[a] = Fusion(a)
    torch.manual_seed(0); prior = PRIOR.Fusion('pruned7')
    for key, value in prior.state_dict().items():
        torch.testing.assert_close(value, models['base7'].state_dict()[key], atol=0, rtol=0)
    with torch.no_grad():
        prior.beta.weight.normal_(std=.1); prior.strength.weight.normal_(std=.1)
    models['base7'].load_state_dict(prior.state_dict())
    x = torch.from_numpy(rng.normal(size=(8, 7)).astype(np.float32))
    for m in (prior, models['base7']):
        opt = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.001)
        b,a = m(x); loss = -b[:,0].log().mean()-a.log().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for key,value in prior.state_dict().items():
        torch.testing.assert_close(value,models['base7'].state_dict()[key],atol=0,rtol=0)
    torch.testing.assert_close(models['mass7'].input.weight[:, KEEP_WITHOUT_GAP],
                               models['mass_no_gap5'].input.weight,atol=0,rtol=0)
    for arm,model in models.items():
        u = rng.normal(size=(8,len(NAMES[arm]))).astype(np.float32)
        with torch.no_grad():
            model.beta.weight.normal_(std=.1); model.strength.weight.normal_(std=.1)
        b,a = model(torch.from_numpy(u))
        nb,na = numpy_forward(model.state_dict(),u,np.zeros(u.shape[1]),np.ones(u.shape[1]))
        np.testing.assert_allclose(b.detach().numpy(),nb,atol=3e-7,rtol=2e-6)
        np.testing.assert_allclose(a.detach().numpy(),na,atol=3e-7,rtol=2e-6)
        restored = Fusion(arm,model.state_dict()); rb,ra = restored(torch.from_numpy(u))
        torch.testing.assert_close(rb,b,atol=0,rtol=0)
        torch.testing.assert_close(ra,torch.full((8,),.3))
    counts={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()}
    assert counts == dict(base7=355,mass7=355,no_gap5=291,mass_no_gap5=291)
    return dict(probability_mass_matches_dense_softmax_and_trie=True,
                empty_uniform_duplicate_history_cases=True,score_shift_invariant=True,
                unchanged_columns_exact=True,deleted_gaps_invariant=True,replaced_top5_invariant=True,
                base7_initialization_and_optimizer_match=True,shared_initial_columns_match=True,
                numpy_forward_all_arms=True,parent_transfer_beta_preserved=True,parameters=counts)
