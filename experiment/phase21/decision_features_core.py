"""Observable start-of-generation statistics for a user-level joint fuser.

No target or fused prediction is an argument to the feature constructor.
Expert probabilities use the existing LIME-derived scores / trie mass rule.
GRAM and both experts are compared on the same legal first-token support.
"""
import math
import numpy as np
import torch
from experiment.phase21 import single_fusion_core as C

NAMES = ['log_history_length', 'entropy_gram', 'entropy_behavior', 'entropy_text',
         'js_behavior_text', 'js_behavior_gram', 'js_text_gram']


def root_mapping(paths):
    assert paths and all(p[0] == 0 and len(p) > 1 for p in paths)
    tokens = np.unique([p[1] for p in paths])
    assert len(tokens) > 1
    groups = np.searchsorted(tokens, [p[1] for p in paths])
    return tokens, groups


def expert_root(scores, groups, branches, scale=20.):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 1 and len(scores) == len(groups) and np.isfinite(scores).all()
    weights = np.exp(scale * (scores - scores.max()))
    mass = np.bincount(groups, weights=weights, minlength=branches)
    return mass / mass.sum()


def features(history_length, gram, behavior, text):
    p = np.stack((gram, behavior, text)).astype(np.float64)
    assert p.ndim == 2 and p.shape[1] > 1 and history_length >= 0
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1., atol=2e-6, rtol=0)
    p /= p.sum(1, keepdims=True)
    def entropy(a):
        return -np.sum(a * np.log(np.maximum(a, 1e-300)), axis=-1)
    ent = entropy(p) / math.log(p.shape[1])
    js = []
    for i, j in ((1, 2), (1, 0), (2, 0)):
        # JS/log(2), not the square-root Jensen-Shannon distance.
        value = (entropy((p[i]+p[j])/2.) - (entropy(p[i])+entropy(p[j]))/2.)/math.log(2.)
        js.append(float(np.clip(value, 0., 1.)))
    return np.asarray([np.log1p(history_length)/5., *ent, *js], dtype=np.float32)


class Fusion(torch.nn.Module):
    def __init__(self, dimension, parent=None):
        super().__init__()
        assert dimension in (7, 10)
        # Match the existing initialization scale and common random prefix;
        # only active feature columns are stored in the new network.
        prototype = C.Fusion()
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


def normalize(user, mean, std):
    return ((user-mean)/std).clamp(-10, 10)


def numpy_forward(state, user, mean, std):
    s = {k: v.detach().cpu().numpy() for k, v in state.items()}
    x = np.clip((user-mean)/std, -10, 10)
    h = np.maximum(x@s['input.weight'].T+s['input.bias'], 0)
    z = h@s['beta.weight'].T+s['beta.bias']
    beta = np.exp(z-z.max(-1, keepdims=True)); beta /= beta.sum(-1, keepdims=True)
    z = (h@s['strength.weight'].T+s['strength.bias']).ravel()
    return beta, np.clip(1/(1+np.exp(-z)), 1e-6, 1.-1e-6)


def tests():
    from experiment.phase20.lime_decode_screen_core import ExpertTrie
    from experiment.phase21.joint_fusion_core import JointFusion
    uniform = np.ones(4)/4
    f = features(3, uniform, uniform, uniform)
    np.testing.assert_allclose(f[1:4], 1.)
    np.testing.assert_allclose(f[4:], 0., atol=1e-7)
    a, b = np.eye(4)[0], np.eye(4)[1]
    f = features(3, a, a, b)
    np.testing.assert_allclose(f[1:4], 0.)
    np.testing.assert_allclose(f[4:], [1., 0., 1.])
    rng = np.random.default_rng(19)
    p = rng.random((3, 6)); p /= p.sum(1, keepdims=True)
    f = features(10, *p)
    swapped = features(10, p[0], p[2], p[1])
    np.testing.assert_allclose(swapped, f[[0,1,3,2,4,6,5]])
    permutation = rng.permutation(6)
    np.testing.assert_allclose(features(10, *p[:, permutation]), f)
    paths = [[0,2,4,1], [0,2,5,1], [0,3,1], [0,6,1]]
    tokens, groups = root_mapping(paths)
    scores = rng.random(4)
    q = expert_root(scores, groups, len(tokens))
    logs = ExpertTrie(paths, range(1,5)).conditional_logs(scores)[(0,)]
    np.testing.assert_allclose(q, [np.exp(logs[int(t)]) for t in tokens], atol=1e-14)
    np.testing.assert_allclose(expert_root(scores+10, groups, len(tokens)), q, atol=1e-14)
    # Direct KL definition verifies the independent entropy identity used above.
    for k, (i,j) in enumerate(((1,2),(1,0),(2,0))):
        m = (p[i]+p[j])/2
        expected = .5*(np.sum(p[i]*np.log(p[i]/m))+np.sum(p[j]*np.log(p[j]/m)))/np.log(2.)
        np.testing.assert_allclose(f[4+k], expected, atol=1e-7)
    counts = {}
    for d in (7,10):
        torch.manual_seed(0); model = Fusion(d)
        x = torch.randn(8,d)
        beta, alpha = model(x)
        torch.testing.assert_close(beta, torch.full((8,2),.5))
        torch.testing.assert_close(alpha, torch.full((8,),.3))
        torch.nn.init.normal_(model.beta.weight, std=.1)
        torch.nn.init.normal_(model.strength.weight, std=.1)
        beta, alpha = model(x)
        nb, na = numpy_forward(model.state_dict(), x.numpy(), np.zeros(d,np.float32), np.ones(d,np.float32))
        np.testing.assert_allclose(beta.detach().numpy(),nb,atol=2e-7)
        np.testing.assert_allclose(alpha.detach().numpy(),na,atol=2e-7)
        counts[d] = sum(p.numel() for p in model.parameters())
    assert counts == {7:355,10:451}
    parent = C.Fusion().state_dict()
    original, compact = JointFusion('user',parent), Fusion(10,parent)
    x = torch.randn(4,10)
    for a,b in zip(original(x),compact(x)):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    # Same minibatch and optimizer must update the compact old10 control as the
    # established joint head, rather than only reproducing its initial outputs.
    oo = torch.optim.AdamW(original.parameters(),lr=.002,weight_decay=.001)
    co = torch.optim.AdamW(compact.parameters(),lr=.002,weight_decay=.001)
    for model,opt in ((original,oo),(compact,co)):
        opt.zero_grad(); beta,alpha = model(x)
        loss = -(alpha*beta[:,0]+(1-alpha)*.2).log().mean()
        loss.backward(); opt.step()
    for a,b in zip(original(x),compact(x)):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    seven = Fusion(7)
    restored = Fusion(7,seven.state_dict())
    for a,b in zip(seven(torch.zeros(2,7)),restored(torch.zeros(2,7))):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    return dict(entropy_endpoints=True, js_endpoints_and_direct_kl=True,
        expert_swap_and_branch_permutation_invariant=True, root_matches_existing_trie=True,
        score_shift_invariant=True, numpy_fuser_parity=True, old_head_forward_exact=True,
        parameters=counts, old_head_optimizer_step_exact=True,
        seven_dimension_parent_reload=True, target_not_in_feature_interface=True)
