"""Prefix-specific strengths for frozen SASRec/BGE probability guidance.

Retains the LIME-derived experts and AGRec-inspired legal-mass preserving rule.
All siblings receive the SAME alpha, so their total native mass is preserved.
"""
import math
import numpy as np
import torch


def path_scores(native, legal, q, alpha, valid):
    if alpha.ndim == 1:
        alpha = alpha[:, None, None]
    mixed = torch.logaddexp(native + torch.log1p(-alpha), legal + q + alpha.log())
    return torch.where(valid, mixed, torch.zeros_like(mixed)).sum(-1) / valid.sum(-1).clamp_min(1)


def pool_kl(native_scores, mixed_scores, valid):
    """Explicit direction: D_KL(p_GRAM || p_mix), with pool temperature 1/20."""
    lg = (20*native_scores).masked_fill(~valid, -torch.inf).log_softmax(-1)
    lm = (20*mixed_scores).masked_fill(~valid, -torch.inf).log_softmax(-1)
    difference = torch.where(valid, lg-lm, torch.zeros_like(lg))
    return (lg.exp()*difference).sum(-1)


def evidence(embeddings, history):
    """Positive BGE support shares; signed contributions reconstruct raw cosine.

    Returns top1 share, top3 share, and weighted relative position (newest=1).
    The shares describe positive support, not a causal attribution of SASRec.
    """
    h = embeddings[torch.as_tensor(history, device=embeddings.device)-1]
    age = torch.arange(len(history)-1, -1, -1, device=h.device, dtype=h.dtype)
    weight = (-.1*age).exp()
    normed = embeddings / (embeddings.norm(dim=1, keepdim=True)+1e-8)
    contribution = (normed @ h.T)*weight
    user_sum = (weight[:, None]*h).sum(0)
    # Upstream normalizes the weighted mean with an additive epsilon.
    raw = contribution.sum(-1)/(user_sum.norm()+1e-8*weight.sum())
    support = contribution.clamp_min(0)
    total = support.sum(-1).clamp_min(1e-12)
    top = support.topk(min(3, len(history)), dim=1).values
    position = torch.linspace(0., 1., len(history), device=h.device) if len(history)>1 else torch.ones(1, device=h.device)
    features = torch.stack((top[:, 0]/total, top.sum(-1)/total,
                            (support*position).sum(-1)/total), -1)
    return features, raw


def prefix_evidence(index, item_features, scores):
    """Expert leaf-mass conditional expectation of evidence under each prefix."""
    weight = np.exp(20*(np.asarray(scores, np.float64)-np.max(scores)))
    node, item = index.node_items, index.item_refs-1
    mass = np.bincount(node, weights=weight[item], minlength=len(index.nodes))
    return np.column_stack([np.bincount(node, weights=(weight*item_features[:, k])[item],
                            minlength=len(index.nodes))/mass for k in range(3)]).astype(np.float32)


class Head(torch.nn.Module):
    def __init__(self, global_only=False, use_evidence=False):
        super().__init__()
        self.global_only, self.use_evidence = global_only, use_evidence
        width = 0 if global_only else 32 + (3 if use_evidence else 0)
        self.weight = torch.nn.Parameter(torch.zeros(width))
        self.bias = torch.nn.Parameter(torch.tensor(math.log(.3/.7)))

    def forward(self, hidden, evidence_features=None):
        base = self.bias.expand(len(hidden))
        if not self.global_only:
            base = base + hidden @ self.weight[:32]
        if self.use_evidence:
            return (base[:, None, None] + evidence_features @ self.weight[32:]).sigmoid().clamp(1e-6, 1-1e-6)
        return base.sigmoid().clamp(1e-6, 1-1e-6)


class PrefixGuide:
    """A policy(prefix, log_legal_mass) supplies a scalar alpha per parent."""
    def __init__(self, index, expert_logs, policy):
        self.index, self.expert_logs, self.policy = index, expert_logs, policy
        self.trace, self.alpha_trace = {}, {}
        self.max_mass_error = 0.

    def __call__(self, input_ids, scores):
        masses = torch.logsumexp(scores, -1)
        prefixes = input_ids.detach().cpu().tolist()
        rr, tt, q, aa = [], [], [], []
        for row, prefix_list in enumerate(prefixes):
            prefix = tuple(prefix_list)
            node = self.index.lookup.get(prefix)
            if node is None:
                assert torch.isneginf(scores[row]).all()
                continue
            tokens = list(self.index.children[node])
            if not tokens:
                continue
            mass = float(masses[row])
            alpha = float(self.policy(prefix, mass))
            assert 0 <= alpha < 1 and math.isfinite(alpha)
            values = scores[row, tokens].detach().cpu().double().numpy()
            if prefix in self.alpha_trace:
                assert abs(self.alpha_trace[prefix]-alpha)<2e-5
            else:
                self.trace[prefix] = (tokens, values, mass)
                self.alpha_trace[prefix] = alpha
            rr.extend([row]*len(tokens)); tt.extend(tokens)
            aa.extend([alpha]*len(tokens))
            q.extend(self.expert_logs[prefix][token] for token in tokens)
        if rr:
            rr = torch.tensor(rr, device=scores.device)
            tt = torch.tensor(tt, device=scores.device)
            # Compute scalar logs in float64 before casting, like ProbabilityGuide.
            left = torch.tensor(np.log1p(-np.asarray(aa)), device=scores.device, dtype=scores.dtype)
            with np.errstate(divide='ignore'):
                right = torch.tensor(np.log(aa), device=scores.device, dtype=scores.dtype)
            q = torch.tensor(q, device=scores.device, dtype=scores.dtype)
            scores[rr, tt] = torch.logaddexp(scores[rr, tt]+left, masses[rr]+q+right)
            finite = torch.isfinite(masses)
            error = (torch.logsumexp(scores, -1)[finite]-masses[finite]).abs().max()
            self.max_mass_error = max(self.max_mass_error, float(error))
            assert self.max_mass_error < 4e-5
        return scores

    def rescore(self, paths):
        result = []
        for path in paths:
            total = 0.
            for d in range(1, len(path)):
                prefix = tuple(path[:d]); token = path[d]
                tokens, native, mass = self.trace[prefix]
                a = self.alpha_trace[prefix]
                lp = native[tokens.index(token)]
                total += lp if a == 0 else np.logaddexp(lp+math.log1p(-a),
                    mass+self.expert_logs[prefix][token]+math.log(a))
            result.append(total/(len(path)-1))
        return np.asarray(result)


def tests():
    from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide
    paths = [[0,2,4,1], [0,2,5,1], [0,3,6,1]]
    index = ExpertTrie(paths, range(1,4)); q = index.conditional_logs(np.array([.1,.7,.5]))
    inputs = torch.tensor([[0,2], [0,3]])
    scores = torch.full((2,9), -torch.inf, dtype=torch.double)
    scores[0,4:6] = torch.log(torch.tensor([.02,.08], dtype=torch.double))
    scores[1,6] = math.log(.6)
    for a in (0., .3, .999):
        actual = PrefixGuide(index,q,lambda p,m:a)(inputs,scores.clone())
        expected = ProbabilityGuide(index,q,a)(inputs,scores.clone())
        torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
    variable = PrefixGuide(index,q,lambda p,m:.3*(1-math.exp(m)))
    value = variable(inputs,scores.clone())
    torch.testing.assert_close(value.exp().sum(-1),scores.exp().sum(-1),atol=1e-14,rtol=1e-14)
    assert variable.alpha_trace[(0,2)]>variable.alpha_trace[(0,3)]
    # Finite, nonzero gradient on a redistribution, including padded candidates.
    ng = torch.tensor([[-.1,-.15,0.]], dtype=torch.double)
    valid = torch.tensor([[True,True,False]])
    torch.testing.assert_close(pool_kl(ng,ng,valid),torch.zeros(1,dtype=torch.double),atol=0,rtol=0)
    a = torch.tensor(.3,dtype=torch.double,requires_grad=True)
    loss = pool_kl(ng,ng+a*torch.tensor([[.02,-.02,0.]]),valid).sum()
    loss.backward(); assert torch.isfinite(a.grad) and abs(float(a.grad))>1e-5
    torch.manual_seed(4); emb = torch.rand(11,6,dtype=torch.double)
    feat, raw = evidence(emb,[1,3,5,7])
    w = torch.exp(-.1*torch.arange(3,-1,-1,dtype=torch.double))
    user = (emb[[0,2,4,6]]*w[:,None]).sum(0)/w.sum()
    expected = (emb/(emb.norm(dim=1,keepdim=True)+1e-8)) @ (user/(user.norm()+1e-8))
    torch.testing.assert_close(raw,expected,atol=1e-12,rtol=1e-12)
    assert bool(((feat>=0)&(feat<=1+1e-12)).all())
    pf = prefix_evidence(index,feat[:3].numpy(),np.array([.1,.7,.5]))
    np.testing.assert_allclose(pf[index.lookup[(0,3)]],feat[2].numpy(),rtol=1e-6)
    native = torch.tensor([[[-2.,-3.],[-1.,0.]]],dtype=torch.double)
    legal = torch.tensor([[[-.2,-.3],[-.1,0.]]],dtype=torch.double)
    expert = torch.tensor([[[-.3,-.5],[-.4,0.]]],dtype=torch.double)
    valid = torch.tensor([[[True,True],[True,False]]])
    alpha = torch.tensor([[[.2,.4],[.6,.1]]],dtype=torch.double,requires_grad=True)
    actual = path_scores(native,legal,expert,alpha,valid)
    probs = (1-alpha)*native.exp()+alpha*(legal+expert).exp()
    expected = torch.where(valid,probs.log(),torch.zeros_like(probs)).sum(-1)/valid.sum(-1)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    torch.autograd.gradcheck(lambda aa:path_scores(native,legal,expert,aa,valid),(alpha,))
    # The actual caches and optimizer use float32. Older torch builds do not
    # consistently promote a Python scalar in where, so cover this dtype too.
    for dtype in (torch.float32,torch.float64):
        n,l,q = [x.to(dtype) for x in (native,legal,expert)]
        aa=alpha.detach().to(dtype).requires_grad_()
        z=path_scores(n,l,q,aa,valid)
        k=pool_kl(z.detach(),z,torch.ones_like(z,dtype=torch.bool)).mean()
        value=-z[:,0].mean()+.01*k
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(aa.grad).all()
    for global_only,use_ev,parameters in ((True,False,1),(False,False,33),(False,True,36)):
        head=Head(global_only,use_ev)
        assert sum(p.numel() for p in head.parameters())==parameters
        hidden=torch.randn(2,32);ev=torch.randn(2,3,4,3)
        aa=head(hidden,ev)
        torch.testing.assert_close(aa,torch.full_like(aa,.3))
        aa.sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    return dict(constant_matches_original=True,zero_exact=True,parent_legal_mass_preserved=True,
        finite_kl_gradient=True,bge_signed_decomposition_exact=True,prefix_evidence_is_conditional_mean=True,
        variable_path_strength_gradient_checked=True,float32_and_float64_losses_verified=True,
        head_parameters=[1,33,36])
