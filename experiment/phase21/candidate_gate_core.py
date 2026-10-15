"""Frozen-expert candidate calibration. Observable feature functions accept no target.

Original expert/gate recipe: LIME-Rec. Contribution definitions follow the
project's expert_evidence_alignment_core.Attribution; this module changes only
the calibration head, not the experts or GRAM's generation mechanism.
"""
import hashlib
from collections import defaultdict

import numpy as np
import torch

ARMS = ('user', 'score', 'evidence', 'shuffled', 'global')
USER_NAMES = (['log_history_length'] + [f'{stat}_{e}' for stat in ('confidence','margin','std','contamination')
              for e in ('sas','cf','text')] + ['history_ratio','w_sas','w_cf','w_text','penalty'])
CAND_NAMES = ([f'{stat}_{e}' for stat in ('score','margin_to_reference','reciprocal_rank','top10')
              for e in ('sas','cf','text')] + ['fused','fused_margin','fused_rr','gap10_11','gap1_10','log_frequency'])
STRUCT_NAMES = ['sum','absolute_mass','positive_fraction','negative_fraction','max_absolute_share',
                'effective_support','recent_absolute_share','cancellation','no_support']
EVID_NAMES = [f'{e}_{side}_{s}' for e in ('cf','text') for side in ('absolute','relative') for s in STRUCT_NAMES]
DU, DC, DE = len(USER_NAMES), len(CAND_NAMES), len(EVID_NAMES)
WIDTH = DU + DC + DE


def fold(user, domain):
    return int.from_bytes(hashlib.sha256(f'candidate-gate-20261010|{domain}|{user}'.encode()).digest()[:8], 'big') % 5


def cohort(s):
    return ('previous512' if s['previously_evaluated'] else 'primary'
            if s['training_holdout_role'] == 'none' else 'seen_fit_user_later_event')


def donor_indices(samples, domain, training):
    # Same split/cohort; sorted neighbouring lengths, then one-step derangement.
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        split = fold(s['user_id'], domain) if training else cohort(s)
        groups[split].append(i)
    result = np.empty(len(samples), np.int32)
    for ix in groups.values():
        ix.sort(key=lambda i:(len(samples[i]['full_history']), samples[i]['user_id']))
        assert len(ix) > 1
        result[ix] = ix[1:] + ix[:1]
    assert all(samples[i]['user_id'] != samples[j]['user_id'] for i,j in enumerate(result))
    return result


def user_features(experts, history, weights, penalty):
    # Torch's topk tie rule matches the original gate implementation on this CPU;
    # full validation's original weights themselves are always loaded, not refit.
    s = torch.from_numpy(np.array(experts, copy=True))
    top2 = s.topk(2, dim=1).values
    mask = torch.zeros(s.shape[1], dtype=torch.bool)
    mask[np.asarray(history, int)-1] = True
    f = torch.cat([torch.tensor([np.log1p(len(history))/5], dtype=torch.float32),
                   top2[:,0]-s.mean(1), top2[:,0]-top2[:,1], s.std(1),
                   mask[s.topk(5,dim=1).indices].float().mean(1), mask.float().mean()[None]])
    return np.concatenate([f.numpy(), weights, [penalty]]).astype(np.float32)


def ranking(scores, eligible):
    ids = np.flatnonzero(eligible)
    order = ids[np.lexsort((ids, -scores[ids]))]
    rank = np.full(len(scores), len(scores)+1, dtype=np.int32)
    rank[order] = np.arange(1,len(order)+1)
    return order, rank


def candidates(experts, fused, history, counts):
    eligible = np.ones(len(fused), bool)
    eligible[np.asarray(history[-20:], int)-1] = False
    orders, ranks = zip(*(ranking(v, eligible) for v in [*experts, fused]))
    ids = np.unique(np.concatenate([o[:50] for o in orders]))
    ref = int(orders[3][9])
    s = experts[:,ids].T
    rr = np.stack([r[ids] for r in ranks[:3]], axis=1)
    x = np.column_stack([s, s-experts[:,ref], 1./rr, (rr<=10).astype(float),
        fused[ids], fused[ids]-fused[ref], 1./ranks[3][ids],
        np.full(len(ids),fused[ref]-fused[orders[3][10]]),
        np.full(len(ids),fused[orders[3][0]]-fused[ref]), np.log1p(counts[ids+1])])
    assert len(ids)<=200 and eligible[ids].all() and x.shape==(len(ids),DC)
    return ids, ref, x.astype(np.float32), ranks[3]


def structure(c):
    total = c.sum(1); a = np.abs(c); mass = a.sum(1)
    den = np.maximum(mass, 1e-12); threshold = np.maximum(mass*1e-6, 1e-12)[:,None]
    result = np.column_stack([total,mass,(c>threshold).mean(1),(c < -threshold).mean(1),
        a.max(1)/den, mass**2/np.maximum((c*c).sum(1),1e-24)/c.shape[1],
        a[:,-5:].sum(1)/den, np.where(mass>1e-12,1-np.abs(total)/den,0), (mass<=1e-12)])
    return result


class Evidence:
    def __init__(self, item_ids, embeddings, cf_model):
        from experiment.phase21.expert_evidence_alignment_core import Attribution
        self.reference = Attribution(item_ids, embeddings, cf_model)
        self.emb = self.reference.emb
        self.unit = self.reference.unit
        self.cf = self.reference.cf

    def contributions(self, ids, history):
        # ids are zero-based catalog indices; histories retain project 1-based IDs.
        hi = np.asarray(history, int)-1
        w = np.exp(-.1*np.arange(len(history)-1,-1,-1)).astype(np.float32)
        mean = (self.emb[hi]*w[:,None]).sum(0)/max(w.sum(),1e-8)
        denom = float(max(w.sum(),1e-8))*float(np.linalg.norm(mean)+1e-8)
        text = (self.unit[ids].astype(np.float64) @ self.emb[hi].astype(np.float64).T)*w[None,:]/denom
        cf = np.zeros((len(ids),len(history)),np.float64)
        offset = max(0,len(history)-20)
        for j,h in enumerate(history[offset:]):
            mapping = self.cf.get(h,{})
            cf[:,offset+j] = np.fromiter((mapping.get(int(i)+1,0.) for i in ids),float,len(ids))*(j+1)/min(len(history),20)
        return cf, text

    def features(self, ids, ref, history):
        values = self.contributions(np.append(ids,ref),history)
        return np.concatenate([z for c in values for z in (structure(c[:-1]),structure(c[:-1]-c[-1:]))],axis=1).astype(np.float32)

    def check(self, ids, ref, history):
        c,t = self.contributions(np.append(ids[:2],ref),history)
        for j,a in enumerate(ids[:2]):
            r = self.reference.pair(int(a)+1,int(ref)+1,history)
            np.testing.assert_allclose(c[[j,-1]].T,r['cf'],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(t[[j,-1]].T,r['text'],atol=1e-9,rtol=1e-9)


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(WIDTH,32)
        self.output = torch.nn.Linear(32,3)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self,x,w0,eta=.1):
        z = self.output(torch.relu(self.input(x)))
        # Subtract a numerically identical zero-logit reference. This implements
        # eta*(softmax(log(w0)+z)-w0) to float32 precision with exact z=0 recovery.
        base = torch.softmax(torch.log(w0),dim=-1)
        return eta*(torch.softmax(torch.log(w0)+z,dim=-1)-base) if eta else torch.zeros_like(z)


def compressed_loss(base, delta, mask, outside, target_base, target_pos, lam=.2):
    """Exact full-catalog CE/KL from shifted candidate logits and outside mass."""
    neg = torch.full_like(base, -float('inf'))
    old = torch.where(mask,base,neg)
    new = torch.where(mask,base+20*delta,neg)
    logz0 = torch.logaddexp(torch.logsumexp(old,1),outside)
    logz = torch.logaddexp(torch.logsumexp(new,1),outside)
    iy = target_pos.clamp_min(0)
    ydelta = delta.gather(1,iy[:,None]).squeeze(1)*(target_pos>=0)
    ce = logz-target_base-20*ydelta
    kl = logz-logz0-(torch.exp(old-logz0[:,None])*20*delta).sum(1)
    return (ce+lam*kl).mean(), ce.mean(), kl.mean()


def utilities(ranks,k=10):
    return np.where(ranks<=k,1/np.log2(ranks+1),0.)


def tests():
    torch.manual_seed(51)
    w=torch.softmax(torch.randn(3,1,3),-1);x=torch.randn(3,8,WIDTH)
    m=Gate();assert torch.equal(m(x,w),torch.zeros(3,8,3))
    with torch.no_grad():m.output.weight.normal_()
    dw=m(x,w);assert float(dw.abs().sum(-1).max())<=.2+1e-6
    assert torch.equal(m(x,w,eta=0),torch.zeros(3,8,3))
    # Two targets within candidate sets and one outside. Full loss gradients,
    # including outside-target rows, must agree with the compressed calculation.
    full=torch.randn(3,13,dtype=torch.float64); ids=torch.tensor([[1,4,8],[0,2,7],[1,5,10]])
    delta=torch.randn(3,3,dtype=torch.float64,requires_grad=True)*.02
    mask=torch.ones(3,3,dtype=torch.bool); targets=torch.tensor([4,12,10]); pos=torch.tensor([1,-1,2])
    outside=full.clone();outside.scatter_(1,ids,-float('inf'))
    a=compressed_loss(full.gather(1,ids),delta,mask,outside.logsumexp(1),full[torch.arange(3),targets],pos)[0]
    changed=full.scatter_add(1,ids,20*delta);p=full.softmax(1)
    b=torch.nn.functional.cross_entropy(changed,targets)+.2*(p*(full.log_softmax(1)-changed.log_softmax(1))).sum(1).mean()
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(a,delta,retain_graph=True)[0];gb=torch.autograd.grad(b,delta)[0]
    torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
    return {'zero_initialization_exact':True,'eta_zero_exact':True,'loss_and_gradient_dense_equivalence':True,
            'outside_targets_retained':True,'bounded_weight_change':True}
