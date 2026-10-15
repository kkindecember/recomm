"""One fixed replacement: arithmetic pooling of expert item probabilities."""
import numpy as np


def softmax(values, axis=-1):
    values=np.asarray(values,dtype=np.float64)
    exp=np.exp(values-values.max(axis=axis,keepdims=True))
    return exp/exp.sum(axis=axis,keepdims=True)


def probability_pool(scores,weights,history_mask,penalty,scale=20.):
    # Only inference-visible quantities are accepted; no target or outcome.
    scores=np.asarray(scores,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    mask=np.asarray(history_mask,dtype=bool)
    assert scores.ndim==2 and weights.shape==(scores.shape[0],) and mask.shape==(scores.shape[1],)
    assert np.isfinite(scores).all() and np.isfinite(weights).all() and (weights>=0).all() and weights.sum()>0
    probabilities=softmax(scale*scores,axis=1)
    contributions=(weights/weights.sum())[:,None]*probabilities
    mixture=contributions.sum(0)
    assert (mixture>0).all() and abs(mixture.sum()-1)<1e-12
    score=np.log(mixture)/scale-float(penalty)*mask
    return score,contributions,probabilities.max(axis=1)


def rank(score,target,eligible):
    if not eligible[target-1]:return 2**30
    value=score[target-1]
    return int(1+np.count_nonzero(eligible&(score>value))+
               np.count_nonzero(eligible[:target-1]&(score[:target-1]==value)))


def top_k(score,eligible,k=10):
    ids=np.flatnonzero(eligible)+1
    assert len(ids)>=k
    threshold=np.partition(score[ids-1],-k)[-k]
    candidates=ids[score[ids-1]>=threshold]
    return candidates[np.lexsort((candidates,-score[candidates-1]))][:k]


def checks():
    rng=np.random.default_rng(20260925)
    scores=rng.uniform(size=(3,19));weights=np.array([.55,.08,.37])
    mask=np.zeros(19,bool);mask[[2,8]]=True;penalty=.043
    new,contrib,peaks=probability_pool(scores,weights,mask,penalty)
    logits=20*scores
    logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None]
    reference=np.logaddexp.reduce(np.log(weights)[:,None]+logp,axis=0)/20-penalty*mask
    np.testing.assert_allclose(new,reference,atol=1e-14,rtol=1e-13)
    for w,sc in [(np.array([0.,1.,0.]),scores),(weights,np.repeat(scores[:1],3,axis=0))]:
        pooled,_,_=probability_pool(sc,w,mask,penalty)
        old=(w[:,None]*sc).sum(0)-penalty*mask
        np.testing.assert_allclose(softmax(20*pooled),softmax(20*old),atol=1e-14,rtol=1e-12)
    shifted,_,_=probability_pool(scores+np.array([1.2,-.7,.3])[:,None],weights,mask,penalty)
    np.testing.assert_allclose(new,shifted,atol=1e-14,rtol=1e-12)
    old=(weights[:,None]*scores).sum(0)-penalty*mask
    geometric=(weights[:,None]*logp).sum(0)-20*penalty*mask
    np.testing.assert_allclose(softmax(20*old),softmax(geometric),atol=1e-14,rtol=1e-12)
    q=np.array([.5,.5,1.,1.,0.]);eligible=np.array([True,False,True,True,True])
    assert top_k(q,eligible,3).tolist()==[3,4,1]
    assert rank(q,1,eligible)==3 and rank(q,2,eligible)==2**30
    return dict(direct_and_logsumexp_agree=True,single_expert_equivalence=True,
                identical_experts_equivalence=True,per_expert_offset_invariance=True,
                geometric_identity_verified=True,filter_and_tie_order_verified=True)
