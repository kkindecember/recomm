"""Small recommendation-trained text projections over frozen BGE vectors.

Train against observed next items and sampled alternatives; unobserved items
are not known dislikes. Hard-candidate sampling uses only frozen SASRec scores.
No new model-novelty claim is made for this bilinear adapter or residual CE.
"""
import math
import numpy as np
import torch
from torch import nn

TRAIN_ARMS=('residual_hard','text_hard','residual_random')


def top50(row):
    row=np.asarray(row)
    boundary=np.partition(row,len(row)-50)[len(row)-50]
    higher=np.flatnonzero(row>boundary)
    equal=np.flatnonzero(row==boundary)[:50-len(higher)]
    chosen=np.r_[higher,equal]
    return chosen[np.lexsort((chosen,-row[chosen]))]+1


def training_pool(proposal,target):
    """Supervised training only. Evaluation proposals never call this function."""
    choices=[int(target)]+[int(i) for i in proposal if int(i)!=int(target)]
    ids=np.ones(51,np.int64)
    valid=np.zeros(51,bool)
    ids[:len(choices)]=choices
    valid[:len(choices)]=True
    assert len(choices)==len(set(choices)) and 50<=len(choices)<=51
    return ids,valid


class TextAdapter(nn.Module):
    def __init__(self,user_width,item_width,rank=32,bound=2.):
        super().__init__()
        self.query=nn.Linear(user_width,rank)
        self.item=nn.Parameter(torch.zeros(rank,item_width))
        self.rank,self.bound=rank,float(bound)

    def semantic_query(self,user):
        return torch.tanh(self.query(user))@self.item/math.sqrt(self.rank)

    def forward(self,user,item_embeddings):
        # Associativity avoids projecting all51 candidate vectors independently.
        query=self.semantic_query(user)
        return self.bound*torch.tanh((query[:,None,:]*item_embeddings).sum(-1))


def objective(delta,sas,valid,arm,regularization=.01):
    logits=delta+(20*sas if arm!='text_hard' else 0.)
    logits=logits.masked_fill(~valid,-torch.inf)
    ce=nn.functional.cross_entropy(logits,torch.zeros(len(delta),dtype=torch.long))
    norm=delta[valid].square().mean()
    return ce+regularization*norm


def numpy_delta(state,user,items,bound=2.):
    query=np.tanh(user@state['query.weight'].T+state['query.bias'])
    # Independent direct candidate projection, rather than query-side factoring.
    projected=items@state['item'].T
    values=np.einsum('br,bkr->bk',query,projected)/math.sqrt(state['item'].shape[0])
    return bound*np.tanh(values)


def tests():
    rng=np.random.default_rng(2153)
    torch.manual_seed(0)
    model=TextAdapter(7,5,rank=3)
    u=torch.tensor(rng.normal(size=(4,7)),dtype=torch.float32)
    e=torch.tensor(rng.normal(size=(4,6,5)),dtype=torch.float32)
    np.testing.assert_array_equal(model(u,e).detach().numpy(),np.zeros((4,6)))
    sas=torch.tensor(rng.random((4,6)),dtype=torch.float32)
    valid=torch.ones((4,6),dtype=torch.bool)
    valid[:,-1]=False
    value=objective(model(u,e),sas,valid,'residual_hard')
    independent=torch.stack([torch.logsumexp((20*sas[j])[valid[j]],0)-20*sas[j,0] for j in range(4)]).mean()
    torch.testing.assert_close(value,independent)
    value.backward()
    assert model.item.grad.abs().sum()>0
    with torch.no_grad():model.item.normal_(std=.2)
    actual=model(u,e)
    state={k:v.detach().numpy() for k,v in model.state_dict().items()}
    np.testing.assert_allclose(numpy_delta(state,u.numpy(),e.numpy()),actual.detach().numpy(),atol=5e-7,rtol=2e-5)
    assert actual.abs().max()<=2.
    for j in range(4):torch.testing.assert_close(model(u[j:j+1],e[j:j+1])[0],actual[j])
    row=np.ones(80)
    np.testing.assert_array_equal(top50(row),np.arange(1,51))
    inside,mask=training_pool(top50(row),2)
    outside,mask2=training_pool(top50(row),75)
    assert inside[0]==2 and mask.sum()==50 and outside[0]==75 and mask2.sum()==51
    assert len(set(inside[mask]))==50 and len(set(outside[mask2]))==51
    return dict(state='PASSED',zero_adapter_exact=True,residual_CE_independent=True,
        first_step_gradient=True,numpy_direct_projection_matches=True,bounded_correction=True,
        batch_invariance=True,deterministic_target_free_top50=True,
        supervised_training_pool_deduplicated=True,targets_absent_from_forward=True)
