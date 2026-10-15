"""Ordinary calibration and request-benefit controls; no target input at prediction."""
import hashlib
import numpy as np
import torch
from experiment.phase20.catalog_scope_core import ranks, utilities
from experiment.phase20.source_competition_core import ResidualRanker, row_losses, target_positions

FEATURES = ('native', 'update', 'sasrec', 'itemcf', 'semantic', 'lime',
            'native_reciprocal_rank', 'mixed_gap10', 'log_frequency', 'log_history_length')


def fold_ids(users, domain):
    return np.array([int.from_bytes(hashlib.sha256(
        f'20260922|clean-benefit-fold|{domain}|{u}'.encode()).digest()[:8], 'big') % 5 for u in users])


def make_features(ids, native, mixed, experts, fused, counts, length, gate_features, history):
    """Single query, fixed native candidate list. Does not accept its target."""
    ids = np.asarray(ids, dtype=np.int64)
    native, mixed = np.asarray(native), np.asarray(mixed)
    valid = ~np.isin(ids, np.asarray(history[-20:], dtype=np.int64))
    assert valid.sum() >= 10
    order = np.lexsort((ids, -native)); nr = np.empty(len(ids), dtype=int); nr[order] = np.arange(1,len(ids)+1)
    boundary = np.sort(mixed[valid])[-10]
    x = np.column_stack((native, mixed-native, experts[:,ids-1].T, fused[ids-1],
                         1/nr, mixed-boundary, np.log1p(counts[ids]), np.full(len(ids),np.log1p(length))))
    assert x.shape == (50,10) and np.isfinite(x).all()
    ni=np.lexsort((ids[valid],-native[valid])); mi=np.lexsort((ids[valid],-mixed[valid]))
    ns=native[valid][ni]; ms=mixed[valid][mi]; delta=(mixed-native)[valid]
    overlap=len(set(ids[valid][ni[:10]]) & set(ids[valid][mi[:10]]))/10
    query=np.concatenate((gate_features,[ns[0]-ns[1],ns[0]-ns[9],ms[0]-ms[1],ms[0]-ms[9],
        delta.mean(),delta.std(),delta.min(),delta.max(),overlap,valid.mean()]))
    return x.astype(np.float32),query.astype(np.float32),valid


def ranking(data, scores):
    return ranks(np.where(data['valid'],scores,-np.inf),data['ids'],data['targets'])


def fit_ridge(x, y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    mean=x.mean(0); std=np.maximum(x.std(0),1e-5); z=(x-mean)/std
    intercept=float(y.mean()); coef=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1]),z.T@(y-intercept)/len(z))
    return dict(mean=mean,std=std,coef=coef,intercept=intercept)


def predict_ridge(model, x):
    return (x-model['mean'])/model['std']@model['coef']+model['intercept']


class LinearCalibration(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.network=torch.nn.Linear(10,1)
        torch.nn.init.zeros_(self.network.weight);torch.nn.init.zeros_(self.network.bias)
    def forward(self,x,prior):
        return prior+2*torch.tanh(self.network(x).squeeze(-1))


def new_head(kind,seed):
    torch.manual_seed(seed)
    return LinearCalibration() if kind=='linear' else ResidualRanker(n_features=10,hidden=32,bound=2.)


def fit_head(data, indices, kind, seed):
    indices=np.asarray(indices)
    pos=target_positions(np.where(data['valid'][indices],data['ids'][indices],0),data['targets'][indices])
    used=indices[pos>=0]
    assert len(used)>0
    valid=data['valid'][used]; raw=data['x'][used]
    mean=raw[valid].mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(raw[valid].std(0,dtype=np.float64),1e-5).astype(np.float32)
    x=torch.from_numpy((raw-mean)/std); v=torch.from_numpy(valid); p=torch.from_numpy(pos[pos>=0])
    prior=torch.from_numpy((10*data['mixed'][used]).astype(np.float32))
    model=new_head(kind,seed); opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    curve=[]
    for step in range(30):
        opt.zero_grad(); total=0.
        for start in range(0,len(used),128):
            sl=slice(start,start+128)
            loss=row_losses(model(x[sl],prior[sl]),v[sl],p[sl]).sum()/len(used)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();curve.append(total)
    return dict(state_dict=model.state_dict(),mean=mean,std=std,kind=kind,seed=seed,used=used,
                fit_indices=indices,curve=curve,optimizer_steps=30)


def predict_head(saved, x, mixed, chunk=128):
    model=new_head(saved['kind'],saved['seed']);model.load_state_dict(saved['state_dict']);model.eval()
    output=[]
    with torch.no_grad():
        for start in range(0,len(x),chunk):
            sl=slice(start,start+chunk)
            z=torch.from_numpy(((x[sl]-saved['mean'])/saved['std']).astype(np.float32))
            p=torch.from_numpy((10*mixed[sl]).astype(np.float32))
            output.append(model(z,p).numpy())
    return np.concatenate(output)


def benefit_labels(data):
    return utilities(ranking(data,data['mixed']))-utilities(ranking(data,data['native']))
