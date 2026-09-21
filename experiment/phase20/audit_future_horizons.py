"""Train-only temporal-label audit. Future occurrence is NOT a false-negative claim."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/future_horizon_audit_20260917'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False); start=time.monotonic(); torch.set_num_threads(4)
    hashes={}; results={}
    def use(path):
        hashes[str(path.relative_to(ROOT))]=sha(path);return path
    def status(state,**kw):
        p=OUT/'status.tmp';p.write_text(json.dumps(dict(state=state,elapsed_seconds=time.monotonic()-start,
            training_steps=0,gpu_used=False,test_read=False,**kw),indent=2)+'\n');p.replace(OUT/'status.json')
    status('RUNNING');use(Path(__file__))
    cfg=json.loads(use(ROOT/'experiment/phase20/gacr_fusion_train_v1.json').read_text())
    try:
        for domain,config in cfg['domains'].items():
            status('RUNNING',domain=domain)
            folder=ROOT/f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
            seqs={}; rows=[]; times=[]
            for row in map(json.loads,use((ROOT/config['validation']).with_name('train.jsonl')).open()):
                u=row['user_id']
                if u not in seqs:
                    assert len(row['history'])==1;seqs[u]=list(row['history'])
                assert seqs[u][-20:]==row['history']
                times.append(len(seqs[u]));seqs[u].append(row['target']);rows.append(row)
            with np.load(use(folder/'training_cache.npz')) as f: a={k:f[k] for k in f.files}
            records=list(map(json.loads,use(folder/'train_candidates.jsonl').open()))
            n=len(records);assert n==len(a['positions'])
            distances=np.full((n,50),-1,np.int64);remaining=np.zeros(n,np.int64)
            users=[]
            for i,row in enumerate(records):
                ix=row['train_index'];orig=rows[ix];u=orig['user_id'];t=times[ix]
                assert row['user_id']==u and row['history']==orig['history'] and row['target']==orig['target']
                np.testing.assert_array_equal(a['candidate_ids'][i],row['union_ids'][:50])
                assert row['target_position']==a['positions'][i]
                seq=seqs[u];remaining[i]=len(seq)-t-1;users.append(u)
                first={}
                for offset,item in enumerate(seq[t+1:],1):first.setdefault(item,offset)
                for j,item in enumerate(a['candidate_ids'][i]):distances[i,j]=first.get(int(item),-1)
                if a['positions'][i]>=0:distances[i,a['positions'][i]]=-1
            # Evaluate exact historical 6->16->1 bounded residual without importing its implementation.
            state=torch.load(use(folder/'B1.pt'),map_location='cpu')
            assert set(state)=={'network.0.weight','network.0.bias','network.2.weight','network.2.bias'},state.keys()
            with torch.no_grad():
                x=torch.as_tensor(a['features'])
                hidden=torch.tanh(torch.nn.functional.linear(x,state['network.0.weight'],state['network.0.bias']))
                residual=.2*torch.tanh(torch.nn.functional.linear(hidden,state['network.2.weight'],state['network.2.bias'])).squeeze(-1)
                g=(torch.as_tensor(a['base'])+residual).numpy().astype(np.float64)
            g=(g-g.mean(1,keepdims=True))/np.maximum(g.std(1,keepdims=True),1e-6)
            pos=np.maximum(a['positions'],0); margins=g-g[np.arange(n),pos,None]
            grad=1/(1+np.exp(-np.clip(margins,-60,60)));grad[np.arange(n),pos]=0
            valid=a['fit']&(a['positions']>=0)
            users=np.asarray(users);domain_result={}
            for horizon in (1,3,5,'all'):
                eligible=valid if horizon=='all' else valid&(remaining>=horizon)
                future=(distances>0) if horizon=='all' else ((distances>0)&(distances<=horizon))
                future&=(np.arange(50)[None]!=pos[:,None])
                count=future.sum(1); mass=(grad*future).sum(1)/np.maximum(grad.sum(1),1e-12)
                within=eligible&(count>0)
                pairs=int(future[eligible].sum());allpairs=int(eligible.sum()*49)
                future_grad=float((grad*future)[eligible].sum());total_grad=float(grad[eligible].sum())
                user_means=[]
                for u in np.unique(users[eligible]):user_means.append(float((count[eligible&(users==u)]>0).mean()))
                domain_result[str(horizon)]=dict(covered_fit_prefixes=int(eligible.sum()),users=int(len(user_means)),
                    any_future_fraction=float((count[eligible]>0).mean()),user_macro_any_fraction=float(np.mean(user_means)),
                    mean_future_negatives=float(count[eligible].mean()),pair_fraction=pairs/allpairs,
                    score_margin_gradient_mass_fraction=future_grad/total_grad,
                    per_pair_gradient_ratio=(future_grad/pairs)/((total_grad-future_grad)/(allpairs-pairs)),
                    rows_with_future_mean_gradient_fraction=float(mass[within].mean()),
                    future_outranks_immediate_fraction=float(((margins>0)&future)[eligible].sum()/pairs),
                    candidate_future_pair_count=pairs)
            np.savez_compressed(OUT/f'{domain}.labels.npz',users=users,distances=distances,remaining=remaining,
                fit=a['fit'],positions=a['positions'],candidate_ids=a['candidate_ids'])
            results[domain]=domain_result
        for p,h in hashes.items():assert sha(ROOT/p)==h,p
        summary=dict(domains=results,input_sha256=hashes,
            interpretation='Descriptive train-only evidence; later interactions are valid immediate-task negatives. '
              'Gradient refers to derivative wrt pair margin, NOT parameter gradient or causal harm. '
              'Finite horizons exclude right-censored prefixes; all includes variable remaining horizons.',
            validation_read=False,test_read=False,training_steps=0)
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');status('COMPLETED')
        print(json.dumps(results,indent=2))
    except Exception as e:status('FAILED',error=repr(e));raise


if __name__=='__main__':main()
