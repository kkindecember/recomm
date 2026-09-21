"""Independent label invariants, checkpoint probability replay, and saved metric audit."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment.phase20.verify_future_factor_screen import sha, z

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/future_label_specificity_20260917'
PARENT=ROOT/'artifacts/phase20/future_factor_screen_20260917'
ARMS=('uniform_mass','shuffle_identity','score_matched_shuffle')


def main():
    torch.set_num_threads(4);s=json.loads((OUT/'summary.json').read_text());checked=0;domains={}
    for p,h in s['input_sha256'].items():assert sha(ROOT/p)==h,p
    for domain in ('Toys','Beauty'):
        with np.load(OUT/f'{domain}.predictions.npz') as f:a={k:f[k] for k in f.files}
        with np.load(PARENT/f'{domain}.audit_validation_cache.npz') as f:v={k:f[k] for k in f.files}
        with np.load(PARENT/f'{domain}.predictions.npz') as f:parent={k:f[k] for k in f.files}
        np.testing.assert_array_equal(a['users'],v['users']);np.testing.assert_array_equal(a['candidate_ids'],v['candidate_ids'])
        with np.load(ROOT/f'artifacts/phase20/future_horizon_audit_20260917/{domain}.labels.npz') as f:b={k:f[k] for k in f.files}
        valid=b['fit']&(b['positions']>=0)&(b['remaining']>=3);pos=b['positions'][valid];d=b['distances'][valid]
        immediate=np.arange(50)[None]==pos[:,None]
        expected=immediate.astype(float)+((d>0)&(d<=3)&~immediate)/(np.maximum(d,0)+1)
        fitusers=set(b['users'][valid]);np.testing.assert_array_equal(a['in_fit'],[u in fitusers for u in a['users']])
        with np.load(OUT/f'{domain}.training_labels.npz') as f:labels={k:f[k] for k in f.files}
        np.testing.assert_array_equal(labels['actual'],expected)
        for arm in ARMS:
            y=labels[arm];np.testing.assert_array_equal(y[immediate],1)
            np.testing.assert_allclose(y.sum(1),expected.sum(1),atol=1e-12,rtol=0)
            if arm!='uniform_mass':np.testing.assert_array_equal(np.sort(y,axis=1),np.sort(expected,axis=1))
            else:np.testing.assert_allclose(y,immediate+(~immediate)*(expected.sum(1,keepdims=True)-1)/49,atol=1e-12,rtol=0)
        names=list(a['arm_names']);errs={}
        for arm in ARMS:
            state=torch.load(OUT/f'{domain}.{arm}.pt',map_location='cpu');values=[]
            with torch.no_grad():
                for offset in range(0,len(v['users']),256):
                    sl=slice(offset,offset+256);x=torch.as_tensor(v['features'][sl]);delta=[]
                    for j in range(2):
                        h=torch.tanh(F.linear(x,state[f'heads.{j}.0.weight'],state[f'heads.{j}.0.bias']))
                        delta.append((2*torch.tanh(F.linear(h,state[f'heads.{j}.2.weight'],state[f'heads.{j}.2.bias'])).squeeze(-1)).numpy().astype(np.float64))
                    prior=v['prior'][sl];root=prior+np.log1p(np.sqrt(1+np.exp(-prior)))
                    probability=1/(1+np.exp(-root-delta[0]))/(1+np.exp(-root-delta[1]))
                    values.append(np.log(probability)-np.log1p(-probability))
            raw=z(np.concatenate(values));errs[arm]=0.
            for fusion in ('current','mean'):
                actual=raw+v['cf_'+fusion];saved=a['scores'][names.index(arm+'.'+fusion)]
                err=float(np.max(np.abs(actual-saved)));errs[arm]=max(errs[arm],err);assert err<1e-11
        for arm in ('soft_future','next_only','frozen'):
            for fusion in ('current','mean'):
                name=arm+'.'+fusion
                np.testing.assert_array_equal(a['scores'][names.index(name)],parent['scores'][list(parent['arm_names']).index(name)])
        for j,name in enumerate(names):
            score=a['scores'][j];pos=a['positions'];ix=np.maximum(pos,0);gold=score[np.arange(len(pos)),ix,None]
            rr=np.where(pos>=0,1+(score>gold).sum(1)+((score==gold)&(np.arange(50)[None]<ix[:,None])).sum(1),51)
            np.testing.assert_array_equal(rr,a['ranks'][j])
            for group,mask in [('all',np.ones(len(rr),bool)),('tail',a['tail_mask']),('non_tail',~a['tail_mask']),('nonfit',~a['in_fit'])]:
                expected=s['domains'][domain]['metrics'][name][group];assert int(mask.sum())==expected['users']
                for k in (5,10,20,50):
                    for metric,value in [(f'H{k}',(rr[mask]<=k).mean()),(f'N{k}',((rr[mask]<=k)/np.log2(rr[mask]+1)).mean())]:
                        assert abs(value-expected[metric])<1e-14;checked+=1
        m=s['domains'][domain]['metrics'];truth=m['soft_future.mean']['all']['N10'];base=m['next_only.mean']['all']['N10'];local=m['score_matched_shuffle.mean']['all']['N10']
        domains[domain]=dict(checkpoint_score_errors=errs,local_shuffle_retains_gain_fraction=(local-base)/(truth-base))
    result=dict(state='VERIFIED',input_hashes=len(s['input_sha256']),metrics_checked=checked,
        checkpoint_count=6,domains=domains,source_sha256=sha(Path(__file__)),test_read=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
