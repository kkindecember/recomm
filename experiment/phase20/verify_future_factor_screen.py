"""Independent checkpoint/metric replay, with explicit probability formulas."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/future_factor_screen_20260917'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def z(x):return (x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),1e-6)


def main():
    torch.set_num_threads(4)
    s=json.loads((OUT/'summary.json').read_text());cfg=json.loads((ROOT/'experiment/phase20/gacr_fusion_train_v1.json').read_text())
    for p,h in s['input_sha256'].items():assert sha(ROOT/p)==h,p
    domains={};checks=0;hashes={}
    for domain,config in cfg['domains'].items():
        with np.load(OUT/f'{domain}.predictions.npz') as f:a={k:f[k] for k in f.files}
        with np.load(ROOT/f'artifacts/phase20/gacr_fusion_train_v1/{domain}/B1.validation_scores.npz') as f:b={k:f[k] for k in f.files}
        order=np.argsort(-b['residual_scores'],axis=1,kind='stable')
        x=np.empty((*order.shape,6),np.float32);lookup={u:i for i,u in enumerate(b['users'])};seen=set()
        for row in map(json.loads,(ROOT/config['features']).open()):
            u=row['user_id'];assert u not in seen;seen.add(u);i=lookup[u]
            np.testing.assert_array_equal(row['union_ids'][:50],b['candidate_ids'][i]);x[i]=row['features'][:50]
        assert seen==set(lookup)
        x=np.take_along_axis(x,order[:,:,None],1);prior=z(np.take_along_axis(b['residual_scores'],order,1))-np.log(49)
        np.testing.assert_array_equal(a['users'],b['users'])
        np.testing.assert_array_equal(a['candidate_ids'],np.take_along_axis(b['candidate_ids'],order,1))
        names=list(a['arm_names']);errors={}
        for arm in ('next_only','soft_future','flat_three','conditional'):
            p=OUT/f'{domain}.{arm}.pt';hashes[str(p.relative_to(ROOT))]=sha(p);state=torch.load(p,map_location='cpu')
            values=[]
            with torch.no_grad():
                for offset in range(0,len(prior),256):
                    sl=slice(offset,offset+256);features=torch.as_tensor(x[sl]);d=[]
                    for j in range(2):
                        h=torch.tanh(F.linear(features,state[f'heads.{j}.0.weight'],state[f'heads.{j}.0.bias']))
                        d.append((2*torch.tanh(F.linear(h,state[f'heads.{j}.2.weight'],state[f'heads.{j}.2.bias'])).squeeze(-1)).numpy().astype(np.float64))
                    root=prior[sl]+np.log1p(np.sqrt(1+np.exp(-prior[sl])))
                    if arm=='flat_three':
                        prob=1/(1+np.exp(-root));lnext=2*np.log(prob)-np.log1p(-prob)+d[0];llater=np.log(prob)+d[1]
                        pred=lnext-np.logaddexp(0,llater)
                    else:
                        probability=1/(1+np.exp(-root-d[0]))/(1+np.exp(-root-d[1]));pred=np.log(probability)-np.log1p(-probability)
                    values.append(pred)
            raw=z(np.concatenate(values));errors[arm]=float(np.max(np.abs(raw-a['scores'][names.index(arm+'.raw')])));assert errors[arm]<1e-11
            for fusion in ('raw','current','mean'):
                actual=raw if fusion=='raw' else raw+a['scores'][names.index('frozen.'+fusion)]-z(prior)
                np.testing.assert_allclose(actual,a['scores'][names.index(arm+'.'+fusion)],atol=1e-11,rtol=0)
        for j,arm in enumerate(names):
            score=a['scores'][j];pos=a['positions'];ix=np.maximum(pos,0);target=score[np.arange(len(pos)),ix,None]
            rr=np.where(pos>=0,1+(score>target).sum(1)+((score==target)&(np.arange(50)[None]<ix[:,None])).sum(1),51)
            np.testing.assert_array_equal(rr,a['ranks'][j])
            for group,mask in [('all',np.ones(len(pos),bool)),('tail',a['tail_mask']),('non_tail',~a['tail_mask'])]:
                m=s['domains'][domain]['metrics'][arm][group];assert m['users']==int(mask.sum())
                for k in (5,10,20,50):
                    for metric,v in [(f'H{k}',(rr[mask]<=k).mean()),(f'N{k}',((rr[mask]<=k)/np.log2(rr[mask]+1)).mean())]:
                        assert abs(m[metric]-v)<1e-14;checks+=1
        with np.load(ROOT/f'artifacts/phase20/future_horizon_audit_20260917/{domain}.labels.npz') as f:
            fitusers=set(f['users'][f['fit']&(f['positions']>=0)&(f['remaining']>=3)])
        in_fit=np.asarray([u in fitusers for u in a['users']]);comparisons={}
        for group,mask in [('all',np.ones(len(in_fit),bool)),('fit',in_fit),('nonfit',~in_fit)]:
            aa=a['ranks'][names.index('soft_future.mean'),mask];bb=a['ranks'][names.index('next_only.mean'),mask]
            n=lambda rr:float(((rr<=10)/np.log2(rr+1)).mean())
            comparisons[group]=dict(users=len(aa),relative_N10=n(aa)/n(bb)-1,
                hit_gained=int(((aa<=10)&(bb>10)).sum()),hit_lost=int(((aa>10)&(bb<=10)).sum()))
        domains[domain]=dict(checkpoint_score_errors=errors,soft_vs_next_only=comparisons)
        np.savez_compressed(OUT/f'{domain}.audit_validation_cache.npz',features=x,prior=prior,users=a['users'],
            candidate_ids=a['candidate_ids'],positions=a['positions'],tail_mask=a['tail_mask'],
            cf_current=a['scores'][names.index('frozen.current')]-z(prior),cf_mean=a['scores'][names.index('frozen.mean')]-z(prior))
        hashes[str((OUT/f'{domain}.audit_validation_cache.npz').relative_to(ROOT))]=sha(OUT/f'{domain}.audit_validation_cache.npz')
    result=dict(state='VERIFIED',input_hashes=len(s['input_sha256']),metrics_checked=checks,
        audit_source_sha256=sha(Path(__file__)),output_sha256=hashes,domains=domains,test_read=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
