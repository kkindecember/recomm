"""Independent NumPy forward/sort and complete rank-summary verification.

Run after candidate_gate_screen completes. Does not import its feature/model
implementation; saved PyTorch weights are only deserialized on CPU.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/candidate_gate_20261010_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
SOURCE=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def softmax(x):
    x=x-x.max(axis=-1,keepdims=True);e=np.exp(x);return e/e.sum(axis=-1,keepdims=True)


def main():
    assert json.loads((OUT/'status.json').read_text())['state']=='COMPLETE'
    cfg=json.loads((OUT/'config.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    du,dc,de=map(lambda k:len(cfg[k]),['user_names','candidate_names','evidence_names'])
    original_manifest=json.loads((SOURCE/'manifest.json').read_text())['sha256']
    manifest=json.loads((OUT/'manifest.json').read_text())['sha256'];checked=0;source_matches=0
    for rel,digest in manifest.items():
        assert sha(ROOT/rel)==digest,rel;checked+=1
        if rel in original_manifest:assert digest==original_manifest[rel];source_matches+=1
    with (ROOT/'artifacts/phase21/full_opportunity_counts_20260925_v1/per_user.csv').open() as f:original=list(csv.DictReader(f))
    rows=[];summary_checks=0;exact=0;maximum_delta=0.;precision_sensitive=0
    for domain in ('Toys','Beauty'):
        folder=OUT/domain/'validation';source=FULL/domain
        a={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ('user','candidate','evidence','shuffled','ids','mask','scores','original_rank')}
        saved=np.load(OUT/domain/'ranks.npz');samples=json.loads((source/'samples.json').read_text())
        full=np.load(source/'fused.npy',mmap_mode='r');experts=np.load(source/'experts.npy',mmap_mode='r')
        w=np.load(source/'gate_prediction.npz')['weights'];records=json.loads((folder/'users.json').read_text())
        assert all(s['user_id']==r['user_id'] for s,r in zip(samples,records))
        cohorts=np.array([r['cohort'] for r in records]);n=len(samples)
        expected=np.array([int(r['LIME']) for r in original if r['domain']==domain])
        # Treat the two reserved absent-rank sentinels equivalently.
        np.testing.assert_array_equal(np.minimum(expected,full.shape[1]+1),saved['original'])
        for co,result in summary[domain]['results'].items():
            mask=np.ones(n,bool) if co=='all' else cohorts==co
            assert int(mask.sum())==result['users']
            for key,metrics in result['metrics'].items():
                rr=saved[key][mask]
                for metric,value in metrics.items():
                    actual=float(np.mean([1/np.log2(int(v)+1) if v<=10 else 0. for v in rr])) if metric=='N10' else float((rr<=int(metric[1:])).mean())
                    assert abs(actual-value)<1e-12,(domain,co,key,metric);summary_checks+=1
        for key in saved.files:
            if key=='original':continue
            arm=key.rsplit('_',1)[0];p=OUT/domain/(key.replace('_0','_seed0').replace('_1','_seed1').replace('_2','_seed2')+'.pt')
            checkpoint=torch.load(p,map_location='cpu');state={k:v.numpy() for k,v in checkpoint['state_dict'].items()}
            fit_users=set(checkpoint['fit_users']);assert len(fit_users)==summary[domain]['train']['users']
            assert not fit_users&{s['user_id'] for s,c in zip(samples,cohorts) if c=='primary'}
            changes=np.flatnonzero(saved[key]!=saved['original'])
            ix=sorted(set([0,n//2,n-1]+changes[:4].tolist()+changes[-4:].tolist()))
            for j in ix:
                cm=a['mask'][j];ids=a['ids'][j,cm]
                assert len(set(ids))==len(ids) and not(set(ids+1)&set(samples[j]['history']))
                m=1 if arm in ('user','global') else len(ids)
                x=np.zeros((m,du+dc+de),np.float32);x[:,:du]=a['user'][j]
                if m!=1:x[:,du:du+dc]=a['candidate'][j,cm]
                if arm in ('evidence','shuffled'):x[:,du+dc:]=a[arm][j,cm]
                x=np.clip((x-checkpoint['mean'])/checkpoint['std'],-10,10)
                hidden=np.maximum(x@state['input.weight'].T+state['input.bias'],0)
                z=hidden@state['output.weight'].T+state['output.bias']
                dw=cfg['eta']*(softmax(np.log(w[j])[None,:]+z)-softmax(np.log(w[j])[None,:]))
                new=np.array(full[j],copy=True)
                if arm=='global':new+=(experts[j]*dw.T).sum(0)
                else:new[ids]+=(a['scores'][j,cm]*dw).sum(1)
                delta=float(np.max(np.abs(new-full[j])));maximum_delta=max(maximum_delta,delta)
                assert delta<=cfg['eta']+1e-6
                if arm!='global':
                    outside=np.ones(new.size,bool);outside[ids]=False
                    np.testing.assert_array_equal(new[outside],full[j,outside])
                new[np.array(samples[j]['history'])-1]=-np.inf
                order=np.lexsort((np.arange(len(new)),-new));target=samples[j]['target']-1
                actual=int(np.flatnonzero(order==target)[0])+1 if np.isfinite(new[target]) else len(new)+1
                expected=int(saved[key][j]);exact+=int(actual==expected)
                if actual!=expected:
                    # Different BLAS summation may reorder almost exact ties;
                    # expose them rather than silently changing saved metrics.
                    rank_window=order[max(0,min(actual,expected)-2):max(actual,expected)+1]
                    separation=float(np.max(np.abs(new[rank_window]-new[target])))
                    assert separation<2e-6,(domain,key,j,actual,expected,separation)
                    precision_sensitive+=1
                rows.append(dict(domain=domain,model=key,user=samples[j]['user_id'],saved_rank=expected,numpy_rank=actual,max_score_change=delta))
    result=dict(state='VERIFIED',manifest_files=checked,original_source_manifest_matches=source_matches,
                all_user_metric_checks=summary_checks,numpy_forward_sort_samples=len(rows),exact_rank_matches=exact,
                precision_sensitive_samples=precision_sensitive,maximum_score_change=maximum_delta,
                candidate_outside_scores_exact=True,primary_users_disjoint=True,
                baseline_ranks_match_prior_independent_audit=True,verification='Same-machine numerical verification, not independent researcher replication',
                verifier_sha256=sha(Path(__file__)),details=rows)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))


if __name__=='__main__':main()
