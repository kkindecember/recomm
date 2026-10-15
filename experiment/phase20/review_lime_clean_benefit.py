"""Standalone cached-output audit. Does not import fitting or evaluation runner."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
def arrays(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def rank_one(ids,score,target,valid):
    ranked=sorted(((-float(s),int(i)) for i,s,v in zip(ids,score,valid) if v))
    found=[j+1 for j,(_,i) in enumerate(ranked) if i==target]
    return found[0] if found else 2**30


def rr(d,score):return np.array([rank_one(i,s,t,v) for i,s,t,v in zip(d['ids'],score,d['targets'],d['valid'])])
def utility(r):return np.array([1/math.log2(int(x)+1) if x<=10 else 0. for x in r])
def metrics(r):
    return {f'{name}{k}':sum((int(x)<=k) if name=='H' else (1/math.log2(int(x)+1) if x<=k else 0.) for x in r)/len(r)
            for k in (5,10,20,50) for name in ('H','N')}


def replay_head(saved,x,prior):
    z=torch.from_numpy(((x-saved['mean'])/saved['std']).astype('float32'));w=saved['state_dict']
    with torch.no_grad():
        if saved['kind']=='linear':value=torch.nn.functional.linear(z,w['network.weight'],w['network.bias'])
        else:
            h=torch.tanh(torch.nn.functional.linear(z,w['network.0.weight'],w['network.0.bias']))
            value=torch.nn.functional.linear(h,w['network.2.weight'],w['network.2.bias'])
        return (torch.from_numpy((10*prior).astype('float32'))+2*torch.tanh(value.squeeze(-1))).numpy()


def domain_audit(domain):
    folder=OUT/domain;d=arrays(folder/'data.npz');samples=read(folder/'samples.json');rec=rows(folder/'candidates.jsonl')
    if (folder/'cache_manifest.json').exists():
        for name,digest in read(folder/'cache_manifest.json')['sha256'].items():assert sha(folder/name)==digest,name
    assert len(rec)==len(d['users'])==read(folder/'cache_summary.json')['users']
    assert read(folder/'cache_summary.json')['model_unchanged']
    count=np.load(folder/'frequencies.npy');experts=np.load(folder/'experts.npy',mmap_mode='r');lime=np.load(folder/'lime.npy',mmap_mode='r')
    gp=arrays(folder/'gate_prediction.npz');gate=torch.load(folder/'gate.pt',map_location='cpu')
    gate_users={s['user_id'] for s in samples if s['role']=='gate_calibration'}
    assert set(gate['fit_users'])==gate_users
    assert not gate_users & set(d['users'])
    q=arrays(ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz')
    held_users={r['user_id'] for r in rows(ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1'/domain/'terminal_holdout.jsonl')}
    pool={str(u) for u in q['users'][q['evaluation_indices']]}-held_users
    expected=sorted(pool,key=lambda u:hashlib.sha256(f'20260922|clean-benefit-evaluation|{domain}|{u}'.encode()).digest())[:512]
    assert expected==d['users'][d['roles']=='development'].tolist()
    assert np.isfinite(experts).all() and np.isfinite(lime).all()
    catalog=read(folder/'catalog.json');paths={tuple(p):i+1 for i,p in enumerate(catalog['decoder_paths'])}
    max_error=0.;guided=[]
    for i,(r,user) in enumerate(zip(rec,d['users'])):
        j=int(d['sample_indices'][i]);s=samples[j]
        assert r['sample_index']==j and r['user_id']==user==s['user_id']
        assert r['target']==d['targets'][i]==s['target'] and r['role']==d['roles'][i]==s['role']
        ids=np.array(r['ids']);native=np.array(r['native']);mixed=np.array(r['mixed'])
        assert len(ids)==len(set(ids))==50
        assert [paths[tuple(p)] for p in r['paths']]==ids.tolist()
        for name,v in [('ids',ids),('native',native),('mixed',mixed)]:np.testing.assert_array_equal(v,d[name][i])
        valid=np.array([int(k) not in s['history'] for k in ids]);np.testing.assert_array_equal(valid,d['valid'][i])
        assert s['history']==s['full_history'][-20:]
        fused=(gp['weights'][j,:,None]*experts[j]).sum(0)
        fused[np.unique(s['full_history'])-1]-=gp['penalty'][j]
        error=float(abs(fused-lime[j]).max());max_error=max(max_error,error)
        np.testing.assert_allclose(fused,lime[j],atol=2e-7,rtol=1e-6)
        # Independent candidate feature reconstruction, including eligibility boundary.
        order=sorted(range(50),key=lambda z:(-native[z],ids[z]));rank={k:v+1 for v,k in enumerate(order)}
        boundary=sorted(mixed[valid],reverse=True)[9]
        rebuilt=np.array([[native[k],mixed[k]-native[k],*experts[j,:,item-1].tolist(),float(lime[j,item-1]),
            1/rank[k],mixed[k]-boundary,math.log1p(int(count[item])),math.log1p(len(s['full_history']))] for k,item in enumerate(ids)],dtype='float32')
        np.testing.assert_allclose(rebuilt,d['x'][i],atol=2e-6,rtol=1e-6)
        ns=sorted(native[valid],reverse=True);ms=sorted(mixed[valid],reverse=True);delta=(mixed-native)[valid]
        no=sorted(np.flatnonzero(valid),key=lambda k:(-native[k],ids[k]))
        mo=sorted(np.flatnonzero(valid),key=lambda k:(-mixed[k],ids[k]))
        overlap=len(set(ids[no[:10]]) & set(ids[mo[:10]]))/10
        qx=np.r_[gp['features'][j],[ns[0]-ns[1],ns[0]-ns[9],ms[0]-ms[1],ms[0]-ms[9],
            delta.mean(),delta.std(),delta.min(),delta.max(),overlap,valid.mean()]].astype('float32')
        np.testing.assert_allclose(qx,d['query_x'][i],atol=2e-6,rtol=1e-6)
        if s['role']=='development':
            g=r['guided'];assert [paths[tuple(p)] for p in g['paths']]==g['ids']
            v=[k not in s['history'] for k in g['ids']]
            guided.append(rank_one(g['ids'],g['scores'],s['target'],v))
    tr=np.flatnonzero(d['roles']=='ranker_fit');dv=np.flatnonzero(d['roles']=='development')
    train={k:v[tr] for k,v in d.items()};dev={k:v[dv] for k,v in d.items()}
    assert not set(train['users']) & set(dev['users'])
    oof=arrays(folder/'oof_predictions.npz');dp=arrays(folder/'development_predictions.npz');savedr=arrays(folder/'ranks.npz')
    folds=np.array([int.from_bytes(hashlib.sha256(f'20260922|clean-benefit-fold|{domain}|{u}'.encode()).digest()[:8],'big')%5 for u in train['users']])
    np.testing.assert_array_equal(folds,oof['folds']);np.testing.assert_array_equal(oof['users'],train['users']);np.testing.assert_array_equal(dp['users'],dev['users'])
    y=utility(rr(train,train['mixed']))-utility(rr(train,train['native']))
    heads=0;score_error=0.
    names=('linear_calibration','mlp_seed0','mlp_seed1','mlp_seed2')
    for f in list(range(5))+['final']:
        fit=np.flatnonzero(folds!=f) if f!='final' else np.arange(len(tr))
        ix=np.flatnonzero(folds==f) if f!='final' else np.arange(len(dv))
        prediction=oof if f!='final' else dp;src=train if f!='final' else dev;tag=f'fold{f}' if f!='final' else 'final'
        c=torch.load(folder/'heads'/f'{tag}.ridge.pt',map_location='cpu');m=c['model']
        assert set(c['fit_users'])==set(train['users'][fit])
        assert not set(c['fit_users']) & set(src['users'][ix])
        xx=train['query_x'][fit].astype(float);mu=xx.mean(0);sd=np.maximum(xx.std(0),1e-5);z=(xx-mu)/sd
        intercept=y[fit].mean();coef=np.linalg.solve(z.T@z+len(fit)*np.eye(z.shape[1]),z.T@(y[fit]-intercept))
        np.testing.assert_allclose(coef,m['coef'],atol=1e-10,rtol=1e-10)
        benefit=((src['query_x'][ix]-mu)/sd)@coef+intercept
        np.testing.assert_allclose(benefit,prediction['benefit'][ix],atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(np.where((benefit>0)[:,None],src['mixed'][ix],src['native'][ix]),prediction['benefit_ridge'][ix])
        constant=src['mixed'][ix] if intercept>0 else src['native'][ix]
        np.testing.assert_array_equal(constant,prediction['constant_gate'][ix])
        for name in names:
            c=torch.load(folder/'heads'/f'{tag}.{name}.pt',map_location='cpu')
            np.testing.assert_array_equal(c['fit_indices'],fit);assert set(c['fit_users'])==set(train['users'][fit])
            assert not set(c['fit_users']) & set(src['users'][ix])
            covered=np.array([np.any((train['ids'][i]==train['targets'][i])&train['valid'][i]) for i in fit])
            np.testing.assert_array_equal(c['used'],fit[covered])
            raw=train['x'][fit[covered]];valid=train['valid'][fit[covered]]
            np.testing.assert_array_equal(c['mean'],raw[valid].mean(0,dtype=float).astype('float32'))
            np.testing.assert_array_equal(c['std'],np.maximum(raw[valid].std(0,dtype=float),1e-5).astype('float32'))
            replay=replay_head(c,src['x'][ix],src['mixed'][ix]);error=float(abs(replay-prediction[name][ix]).max());score_error=max(score_error,error)
            np.testing.assert_allclose(replay,prediction[name][ix],atol=2e-6,rtol=1e-6)
            assert c['optimizer_steps']==30 and len(c['curve'])==30;heads+=1
    report=read(folder/'analysis.json');checked=0
    for label,src,pred in [('oof',train,oof),('development',dev,dp)]:
        prefix='oof' if label=='oof' else 'dev'
        for name in ('GRAM','fixed_late','constant_gate','benefit_ridge')+names:
            r=rr(src,pred[name]);np.testing.assert_array_equal(r,savedr[f'{prefix}_{name}'])
            for key,value in metrics(r).items():assert abs(value-report[label]['metrics'][name][key])<1e-12
            checked+=len(r)
    np.testing.assert_array_equal(guided,savedr['dev_fixed_decode'])
    for key,value in metrics(guided).items():assert abs(value-report['development']['metrics']['fixed_decode'][key])<1e-12
    # Rebuild frozen advancement conditions, not just their displayed scores.
    for name in ('constant_gate','benefit_ridge')+names:
        wins=0
        for f in range(5):
            ix=np.flatnonzero(folds==f)
            values={a:metrics(savedr['oof_'+a][ix]) for a in (name,'GRAM','fixed_late')}
            for a,m in values.items():
                for key,v in m.items():assert abs(v-report['folds'][str(f)][a][key])<1e-12
            wins+=all(values[name]['N10']>values[a]['N10'] for a in ('GRAM','fixed_late'))
        devm={a:metrics(savedr['dev_'+a]) for a in (name,'GRAM','fixed_late')}
        trm={a:metrics(savedr['oof_'+a]) for a in (name,'GRAM','fixed_late')}
        dv=all(devm[name]['N10']>=1.01*devm[a]['N10'] and devm[name]['H10']>=devm[a]['H10'] for a in ('GRAM','fixed_late'))
        tv=wins>=4 and all(trm[name]['N10']>trm[a]['N10'] for a in ('GRAM','fixed_late'))
        assert report['checks'][name]==dict(development_pass=dv,oof_pass=tv,positive_folds_vs_both=wins,passed=dv and tv)
    return dict(state='VERIFIED',users=len(rec),replayed_heads=heads,checked_ranks=checked+len(guided),
        max_lime_score_error=max_error,max_head_score_error=score_error,complete_user_lists=True,
        gate_fit_disjoint=True,ranker_fit_disjoint=True,predictions_replayed=True,independent_metric_sorting=True)


def main():
    torch.set_num_threads(2);started=time.monotonic();assert read(OUT/'status.json')['state']=='ANALYZED'
    manifest=read(OUT/'manifest.json')['sha256']
    for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
    result={d:domain_audit(d) for d in ('Toys','Beauty')}
    summary=read(OUT/'summary.json')
    for name,passed in summary['decision'].items():
        if name=='mlp_all_three_seeds':expected=all(summary['decision'][f'mlp_seed{i}'] for i in range(3))
        else:expected=all(read(OUT/d/'analysis.json')['checks'][name]['passed'] for d in ('Toys','Beauty'))
        assert passed==expected
    result=dict(state='VERIFIED',domains=result,input_hashes=len(manifest),seconds=time.monotonic()-started,
        reviewer_sha256=sha(Path(__file__)),note='Standalone recomputation in same assistant session; not an external independent reviewer.')
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    summary.update(state='COMPLETED',independent_verification_pending=False,verification=result)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    status=read(OUT/'status.json');status.update(state='COMPLETED',verified=True,independent_verification_pending=False)
    status['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
