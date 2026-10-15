"""Independent CSV accounting and real-row log-domain pooling verification."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/fusion_pooling_20260925_v1'
SOURCE=ROOT/'artifacts/phase21/full_validation_20260924_v1'
summary=json.loads((OUT/'summary.json').read_text())
assert summary['state']=='COMPLETED'
result={}
for domain in ('Toys','Beauty'):
    with (OUT/(domain+'_per_user.csv')).open() as f:
        rows=list(csv.DictReader(f))
    assert len({r['user_id'] for r in rows})==len(rows)
    groups={'all':rows,'previous512':[r for r in rows if r['previously_evaluated']=='True'],
            'remaining_validation_events':[r for r in rows if r['previously_evaluated']=='False']}
    detail={}
    for name,group in groups.items():
        old=np.array([int(r['old_rank']) for r in group])
        new=np.array([int(r['new_rank']) for r in group])
        original=summary['domains'][domain][name]
        for kind,ranks in (('old',old),('new',new)):
            for k in (10,50):
                h=sum(int(r<=k) for r in ranks)/len(ranks)
                n=sum(1/np.log2(r+1) for r in ranks if r<=k)/len(ranks)
                assert abs(h-original[kind]['H'+str(k)])<1e-12
                assert abs(n-original[kind]['N'+str(k)])<1e-12
        gained=int(((old>10)&(new<=10)).sum())
        lost=int(((old<=10)&(new>10)).sum())
        assert (gained,lost)==(original['gained10'],original['lost10'])
        assert gained-lost==int((new<=10).sum()-(old<=10).sum())
        unsupported=np.array([all(int(r[e+'_rank'])>10 for e in ('SASRec','ItemCF','semantic')) for r in group])
        detail[name]=dict(users=len(group),original_hits=int((old<=10).sum()),new_hits=int((new<=10).sum()),
            none_expert_top10_old_fusion_hits=int((unsupported&(old<=10)).sum()),
            none_expert_top10_new_fusion_hits=int((unsupported&(new<=10)).sum()),
            lost_none_expert_top10=int((unsupported&(old<=10)&(new>10)).sum()),
            mean_top10_overlap=float(np.mean([int(r['top10_overlap']) for r in group])))
    source=SOURCE/domain
    samples=json.loads((source/'samples.json').read_text())
    scores=np.load(source/'experts.npy',mmap_mode='r',allow_pickle=False)
    with np.load(source/'gate_prediction.npz',allow_pickle=False) as z:
        weights=z['weights'];penalties=z['penalty']
    checked=[]
    for i in (0,len(samples)//2,len(samples)-1):
        logits=20*scores[i].astype(float)
        logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None]
        w=weights[i].astype(float);w/=w.sum()
        value=np.logaddexp.reduce(logp+np.log(w)[:,None],axis=0)/20
        sample=samples[i]
        value[np.array(sample['full_history'])-1]-=penalties[i]
        ids=np.setdiff1d(np.arange(1,len(value)+1),sample['history'][-20:])
        ordered=ids[np.lexsort((ids,-value[ids-1]))]
        rank=int(np.flatnonzero(ordered==sample['target'])[0])+1
        assert rank==int(rows[i]['new_rank'])
        assert int(ordered[9])==int(rows[i]['new10_boundary_item'])
        checked.append(i)
    known=[{k:r[k] for k in ('user_id','target','old_rank','new_rank','SASRec_rank','ItemCF_rank','semantic_rank')}
           for r in rows if r['user_id'] in ('A18G3K8RVJD28','A1JEA7U282WD4','A3P1SVONO50V1B')]
    result[domain]=dict(unique_users=True,all_reported_metrics_and_hit_counts_match_csv=True,
        independent_log_domain_full_sort_rows=checked,cohorts=detail,previously_discussed_cases=known)
result['users']=sum(v['cohorts']['all']['users'] for v in result.values())
assert result['users']==41775
(OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
