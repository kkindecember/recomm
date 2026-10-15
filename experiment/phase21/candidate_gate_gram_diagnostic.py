"""Separate candidate-access and ordering evidence in existing generation outputs.

No new training or generation. Conditional groups explain arithmetic; they are
outcome-defined and must not be interpreted as causal or attainable gains.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT=ROOT/'artifacts/phase21/candidate_gate_20261010_v1'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def utility(r):return np.where(r<=10,1/np.log2(r+1),0.)


def main():
    source=ROOT/'artifacts/phase21/full_opportunity_counts_20260925_v1/per_user.csv'
    with source.open() as f:rows=list(csv.DictReader(f))
    result={};manifest={str(source.relative_to(ROOT)):sha(source),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for domain in ('Toys','Beauty'):
        folder=FULL/domain;samples=json.loads((folder/'samples.json').read_text())
        with np.load(folder/'ranks.npz') as f:original={k:f[k].copy() for k in ('GRAM','late','guided')}
        ranks={k:[] for k in original};same_set=0;dups=0
        with (folder/'predictions.jsonl').open() as f:
            for j,line in enumerate(f):
                record=json.loads(line);s=samples[j];assert record['user_id']==s['user_id']
                same_set+=int(set(record['native']['ids'])==set(record['late']['ids']))
                for name,field in (('GRAM','native'),('late','late'),('guided','guided')):
                    v=record[field];dups+=int(len(v['ids'])!=len(set(v['ids'])))
                    selected=[(a,b) for a,b in zip(v['ids'],v['scores']) if a not in set(s['history'])]
                    selected.sort(key=lambda x:(-x[1],x[0]))
                    rank=next((i+1 for i,(a,b) in enumerate(selected) if a==s['target']),2**30)
                    ranks[name].append(rank)
        assert len(ranks['GRAM'])==len(samples) and same_set==len(samples)
        ranks={k:np.array(v) for k,v in ranks.items()}
        for k in ranks:np.testing.assert_array_equal(ranks[k],original[k])
        rr=[r for r in rows if r['domain']==domain]
        assert [r['user_id'] for r in rr]==[s['user_id'] for s in samples]
        for k in ranks:np.testing.assert_array_equal(ranks[k],np.array([int(r[k]) for r in rr]))
        er=np.array([[int(r[k]) for k in ('SASRec','ItemCF','semantic','LIME')] for r in rr])
        result[domain]=dict(recomputed_prediction_ranks_exact=True,late_candidate_set_exact=True,duplicate_prediction_lists=dups,cohorts={})
        primary=np.array([not s['previously_evaluated'] and s['training_holdout_role']=='none' for s in samples])
        for name,mask in [('all',np.ones(len(samples),bool)),('primary',primary)]:
            r={k:v[mask] for k,v in ranks.items()};g,l,a=r['GRAM'],r['late'],r['guided'];e=er[mask];n=len(g)
            on=g<=50;ng,nl,na=utility(g),utility(l),utility(a)
            opportunity=(g>10)&(e[:,:3].min(1)<=10)
            unconverted=opportunity&(a>10)
            diff=na-nl
            result[domain]['cohorts'][name]=dict(users=n,
                metrics={k:dict(N10=float(utility(v).mean()),H10=float((v<=10).mean()),H50=float((v<=50).mean())) for k,v in r.items()},
                native_target_in50=int(on.sum()),native_target_rank11_50=int(((g>10)&on).sum()),
                guided_new_target_in50=int(((g>50)&(a<=50)).sum()),guided_lost_target_in50=int(((g<=50)&(a>50)).sum()),
                guided_new_target_top10_outside_native50=int(((g>50)&(a<=10)).sum()),
                guided_gained10_vs_late=int(((l>10)&(a<=10)).sum()),guided_lost10_vs_late=int(((l<=10)&(a>10)).sum()),
                same_candidate_late_delta_N10=float((nl-ng).mean()),
                guided_vs_late_delta_N10=float(diff.mean()),
                guided_vs_late_delta_N10_from_native_in50=float(diff[on].sum()/n),
                guided_vs_late_delta_N10_from_native_outside50=float(diff[~on].sum()/n),
                expert_top10_complementary_opportunities=int(opportunity.sum()),
                guided_converted_opportunities=int((opportunity&(a<=10)).sum()),
                unconverted=int(unconverted.sum()),fusion_already_out10=int((unconverted&(e[:,3]>10)).sum()),
                fusion_in10_guided11_50=int((unconverted&(e[:,3]<=10)&(a<=50)).sum()),
                fusion_in10_guided_out50=int((unconverted&(e[:,3]<=10)&(a>50)).sum()))
        for filename in ('samples.json','ranks.npz','predictions.jsonl'):
            manifest[str((folder/filename).relative_to(ROOT))]=sha(folder/filename)
    OUT.mkdir(exist_ok=True)
    (OUT/'gram_diagnostic.json').write_text(json.dumps(dict(results=result,input_sha256=manifest,
        note='Old single-upstream-seed validation outputs; no new generation; conditional groups are descriptive.'),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
