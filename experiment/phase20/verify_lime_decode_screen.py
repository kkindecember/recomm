"""Independent CPU audit of screen predictions; imports no experiment/model code."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/lime_decode_screen_20260921_v1'


def read(p):return json.loads(p.read_text())


def measured(ranks):
    return {f'{m}{k}':sum((int(r<=k) if m=='H' else int(r<=k)/math.log2(r+1))
            for r in ranks)/len(ranks) for k in (5,10,20,50) for m in ('N','H')}


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT.relative_to(ROOT)))
    args=parser.parse_args()
    OUT=ROOT/args.output
    summary=read(OUT/'summary.json'); checks={}
    assert summary['test_read'] is False
    for domain in ('Toys','Beauty'):
        folder=OUT/domain; selection=read(folder/'selection.json')
        samples=read(folder/'samples.json'); sig=np.load(folder/'signals.npz',allow_pickle=False)
        rows={s:[json.loads(l) for l in (folder/f'{s}.jsonl').read_text().splitlines()]
              for s in ('tuning','evaluation')}
        budget=read(folder/'budget.json') if (folder/'budget.json').exists() else dict(tuning_users=128,evaluation_users=512)
        assert len(rows['tuning'])==budget['tuning_users'] and len(rows['evaluation'])==budget['evaluation_users']
        assert set(r['user_id'] for r in rows['tuning']).isdisjoint(r['user_id'] for r in rows['evaluation'])
        n=0
        for split, group in rows.items():
            for r in group:
                j=r['sample_index']; assert r['split']==split and samples[j]['split']==split
                assert r['target']==samples[j]['target'] and r['history']==samples[j]['history']
                h=set(r['history'])
                for name,arm in r['arms'].items():
                    if name=='LIME': expected=sig['lime_top50'][j].tolist()
                    else:
                        raw=r['candidate_scores'][name]
                        assert len(raw['ids'])==50 and len(set(raw['ids']))==50
                        assert all(math.isfinite(v) for v in raw['scores'])
                        expected=[i for i,s in sorted(zip(raw['ids'],raw['scores']),key=lambda a:(-a[1],a[0])) if i not in h]
                    assert expected==arm['items'] and not (set(expected)&h)
                    rr=next((i+1 for i,v in enumerate(expected) if v==r['target']),51)
                    assert rr==arm['rank'];n+=1
        for signal in ('sasrec','lime'):
            scores=[]
            for a in (0.,.1,.3):
                name='GRAM' if not a else f'{signal}_{a}'
                v=measured([r['arms'][name]['rank'] for r in rows['tuning']])['N10']
                scores.append((v,-a,a))
            assert max(scores)[2]==selection['selected_alpha'][signal]
        s=summary['domains'][domain]
        for name,table in s['metrics'].items():
            ranks=[r['arms'][name]['rank'] for r in rows['evaluation']]
            for metric,value in measured(ranks).items():assert abs(value-table[metric])<1e-12
        for name,table in s['comparisons'].items():
            left,right=name.split('_vs_',1)
            if right=='late':right=left+'_late'
            a=[r['arms'][left]['rank'] for r in rows['evaluation']]
            b=[r['arms'][right]['rank'] for r in rows['evaluation']]
            delta=measured(a)['N10']-measured(b)['N10']
            assert abs(delta-table['delta_N10'])<1e-12
            assert sum(x<=10 and y>10 for x,y in zip(a,b))==table['gained_H10']
            assert sum(x>10 and y<=10 for x,y in zip(a,b))==table['lost_H10']
            assert sum(x<=50 and y>50 for x,y in zip(a,b))==table['gained_H50']
            assert sum(x>50 and y<=50 for x,y in zip(a,b))==table['lost_H50']
        checks[domain]=dict(predictions_checked=n,metrics_and_differences_checked=True,
            selection_checked=True,cohorts_disjoint=True)
        sig.close()
    digest=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()
    result=dict(checks=checks,summary_sha256=digest,independent_of_runner_code=True,test_read=False)
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
