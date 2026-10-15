"""Post-hoc CPU controls: account for ranking harm, no new model or decoding."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/lime_generation_trace_20260922_v1'
PARENT=ROOT/'artifacts/phase20/lime_eligible_path_20260922_v1'


def read(p):return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f:return [json.loads(line) for line in f if line.strip()]


def value(r):return 1/math.log2(r+1) if r<=10 else 0.


def rank(ids,scores,history,target):
    pairs=sorted([(int(i),float(v)) for i,v in zip(ids,scores) if i not in set(history)],key=lambda x:(-x[1],x[0]))
    order=[i for i,_ in pairs]
    return order.index(target)+1 if target in order else 51


def metrics(rr):
    return dict(N10=float(np.mean([value(r) for r in rr])),H10=float(np.mean([r<=10 for r in rr])),
                H50=float(np.mean([r<=50 for r in rr])))


def main():
    note=dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),post_hoc=True,
        trigger='Trace audit found most harmful cases retain their target; inspect score/order versus membership.',
        predeclared_before_this_control='On the fixed generated candidates, compare existing fused score, native GRAM path score, and direct frozen LIME item score; no tuning.',
        test_read=False,new_decodes=0,model_updates=0,
        limitation='128-user target-independent subsamples are still reused development; enriched cases are not aggregate performance evidence.')
    (OUT/'cached_control_protocol.json').write_text(json.dumps(note,indent=2)+'\n')
    results={}
    for d in ('Toys','Beauty'):
        complete=rows(PARENT/d/'evaluation.jsonl');counts=Counter();deltas=Counter();totals=Counter()
        for r in complete:
            g,l,f=(r['arms'][k]['rank'] for k in ('GRAM','fixed_late','fixed'))
            totals['score']+=value(l)-value(g);totals['membership']+=value(f)-value(l)
            if value(f)>=value(g):continue
            counts['hurt_users']+=1;counts['target_retained' if f<=50 else 'target_lost']+=1
            sh=value(l)<value(g);ch=value(f)<value(l)
            counts['both' if sh and ch else 'score_only' if sh else 'membership_only' if ch else 'neither']+=1
            deltas['score']+=value(l)-value(g);deltas['membership']+=value(f)-value(l)
            deltas['total']+=value(f)-value(g)
        records=rows(OUT/d/'diagnostics.jsonl');samples=read(OUT/d/'samples.json')
        with np.load(OUT/d/'signals.npz',allow_pickle=False) as f:signals=f['lime'].copy()
        output=[];max_shared_error=0.
        for j,r in enumerate(records):
            trace=ROOT/r['traces']['fixed']['path']
            with gzip.open(trace,'rt') as f:p=json.load(f)
            lookup={tuple(prefix):dict(zip(row['tokens'],row['native'])) for s in p['steps'] for prefix,row in zip(s['prefixes'],s['rows'])}
            native=[sum(lookup[tuple(path[:k])][path[k]] for k in range(1,len(path)))/(len(path)-1) for path in p['result']['paths']]
            source=next(x for x in complete if x['user_id']==r['user_id'])
            baseline=dict(zip(source['candidate_scores']['GRAM']['ids'],source['candidate_scores']['GRAM']['scores']))
            for item,v in zip(p['result']['ids'],native):
                if item in baseline:
                    e=abs(v-baseline[item]);max_shared_error=max(max_shared_error,e);assert e<1e-5
            scores=signals[j][np.asarray(p['result']['ids'])-1]
            ranks=dict(GRAM=source['arms']['GRAM']['rank'],fixed=source['arms']['fixed']['rank'],
                fixed_late=source['arms']['fixed_late']['rank'],
                guided_candidates_native_score=rank(p['result']['ids'],native,samples[j]['history'],r['target']),
                guided_candidates_LIME_score=rank(p['result']['ids'],scores,samples[j]['history'],r['target']))
            assert (ranks['fixed']<=50)==(ranks['guided_candidates_native_score']<=50)==(ranks['guided_candidates_LIME_score']<=50)
            output.append(dict(user_id=r['user_id'],groups=r['groups'],ranks=ranks))
        rep=[r for r in output if 'representative' in r['groups']]
        results[d]=dict(full_512_harm_counts=dict(counts),full_512_harmed_subset_N10_components={k:v/512 for k,v in deltas.items()},
            full_512_overall_N10_components={k:v/512 for k,v in totals.items()},
            representative_users=len(rep),representative_metrics={k:metrics([r['ranks'][k] for r in rep]) for k in rep[0]['ranks']},
            maximum_native_score_error_on_shared_paths=max_shared_error,
            hurt_users_recovered_by_native_score=sum('guided_hurt_vs_GRAM' in r['groups'] and value(r['ranks']['guided_candidates_native_score'])>=value(r['ranks']['GRAM']) for r in output),
            hurt_users_recovered_by_LIME_score=sum('guided_hurt_vs_GRAM' in r['groups'] and value(r['ranks']['guided_candidates_LIME_score'])>=value(r['ranks']['GRAM']) for r in output))
        with (OUT/d/'cached_control_ranks.jsonl').open('w') as f:
            for r in output:f.write(json.dumps(r)+'\n')
    (OUT/'cached_controls.json').write_text(json.dumps(dict(protocol=note,domains=results),indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
