"""Recheck existing expert removal predictions for current exploration criteria."""
import json
from pathlib import Path
import numpy as np
from experiment.phase21.candidate_gate_screen import ROOT,write,sha

OUT=ROOT/'artifacts/phase21/expert_simplification_audit_20261014_v1'
REM=ROOT/'artifacts/phase21/expert_resource_recovery_20260923_v1/removal'
TEXT=ROOT/'artifacts/phase21/text_contribution_20260923_v1'


def rank(v,target,history):
    pairs=[(i,s) for i,s in zip(v['ids'],v['scores']) if i not in history]
    pairs.sort(key=lambda z:(-z[1],z[0]))
    return next((j+1 for j,(i,_) in enumerate(pairs) if i==target),2**30)


def utility(r):
    return np.where(r<=10,1/np.log2(r+1),0.)


def main():
    OUT.mkdir(exist_ok=False);all_results={};manifest={}
    for d in ('Toys','Beauty'):
        results={};reference=None;native=None
        for arm in ('real','without_itemcf','without_sasrec','without_text'):
            p=(TEXT if arm in ('real','without_text') else REM)/arm/d
            samples=json.loads((p/'samples.json').read_text());samples=[s for s in samples if s['role']=='development']
            rows=[json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
            assert len(rows)==len(samples)==512
            config=json.loads((p/'data_config.json').read_text())
            if reference is None:reference=config['parent_sha256']
            assert config['parent_sha256']==reference
            ranks={key:[] for key in ('native','guided','late')}
            for s,row in zip(samples,rows):
                assert (s['user_id'],s['history'],s['target'])==(row['user_id'],row['history'],row['target'])
                for key in ranks:ranks[key].append(rank(row[key],row['target'],row['history']))
            ranks={k:np.array(v) for k,v in ranks.items()}
            saved=np.load(p/'ranks.npz')
            for key,other in [('native','GRAM'),('guided','guided'),('late','late')]:np.testing.assert_array_equal(ranks[key],saved[other])
            np.testing.assert_array_equal(saved['users'],np.array([s['user_id'] for s in samples]))
            if native is None:native=ranks['native'];users=saved['users'].copy()
            np.testing.assert_array_equal(native,ranks['native']);np.testing.assert_array_equal(users,saved['users'])
            results[arm]=ranks
            for name in ('predictions.jsonl','ranks.npz','data_config.json','fit_checks.json'):
                q=p/name
                if q.exists():manifest[str(q.relative_to(ROOT))]=sha(q)
        means={a:float(utility(rr['guided']).mean()) for a,rr in results.items()}
        base=float(utility(native).mean());detail={}
        for arm,rr in results.items():
            diff=utility(rr['guided'])-utility(native);rng=np.random.default_rng(20261014)
            boot=diff[rng.integers(0,512,size=(2000,512))].mean(1)
            detail[arm]=dict(N10=means[arm],H10=float((rr['guided']<=10).mean()),H50=float((rr['guided']<=50).mean()),
                relative_vs_GRAM=means[arm]/base-1,relative_vs_full=means[arm]/means['real']-1,
                paired_delta_vs_GRAM=float(diff.mean()),descriptive_ci95=np.quantile(boot,[.025,.975]).tolist())
        all_results[d]=dict(users=512,parent_sha256=reference,GRAM_N10=base,arms=detail)
    write(OUT/'summary.json',all_results);write(OUT/'manifest.json',dict(sha256=manifest))
    write(OUT/'verification.json',dict(state='VERIFIED_EXISTING_PREDICTIONS',domains=2,users_per_domain=512,arms=4,rank_arrays_recomputed=24,
          same_user_order=True,same_parent_per_domain=True,training_rerun=False,generation_rerun=False,test_read=False))
    print(json.dumps(all_results,indent=2))


if __name__=='__main__':main()
