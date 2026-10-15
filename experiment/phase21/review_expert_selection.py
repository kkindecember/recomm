"""Independent prediction-level review of completed expert-selection runs.

No fitting, decoding, candidate-weight scheduling, or test-set access.
"""
import json
from pathlib import Path

import numpy as np

from experiment.phase21.candidate_gate_screen import ROOT,write,sha

SOURCE=ROOT/'artifacts/phase21/expert_selection_20261014_v1'
DUAL=ROOT/'artifacts/phase21/expert_resource_recovery_20260923_v1/removal/without_itemcf'
OUT=SOURCE/'review'


def rank(row, target, history):
    seen=set(history)
    pairs=sorted(((i,s) for i,s in zip(row['ids'],row['scores']) if i not in seen),key=lambda p:(-p[1],p[0]))
    return next((j+1 for j,(i,_) in enumerate(pairs) if i==target),2**30)


def utility(r,k=10):
    return np.where(r<=k,1/np.log2(r+1),0.)


def main():
    assert json.loads((SOURCE/'status.json').read_text())['state']=='COMPLETED'
    OUT.mkdir(exist_ok=False)
    summary=json.loads((SOURCE/'summary.json').read_text())
    results={};manifest={};count=0;checks=0
    for domain in ('Toys','Beauty'):
        rr={};users=None;native=None;targets=None;parent=None
        for arm in ('dual','sas_only','text_only'):
            folder=(DUAL/domain) if arm=='dual' else SOURCE/arm/domain
            dc=json.loads((folder/'data_config.json').read_text())
            if parent is None:parent=dc['parent_sha256']
            assert dc['parent_sha256']==parent
            samples=[s for s in json.loads((folder/'samples.json').read_text()) if s['role']=='development']
            predictions=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
            assert len(samples)==len(predictions)==512
            ix=np.array([s['user_id'] for s in samples]);yy=np.array([s['target'] for s in samples])
            if users is None:users=ix;targets=yy
            np.testing.assert_array_equal(users,ix);np.testing.assert_array_equal(targets,yy)
            recomputed={k:[] for k in ('native','guided','late')}
            for sample,p in zip(samples,predictions):
                assert (sample['user_id'],sample['target'],sample['history'])==(p['user_id'],p['target'],p['history'])
                for scope in recomputed:
                    v=p[scope];assert len(v['ids'])==len(set(v['ids']))==len(v['scores'])==50
                    assert np.isfinite(v['scores']).all()
                    recomputed[scope].append(rank(v,p['target'],p['history']))
                count+=1
            saved=np.load(folder/'ranks.npz')
            np.testing.assert_array_equal(saved['users'],users)
            for scope,name in (('native','GRAM'),('guided','guided'),('late','late')):
                np.testing.assert_array_equal(recomputed[scope],saved[name]);checks+=1
            if native is None:native=np.asarray(recomputed['native'])
            np.testing.assert_array_equal(native,recomputed['native'])
            rr[arm]=np.array(recomputed['guided'])
            for name in ('predictions.jsonl','ranks.npz','summary.json','data_config.json'):
                p=folder/name;manifest[str(p.relative_to(ROOT))]=sha(p)
        rr['GRAM']=native
        for arm,r in rr.items():
            for k in (5,10,20,50):
                assert abs(float(utility(r,k).mean())-summary[domain]['metrics'][arm]['N'+str(k)])<1e-14
                assert abs(float((r<=k).mean())-summary[domain]['metrics'][arm]['H'+str(k)])<1e-14
        x,y,z=[rr[a] for a in ('sas_only','text_only','dual')]
        hit=dict(sas_hits=int((x<=10).sum()),text_hits=int((y<=10).sum()),dual_hits=int((z<=10).sum()),
                 joint_hits=int(((x<=10)&(y<=10)).sum()),sas_exclusive=int(((x<=10)&(y>10)).sum()),
                 text_exclusive=int(((y<=10)&(x>10)).sum()),
                 dual_retained_sas_exclusive=int(((x<=10)&(y>10)&(z<=10)).sum()),
                 dual_retained_text_exclusive=int(((y<=10)&(x>10)&(z<=10)).sum()),
                 dual_new_beyond_single_hits=int(((x>10)&(y>10)&(z<=10)).sum()),
                 dual_lost_joint_hits=int(((x<=10)&(y<=10)&(z>10)).sum()))
        comparisons={}
        for a in ('GRAM','sas_only','text_only'):
            r=rr[a];delta=utility(z)-utility(r)
            assert abs(delta.mean()-summary[domain]['comparisons']['dual_vs_'+a]['delta_N10'])<1e-14
            both=(z<=10)&(r<=10);gained=(z<=10)&(r>10);lost=(z>10)&(r<=10)
            parts={k:float(delta[m].sum()/len(delta)) for k,m in [('both_hit_rank_change',both),('new_hits',gained),('lost_hits',lost)]}
            assert abs(sum(parts.values())-delta.mean())<1e-14
            comparisons[a]=dict(dual_relative_improvement=float(utility(z).mean()/utility(r).mean()-1),
                removing_expert_relative_change=float(utility(r).mean()/utility(z).mean()-1) if a!='GRAM' else None,
                per_user_N10_decomposition=parts,
                paired_interval_includes_zero=bool(summary[domain]['comparisons']['dual_vs_'+a]['descriptive_95pct_N10'][0]<=0<=summary[domain]['comparisons']['dual_vs_'+a]['descriptive_95pct_N10'][1]))
        results[domain]=dict(users=512,hit_overlap=hit,comparisons=comparisons)
    write(OUT/'diagnostic.json',results)
    write(OUT/'manifest.json',dict(sha256=manifest))
    write(OUT/'verification.json',dict(state='VERIFIED_SAVED_PREDICTIONS',prediction_records_rechecked=count,
        rank_arrays_recomputed=checks,aggregate_metrics_rechecked=64,same_user_order_and_targets=True,
        same_parent_checkpoint_per_domain=True,all_expected_users_present=True,
        new_training=False,new_generation=False,test_read=False))
    # Assumptions and interpretation checks: descriptive comparisons only.
    scan={
      'Simpson_paradox':'Reported domains separately; N10 directions agree. H10 differs by domain/comparator.',
      'ecological_fallacy':'Paired user-level ranks; no claim every user benefits.',
      'Berkson_selection':'Repeated development pool selected for exploration; generalization remains unconfirmed.',
      'collider_bias':'Primary comparisons use all fixed 512 users; hit-overlap groups are descriptive, outcome-defined.',
      'base_rate_neglect':'Report absolute N10/H10 and hit counts alongside relative changes.',
      'regression_to_mean':'Same-user matched controls; selection after previous exploration remains a limitation.',
      'survivorship_bias':'All expected users in all four completed jobs retained; no successful-row-only filtering.',
      'look_elsewhere':'All domains and planned controls reported; intervals unadjusted, no confirmatory significance claim.',
      'forking_paths':'Expert shortlist selected using prior results; current protocol fixed before new generation.',
      'causal_language':'Conditional component comparison with refitted common fusion; not a general causal claim about user behavior.',
      'reverse_causality':'No target supplied as inference input; only held-out development targets used in outcome calculations.'}
    write(OUT/'interpretation_checks.json',dict(coverage='11/11',checks=scan,
        decision='SASRec + text is a provisional exploration configuration; necessity is not established robustly.'))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
