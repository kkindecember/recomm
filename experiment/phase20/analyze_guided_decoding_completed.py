"""Read-only scientific audit of completed frozen-search outputs; no GPU calls."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now

OUT=ROOT/'artifacts/phase20/guided_decoding_20260919_v1'
REVIEW=OUT/'completed_review'


def independent_rank(ids,scores,target):
    ordered=sorted(zip(ids,scores),key=lambda pair:-pair[1])
    return next((i+1 for i,(item,_) in enumerate(ordered) if item==target),51)


def metrics(ranks):
    result={}
    for k in (5,10,20,50):
        hits=[r for r in ranks if r<=k]
        result[f'H{k}']=len(hits)/len(ranks)
        result[f'N{k}']=sum(1/math.log2(r+1) for r in hits)/len(ranks)
    return result


def top10(row,head):
    return [i for i,_ in sorted(zip(row['candidate_ids'],row['scores'][head]),key=lambda pair:-pair[1])[:10]]


def gain(rank): return 1/math.log2(rank+1) if rank<=10 else 0.


def audit_domain(domain):
    folder=OUT/domain; summary=read(folder/'summary.json'); verification=read(folder/'verification.json')
    selection=read(folder/'selection.json'); cohorts=read(folder/'cohorts.json')
    for name,expected in verification['output_sha256'].items(): assert sha(folder/name)==expected
    assert sha(folder/'tuning.jsonl')==selection['tuning_sha256']
    assert selection['evaluation_decodes_so_far']==0
    assert selection['selected']==summary['selected_strengths']
    groups=[{r['user_id'] for r in cohorts[g]} for g in ('smoke','tuning','evaluation')]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    rows={a:[] for a in ('vanilla','real_cf','shuffled_cf','popularity')}; checked=0
    with (folder/'evaluation.jsonl').open() as f:
        for line in f:
            r=json.loads(line); ids=r['candidate_ids']; assert len(set(ids))==50
            for key,scores in r['scores'].items():
                assert len(scores)==50 and np.isfinite(scores).all()
                assert independent_rank(ids,scores,r['target'])==r['ranks'][key]
                checked+=1
            if r['arm']!='vanilla': assert r['strength']==selection['selected'][r['arm']]
            rows[r['arm']].append(r)
    for arm,records in rows.items():
        assert [r['user_id'] for r in records]==[e['user_id'] for e in cohorts['evaluation']]
        for head in records[0]['ranks']:
            actual=metrics([r['ranks'][head] for r in records]); expected=summary['metrics'][arm+'.'+head]
            assert max(abs(actual[k]-expected[k]) for k in actual)<1e-12
    contrasts={}; rng=np.random.RandomState(read(OUT/'config.json')['selection_seed'])
    samples=rng.randint(0,512,(read(OUT/'config.json')['bootstrap_repeats'],512))
    for name,expected in summary['contrasts'].items():
        aa,bb=name.split('__vs__'); arm_a,head_a=aa.split('.',1); arm_b,head_b=bb.split('.',1)
        a=np.array([r['ranks'][head_a] for r in rows[arm_a]]); b=np.array([r['ranks'][head_b] for r in rows[arm_b]])
        difference=np.array([gain(x)-gain(y) for x,y in zip(a,b)])
        ci=np.percentile(difference[samples].mean(1),[2.5,97.5])
        np.testing.assert_allclose(ci,expected['descriptive_95pct_N10'],rtol=0,atol=1e-12)
        assert abs(difference.mean()-expected['delta_N10'])<1e-12
        contrasts[name]=dict(delta_N10=float(difference.mean()),descriptive_95pct_N10=ci.tolist())
    categories={k:dict(users=0,N10_sum_change=0.,top10_gained=0,top10_lost=0) for k in
                ('target_newly_covered','target_lost_coverage','target_covered_in_both','target_absent_in_both')}
    ordered_equal=sets_equal=target_top10_equal=new_items_in_final_top10=0
    user_rows=[]
    for b,c in zip(rows['vanilla'],rows['real_cf']):
        assert b['user_id']==c['user_id'] and b['target']==c['target']
        before=b['target'] in b['candidate_ids']; after=c['target'] in c['candidate_ids']
        category=('target_covered_in_both' if before and after else 'target_newly_covered' if after else
                  'target_lost_coverage' if before else 'target_absent_in_both')
        br,cr=b['ranks']['softmax'],c['ranks']['softmax']; delta=gain(cr)-gain(br)
        part=categories[category]; part['users']+=1;part['N10_sum_change']+=delta
        part['top10_gained']+=int(cr<=10<br);part['top10_lost']+=int(br<=10<cr)
        late=top10(b,'late_real_cf'); early=top10(c,'guided')
        ordered_equal+=int(late==early);sets_equal+=int(set(late)==set(early))
        target_top10_equal+=int(min(b['ranks']['late_real_cf'],11)==min(c['ranks']['guided'],11))
        nnew=len(set(top10(c,'softmax'))-set(b['candidate_ids']));new_items_in_final_top10+=nnew
        user_rows.append(dict(user_id=b['user_id'],target=b['target'],category=category,baseline_rank=br,
            guided_pipeline_rank=cr,N10_change=delta,new_items_in_final_top10=nnew,
            late_only_rank=b['ranks']['late_real_cf'],guided_score_rank=c['ranks']['guided']))
    for part in categories.values(): part['contribution_to_mean_N10_change']=part['N10_sum_change']/512
    with (REVIEW/f'{domain}.mechanism.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(user_rows[0]));writer.writeheader();writer.writerows(user_rows)
    baseline=summary['metrics']['vanilla.softmax']; current=summary['metrics']['real_cf.softmax']
    return dict(state='CACHED_OUTPUT_AUDIT_PASSED',checked_ranks=checked,users=512,
        baseline=baseline,real_cf=current,relative_N10_percent=(current['N10']/baseline['N10']-1)*100,
        primary_comparison=summary['contrasts']['real_cf.softmax__vs__vanilla.softmax'],
        low_frequency_baseline=summary['low_frequency_metrics']['vanilla.softmax'],
        low_frequency_real_cf=summary['low_frequency_metrics']['real_cf.softmax'],
        paired_intervals_recomputed=contrasts,coverage_decomposition=categories,
        early_vs_late=dict(identical_top10_item_order_users=ordered_equal,identical_top10_item_set_users=sets_equal,
                          identical_target_rank_at10_users=target_top10_equal),
        new_items_in_guided_pipeline_top10_total=new_items_in_final_top10,
        source_sha256={name:sha(folder/name) for name in ('summary.json','verification.json','selection.json','evaluation.jsonl')})


def main():
    assert read(OUT/'status.json')['state']=='COMPLETED'
    REVIEW.mkdir(exist_ok=False)
    results={domain:audit_domain(domain) for domain in ('Toys','Beauty')}
    write(REVIEW/'audit.json',dict(created_at=now(),state='CACHED_OUTPUT_AUDIT_PASSED',domains=results,
        new_training_steps=0,new_gpu_inference=0,test_read=False,
        limitations='Exploratory reused-validation sample; no new training seed or independent test; checkpoint replay done by original worker.',
        analysis_source_sha256=sha(Path(__file__))))
    for domain,result in results.items():
        print(domain,json.dumps({k:result[k] for k in ('relative_N10_percent','primary_comparison','coverage_decomposition','early_vs_late','new_items_in_guided_pipeline_top10_total')},ensure_ascii=False))


if __name__=='__main__':main()
