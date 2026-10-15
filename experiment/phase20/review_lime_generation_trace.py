"""Independent retrospective audit of saved beam traces; never loads a model."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/lime_generation_trace_20260922_v1'
PARENT=ROOT/'artifacts/phase20/lime_eligible_path_20260922_v1'


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f:return [json.loads(line) for line in f if line.strip()]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def score(rank):
    return 1/math.log2(rank+1) if rank<=10 else 0.


def main():
    assert read(OUT/'status.json')['state'] in {'GENERATED','COMPLETED'}
    assert all(read(OUT/d/'status.json')['state']=='COMPLETED' for d in ('Toys','Beauty'))
    domains={};case_rows=[];audit={}
    for domain in ('Toys','Beauty'):
        folder=OUT/domain;samples=read(folder/'samples.json');records=rows(folder/'diagnostics.jsonl')
        cfg=read(folder/'data_config.json');catalog=read(ROOT/cfg['prepared']/'catalog.json')
        original=rows(PARENT/domain/'evaluation.jsonl');ref={r['user_id']:r for r in original}
        assert len(original)==512 and len(samples)==len(records)
        representative={r['user_id'] for r in sorted(original,key=lambda r:hashlib.sha256(
            f'20260922|trace-representative|{domain}|{r["user_id"]}'.encode()).digest())[:128]}
        expected=[]
        for r in original:
            g,f,l=(r['arms'][k]['rank'] for k in ('GRAM','fixed','fixed_late'))
            if r['user_id'] in representative or (g<=50)!=(f<=50) or score(g)!=score(f) or score(f)!=score(l):
                expected.append(r['user_id'])
        assert [s['user_id'] for s in samples]==expected
        descendants=defaultdict(list)
        for item,path in enumerate(catalog['decoder_paths'],1):
            for length in range(1,len(path)+1):descendants[tuple(path[:length])].append(item-1)
        descendants={p:np.asarray(ids,dtype=np.int64) for p,ids in descendants.items()}
        with np.load(folder/'signals.npz',allow_pickle=False) as f:signals={k:f[k] for k in f.files}
        path_checks=0;event_checks=0;max_path_error=0.
        for j,(sample,record) in enumerate(zip(samples,records)):
            assert sample['user_id']==record['user_id'] and sample['target']==record['target']
            assert ('representative' in sample['groups'])==(sample['user_id'] in representative)
            original_row=ref[sample['user_id']];target=sample['target'];target_path=catalog['decoder_paths'][target-1]
            excluded=set(sample['history']);allowed=np.array([i not in excluded for i in range(1,len(catalog['items']))])
            s=signals['lime'][j].astype(float);weights=np.exp(20*(s-s.max()))
            lime_rank=1+int(np.count_nonzero(s[allowed]>s[target-1]))
            for arm in ('GRAM','fixed'):
                trace=record['traces'][arm];path=ROOT/trace['path'];assert sha(path)==trace['sha256']
                with gzip.open(path,'rt') as f:payload=json.load(f)
                result=payload['result'];steps=payload['steps'];a=record['analysis'][arm]
                assert result['paths']==original_row['paths'][arm]
                assert result['ids']==original_row['candidate_scores'][arm]['ids']
                np.testing.assert_allclose(result['scores'],original_row['candidate_scores'][arm]['scores'],atol=5e-6,rtol=1e-5)
                candidates=sorted([(i,v) for i,v in zip(result['ids'],result['scores']) if i not in excluded],key=lambda p:(-p[1],p[0]))
                ordered=[i for i,_ in candidates]
                assert ordered==original_row['arms'][arm]['items']
                rank=ordered.index(target)+1 if target in ordered else 51
                assert rank==a['final_rank'] and lime_rank==a['lime_catalog_rank']
                lookup={tuple(p):dict(zip(r['tokens'],r['used'])) for step in steps for p,r in zip(step['prefixes'],step['rows'])}
                for p,value in zip(result['paths'],result['scores']):
                    rebuilt=sum(lookup[tuple(p[:d])][p[d]] for d in range(1,len(p)))/(len(p)-1)
                    error=abs(rebuilt-value);max_path_error=max(max_path_error,error);assert error<1e-5
                    path_checks+=1
                loss_depth=None;eos_status=None;events=[]
                for step in steps:
                    d=step['depth']
                    if d>=len(target_path):continue
                    matching=[k for k,p in enumerate(step['prefixes']) if p==target_path[:d] and
                              step['parent_scores'][k] is not None and step['parent_scores'][k]>-1e8]
                    if not matching:continue
                    k=max(matching,key=lambda k:step['parent_scores'][k]);r=step['rows'][k];t=target_path[d]
                    pos=r['tokens'].index(t);native=r['native'];native_rank=1+sum(v>native[pos] for v in native)
                    kept=any(i==k and token==t for i,token in zip(step['selected']['next_beam_indices'],step['selected']['next_beam_tokens']))
                    prefix=tuple(target_path[:d]);mass=weights[descendants[prefix]].sum()
                    q=weights[descendants[prefix+(t,)]].sum()/mass
                    logratio=math.log(q)-native[pos]+r['legal_log_mass']
                    events.append((d,native_rank,kept,logratio))
                    if t==1:
                        eos_status=any(h['prefix']==target_path[:-1] for h in step['finished']);break
                    if not kept:loss_depth=d;break
                assert len(events)==len(a['events'])
                for (d,nr,kept,logratio),event in zip(events,a['events']):
                    assert d==event['depth'] and nr==event['native_child_rank'] and kept==event['kept_active']
                    assert abs(logratio-event['expert_log_ratio'])<1e-9
                    event_checks+=1
                assert loss_depth==(a['first_loss']['depth'] if a['first_loss'] else None)
                if target in excluded:outcome='excluded_by_existing_history_rule'
                elif target in result['ids']:outcome='returned_top10' if rank<=10 else 'returned_below_top10'
                elif loss_depth is not None:outcome='prefix_pruned'
                elif eos_status is not None:outcome='finished_queue_or_final_selection' if eos_status else 'eos_not_admitted'
                else:outcome='search_stopped_before_target_completion'
                assert outcome==a['outcome']
                loss=a['first_loss']
                case_rows.append(dict(domain=domain,user_id=sample['user_id'],arm=arm,groups=';'.join(sample['groups']),
                    target=target,rank=rank,outcome=outcome,lime_rank=lime_rank,
                    SASRec_rank=record['experts']['SASRec']['catalog_rank'],ItemCF_rank=record['experts']['ItemCF']['catalog_rank'],
                    BGE_rank=record['experts']['BGE']['catalog_rank'],prune_depth=loss['depth'] if loss else '',
                    forced=loss['forced'] if loss else '',native_child_rank=loss['native_child_rank'] if loss else '',
                    expert_child_rank=loss['expert_child_rank'] if loss else '',expert_log_ratio=loss['expert_log_ratio'] if loss else '',
                    global_rank=loss['global_rank_min'] if loss else '',margin=loss['margin_to_active_cutoff'] if loss else ''))
        groups={}
        names=['representative','all_selected']+sorted({g for r in records for g in r['groups'] if g!='representative'})
        for group in names:
            rr=[r for r in records if group=='all_selected' or group in r['groups']]
            data={}
            for arm in ('GRAM','fixed'):
                values=[r['analysis'][arm] for r in rr];pruned=[v for v in values if v['first_loss']]
                branches=[v for v in pruned if not v['first_loss']['forced']]
                misses=[v for v in values if not v['raw_target_present']]
                data[arm]=dict(users=len(values),outcomes=dict(Counter(v['outcome'] for v in values)),
                    prune_depth=dict(Counter(str(v['first_loss']['depth']) for v in pruned)),
                    pruned=len(pruned),branching_pruned=len(branches),
                    expert_raises_probability=sum(v['first_loss']['expert_log_ratio']>1e-6 for v in branches),
                    locally_native_top1=sum(v['first_loss']['native_child_rank']==1 for v in pruned),
                    forced_prune=sum(v['first_loss']['forced'] for v in pruned),
                    misses=len(misses),misses_lime_top10=sum(v['lime_catalog_rank']<=10 for v in misses),
                    misses_lime_top50=sum(v['lime_catalog_rank']<=50 for v in misses),
                    misses_lime_below50=sum(v['lime_catalog_rank']>50 for v in misses),
                    branching_boosted_lime_top50=sum(v['first_loss']['expert_log_ratio']>1e-6 and v['lime_catalog_rank']<=50 for v in branches))
            data['expert_top50_users']={label:sum(r['experts'][label]['catalog_rank']<=50 for r in rr) for label in ('SASRec','ItemCF','BGE')}
            data['missed_by_fixed_but_some_expert_top50']=sum(not r['analysis']['fixed']['raw_target_present'] and
                any(r['experts'][label]['catalog_rank']<=50 for label in ('SASRec','ItemCF','BGE')) for r in rr)
            groups[group]=data
        domains[domain]=groups
        audit[domain]=dict(users=len(records),representative=128,completed_paths_rebuilt=path_checks,
            target_events_recomputed=event_checks,maximum_path_score_error=max_path_error,
            sampling_recomputed=True,trace_hashes_verified=True,original_output_parity=True)
    with (OUT/'case_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(case_rows[0]));w.writeheader();w.writerows(case_rows)
    write(OUT/'independent_review.json',dict(state='VERIFIED',at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          checks=audit,groups=domains,no_runner_or_observer_import=True,test_read=False,
          statistical_scope='Descriptive development diagnosis; outcome-enriched cases reported separately; no significance tests or causal identification.'))
    print(json.dumps(dict(checks=audit,representative={d:domains[d]['representative'] for d in domains}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
