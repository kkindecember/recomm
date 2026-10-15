"""Frozen two-arm generation diagnostics; no training or new recommendation policy."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20.lime_generation_trace_core import BeamObserver, validate_steps, target_path_analysis

ROOT = base.ROOT
OUT = ROOT/'artifacts/phase20/lime_generation_trace_20260922_v1'
PARENT = ROOT/'artifacts/phase20/lime_eligible_path_20260922_v1'
PROTOCOL = ROOT/'plan/第二十阶段/LIME生成过程诊断_20260922.md'
REPORT = ROOT/'report/第二十阶段/LIME生成过程诊断_20260922.md'
MODULE = 'experiment.phase20.lime_generation_trace'
GPUS = {'Toys':'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2',
        'Beauty':'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253'}


def ndcg(rank):
    return 1/math.log2(rank+1) if rank <= 10 else 0.


def load_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def groups_for(row):
    a = row['arms']; groups = []
    g,f,l = (a[k]['rank'] for k in ('GRAM','fixed','fixed_late'))
    if (g <= 50) != (f <= 50): groups.append('coverage_changed')
    if ndcg(f) > ndcg(g): groups.append('guided_helped_vs_GRAM')
    if ndcg(f) < ndcg(g): groups.append('guided_hurt_vs_GRAM')
    if ndcg(f) > ndcg(l): groups.append('guided_helped_vs_same_rule_late')
    if ndcg(f) < ndcg(l): groups.append('guided_hurt_vs_same_rule_late')
    return groups


def prepare():
    assert base.read(PARENT/'status.json')['state'] == 'COMPLETED'
    inherited = base.read(PARENT/'manifest.json')['sha256']
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'status.json',dict(state='PREPARING',updated_at=base.stamp()))
    manifest = {}
    def use(path):
        rel = str(Path(path).relative_to(ROOT))
        assert '/test/' not in rel and 'test.json' not in rel
        actual = base.sha(path)
        if rel in inherited: assert actual == inherited[rel],rel
        manifest[rel] = actual
        return path
    config = dict(domains=base.DOMAINS,gpus=GPUS,representative_users=128,beam_size=50,
                  alpha=.3,score_scale=20.,minimum_free_mib=12288,allocation_limit_gib=11,
                  worker_timeout_seconds=None,test_read=False,optimizer_steps=0,
                  diagnostic_only=True,automatic_new_experiments=False)
    base.write(OUT/'config.json',config)
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    for domain in base.DOMAINS:
        folder=OUT/domain;folder.mkdir();(folder/'traces').mkdir()
        old_samples=base.read(use(PARENT/domain/'samples.json'))
        records=load_rows(use(PARENT/domain/'evaluation.jsonl'))
        representative={r['user_id'] for r in sorted(records,key=lambda r:hashlib.sha256(
            f'20260922|trace-representative|{domain}|{r["user_id"]}'.encode()).digest())[:128]}
        selected=[];refs=[]
        for row in records:
            groups=groups_for(row)
            if row['user_id'] in representative:groups=['representative']+groups
            if not groups:continue
            original=old_samples[row['sample_index']]
            assert all(row[k]==original[k] for k in ('user_id','target','history'))
            selected.append(dict(original,parent_sample_index=row['sample_index'],groups=groups))
            refs.append({k:row[k] for k in ('user_id','target','history','arms','candidate_scores','paths')})
        cfg=base.read(use(PARENT/domain/'data_config.json'))
        for path in [ROOT/cfg['parent_checkpoint'],ROOT/cfg['original_config'],
                     ROOT/cfg['prepared']/'catalog.json',ROOT/cfg['prepared']/'validation.jsonl']:
            use(path)
        indices=np.asarray([s['parent_sample_index'] for s in selected])
        with np.load(use(PARENT/domain/'signals.npz'),allow_pickle=False) as f:
            signals={k:f[k][indices] for k in f.files}
        # Expert scores are diagnostic inputs only; frozen fused scores remain authoritative.
        query_indices=np.asarray([s['query_index'] for s in selected])
        recovery=ROOT/f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        for label,path in [('SASRec',recovery/'sasrec_scores_seed0.npy'),
                           ('ItemCF',ROOT/f'artifacts/phase20/lime_comparison_20260920_v1/{domain}/itemcf_scores.npy'),
                           ('BGE',recovery/'semantic_scores.npy')]:
            raw=np.asarray(np.load(use(path),mmap_mode='r')[query_indices])
            valid=raw > -1e8
            low=np.where(valid,raw,np.inf).min(1,keepdims=True)
            high=np.where(valid,raw,-np.inf).max(1,keepdims=True)
            signals[label]=(raw-low)/(high-low+1e-8)
        np.savez_compressed(folder/'signals.npz',**signals)
        base.write(folder/'samples.json',selected)
        base.write(folder/'reference.json',refs)
        base.write(folder/'data_config.json',cfg)
        base.write(folder/'selection.json',dict(users=len(selected),representative=128,
            group_counts=dict(Counter(g for s in selected for g in s['groups'])),
            outcome_enrichment_explicit=True,selected_before_new_decoding=True))
        base.write(folder/'status.json',dict(state='PREPARED',domain=domain,users_total=len(selected),
                                           users_completed=0,updated_at=base.stamp()))
        for name in ('signals.npz','samples.json','reference.json','data_config.json','selection.json'):
            use(folder/name)
    for name in ('lime_generation_trace.py','lime_generation_trace_core.py','test_lime_generation_trace.py',
                 'run_lime_generation_trace.sh','lime_decode_screen.py','lime_decode_screen_core.py',
                 'lime_path_mixture_core.py','guided_decoding_core.py'):
        use(ROOT/'experiment/phase20'/name)
    for path in (PROTOCOL,OUT/'protocol.md',OUT/'config.json',ROOT/'experiment/phase18/core/diff_generation_cache.py'):
        use(path)
    base.write(OUT/'manifest.json',dict(sha256=manifest,created_at=base.stamp()))
    base.write(OUT/'status.json',dict(state='PREPARED',updated_at=base.stamp(),test_read=False,
                                   domains={d:base.read(OUT/d/'selection.json') for d in base.DOMAINS}))
    print(json.dumps(base.read(OUT/'status.json')),flush=True)


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    return value


def restore_trace(value):
    # Only trace score arrays contain nulls; they encode masked -infinity.
    if value is None:return -math.inf
    if isinstance(value,dict):return {k:restore_trace(v) for k,v in value.items()}
    if isinstance(value,list):return [restore_trace(v) for v in value]
    return value


def summarize(records):
    output={}
    group_names=['representative','all_selected_cases']+sorted({g for r in records for g in r['groups'] if g!='representative'})
    for group in group_names:
        cohort=[r for r in records if group=='all_selected_cases' or group in r['groups']]
        arms={}
        for name in ('GRAM','fixed'):
            values=[r['analysis'][name] for r in cohort]
            pruned=[v for v in values if v['first_loss'] is not None and not v['raw_target_present']]
            losses=[v['first_loss'] for v in pruned]
            branches=[v for v in pruned if not v['first_loss']['forced']]
            arms[name]=dict(outcomes=dict(Counter(v['outcome'] for v in values)),
                first_prune_depth=dict(Counter(str(e['depth']) for e in losses)),
                pruned_users=len(pruned),forced_at_prune=sum(e['forced'] for e in losses),
                native_local_top1_at_prune=sum(e['native_child_rank']==1 for e in losses),
                branching_pruned_users=len(branches),
                branching_expert_raises_target_probability=sum(v['first_loss']['expert_log_ratio']>1e-6 for v in branches),
                branching_expert_target_child_top1=sum(v['first_loss']['expert_child_rank']==1 for v in branches),
                pruned_despite_lime_item_top50=sum(v['lime_catalog_rank']<=50 for v in pruned),
                pruned_despite_lime_item_top10=sum(v['lime_catalog_rank']<=10 for v in pruned),
                branching_raised_and_lime_item_top50=sum(v['first_loss']['expert_log_ratio']>1e-6 and v['lime_catalog_rank']<=50 for v in branches),
                missed_item_and_lime_item_below50=sum(not v['raw_target_present'] and v['lime_catalog_rank']>50 for v in values))
        output[group]=dict(users=len(cohort),arms=arms)
    return output


class Worker(base.Worker):
    def __init__(self,domain):
        base.OUT=OUT
        super().__init__(domain)
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        install_cross_cache_reuse(self.model,50)
        self.references=base.read(self.folder/'reference.json')
        self.users_completed=0

    def status(self,state='RUNNING',**kw):
        super().status(state,users_completed=getattr(self,'users_completed',0),
                       users_total=len(getattr(self,'samples',[])),**kw)

    def observed_decode(self,ctx,q,alpha,observe=True):
        import torch
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        from experiment.phase20.lime_path_mixture_core import PathMixtureGuide
        from contextlib import nullcontext
        begin=time.monotonic();guide=PathMixtureGuide(self.index,q,alpha,0.)
        observer=BeamObserver(guide)
        processors=[guide,observer] if observe else [guide]
        with torch.no_grad(), observer.installed() if observe else nullcontext():
            output=super(GRAM,self.model).generate(input_ids=ctx['ids'].reshape(1,-1),
                attention_mask=ctx['mask'].reshape(1,-1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden']),
                max_length=self.max_length,prefix_allowed_tokens_fn=self.prefix_fn,
                num_beams=50,num_return_sequences=50,return_dict_in_generate=True,output_scores=True,
                length_penalty=1.,logits_processor=LogitsProcessorList(processors))
        paths=[p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids=[self.index.path_to_item[tuple(p)] for p in paths]
        scores=output.sequences_scores.detach().cpu().double().numpy()
        np.testing.assert_allclose(guide.rescore(paths),scores,atol=2e-6,rtol=1e-5)
        assert len(set(ids))==50 and np.isfinite(scores).all()
        checked=validate_steps(observer.steps) if observe else None
        del output
        torch.cuda.synchronize();self.seconds.append(time.monotonic()-begin);self.completed+=1
        return dict(ids=ids,paths=paths,scores=scores.tolist(),observer=observer,checks=checked)

    def expert_target_paths(self,j,target_path):
        output={}
        for label in ('SASRec','ItemCF','BGE'):
            scores=self.signals[label][j].astype(np.float64)
            weights=np.exp(20*(scores-scores.max()))
            mass=np.bincount(self.index.node_items,weights=weights[self.index.item_refs-1],minlength=len(self.index.nodes))
            records=[]
            for depth in range(1,len(target_path)):
                parent=self.index.lookup[tuple(target_path[:depth])]
                children=self.index.children[parent]
                token=target_path[depth]
                values={t:mass[c]/mass[parent] for t,c in children.items()}
                records.append(dict(depth=depth,children=len(children),
                    probability=float(values[token]),child_rank=1+int(sum(v>values[token] for v in values.values()))))
            allowed=np.ones(len(scores),dtype=bool)
            allowed[np.asarray(self.samples[j]['history'],dtype=np.int64)-1]=False
            score=scores[self.samples[j]['target']-1]
            output[label]=dict(catalog_rank=1+int(np.sum(scores[allowed]>score)),
                               catalog_rank_tie_end=int(np.sum(scores[allowed]>=score)),path=records)
        return output

    def run(self):
        from experiment.phase20.lime_decode_screen_core import ordered,target_rank
        self.stage='trace';records=[];max_difference=0.;smoke=[]
        with (self.folder/'diagnostics.jsonl').open('x') as handle:
            for j,sample in enumerate(self.samples):
                self.status(current_sample=j)
                ctx=self.context(j);q=self.index.conditional_logs(self.signals['lime'][j])
                target_path=self.catalog['decoder_paths'][sample['target']-1]
                analyses={};parity={};checks={};trace_files={}
                for name,alpha in [('GRAM',0.),('fixed',.3)]:
                    self.status(current_sample=j,current_arm=name)
                    out=self.observed_decode(ctx,q,alpha)
                    ref=self.references[j]
                    assert out['ids']==ref['candidate_scores'][name]['ids'],('candidate identity',j,name)
                    assert out['paths']==ref['paths'][name],('path identity',j,name)
                    expected=ref['candidate_scores'][name]['scores']
                    np.testing.assert_allclose(out['scores'],expected,atol=5e-6,rtol=1e-5)
                    difference=float(np.max(np.abs(np.asarray(out['scores'])-expected)))
                    max_difference=max(max_difference,difference)
                    items=ordered(out['ids'],out['scores'],sample['history'])
                    rank=target_rank(items,sample['target'])
                    assert items==ref['arms'][name]['items'] and rank==ref['arms'][name]['rank']
                    if j<3:
                        plain=self.observed_decode(ctx,q,alpha,False)
                        assert plain['paths']==out['paths']
                        np.testing.assert_array_equal(plain['scores'],out['scores'])
                        smoke.append(dict(sample=j,arm=name,observer_identity_exact=True))
                    payload=dict(steps=out['observer'].steps,final=out['observer'].final,
                                 result={k:out[k] for k in ('ids','paths','scores')})
                    filename=f'{j:04d}.{name}.json.gz';path=self.folder/'traces'/filename
                    with gzip.open(path,'wt',encoding='utf-8') as f:
                        json.dump(json_safe(payload),f,separators=(',',':'),allow_nan=False)
                    trace_files[name]=dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path))
                    analyses[name]=target_path_analysis(out['observer'].steps,target_path,out['ids'],sample['target'],
                        rank,q,self.signals['lime'][j],sample['history'])
                    checks[name]=out['checks'];parity[name]=dict(paths_and_ranks_exact=True,max_abs_score_error=difference)
                record=dict(user_id=sample['user_id'],target=sample['target'],sample_index=j,
                    parent_sample_index=sample['parent_sample_index'],groups=sample['groups'],
                    analysis=analyses,experts=self.expert_target_paths(j,target_path),
                    cache_parity=parity,step_checks=checks,traces=trace_files)
                handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush();records.append(record)
                self.users_completed=j+1;self.status(current_sample=j)
                del ctx,out
        assert self.digest()==self.initial_digest
        for path,digest in self.loaded_source.items():assert base.sha(ROOT/path)==digest,path
        base.write(self.folder/'summary.json',dict(state='COMPLETED',domain=self.domain,users=len(records),
            groups=summarize(records),model_unchanged=True,test_read=False,optimizer_steps=0,
            observer_identity_checks=smoke,maximum_cache_score_difference=max_difference,
            elapsed_seconds=time.monotonic()-self.started,completed_decodes=self.completed))
        self.stage='done';self.status('COMPLETED')


def supervise():
    active={};pending=list(base.DOMAINS);done={};started=time.monotonic()
    while pending or active:
        for domain in list(pending):
            if base.gpu_free(GPUS[domain])<12288:continue
            with (OUT/domain/'worker.log').open('x') as log:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[domain],PYTHONUNBUFFERED='1',
                         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                child=subprocess.Popen([sys.executable,'-m',MODULE,'worker','--domain',domain],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            active[domain]=child;pending.remove(domain)
        for domain,child in list(active.items()):
            code=child.poll()
            if code is not None:
                done[domain]=code;del active[domain]
                if code and base.read(OUT/domain/'status.json').get('state')!='FAILED':
                    base.write(OUT/domain/'status.json',dict(state='FAILED',exit_code=code,updated_at=base.stamp()))
        state='RUNNING' if active or pending else ('GENERATED' if all(c==0 for c in done.values()) else 'FAILED')
        base.write(OUT/'status.json',dict(state=state,updated_at=base.stamp(),supervisor_pid=os.getpid(),
            active={d:p.pid for d,p in active.items()},pending=pending,exit_codes=done,
            domains={d:base.read(OUT/d/'status.json') for d in base.DOMAINS},
            elapsed_seconds=time.monotonic()-started,test_read=False,automatic_new_experiments=False))
        if pending or active:time.sleep(15)
    if all(c==0 for c in done.values()):verify()


def verify():
    for path,digest in base.read(OUT/'manifest.json')['sha256'].items():
        assert base.sha(ROOT/path)==digest,path
    checks={};summaries={}
    for domain in base.DOMAINS:
        folder=OUT/domain;samples=base.read(folder/'samples.json');records=load_rows(folder/'diagnostics.jsonl')
        summary=base.read(folder/'summary.json');references=base.read(folder/'reference.json')
        assert len(records)==len(samples)==summary['users']
        assert sum('representative' in s['groups'] for s in samples)==128
        steps_count=0
        for j,(sample,record) in enumerate(zip(samples,records)):
            assert all(sample[k]==record[k] for k in ('user_id','target','groups'))
            for arm,trace in record['traces'].items():
                path=ROOT/trace['path'];assert base.sha(path)==trace['sha256']
                with gzip.open(path,'rt') as f:payload=restore_trace(json.load(f))
                checked=validate_steps(payload['steps']);assert checked==record['step_checks'][arm]
                assert payload['result']['ids']==references[j]['candidate_scores'][arm]['ids']
                assert payload['result']['paths']==references[j]['paths'][arm]
                steps_count+=len(payload['steps'])
            for arm in ('GRAM','fixed'):
                a=record['analysis'][arm];ref=references[j]['arms'][arm]
                assert a['final_rank']==ref['rank']
                assert a['raw_target_present']==(sample['target'] in references[j]['candidate_scores'][arm]['ids'])
                if a['raw_target_present']:assert a['first_loss'] is None
        assert summarize(records)==summary['groups']
        assert summary['model_unchanged'] and all(s['observer_identity_exact'] for s in summary['observer_identity_checks'])
        checks[domain]=dict(users=len(records),beam_steps_checked=steps_count,cache_paths_and_ranks_exact=True)
        summaries[domain]=summary
    base.write(OUT/'verification.json',dict(state='VERIFIED',verified_at=base.stamp(),domains=checks,
                                          frozen_inputs_unchanged=True,test_read=False))
    base.write(OUT/'summary.json',dict(domains=summaries,verification=checks,test_read=False,
        caveat='Retrospective development diagnosis; enriched cases are not population rates; local support is not causal proof.'))
    previous=base.read(OUT/'status.json')
    base.write(OUT/'status.json',dict(previous,state='COMPLETED',updated_at=base.stamp(),verified=True))
    print(json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','supervise','worker','verify'])
    parser.add_argument('--domain',choices=base.DOMAINS);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='launch':
        assert base.read(OUT/'status.json')['state']=='PREPARED'
        assert not (OUT/'launch.json').exists()
        with (OUT/'supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m',MODULE,'supervise'],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.write(OUT/'launch.json',dict(pid=child.pid,started_at=base.stamp()))
        print(json.dumps(dict(pid=child.pid,status=str(OUT/'status.json'))),flush=True)
    elif args.action=='worker':
        try:Worker(args.domain).run()
        except Exception:
            detail=traceback.format_exc();print(detail,flush=True)
            base.write(OUT/args.domain/'failure.json',dict(error=detail,at=base.stamp()))
            previous=base.read(OUT/args.domain/'status.json')
            base.write(OUT/args.domain/'status.json',dict(previous,state='FAILED',error=detail,updated_at=base.stamp()))
            raise
    elif args.action=='supervise':
        try:supervise()
        except Exception:
            detail=traceback.format_exc();print(detail,flush=True)
            previous=base.read(OUT/'status.json')
            base.write(OUT/'status.json',dict(previous,state='FAILED',error=detail,updated_at=base.stamp()))
            raise
    else:verify()


if __name__=='__main__':main()
