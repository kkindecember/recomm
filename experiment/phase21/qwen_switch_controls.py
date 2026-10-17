"""Finish Qwen4's matched 2x2 switches; compare both frozen text experts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import selfmatch_history_ablation as A
from experiment.phase21 import qwen_switch_controls_core as C
from experiment.phase20.catalog_scope_core import metrics,contrast

F,B = A.F,A.B
ROOT = A.ROOT
LONG,PAIR = F.OUT,A.OUT
SINGLE = ROOT/'artifacts/phase21/single_switch_controls_1024_20261017_v1'
QWEN = B.OLD.QWEN
OUT = ROOT/'artifacts/phase21/qwen_switch_controls_1024_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.56_Qwen4开关补齐与最终文本专家匹配筛选_20261017.md'
REPORT = ROOT/'report/第二十一阶段/Qwen4开关补齐与最终文本专家匹配筛选_20261017.md'
MODULE = 'experiment.phase21.qwen_switch_controls'
DOMAINS = ('Toys','Beauty')
JOBS = {f'{d}_{i}':dict(domain=d,shard=i) for i in range(4) for d in DOMAINS}
CFG = dict(A.CFG,users_per_domain=1024,shards_per_domain=4,users_per_job=256,
    arms=list(C.ARMS),fresh_arms=list(C.FRESH),content_model='Qwen3-Embedding-4B',
    beta_single=1.,max_parallel=None,project_gpu_task_cap=None,avoid_active_source_gpus=False,
    gpu_admission='free memory only; same GPU sharing allowed; no old queue gate',
    source=str(PAIR.relative_to(ROOT)),fresh_generations_total=12288,
    primary_setting='loo_penalty',selection_rule='worst-domain relative N10 vs common SASRec+penalty, then mean relative N10, mean relative H10, BGE',
    stop_after='complete all prescribed Qwen switches and matched development text selection; no automatic next run')
read,write,sha,stamp = A.read,A.write,A.sha,A.stamp
SETTINGS = {
    'raw_none':('pair_qwen_raw','pair_bge_raw','sas_only','qwen_raw','bge_raw'),
    'loo_none':('pair_qwen_loo','pair_bge_loo','sas_only','qwen_loo','bge_loo'),
    'raw_penalty':('pair_qwen_raw_penalty','pair_bge_raw_penalty','sas_penalty','qwen_raw_penalty','bge_raw_penalty'),
    'loo_penalty':('pair_qwen_loo_penalty','pair_bge_loo_penalty','sas_penalty','qwen_loo_penalty','bge_loo_penalty')}


def indices(shard):
    return list(range(shard,1024,4))


def tests():
    return dict(state='PASSED',score_and_selection=C.tests(),leave_self_out=A.C.tests(),
                shards_exact_partition=sorted(j for k in range(4) for j in indices(k))==list(range(1024)))


def prepare():
    torch.set_num_threads(2)
    checks = tests()
    for source in (LONG,PAIR,SINGLE):
        assert read(source/'status.json')['state']=='COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',CFG);write(OUT/'tests.json',checks)
    write(OUT/'status.json',dict(state='PREPARING',updated_at=stamp(),test_read=False))
    shutil.copy2(PLAN,OUT/'protocol.md')
    known = {}
    for source in (LONG,PAIR,SINGLE,B.OLD.OUT):
        known.update(read(source/'manifest.json')['sha256'])
        for domain in DOMAINS:
            q = source/domain/'output_manifest.json'
            if q.exists():known.update(read(q)['sha256'])
    manifest,inputs = {},{}
    def use(path):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert rel not in known or known[rel]==digest,rel
        manifest[rel] = digest
        return rel
    common = []
    for rel in read(LONG/'manifest.json')['sha256']:
        if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/')):
            common.append(use(ROOT/rel))
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and name.endswith('.py') and ROOT in Path(name).resolve().parents:
            common.append(use(Path(name).resolve()))
    common += [use(PLAN),use(OUT/'config.json')]
    for domain in DOMAINS:
        folder = OUT/domain;folder.mkdir();files=[]
        for name in ('samples.json','catalog.json','data_config.json','teacher_checks.json'):
            src = PAIR/domain/name;files.append(use(src))
            shutil.copy2(src,folder/name);files.append(use(folder/name))
        samples = read(folder/'samples.json')
        long_samples = read(LONG/domain/'samples.json')
        assert samples==read(SINGLE/domain/'samples.json')
        assert len(samples)==len({s['user_id'] for s in samples})==1024
        fit_path = B.OUT/domain/'fit_samples.json';files.append(use(fit_path))
        assert {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in read(fit_path))
        for j,s in enumerate(samples):
            assert s['source_validation_index']==j+1024
            assert all(s[k]==long_samples[j+1024][k] for k in ('user_id','target','history','full_history','evaluation_block'))
        files.append(use(LONG/domain/'samples.json'))
        long_path = LONG/domain/'predictions.jsonl';files.append(use(long_path))
        longs = {}
        with long_path.open() as stream:
            for line in stream:
                row = json.loads(line);ix = row['sample_index']
                if ix>=2048:break
                if ix>=1024:longs[ix-1024]=row
        assert sorted(longs)==list(range(1024))
        old = {}
        for source in (PAIR,SINGLE):
            path = source/domain/'predictions.jsonl';files.append(use(path))
            old[source] = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(old[source])==1024
        refs,baselines = [],[]
        for j,s in enumerate(samples):
            pair,single,raw = old[PAIR][j],old[SINGLE][j],longs[j]
            for row in (pair,single,raw):
                assert all(row[k]==s[k] for k in ('user_id','target','history','evaluation_block'))
            assert pair['native']==raw['native'] and pair['sas_only']==raw['SASRec__only']
            assert pair['pair_raw']==raw['SASRec__BGE'] and pair['bge_only']==raw['BGE__only']
            ref=dict(native=raw['native'],sas_only=raw['SASRec__only'],qwen_only=raw['Qwen4__only'],
                     pair_raw=raw['SASRec__Qwen4'])
            refs.append(ref)
            baselines.append(dict(sample_index=j,user_id=s['user_id'],target=s['target'],history=s['history'],
                evaluation_block=s['evaluation_block'],native=ref['native'],sas_only=ref['sas_only'],
                sas_penalty=single['sas_penalty'],qwen_raw=ref['qwen_only'],pair_qwen_raw=ref['pair_raw'],
                bge_raw=pair['bge_only'],bge_loo=single['bge_loo'],bge_raw_penalty=single['bge_raw_penalty'],
                bge_loo_penalty=single['bge_loo_penalty'],pair_bge_raw=pair['pair_raw'],
                pair_bge_loo=pair['pair_loo'],pair_bge_raw_penalty=pair['pair_raw_penalty'],
                pair_bge_loo_penalty=pair['pair_loo_penalty']))
        write(folder/'references.json',refs)
        with (folder/'comparison_predictions.jsonl').open('x') as stream:
            for row in baselines:stream.write(json.dumps(row)+'\n')
        source = B.OUT/domain/'features/validation/experts.npy';files.append(use(source))
        raw_all = np.load(source,mmap_mode='r')[1536:2560]
        raw = np.asarray(raw_all[:,:,[0,4]])
        np.testing.assert_array_equal(raw[:,:,0],np.load(PAIR/domain/'raw_experts.npy',mmap_mode='r')[:,:,0])
        files.append(use(PAIR/domain/'raw_experts.npy'))
        np.save(folder/'raw_experts.npy',raw)
        embedding_path = QWEN/domain/'qwen4_embeddings.npz';files.append(use(embedding_path))
        packed = np.load(embedding_path)
        assert packed['item_ids'].tolist()==read(folder/'catalog.json')['items'][1:]
        embeddings = packed['embeddings']
        assert embeddings.shape[1]>768,'Expected Qwen4 embedding, not cached BGE'
        loo = np.lib.format.open_memmap(folder/'loo_text.npy',mode='w+',dtype='float32',shape=raw.shape[:2])
        for start in range(0,1024,32):
            stop = start+32
            loo[start:stop] = A.C.leave_self_out([s['full_history'] for s in samples[start:stop]],embeddings,raw[start:stop,:,1])
            write(OUT/'status.json',dict(state='PREPARING',stage='qwen_leave_self_out',domain=domain,
                users_completed=stop,users_total=1024,updated_at=stamp(),test_read=False))
        loo.flush()
        up,changed = 0,0
        for j,s in enumerate(samples):
            mask = np.isin(np.arange(1,raw.shape[1]+1),s['full_history'])
            np.testing.assert_array_equal(loo[j,~mask],raw[j,~mask,1])
            up += int((loo[j,mask]>raw[j,mask,1]+3e-6).sum())
            changed += int((loo[j,mask]!=raw[j,mask,1]).sum())
            values = C.scores(raw[j],loo[j],s['full_history'])
            np.testing.assert_array_equal(values['pair_qwen_raw'],F.fused(raw_all[j],'SASRec__Qwen4'))
        # Independently reconstruct actual residual prototypes for three users.
        independent = []
        candidates = embeddings/(np.linalg.norm(embeddings,axis=1,keepdims=True)+1e-8)
        for j in (0,512,1023):
            history=np.asarray(samples[j]['full_history'])-1
            w=np.exp(-.1*np.arange(len(history)-1,-1,-1)).astype(np.float32)
            full=(embeddings[history]*w[:,None]).sum(0)/max(w.sum(),1e-8)
            full/=np.linalg.norm(full)+1e-8
            cos=full@candidates.T;low,span=float(cos.min()),float(np.ptp(cos))
            errors=[]
            for item in np.unique(history):
                keep=history!=item
                if not keep.any() or span<=1e-7:value=0.
                else:
                    weights=np.exp(-.1*np.arange(len(history)-1,-1,-1))
                    vector=(embeddings[history[keep]].astype(float)*weights[keep,None]).sum(0)/weights[keep].sum()
                    vector/=np.linalg.norm(vector)+1e-8
                    value=float(np.clip((vector@candidates[item].astype(float)-low)/(span+1e-8),0.,1.))
                errors.append(abs(value-float(loo[j,item])))
            assert max(errors)<2e-5,(domain,j,max(errors))
            independent.append(dict(row=j,max_residual_prototype_error=max(errors)))
        assert up==0 and changed>0
        write(folder/'scoring_checks.json',dict(state='PASSED',users=1024,embedding=str(embedding_path.relative_to(ROOT)),
            embedding_dimension=embeddings.shape[1],nonhistorical_scores_exact=True,sas_scores_exact=True,
            historical_upward_changes=up,historical_changed_scores=changed,actual_residual_checks=independent,
            target_not_used_for_scoring=True,target_in_history=sum(s['target'] in s['full_history'] for s in samples)))
        dc=read(folder/'data_config.json')
        for key in ('parent_checkpoint','original_config','embedding'):files.append(use(ROOT/dc[key]))
        for name in ('references.json','comparison_predictions.jsonl','raw_experts.npy','loo_text.npy','scoring_checks.json'):
            files.append(use(folder/name))
        inputs[domain]=list(dict.fromkeys(common+files))
    for job,spec in JOBS.items():
        folder=OUT/'jobs'/job;folder.mkdir(parents=True)
        write(folder/'status.json',dict(state='PREPARED',**spec,users_completed=0,users_total=256,updated_at=stamp()))
    for rel in manifest:
        source=ROOT/rel
        if source.suffix in ('.py','.sh','.md'):
            dest=OUT/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=inputs))
    write(OUT/'status.json',dict(state='PREPARED',users_completed=0,users_total=2048,
        fresh_generations_total=12288,updated_at=stamp(),test_read=False))
    REPORT.write_text('# Qwen4开关补齐与最终文本专家匹配筛选\n\n状态：PREPARED；尚未产生Qwen4修正版本的效果结论。\n')
    print(json.dumps(dict(state='PREPARED',hashes=len(manifest),users_per_domain=1024,fresh_generations=12288)),flush=True)


class Generation(A.Generation):
    def scores(self,j):
        return C.scores(self.raw[j],self.loo[j],self.truth[j]['full_history'])

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU memory lease exited'
        rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json',dict(state=state,stage=self.stage,job=self.job,
            domain=self.spec['domain'],shard=self.spec['shard'],pid=os.getpid(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),users_completed=self.users_completed,users_total=256,
            completed_decodes=self.completed,fresh_generations_completed=self.users_completed*6,
            median_seconds_per_user=rate,remaining_generation_seconds_estimate=(256-self.users_completed)*rate if rate else None,
            updated_at=stamp(),history_filter=False,training_steps=0,test_read=False,**extra))

    def run(self):
        ix=indices(self.spec['shard']);self.stage='matching_native_singles_pair_smoke';self.status();checks=[]
        for j in (ix[0],ix[len(ix)//2],ix[-1]):
            ctx=self.context(j);native=self.decode(ctx)
            B.Generation.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),self.refs[j]['native'])
            for name,value in [('sas_only',self.raw[j,:,0]),('qwen_only',self.raw[j,:,1]),('pair_raw',self.scores(j)['pair_qwen_raw'])]:
                q=self.index.conditional_logs(value);decoded=self.decode(ctx,q,.3)
                B.Generation.equal(dict(ids=decoded['ids'],scores=decoded['scores'].tolist()),self.refs[j][name])
            zero=self.decode(ctx,q,0.)
            assert zero['paths']==native['paths'];np.testing.assert_array_equal(zero['scores'],native['scores'])
            plain=self.decode(ctx,native_entry=True,record=False)
            assert plain['paths']==native['paths'];np.testing.assert_array_equal(plain['scores'],native['scores'])
            self.samples[j]['target']=2;other=self.context(j);self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],other['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],other['mask'],atol=0,rtol=0)
            checks.append(dict(row=j,native_and_sas_qwen_pair_references_match=True,alpha_zero_exact=True,
                native_entry_exact=True,target_placeholder_invariant=True))
        write(self.jobfolder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='six_qwen_switch_generations';self.status()
        with (self.jobfolder/'predictions.jsonl').open('x') as stream:
            for j in ix:
                began=time.monotonic();ctx=self.context(j);s=self.truth[j]
                row=dict(sample_index=j,user_id=s['user_id'],target=s['target'],history=s['history'],
                    evaluation_block=s['evaluation_block'],standalone_ranks={},diagnostics={})
                for name,value in self.scores(j).items():
                    if name in C.FRESH:
                        decoded=self.decode(ctx,self.index.conditional_logs(value),.3)
                        row[name]=dict(ids=decoded['ids'],scores=decoded['scores'].tolist())
                    row['standalone_ranks'][name]=F.full_rank(value,s['target'])
                    weights=np.exp(20*(value.astype(float)-float(value.max())))
                    mask=np.isin(np.arange(1,len(value)+1),s['full_history'])
                    row['diagnostics'][name]=dict(expert_history_probability=float(weights[mask].sum()/weights.sum()))
                stream.write(json.dumps(row)+'\n');stream.flush()
                self.users_completed+=1;self.row_seconds.append(time.monotonic()-began)
                if self.users_completed==1 or self.users_completed%8==0 or j==ix[-1]:
                    os.fsync(stream.fileno());self.status()
        assert self.digest()==self.initial_digest
        write(self.jobfolder/'runtime_checks.json',dict(state='PASSED',users=256,all_six_fresh_arms=True,
            gram_digest_unchanged=True,targets_only_for_metrics=True,history_filter=False,training_steps=0,test_read=False))
        self.stage='complete';self.status('COMPLETED')


def worker(job):
    F.OUT,F.CFG,F.JOBS,F.MODULE=OUT,CFG,JOBS,MODULE
    return F.worker(job)


def summarize():
    domains={}
    for domain in DOMAINS:
        folder=OUT/domain;samples=read(folder/'samples.json')
        manifest=read(OUT/'manifest.json')
        for rel in manifest['domain_inputs'][domain]:assert sha(ROOT/rel)==manifest['sha256'][rel],rel
        baselines=[json.loads(line) for line in (folder/'comparison_predictions.jsonl').read_text().splitlines()]
        rows={}
        for job,spec in JOBS.items():
            if spec['domain']!=domain:continue
            path=OUT/'jobs'/job
            assert read(path/'status.json')['state']=='COMPLETED'
            assert read(path/'runtime_checks.json')['state']=='PASSED' and read(path/'lease_status.json')['state']=='released'
            got=[json.loads(line) for line in (path/'predictions.jsonl').read_text().splitlines()]
            assert [r['sample_index'] for r in got]==indices(spec['shard'])
            for row in got:
                assert row['sample_index'] not in rows;rows[row['sample_index']]=row
        assert sorted(rows)==list(range(1024))
        methods=[k for k,v in baselines[0].items() if isinstance(v,dict) and 'ids' in v]+list(C.FRESH)
        rr={k:[] for k in methods};pure={a:[] for a in C.ARMS};history={k:[] for k in methods}
        raw=np.load(folder/'raw_experts.npy',mmap_mode='r');loo=np.load(folder/'loo_text.npy',mmap_mode='r')
        with (folder/'predictions.jsonl').open('x') as stream:
            for j,s in enumerate(samples):
                fresh=rows[j];baseline=baselines[j]
                assert all(fresh[k]==baseline[k]==s[k] for k in ('user_id','target','history','evaluation_block'))
                combined=dict(baseline,**{k:v for k,v in fresh.items() if k not in baseline})
                for arm in methods:
                    prediction=combined[arm];ids,ss=prediction['ids'],prediction['scores']
                    assert len(ids)==len(set(ids))==len(ss)==50 and np.isfinite(ss).all()
                    items=[i for i,_ in sorted(zip(ids,ss),key=lambda p:(-p[1],p[0]))]
                    rank=items.index(s['target'])+1 if s['target'] in items else 2**30
                    assert rank==B.rank(prediction,s['target']);rr[arm].append(rank)
                    history[arm].append(float(np.isin(items[:10],s['full_history']).mean()))
                for arm,value in C.scores(raw[j],loo[j],s['full_history']).items():
                    rank=F.full_rank(value,s['target']);assert rank==fresh['standalone_ranks'][arm];pure[arm].append(rank)
                stream.write(json.dumps(combined)+'\n')
        rr={k:np.asarray(v) for k,v in rr.items()};pure={k:np.asarray(v) for k,v in pure.items()}
        np.savez_compressed(folder/'ranks.npz',users=np.array([s['user_id'] for s in samples]),**rr)
        np.savez_compressed(folder/'standalone_ranks.npz',**pure)
        groups=dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024))
        blocks=np.array([s['evaluation_block'] for s in samples]);groups.update({f'block{k+1}':np.flatnonzero(blocks==k) for k in range(5)})
        pairs=[]
        for q,b,sas,qt,bt in SETTINGS.values():pairs.extend([(q,b),(q,sas),(q,qt)])
        pairs += [(f'{prefix}_loo',f'{prefix}_raw') for prefix in ('pair_qwen','qwen')]
        pairs += [(f'{prefix}_loo_penalty',f'{prefix}_raw_penalty') for prefix in ('pair_qwen','qwen')]
        pairs += [(f'{prefix}_raw_penalty',f'{prefix}_raw') for prefix in ('pair_qwen','qwen')]
        pairs += [(f'{prefix}_loo_penalty',f'{prefix}_loo') for prefix in ('pair_qwen','qwen')]
        pairs += [('pair_bge_loo_penalty','sas_penalty'),('pair_bge_loo_penalty','bge_loo_penalty')]
        domains[domain]={}
        for group,ix in groups.items():
            domains[domain][group]=dict(users=len(ix),metrics={a:metrics(v[ix]) for a,v in rr.items()},
                standalone_metrics={a:metrics(v[ix]) for a,v in pure.items()},
                mean_history_fraction_top10={a:float(np.asarray(v)[ix].mean()) for a,v in history.items()},
                comparisons={a+'_vs_'+b:contrast(rr[a][ix],rr[b][ix],2000,20261017) for a,b in pairs})
        write(folder/'summary.json',domains[domain])
        paths=[folder/name for name in ('predictions.jsonl','ranks.npz','standalone_ranks.npz','summary.json')]
        paths += [OUT/'jobs'/job/name for job,spec in JOBS.items() if spec['domain']==domain
                  for name in ('predictions.jsonl','smoke.json','runtime_checks.json','lease_status.json')]
        write(folder/'output_manifest.json',dict(sha256={str(path.relative_to(ROOT)):sha(path) for path in paths}))
    choice=C.choose({d:values['all1024']['metrics'] for d,values in domains.items()})
    choice['corrected_qwen_minus_bge']={d:values['all1024']['comparisons']['pair_qwen_loo_penalty_vs_pair_bge_loo_penalty'] for d,values in domains.items()}
    write(OUT/'text_expert_decision.json',choice)
    write(OUT/'summary.json',dict(domains=domains,selection=choice,config=CFG,settings=SETTINGS,
        test_read=False,intervals='unadjusted descriptive paired-user bootstrap on reused development users',
        statistical_review=read(LONG/'summary.json')['statistical_review'],next_experiment_started=False))
    lines=['# Qwen4开关补齐与最终文本专家匹配筛选','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run / validate',
        '- Verification Status: ANALYZED; saved ranks independently recomputed','',
        '每域1024相同用户、α=.3、双专家β各.5、单专家1，惩罚.05，历史商品始终合法。','']
    for domain in DOMAINS:
        mm=domains[domain]['all1024']['metrics'];lines += [f'## {domain}','','| 设置 | SAS+Qwen4 N10 | SAS+BGE N10 | 匹配SASRec N10 | Qwen4单 N10 | BGE单 N10 |','|---|---:|---:|---:|---:|---:|']
        for setting,arms in SETTINGS.items():lines.append('| '+setting+' | '+' | '.join(f'{mm[a]["N10"]:.6f}' for a in arms)+' |')
        c=choice['corrected_qwen_minus_bge'][domain]
        lines += ['',f'两项都开时Qwen4组合减BGE组合N10={c["delta_N10"]:+.6f}，描述性95%区间={c["descriptive_95pct_N10"]}。','']
    chosen=choice['selected_development_text_expert'];candidate=choice['candidates'][chosen]
    lines += ['## 统一文本专家选择','',f'按预定最差域优先规则，当前协议的开发候选选择：{chosen}。',
        '两域均值是否均超过两个匹配单专家：'+str(candidate['both_domains_above_both_singles_by_mean'])+'。',
        '这是现有模型、两开关固定条件下的候选选择；区间跨零不能解释为已证明优越，均值未达标也不宣布双专家目标达成。',
        '逐域、前后512、五块、全部19种预测和8种Qwen纯专家排名均保存。前后批次/五块不是独立复现。',
        '全部比较未经多重校正，重复开发用户、固定单个上游模型及未覆盖训练种子的不确定性保留；不是独立test确认。',
        '评分/编码不使用目标，所有预定用户和负结果保留；无训练、无新文本编码、无历史过滤。',
        '统计谬误11/11沿用固定专家筛选审计并按本轮分域/匹配对照解释；不以跨域平均掩盖弱域，不挑开关或最好分组，不把数值复算当训练复现。']
    REPORT.write_text('\n'.join(lines)+'\n')


def available(inventory,active):
    free={g['uuid']:float(g['free_mib']) for g in inventory}
    for job,task in active.items():
        if not (OUT/'jobs'/job/'smoke.json').exists():free[task['gpu']['uuid']]-=CFG['total_lease_mib']
    return sorted([dict(g,effective_free_mib=free[g['uuid']]) for g in inventory if free[g['uuid']]>=CFG['minimum_free_mib']],
                  key=lambda g:(g['utilization'],-g['effective_free_mib'],g['index']))


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    pending,active,finished=list(JOBS),{},{}
    attempts={job:0 for job in JOBS};began=time.monotonic()
    write(OUT/'runtime_resource_policy.json',dict(project_gpu_task_cap=None,share_gpu_if_memory_available=True,
        older_pending_blocks_admission=False,minimum_free_mib=18432,total_lease_mib=16384,
        authorization='不要设限gpu gpu有位置就可以用'))
    try:
        while pending or active:
            for job,task in list(active.items()):
                code=task['process'].poll()
                if code is None:continue
                task['log'].close();status=read(OUT/'jobs'/job/'status.json')
                if code==75 and status['state']=='ADMISSION_DEFERRED':pending.append(job)
                else:
                    lease_path=OUT/'jobs'/job/'lease_status.json'
                    finished[job]=dict(exit_code=code,state=status['state'],lease_released=read(lease_path)['state']=='released' if lease_path.exists() else True)
                del active[job]
            inventory=B.S.OLD.gpu_inventory()
            for job in list(pending):
                choices=available(inventory,active)
                if not choices:break
                gpu=choices[0];pending.remove(job);attempts[job]+=1
                log=(OUT/'jobs'/job/f'worker_attempt{attempts[job]}.log').open('x')
                write(OUT/'jobs'/job/'status.json',dict(state='STARTING',**JOBS[job],users_completed=0,users_total=256,
                    gpu_index=gpu['index'],gpu_uuid=gpu['uuid'],updated_at=stamp()))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',job],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[job]=dict(process=process,log=log,gpu=gpu)
            if pending and time.monotonic()-began>CFG['resource_wait_seconds']:
                for job in pending:finished[job]=dict(state='RESOURCE_WAIT_LIMIT',exit_code=124,lease_released=True)
                pending=[]
            statuses={job:read(OUT/'jobs'/job/'status.json') for job in JOBS}
            rates=[s['median_seconds_per_user'] for s in statuses.values() if s.get('median_seconds_per_user')]
            estimate=sum((256-s.get('users_completed',0))*(s.get('median_seconds_per_user') or float(np.median(rates))) for s in statuses.values())/max(1,len(active)) if rates else None
            write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_FREE_GPU_MEMORY' if pending else 'FINALIZING',
                users_completed=sum(s.get('users_completed',0) for s in statuses.values()),users_total=2048,
                fresh_generations_total=12288,remaining_generation_seconds_estimate=estimate,pending=pending,finished=finished,
                active={job:dict(pid=t['process'].pid,gpu_index=t['gpu']['index'],gpu_uuid=t['gpu']['uuid']) for job,t in active.items()},
                jobs=statuses,project_gpu_task_cap=None,pid=os.getpid(),updated_at=stamp(),test_read=False))
            if pending or active:time.sleep(5)
        complete=len(finished)==len(JOBS) and all(s['exit_code']==0 and s['state']=='COMPLETED' and s['lease_released'] for s in finished.values())
        if complete:summarize()
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
            users_completed=2048 if complete else sum(s.get('users_completed',0) for s in statuses.values()),users_total=2048,
            updated_at=stamp(),report=str(REPORT.relative_to(ROOT)),next_experiment_started=False,test_read=False))
    finally:
        for task in active.values():B.stop(task['process']);task['log'].close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','supervise','worker','generate','review'))
    parser.add_argument('--job',choices=JOBS);parser.add_argument('--lease-pid',type=int);args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Qwen switch run interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='test':print(json.dumps(tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='worker':return worker(args.job)
        elif args.mode=='review':summarize()
        else:
            # Let CLEAN.Worker initialize torch inter-op threads exactly once.
            A.OUT,A.CFG,A.JOBS=OUT,CFG,JOBS
            Generation(args.job,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            folder=OUT/'jobs'/args.job if args.job else OUT
            value=dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(folder/'failure.json',value);write(folder/'status.json',value)
        raise


if __name__=='__main__':
    sys.exit(main())
