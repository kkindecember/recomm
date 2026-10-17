"""Supplement matching single-expert switches without changing active runs."""
import argparse
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
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT=A.ROOT
SOURCE=A.OUT
ORIGINAL=A.SOURCE
OUT=ROOT/'artifacts/phase21/single_switch_controls_1024_20261017_v1'
PLAN=ROOT/'plan/第二十一阶段/21.52_单专家历史惩罚与文本自匹配匹配控制_20261017.md'
REPORT=ROOT/'report/第二十一阶段/单专家开关与双专家匹配控制_20261017.md'
MODULE='experiment.phase21.single_switch_controls'
DOMAINS=A.DOMAINS
JOBS=A.JOBS
ARMS=('sas_penalty','bge_loo','bge_raw_penalty','bge_loo_penalty')
CFG=dict(A.CFG,arms=list(ARMS),fresh_arms=list(ARMS),max_parallel=2,global_project_gpu_cap=4,
    source=str(SOURCE.relative_to(ROOT)),beta_single=1.,stop_after='complete matching single switches; joint review if source complete')
read,write,sha,stamp=A.read,A.write,A.sha,A.stamp


def scores(raw,loo,history,penalty=.05):
    result=dict(sas_penalty=np.asarray(raw[:,0]).copy(),bge_loo=np.asarray(loo).copy(),
        bge_raw_penalty=np.asarray(raw[:,1]).copy(),bge_loo_penalty=np.asarray(loo).copy())
    ids=np.unique(np.asarray(history,int))-1
    for key in ('sas_penalty','bge_raw_penalty','bge_loo_penalty'):
        result[key][ids]-=np.float32(penalty)
    return result


def tests():
    raw=np.arange(14,dtype=np.float32).reshape(7,2)/20
    loo=raw[:,1]*.8
    values=scores(raw,loo,[1,7])
    keep=np.array([False,True,True,True,True,True,False])
    for arm,base in [('sas_penalty',raw[:,0]),('bge_raw_penalty',raw[:,1]),('bge_loo_penalty',loo)]:
        np.testing.assert_array_equal(values[arm][keep],base[keep])
        np.testing.assert_allclose(values[arm][~keep],base[~keep]-.05,atol=3e-8)
    np.testing.assert_array_equal(values['bge_loo'],loo)
    zero=scores(raw,loo,[1,7],0.)
    np.testing.assert_array_equal(zero['sas_penalty'],raw[:,0])
    np.testing.assert_array_equal(zero['bge_raw_penalty'],raw[:,1])
    return dict(state='PASSED',matching_score_formulas=True,full_history_penalty=True,zero_penalty_exact=True)


def prepare():
    checks=tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',CFG)
    write(OUT/'tests.json',checks)
    shutil.copy2(PLAN,OUT/'protocol.md')
    expected=read(SOURCE/'manifest.json')['sha256']
    manifest={}
    def use(path,digest=None):
        rel=str(path.relative_to(ROOT))
        value=manifest.get(rel) or sha(path)
        assert digest is None or digest==value,rel
        manifest[rel]=value
        return rel
    common=[use(ROOT/rel,digest) for rel,digest in expected.items()
            if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/'))]
    for path in (Path(__file__),PLAN,OUT/'config.json'):
        rel=use(path)
        if rel not in common:common.append(rel)
    inputs={}
    for d in DOMAINS:
        folder=OUT/d
        folder.mkdir()
        files=[]
        for name in ('samples.json','references.json','catalog.json','data_config.json','teacher_checks.json'):
            source=SOURCE/d/name
            files.append(use(source,expected[str(source.relative_to(ROOT))]))
            shutil.copy2(source,folder/name)
            files.append(use(folder/name))
        for name in ('raw_experts.npy','loo_text.npy'):
            source=SOURCE/d/name
            files.append(use(source,expected[str(source.relative_to(ROOT))]))
            (folder/name).symlink_to(source)
            files.append(use(folder/name))
        samples=read(folder/'samples.json')
        assert len(samples)==len({s['user_id'] for s in samples})==1024
        inputs[d]=common+files
    for job,spec in JOBS.items():
        p=OUT/'jobs'/job
        p.mkdir(parents=True)
        write(p/'status.json',dict(state='PREPARED',**spec,users_completed=0,users_total=512,updated_at=stamp()))
    for rel in manifest:
        source=ROOT/rel
        if source.suffix in ('.py','.md','.sh'):
            dest=OUT/'sources'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=inputs))
    write(OUT/'status.json',dict(state='PREPARED',users_completed=0,users_total=2048,
        fresh_generations_total=8192,updated_at=stamp(),test_read=False))
    REPORT.write_text('# 单专家开关与双专家匹配控制\n\n状态：PREPARED，等待新增生成。\n')
    print(json.dumps(dict(state='PREPARED',checks=checks,hashes=len(manifest))),flush=True)


class Generation(A.Generation):
    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None
        rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json',dict(state=state,stage=self.stage,job=self.job,
            domain=self.spec['domain'],pid=os.getpid(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            users_completed=self.users_completed,users_total=512,completed_decodes=self.completed,
            fresh_generations_completed=self.users_completed*4,median_seconds_per_user=rate,
            remaining_generation_seconds_estimate=(512-self.users_completed)*rate if rate else None,
            updated_at=stamp(),history_filter=False,training_steps=0,test_read=False,**extra))

    def run(self):
        ix=A.indices(self.spec['shard'])
        self.stage='matching_single_smoke'
        self.status()
        checks=[]
        for j in (ix[0],ix[len(ix)//2],ix[-1]):
            ctx=self.context(j)
            native=self.decode(ctx)
            A.B.Generation.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),self.refs[j]['native'])
            for column,key in ((0,'sas_only'),(1,'bge_only')):
                q=self.index.conditional_logs(self.raw[j,:,column])
                prediction=self.decode(ctx,q,.3)
                A.B.Generation.equal(dict(ids=prediction['ids'],scores=prediction['scores'].tolist()),self.refs[j][key])
            zero=self.decode(ctx,q,0.)
            assert zero['paths']==native['paths']
            np.testing.assert_array_equal(zero['scores'],native['scores'])
            self.samples[j]['target']=2
            changed=self.context(j)
            self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],changed['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],atol=0,rtol=0)
            checks.append(dict(row=j,native_and_both_singles_match=True,alpha_zero_exact=True,target_placeholder_invariant=True))
        write(self.jobfolder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='matching_single_switch_generation'
        self.status()
        with (self.jobfolder/'predictions.jsonl').open('x') as stream:
            for j in ix:
                began=time.monotonic()
                ctx=self.context(j)
                s=self.truth[j]
                row=dict(sample_index=j,user_id=s['user_id'],target=s['target'],history=s['history'],
                    evaluation_block=s['evaluation_block'],standalone_ranks={})
                for arm,value in scores(self.raw[j],self.loo[j],s['full_history']).items():
                    decoded=self.decode(ctx,self.index.conditional_logs(value),.3)
                    row[arm]=dict(ids=decoded['ids'],scores=decoded['scores'].tolist())
                    row['standalone_ranks'][arm]=A.F.full_rank(value,s['target'])
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                self.users_completed+=1
                self.row_seconds.append(time.monotonic()-began)
                if self.users_completed==1 or self.users_completed%8==0 or j==ix[-1]:
                    os.fsync(stream.fileno())
                    self.status()
        assert self.digest()==self.initial_digest
        write(self.jobfolder/'runtime_checks.json',dict(state='PASSED',users=512,gram_digest_unchanged=True,
            all_four_single_switches=True,history_filter=False,test_read=False))
        self.stage='complete'
        self.status('COMPLETED')


def bind():
    A.OUT,A.CFG,A.JOBS=OUT,CFG,JOBS


def worker(job):
    A.F.OUT,A.F.CFG,A.F.JOBS,A.F.MODULE=OUT,CFG,JOBS,MODULE
    return A.F.worker(job)


def external_active():
    active=[]
    for source in (ORIGINAL,SOURCE):
        s=read(source/'status.json')
        active.extend(s.get('active',{}).values())
    return active


def summarize():
    domains={}
    for d in DOMAINS:
        folder=OUT/d
        samples=read(folder/'samples.json')
        rows={}
        for job,spec in JOBS.items():
            if spec['domain']!=d:continue
            p=OUT/'jobs'/job
            assert read(p/'runtime_checks.json')['state']=='PASSED'
            assert read(p/'lease_status.json')['state']=='released'
            for line in (p/'predictions.jsonl').read_text().splitlines():
                row=json.loads(line)
                assert row['sample_index'] not in rows
                rows[row['sample_index']]=row
        assert sorted(rows)==list(range(1024))
        raw=np.load(folder/'raw_experts.npy',mmap_mode='r')
        loo=np.load(folder/'loo_text.npy',mmap_mode='r')
        rr={a:[] for a in ARMS}
        with (folder/'predictions.jsonl').open('x') as stream:
            for j,s in enumerate(samples):
                row=rows[j]
                assert all(row[k]==s[k] for k in ('user_id','target','history','evaluation_block'))
                values=scores(raw[j],loo[j],s['full_history'])
                for arm in ARMS:
                    rr[arm].append(A.B.rank(row[arm],s['target']))
                    assert A.F.full_rank(values[arm],s['target'])==row['standalone_ranks'][arm]
                stream.write(json.dumps(row)+'\n')
        np.savez_compressed(folder/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**rr)
        domains[d]={a:metrics(v) for a,v in rr.items()}
        paths=[folder/'predictions.jsonl',folder/'ranks.npz']
        write(folder/'output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    write(OUT/'summary.json',dict(domains=domains,test_read=False,matching_pair_review=joint_review()))


def joint_review():
    if read(SOURCE/'status.json')['state']!='COMPLETED':
        write(OUT/'joint_review_status.json',dict(state='WAITING_FOR_PAIR_EXPERIMENT',updated_at=stamp(),source=str(SOURCE)))
        return 'WAITING_FOR_PAIR_EXPERIMENT'
    result={}
    lines=['# 单专家开关与双专家匹配控制','','各设置均比较双专家与两个匹配单专家；α=.3、单专家β=1、双专家β各.5，历史商品始终合法。','']
    settings={'raw_none':('pair_raw','sas_only','bge_only'),
        'loo_none':('pair_loo','sas_only','bge_loo'),
        'raw_penalty':('pair_raw_penalty','sas_penalty','bge_raw_penalty'),
        'loo_penalty':('pair_loo_penalty','sas_penalty','bge_loo_penalty')}
    for d in DOMAINS:
        p=OUT/d
        samples=read(p/'samples.json')
        pairrows=[json.loads(x) for x in (SOURCE/d/'predictions.jsonl').read_text().splitlines()]
        singles=dict(np.load(p/'ranks.npz'))
        singles.pop('users')
        rr={k:np.asarray(v) for k,v in singles.items()}
        for arm in ('native','sas_only','bge_only','pair_raw','pair_loo','pair_raw_penalty','pair_loo_penalty'):
            rr[arm]=np.asarray([A.B.rank(row[arm],s['target']) for row,s in zip(pairrows,samples)])
        assert len(pairrows)==len(samples)==1024 and all(r['user_id']==s['user_id'] for r,s in zip(pairrows,samples))
        groups=dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024))
        blocks=np.asarray([s['evaluation_block'] for s in samples])
        groups.update({f'block{b+1}':np.flatnonzero(blocks==b) for b in range(5)})
        result[d]={}
        for name,ix in groups.items():
            result[d][name]=dict(users=len(ix),settings={key:dict(metrics={a:metrics(rr[a][ix]) for a in arms},
                comparisons={arms[0]+'_vs_'+b:contrast(rr[arms[0]][ix],rr[b][ix],repeats=2000,seed=20261017) for b in arms[1:]}) for key,arms in settings.items()})
        lines += [f'## {d}','','| 设置 | 双专家 N10 | SASRec单专家 N10 | BGE单专家 N10 |','|---|---:|---:|---:|']
        for key,arms in settings.items():
            table=result[d]['all1024']['settings'][key]['metrics']
            lines.append(f'| {key} | {table[arms[0]]["N10"]:.6f} | {table[arms[1]]["N10"]:.6f} | {table[arms[2]]["N10"]:.6f} |')
        lines.append('')
    write(OUT/'joint_review.json',dict(domains=result,test_read=False,settings=settings,
        intervals='descriptive unadjusted paired bootstrap, reused development cohort',independent_test=False))
    write(OUT/'joint_review_status.json',dict(state='COMPLETED',updated_at=stamp()))
    lines.append('这里只比较当前固定权重协议，不代表每种方法都已最优校准；配对区间为开发集描述性证据，未经多重比较校正。')
    REPORT.write_text('\n'.join(lines)+'\n')
    return 'COMPLETED'


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    pending=list(JOBS)
    active={}
    finished={}
    attempts={j:0 for j in JOBS}
    began=time.monotonic()
    try:
        while pending or active:
            for job,t in list(active.items()):
                code=t['process'].poll()
                if code is not None:
                    t['log'].close()
                    p=OUT/'jobs'/job
                    s=read(p/'status.json')
                    if code==75 and s['state']=='ADMISSION_DEFERRED':pending.append(job)
                    else:finished[job]=dict(exit_code=code,state=s['state'],lease_released=read(p/'lease_status.json')['state']=='released' if (p/'lease_status.json').exists() else True)
                    del active[job]
            external=external_active()
            excluded={t['gpu_uuid'] for t in external}|{t['gpu']['uuid'] for t in active.values()}
            available=sorted([g for g in A.B.S.OLD.gpu_inventory() if g['uuid'] not in excluded and g['free_mib']>=CFG['minimum_free_mib']],key=lambda g:(g['utilization'],-g['free_mib'],g['index']))
            for job in list(pending):
                if not available or len(active)>=CFG['max_parallel'] or len(active)+len(external)>=CFG['global_project_gpu_cap']:break
                gpu=available.pop(0)
                pending.remove(job)
                attempts[job]+=1
                log=(OUT/'jobs'/job/f'worker_attempt{attempts[job]}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                write(OUT/'jobs'/job/'status.json',dict(state='STARTING',users_completed=0,users_total=512,updated_at=stamp(),gpu_index=gpu['index']))
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',job],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[job]=dict(process=process,log=log,gpu=gpu)
            if pending and time.monotonic()-began>CFG['resource_wait_seconds']:
                for job in pending:finished[job]=dict(exit_code=124,state='RESOURCE_WAIT_LIMIT',lease_released=True)
                pending=[]
            statuses={j:read(OUT/'jobs'/j/'status.json') for j in JOBS}
            rates=[s.get('median_seconds_per_user') for s in statuses.values() if s.get('median_seconds_per_user')]
            remaining=sum((512-s.get('users_completed',0))*(s.get('median_seconds_per_user') or float(np.median(rates))) for s in statuses.values())/max(1,len(active)) if rates else None
            write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',updated_at=stamp(),pid=os.getpid(),
                users_completed=sum(s.get('users_completed',0) for s in statuses.values()),users_total=2048,fresh_generations_total=8192,
                remaining_generation_seconds_estimate=remaining,pending=pending,finished=finished,jobs=statuses,
                active={j:dict(pid=t['process'].pid,gpu_index=t['gpu']['index'],gpu_uuid=t['gpu']['uuid']) for j,t in active.items()},test_read=False))
            if pending or active:time.sleep(10)
        complete=len(finished)==len(JOBS) and all(t['exit_code']==0 and t['state']=='COMPLETED' and t['lease_released'] for t in finished.values())
        if complete:summarize()
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,updated_at=stamp(),
            users_completed=2048 if complete else sum(s.get('users_completed',0) for s in statuses.values()),users_total=2048,test_read=False,
            report=str(REPORT.relative_to(ROOT)),next_experiment_started=False))
    finally:
        for t in active.values():A.B.stop(t['process']);t['log'].close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','supervise','worker','generate','review'))
    parser.add_argument('--job',choices=JOBS)
    parser.add_argument('--lease-pid',type=int)
    args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Single switch control interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='test':print(json.dumps(tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='review':joint_review()
        elif args.mode=='worker':return worker(args.job)
        else:bind();Generation(args.job,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            p=OUT/'jobs'/args.job if args.job else OUT
            value=dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(p/'failure.json',value);write(p/'status.json',value)
        raise


if __name__=='__main__':sys.exit(main())
