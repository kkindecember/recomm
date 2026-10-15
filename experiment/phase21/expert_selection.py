"""Expert selection only: frozen dual baseline vs matched refitted singles.

Process-local adapter of existing ablation code. No candidate-weight training.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment.phase21 import text_contribution as T
from experiment.phase20.catalog_scope_core import metrics,contrast

ROOT=T.ROOT
OUT=ROOT/'artifacts/phase21/expert_selection_20261014_v1'
PLAN=ROOT/'plan/第二十一阶段/21.21_先确定双专家组合_20261014.md'
REPORT=ROOT/'report/第二十一阶段/双专家组合选择_20261014.md'
DUAL=ROOT/'artifacts/phase21/expert_resource_recovery_20260923_v1/removal/without_itemcf'
ARMS=('sas_only','text_only')
KEPT={'sas_only':0,'text_only':2}
GPUS=['GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc','GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253']
read,write,sha,stamp=T.read,T.write,T.sha,T.stamp


def transform_scores(raw,arm,donors=None):
    result=np.zeros_like(raw,dtype=np.float32)
    result[:,KEPT[arm]]=raw[:,KEPT[arm]]
    return result


def fuse(gate,scores,lengths,history,arm):
    from scripts.evaluation.run_repeat_aware_gate import features
    x=features(scores,lengths,history).clone()
    keep=KEPT[arm]
    for e in range(3):
        if e!=keep:x[:,[1+e,4+e,7+e,10+e]]=0.
    weights,penalty=gate(x)
    mask=weights.new_zeros(3);mask[keep]=1.
    weights=weights*mask;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
    return (weights[:,:,None]*scores).sum(1)-penalty[:,None]*history,x,weights,penalty


def bind():
    T.OUT=OUT;T.PLAN=PLAN;T.REPORT=REPORT;T.ARMS=ARMS
    T.GPUS=dict(sas_only=GPUS[0],text_only=GPUS[1])
    T.transform_scores=transform_scores;T.fuse=fuse


def tests():
    import torch
    sys.path.insert(0,str(ROOT/'third_party/LIME-Rec'))
    from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate
    torch.manual_seed(0);gate=RepeatAwareGate(14,[.6,.15,.25],.1)
    raw=np.random.default_rng(3).random((2,3,17),dtype=np.float32)
    lengths=torch.tensor([2.,3.]);history=torch.zeros(2,17,dtype=torch.bool);history[:,0]=True
    for arm in ARMS:
        changed=raw.copy()
        for e in range(3):
            if e!=KEPT[arm]:changed[:,e]=500.
        a=fuse(gate,torch.from_numpy(transform_scores(raw,arm)),lengths,history,arm)
        b=fuse(gate,torch.from_numpy(transform_scores(changed,arm)),lengths,history,arm)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
        torch.testing.assert_close(a[2][:,KEPT[arm]],torch.ones(2),atol=0,rtol=0)
        for e in range(3):
            if e!=KEPT[arm]:assert torch.count_nonzero(a[1][:,[1+e,4+e,7+e,10+e]])==0
    return dict(dropped_score_and_feature_invariance=True,single_weight_one=True)


def prepare():
    bind();checks=tests();T.prepare()
    cfg=read(OUT/'config.json')
    cfg.update(worker_timeout_seconds=21600,supervisor_timeout_seconds=86400,resource_wait_seconds=86400,
               minimum_free_mib=32768,total_lease_mib=30720,expected_workload_peak_mib=12288,max_parallel=2,
               candidate_weight_experiment=False,forecast_policy='advisory_only',gpu_pool=GPUS)
    write(OUT/'config.json',cfg)
    for arm in ARMS:write(OUT/arm/'config.json',cfg)
    manifest=read(OUT/'manifest.json')['sha256']
    for d in ('Toys','Beauty'):
        for name in ('predictions.jsonl','ranks.npz','summary.json','data_config.json','gate.pt','fit_checks.json'):
            p=DUAL/d/name;manifest[str(p.relative_to(ROOT))]=sha(p)
        expected=[s for s in read(OUT/ARMS[0]/d/'samples.json') if s['role']=='development']
        rr=np.load(DUAL/d/'ranks.npz')
        np.testing.assert_array_equal(rr['users'],np.array([s['user_id'] for s in expected]))
    for p in [Path(__file__),ROOT/'experiment/phase21/run_expert_selection.sh',ROOT/'experiment/gpu_memory_lease.py',ROOT/'experiment/GPU_MEMORY_LEASE_POLICY.md',OUT/'config.json',*[OUT/a/'config.json' for a in ARMS]]:
        manifest[str(p.relative_to(ROOT))]=sha(p)
        if p.suffix in ('.py','.sh','.md'):
            import shutil
            dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,created_at=stamp()))
    write(OUT/'tests.json',checks)
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),scope='expert_selection_only',test_read=False))
    REPORT.write_text('# 双专家组合选择\n\n状态：PREPARED。旧双专家参照已核验；单专家真实生成尚未完成。仅确定专家，不继续候选权重。\n')


class Worker(T.Worker):
    def status(self,state='RUNNING',**values):
        assert self.lease.poll() is None,'GPU lease exited unexpectedly'
        super().status(state,**values)

    def smoke_control(self):
        self.stage='native_zero_and_dual_parity';self.status()
        longest=max(self.development,key=lambda j:len(self.samples[j]['history']))
        js=list(dict.fromkeys([longest,*self.development]))[:3]
        dual=np.load(DUAL/self.domain/'fused.npy',mmap_mode='r')
        dual_rows={r['user_id']:r for r in T.clean.rows(DUAL/self.domain/'predictions.jsonl')}
        checks=[]
        for j in js:
            s=self.samples[j];ctx=self.context(j);native=self.decode(ctx)
            q=self.index.conditional_logs(self.fused[j]);zero=self.decode(ctx,q,0.)
            assert native['paths']==zero['paths'];np.testing.assert_array_equal(native['scores'],zero['scores'])
            ref=self.references[s['user_id']]
            assert native['paths']==ref['paths'];np.testing.assert_array_equal(native['scores'],ref['native'])
            dq=self.index.conditional_logs(dual[j]);old=self.decode(ctx,dq,.3)
            saved=dual_rows[s['user_id']]['guided']
            assert old['ids']==saved['ids'];np.testing.assert_allclose(old['scores'],saved['scores'],atol=2e-6,rtol=1e-5)
            checks.append(dict(user=s['user_id'],native_exact=True,alpha_zero_exact=True,dual_baseline_reproduced=True))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))


def worker(arm,domain):
    bind();folder=OUT/arm/domain;cfg=read(OUT/'config.json');lease=None
    def interrupted(signum,frame):raise InterruptedError('Worker received termination')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for rel,digest in read(OUT/'manifest.json')['sha256'].items():
            if rel.startswith(('experiment/','third_party/','GRAM/src/')):assert sha(ROOT/rel)==digest,rel
        dc=read(folder/'data_config.json');assert sha(ROOT/dc['parent_checkpoint'])==dc['parent_sha256']
        with (folder/'lease.log').open('x') as log:
            lease_path=folder/'lease_status.json'
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0','--total-lease-mib','30720','--expected-workload-peak-mib','12288','--status-path',str(lease_path)],stdout=log,stderr=subprocess.STDOUT)
            begin=time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state')!='holding':
                assert lease.poll() is None,'GPU lease startup failed';assert time.monotonic()-begin<180
                time.sleep(1)
            Worker.lease=lease
            Worker(arm,domain).run()
    except BaseException as e:
        (folder/'failure.txt').write_text(traceback.format_exc())
        old=read(folder/'status.json')
        write(folder/'status.json',dict(state='FAILED',error=repr(e),previous=old,updated_at=stamp()))
        raise
    finally:
        if lease is not None and lease.poll() is None:
            lease.terminate()
            try:lease.wait(timeout=15)
            except subprocess.TimeoutExpired:lease.kill();lease.wait()


def summarize():
    result={};lines=['# 双专家组合选择','','状态：COMPLETED。冻结生成器、同一512用户、每域一个上游seed；仅探索性结果。','',
        '| 域 | GRAM N10 | 双专家 N10 | 仅SASRec N10 | 仅文本 N10 |','|---|---:|---:|---:|---:|']
    for d in ('Toys','Beauty'):
        old=np.load(DUAL/d/'ranks.npz');ranks={'GRAM':old['GRAM'],'dual':old['guided']}
        for arm in ARMS:
            assert read(OUT/arm/d/'status.json')['state']=='COMPLETED'
            a=np.load(OUT/arm/d/'ranks.npz');np.testing.assert_array_equal(a['users'],old['users']);np.testing.assert_array_equal(a['GRAM'],old['GRAM'])
            ranks[arm]=a['guided']
        ms={k:metrics(v) for k,v in ranks.items()}
        cs={f'dual_vs_{b}':contrast(ranks['dual'],ranks[b],repeats=2000,seed=20261014) for b in ('GRAM',*ARMS)}
        result[d]=dict(users=512,metrics=ms,comparisons=cs,
            above_GRAM=ms['dual']['N10']>ms['GRAM']['N10'],
            retained_experts_directionally_useful=all(ms['dual']['N10']>ms[a]['N10'] for a in ARMS))
        lines.append(f'| {d} | {ms["GRAM"]["N10"]:.6f} | {ms["dual"]["N10"]:.6f} | {ms["sas_only"]["N10"]:.6f} | {ms["text_only"]["N10"]:.6f} |')
    result['scope']='exploratory; single upstream seed; no candidate-weight experiment scheduled'
    lines+=['','双专家优于仅SASRec，才支持文本的条件贡献；优于仅文本，才支持SASRec的条件贡献。逐域差值、配对描述区间、H10/H50与新增/丢失用户见 summary.json。方向为正也不等于已证明稳定性；不把跨0区间解释为显著。','',
            '[结果](../../artifacts/phase21/expert_selection_20261014_v1/summary.json)；[协议](../../plan/第二十一阶段/21.21_先确定双专家组合_20261014.md)。']
    write(OUT/'summary.json',result);REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    cfg=read(OUT/'config.json');start=time.monotonic();pending=[(a,d) for a in ARMS for d in ('Toys','Beauty')];active={};finished={}
    write(OUT/'status.json',dict(state='VERIFYING_INPUTS',pid=os.getpid(),updated_at=stamp()))
    for i,(rel,digest) in enumerate(read(OUT/'manifest.json')['sha256'].items(),1):
        assert sha(ROOT/rel)==digest,rel
        if i%20==0:write(OUT/'status.json',dict(state='VERIFYING_INPUTS',verified_files=i,pid=os.getpid(),updated_at=stamp()))
    while pending or active:
        for key,job in list(active.items()):
            code=job['process'].poll()
            if code is None and time.monotonic()-job['start']>21900:
                job['process'].terminate()
                try:code=job['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:job['process'].kill();code=job['process'].wait()
                st=read(OUT/job['arm']/job['domain']/'status.json')
                write(OUT/job['arm']/job['domain']/'status.json',dict(state='RESOURCE_TIME_LIMIT',previous=st,exit_code=code,updated_at=stamp()))
            if code is not None:
                job['log'].close();finished[key]=dict(exit_code=code,state=read(OUT/job['arm']/job['domain']/'status.json').get('state'));del active[key]
        if time.monotonic()-start>cfg['supervisor_timeout_seconds'] and pending:
            for a,d in pending:
                write(OUT/a/d/'status.json',dict(state='RESOURCE_WAIT_LIMIT',updated_at=stamp()))
                finished[a+'/'+d]=dict(state='RESOURCE_WAIT_LIMIT')
            pending=[]
        if pending and len(active)<cfg['max_parallel']:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            free={line.split(',')[0].strip():int(line.split(',')[1]) for line in output.splitlines()}
            occupied={job['gpu'] for job in active.values()}
            for gpu in GPUS:
                if pending and len(active)<cfg['max_parallel'] and gpu not in occupied and free[gpu]>=cfg['minimum_free_mib']:
                    arm,domain=pending.pop(0);folder=OUT/arm/domain
                    log=(folder/'worker.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
                    p=subprocess.Popen([sys.executable,'-u','-m','experiment.phase21.expert_selection','worker',arm,domain],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
                    active[arm+'/'+domain]=dict(process=p,start=time.monotonic(),arm=arm,domain=domain,gpu=gpu,log=log)
                    write(folder/'status.json',dict(state='STARTING',pid=p.pid,gpu=gpu,updated_at=stamp()))
        snapshot={key:dict(pid=j['process'].pid,gpu=j['gpu'],worker=read(OUT/j['arm']/j['domain']/'status.json')) for key,j in active.items()}
        write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES',scope='expert_selection_only',pid=os.getpid(),updated_at=stamp(),elapsed_seconds=time.monotonic()-start,active=snapshot,pending=[a+'/'+d for a,d in pending],finished=finished,test_read=False))
        if pending or active:time.sleep(10)
    if all(v.get('state')=='COMPLETED' and v.get('exit_code')==0 for v in finished.values()):
        summarize();state='COMPLETED'
    else:state='INCOMPLETE'
    write(OUT/'status.json',dict(state=state,scope='expert_selection_only',updated_at=stamp(),finished=finished,test_read=False,candidate_weights_scheduled=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','supervise','worker','test']);p.add_argument('arm',nargs='?');p.add_argument('domain',nargs='?');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='worker':worker(a.arm,a.domain)
    elif a.mode=='test':print(tests())
    else:
        try:supervise()
        except BaseException as e:
            write(OUT/'status.json',dict(state='SUPERVISOR_FAILED',error=repr(e),traceback=traceback.format_exc(),updated_at=stamp()))
            raise
