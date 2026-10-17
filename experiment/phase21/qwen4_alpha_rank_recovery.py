"""Diagnose teacher arithmetic and explicitly resume only failed Beauty.

The original scientific files, losses, tolerances and completed Toys rows stay
unchanged. A failing batched gold path is recomputed with the original single
path implementation; recovery is accepted only if the original guard passes.
"""
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
from experiment.phase21 import qwen4_alpha_rank as R
from experiment.phase18.core.diff_data import DiffDataset

ROOT, OUT = R.ROOT, R.OUT
REC = OUT/'numerical_recovery_v1'
MODULE = 'experiment.phase21.qwen4_alpha_rank_recovery'
PLAN = ROOT/'plan/第二十一阶段/21.46_Qwen4强度对照Beauty数值核验修复_20261017.md'
read, write, sha, stamp = R.read, R.write, R.sha, R.stamp
ATOL, RTOL = 4e-5, 2e-5


def close(a,b):
    return bool(np.allclose(a,b,atol=ATOL,rtol=RTOL))


class DiagnosticTeacher(R.Teacher):
    def status(self,state='RUNNING',**extra):
        write(REC/'probe_status.json',dict(state=state,stage=self.stage,
            updated_at=stamp(),pid=os.getpid(),**extra))


def probe():
    R.CLEAN.OUT=OUT;w=DiagnosticTeacher('Beauty')
    actual=read(w.folder/'fit_samples.json')
    w.samples=[dict(s,target=1) for s in actual];w.data=DiffDataset(w.samples,w.catalog)
    pool=np.load(w.folder/'pool.npz');old=np.load(w.folder/'teacher.npz')
    assert old['users'].tolist()==pool['users'].tolist()==[s['user_id'] for s in actual]
    assert w.initial_digest==read(w.folder/'teacher_checks.json')['model_digest']
    selected=sorted(set([0,len(actual)//2,len(actual)-1,*range(864,928)]))
    checks=[];failed=[];negative_errors=[]
    for j in selected:
        ctx=w.context(j)
        paths=[w.catalog['decoder_paths'][item-1] for item in pool['ids'][j,pool['valid'][j]]]
        nt,mt=w.values(ctx,paths)
        single_n,single_m=R.OLD.Teacher.path_values(w,ctx,paths[0]);length=len(single_n)
        np.testing.assert_allclose(single_n,old['native'][j,:length],atol=ATOL,rtol=RTOL)
        np.testing.assert_allclose(single_m,old['legal'][j,:length],atol=ATOL,rtol=RTOL)
        mismatch=not (close(nt[0,:length],old['native'][j,:length]) and close(mt[0,:length],old['legal'][j,:length]))
        if mismatch:failed.append(j)
        checks.append(dict(row=j,user_id=actual[j]['user_id'],batched_original_guard_passed=not mismatch,
            batch_vs_saved_native_max_abs=float(np.max(np.abs(nt[0,:length]-old['native'][j,:length]))),
            single_vs_saved_native_max_abs=float(np.max(np.abs(single_n-old['native'][j,:length])))))
        if mismatch or j in (0,len(actual)//2,len(actual)-1):
            for col,path in enumerate(paths[1:],1):
                a,b=R.OLD.Teacher.path_values(w,ctx,path)
                negative_errors.append(max(float(np.max(np.abs(a-nt[col,:len(a)]))),float(np.max(np.abs(b-mt[col,:len(b)])))))
        write(REC/'probe_progress.json',dict(completed=len(checks),total=len(selected),failed_batch_rows=failed))
    assert failed,'The reported arithmetic failure was not reproduced; do not assume its cause'
    assert w.digest()==w.initial_digest
    result=dict(state='PASSED',updated_at=stamp(),checks=checks,batched_failure_reproduced_rows=failed,
        single_path_original_guard_passed_for_all=True,gram_weights_unchanged=True,
        compared_negative_paths=len(negative_errors),negative_batch_vs_single_max_abs=max(negative_errors),
        original_guard=dict(atol=ATOL,rtol=RTOL),scientific_configuration_unchanged=True)
    write(REC/'diagnosis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


def prepare():
    assert read(OUT/'status.json')['state']=='INCOMPLETE'
    assert read(OUT/'Toys/status.json')['state']=='COMPLETED'
    assert read(OUT/'Beauty/status.json')['state']=='FAILED'
    assert read(REC/'diagnosis.json')['state']=='PASSED'
    for d in R.DOMAINS:assert read(OUT/d/'lease_status.json')['state']=='released'
    # No incomplete scientific outputs are silently overwritten.
    assert not any((OUT/'Beauty'/n).exists() for n in ('path_teacher.npz','training.json','predictions.jsonl'))
    archive=REC/'before';archive.mkdir(exist_ok=False)
    for src in [OUT/'status.json',OUT/'Beauty/status.json',OUT/'Beauty/failure.json',
                OUT/'Beauty/worker_attempt1.log',OUT/'Beauty/lease.log',OUT/'Beauty/lease_status.json',
                OUT/'Beauty/lease_process.json',OUT/'supervisor.log']:
        dest=archive/src.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    preserved={}
    for name in ('predictions.jsonl','training.json','training_manifest.json','runtime_checks.json','status.json'):
        path=OUT/'Toys'/name;preserved[str(path.relative_to(ROOT))]=sha(path)
    for rel,digest in read(OUT/'manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest,rel
    write(REC/'preserved_toys.json',preserved)
    manifest={}
    for src in (Path(__file__),PLAN,REC/'diagnosis.json',REC/'preserved_toys.json',OUT/'manifest.json',OUT/'config.json'):
        manifest[str(src.relative_to(ROOT))]=sha(src)
        dest=REC/'sources'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    write(REC/'manifest.json',dict(sha256=manifest))
    write(REC/'status.json',dict(state='PREPARED',updated_at=stamp(),resume_domain='Beauty',
        toys_reused=True,original_tolerances_unchanged=True,automatic_retry=False,test_read=False))
    print('PREPARED: only Beauty will resume; original failure records archived',flush=True)


class VerifiedTeacher(R.Teacher):
    def __init__(self,domain):
        super().__init__(domain)
        self.saved_teacher=np.load(self.folder/'teacher.npz')
        self.fallbacks=[];self.maximum_batched_gold_error=0.

    def context(self,j,override_index=None):
        self.current_row=j
        return super().context(j,override_index=j)

    def values(self,ctx,paths):
        nt,mt=super().values(ctx,paths)
        j=self.current_row;length=len(paths[0])-1
        old_n=self.saved_teacher['native'][j,:length];old_m=self.saved_teacher['legal'][j,:length]
        assert self.samples[j]['user_id']==self.saved_teacher['users'][j]
        assert paths[0]==self.catalog['decoder_paths'][int(self.saved_teacher['targets'][j])]
        difference=max(float(np.max(np.abs(nt[0,:length]-old_n))),float(np.max(np.abs(mt[0,:length]-old_m))))
        self.maximum_batched_gold_error=max(self.maximum_batched_gold_error,difference)
        if not (close(nt[0,:length],old_n) and close(mt[0,:length],old_m)):
            single_n,single_m=R.OLD.Teacher.path_values(self,ctx,paths[0])
            # Same original strict guard: no unchecked acceptance or loosening.
            np.testing.assert_allclose(single_n,old_n,atol=ATOL,rtol=RTOL)
            np.testing.assert_allclose(single_m,old_m,atol=ATOL,rtol=RTOL)
            nt[0,:length]=single_n;mt[0,:length]=single_m
            self.fallbacks.append(dict(row=j,user_id=self.samples[j]['user_id'],
                batch_max_absolute_difference=difference,
                single_max_absolute_difference=max(float(np.max(np.abs(single_n-old_n))),float(np.max(np.abs(single_m-old_m))))))
        if self.fallbacks or j%32==0:
            write(REC/'runtime_numeric_checks.json',dict(state='RUNNING',users_checked=j+1,
                maximum_batched_gold_error=self.maximum_batched_gold_error,fallbacks=self.fallbacks,
                original_guard=dict(atol=ATOL,rtol=RTOL),all_fallbacks_pass_original_guard=True))
        return nt,mt


def bind():
    R.MODULE=MODULE
    R.Teacher=VerifiedTeacher


def worker():
    p=OUT/'Beauty';lease=child=None
    def interrupted(signum,frame):raise InterruptedError('Recovery worker interrupted or timed out')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(R.CFG['worker_timeout_seconds'])
    try:
        R.progress('Beauty','verify_original_and_recovery_inputs')
        assert read(OUT/'config.json')==R.CFG
        original=read(OUT/'manifest.json')
        for rel in original['domain_inputs']['Beauty']:assert sha(ROOT/rel)==original['sha256'][rel],rel
        for rel,digest in read(REC/'manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest,rel
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        free=next(g['free_mib'] for g in R.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
        assert free>=R.CFG['minimum_free_mib'],'GPU became unavailable; no automatic retry'
        with (REC/'lease.log').open('x') as log:
            leasepath=p/'lease_status.json'
            # Archive the old released status before a new sidecar can be read.
            leasepath.unlink()
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib','16384','--expected-workload-peak-mib','12288','--status-path',str(leasepath)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/'lease_process.json',dict(pid=lease.pid,gpu_uuid=gpu,started_at=stamp(),
                recovery=str(REC.relative_to(ROOT))))
            started=time.monotonic()
            while not leasepath.exists() or read(leasepath).get('state')!='holding':
                assert lease.poll() is None,'Lease exited';assert time.monotonic()-started<180,'Lease timeout';time.sleep(1)
            for phase in ('cache','fit','generate'):
                assert lease.poll() is None,'Lease exited'
                child=subprocess.Popen([sys.executable,'-u','-m',MODULE,phase,'--domain','Beauty',
                    '--lease-pid',str(lease.pid)],cwd=ROOT,start_new_session=True)
                assert child.wait()==0,('Recovery phase failed; no automatic retry',phase)
        return 0
    finally:
        signal.alarm(0);R.B.stop(child);R.B.stop(lease)


def supervise():
    assert read(REC/'status.json')['state']=='PREPARED'
    for rel,digest in read(REC/'manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest,rel
    began=time.monotonic();process=None
    try:
        while process is None:
            inventory=R.B.S.OLD.gpu_inventory()
            available=sorted([g for g in inventory if g['free_mib']>=R.CFG['minimum_free_mib']],
                key=lambda g:(g['utilization'],-g['free_mib'],g['index']))
            if available:
                gpu=available[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                log=(REC/'worker.log').open('x')
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain','Beauty'],cwd=ROOT,
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            else:
                status=dict(state='WAITING_FOR_RESOURCES',updated_at=stamp(),test_read=False,
                    recovery=str(REC.relative_to(ROOT)),finished={'Toys':dict(state='COMPLETED',reused=True)})
                write(OUT/'status.json',status);write(REC/'status.json',status)
                assert time.monotonic()-began<R.CFG['resource_wait_seconds'],'Resource wait timeout'
                time.sleep(10)
        while process.poll() is None:
            if time.monotonic()-began>R.CFG['worker_timeout_seconds']+120:raise TimeoutError('Recovery hard timeout')
            status=dict(state='RUNNING',stage='explicit_numerical_recovery',updated_at=stamp(),test_read=False,
                recovery=str(REC.relative_to(ROOT)),elapsed_seconds=time.monotonic()-began,
                finished={'Toys':dict(state='COMPLETED',reused=True)},
                active={'Beauty':dict(pid=process.pid,gpu_index=gpu['index'],gpu_uuid=gpu['uuid'],worker=read(OUT/'Beauty/status.json'))})
            write(OUT/'status.json',status);write(REC/'status.json',status);time.sleep(10)
        log.close()
        assert process.returncode==0,('Recovery failed; no automatic retry',process.returncode)
        assert read(OUT/'Beauty/status.json')['state']=='COMPLETED'
        assert read(OUT/'Beauty/lease_status.json')['state']=='released'
        for rel,digest in read(REC/'preserved_toys.json').items():assert sha(ROOT/rel)==digest,rel
        R.summarize()
        result=dict(state='COMPLETED',updated_at=stamp(),test_read=False,
            recovery=str(REC.relative_to(ROOT)),finished={d:dict(state='COMPLETED',reused=d=='Toys',lease_released=True) for d in R.DOMAINS},
            scientific_configuration_unchanged=True,report=str(R.REPORT.relative_to(ROOT)),elapsed_seconds=time.monotonic()-began)
        write(OUT/'status.json',result);write(REC/'status.json',result)
        numeric=read(REC/'runtime_numeric_checks.json');numeric['state']='PASSED';write(REC/'runtime_numeric_checks.json',numeric)
    finally:R.B.stop(process)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('probe','prepare','supervise','worker','cache','fit','generate'))
    parser.add_argument('--domain',choices=('Beauty',));parser.add_argument('--lease-pid',type=int);args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Numerical recovery interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='probe':probe()
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        else:
            bind()
            if args.mode=='worker':return worker()
            elif args.mode=='cache':R.cache('Beauty')
            elif args.mode=='fit':R.fit('Beauty')
            else:R.Generation('Beauty',args.lease_pid).run()
        return 0
    except BaseException as exc:
        details=dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
        write(REC/(args.mode+'_failure.json'),details)
        if args.mode in ('worker','cache','fit','generate'):write(OUT/'Beauty/status.json',details)
        if args.mode=='supervise':write(OUT/'status.json',details)
        write(REC/'status.json',details)
        raise


if __name__=='__main__':sys.exit(main())
