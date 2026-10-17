"""Resource-only Toys recovery; retain the completed Beauty control unchanged."""
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
from experiment.phase21 import global_kl_control as G

ROOT=G.ROOT
ORIGINAL=G.OUT
OUT=ROOT/'artifacts/phase21/global_kl_control_20261017_v1_recovery1'
PLAN=ROOT/'plan/第二十一阶段/21.52_全局KL对照OOM后的Toys单域恢复_20261017.md'
REPORT=ROOT/'report/第二十一阶段/全局alpha同KL损失匹配对照_恢复_20261017.md'
MODULE='experiment.phase21.global_kl_control_recovery'
CFG=dict(G.CFG,workload_preallocation_mib=10240,stable_snapshots=3,poll_seconds=5,
    excluded_gpu_uuids=['GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2'],
    max_parallel=1,reused_domain='Beauty',regenerated_domain='Toys')
read,write,sha,stamp=G.read,G.write,G.sha,G.stamp


def bind():
    G.OUT,G.CFG,G.REPORT=OUT,CFG,REPORT


def progress(state,stage,**extra):
    write(OUT/'Toys/status.json',dict(state=state,stage=stage,pid=os.getpid(),
        updated_at=stamp(),test_read=False,**extra))


def prepare():
    assert read(ORIGINAL/'Beauty/status.json')['state']=='COMPLETED'
    assert read(ORIGINAL/'Beauty/runtime_checks.json')['state']=='PASSED'
    assert not (ORIGINAL/'Toys/predictions.jsonl').exists()
    OUT.mkdir(exist_ok=False);write(OUT/'config.json',CFG);shutil.copy2(PLAN,OUT/'protocol.md')
    manifest={};domain_inputs={};common=[]
    original_manifest=read(ORIGINAL/'manifest.json')['sha256']
    def use(path,expected=None):
        rel=str(path.relative_to(ROOT));digest=manifest.get(rel) or sha(path)
        assert expected is None or digest==expected,rel
        manifest[rel]=digest;return rel
    for rel,digest in original_manifest.items():
        if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/')):
            common.append(use(ROOT/rel,digest))
    for f in (Path(__file__),ROOT/'experiment/phase21/run_global_kl_control_recovery.sh',PLAN,OUT/'config.json'):
        common.append(use(f))
    for d in G.DOMAINS:
        p=OUT/d;p.mkdir();files=[]
        train_manifest=read(ORIGINAL/d/'training_manifest.json')['sha256']
        for name in ('catalog.json','data_config.json','fit_samples.json','samples.json','teacher.npz',
            'teacher_checks.json','pool.npz','path_teacher.npz','path_teacher_checks.json','expert_path_logs.npy',
            'signals.npy','source_alphas.npz','global_nll.pt','user_kl1.pt','source_training.json',
            'training_signal_audit.json','references.json','global_kl1.pt','alpha.npy','training.json','replay_checks.json'):
            src=ORIGINAL/d/name;rel=str(src.relative_to(ROOT))
            files.append(use(src,train_manifest.get(rel,original_manifest.get(rel))))
            shutil.copy2(src,p/name);files.append(use(p/name))
        model_files=[p/n for n in ('global_kl1.pt','alpha.npy','training.json','replay_checks.json')]
        write(p/'training_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in model_files}))
        files.append(use(p/'training_manifest.json'))
        cfg=read(p/'data_config.json');files.append(use(ROOT/cfg['parent_checkpoint'],cfg['parent_sha256']))
        if d=='Beauty':
            for name in ('predictions.jsonl','smoke.json','runtime_checks.json','status.json'):
                src=ORIGINAL/d/name;files.append(use(src));shutil.copy2(src,p/name);files.append(use(p/name))
            samples=read(p/'samples.json');refs=read(p/'references.json')
            rows=[json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()]
            assert len(rows)==len(samples)==512
            for row,s,ref in zip(rows,samples,refs):
                assert all(row[k]==s[k] for k in ('user_id','target','history'))
                assert all(row[k]==ref[k] for k in G.REFERENCES)
                G.independent_rank(row[G.ARM],s['target'])
            write(p/'reused_result.json',dict(source=str((ORIGINAL/d).relative_to(ROOT)),
                prediction_sha256=sha(p/'predictions.jsonl'),users=512,rerun=False))
        else:progress('PREPARED','waiting_for_stable_gpu',users_completed=0,users_total=512)
        domain_inputs[d]=common+files
    for rel in manifest:
        f=ROOT/rel
        if f.suffix in ('.py','.sh','.md'):
            dest=OUT/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=domain_inputs))
    write(OUT/'diagnosis.json',dict(known_failure='Toys CUDA OOM at model.to(cuda), before first prediction',
        original_gpu_uuid=CFG['excluded_gpu_uuids'][0],scientific_inputs_unchanged=True,
        beauty_users_retained=512,toys_prior_predictions=0,
        limitation='Load-time free memory was not logged; exact competing allocation is unverified.',
        fix='Stable admission plus same-process caching-allocator reservation before model load.'))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),reused_beauty=True,test_read=False))
    write(ORIGINAL/'recovery_pointer.json',dict(status_path=str(OUT/'status.json'),
        original_failure_retained=True,updated_at=stamp()))
    REPORT.write_text('# 全局α同KL对照：Toys资源恢复\n\nBeauty512完整复用；仅补Toys。尚无完整两域新结论。\n')
    print(json.dumps(dict(state='PREPARED',beauty_reused=512,hashes=len(manifest))),flush=True)


def memory_snapshot():
    free,total=torch.cuda.mem_get_info()
    return dict(free_mib=free/2**20,total_mib=total/2**20,
        allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20)


def generate(lease_pid,attempt):
    # No CPU tensor operations before CLEAN.Worker sets its interop thread count.
    progress('RUNNING','reserve_working_memory',attempt=attempt)
    before=None
    try:
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
        before=memory_snapshot()
        reserve=torch.empty(CFG['workload_preallocation_mib']*2**20,device='cuda:0',dtype=torch.uint8)
        torch.cuda.synchronize();held=memory_snapshot();del reserve
        cached=memory_snapshot();assert cached['reserved_mib']>=CFG['workload_preallocation_mib']
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():raise
        write(OUT/'Toys'/f'reservation_attempt{attempt}.json',dict(state='ADMISSION_DEFERRED',
            reason=repr(exc),before=before,model_loaded=False,predictions_written=False,updated_at=stamp()))
        progress('ADMISSION_DEFERRED','working_memory_reservation_failed',attempt=attempt)
        return 75
    write(OUT/'Toys'/f'reservation_attempt{attempt}.json',dict(state='RESERVED',before=before,
        during_reservation=held,cached_for_model=cached,updated_at=stamp()))
    torch.cuda.reset_peak_memory_stats()
    class ReservedGeneration(G.Generation):
        def status(self,state='RUNNING',**extra):
            assert torch.cuda.memory_reserved()/2**20>=CFG['workload_preallocation_mib'],'Workload cache was released'
            super().status(state,**extra)
    ReservedGeneration('Toys',lease_pid).run()
    write(OUT/'Toys/reservation_runtime.json',dict(state='PASSED',memory=memory_snapshot(),
        scientific_rule_changed=False,total_lease_mib=16384,sidecar_mib=4096,
        workload_cached_mib=CFG['workload_preallocation_mib']))
    return 0


def worker(attempt):
    lease=child=None
    def interrupted(signum,frame):raise InterruptedError('Recovery interrupted or timed out')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    p=OUT/'Toys'
    try:
        manifest=read(OUT/'manifest.json');assert read(OUT/'config.json')==CFG
        for rel in manifest['domain_inputs']['Toys']:assert sha(ROOT/rel)==manifest['sha256'][rel],rel
        G.verify_training('Toys')
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        free=next(g['free_mib'] for g in G.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
        write(p/f'admission_attempt{attempt}.json',dict(gpu_uuid=gpu,free_before_lease_mib=free,updated_at=stamp()))
        if free<CFG['minimum_free_mib']:
            progress('ADMISSION_DEFERRED','free_memory_changed_before_lease',free_mib=free);return 75
        with (p/f'lease_attempt{attempt}.log').open('x') as log:
            lease_path=p/f'lease_status_attempt{attempt}.json'
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib','16384','--expected-workload-peak-mib','12288','--status-path',str(lease_path)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/f'lease_process_attempt{attempt}.json',dict(pid=lease.pid,gpu_uuid=gpu,updated_at=stamp()))
            began=time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state')!='holding':
                assert lease.poll() is None,'Lease exited';assert time.monotonic()-began<180,'Lease timeout';time.sleep(1)
            free_after=next(g['free_mib'] for g in G.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
            write(p/f'post_lease_attempt{attempt}.json',dict(gpu_uuid=gpu,free_mib=free_after,updated_at=stamp()))
            if free_after<CFG['workload_preallocation_mib']+2048:
                progress('ADMISSION_DEFERRED','free_memory_changed_after_lease',free_mib=free_after);return 75
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'generate','--lease-pid',str(lease.pid),
                '--attempt',str(attempt)],cwd=ROOT,start_new_session=True)
            code=child.wait()
            if code==75 and read(p/'status.json')['state']=='ADMISSION_DEFERRED':return 75
            assert code==0,('Generation failed; see failure_generate.json',code)
        return 0
    finally:
        signal.alarm(0);G.B.stop(child);G.B.stop(lease)


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    began=time.monotonic();stable={};attempt=0;active=None;done=False;finished={
        'Beauty':dict(exit_code=0,state='COMPLETED',lease_released=True,reused_from=str(ORIGINAL/'Beauty'))}
    try:
        while not done:
            if active and active['process'].poll() is not None:
                code=active['process'].returncode;active['log'].close();status=read(OUT/'Toys/status.json')
                released=read(OUT/'Toys'/f'lease_status_attempt{attempt}.json').get('state')=='released' if (OUT/'Toys'/f'lease_status_attempt{attempt}.json').exists() else True
                write(OUT/'Toys'/f'attempt{attempt}_result.json',dict(exit_code=code,status=status,lease_released=released))
                if code==75 and status['state']=='ADMISSION_DEFERRED':stable={}
                else:
                    finished['Toys']=dict(exit_code=code,state=status['state'],lease_released=released);done=True
                active=None
            inventory=G.B.S.OLD.gpu_inventory()
            if not active and not done:
                eligible=[]
                for gpu in inventory:
                    ok=gpu['uuid'] not in CFG['excluded_gpu_uuids'] and gpu['free_mib']>=CFG['minimum_free_mib']
                    stable[gpu['uuid']]=stable.get(gpu['uuid'],0)+1 if ok else 0
                    if stable[gpu['uuid']]>=CFG['stable_snapshots']:eligible.append(gpu)
                if eligible:
                    gpu=sorted(eligible,key=lambda g:(-g['free_mib'],g['utilization'],g['index']))[0];attempt+=1
                    log=(OUT/'Toys'/f'worker_attempt{attempt}.log').open('x')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    proc=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--attempt',str(attempt)],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active=dict(process=proc,log=log,gpu=gpu)
                elif time.monotonic()-began>CFG['resource_wait_seconds']:
                    finished['Toys']=dict(exit_code=124,state='RESOURCE_WAIT_LIMIT',lease_released=True);done=True
            write(OUT/'status.json',dict(state='RUNNING' if active else 'FINALIZING' if done else 'WAITING_FOR_RESOURCES',
                pid=os.getpid(),updated_at=stamp(),elapsed_seconds=time.monotonic()-began,
                pending=['Toys'] if not active and not done else [],finished=finished,reused_beauty=True,
                active={'Toys':dict(pid=active['process'].pid,gpu_index=active['gpu']['index'],
                    gpu_uuid=active['gpu']['uuid'],worker=read(OUT/'Toys/status.json'))} if active else {},
                gpu_inventory=inventory,test_read=False))
            if not done:time.sleep(CFG['poll_seconds'])
        complete=all(v['exit_code']==0 and v['state']=='COMPLETED' and v['lease_released'] for v in finished.values())
        if complete:
            G.summarize()
            with REPORT.open('a') as stream:stream.write('\n资源恢复记录：Beauty512预测原样复用，Toys在独立目录补齐；α和科学输入未变。\n')
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
            updated_at=stamp(),elapsed_seconds=time.monotonic()-began,report=str(REPORT),test_read=False))
    finally:
        if active:G.B.stop(active['process']);active['log'].close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','supervise','worker','generate'))
    parser.add_argument('--lease-pid',type=int);parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    bind()
    def interrupted(signum,frame):raise InterruptedError('Recovery interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='worker':return worker(args.attempt)
        else:return generate(args.lease_pid,args.attempt)
        return 0
    except BaseException as exc:
        if OUT.exists():
            p=OUT/'Toys' if args.mode in ('worker','generate') else OUT
            write(p/('failure_'+args.mode+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            if args.mode=='worker' and (p/'failure_generate.json').exists():
                error=read(p/'failure_generate.json')['error']
            else:error=repr(exc)
            write(p/'status.json',dict(state='FAILED',stage=args.mode,error=error,updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':sys.exit(main())
