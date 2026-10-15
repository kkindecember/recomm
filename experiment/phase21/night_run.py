"""Six-slot handover and explicit post-reboot continuation; scientific recipes unchanged."""
import argparse
import fcntl
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
from experiment.phase21 import text_contribution as runner
from experiment.phase21 import expert_resource_recovery as recovery
from experiment.phase21 import text_adaptation as text_run
from experiment.phase21.night_resume_core import durable_file, prediction_prefix, process_identity, identity_alive

ROOT = runner.ROOT
OUT = ROOT/'artifacts/phase21/night_run_20260923_v1'
PLAN = ROOT/'plan/第二十一阶段/21.4C_重启前并行与断点恢复_20260923.md'
MODULE = 'experiment.phase21.night_run'
read, sha, stamp = runner.read, runner.sha, runner.stamp


def write(path, obj):
    runner.write(path, obj); durable_file(path)


def key(job):
    return '/'.join(str(job[x]) for x in ('pipeline','kind','group','arm','domain') if x in job)


def folder(job):
    if job['pipeline']=='recovery': return recovery.OUT/job['group']/job['arm']/job['domain']
    return text_run.job_folder(job)


def snapshot(job):
    return read(folder(job)/'status.json')


def complete(job):
    p=folder(job)
    if snapshot(job)['state']!='COMPLETED' or not (p/'summary.json').exists(): return False
    if job.get('kind','decode')=='decode':
        assert read(p/'summary.json')['model_unchanged'] and (p/'ranks.npz').exists()
    return True


class ContinueGeneration:
    def fit_signals(self):
        import torch
        names=('gate.pt','fused.npy','gate_prediction.npz','fit_checks.json')
        if not all((self.folder/n).exists() for n in names):
            assert not (self.folder/'predictions.jsonl').exists(), 'Saved predictions require complete original fusion inputs'
            partial=[self.folder/n for n in names+('signal_audit.json','mechanism_scores.npy') if (self.folder/n).exists()]
            if partial:
                dest=self.folder/'attempts'/f'incomplete_fit_{int(time.time())}_{os.getpid()}'; dest.mkdir(parents=True)
                for p in partial: p.rename(dest/p.name)
            return super().fit_signals()
        assert read(self.folder/'fit_checks.json')['state']=='PASSED'
        saved=torch.load(self.folder/'gate.pt',map_location='cpu')
        assert saved['arm']==self.arm and not saved['development_labels_used']
        expected=sorted(s['user_id'] for s in self.samples if s['role']=='gate_calibration')
        assert saved['fit_users']==expected
        self.fit_steps=saved['optimizer_steps']; assert self.fit_steps==dict(Toys=12,Beauty=16)[self.domain]
        self.fused=np.load(self.folder/'fused.npy',mmap_mode='r')
        assert self.fused.shape==(len(self.samples),self.raw.shape[-1])
        for begin in range(0,len(self.fused),64): assert np.isfinite(self.fused[begin:begin+64]).all()
        self.stage='reuse_completed_fusion'; self.status()

    def run(self):
        self.fit_signals()
        frozen={n:sha(self.folder/n) for n in ('gate.pt','fused.npy','fit_checks.json')}
        saved_path=self.folder/'continuation_inputs.json'
        if saved_path.exists(): assert read(saved_path)['sha256']==frozen
        else: write(saved_path,dict(sha256=frozen,created_at=stamp()))
        expected=[(j,self.samples[j]) for j in self.development]
        records=prediction_prefix(self.folder/'predictions.jsonl',expected,repair=True)
        for record in records:
            ref=self.references[record['user_id']]
            assert record['native']['ids']==ref['ids']
            np.testing.assert_array_equal(record['native']['scores'],ref['native'])
        resumed=len(records); self.users_completed=resumed
        write(self.folder/'continuation.json',dict(resumed_users=resumed,total=512,at=stamp(),pid=os.getpid()))
        self.smoke_control(); self.stage='actual_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('a') as handle:
            for j,s in expected[resumed:]:
                ctx=self.context(j); q=self.index.conditional_logs(self.fused[j]); native=self.decode(ctx)
                ref=self.references[s['user_id']]
                assert native['paths']==ref['paths']; np.testing.assert_array_equal(native['scores'],ref['native'])
                late=native['guide'].rescore(native['paths'],q,.3); guided=self.decode(ctx,q,.3)
                record=dict(user_id=s['user_id'],target=s['target'],history=s['history'],sample_index=j,
                    native=dict(ids=native['ids'],scores=native['scores'].tolist()),
                    late=dict(ids=native['ids'],scores=late.tolist()),
                    guided=dict(ids=guided['ids'],scores=guided['scores'].tolist()))
                handle.write(json.dumps(record)+'\n'); handle.flush(); os.fsync(handle.fileno())
                self.users_completed+=1
                if self.users_completed%8==0 or self.users_completed==resumed+1: self.status()
                del ctx,native,guided
        assert self.digest()==self.initial_digest
        report=runner.verify_domain(self.arm,self.domain)
        report.update(model_unchanged=True,fusion_optimizer_steps=self.fit_steps,
            resumed_users=resumed,new_users_this_attempt=512-resumed,total_decodes_this_attempt=self.completed,
            seconds_this_attempt=time.monotonic()-self.started)
        write(self.folder/'summary.json',report)
        for name in ('ranks.npz','gate.pt','fused.npy'): durable_file(self.folder/name)
        self.stage='complete'; self.status(state='COMPLETED')


def archive_incomplete_cache(job):
    """A partial score matrix has no trusted row commit marker: preserve, recompute only this cache job."""
    p=folder(job); tag=f'attempt_{int(time.time())}_{os.getpid()}'
    for root in [p]+[text_run.OUT/a/job['domain'] for a in (('D',) if job['kind']=='cache_D' else ('B','C'))]:
        names=('semantic.npy','fused.npy','gate.pt','gate_prediction.npz','fit_checks.json','cache_manifest.json',
               'score_audit.json','history_dispersion.json','weight_examples.json','summary.json',
               'A_ranks.npz','B_ranks.npz','C_ranks.npz','D_ranks.npz','failure.txt')
        existing=[root/n for n in names if (root/n).exists()]
        if existing:
            dest=root/'attempts'/tag; dest.mkdir(parents=True)
            for src in existing: src.rename(dest/src.name)


def worker(job):
    for rel,digest in read(OUT/'manifest.json')['sha256'].items():
        if rel.startswith('experiment/'): assert sha(ROOT/rel)==digest,rel
    if job['pipeline']=='recovery':
        recovery.RecoveryMixin=type('ContinuableRecovery',(ContinueGeneration,recovery.RecoveryMixin),{})
        recovery.worker(job['group'],job['arm'],job['domain'])
    else:
        if job['kind']=='decode':
            text_run.GenerationWorker=type('ContinuableText',(ContinueGeneration,text_run.GenerationWorker),{})
        else: archive_incomplete_cache(job)
        text_run.worker(job)


def parse_worker(cmd):
    if '-m' not in cmd or 'worker' not in cmd: return None
    module=cmd[cmd.index('-m')+1]
    if module not in (recovery.MODULE,text_run.MODULE,MODULE): return None
    if module==MODULE: return json.loads(cmd[cmd.index('--job')+1])
    value=lambda flag:cmd[cmd.index(flag)+1]
    if module==recovery.MODULE: return dict(pipeline='recovery',group=value('--group'),arm=value('--arm'),domain=value('--domain'))
    job=dict(pipeline='text',kind=value('--kind'),domain=value('--domain'))
    if '--arm' in cmd: job['arm']=value('--arm')
    return job


def discover():
    active={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid!=os.getuid(): continue
            ident=process_identity(int(p.name))
            if not ident: continue
            cmd=p.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            job=parse_worker(cmd)
            if job is None: continue
            env=p.joinpath('environ').read_bytes().split(b'\0')
            gpu=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
            elapsed=float(Path('/proc/uptime').read_text().split()[0])-int(ident['start'])/os.sysconf('SC_CLK_TCK')
            k=key(job); assert k not in active, f'Duplicate running job: {k}'
            active[k]=dict(job=job,identity=ident,gpu=gpu,started_wall=time.time()-elapsed,adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError): continue
    return active


def takeover():
    stopped=[]
    for root,module in ((recovery.OUT,recovery.MODULE),(text_run.OUT,text_run.MODULE)):
        state=read(root/'status.json'); pid=state.get('pid')
        if not pid or not process_identity(pid): continue
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
        assert module in cmd and 'worker' not in cmd, (pid,cmd)
        # Stop only the dispatcher PID. Workers were started in separate sessions.
        os.kill(pid,signal.SIGSTOP)
        snapshot_path=OUT/f'handover_{root.name}_{int(time.time())}.json'
        write(snapshot_path,dict(status=read(root/'status.json'),launches=read(root/'launch.json') if (root/'launch.json').exists() else None))
        os.kill(pid,signal.SIGKILL); stopped.append(pid)
    write(OUT/'handover.json',dict(stopped_dispatchers_only=stopped,at=stamp(),workers=discover()))


def base_jobs():
    jobs=[dict(pipeline='recovery',**j) for j in read(recovery.OUT/'jobs.json')['jobs']]
    return [dict(pipeline='text',kind='cache_main',domain=d) for d in text_run.DOMAINS]+jobs


def extend_text(jobs):
    exists={key(j) for j in jobs}
    def add(new):
        for j in new:
            if key(j) not in exists: jobs.append(j); exists.add(key(j))
    mains=[dict(pipeline='text',kind='cache_main',domain=d) for d in text_run.DOMAINS]
    if not all(complete(j) for j in mains): return 'CACHE_SCREEN'
    screen={d:read(text_run.OUT/'cache_main'/d/'summary.json') for d in text_run.DOMAINS}
    verdict=text_run.core.screen_rule(screen)
    if not (text_run.OUT/'screen_summary.json').exists():
        write(text_run.OUT/'screen_summary.json',dict(domains=screen,verdict=verdict,test_read=False))
        text_run.write_screen_report(screen,verdict)
    if not verdict['pass_for_actual_generation']: return 'COMPLETED_SCREEN_ONLY'
    decodes=[dict(pipeline='text',kind='decode',arm=a,domain=d) for d in text_run.DOMAINS for a in ('B','C')]
    add(decodes)
    if not all(complete(j) for j in decodes): return 'ACTUAL_GENERATION_ABC'
    path=text_run.OUT/'generation_ABC.json'
    result=read(path) if path.exists() else text_run.generation_summary(('A','B','C'))
    if not result['continue_to_D']: return 'COMPLETED'
    caches=[dict(pipeline='text',kind='cache_D',domain=d) for d in text_run.DOMAINS]; add(caches)
    if not all(complete(j) for j in caches): return 'CONDITIONAL_D_CACHE'
    decodes=[dict(pipeline='text',kind='decode',arm='D',domain=d) for d in text_run.DOMAINS]; add(decodes)
    if not all(complete(j) for j in decodes): return 'CONDITIONAL_D_GENERATION'
    if not (text_run.OUT/'generation_with_D.json').exists(): text_run.generation_summary(('A','B','C','D'))
    return 'COMPLETED'


def prepare():
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',dict(max_parallel=6,minimum_free_mib=16384,gpu_pool=recovery.GPU_POOL,
        worker_timeout_seconds=14400,supervisor_timeout_seconds=172800,auto_start_after_reboot=False,
        source_roots=[str(recovery.OUT),str(text_run.OUT)],test_read=False))
    shutil.copy2(PLAN,OUT/'protocol.md')
    files=[Path(__file__),ROOT/'experiment/phase21/night_resume_core.py',ROOT/'experiment/phase21/test_night_resume.py',
           ROOT/'experiment/phase21/run_night.sh',PLAN,OUT/'config.json',recovery.OUT/'manifest.json',text_run.OUT/'manifest.json']
    write(OUT/'manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in files},at=stamp()))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp()))


def sync_progress(active):
    for r in active.values():
        p=folder(r['job'])/'predictions.jsonl'
        if p.exists(): durable_file(p)


def supervise(initial=False,retry_failed=False):
    lock=(OUT/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(OUT/'config.json')
    for rel,digest in read(OUT/'manifest.json')['sha256'].items(): assert sha(ROOT/rel)==digest,rel
    if initial: takeover()
    active=discover(); jobs=base_jobs(); children={}; blocked={}; started=time.monotonic(); last_sync=0
    for j in jobs:
        if snapshot(j)['state']=='FAILED' and not retry_failed and key(j) not in active: blocked[key(j)]=snapshot(j)
    attempt=f'{int(time.time())}_{os.getpid()}'
    launches=[]; summaries={}
    while True:
        now=time.time()
        if time.monotonic()-started>cfg['supervisor_timeout_seconds']:
            write(OUT/'status.json',dict(state='RESOURCE_TIME_LIMIT',reason='48h dispatcher limit',active=active,updated_at=stamp()))
            return
        for k,r in list(active.items()):
            if k in children: children[k].poll()
            if identity_alive(r['identity']) and now-r['started_wall']>cfg['worker_timeout_seconds']:
                os.killpg(r['identity']['pid'],signal.SIGKILL if r.get('terminating') else signal.SIGTERM)
                r['terminating']=True
                write(folder(r['job'])/'status.json',dict(state='RESOURCE_TIME_LIMIT',reason='Actual worker 4h limit',updated_at=stamp()))
            if not identity_alive(r['identity']):
                del active[k]
                if not complete(r['job']):
                    s=snapshot(r['job'])
                    if s['state'] not in ('FAILED','RESOURCE_TIME_LIMIT'):
                        s=dict(state='INTERRUPTED',previous=s,updated_at=stamp()); write(folder(r['job'])/'status.json',s)
                    blocked[k]=s
        phase=extend_text(jobs)
        if not retry_failed:
            for j in jobs:
                if key(j) not in active and snapshot(j)['state']=='FAILED': blocked[key(j)]=snapshot(j)
        candidates=[j for j in jobs if key(j) not in active and key(j) not in blocked and not complete(j)]
        candidates.sort(key=lambda j:(j['pipeline']!='text',j.get('group')!='removal',j['domain']!='Toys',key(j)))
        query_error=None; free={}
        if candidates and len(active)<cfg['max_parallel']:
            try:
                output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30)
                free={s.split(',')[0].strip():int(s.split(',')[1]) for s in output.splitlines()}
            except (subprocess.SubprocessError,ValueError) as error: query_error=repr(error)
        busy={r['gpu'] for r in active.values()}
        available=sorted([g for g in cfg['gpu_pool'] if g not in busy and free.get(g,0)>=cfg['minimum_free_mib']],key=lambda g:-free[g])
        for job in candidates:
            if not available or len(active)>=cfg['max_parallel']: break
            gpu=available.pop(0); k=key(job); p=folder(job)
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            write(p/'status.json',dict(state='STARTING',updated_at=stamp(),gpu=gpu,attempt=attempt))
            with (p/f'worker_{attempt}.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',json.dumps(job)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[k]=process; ident=process_identity(process.pid); assert ident
            active[k]=dict(job=job,identity=ident,gpu=gpu,started_wall=time.time(),adopted=False)
            launches.append(active[k]); write(OUT/f'launch_{attempt}.json',dict(launches=launches,supervisor_pid=os.getpid()))
        for group in recovery.GROUPS:
            relevant=[j for j in jobs if j.get('group')==group]
            if group not in summaries and all(complete(j) for j in relevant):
                group_status=recovery.OUT/group/'status.json'
                if group_status.exists() and read(group_status)['state']=='COMPLETED': summaries[group]='COMPLETED'
                else:
                    try:
                        recovery.bind(group).summarize(); summaries[group]='COMPLETED'
                        write(group_status,dict(state='COMPLETED',updated_at=stamp()))
                    except Exception:
                        summaries[group]='FAILED'; (OUT/f'{group}_summary_failure.txt').write_text(traceback.format_exc())
        if now-last_sync>60: sync_progress(active); last_sync=now
        pending=[j for j in jobs if key(j) not in active and key(j) not in blocked and not complete(j)]
        details={k:dict(**r,worker=snapshot(r['job'])) for k,r in active.items()}
        terminal=not active and not pending
        state=('COMPLETED' if not blocked and 'FAILED' not in summaries.values() else 'INCOMPLETE') if terminal else ('RUNNING' if active else 'WAITING_FOR_RESOURCES')
        status=dict(state=state,pid=os.getpid(),updated_at=stamp(),active=details,pending=pending,blocked=blocked,
            text_phase=phase,summaries=summaries,max_parallel=cfg['max_parallel'],resource_query_error=query_error,test_read=False,
            resume_command='bash experiment/phase21/run_night.sh resume',auto_start_after_reboot=False)
        write(OUT/'status.json',status)
        for pipeline,root in (('text',text_run.OUT),('recovery',recovery.OUT)):
            a={k:r for k,r in details.items() if r['job']['pipeline']==pipeline}
            pp=[j for j in pending if j['pipeline']==pipeline]; bb={k:s for k,s in blocked.items() if k.startswith(pipeline+'/')}
            sub_state='RUNNING' if a else ('WAITING_FOR_RESOURCES' if pp else ('INCOMPLETE' if bb else 'COMPLETED'))
            if pipeline=='text' and phase=='COMPLETED_SCREEN_ONLY': sub_state=phase
            write(root/'status.json',dict(state=sub_state,pid=os.getpid(),updated_at=stamp(),managed_by=str(OUT),active=a,pending=pp,blocked=bb,
                phase=phase if pipeline=='text' else 'EXPERT_CHECKS',test_read=False))
        if terminal: break
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['start','resume','worker'])
    parser.add_argument('--job'); parser.add_argument('--retry-failed',action='store_true'); args=parser.parse_args()
    if args.action=='worker':
        worker(json.loads(args.job)); return
    if args.action=='start': prepare()
    try: supervise(initial=args.action=='start',retry_failed=args.retry_failed)
    except BlockingIOError:
        print('A night controller is already running; leave it and its workers unchanged.',flush=True)
        return
    except BaseException as error:
        (OUT/f'controller_failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        write(OUT/'status.json',dict(state='CONTROLLER_ERROR',error=repr(error),updated_at=stamp(),
              worker_policy='Leave already running workers alive; explicit resume adopts them'))
        raise


if __name__=='__main__': main()
