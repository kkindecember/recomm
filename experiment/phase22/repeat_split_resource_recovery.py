"""One documented OOM recovery; adopt live workers without stopping them."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase22 import repeat_split_parallel as P
from experiment.phase21.night_resume_core import process_identity, identity_alive

OUT, ROOT = P.OUT, P.ROOT
FOLDER = OUT/'resource_recovery_20261017'
PLAN = ROOT/'plan/第二十二阶段/22.3_Beauty显存故障接续记录_20261017.md'
MODULE = 'experiment.phase22.repeat_split_resource_recovery'
TARGET = 'Beauty_s2'
BAD_GPU = 'GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160'


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []


def eligible(gpus,active):
    slots = Counter(t['gpu']['uuid'] for t in active.values())
    return sorted([g for g in gpus if g['uuid']!=BAD_GPU and slots[g['uuid']]<2
        and g['free_mib'] >= (32768 if slots[g['uuid']] else P.CFG['minimum_free_mib'])],
        key=lambda g:(-g['free_mib'],g['utilization'],g['index']))


def tests():
    def gpu(uuid,free):
        return dict(uuid=uuid,free_mib=free,index=0,utilization=0)
    a={'job':dict(gpu=gpu('a',40000))}
    assert [g['uuid'] for g in eligible([gpu('a',33000),gpu('b',19000),gpu(BAD_GPU,49000)],a)]==['a','b']
    assert not eligible([gpu('a',32000)],a)
    a['job2']=dict(gpu=gpu('a',49000))
    assert not eligible([gpu('a',49000)],a)
    assert not eligible([gpu('b',18000)],{})
    return dict(state='PASSED',failed_gpu_excluded=True,double_slot_headroom=True,
                max_two_tasks_per_gpu=True,ordinary_admission_preserved=True)


def freeze():
    assert not (FOLDER/'manifest.json').exists(),'Recovery already prepared'
    FOLDER.mkdir(parents=True,exist_ok=True)
    folder=OUT/'jobs'/TARGET
    previous=P.read(folder/'status.json')
    assert previous['state']=='FAILED' and previous['users_completed']==0
    logs=sorted(folder.glob('worker_*.log'))
    assert logs and 'RuntimeError: CUDA out of memory.' in logs[-1].read_text()
    assert P.read(folder/'gpu_lease_status.json')['state']=='released'
    frozen={}
    for p in (PLAN,Path(__file__),ROOT/'experiment/phase22/run_repeat_split_resource_recovery.sh',
              folder/'status.json',folder/'gpu_lease_status.json',logs[-1]):
        frozen[str(p.relative_to(ROOT))]=P.sha(p)
        dest=FOLDER/'before'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    P.write(FOLDER/'manifest.json',dict(sha256=frozen,source_manifest_sha256=P.sha(OUT/'manifest.json'),
        target=TARGET,reason='documented shared-GPU OOM before formal predictions',
        one_compute_retry_only=True,tests=tests(),created_at=P.stamp()))


def handoff():
    s=P.read(OUT/'status.json')
    assert not s.get('pending'),'Only a fully launched controller can be adopted'
    assert TARGET not in s.get('active',{})
    pid=s['pid']; identity=process_identity(pid)
    argv=command(pid)
    assert identity and b'experiment.phase22.repeat_split_parallel' in argv and b'supervise' in argv,argv
    for job in P.JOBS:
        if job.endswith('_fit'):
            assert P.read(OUT/'jobs'/job/'status.json')['state']=='COMPLETED'
    # Stop only the old controller so its snapshot cannot change during adoption.
    os.kill(pid,signal.SIGSTOP)
    try:
        assert identity_alive(identity)
        snapshot=P.read(OUT/'status.json')
        active={}
        for job,t in snapshot.get('active',{}).items():
            child=t['pid']; ci=process_identity(child)
            if ci:
                cmd=command(child)
                assert b'experiment.phase22.repeat_split_parallel' in cmd and b'worker' in cmd
                assert job.encode() in cmd
                active[job]=dict(identity=ci,gpu=t['gpu'],adopted=True)
            else:
                assert P.read(OUT/'jobs'/job/'status.json')['state'] in ('COMPLETED','FAILED')
        P.write(FOLDER/'handoff.json',dict(old_controller=identity,root_before=snapshot,
            active_workers=active,created_at=P.stamp(),scientific_worker_interruptions=0))
        assert identity_alive(identity)
        # SIGKILL bypasses the old controller's finally block which would stop workers.
        # These workers have independent process sessions; never signal their groups.
        os.kill(pid,signal.SIGKILL)
    except BaseException:
        if identity_alive(identity):
            os.kill(pid,signal.SIGCONT)
        raise
    for _ in range(20):
        if not identity_alive(identity):
            break
        time.sleep(.25)
    assert not identity_alive(identity)
    lock=(OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for job,t in active.items():
        assert identity_alive(t['identity']) or P.read(OUT/'jobs'/job/'status.json')['state'] in ('COMPLETED','FAILED')
    return lock,active,snapshot


def supervise():
    guard=(FOLDER/'recovery.lock').open('a'); fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=P.read(FOLDER/'manifest.json')
    assert P.sha(OUT/'manifest.json')==manifest['source_manifest_sha256']
    for rel,digest in manifest['sha256'].items():
        assert P.sha(ROOT/rel)==digest,rel
    P.R.verify()
    lock,active,before=handoff()
    began=time.monotonic(); waiting=True; attempt=0; children={}; logs={}; finished={}
    for job in P.JOBS:
        if job not in active and job!=TARGET:
            st=P.read(OUT/'jobs'/job/'status.json')
            finished[job]=dict(state=st['state'],exit_code=0 if st['state']=='COMPLETED' else 1)
    P.status(FOLDER,'RUNNING',stage='adopted_existing_workers',active=list(active),pending=[TARGET])
    try:
        while active or waiting:
            for job,t in list(active.items()):
                if job in children:
                    children[job].poll()
                if identity_alive(t['identity']):
                    continue
                st=P.read(OUT/'jobs'/job/'status.json')
                code=children[job].wait() if job in children else (0 if st['state']=='COMPLETED' else 1)
                if job in logs:
                    logs.pop(job).close()
                if job==TARGET and code==75 and st['state']=='ADMISSION_DEFERRED':
                    waiting=True
                else:
                    finished[job]=dict(exit_code=code,state=st['state'])
                del active[job]
            error=None
            if waiting:
                try:
                    available=eligible(P.R.inventory(),active)
                except (OSError,subprocess.SubprocessError,ValueError) as e:
                    available=[]; error=repr(e)
                if available and len(active)<8:
                    gpu=available[0]; folder=OUT/'jobs'/TARGET; attempt+=1
                    P.status(folder,'STARTING',domain='Beauty',kind='generate',shard=2,
                        users_completed=0,users_total=5591,gpu_uuid=gpu['uuid'],gpu_index=gpu['index'],
                        resource_recovery=True,prior_failure='shared-GPU OOM; preserved in recovery/before')
                    logpath=folder/f'worker_resource_recovery_{time.time_ns()}.log'
                    log=logpath.open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m',P.MODULE,'worker','--job',TARGET],
                        cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']),stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                    ci=process_identity(proc.pid); assert ci
                    children[TARGET]=proc; logs[TARGET]=log
                    active[TARGET]=dict(identity=ci,gpu=gpu,adopted=False)
                    waiting=False
                    P.write(folder/f'launch_resource_recovery_{proc.pid}.json',dict(pid=proc.pid,gpu=gpu,
                        log=str(logpath),identity=ci,started_at=P.stamp(),source_manifest_sha256=manifest['source_manifest_sha256']))
                elif time.monotonic()-began>P.CFG['resource_wait_seconds']:
                    P.status(OUT/'jobs'/TARGET,'RESOURCE_WAIT_LIMIT',users_completed=0,users_total=5591)
                    finished[TARGET]=dict(state='RESOURCE_WAIT_LIMIT',exit_code=124); waiting=False
            states={j:P.read(OUT/'jobs'/j/'status.json') for j in P.JOBS}
            domains={}
            for d in P.R.DOMAINS:
                members={j:st for j,st in states.items() if P.JOBS[j]['domain']==d}
                done=sum(st.get('users_completed',0) for j,st in members.items() if P.JOBS[j]['kind']=='generate')
                state='COMPLETED' if all(st['state']=='COMPLETED' for st in members.values()) else (
                    'INCOMPLETE' if any(st['state']=='FAILED' for st in members.values()) else 'RUNNING')
                domains[d]=dict(state=state,users_completed=done,users_total=P.R.COUNTS[d],jobs=members,updated_at=P.stamp())
                P.write(OUT/d/'status.json',domains[d])
            common=dict(users_completed=sum(d['users_completed'] for d in domains.values()),users_total=sum(P.R.COUNTS.values()),
                active={j:dict(pid=t['identity']['pid'],gpu=t['gpu'],adopted=t['adopted']) for j,t in active.items()},
                pending=[TARGET] if waiting else [],finished=finished,domains=domains,
                resource_recovery=dict(job=TARGET,attempts=attempt,original_failure_preserved=True,
                    handoff=str((FOLDER/'handoff.json').relative_to(ROOT))),resource_query_error=error,
                elapsed_seconds=before.get('elapsed_seconds',0)+time.monotonic()-began,
                report=str(P.REPORT.relative_to(ROOT)))
            P.status(OUT,'RUNNING' if active else 'WAITING_FOR_RESOURCES' if waiting else 'FINALIZING',**common)
            P.status(FOLDER,'RUNNING' if active or waiting else 'FINALIZING',active=list(active),pending=[TARGET] if waiting else [],finished=finished)
            if active or waiting:
                time.sleep(15)
        ok=len(finished)==len(P.JOBS) and all(v['state']=='COMPLETED' and v['exit_code']==0 for v in finished.values())
        if ok:
            P.status(OUT,'SUMMARIZING',**common)
            P.merge()
        P.status(OUT,'COMPLETED' if ok else 'INCOMPLETE',**common,next_experiment_started=False)
        P.status(FOLDER,'COMPLETED' if ok else 'INCOMPLETE',finished=finished)
    finally:
        # Never stop adopted scientific workers if this auxiliary controller fails.
        for job,proc in children.items():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=60)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    pass
        for log in logs.values():
            log.close()


def start():
    freeze()
    session='p22_repeat_split_resource_20261017'
    found=subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert found.returncode!=0,'Recovery session already exists'
    command='bash '+shlex.quote(str(ROOT/'experiment/phase22/run_repeat_split_resource_recovery.sh'))+' supervise'
    command+=' >> '+shlex.quote(str(FOLDER/'controller.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'bash','-lc',command],cwd=ROOT,check=True)
    print(json.dumps(dict(state='STARTED',session=session,status=str(OUT/'status.json'))),flush=True)


def main():
    P.bind()
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('test','start','supervise'))
    args=p.parse_args()
    if args.action=='test':
        print(json.dumps(tests()),flush=True)
    elif args.action=='start':
        start()
    else:
        try:
            supervise()
        except BaseException as e:
            (FOLDER/f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
            P.status(FOLDER,'FAILED',error=repr(e))
            raise


if __name__=='__main__':
    main()
