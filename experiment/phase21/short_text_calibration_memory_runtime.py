"""User-authorized scheduler override: admit by free GPU memory, no task cap.

Frozen scientific code, configurations, completed CPU work and running jobs are
preserved. Only the idle queue supervisor is retired. GPU workers still verify
their original scientific manifests and retain their per-task memory budgets.
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

from experiment.phase21 import short_text_calibration as S

ROOT,OUT = S.ROOT,S.OUT
ARCHIVE = OUT/'scheduler_memory_override_20261017'
POLICY = dict(
    authorization='不要设限gpu gpu有位置就可以用',
    project_gpu_task_cap=None,max_new_gpu_jobs=None,older_pending_blocks_admission=False,
    share_gpu_if_memory_available=True,minimum_free_mib=18432,total_lease_mib=16384,
    expected_workload_peak_mib=12288,allocation_limit_gib=11,
    scientific_config_and_sources_unchanged=True,completed_CPU_work_reused=True,
    scientific_auto_retry=False)
GPU_MODULE = 'experiment.phase21.short_text_calibration_gpu_runtime'


def exists(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def retire_idle_supervisor():
    status = S.read(OUT/'status.json')
    pid = int(status['pid'])
    proc = Path('/proc')/str(pid)
    assert proc.stat().st_uid==os.getuid()
    command = proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert S.MODULE.encode() in command and b'supervise' in command,command
    assert status['state']=='WAITING_FOR_GPU_SLOT' and not status['active']
    ARCHIVE.mkdir(exist_ok=False)
    S.write(ARCHIVE/'previous_status.json',status)
    S.write(ARCHIVE/'authorization.json',dict(updated_at=S.stamp(),policy=POLICY,retired_pid=pid))
    # Freeze only the idle Python supervisor, then verify it has no workers.
    # This prevents a launch between the status check and the handoff.
    os.kill(pid,signal.SIGSTOP)
    clean = False
    try:
        current = S.read(OUT/'status.json')
        children = proc.joinpath('task',str(pid),'children').read_text().split()
        assert not children and not current['active'], 'Supervisor has a worker; leave it intact'
        os.kill(pid,signal.SIGTERM)
        os.kill(pid,signal.SIGCONT)
        clean = True
    finally:
        if not clean and exists(pid):
            os.kill(pid,signal.SIGCONT)
    began = time.monotonic()
    while exists(pid):
        assert time.monotonic()-began<20,'Retired supervisor did not exit'
        time.sleep(.1)
    # The retired CLI records SIGTERM as an interruption. Preserve this as
    # scheduler history instead of leaving a misleading scientific failure.
    failure = OUT/'failure.json'
    if failure.exists():
        value = S.read(failure)
        assert 'InterruptedError' in value.get('error',''),value
        shutil.move(str(failure),str(ARCHIVE/'retired_supervisor_interruption.json'))
    S.write(ARCHIVE/'handoff.json',dict(state='COMPLETED',updated_at=S.stamp(),retired_pid=pid,
        no_active_workers_interrupted=True,scientific_jobs_restarted=0))
    return status


def choices(inventory,active):
    free = {g['uuid']:float(g['free_mib']) for g in inventory}
    # Account for launches whose sidecar/model has not yet appeared in measured
    # GPU memory. This is a memory reservation, not a project concurrency cap.
    for task in active.values():
        job = OUT/'jobs'/task['job']
        smoke = job/'smoke.json'
        if not smoke.exists():
            free[task['gpu']['uuid']] -= 16384
    return sorted([dict(g,effective_free_mib=free[g['uuid']]) for g in inventory
                   if free[g['uuid']]>=18432],
                  key=lambda g:(g['utilization'],-g['effective_free_mib'],g['index']))


def recover_pre_model_startup():
    previous = S.read(OUT/'status.json')
    assert previous['state']=='STARTUP_ERROR_CORRECTING' and not previous['active']
    if (ARCHIVE/'thread_entrypoint_fix.json').exists() and all(
            S.read(OUT/'jobs'/j/'status.json')['state']=='PREPARED' for j in S.JOBS):
        # The startup archives were already committed before the controller
        # discovered that the old queue snapshot had an empty pending list.
        assert all(not (OUT/'jobs'/j/'predictions.jsonl').exists() for j in S.JOBS)
        previous['pending'],previous['finished'] = list(S.JOBS),{}
        failure = OUT/'memory_runtime_failure.json'
        if failure.exists():
            shutil.move(str(failure),str(ARCHIVE/'pending_queue_handoff_fix.json'))
        return previous
    for job in S.JOBS:
        p = OUT/'jobs'/job
        assert not (p/'predictions.jsonl').exists() and not (p/'smoke.json').exists()
        assert S.read(p/'lease_status.json')['state']=='released'
        log = (p/'memory_worker_attempt1.log').read_text()
        assert 'cannot set number of interop threads' in log
        archive = p/'pre_model_thread_initialization_failure'
        archive.mkdir(exist_ok=False)
        for path in list(p.iterdir()):
            if path.is_file():
                shutil.move(str(path),str(archive/path.name))
        count = len(S.read(S.GUIDED/S.JOBS[job]['domain']/'samples.json'))
        S.write(p/'status.json',dict(state='PREPARED',users_completed=0,users_total=count,updated_at=S.stamp(),
            startup_entrypoint_corrected=True,test_read=False))
    S.write(ARCHIVE/'thread_entrypoint_fix.json',dict(state='CORRECTED',updated_at=S.stamp(),
        error='CPU-oriented main set interop threads, then GRAM constructor attempted a second initialization',
        fix='GPU-specific entrypoint lets GRAM initialize threads once',
        no_scientific_output_from_failed_startup=True,failed_logs_preserved=True,
        scientific_sources_and_CPU_outputs_unchanged=True))
    failure = OUT/'memory_runtime_failure.json'
    if failure.exists():
        shutil.move(str(failure),str(ARCHIVE/'pre_model_memory_runtime_failure.json'))
    previous['pending'],previous['finished'] = list(S.JOBS),{}
    return previous


def run(recover_startup=False):
    previous = recover_pre_model_startup() if recover_startup else retire_idle_supervisor()
    S.write(OUT/'runtime_resource_policy.json',dict(updated_at=S.stamp(),**POLICY))
    for domain in S.DOMAINS:
        assert S.read(OUT/domain/'status.json')['state']=='CPU_COMPLETED'
        for rel,digest in S.read(OUT/domain/'cpu_output_manifest.json')['sha256'].items():
            assert S.sha(ROOT/rel)==digest,rel
    assert (OUT/'cached_summary.json').exists()
    pending = list(previous['pending'])
    assert sorted(pending)==sorted(S.JOBS) and not previous['finished']
    active,finished = {},{}
    attempts = {j:(1 if recover_startup else 0) for j in S.JOBS}
    began = time.monotonic()
    start_elapsed = previous.get('elapsed_seconds',0.)
    while pending or active:
        for job,task in list(active.items()):
            code = task['process'].poll()
            if code is None:
                continue
            task['log'].close()
            state = S.read(OUT/'jobs'/job/'status.json')['state']
            if code==75 and state=='ADMISSION_DEFERRED':
                pending.append(job)
            else:
                finished[job] = dict(exit_code=code,state=state)
                assert code==0 and state=='COMPLETED','Scientific failure; no automatic retry'
                assert S.read(OUT/'jobs'/job/'lease_status.json')['state']=='released'
            del active[job]
        inventory = S.F.B.S.OLD.gpu_inventory()
        # No limit on job count, no old-queue gate, no GPU-UUID exclusion.
        for job in list(pending):
            available = choices(inventory,active)
            if not available:
                break
            gpu = available[0]
            pending.remove(job)
            attempts[job]+=1
            log = (OUT/'jobs'/job/f'memory_worker_attempt{attempts[job]}.log').open('x')
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            process = subprocess.Popen([sys.executable,'-u','-m',GPU_MODULE,'worker','--job',job],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[job] = dict(job=job,gpu=gpu,process=process,log=log)
        external,external_pending = S.external_state()
        S.write(OUT/'status.json',dict(
            state='GUIDED_CALIBRATION_RUNNING' if active else 'WAITING_FOR_FREE_GPU_MEMORY' if pending else 'FINALIZING',
            updated_at=S.stamp(),pid=os.getpid(),pending=pending,finished=finished,
            cached_results='cached_report.md',external_active=external,external_pending=external_pending,
            active={j:dict(pid=t['process'].pid,gpu_uuid=t['gpu']['uuid'],gpu_index=t['gpu']['index'])
                    for j,t in active.items()},jobs={j:S.read(OUT/'jobs'/j/'status.json') for j in S.JOBS},
            runtime_resource_policy='runtime_resource_policy.json',project_gpu_task_cap=None,
            admission='free GPU memory only; old queue does not block',
            elapsed_seconds=start_elapsed+time.monotonic()-began,test_read=False))
        assert time.monotonic()-began<86400,'Resource wait limit'
        if pending or active:
            time.sleep(5)
    S.finalize_guided()
    S.summarize('final')
    path = OUT/'final_summary.json'
    summary = S.read(path)
    summary['runtime_resource_policy'] = POLICY
    S.write(path,summary)
    with S.REPORT.open('a') as stream:
        stream.write('\n用户随后授权取消项目GPU数量与旧队列优先限制；实际按空闲显存准入，原科学配置中的调度字段为历史快照。有效调度见runtime_resource_policy.json。科学输入、训练、候选和指标协议均未改变。\n')
    S.write(OUT/'status.json',dict(state='COMPLETED',updated_at=S.stamp(),users_completed=2048,users_total=2048,
        elapsed_seconds=start_elapsed+time.monotonic()-began,finished=finished,
        report=str(S.REPORT.relative_to(ROOT)),runtime_resource_policy='runtime_resource_policy.json',
        project_gpu_task_cap=None,test_read=False,next_experiment_started=False))


def test():
    inventory = [dict(uuid='x',free_mib=50000,utilization=0,index=0),
                 dict(uuid='y',free_mib=19000,utilization=40,index=1)]
    assert len(choices(inventory,{}))==2
    active = {'a':dict(job='nonexistent_for_fixture',gpu=inventory[0])}
    assert choices(inventory,active)[0]['effective_free_mib']==50000-16384
    assert len(choices([dict(inventory[0],free_mib=20000)],active))==0
    print(json.dumps(dict(state='PASSED',no_project_task_cap=True,no_old_queue_gate=True,
        GPU_sharing_admitted_by_free_memory=True,starting_jobs_memory_reserved=True)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','run','resume-startup'))
    args = parser.parse_args()
    if args.mode=='test':
        test()
    else:
        try:
            run(recover_startup=args.mode=='resume-startup')
        except BaseException as exc:
            S.write(OUT/'memory_runtime_failure.json',dict(state='FAILED',updated_at=S.stamp(),
                error=repr(exc),traceback=traceback.format_exc()))
            raise
