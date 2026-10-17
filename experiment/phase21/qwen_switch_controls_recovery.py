"""User-triggered one-shot OOM recovery; frozen scientific generation reused."""
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
from experiment.phase21 import qwen_switch_controls as Q

ROOT,ORIGINAL=Q.ROOT,Q.OUT
OUT=ORIGINAL/'recovery_20261017_v1'
JOB='Beauty_1'
MODULE='experiment.phase21.qwen_switch_controls_recovery'
PLAN=ROOT/'plan/第二十一阶段/21.58_Qwen4共享显存OOM缺失分片恢复_20261017.md'
read,write,sha,stamp=Q.read,Q.write,Q.sha,Q.stamp


def required_free_mib(reserved):
    return max(1024.,Q.CFG['expected_workload_peak_mib']-reserved)


def prepare():
    assert read(ORIGINAL/'status.json')['state']=='INCOMPLETE'
    failed=ORIGINAL/'jobs'/JOB
    assert read(failed/'status.json')['state']=='FAILED'
    assert read(failed/'lease_status.json')['state']=='released'
    assert 'CUDA out of memory' in (failed/'worker_attempt1.log').read_text()
    raw=(failed/'predictions.jsonl').read_bytes()
    assert raw.endswith(b'\n')
    prefix=[json.loads(line) for line in raw.splitlines()]
    full=Q.indices(Q.JOBS[JOB]['shard'])
    assert [r['sample_index'] for r in prefix]==full[:len(prefix)] and len(prefix)==12
    missing=full[len(prefix):];assert len(missing)==244
    assert required_free_mib(3584)==8704 and required_free_mib(12000)==1024
    OUT.mkdir(exist_ok=False)
    snapshot=OUT/'original_failure'/JOB;shutil.copytree(failed,snapshot)
    shutil.copy2(ORIGINAL/'status.json',OUT/'original_root_status.json')
    shutil.copy2(PLAN,OUT/'protocol.md')
    completed={}
    for job in Q.JOBS:
        if job==JOB:continue
        folder=ORIGINAL/'jobs'/job
        assert read(folder/'status.json')['state']=='COMPLETED' and read(folder/'lease_status.json')['state']=='released'
        assert read(folder/'runtime_checks.json')['state']=='PASSED'
        for name in ('predictions.jsonl','status.json','smoke.json','runtime_checks.json','lease_status.json'):
            path=folder/name;completed[str(path.relative_to(ROOT))]=sha(path)
    saved_hashes={str(p.relative_to(ROOT)):sha(p) for p in failed.iterdir() if p.is_file()}
    for path in snapshot.iterdir():
        if path.is_file():assert sha(path)==saved_hashes[str((failed/path.name).relative_to(ROOT))]
    write(OUT/'resume_protocol.json',dict(job=JOB,domain='Beauty',original_users=256,
        preserved_users=len(prefix),missing_users=len(missing),missing_indices=missing,
        prefix_sha256=sha(failed/'predictions.jsonl'),original_failed_files=saved_hashes,
        untouched_completed_files=completed,failed_gpu_uuid=read(failed/'lease_process.json')['gpu_uuid'],
        recovery_reason='Shared GPU global OOM, 41.62MiB free; one user-triggered recovery.',
        scientific_auto_retry=False,per_decode_workload_budget_mib=12288,minimum_free_mib=18432,
        same_frozen_scientific_configuration=True,test_read=False))
    shutil.copy2(ORIGINAL/'config.json',OUT/'config.json')
    (OUT/'Beauty').mkdir();(OUT/'jobs'/JOB).mkdir(parents=True)
    original_manifest=read(ORIGINAL/'manifest.json')
    manifest=dict(original_manifest['sha256'])
    inputs=list(original_manifest['domain_inputs']['Beauty'])
    for name in ('samples.json','catalog.json','data_config.json','teacher_checks.json','references.json','raw_experts.npy','loo_text.npy'):
        path=OUT/'Beauty'/name;path.symlink_to(ORIGINAL/'Beauty'/name)
        rel=str(path.relative_to(ROOT));manifest[rel]=sha(path);inputs.append(rel)
    for path in (Path(__file__).resolve(),PLAN,OUT/'config.json',OUT/'resume_protocol.json',snapshot/'predictions.jsonl'):
        rel=str(path.relative_to(ROOT));manifest[rel]=sha(path);inputs.append(rel)
    for rel in inputs:assert sha(ROOT/rel)==manifest[rel],rel
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs={'Beauty':inputs}))
    write(OUT/'jobs'/JOB/'status.json',dict(state='PREPARED',users_completed=0,users_total=244,updated_at=stamp()))
    write(OUT/'status.json',dict(state='PREPARED',users_completed=0,users_total=244,updated_at=stamp(),test_read=False))
    print(json.dumps(dict(state='PREPARED',preserved_total_users=1804,missing_users=244,old_seven_shards_unchanged=True)),flush=True)


class Generation(Q.Generation):
    def __init__(self,*args,**kwargs):
        self.protocol=read(OUT/'resume_protocol.json')
        self.memory_wait_seconds=0.
        super().__init__(*args,**kwargs)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None
        rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        total=self.protocol['missing_users']
        write(self.jobfolder/'status.json',dict(state=state,stage=self.stage,job=JOB,domain='Beauty',
            pid=os.getpid(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),users_completed=self.users_completed,
            users_total=total,preserved_original_users=12,full_shard_users_completed=12+self.users_completed,
            fresh_generations_completed=self.users_completed*6,completed_decodes=self.completed,
            median_seconds_per_user=rate,remaining_generation_seconds_estimate=(total-self.users_completed)*rate if rate else None,
            memory_wait_seconds=self.memory_wait_seconds,updated_at=stamp(),test_read=False,**extra))

    def decode(self,*args,**kwargs):
        began=time.monotonic();original_stage=self.stage;last_status=0.
        while True:
            free,_=torch.cuda.mem_get_info()
            reserved=torch.cuda.memory_reserved()/2**20
            required=required_free_mib(reserved)
            if free/2**20>=required:break
            assert self.lease.poll() is None
            self.stage='WAITING_FOR_SHARED_MEMORY_HEADROOM'
            if time.monotonic()-last_status>=5:
                self.status(free_mib=free/2**20,reserved_mib=reserved,required_free_mib=required)
                last_status=time.monotonic()
            time.sleep(2)
        self.memory_wait_seconds+=time.monotonic()-began
        self.stage=original_stage
        return super().decode(*args,**kwargs)

    def run(self):
        prefix=[json.loads(line) for line in (OUT/'original_failure'/JOB/'predictions.jsonl').read_text().splitlines()]
        replay=[];self.stage='verify_all_12_preserved_users';self.status()
        for row in prefix:
            j=row['sample_index'];ctx=self.context(j)
            for arm,value in self.scores(j).items():
                if arm not in Q.C.FRESH:continue
                prediction=self.decode(ctx,self.index.conditional_logs(value),.3)
                Q.B.Generation.equal(dict(ids=prediction['ids'],scores=prediction['scores'].tolist()),row[arm])
            replay.append(j)
            write(self.jobfolder/'prefix_replay.json',dict(state='PASSED_SO_FAR',users_completed=len(replay),users_total=12,indices=replay))
        write(self.jobfolder/'prefix_replay.json',dict(state='PASSED',users=12,arms_per_user=6,
            all_ids_and_scores_match_original=True,original_predictions_kept=True))
        # Same frozen run()/score()/decode math, only missing user indices.
        super().run()
        rows=[json.loads(line) for line in (self.jobfolder/'predictions.jsonl').read_text().splitlines()]
        assert [r['sample_index'] for r in rows]==self.protocol['missing_indices']
        check=read(self.jobfolder/'runtime_checks.json')
        check.update(users=len(rows),original_shard_users=256,preserved_original_users=12,
            preserved_prefix_replay_passed=True,scientific_source_unchanged=True,
            per_decode_headroom_guard=True,memory_wait_seconds=self.memory_wait_seconds)
        write(self.jobfolder/'runtime_checks.json',check)
        self.status('COMPLETED')


def worker():
    Q.F.OUT,Q.F.CFG,Q.F.JOBS,Q.F.MODULE=OUT,Q.CFG,{JOB:Q.JOBS[JOB]},MODULE
    return Q.F.worker(JOB)


def commit_and_summarize():
    protocol=read(OUT/'resume_protocol.json');folder=OUT/'jobs'/JOB;old=ORIGINAL/'jobs'/JOB
    assert read(folder/'status.json')['state']=='COMPLETED' and read(folder/'lease_status.json')['state']=='released'
    assert read(folder/'runtime_checks.json')['state']=='PASSED' and read(folder/'prefix_replay.json')['state']=='PASSED'
    for rel,digest in protocol['untouched_completed_files'].items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in protocol['original_failed_files'].items():assert sha(ROOT/rel)==digest,rel
    prefix=(old/'predictions.jsonl').read_bytes();new=(folder/'predictions.jsonl').read_bytes()
    assert sha(old/'predictions.jsonl')==protocol['prefix_sha256'] and new.endswith(b'\n')
    merged=prefix+new;records=[json.loads(line) for line in merged.splitlines()]
    assert [r['sample_index'] for r in records]==Q.indices(Q.JOBS[JOB]['shard'])
    samples=read(ORIGINAL/'Beauty/samples.json')
    for row in records:
        s=samples[row['sample_index']]
        assert all(row[k]==s[k] for k in ('user_id','target','history','evaluation_block'))
        for arm in Q.C.FRESH:
            p=row[arm];assert len(p['ids'])==len(set(p['ids']))==len(p['scores'])==50 and np.isfinite(p['scores']).all()
    receipt=dict(state='COMPLETED',updated_at=stamp(),preserved_users=12,resumed_users=244,total_users=256,
        original_prefix_byte_identical=True,all_12_original_users_replayed=True,old_seven_shards_hashes_unchanged=True,
        recovery_path=str(OUT.relative_to(ROOT)),scientific_inputs_unchanged=True,test_read=False)
    write(OUT/'commit_review.json',receipt)
    temporary=old/'predictions.recovered.jsonl.tmp';temporary.write_bytes(merged);temporary.replace(old/'predictions.jsonl')
    (old/'failure.json').replace(old/'failure_before_recovery.json')
    for name in ('lease_status.json','lease_process.json'):
        shutil.copy2(folder/name,old/name)
    shutil.copy2(folder/'smoke.json',old/'recovery_smoke.json')
    shutil.copy2(folder/'prefix_replay.json',old/'recovery_prefix_replay.json')
    shutil.copy2(folder/'successful_worker.log',old/'worker_recovery1.log')
    checks=read(folder/'runtime_checks.json');checks.update(users=256,recovered_after_shared_gpu_oom=True)
    write(old/'runtime_checks.json',checks)
    status=read(folder/'status.json');status.update(users_completed=256,users_total=256,
        fresh_generations_completed=1536,recovered_after_shared_gpu_oom=True,remaining_generation_seconds_estimate=0.)
    write(old/'status.json',status);write(old/'recovery_receipt.json',receipt)
    Q.summarize()
    previous=read(OUT/'original_root_status.json');finished=previous['finished']
    finished[JOB]=dict(exit_code=0,state='COMPLETED',lease_released=True,recovered_after_shared_gpu_oom=True)
    write(ORIGINAL/'status.json',dict(state='COMPLETED',finished=finished,users_completed=2048,users_total=2048,
        updated_at=stamp(),report=str(Q.REPORT.relative_to(ROOT)),recovery=str(OUT.relative_to(ROOT)),
        next_experiment_started=False,test_read=False))
    write(OUT/'status.json',dict(state='COMPLETED',users_completed=244,users_total=244,updated_at=stamp(),
        original_experiment_complete=True,test_read=False))


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    protocol=read(OUT/'resume_protocol.json');began=time.monotonic();task=None;attempt=0
    try:
        with (OUT/'memory_trace.jsonl').open('x') as trace:
          while True:
            while task is None:
                candidates=sorted([g for g in Q.B.S.OLD.gpu_inventory() if g['free_mib']>=18432],
                    key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
                if candidates:
                    gpu=candidates[0];attempt+=1
                    log_path=OUT/'jobs'/JOB/f'worker_attempt{attempt}.log';log=log_path.open('x')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                    process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',JOB],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    task=dict(process=process,log=log,gpu=gpu,log_path=log_path)
                    write(OUT/'launch.json',dict(updated_at=stamp(),gpu=gpu,selection='most currently free memory',one_user_triggered_recovery=True,admission_attempt=attempt))
                    break
                write(ORIGINAL/'status.json',dict(state='WAITING_FOR_RECOVERY_GPU_MEMORY',users_completed=1804,users_total=2048,
                    missing_users=244,recovery=str(OUT.relative_to(ROOT)),updated_at=stamp(),test_read=False))
                assert time.monotonic()-began<86400;time.sleep(5)
            while task['process'].poll() is None:
                s=read(OUT/'jobs'/JOB/'status.json')
                inventory=Q.B.S.OLD.gpu_inventory();g=next(g for g in inventory if g['uuid']==task['gpu']['uuid'])
                trace.write(json.dumps(dict(updated_at=stamp(),gpu=g,job_state=s.get('state'),stage=s.get('stage'),users_completed=s.get('users_completed',0)))+'\n');trace.flush()
                current=dict(state='RECOVERING_SHARED_GPU_OOM',users_completed=1804+s.get('users_completed',0),
                    users_total=2048,missing_users=244-s.get('users_completed',0),preserved_users=1804,
                    active={JOB:dict(pid=task['process'].pid,gpu_index=g['index'],gpu_uuid=g['uuid'])},
                    recovery_job_status=s,recovery=str(OUT.relative_to(ROOT)),updated_at=stamp(),test_read=False)
                write(ORIGINAL/'status.json',current)
                write(OUT/'status.json',dict(state='RUNNING',users_completed=s.get('users_completed',0),users_total=244,updated_at=stamp(),job_status=s,test_read=False))
                time.sleep(5)
            code=task['process'].returncode;task['log'].close()
            if code==75 and read(OUT/'jobs'/JOB/'status.json')['state']=='ADMISSION_DEFERRED':
                task=None
                assert time.monotonic()-began<86400
                continue
            assert code==0, 'Recovery failed; no further automatic retry'
            shutil.copy2(task['log_path'],OUT/'jobs'/JOB/'successful_worker.log')
            commit_and_summarize()
            break
    finally:
        if task is not None:
            Q.B.stop(task['process']);task['log'].close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','supervise','worker','generate'))
    p.add_argument('--job',choices=(JOB,));p.add_argument('--lease-pid',type=int);args=p.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Qwen OOM recovery interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='worker':return worker()
        else:
            protocol=read(OUT/'resume_protocol.json')
            Q.OUT,Q.A.OUT,Q.A.JOBS=OUT,OUT,Q.JOBS
            Q.indices=lambda shard:protocol['missing_indices']
            Generation(JOB,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            folder=OUT/'jobs'/JOB if args.mode in ('worker','generate') else OUT
            value=dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(folder/'failure.json',value);write(folder/'status.json',value)
            if args.mode=='supervise':
                write(ORIGINAL/'status.json',dict(state='RECOVERY_FAILED',recovery=str(OUT.relative_to(ROOT)),error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':sys.exit(main())
