"""Eight disjoint user shards, one shared scalar fit per domain, dynamic GPUs."""
import argparse
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

import numpy as np
from experiment.phase22 import repeat_split as R
from experiment.phase22 import repeat_split_core as C

ROOT = R.ROOT
OUT = ROOT/'artifacts/phase22/repeat_split_minimal_20261017_parallel_v2'
PLAN = ROOT/'plan/第二十二阶段/22.2_多卡分片执行补充_20261017.md'
REPORT = ROOT/'report/第二十二阶段/探索重复拆分最小全量验证_多卡_20261017.md'
MODULE = 'experiment.phase22.repeat_split_parallel'
CFG = dict(R.CFG, max_parallel=8, shards_per_domain=4, total_lease_mib=16384,
    minimum_free_mib=18432, expected_workload_peak_mib=12288,
    shared_rho_fit=True, gpu_policy='one project task per GPU, all eligible visible GPUs')
JOBS = {d+'_fit':dict(domain=d,kind='fit') for d in R.DOMAINS}
JOBS.update({d+'_s'+str(s):dict(domain=d,kind='generate',shard=s)
             for d in R.DOMAINS for s in range(CFG['shards_per_domain'])})
read, write, sha, stamp, status = R.read, R.write, R.sha, R.stamp, R.status


def bind():
    R.OUT, R.CFG, R.PLAN, R.REPORT = OUT, CFG, PLAN, REPORT


def indices(domain, shard):
    return list(range(shard,R.COUNTS[domain],CFG['shards_per_domain']))


def recover(path, order, samples):
    count = 0
    if not path.exists():
        return count
    with path.open('r+b') as f:
        while True:
            start = f.tell(); line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                f.truncate(start); f.flush(); os.fsync(f.fileno()); break
            row = json.loads(line)
            assert count < len(order)
            j = order[count]
            assert row['sample_index'] == j
            assert all(row[k] == samples[j][k] for k in ('user_id','target','history'))
            assert set(row['methods']) == set(R.ARMS)
            for a in R.ARMS:
                C.ordered(row['methods'][a])
            count += 1
    return count


def runtime_tests():
    from tempfile import TemporaryDirectory
    for d in R.DOMAINS:
        all_indices = [j for s in range(4) for j in indices(d,s)]
        assert len(all_indices) == len(set(all_indices)) == R.COUNTS[d]
        assert sorted(all_indices) == list(range(R.COUNTS[d]))
        assert [list(range(s,R.COUNTS[d],4)) for s in range(4)] == [indices(d,s) for s in range(4)]
    with TemporaryDirectory(dir='/tmp',prefix='p22_shard_') as tmp:
        p = Path(tmp)/'records'
        samples = [dict(user_id=str(j),target=j+1,history=[99]) for j in range(9)]
        order = [1,5]
        rows = [dict(sample_index=j,**samples[j],methods={a:dict(ids=list(range(1,51)),scores=list(range(50))) for a in R.ARMS}) for j in order]
        content = ''.join(json.dumps(row)+'\n' for row in rows)
        p.write_text(content+'{"partial')
        assert recover(p,order,samples) == 2 and p.read_text() == content
        p.write_text(json.dumps(rows[1])+'\n')
        try:
            recover(p,order,samples)
            raise RuntimeError('Wrong shard row accepted')
        except AssertionError:
            pass
    return dict(state='PASSED',complete_disjoint_shards=True,deterministic_order=True,
                shard_resume_partial_tail=True,wrong_shard_rejected=True)


def prepare():
    checks = runtime_tests()
    R.prepare()
    write(OUT/'runtime_tests.json',checks)
    manifest = read(OUT/'manifest.json')
    original = ROOT/'plan/第二十二阶段/22.1_探索重复拆分最小全量验证_20261017.md'
    rel = str(original.relative_to(ROOT))
    manifest['sha256'][rel] = sha(original)
    for paths in manifest['domain_inputs'].values():
        paths.append(rel)
    dest = OUT/'sources'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(original,dest)
    write(OUT/'manifest.json',manifest)
    for job, spec in JOBS.items():
        total = 0 if spec['kind']=='fit' else len(indices(spec['domain'],spec['shard']))
        status(OUT/'jobs'/job,'PREPARED',**spec,users_completed=0,users_total=total)
    status(OUT,'PREPARED',users_completed=0,users_total=sum(R.COUNTS.values()),
           pending=list(JOBS),new_generations_total=4*sum(R.COUNTS.values()),max_parallel=8)


class Generation(R.Generation):
    def __init__(self,job):
        self.job, self.spec = job, JOBS[job]
        self.jobfolder = OUT/'jobs'/job
        super().__init__(self.spec['domain'])
        self.domainfolder = self.folder

    def status(self,state='RUNNING',**kw):
        total = 0 if self.spec['kind']=='fit' else len(indices(self.domain,self.spec['shard']))
        done = getattr(self,'users_completed',0)
        times = getattr(self,'row_times',[])
        estimate = float(np.median(times[-64:]))*(total-done) if times else None
        values = dict(**self.spec,job=self.job,stage=self.stage,users_completed=done,users_total=total,
            rho=getattr(self,'rho',None),remaining_seconds_estimate=estimate,
            elapsed_seconds=time.monotonic()-self.started,completed_decodes=self.completed,
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
        values.update(kw)
        status(self.jobfolder,state,**values)

    def fit(self):
        self.fit_rho()
        self.smoke()
        assert self.digest() == self.initial_digest
        write(self.domainfolder/'fit_complete.json',dict(state='PASSED',rho=self.rho,
            model_digest=self.initial_digest,fit_once=True,validation_labels_used=False))
        self.stage = 'fit_complete'; self.status('COMPUTE_COMPLETED')

    def generate(self):
        for name,digest in read(self.domainfolder/'fit_manifest.json')['sha256'].items():
            assert sha(self.domainfolder/name) == digest
        assert read(self.domainfolder/'fit_complete.json')['state'] == 'PASSED'
        self.rho = read(self.domainfolder/'rho.json')['rho']
        self.folder = self.jobfolder
        self.smoke()
        order = indices(self.domain,self.spec['shard'])
        self.users_completed = recover(self.folder/'predictions.jsonl',order,self.truth)
        self.stage = 'sharded_full_validation_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('a') as stream:
            for j in order[self.users_completed:]:
                began = time.monotonic()
                s = self.truth[j]; ctx = self.context(j)
                raw = (self.weights[j,:,None]*np.asarray(self.raw[j])).sum(0)
                bounded = raw.copy()
                bounded[np.unique(s['full_history'])-1] -= self.penalties[j]
                np.testing.assert_allclose(bounded,self.bounded[j],atol=2e-6,rtol=1e-5)
                h = s['full_history']
                methods = dict(native=self.references[j]['native'],bounded=self.references[j]['guided'])
                for arm,q,alpha,counts in (
                    ('raw',self.index.conditional_logs(raw),.3,None),
                    ('split_zero',C.conditional_logs(self.index,raw,h,0.),.3,None),
                    ('split_learned',C.conditional_logs(self.index,raw,h,self.rho),.3,None),
                    ('native_norepeat',None,0.,C.novel_counts(self.index,h))):
                    if self.users_completed%8 == 0:
                        self.status(current_arm=arm,current_sample_index=j)
                    result = self.decode(ctx,q,alpha,counts)
                    methods[arm] = dict(ids=result['ids'],scores=result['scores'].tolist())
                    if arm == 'native_norepeat':
                        assert set(result['ids']).isdisjoint(h)
                row = dict(sample_index=j,user_id=s['user_id'],target=s['target'],history=s['history'],
                    rho=self.rho,original_history_mass=C.components(raw,h)[1],methods=methods)
                stream.write(json.dumps(row,separators=(',',':'))+'\n'); stream.flush(); os.fsync(stream.fileno())
                self.users_completed += 1; self.row_times.append(time.monotonic()-began)
                if self.users_completed == 1 or self.users_completed%8 == 0 or j == order[-1]:
                    self.status(last_completed_user=s['user_id'])
                del ctx,methods,result
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json',dict(state='PASSED',users=len(order),
            indices=order,gram_digest_unchanged=True,model_digest=self.initial_digest,
            rho=self.rho,shared_fit_manifest_sha256=sha(self.domainfolder/'fit_manifest.json')))
        self.stage = 'generation_complete'; self.status('COMPUTE_COMPLETED')


def worker(job):
    spec = JOBS[job]; folder = OUT/'jobs'/job
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease = child = None; code = 1
    try:
        R.verify(spec['domain'])
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = {g['uuid']:g['free_mib'] for g in R.inventory()}
        if free.get(gpu,0) < CFG['minimum_free_mib']:
            status(folder,'ADMISSION_DEFERRED',free_mib=free.get(gpu))
            return 75
        leasepath = folder/'gpu_lease_status.json'
        if leasepath.exists():
            leasepath.rename(folder/f'lease_previous_{time.time_ns()}.json')
        with (folder/'lease.log').open('a') as log:
            lease = subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),
                '--gpu','0','--total-lease-mib',str(CFG['total_lease_mib']),
                '--expected-workload-peak-mib',str(CFG['expected_workload_peak_mib']),
                '--status-path',str(leasepath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            began = time.monotonic()
            while not leasepath.exists() or read(leasepath).get('state')!='holding':
                assert lease.poll() is None,'Lease failed'
                assert time.monotonic()-began < 180,'Lease startup timeout'
                time.sleep(1)
            child = subprocess.Popen([sys.executable,'-u','-m',MODULE,'compute','--job',job],cwd=ROOT)
            began = time.monotonic()
            while child.poll() is None:
                assert lease.poll() is None,'GPU lease exited'
                if time.monotonic()-began > CFG['worker_timeout_seconds']:
                    raise TimeoutError('48h job computation limit')
                time.sleep(5)
            assert child.returncode == 0,'Compute failed; see worker log'
        code = 0
    except BaseException as error:
        (folder/f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        old = read(folder/'status.json')
        status(folder,'FAILED',error=repr(error),users_completed=old.get('users_completed',0),
               users_total=old.get('users_total',0),**spec)
    finally:
        R.stop(child); R.stop(lease)
        if code == 0:
            assert read(folder/'gpu_lease_status.json')['state']=='released'
            old = read(folder/'status.json')
            status(folder,'COMPLETED',stage='complete',**spec,rho=old.get('rho'),
                   users_completed=old.get('users_completed',0),users_total=old.get('users_total',0),lease_released=True)
    return code


def merge():
    for d in R.DOMAINS:
        folder = OUT/d; sources = [OUT/'jobs'/(d+'_s'+str(s)) for s in range(4)]
        release = {}
        for job in [d+'_fit']+[d+'_s'+str(s) for s in range(4)]:
            p = OUT/'jobs'/job
            assert read(p/'status.json')['state']=='COMPLETED'
            assert read(p/'gpu_lease_status.json')['state']=='released'
            release[job] = sha(p/'gpu_lease_status.json')
        for s,p in enumerate(sources):
            ck = read(p/'runtime_checks.json')
            assert ck['state']=='PASSED' and ck['indices']==indices(d,s)
        samples = read(folder/'samples.json')
        handles = [(p/'predictions.jsonl').open() for p in sources]
        tmp = folder/'predictions.partial.jsonl'
        try:
            with tmp.open('w') as output:
                for j,s in enumerate(samples):
                    line = handles[j%4].readline()
                    row = json.loads(line)
                    assert row['sample_index']==j and all(row[k]==s[k] for k in ('user_id','target','history'))
                    assert set(row['methods'])==set(R.ARMS)
                    output.write(line)
                assert all(not h.readline() for h in handles)
                output.flush(); os.fsync(output.fileno())
        finally:
            for h in handles:
                h.close()
        tmp.replace(folder/'predictions.jsonl')
        write(folder/'runtime_checks.json',dict(state='PASSED',users=len(samples),
            complete_disjoint_shards=True,all_shard_digests_unchanged=True,
            model_digest=read(folder/'fit_complete.json')['model_digest']))
        write(folder/'gpu_lease_status.json',dict(state='released',kind='all_five_domain_tasks_verified',sources_sha256=release))
    R.summarize()


def supervise(resume=False):
    lock = (OUT/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert resume or read(OUT/'status.json')['state']=='PREPARED'
    status(OUT,'VERIFYING_INPUTS',users_completed=0,users_total=sum(R.COUNTS.values()))
    R.verify()
    finished,active,attempts = {},{}, {j:0 for j in JOBS}
    for job in JOBS:
        with (OUT/'jobs'/job/'worker.lock').open('a') as probe:
            fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        st = read(OUT/'jobs'/job/'status.json')
        if st['state']=='COMPLETED':
            finished[job] = dict(exit_code=0,state='COMPLETED')
        elif not resume:
            assert st['state']=='PREPARED'
    pending = [j for j in JOBS if j not in finished]
    began = time.monotonic()
    try:
        while pending or active:
            for job,t in list(active.items()):
                code = t['process'].poll()
                if code is None:
                    continue
                t['log'].close(); st = read(OUT/'jobs'/job/'status.json')
                if code==75 and st['state']=='ADMISSION_DEFERRED':
                    pending.append(job)
                else:
                    finished[job] = dict(exit_code=code,state=st['state'])
                del active[job]
            # A failed scientific fit cannot authorize its dependent generation.
            for job in list(pending):
                spec = JOBS[job]; parent = spec['domain']+'_fit'
                if spec['kind']=='generate' and parent in finished and finished[parent]['exit_code']!=0:
                    status(OUT/'jobs'/job,'DEPENDENCY_FAILED',**spec,users_completed=0,
                           users_total=len(indices(spec['domain'],spec['shard'])))
                    finished[job] = dict(exit_code=1,state='DEPENDENCY_FAILED'); pending.remove(job)
            error = None
            try:
                gpus = R.inventory()
            except (OSError,subprocess.SubprocessError,ValueError) as e:
                gpus = []; error = repr(e)
            used = {t['gpu']['uuid'] for t in active.values()}
            available = sorted([g for g in gpus if g['uuid'] not in used and g['free_mib']>=CFG['minimum_free_mib']],
                               key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
            # Interleave domains to avoid giving every newly free card to one domain.
            queue = sorted(pending,key=lambda j:(0 if JOBS[j]['kind']=='fit' else 1,
                JOBS[j].get('shard',-1),JOBS[j]['domain']))
            for job in queue:
                if not available or len(active)>=CFG['max_parallel']:
                    break
                spec = JOBS[job]
                if spec['kind']=='generate' and (spec['domain']+'_fit' not in finished or
                        finished[spec['domain']+'_fit']['exit_code']!=0):
                    continue
                gpu = available.pop(0); pending.remove(job); attempts[job]+=1
                old = read(OUT/'jobs'/job/'status.json')
                status(OUT/'jobs'/job,'STARTING',**spec,users_completed=old.get('users_completed',0),
                       users_total=old.get('users_total',0),gpu_index=gpu['index'],gpu_uuid=gpu['uuid'])
                path = OUT/'jobs'/job/f'worker_{time.time_ns()}.log'; log = path.open('x')
                proc = subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',job],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[job] = dict(process=proc,log=log,gpu=gpu)
                write(OUT/'jobs'/job/f'launch_{proc.pid}.json',dict(pid=proc.pid,gpu=gpu,log=str(path),started_at=stamp()))
            if pending and time.monotonic()-began>CFG['resource_wait_seconds']:
                for job in pending:
                    status(OUT/'jobs'/job,'RESOURCE_WAIT_LIMIT',users_completed=0,users_total=0)
                    finished[job] = dict(exit_code=124,state='RESOURCE_WAIT_LIMIT')
                pending.clear()
            states = {j:read(OUT/'jobs'/j/'status.json') for j in JOBS}
            domains = {}
            for d in R.DOMAINS:
                members = {j:s for j,s in states.items() if JOBS[j]['domain']==d}
                done = sum(s.get('users_completed',0) for j,s in members.items() if JOBS[j]['kind']=='generate')
                state = 'COMPLETED' if all(s['state']=='COMPLETED' for s in members.values()) else (
                    'INCOMPLETE' if any(s['state'] in ('FAILED','DEPENDENCY_FAILED') for s in members.values()) else 'RUNNING')
                domains[d] = dict(state=state,users_completed=done,users_total=R.COUNTS[d],jobs=members,updated_at=stamp())
                write(OUT/d/'status.json',domains[d])
            status(OUT,'RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                users_completed=sum(d['users_completed'] for d in domains.values()),users_total=sum(R.COUNTS.values()),
                active={j:dict(pid=t['process'].pid,gpu=t['gpu']) for j,t in active.items()},
                pending=pending,finished=finished,domains=domains,resource_query_error=error,
                new_generations_total=4*sum(R.COUNTS.values()),max_parallel=CFG['max_parallel'],
                elapsed_seconds=time.monotonic()-began,report=str(REPORT.relative_to(ROOT)))
            if active or pending:
                time.sleep(15)
        ok = len(finished)==len(JOBS) and all(s['exit_code']==0 and s['state']=='COMPLETED' for s in finished.values())
        if ok:
            status(OUT,'SUMMARIZING',users_completed=sum(R.COUNTS.values()),users_total=sum(R.COUNTS.values()))
            merge()
        status(OUT,'COMPLETED' if ok else 'INCOMPLETE',finished=finished,domains=domains,
            users_completed=sum(d['users_completed'] for d in domains.values()),users_total=sum(R.COUNTS.values()),
            report=str(REPORT.relative_to(ROOT)),next_experiment_started=False)
    finally:
        for t in active.values():
            try:
                os.killpg(t['process'].pid,signal.SIGTERM); t['process'].wait(timeout=60)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                pass
            t['log'].close()


def start(resume=False):
    assert (OUT/'manifest.json').exists()
    session = 'p22_repeat_split_parallel_20261017'
    found = subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert found.returncode!=0,'Background session already exists'
    command = 'bash '+shlex.quote(str(ROOT/'experiment/phase22/run_repeat_split_parallel.sh'))
    command += ' '+('resume-controller' if resume else 'supervise')
    command += ' >> '+shlex.quote(str(OUT/'controller.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'bash','-lc',command],cwd=ROOT,check=True)
    write(OUT/f'background_launch_{time.time_ns()}.json',dict(session=session,launched_at=stamp(),resume=resume))
    print(json.dumps(dict(state='STARTED',session=session,status=str(OUT/'status.json'))),flush=True)


def main():
    bind()
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('test','prepare','start','status','resume','supervise','resume-controller','worker','compute'))
    p.add_argument('--job',choices=JOBS); args = p.parse_args()
    def interrupt(signum,frame):
        raise InterruptedError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    if args.action=='test':
        from experiment.phase22.test_repeat_split import tests
        print(json.dumps(dict(core=tests(),runtime=runtime_tests())),flush=True)
    elif args.action=='prepare':
        prepare()
    elif args.action in ('start','resume'):
        start(args.action=='resume')
    elif args.action=='status':
        print((OUT/'status.json').read_text())
    elif args.action=='worker':
        sys.exit(worker(args.job))
    elif args.action=='compute':
        gen = Generation(args.job)
        if JOBS[args.job]['kind']=='fit':
            gen.fit()
        else:
            gen.generate()
    else:
        try:
            supervise(args.action=='resume-controller')
        except BaseException as error:
            (OUT/f'controller_failure_{os.getpid()}.txt').write_text(traceback.format_exc())
            status(OUT,'FAILED',error=repr(error))
            raise


if __name__=='__main__':
    main()
