"""User-authorized 16 GiB leases and lossless continuation of joint fusion.

The frozen scientific sources/configuration and completed training are reused.
This separate operational runner changes GPU admission, lease size, and permits
appending after verified complete prediction rows. It does not select results.
"""
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
from experiment.phase21 import joint_fusion as J

ROOT, OUT = J.ROOT, J.OUT
MODULE = 'experiment.phase21.joint_fusion_resource_resume'
RESOURCE = dict(total_lease_mib=16384, expected_workload_peak_mib=12288,
    sidecar_reserved_mib=4096, minimum_free_mib=18432, max_parallel=2,
    allocation_limit_gib=11, worker_timeout_seconds=21600,
    user_authorization='2026-10-15: 30g太多了是不是 你可以少放点呀',
    override_scope='this joint_fusion_20261015_v2 run; scientific configuration unchanged')
read, write, sha, stamp = J.read, J.write, J.sha, J.stamp


def completed_prefix(raw):
    """Only an incomplete final line may be discarded; corrupt full rows fail."""
    parts = raw.splitlines(keepends=True)
    rows, prefix = [], bytearray()
    for j, line in enumerate(parts):
        if not line.endswith(b'\n'):
            assert j == len(parts)-1
            return rows, bytes(prefix), line
        rows.append(json.loads(line))
        prefix.extend(line)
    return rows, bytes(prefix), b''


def tests():
    raw = b'{"user_id":"a"}\n{"user_id":"b"}\n'
    rows, prefix, tail = completed_prefix(raw+b'{"user_id":')
    assert len(rows)==2 and prefix==raw and tail==b'{"user_id":'
    assert completed_prefix(raw)==(rows,raw,b'')
    assert completed_prefix(b'')==([],b'',b'')
    try:
        completed_prefix(b'bad complete row\n')
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError('Corrupt complete record accepted')
    assert RESOURCE['total_lease_mib']-RESOURCE['expected_workload_peak_mib']==RESOURCE['sidecar_reserved_mib']
    assert RESOURCE['minimum_free_mib']>RESOURCE['total_lease_mib']
    return dict(complete_prefix_preserved=True, partial_tail_only_recovery=True,
                corrupt_complete_record_rejected=True, resource_budget_consistent=True)


def prepare():
    # Run only after the previous supervisor and workers have stopped.
    assert all((OUT/d/'training_manifest.json').exists() for d in J.DOMAINS), 'Finish existing training before switching'
    dest=OUT/'resource_resume'; dest.mkdir(exist_ok=False)
    for p in (Path(__file__),ROOT/'experiment/phase21/run_joint_fusion_resource_resume.sh'):
        shutil.copy2(p,dest/p.name)
    write(dest/'config.json',RESOURCE);write(dest/'tests.json',tests())
    before={}
    for d in J.DOMAINS:
        folder=OUT/d; archive=dest/d;archive.mkdir()
        for name in ('status.json','lease_status.json','lease_process.json','runtime_checks.json','smoke.json'):
            if (folder/name).exists(): shutil.copy2(folder/name,archive/name)
        source=folder/'predictions.jsonl'
        raw=source.read_bytes() if source.exists() else b''
        rows,prefix,tail=completed_prefix(raw)
        if source.exists(): shutil.copy2(source,archive/'predictions_before_resume.jsonl')
        if tail:
            with source.open('wb') as handle:
                handle.write(prefix);handle.flush();os.fsync(handle.fileno())
        samples=read(folder/'samples.json')
        assert len(rows)<=len(samples)==512
        keys=('native',)+tuple(a+s for a in J.ARMS for s in ('_late','_guided'))
        refs={r['user_id']:r for r in read(folder/'references.json')}
        for row,sample in zip(rows,samples):
            assert all(row[k]==sample[k] for k in ('user_id','target','history'))
            for key in keys:
                J.S.OLD.record_rank(row[key],sample)
                if key in refs[row['user_id']]:
                    ref=refs[row['user_id']][key]
                    assert row[key]['ids']==ref['ids']
                    np.testing.assert_allclose(row[key]['scores'],ref['scores'],atol=2e-6,rtol=1e-5)
        old_lease=read(archive/'lease_process.json')
        before[d]=dict(completed_users=len(rows), preserved_prefix_bytes=len(prefix),
            preserved_prefix_sha256=hashlib.sha256(prefix).hexdigest(),discarded_partial_tail_bytes=len(tail),
            old_lease=old_lease,training_manifest_sha256=sha(folder/'training_manifest.json'))
        write(folder/'status.json',dict(state='PREPARED',stage='resource_resume',users_completed=len(rows),
            users_total=512,updated_at=stamp(),effective_resource=RESOURCE))
    write(dest/'snapshot.json',before)
    shutil.copy2(OUT/'status.json',dest/'root_status_before_resume.json')
    write(dest/'manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),ROOT/'experiment/phase21/run_joint_fusion_resource_resume.sh',dest/'config.json',dest/'snapshot.json')}))
    write(OUT/'status.json',dict(state='PREPARED',stage='resource_resume',updated_at=stamp(),effective_resource=RESOURCE))
    (dest/'authorization.md').write_text('# 本轮显存预留调整\n\n用户明确要求降低30 GiB预留，覆盖本轮默认租约规则。'
        '总预算改为16 GiB，其中工作峰值预算仍为12 GiB、预占4 GiB，启动要求18 GiB。'
        '模型的11 GiB分配上限、训练参数、数据、权重、评分与解码规则不变。'
        '复用两域已完成训练，核验已有完整预测后从断点追加；不按结果挑选或丢弃用户。'
        '历史config.json保留原始记录，实际资源设置见resource_resume/config.json及根status.json。\n')
    print(json.dumps(before),flush=True)


class Generation(J.Generation):
    def run(self):
        before=read(OUT/'resource_resume/snapshot.json')[self.domain]
        source=self.folder/'predictions.jsonl'
        raw=source.read_bytes() if source.exists() else b''
        rows,prefix,tail=completed_prefix(raw)
        assert not tail and len(rows)==before['completed_users']
        assert hashlib.sha256(prefix).hexdigest()==before['preserved_prefix_sha256']
        self.users_completed=len(rows)
        self.alpha=dict(np.load(self.folder/'alphas.npz'))
        assert self.initial_digest==read(self.folder/'teacher_checks.json')['model_digest']

        def prediction(j):
            sample=self.samples[j]
            ctx=self.context(j);native=self.decode(ctx)
            ref=self.reference[sample['user_id']]
            assert native['ids']==ref['native']['ids']
            np.testing.assert_allclose(native['scores'],ref['native']['scores'],atol=2e-6,rtol=1e-5)
            row=dict(user_id=sample['user_id'],target=sample['target'],history=sample['history'],
                native=dict(ids=native['ids'],scores=native['scores'].tolist()))
            for arm in J.ARMS:
                a=float(self.alpha[arm][j])
                q=self.index.conditional_logs(self.signals[arm][j],temperature_scale=20.)
                late=native['guide'].rescore(native['paths'],q,a)
                guided=self.decode(ctx,q,a)
                row[arm+'_late']=dict(ids=native['ids'],scores=late.tolist())
                row[arm+'_guided']=dict(ids=guided['ids'],scores=guided['scores'].tolist())
                for suffix in ('_late','_guided'):
                    key=arm+suffix
                    if key in ref:
                        assert row[key]['ids']==ref[key]['ids'],(j,key)
                        np.testing.assert_allclose(row[key]['scores'],ref[key]['scores'],atol=2e-6,rtol=1e-5)
            return row

        self.stage='resume_decoder_parity';self.status(effective_resource=RESOURCE)
        checks=[]
        for j in (0,256,511):
            ctx=self.context(j);native=self.decode(ctx)
            plain=self.decode(ctx,native_entry=True,record=False)
            q=self.index.conditional_logs(self.signals['bge_fixed'][j])
            zero=self.decode(ctx,q,0.)
            for other in (plain,zero):
                assert native['paths']==other['paths']
                np.testing.assert_array_equal(native['scores'],other['scores'])
            checks.append(dict(user_id=self.samples[j]['user_id'],alpha0_exact=True,native_entry_exact=True))
        if rows:
            replay=prediction(len(rows)-1)
            for key,value in replay.items():
                if isinstance(value,dict):
                    assert value['ids']==rows[-1][key]['ids'],key
                    np.testing.assert_allclose(value['scores'],rows[-1][key]['scores'],atol=2e-6,rtol=1e-5)
                else:
                    assert value==rows[-1][key]
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks,
            resumed_users=len(rows),preserved_prefix_sha256=before['preserved_prefix_sha256'],
            last_saved_row_all_20_arms_replayed=bool(rows)))
        self.stage='matched_generation';self.status(effective_resource=RESOURCE)
        with source.open('a') as stream:
            for j in range(len(rows),len(self.samples)):
                begin=time.monotonic();row=prediction(j)
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                self.users_completed=j+1;self.row_seconds.append(time.monotonic()-begin)
                if j==len(rows) or (j+1)%8==0 or j==511:
                    self.status(arms_per_user=len(J.ARMS),resumed_users=len(rows),effective_resource=RESOURCE)
        assert self.digest()==self.initial_digest
        with source.open('rb') as handle:
            assert hashlib.sha256(handle.read(before['preserved_prefix_bytes'])).hexdigest()==before['preserved_prefix_sha256']
        write(self.folder/'runtime_checks.json',dict(state='PASSED',native_reproduced_users=512,
            four_historical_fixed_arms_reproduced_users=512,gram_model_digest_unchanged=True,
            gram_optimizer_steps=0,all_arms_generated=True,resumed_users=len(rows),preserved_prefix_unchanged=True,
            effective_resource=RESOURCE))
        self.stage='complete';self.status('COMPLETED',effective_resource=RESOURCE)


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()


def worker(domain):
    folder=OUT/domain;lease=None;child=None
    def interrupted(signum,frame): raise InterruptedError('Resource-resume worker interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(RESOURCE['worker_timeout_seconds'])
    try:
        J.progress(domain,'verify_resume_inputs',effective_resource=RESOURCE)
        assert read(OUT/'config.json')==J.CFG
        for name in ('manifest.json','resource_resume/manifest.json',domain+'/training_manifest.json'):
            for relative,digest in read(OUT/name)['sha256'].items():
                assert sha(ROOT/relative)==digest,relative
        before=read(OUT/'resource_resume/snapshot.json')[domain]
        assert sha(folder/'training_manifest.json')==before['training_manifest_sha256']
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        free=next(g['free_mib'] for g in J.S.OLD.gpu_inventory() if g['uuid']==gpu)
        if free<RESOURCE['minimum_free_mib']:
            write(folder/'status.json',dict(state='ADMISSION_DEFERRED',free_mib=free,
                required_free_mib=RESOURCE['minimum_free_mib'],updated_at=stamp()))
            return 75
        with (folder/'lease_resource_resume.log').open('a') as log:
            path=folder/'lease_status.json'
            write(path,dict(state='starting',**RESOURCE))
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib',str(RESOURCE['total_lease_mib']),
                '--expected-workload-peak-mib',str(RESOURCE['expected_workload_peak_mib']),
                '--status-path',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(folder/'lease_process.json',dict(pid=lease.pid,gpu_uuid=gpu,started_at=stamp(),effective_resource=RESOURCE))
            start=time.monotonic()
            while read(path).get('state')!='holding':
                assert lease.poll() is None,'Lease failed'
                assert time.monotonic()-start<180,'Lease startup timed out'
                time.sleep(1)
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'generate','--domain',domain,
                '--lease-pid',str(lease.pid)],cwd=ROOT,start_new_session=True)
            code=child.wait()
            assert code==0,('Generator failed',code)
        return 0
    except BaseException as exc:
        write(folder/'status.json',dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
        raise
    finally:
        signal.alarm(0);stop(child);stop(lease)


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    start=time.monotonic();pending=list(J.DOMAINS);active={};finished={};attempts={d:0 for d in J.DOMAINS}
    previous=read(OUT/'resource_resume/snapshot.json')
    try:
        while pending or active:
            for d,job in list(active.items()):
                code=job['process'].poll()
                if code is None and time.monotonic()-job['started']>RESOURCE['worker_timeout_seconds']+120:
                    stop(job['process']);code=job['process'].returncode
                if code is not None:
                    job['log'].close();state=read(OUT/d/'status.json')
                    if code==75 and state['state']=='ADMISSION_DEFERRED': pending.append(d)
                    else:
                        finished[d]=dict(exit_code=code,state=state['state'],
                            lease_released=read(OUT/d/'lease_status.json').get('state')=='released')
                    del active[d]
            inventory=J.S.OLD.gpu_inventory()
            occupied={v['gpu']['uuid'] for v in active.values()}
            free=sorted([g for g in inventory if g['uuid'] not in occupied and g['free_mib']>=RESOURCE['minimum_free_mib']],
                        key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
            for d in list(pending):
                if not free or len(active)>=RESOURCE['max_parallel']:break
                preferred=previous[d]['old_lease']['gpu_uuid']
                gpu=next((g for g in free if g['uuid']==preferred),free[0]);free.remove(gpu)
                pending.remove(d);attempts[d]+=1
                log=(OUT/d/f'resource_resume_attempt{attempts[d]}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                         OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                write(OUT/d/'status.json',dict(state='STARTING',stage='resource_resume',gpu_index=gpu['index'],
                    gpu_uuid=gpu['uuid'],updated_at=stamp(),users_completed=previous[d]['completed_users']))
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',d],cwd=ROOT,
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[d]=dict(process=process,log=log,gpu=gpu,started=time.monotonic())
            if pending and time.monotonic()-start>J.CFG['resource_wait_seconds']:
                for d in pending:finished[d]=dict(exit_code=124,state='RESOURCE_WAIT_LIMIT',lease_released=True)
                pending=[]
            write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(),updated_at=stamp(),elapsed_seconds_since_resource_resume=time.monotonic()-start,
                pending=pending,finished=finished,effective_resource=RESOURCE,required_free_mib=RESOURCE['minimum_free_mib'],
                active={d:dict(pid=v['process'].pid,gpu_index=v['gpu']['index'],gpu_uuid=v['gpu']['uuid'],
                    worker=read(OUT/d/'status.json')) for d,v in active.items()},gpu_inventory=inventory,test_read=False))
            if pending or active:time.sleep(10)
        complete=len(finished)==2 and all(v['exit_code']==0 and v['state']=='COMPLETED' and v['lease_released'] for v in finished.values())
        if complete:
            J.summarize()
            with J.REPORT.open('a') as stream:
                stream.write('\n运行期间按用户要求将每域显存预算由30 GiB调整为16 GiB；训练产物复用、预测从核验断点续写，评分规则不变。详见resource_resume记录。\n')
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
            updated_at=stamp(),effective_resource=RESOURCE,test_read=False,report=str(J.REPORT.relative_to(ROOT))))
    finally:
        for job in active.values():stop(job['process']);job['log'].close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','supervise','worker','generate'))
    parser.add_argument('--domain',choices=J.DOMAINS);parser.add_argument('--lease-pid',type=int)
    args=parser.parse_args();J.bind()
    def interrupted(signum,frame):raise InterruptedError('Resource-resume interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='test':print(json.dumps(tests()))
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='worker':return worker(args.domain)
        else:
            assert args.domain and args.lease_pid
            lease=J.S.LeaseHandle(args.lease_pid);assert lease.poll() is None
            J.LEASE=lease
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            Generation(args.domain,lease).run()
        return 0
    except BaseException as exc:
        dest=OUT/args.domain if args.domain else OUT
        write(dest/'resource_resume_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
        if args.mode not in ('worker','test'):
            write(dest/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp()))
        raise


if __name__=='__main__':sys.exit(main())
