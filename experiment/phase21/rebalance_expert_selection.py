"""Live supervisor handoff: widen only GPU resources; preserve running workers.

The old supervisor is stopped (not its workers) until its children finish.
Scientific sources/configuration remain immutable. New workers use the original
entry point and original manifest. This supervisor owns the root status file.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase21 import expert_selection as E

OUT=E.OUT
OLD_SUPERVISOR=2607134


def proc(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text()
        fields=raw[raw.rfind(')')+2:].split()
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return dict(state=fields[0],start=fields[19],exit_code=int(fields[49]),args=[x.decode() for x in args if x])
    except (FileNotFoundError,ProcessLookupError):return None


def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in output.splitlines():
        ix,gpu,free,util=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(ix),gpu=gpu,free=int(free),utilization=int(util)))
    return rows


def stop_old_supervisor(identity):
    p=proc(OLD_SUPERVISOR)
    if p and p['start']==identity['start'] and p['state']!='Z':
        # Workers have exited; resuming only delivers the pending default TERM.
        os.kill(OLD_SUPERVISOR,signal.SIGTERM)
        os.kill(OLD_SUPERVISOR,signal.SIGCONT)


def main():
    cfg=E.read(OUT/'config.json')
    assert not (OUT/'resource_override.json').exists(),'A resource handoff already exists'
    inventory=gpu_inventory()
    assert len(inventory)==8
    old=proc(OLD_SUPERVISOR)
    assert old and old['state']!='Z'
    assert old['args'][-3:]==['-m','experiment.phase21.expert_selection','supervise']
    assert Path(f'/proc/{OLD_SUPERVISOR}').stat().st_uid==os.getuid()
    # Fail before stopping the old supervisor if any frozen input changed.
    for rel,digest in E.read(OUT/'manifest.json')['sha256'].items():assert E.sha(E.ROOT/rel)==digest,rel
    os.kill(OLD_SUPERVISOR,signal.SIGSTOP)
    active={};finished={};pending=[];ownership_acquired=False
    try:
        for _ in range(100):
            if proc(OLD_SUPERVISOR)['state'] in ('T','t'):break
            time.sleep(.01)
        else:raise RuntimeError('Old supervisor did not stop')
        snapshot=E.read(OUT/'status.json')
        assert snapshot['state']=='RUNNING' and snapshot['pid']==OLD_SUPERVISOR
        child_ids=[int(x) for x in Path(f'/proc/{OLD_SUPERVISOR}/task/{OLD_SUPERVISOR}/children').read_text().split()]
        live_children=[]
        for pid in child_ids:
            p=proc(pid)
            if p and p['state']!='Z':live_children.append(pid)
        expected_pids=[v['pid'] for v in snapshot['active'].values() if proc(v['pid']) and proc(v['pid'])['state']!='Z']
        assert set(live_children)==set(expected_pids),(live_children,expected_pids)
        for arm in E.ARMS:
            for domain in ('Toys','Beauty'):
                key=arm+'/'+domain;folder=OUT/arm/domain;state=E.read(folder/'status.json')
                live=proc(snapshot['active'][key]['pid']) if key in snapshot['active'] else None
                if key in snapshot['active'] and live and live['state']!='Z':
                    a=snapshot['active'][key];p=proc(a['pid'])
                    assert p and p['args'][-5:]==['-m','experiment.phase21.expert_selection','worker',arm,domain]
                    active[key]=dict(pid=a['pid'],identity=p['start'],gpu=a['gpu'],arm=arm,domain=domain,
                        inherited=True,start=time.monotonic()-state.get('elapsed_seconds',0))
                elif state['state']=='COMPLETED':finished[key]=dict(state='COMPLETED',exit_code=0)
                else:
                    assert state['state']=='PREPARED',(key,state)
                    assert not (folder/'worker.log').exists(),key
                    pending.append((arm,domain))
        E.write(OUT/'resource_override.json',dict(created_at=E.stamp(),reason='User observed available GPUs outside original pool',
            old_supervisor_pid=OLD_SUPERVISOR,new_supervisor_pid=os.getpid(),gpu_pool=[r['gpu'] for r in inventory],
            max_parallel=cfg['max_parallel'],minimum_free_mib=cfg['minimum_free_mib'],
            scientific_settings_unchanged=True,running_workers_preserved=[j['pid'] for j in active.values()],
            runner_sha256=E.sha(Path(__file__)),original_manifest_sha256=E.sha(OUT/'manifest.json')))
        ownership_acquired=True
        start=time.monotonic();old_stopped=True
        while pending or active:
            for key,job in list(active.items()):
                if job['inherited']:
                    p=proc(job['pid'])
                    code=None if p and p['start']==job['identity'] and p['state']!='Z' else (
                        os.waitstatus_to_exitcode(p['exit_code']) if p and p['state']=='Z' else -1)
                else:code=job['process'].poll()
                if code is None and time.monotonic()-job['start']>21900:
                    os.kill(job['pid'],signal.SIGTERM)
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        p=proc(job['pid'])
                        if p is None or p['state']=='Z':break
                        time.sleep(.5)
                    else:os.kill(job['pid'],signal.SIGKILL)
                    code=124
                    E.write(OUT/job['arm']/job['domain']/'status.json',dict(state='RESOURCE_TIME_LIMIT',updated_at=E.stamp()))
                if code is not None:
                    state=E.read(OUT/job['arm']/job['domain']/'status.json')
                    finished[key]=dict(exit_code=code,state=state['state'])
                    if not job['inherited']:job['log'].close()
                    del active[key]
            if old_stopped and not any(j['inherited'] for j in active.values()):
                stop_old_supervisor(old);old_stopped=False
            if pending and time.monotonic()-start>cfg['supervisor_timeout_seconds']:
                for a,d in pending:
                    E.write(OUT/a/d/'status.json',dict(state='RESOURCE_WAIT_LIMIT',updated_at=E.stamp()))
                    finished[a+'/'+d]=dict(state='RESOURCE_WAIT_LIMIT',exit_code=124)
                pending=[]
            inventory=gpu_inventory()
            occupied={j['gpu'] for j in active.values()}
            for gpu in sorted(inventory,key=lambda r:(-r['free'],r['utilization'],r['index'])):
                if not pending or len(active)>=cfg['max_parallel']:break
                if gpu['gpu'] in occupied or gpu['free']<cfg['minimum_free_mib']:continue
                arm,domain=pending.pop(0);folder=OUT/arm/domain
                log=(folder/'worker.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['gpu'])
                process=subprocess.Popen([sys.executable,'-u','-m','experiment.phase21.expert_selection','worker',arm,domain],
                    stdout=log,stderr=subprocess.STDOUT,env=env,cwd=E.ROOT,start_new_session=True)
                active[arm+'/'+domain]=dict(process=process,pid=process.pid,inherited=False,start=time.monotonic(),
                    gpu=gpu['gpu'],arm=arm,domain=domain,log=log)
                E.write(folder/'status.json',dict(state='STARTING',pid=process.pid,gpu=gpu['gpu'],gpu_index=gpu['index'],updated_at=E.stamp()))
            jobs={k:dict(pid=j['pid'],gpu=j['gpu'],inherited=j['inherited'],worker=E.read(OUT/j['arm']/j['domain']/'status.json')) for k,j in active.items()}
            E.write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES',scope='expert_selection_only',
                pid=os.getpid(),updated_at=E.stamp(),active=jobs,pending=[a+'/'+d for a,d in pending],finished=finished,
                gpu_pool_indices=[r['index'] for r in inventory],resource_override='resource_override.json',test_read=False))
            if pending or active:time.sleep(10)
        if old_stopped:stop_old_supervisor(old)
        if all(x['state']=='COMPLETED' and x.get('exit_code')==0 for x in finished.values()):
            E.summarize();state='COMPLETED'
        else:state='INCOMPLETE'
        E.write(OUT/'status.json',dict(state=state,scope='expert_selection_only',updated_at=E.stamp(),finished=finished,
            resource_override='resource_override.json',test_read=False,candidate_weights_scheduled=False))
    except BaseException:
        if not ownership_acquired:
            os.kill(OLD_SUPERVISOR,signal.SIGCONT)
        else:
            E.write(OUT/'status.json',dict(state='SUPERVISOR_FAILED',traceback=traceback.format_exc(),updated_at=E.stamp(),
                active_pids=[j['pid'] for j in active.values()],old_supervisor_stopped=bool(proc(OLD_SUPERVISOR) and proc(OLD_SUPERVISOR)['state'] in ('T','t'))))
        raise


if __name__=='__main__':main()
