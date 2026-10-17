"""Short, isolated CPU/GPU and matched GRAM replay benchmark.

No training, candidate selection, test access, or changes to active workers.
Only the benchmark child processes change their own CPU affinity. GPU probes
briefly share resources with running experiments; outputs are diagnostic.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/old10_expert_selection_20261016_v1'
OUT = ROOT/'artifacts/phase21/runtime_resource_benchmark_20261017_v1'
MODULE = 'experiment.phase21.benchmark_runtime_resources'
DEVICES = [(2,'GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059'),
           (7,'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253'),
           (4,'GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09'),
           (2,'GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059')]


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def optional(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def gpu_state():
    fields = 'index,uuid,utilization.gpu,memory.free,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active'
    return subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()


def numbers(path):
    text = optional(path)
    return {line.split()[0].rstrip(':'):int(line.split()[1]) for line in text.splitlines()
            if len(line.split())>1 and line.split()[1].isdigit()} if text else {}


def task_snapshot(pid):
    p = Path('/proc')/str(pid)
    command = optional(p/'cmdline')
    if not command:return dict(available=False)
    assert 'experiment.phase21.old10_expert_selection' in command
    status = optional(p/'status')
    data = (p/'stat').read_text().rsplit(')',1)[1].split()
    sched = [int(x) for x in (p/'schedstat').read_text().split()]
    cpu = int(data[36])
    cgroups = optional(p/'cgroup')
    limits = {}
    for line in (cgroups or '').splitlines():
        _, controllers, relative = line.split(':',2)
        if not controllers:
            candidates = [Path('/sys/fs/cgroup')/relative.lstrip('/')]
        elif 'cpu' in controllers.split(','):
            candidates = [Path('/sys/fs/cgroup')/name/relative.lstrip('/') for name in ('cpu,cpuacct','cpu')]
        else:continue
        for base in candidates:
            for directory in [base,*base.parents]:
                if directory == Path('/sys/fs/cgroup').parent:break
                found = {name:optional(directory/name) for name in
                         ('cpu.max','cpu.stat','cpu.cfs_quota_us','cpu.cfs_period_us','cpuset.cpus.effective')}
                found = {k:v for k,v in found.items() if v is not None}
                if found:limits[str(directory)] = found
    prefix = Path('/sys/devices/system/cpu')/f'cpu{cpu}'
    frequency = {name:optional(prefix/'cpufreq'/name) for name in
                 ('scaling_cur_freq','scaling_max_freq','cpuinfo_max_freq','scaling_governor')}
    return dict(available=True,monotonic=time.monotonic(),cpu=cpu,
        cpu_runtime_ns=sched[0],runqueue_wait_ns=sched[1],timeslices=sched[2],
        minor_faults=int(data[7]),major_faults=int(data[9]),
        io=numbers(p/'io'),status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines()
            if line.split(':',1)[0] in ('State','Threads','VmRSS','VmSwap','Cpus_allowed_list','voluntary_ctxt_switches','nonvoluntary_ctxt_switches')},
        cgroup=cgroups,cpu_limits=limits,frequency=frequency,
        thermal={name:optional(prefix/'thermal_throttle'/name) for name in ('core_throttle_count','package_throttle_count')})


def audit():
    before = {d:task_snapshot(read(RUN/d/'status.json')['pid']) for d in ('Toys','Beauty')}
    time.sleep(8)
    after = {d:task_snapshot(read(RUN/d/'status.json')['pid']) for d in ('Toys','Beauty')}
    derived = {}
    for d,a in before.items():
        b=after[d]
        if not a['available'] or not b['available']:
            derived[d]=dict(available=False);continue
        seconds=b['monotonic']-a['monotonic']
        derived[d]=dict(window_seconds=seconds,
            main_thread_runtime_seconds=(b['cpu_runtime_ns']-a['cpu_runtime_ns'])/1e9,
            main_thread_scheduler_wait_seconds=(b['runqueue_wait_ns']-a['runqueue_wait_ns'])/1e9,
            main_thread_scheduler_wait_percent=100*(b['runqueue_wait_ns']-a['runqueue_wait_ns'])/1e9/seconds,
            major_faults=b['major_faults']-a['major_faults'],
            io_delta={k:b['io'].get(k,0)-a['io'].get(k,0) for k in b['io']})
    result=dict(before=before,after=after,derived=derived,
        cpu_pressure=optional('/proc/pressure/cpu'),io_pressure=optional('/proc/pressure/io'),
        memory_pressure=optional('/proc/pressure/memory'),intel_pstate_no_turbo=optional('/sys/devices/system/cpu/intel_pstate/no_turbo'))
    write(OUT/'task_resource_audit.json',result)
    print('task_resource_audit',json.dumps(derived),flush=True)


def cpu_times():
    values={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields[0].startswith('cpu') and fields[0][3:].isdigit():
            v=list(map(int,fields[1:]));values[int(fields[0][3:])]=(sum(v[:8]),v[3]+v[4])
    return values


def choose_cpus():
    allowed=os.sched_getaffinity(0);a=cpu_times();time.sleep(.5);b=cpu_times()
    order=sorted(allowed,key=lambda c:1-(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]))
    selected=[];used=set()
    for cpu in order:
        siblings=optional(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list') or str(cpu)
        if siblings not in used:
            selected.append(cpu);used.add(siblings)
        if len(selected)==2:break
    return selected


def cpu_benchmark(cpus):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    previous=os.sched_getaffinity(0);results=[]
    torch.manual_seed(7)
    a=torch.randn(1024,1024);b=torch.randn_like(a);out=torch.empty_like(a)
    def loop():
        value=7
        for _ in range(2000000):value=(1664525*value+1013904223)&0xffffffff
        return value
    for cpu in cpus:
        os.sched_setaffinity(0,{cpu});loop()
        py=[];wall=[];process=[]
        for _ in range(5):
            t=time.perf_counter();loop();py.append(time.perf_counter()-t)
        for _ in range(3):torch.mm(a,b,out=out)
        for _ in range(5):
            t,p=time.perf_counter(),time.process_time()
            for _ in range(10):torch.mm(a,b,out=out)
            wall.append(time.perf_counter()-t);process.append(time.process_time()-p)
        results.append(dict(cpu=cpu,python_2m_loop_seconds=py,matrix10_wall_seconds=wall,
            matrix10_process_seconds=process,matrix_single_thread_GFLOPs=20*1024**3/np.median(wall)/1e9,
            frequency={k:optional(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/{k}') for k in
                       ('scaling_cur_freq','scaling_max_freq','cpuinfo_max_freq','scaling_governor')}))
    os.sched_setaffinity(0,previous)
    write(OUT/'cpu_benchmark.json',dict(results=results,threads=1,torch_version=torch.__version__,healthy_historical_baseline_available=False))
    print('cpu_benchmark',json.dumps(results),flush=True)


def gpu_benchmark(folder):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    results={}
    for label,n,dtype,iterations in [('fp32_4096',4096,torch.float32,10),
                                    ('fp16_4096',4096,torch.float16,10),
                                    ('small_fp32_launch',256,torch.float32,200)]:
        a=torch.randn(n,n,device='cuda',dtype=dtype);b=torch.randn_like(a);out=torch.empty_like(a)
        for _ in range(5):torch.mm(a,b,out=out)
        torch.cuda.synchronize();elapsed=[];wall=[]
        for _ in range(7):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();begin.record()
            for _ in range(iterations):torch.mm(a,b,out=out)
            end.record();end.synchronize()
            wall.append(time.perf_counter()-start);elapsed.append(begin.elapsed_time(end)/1000)
        results[label]=dict(iterations=iterations,gpu_seconds=elapsed,wall_seconds=wall,
            effective_TFLOPs=2*n**3*iterations/np.median(elapsed)/1e12)
        del a,b,out
    torch.cuda.empty_cache()
    results['name']=torch.cuda.get_device_name(0)
    write(folder/'gpu_microbench.json',results)
    print('gpu_microbench',folder.name,json.dumps(results),flush=True)


def replay(folder):
    import numpy as np
    import torch
    from experiment.phase20 import lime_clean_benefit as clean
    from experiment.phase18.core.diff_data import DiffDataset
    clean.OUT=folder
    class Probe(clean.Worker):
        def status(self,state='RUNNING',**kw):
            write(folder/'gram_probe_status.json',dict(state=state,stage=self.stage,**kw))
        def context(self,j,override_index=None):return super().context(j,override_index=j)
    worker=Probe('Beauty')
    expected=read(folder/'Beauty/expected.json')
    worker.samples=[dict(s,target=1) for s in worker.samples]
    worker.data=DiffDataset(worker.samples,worker.catalog)
    scores=np.load(folder/'Beauty/scores.npy');alphas=np.load(folder/'Beauty/alpha.npy')
    checks=[];timings=[]
    for j in range(len(worker.samples)):
        ctx=worker.context(j);q=worker.index.conditional_logs(scores[j])
        for key,a,expert in [('native',0.,None),('guided',float(alphas[j]),q)]:
            out=worker.decode(ctx,expert,a)
            actual=dict(zip(out['ids'],out['scores']));reference=expected[j][key]
            assert set(actual)==set(reference['ids'])
            np.testing.assert_allclose([actual[i] for i in reference['ids']],reference['scores'],atol=3e-6,rtol=2e-5)
        checks.append(dict(row=j,same_native_and_guided_predictions=True))
        for repetition in range(5):
            torch.cuda.synchronize();start=time.perf_counter();ctx=worker.context(j);torch.cuda.synchronize()
            encoder=time.perf_counter()-start
            start=time.perf_counter();q=worker.index.conditional_logs(scores[j]);expert_cpu=time.perf_counter()-start
            start=time.perf_counter();worker.decode(ctx);native=time.perf_counter()-start
            start=time.perf_counter();worker.decode(ctx,q,float(alphas[j]));guided=time.perf_counter()-start
            timings.append(dict(row=j,repetition=repetition,encoder_seconds=encoder,
                expert_cpu_seconds=expert_cpu,native_seconds=native,guided_seconds=guided))
    assert worker.digest()==worker.initial_digest
    result=dict(checks=checks,timings=timings,model_digest=worker.initial_digest,
        medians={k:float(np.median([x[k] for x in timings])) for k in
                 ('encoder_seconds','expert_cpu_seconds','native_seconds','guided_seconds')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),test_read=False,training_steps=0)
    write(folder/'gram_replay.json',result)
    print('gram_replay',folder.name,json.dumps(result['medians']),flush=True)


def device(label,cpus):
    os.sched_setaffinity(0,set(cpus))
    folder=OUT/label
    write(folder/'gpu_before.json',dict(csv=gpu_state()))
    gpu_benchmark(folder)
    replay(folder)
    write(folder/'gpu_after.json',dict(csv=gpu_state()))


def run():
    import numpy as np
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json',dict(state='RUNNING',stage='resource_audit',test_read=False))
    write(OUT/'gpu_initial.json',dict(csv=gpu_state()))
    audit();cpus=choose_cpus();cpu_benchmark(cpus)
    samples=read(RUN/'Beauty/samples.json');refs=read(RUN/'Beauty/references.json')
    indices=[512,576,640];expected={}
    with (RUN/'Beauty/predictions.jsonl').open() as stream:
        for j,line in enumerate(stream):
            if j in indices:
                row=json.loads(line)
                assert row['user_id']==samples[j]['user_id']
                expected[j]=dict(native=refs[j]['native'],guided=row['SASRec__BGE'])
            if j>=max(indices):break
    assert len(expected)==3
    alpha=np.load(RUN/'Beauty/alphas.npz')['SASRec__BGE'][indices]
    scores=np.load(RUN/'Beauty/signals/SASRec__BGE.npy',mmap_mode='r')[indices]
    result={}
    for sequence,(index,uuid) in enumerate(DEVICES):
        label=f'pass{sequence}_gpu{index}';folder=OUT/label
        (folder/'Beauty').mkdir(parents=True)
        write(folder/'config.json',dict(worker_timeout_seconds=180,test_read=False,diagnostic_only=True))
        write(folder/'Beauty/samples.json',[dict(samples[j],validation_index=k) for k,j in enumerate(indices)])
        write(folder/'Beauty/expected.json',[expected[j] for j in indices])
        for name in ('data_config.json','catalog.json'):shutil.copy2(RUN/'Beauty'/name,folder/'Beauty'/name)
        np.save(folder/'Beauty/alpha.npy',alpha);np.save(folder/'Beauty/scores.npy',scores)
        write(OUT/'status.json',dict(state='RUNNING',stage=label,cpu_affinity=cpus,test_read=False))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (folder/'worker.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m',MODULE,'device','--label',label,'--cpus',','.join(map(str,cpus))],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        result[label]=dict(gpu_index=index,gpu_uuid=uuid,microbench=read(folder/'gpu_microbench.json'),replay=read(folder/'gram_replay.json'))
        write(OUT/'progress.json',result)
        print('device_completed',label,result[label]['replay']['medians'],flush=True)
    write(OUT/'summary.json',dict(devices=result,cpu=read(OUT/'cpu_benchmark.json'),
        task_audit=read(OUT/'task_resource_audit.json'),test_read=False,training_steps=0,
        limits='Concurrent shared machine; no healthy same-machine historical benchmark. Device comparisons are sequential and observational. Benchmark briefly competed with active workers.'))
    write(OUT/'status.json',dict(state='COMPLETED',test_read=False))
    print('COMPLETED',str(OUT),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','device'))
    parser.add_argument('--label');parser.add_argument('--cpus');args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:device(args.label,list(map(int,args.cpus.split(','))))
    except BaseException as error:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(error)))
        raise
