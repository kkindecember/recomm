"""Matched replay after briefly suspending only our own generator launches.

Load the diagnostic model first, then SIGSTOP the verified original generator.
SIGCONT is guaranteed by finally and an independent 90-second watchdog. The
lease and supervisor keep running. No original code, parameters or files change.
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

from experiment.phase21 import benchmark_runtime_resources as B

OUT = B.ROOT/'artifacts/phase21/runtime_resource_isolation_20261017_v1'
MODULE = 'experiment.phase21.benchmark_runtime_isolation'
CASES = [('gpu7','Toys',7,'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253','pass1_gpu7'),
         ('gpu4','Beauty',4,'GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09','pass2_gpu4')]


def device(label,domain,cpus):
    os.sched_setaffinity(0,set(cpus))
    folder = OUT/label
    from experiment.phase20 import lime_clean_benefit as clean
    Original = clean.Worker
    original_status = B.read(B.RUN/domain/'status.json')
    pid = original_status['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
    assert 'experiment.phase21.old10_expert_selection generate' in command
    assert '--domain '+domain in command
    assert original_status['gpu_uuid'] == os.environ['CUDA_VISIBLE_DEVICES']
    paused = False; watchdog = None; paused_at = None
    class LoadedPause(Original):
        def __init__(self,*args,**kwargs):
            nonlocal paused, watchdog, paused_at
            super().__init__(*args,**kwargs)
            # The independent process restores scheduling even if this child dies.
            program = 'import os,signal,time;time.sleep(90);os.kill('+str(pid)+',signal.SIGCONT)'
            watchdog = subprocess.Popen([sys.executable,'-c',program],start_new_session=True,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            os.kill(pid,signal.SIGSTOP); paused = True; paused_at = time.monotonic()
            time.sleep(.3)
            B.write(folder/'pause.json',dict(state='PAUSED',generator_pid=pid,watchdog_pid=watchdog.pid))
    clean.Worker = LoadedPause
    try:
        B.replay(folder)
    finally:
        clean.Worker = Original
        if paused:
            os.kill(pid,signal.SIGCONT)
            B.write(folder/'pause.json',dict(state='RESUMED',generator_pid=pid,
                paused_seconds=time.monotonic()-paused_at,watchdog_pid=watchdog.pid))
        if watchdog is not None:
            watchdog.terminate(); watchdog.wait(timeout=5)


def run():
    OUT.mkdir(exist_ok=False)
    B.write(OUT/'status.json',dict(state='RUNNING',test_read=False))
    cpus = B.read(B.OUT/'cpu_benchmark.json')['results']
    cpus = [row['cpu'] for row in cpus]
    results = {}
    for label,domain,index,uuid,source_label in CASES:
        folder = OUT/label
        (folder/'Beauty').mkdir(parents=True)
        source = B.OUT/source_label
        shutil.copy2(source/'config.json',folder/'config.json')
        for path in (source/'Beauty').iterdir():
            if path.is_file():shutil.copy2(path,folder/'Beauty'/path.name)
        B.write(OUT/'status.json',dict(state='RUNNING',stage=label,test_read=False))
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OPENBLAS_NUM_THREADS='2',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        with (folder/'worker.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m',MODULE,'device','--label',label,
                '--domain',domain,'--cpus',','.join(map(str,cpus))],cwd=B.ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        result = B.read(folder/'gram_replay.json')
        result.update(gpu_index=index,gpu_uuid=uuid,pause=B.read(folder/'pause.json'))
        results[label] = result
        B.write(OUT/'progress.json',results)
        print(label,json.dumps(dict(medians=result['medians'],pause=result['pause'])),flush=True)
    B.write(OUT/'summary.json',dict(cases=results,test_read=False,training_steps=0,
        caveat='Other users continue running. Only our own generator is briefly suspended; this is not exclusive hardware or a healthy historical baseline.'))
    B.write(OUT/'status.json',dict(state='COMPLETED',test_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','device'))
    parser.add_argument('--label');parser.add_argument('--domain');parser.add_argument('--cpus');args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:device(args.label,args.domain,list(map(int,args.cpus.split(','))))
    except BaseException as error:
        if OUT.exists():B.write(OUT/'failure.json',dict(error=repr(error)))
        raise
