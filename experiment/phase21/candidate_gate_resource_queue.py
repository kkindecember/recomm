"""Bounded GPU resource queue; a device change never changes model/sample settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/candidate_gate_generation_resume_20261014_v1'
GPUS=['GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc','GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253']


def worker(domain,gpu):
    from experiment.phase21 import resume_candidate_gate_generation as R
    G=R.G;G.OUT=OUT;G.Worker=R.ResumeWorker
    original_read=G.read
    def resource_read(path):
        value=original_read(path)
        if Path(path)==OUT/'config.json':
            value['gpus'][domain]=gpu
        return value
    G.read=resource_read
    G.worker(domain)


def queue(domain):
    from experiment.phase21.candidate_gate_screen import write,sha
    start=time.monotonic();folder=OUT/domain
    assert not (folder/'lease.log').exists(), 'A previous GPU worker already initialized; audit before retry'
    while time.monotonic()-start<21600:
        free=[]
        for gpu in GPUS:
            value=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(value.strip())>=32768:free.append((int(value.strip()),gpu))
        if free:
            _,gpu=max(free)
            write(folder/'resource_amendment.json',dict(reason='original GPU failed admission before model or lease initialization',device=gpu,scientific_settings_unchanged=True,wait_seconds=time.monotonic()-start,runner_sha256=sha(Path(__file__))))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
            with (folder/'queued_worker.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','experiment.phase21.candidate_gate_resource_queue','worker',domain,gpu],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write(folder/'queue_status.json',dict(state='RUNNING',pid=process.pid,gpu=gpu))
                try:code=process.wait(timeout=21900)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    code=124
                write(folder/'queue_status.json',dict(state='COMPLETED' if code==0 else 'FAILED',exit_code=code))
                if code:raise RuntimeError('Queued worker exit '+str(code))
            # The other fixed-cohort worker may still be active.
            from experiment.phase21 import candidate_gate_generation as G
            G.OUT=OUT
            while time.monotonic()-start<43200:
                states={d:json.loads((OUT/d/'status.json').read_text())['state'] for d in ('Toys','Beauty')}
                if all(v=='COMPLETED' for v in states.values()):G.summarize();return
                if any(v=='FAILED' for v in states.values()):raise RuntimeError(states)
                time.sleep(10)
            raise TimeoutError('Waiting for paired summary')
        write(folder/'queue_status.json',dict(state='WAITING_FOR_RESOURCES',elapsed_seconds=time.monotonic()-start,minimum_free_mib=32768))
        time.sleep(15)
    raise TimeoutError('GPU queue time budget')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['queue','worker']);p.add_argument('domain');p.add_argument('gpu',nargs='?');a=p.parse_args()
    if a.mode=='queue':queue(a.domain)
    else:worker(a.domain,a.gpu)
