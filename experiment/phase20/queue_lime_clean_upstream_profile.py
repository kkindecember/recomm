"""Wait at most ten minutes for the fixed GPU, then one bounded profile attempt."""
import os
import signal
import subprocess
import sys
import time

from experiment.phase20.guided_decoding_job import ROOT,read,write,sha,now


def main():
    config=ROOT/'experiment/phase20/lime_clean_upstream_toys_v1.json';c=read(config)
    root=ROOT/c['output'];out=root/'gpu_queue';out.mkdir(exist_ok=False)
    assert read(root/'cpu_smoke/summary.json')['state']=='COMPLETED'
    assert not (root/'profile').exists()
    initial_config=sha(config);start=time.monotonic();child=None
    def status(state,**extra):
        value=dict(state=state,pid=os.getpid(),updated_at=now(),elapsed_seconds=time.monotonic()-start,
            gpu_uuid=c['gpu_uuid'],minimum_free_mib=c['minimum_free_mib'],max_wait_seconds=600,
            formal_training_started=False,**extra)
        write(out/'status.json',value);print(value,flush=True)
    with (out/'run.log').open('x',buffering=1) as log:
        for attempt in range(21):
            r=subprocess.run(['nvidia-smi','--id='+c['gpu_uuid'],'--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True)
            free=int(r.stdout.split(',')[0]) if r.returncode==0 else -1
            if free>=c['minimum_free_mib']:break
            status('WAITING_FOR_GPU',free_mib=free,attempt=attempt)
            if time.monotonic()-start>=600:
                status('RESOURCE_WAIT_EXPIRED',free_mib=free);return
            time.sleep(min(30,600-(time.monotonic()-start)))
        assert sha(config)==initial_config
        command=['bash','experiment/phase20/run_lime_clean_upstream.sh','profile']
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('PROFILING',child_pid=child.pid,launch_free_mib=free,command=command)
        try:rc=child.wait(timeout=c['profile_timeout_seconds']+30)
        except subprocess.TimeoutExpired:
            status('HARD_TIMEOUT',child_pid=child.pid)
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            return
        state=read(root/'profile/status.json') if (root/'profile/status.json').exists() else {}
        status('COMPLETED' if rc==0 and state.get('state')=='COMPLETED' else 'FAILED',
            child_pid=child.pid,exit_code=rc,profile_state=state.get('state'))


if __name__=='__main__':main()
