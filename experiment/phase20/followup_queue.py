"""Launch authorized GACR evaluation after smoke and resource admission."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/followup_queue'
child = None
stopping = False


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save(state, **values):
    row = {'state': state, 'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'automatic_retry': False, 'test_read': False, **values}
    tmp = OUT / 'status.tmp.json'
    tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(OUT / 'status.json')


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)


def alive_matching(pid, fragment):
    if not pid:
        return False
    try:
        return fragment in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
    except FileNotFoundError:
        return False


def main():
    global child
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    config = read(ROOT / 'experiment/phase20/gacr_v6_pcrf_toys_v1.json')
    output = ROOT / config['output']
    started = time.time()
    while not stopping:
        if time.time()-started > 18*3600:
            save('STOPPED', reason='18-hour prerequisite/resource wait limit; no launch')
            return
        smoke = read(ROOT / config['smoke_output'] / 'smoke.json')
        smoke_status = read(ROOT / config['smoke_output'] / 'status.json')
        if smoke_status.get('state') in ['FAILED', 'STOPPED']:
            save('FAILED', reason='GACR engineering check did not pass', dependency=smoke_status)
            return
        sft = read(ROOT / 'artifacts/phase20/round1_sft_full_v1/status.json')
        sft_active = alive_matching(sft.get('pid'), b'experiment.phase20.evaluate_saved')
        sft_ready = (not config.get('wait_for_sft', True) or
                     (sft.get('state') in ['COMPLETED', 'FAILED', 'STOPPED'] and not sft_active))
        if smoke.get('state') != 'PASSED' or not sft_ready:
            save('WAITING', sft_state=sft.get('state', 'STARTING'), smoke_state=smoke.get('state', smoke_status.get('state', 'RUNNING')),
                 reason='Wait for configured prerequisites and successful GACR engineering check')
            time.sleep(30)
            continue
        if (output / 'manifest.json').exists():
            save('FAILED', reason='Existing GACR attempt; refusing restart or overwrite')
            return
        try:
            query = subprocess.check_output(['nvidia-smi', '--id='+config['gpu_uuid'],
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20)
            free = int(query.strip())
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            save('WAITING_FOR_RESOURCE', reason=repr(error))
            time.sleep(30)
            continue
        if free < config['minimum_free_mib']:
            save('WAITING_FOR_RESOURCE', free_mib=free, required_mib=config['minimum_free_mib'])
            time.sleep(30)
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'experiment.phase20.evaluate_gacr_pcrf',
                   '--config', 'experiment/phase20/gacr_v6_pcrf_toys_v1.json', '--mode', 'run']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (output / 'run.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launched = time.time()
            timeout_sent = None
            while child.poll() is None:
                detail = read(output / 'status.json')
                save('STOPPING' if stopping else 'RUNNING', child_pid=child.pid, output=config['output'],
                     child_status=detail, elapsed_seconds=round(time.time()-launched, 1))
                if time.time()-launched > config['max_wall_seconds']+90 and timeout_sent is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    timeout_sent = time.time()
                if timeout_sent is not None and time.time()-timeout_sent > 60:
                    os.killpg(child.pid, signal.SIGKILL)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        detail = read(output / 'status.json')
        summary = read(output / 'summary.json')
        success = child.returncode == 0 and detail.get('state') == 'COMPLETED' and summary.get('state') == 'COMPLETED'
        save('COMPLETED' if success else 'STOPPED' if stopping else 'FAILED', child_exit_code=child.returncode,
             child_status=detail, output=config['output'])
        return
    save('STOPPED', reason='User stopped follow-up queue before launch')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save('FAILED', error=repr(error))
        raise
