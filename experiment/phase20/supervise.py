"""Run Toys, then conditionally Beauty, in one bounded background session."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'artifacts/phase20'
child = None
stopping = False


def save(state, **values):
    path = OUTPUT/'status.json'
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps({'state': state, 'stage': 20, 'supervisor_pid': os.getpid(),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'tmux_session': 's20_sprec_gram_v1', 'test_read': False, **values}, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)


def launch(domain, mode):
    global child
    config_path = ROOT/f'experiment/phase20/sprec_{domain}_v1.json'
    config = json.loads(config_path.read_text())
    output = ROOT/config['smoke_output' if mode == 'smoke' else 'output']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError(f'Will not overwrite an existing {mode}: {output}')
    args = [sys.executable, '-u', '-m', 'experiment.phase20.run_preference',
            '--config', str(config_path.relative_to(ROOT)), '--mode', mode]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with (output/'run.log').open('a') as log:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while True:
            status_path = output/'status.json'
            detail = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'STARTING'}
            save('STOPPING' if stopping else 'RUNNING', dataset=domain.title(), mode=mode,
                 child_pid=child.pid, child_status=detail, output=str(output.relative_to(ROOT)))
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue
    if code != 0:
        raise RuntimeError(f'{domain} {mode} exited with {code}; no automatic retry')


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        toys_config = json.loads((ROOT/'experiment/phase20/sprec_toys_v1.json').read_text())
        smoke_path = ROOT/toys_config['smoke_output']/'smoke.json'
        if not smoke_path.exists() or json.loads(smoke_path.read_text()).get('state') != 'PASSED':
            raise RuntimeError('Toys engineering smoke must pass before background launch')
        launch('toys', 'run')
        toys = json.loads((ROOT/toys_config['output']/'summary.json').read_text())
        if stopping:
            save('STOPPED', reason='User stop requested')
        elif toys['promising_for_beauty']:
            launch('beauty', 'smoke')
            if stopping:
                save('STOPPED', reason='User stop requested before Beauty training')
                return
            launch('beauty', 'run')
            save('COMPLETED', completed_domains=['Toys', 'Beauty'], scientific_results='See each domain summary.json')
        else:
            save('COMPLETED', completed_domains=['Toys'], beauty='NOT_TRIGGERED',
                 reason='No positive full-validation raw or incremental PCRF NDCG signal under the fixed rule')
    except BaseException as error:
        save('STOPPED' if stopping else 'FAILED', error=repr(error), automatic_retry=False)
        raise


if __name__ == '__main__':
    main()
