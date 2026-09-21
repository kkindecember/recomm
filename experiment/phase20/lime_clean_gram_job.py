"""Two-domain background queue. No training before frozen checks and GPU admission."""
import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now, gpu_candidates

CONFIG = ROOT / os.environ.get('RECOMM_CLEAN_GRAM_CONFIG', 'experiment/phase20/lime_clean_gram_20260920_v1.json')
TERMINAL = {'COMPLETED', 'FAILED', 'RESOURCE_WAIT_EXPIRED', 'COST_GATE_FAILED', 'STOPPED'}


def eligible_gpus(candidates, busy, minimum_free_mib, required_uuid=None):
    return [g for g in candidates if g['free_mib'] >= minimum_free_mib
            and g['uuid'] not in busy and (required_uuid is None or g['uuid'] == required_uuid)]


def profile_passed(summary, manifest):
    return (summary.get('state') == 'PASSED' and summary.get('input_sha256') == manifest
            and summary.get('profile_weights_discarded') is True)


def validate_launch(config, root):
    if config.get('execution_mode_evidence'):
        evidence = read(ROOT / config['execution_mode_evidence'])
        assert evidence['state'] == 'COMPLETED' and evidence['reports']['native_tf32']['numerical_check_passed']
    prepared = read(root / 'prepared.json')
    assert prepared['state'] == 'PREPARED' and prepared['config_sha256'] == sha(CONFIG)
    tests = read(root / 'tests.json')
    assert tests['state'] == 'PASSED'
    assert tests['source_sha256'] == prepared['source_sha256'], 'Tests must cover frozen sources'
    for domain in config['domains']:
        manifest = read(root / domain / 'input_manifest.json')
        smoke = read(root / domain / 'cpu_check/summary.json')
        assert smoke['state'] == 'PASSED' and smoke['input_sha256'] == manifest
        assert all(s['exact_resume'] for s in smoke['stages'].values())
        assert not (root / domain / 'profile').exists()
        assert not (root / domain / 'train').exists()
        for name, digest in manifest.items():
            assert sha(ROOT / name) == digest, f'Changed frozen file: {name}'
    assert shutil.disk_usage(root).free >= config['minimum_disk_free_gib'] * 2**30


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            # The last completed atomic checkpoint remains safe even if mid-update.
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def supervise(config, root):
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _supervise(config, root)


def _supervise(config, root):
    start = time.monotonic()
    jobs = {}
    states = {d: dict(state='WAITING_GPU', next_action='profile', stage='等待正式批量检查所需GPU',
              formal_training_started=False) for d in config['domains']}
    waiting_since = {d: start for d in config['domains']}
    observations = []

    def status(state, **extra):
        write(root / 'status.json', dict(state=state, updated_at=now(), supervisor_pid=os.getpid(),
            elapsed_seconds=time.monotonic()-start, domains=states, gpu_observations=observations,
            minimum_free_mib=config['minimum_free_mib'], active_workers=len(jobs),
            test_read=False, new_recommendation_metrics=False, **extra))

    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        validate_launch(config, root)
        while True:
            for domain, job in list(jobs.items()):
                path = root / domain / job['action'] / 'status.json'
                worker = read(path) if path.exists() else {}
                states[domain]['worker'] = worker
                code = job['child'].poll()
                elapsed = time.monotonic()-job['started']
                limit = config['profile_timeout_seconds'] if job['action'] == 'profile' else config['training_timeout_seconds']
                if code is None and elapsed > limit+300:
                    states[domain].update(state='FAILED', error=f'{job["action"]} hard timeout; stopping own worker')
                    status('RUNNING', stage='记录硬超时后停止子进程')
                    stop_child(job['child'])
                    code = job['child'].returncode
                if code is None:
                    continue
                job['log'].close()
                del jobs[domain]
                states[domain].update(worker_alive=False, exit_code=code)
                summary_path = root / domain / job['action'] / 'summary.json'
                summary = read(summary_path) if summary_path.exists() else {}
                if code != 0:
                    states[domain].update(state='FAILED', error=worker.get('error', f'Worker exit {code}'))
                elif job['action'] == 'profile':
                    manifest = read(root / domain / 'input_manifest.json')
                    if profile_passed(summary, manifest):
                        states[domain].update(state='WAITING_GPU', next_action='train', stage='批量检查通过，等待同卡正式训练',
                            required_gpu_uuid=job['gpu']['uuid'], profile=summary)
                        waiting_since[domain] = time.monotonic()
                    else:
                        states[domain].update(state='COST_GATE_FAILED' if summary.get('state') == 'COST_GATE_FAILED' else 'FAILED',
                            stage='检查未通过，不进入正式训练', profile=summary)
                elif summary.get('state') == 'COMPLETED' and worker.get('state') == 'COMPLETED':
                    states[domain].update(state='COMPLETED', stage='GRAM/C1已完成，候选与排序验证待接续')
                else:
                    states[domain].update(state='FAILED', error='Worker exited without a valid completion receipt')
            if all(s['state'] in TERMINAL for s in states.values()):
                break
            pending = [d for d, state in states.items() if state['state'] == 'WAITING_GPU']
            observations = gpu_candidates() if pending else observations
            busy = {job['gpu']['uuid'] for job in jobs.values()}
            for domain in pending:
                if time.monotonic()-waiting_since[domain] > config['resource_wait_seconds']:
                    states[domain].update(state='RESOURCE_WAIT_EXPIRED', stage='24小时等待结束，未自动改变显存或批量设置')
                    continue
                if len(jobs) >= config['max_parallel_workers']:
                    break
                candidates = eligible_gpus(observations, busy, config['minimum_free_mib'],
                                           states[domain].get('required_gpu_uuid'))
                if not candidates:
                    continue
                assert shutil.disk_usage(root).free >= config['minimum_disk_free_gib'] * 2**30
                gpu = candidates[0]
                action = states[domain]['next_action']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS=str(config['cpu_threads']),
                           OPENBLAS_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']),
                           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command = [sys.executable, '-B', '-m', 'experiment.phase20.lime_clean_gram_worker', action,
                           '--domain', domain, '--gpu-uuid', gpu['uuid']]
                log = (root / domain / f'{action}.host.log').open('x', buffering=1)
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                jobs[domain] = dict(child=child, started=time.monotonic(), gpu=gpu, action=action, log=log)
                busy.add(gpu['uuid'])
                states[domain].update(state='PROFILING' if action == 'profile' else 'TRAINING',
                    stage='正式批量检查' if action == 'profile' else 'GRAM接续C1正式训练',
                    worker_pid=child.pid, worker_alive=True, gpu=gpu,
                    formal_training_started=action == 'train')
                write(root / domain / f'{action}.launch.json', dict(pid=child.pid, created_at=now(), command=command, gpu=gpu))
            status('RUNNING' if jobs else 'WAITING_GPU', stage='两域独立排队；检查通过后自动GRAM→C1')
            time.sleep(config['status_interval_seconds'])
        write(root / 'summary.json', dict(states=states, completed_at=now(), test_read=False,
              new_recommendation_metrics=False))
        status('COMPLETED' if all(s['state'] == 'COMPLETED' for s in states.values()) else 'FINISHED_WITH_FAILURES',
               stage='查看各域终态；失败不自动重试')
    except BaseException as error:
        for domain, job in jobs.items():
            stop_child(job['child'])
            job['log'].close()
            states[domain].update(state='STOPPED', worker_alive=False)
        (root / f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        status('FAILED', stage='后台调度停止', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'supervise', 'status'))
    args = parser.parse_args()
    config = read(CONFIG)
    root = ROOT / config['output']
    if args.action == 'status':
        path = root / 'status.json'
        print(path.read_text() if path.exists() else 'No supervisor started', flush=True)
    elif args.action == 'launch':
        validate_launch(config, root)
        assert not (root / 'launch.json').exists(), 'Existing attempt; inspect state rather than relaunching'
        # Acquire an exclusive launch marker to prevent concurrent double starts.
        with (root / 'launch.json').open('x') as marker:
            import json
            json.dump(dict(state='LAUNCHING', created_at=now()), marker)
        with (root / 'supervisor.log').open('x', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_clean_gram_job', 'supervise'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(root / 'launch.json', dict(pid=child.pid, created_at=now(), status_path=str(root / 'status.json')))
        print(f'Background supervisor PID {child.pid}; status: {root}/status.json', flush=True)
    else:
        supervise(config, root)


if __name__ == '__main__':
    main()
