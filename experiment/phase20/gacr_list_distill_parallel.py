"""Adopt the live cache, then run the three frozen arms on distinct GPUs.

The original experiment/config remain immutable. Briefly pause the independent
cache process while retiring only the old supervisor, then resume that same cache
PID. Each arm owns its own train/evaluate directories, and one controller owns
status/report/aggregation. No cache/model computation is repeated.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20.gacr_list_distill_queue import ARMS, child_path, report
from experiment.phase20.gacr_fusion_queue import read, write

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'experiment/phase20/gacr_list_distill_toys_gpu1_v2.json'
GPU_POOL = [1, 4, 6, 0, 3, 5, 2, 7]
PREFERRED = {'CE_anchor': 4, 'true_teacher': 1, 'shuffled_teacher': 6}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def process(pid):
    try:
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rpartition(') ')[2].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
                'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'uid': base.stat().st_uid,
                'argv': (base / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    live = process(expected['pid'])
    return bool(live and live['state'] != 'Z' and all(
        live[k] == expected[k] for k in ('start_ticks', 'uid', 'argv')))


def checked_signal(expected, sig, group=False):
    if not same_process(expected):
        raise RuntimeError('Process identity changed; refusing signal: ' + str(expected['pid']))
    assert expected['uid'] == os.getuid()
    if group:
        assert process(expected['pid'])['pgid'] == expected['pid']
        os.killpg(expected['pid'], sig)
    else:
        os.kill(expected['pid'], sig)


def validate_handoff(status):
    assert status['state'] == 'RUNNING' and status['current_task'] == 'cache'
    assert status['new_optimizer_steps'] == 0
    states = {t['id']: t['state'] for t in status['tasks']}
    assert states['profile'] == 'COMPLETED' and states['cache'] == 'RUNNING'
    assert all(v == 'PENDING' for k, v in states.items() if k not in ('profile', 'cache'))


def ready_tasks(tasks):
    states = {t['id']: t['state'] for t in tasks}
    if states['cache'] != 'COMPLETED':
        return []
    ready = []
    for t in tasks:
        if t['state'] != 'PENDING':
            continue
        if t['mode'] == 'train' or (
            t['mode'] == 'evaluate' and states[t['arm'] + '_train'] == 'COMPLETED'
        ) or (t['mode'] == 'aggregate' and all(states[a + '_evaluate'] == 'COMPLETED' for a in ARMS)):
            ready.append(t)
    return ready


def choose_gpu(arm, bindings, boards, minimum):
    if arm in bindings:
        uuid = bindings[arm]['uuid']
        return next((b for b in boards if b['uuid'] == uuid and b['free'] >= minimum), None)
    occupied = {b['uuid'] for b in bindings.values()}
    order = [PREFERRED[arm]] + [i for i in GPU_POOL if i != PREFERRED[arm]]
    for index in order:
        for board in boards:
            if (board['index'] == index and board['uuid'] not in occupied
                    and board['name'] == 'NVIDIA RTX A6000' and board['free'] >= minimum):
                return board
    return None


def worker_config(original, board, path):
    config = copy.deepcopy(original)
    config['gpu_uuid'] = board['uuid']
    config['config_path'] = path
    config['resource_revision'] = {
        'previous_config': CONFIG, 'previous_config_sha256': digest(ROOT / CONFIG),
        'reason': 'User requested distinct GPUs on 2026-09-15; scientific settings unchanged.',
        'gpu_index': board['index'], 'controller': str(Path(__file__).relative_to(ROOT)),
    }
    return config


def verify_sources(config):
    for path, expected in {**config['source_sha256'], **config['input_sha256']}.items():
        assert digest(ROOT / path) == expected, ('changed_frozen_source_or_input', path)


class Controller:
    def __init__(self):
        self.config = read(ROOT / CONFIG)
        self.out = ROOT / self.config['output']
        self.meta = self.out / 'parallel_v2'
        self.meta.mkdir(exist_ok=True)
        self.lock = (self.meta / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (self.meta / 'status.json').exists(), 'Existing parallel attempt; no restart/overwrite'
        self.started = time.monotonic()
        self.tasks = []
        self.bindings, self.workers, self.active, self.wait_since = {}, {}, {}, {}
        self.stopping, self.paused, self.retired = False, False, False
        self.cache_paused, self.cache_complete = False, False
        self.owns_status = False
        self.old, self.cache, self.receipt = None, None, {}
        self.source_hash = digest(__file__)
        self.config_hash = digest(ROOT / CONFIG)
        self.identity = process(os.getpid())

    def stop(self, *_):
        self.stopping = True

    def take_over(self):
        verify_sources(self.config)
        before = read(self.out / 'status.json')
        validate_handoff(before)
        self.old, self.cache = process(before['pid']), process(before['child_pid'])
        assert self.old and self.cache
        assert self.old['start_ticks'] == before['pid_start_ticks']
        assert self.cache['ppid'] == self.old['pid'] and self.cache['pgid'] == self.cache['pid']
        assert 'experiment.phase20.gacr_list_distill_queue' in self.old['argv']
        assert self.old['argv'][-2:] == ['--config', CONFIG]
        assert self.cache['argv'][-4:] == ['--config', CONFIG, '--mode', 'cache']
        assert self.old['uid'] == self.cache['uid'] == os.getuid()
        for arm in ARMS:
            assert not (self.out / arm).exists(), ('arm_already_started', arm)
        write(self.meta / 'before_handoff.json', before)
        self.receipt = {'original_supervisor': self.old, 'adopted_cache': self.cache,
                        'controller': self.identity, 'source_sha256': self.source_hash,
                        'original_config_sha256': self.config_hash,
                        'authorization': 'User: 我想分卡运行', 'test_read': False}
        write(self.meta / 'handoff.json', self.receipt)
        # The cache has its own session, so the terminal's foreground-job
        # handling cannot resume it. Freeze it briefly to close the completion
        # race, retire only the old supervisor, then resume the same cache PID.
        checked_signal(self.cache, signal.SIGSTOP)
        self.cache_paused = True
        for _ in range(100):
            if process(self.cache['pid'])['state'] == 'T':
                break
            time.sleep(.01)
        assert process(self.cache['pid'])['state'] == 'T'
        stable = read(self.out / 'status.json')
        validate_handoff(stable)
        assert stable['child_pid'] == self.cache['pid']
        assert read(self.out / 'cache/status.json')['state'] == 'RUNNING'
        self.tasks = copy.deepcopy(stable['tasks'])
        self.owns_status = True
        self.retire_old()
        checked_signal(self.cache, signal.SIGCONT)
        self.cache_paused = False
        self.cache_deadline = time.monotonic() + max(0, self.config['cache_budget_seconds']
            - read(self.out / 'cache/status.json').get('elapsed_seconds', 0)) + 60
        self.receipt.update(state='CACHE_ADOPTED_SUPERVISOR_RETIRED', cache_recomputed=False,
                            cache_same_pid_resumed=True)
        write(self.meta / 'handoff.json', self.receipt)
        self.save('RUNNING', 'cache')

    def retire_old(self):
        if self.old and not self.retired:
            # The old handler forwards SIGTERM to its child; never invoke it.
            # SIGKILL here addresses only the verified supervisor PID.
            checked_signal(self.old, signal.SIGKILL)
            self.retired = True
            self.paused = False
            self.original_lock = (self.out / 'queue.lock').open('a')
            for _ in range(100):
                try:
                    fcntl.flock(self.original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.05)
            else:
                raise RuntimeError('Old supervisor did not release queue lock')
            self.receipt['state'] = 'PARALLEL_CONTROLLER_OWNS_QUEUE'
            write(self.meta / 'handoff.json', self.receipt)

    def save(self, state, current=None, reason=None):
        details = {t['id']: read(child_path(self.out, t) / 'status.json') for t in self.tasks}
        for task in self.tasks:
            task['stage'] = details[task['id']].get('stage', '')
        profile = read(self.out / 'profile/profile.json')
        remaining = []
        for arm in ARMS:
            seconds = 0
            for mode in ('train', 'evaluate'):
                task = next(t for t in self.tasks if t['id'] == arm + '_' + mode)
                if task['state'] in ('PENDING', 'RUNNING', 'WAITING_FOR_RESOURCE'):
                    estimate = (profile['estimated_training_seconds'][arm] if mode == 'train'
                                else profile['estimated_evaluation_seconds_per_arm'])
                    seconds += max(0, estimate - details[task['id']].get('elapsed_seconds', 0))
            remaining.append(seconds)
        cache_detail = details.get('cache', {})
        cache_remaining = (cache_detail.get('stage_eta_seconds', 0) or 0) if not self.cache_complete else 0
        value = {'state': state, 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'pid': os.getpid(), 'pid_start_ticks': self.identity['start_ticks'], 'controller_identity': self.identity,
            'controller_module': 'experiment.phase20.gacr_list_distill_parallel',
            'elapsed_seconds': round(time.monotonic()-self.started, 1), 'heartbeat_interval_seconds': 30,
            'scheduling': 'shared cache -> up to three distinct GPU train/evaluate chains -> one aggregate',
            'current_task': current, 'tasks': self.tasks, 'task_status': details,
            'completed_tasks': sum(t['state'] == 'COMPLETED' for t in self.tasks), 'total_tasks': len(self.tasks),
            'gpu_uuid': self.config['gpu_uuid'], 'arm_gpu_bindings': self.bindings,
            'preferred_gpu_indices': PREFERRED, 'minimum_free_mib': self.config['minimum_free_mib'],
            'new_optimizer_steps': sum(d.get('new_optimizer_steps', 0) for d in details.values()),
            'planned_new_optimizer_steps': 513, 'engineering_optimizer_steps_discarded': 1,
            'projected_remaining_seconds': round(cache_remaining + max(remaining) + 300, 1),
            'eta_assumption': 'three GPUs available; per-GPU throughput from original profile; excludes resource waits',
            'active_children': {k: {'pid': w['identity']['pid'], 'gpu': w['board']} for k, w in self.active.items()},
            'adopted_cache_pid': self.cache['pid'], 'original_supervisor_paused': self.paused,
            'original_supervisor_retired': self.retired,
            'child_status': cache_detail if not self.cache_complete else None,
            'reason': reason, 'test_read': False, 'automatic_retry': False, 'automatic_next_experiment': False}
        if state == 'COMPLETED':
            value['projected_remaining_seconds'] = 0
        write(self.meta / 'status.json', value)
        if self.owns_status:
            write(self.out / 'status.json', value)
            display = copy.deepcopy(self.config)
            display['gpu_uuid'] = ', '.join(f'{a}=GPU{b["index"]}' for a, b in self.bindings.items()) or '缓存GPU1；三臂待分配'
            report(display, value)
            target = ROOT / self.config['report']
            text = target.read_text().replace('通过后依次生成父保持缓存、训练三臂、对三臂各完整生成',
                '共用父保持缓存完成后，三臂分卡各自训练、完成后各完整生成')
            text = text.replace('profile后队列约', '原单卡串行工作量约')
            notice = ('\n**2026-09-15分卡调度：** 当前状态'+state+'；三臂优先GPU4/1/6，'
                '仅在空闲显存≥30,000 MiB时分配不同A6000，实际分配见status。每臂训练后在原卡验证，'
                '全部完成后汇总。原缓存继续计算，训练协议、步数、输入和模型代码保持冻结。\n\n'
                f'三卡可用且吞吐接近profile时，估计剩余{value["projected_remaining_seconds"]/3600:.2f}小时；'
                '不含等卡时间，共享负载可能使估算变化。\n')
            text = text.replace('\n## Material Passport', notice + '\n## Material Passport', 1)
            temp = target.with_suffix('.parallel.tmp.md'); temp.write_text(text); temp.replace(target)

    def await_cache(self):
        while same_process(self.cache):
            if self.stopping:
                raise InterruptedError('User stopped parallel controller')
            assert not same_process(self.old), 'Original supervisor must no longer schedule tasks'
            if time.monotonic() > self.cache_deadline:
                raise TimeoutError('Adopted cache exceeded original budget')
            self.save('RUNNING', 'cache')
            time.sleep(30)
        status = read(self.out / 'cache/status.json')
        manifest = read(self.out / 'cache/manifest.json')
        assert status['state'] == manifest['state'] == 'COMPLETED', ('adopted_cache_failed', status)
        assert manifest['cohort'] == self.config['frozen_cohort']
        for file, key in [('distillation_cache.npz', 'distillation_cache_sha256'),
                          ('training_order.npy', 'training_order_sha256')]:
            assert digest(self.out / 'cache' / file) == manifest[key]
        write(self.meta / 'cache_reuse.json', {'manifest': manifest, 'verified_at': time.strftime('%FT%T%z'),
            'exit_code': None, 'verification': 'Completed child status + exited process + manifest/file hashes; adopted process is not a direct child.',
            'cache_recomputed': False, 'test_read': False})
        self.cache_complete = True
        next(t for t in self.tasks if t['id'] == 'cache').update(state='COMPLETED', exit_code=None)

    def boards(self):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=20)
        result = []
        for line in raw.strip().splitlines():
            index, uuid, name, free = [part.strip() for part in line.split(',')]
            result.append(dict(index=int(index), uuid=uuid, name=name, free=int(free)))
        return result

    def launch(self, task, board):
        assert digest(__file__) == self.source_hash and digest(ROOT / CONFIG) == self.config_hash
        target = child_path(self.out, task)
        assert not target.exists(), ('existing_task_output', str(target))
        arm = task['arm']
        if arm and arm not in self.bindings:
            self.bindings[arm] = board
            path = str((self.meta / (arm + '.json')).relative_to(ROOT))
            config = worker_config(self.config, board, path)
            config['source_sha256'][str(Path(__file__).relative_to(ROOT))] = self.source_hash
            write(ROOT / path, config)
            self.workers[arm] = path
        config_path = self.workers[arm] if arm else CONFIG
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=board['uuid'] if board else '',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [sys.executable, '-u', '-m', 'experiment.phase20.gacr_list_distill', '--config', config_path,
                   '--mode', task['mode']]
        if arm:
            command += ['--arm', arm]
        log = (self.out / (task['id'] + '.log')).open('x')
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        identity = process(child.pid)
        self.active[task['id']] = {'child': child, 'identity': identity, 'log': log, 'board': board,
            'deadline': time.monotonic() + self.config[task['mode'] + '_budget_seconds'] + 60}
        task.update(state='RUNNING', pid=child.pid, gpu_index=board['index'] if board else None)
        write(self.meta / (task['id'] + '.launch.json'), {'identity': identity, 'command': command,
            'CUDA_VISIBLE_DEVICES': environment['CUDA_VISIBLE_DEVICES'], 'config_sha256': digest(ROOT / config_path),
            'board': board, 'test_read': False})

    def execute(self):
        self.take_over()
        self.await_cache()
        while True:
            if self.stopping:
                raise InterruptedError('User stopped parallel controller')
            for task_id, running in list(self.active.items()):
                code = running['child'].poll()
                if code is None:
                    if time.monotonic() > running['deadline']:
                        raise TimeoutError('Task exceeded frozen budget: ' + task_id)
                    continue
                running['log'].close()
                task = next(t for t in self.tasks if t['id'] == task_id)
                detail = read(child_path(self.out, task) / 'status.json')
                task.update(state='COMPLETED' if code == 0 and detail.get('state') == 'COMPLETED' else 'FAILED', exit_code=code)
                del self.active[task_id]
                if task['state'] != 'COMPLETED':
                    raise RuntimeError('Task failed; no automatic retry: ' + task_id)
            if all(t['state'] == 'COMPLETED' for t in self.tasks):
                self.save('COMPLETED', reason='All three arms trained/evaluated on distinct GPUs and aggregated.')
                return
            boards = self.boards()
            for task in ready_tasks(self.tasks):
                board = None
                if task['arm']:
                    board = choose_gpu(task['arm'], self.bindings, boards, self.config['minimum_free_mib'])
                    if board is None:
                        start = self.wait_since.setdefault(task['id'], time.monotonic())
                        task['resource_wait_seconds'] = round(time.monotonic() - start, 1)
                        if task['resource_wait_seconds'] > self.config['resource_wait_seconds']:
                            raise TimeoutError('Resource wait exceeded original 1h budget: ' + task['id'])
                        continue
                self.launch(task, board)
            current = ','.join(self.active) or 'waiting_for_gpu'
            self.save('RUNNING' if self.active else 'WAITING_FOR_RESOURCE', current)
            time.sleep(30)

    def cleanup(self, error):
        if not self.owns_status:
            # A failed preflight/race check must not interrupt the original run.
            if self.cache_paused and same_process(self.cache):
                checked_signal(self.cache, signal.SIGCONT)
                self.cache_paused = False
            write(self.meta / 'error.json', {'error': str(error), 'traceback': traceback.format_exc(),
                'original_queue_preserved': True})
            return
        # Stop only tasks belonging to this attempt. Preserve every artifact.
        if self.cache_paused and same_process(self.cache):
            checked_signal(self.cache, signal.SIGCONT)
            self.cache_paused = False
        targets = [w['identity'] for w in self.active.values() if w['child'].poll() is None]
        if self.cache and same_process(self.cache):
            targets.append(self.cache)
        for identity in targets:
            if same_process(identity):
                checked_signal(identity, signal.SIGTERM, group=True)
        deadline = time.monotonic() + 60
        while targets and time.monotonic() < deadline:
            targets = [p for p in targets if same_process(p)]
            if targets:
                time.sleep(.2)
        for identity in targets:
            checked_signal(identity, signal.SIGKILL, group=True)
        self.retire_old()
        for task in self.tasks:
            if task['state'] == 'RUNNING':
                task['state'] = 'STOPPED'
        if self.owns_status:
            self.save('STOPPED' if isinstance(error, InterruptedError) else 'FAILED', reason=str(error))
        write(self.meta / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / CONFIG)
    out = ROOT / config['output']
    if args.preflight:
        verify_sources(config)
        validate_handoff(read(out / 'status.json'))
        print(json.dumps({'state': 'PASSED', 'scientific_protocol_unchanged': True,
            'preferred_gpu_indices': PREFERRED, 'minimum_free_mib': config['minimum_free_mib']}))
        return
    if args.stop:
        status = read(out / 'status.json')
        assert status['state'] in ('RUNNING', 'WAITING_FOR_RESOURCE')
        checked_signal(status['controller_identity'], signal.SIGTERM)
        print('Stop requested for verified parallel controller', status['pid'])
        return
    controller = Controller()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, controller.stop)
    try:
        controller.execute()
    except BaseException as error:
        controller.cleanup(error)
        raise


if __name__ == '__main__':
    main()
