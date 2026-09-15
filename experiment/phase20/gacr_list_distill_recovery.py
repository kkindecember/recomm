"""Recover the failed list-distillation attempt without changing its model code.

Reuse verified profile/cache in a new directory. GPU admission has a launch
margin and two observations; a worker waits in the same process after CUDA init.
Training, evaluation, initialization and checkpoint rules use the frozen Work.
"""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20.gacr_fusion_queue import read, write
from experiment.phase20.gacr_list_distill_queue import ARMS, child_path, report
from experiment.phase20.gacr_list_distill_parallel import (
    digest, process, same_process, checked_signal, choose_gpu, ready_tasks, verify_sources,
)

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = 'experiment/phase20/gacr_list_distill_toys_gpu1_v2.json'
CONFIG = 'experiment/phase20/gacr_list_distill_toys_recovery_v3.json'
OUTPUT = 'artifacts/phase20/gacr_list_distill_toys_recovery_v3'
MODULE = 'experiment.phase20.gacr_list_distill_recovery'
TERMINAL = {'COMPLETED', 'FAILED', 'STOPPED'}


def wait_for_memory(query, minimum, deadline, observe, clock=time.monotonic, sleep=time.sleep):
    """No model/optimizer work or process restart occurs while waiting."""
    while True:
        now = clock()
        if now >= deadline:
            raise TimeoutError('GPU resource wait exceeded the original 1 hour budget')
        free = query()
        observe(free, free >= minimum, max(0., deadline - now))
        if free >= minimum:
            return free
        sleep(min(10., max(0., deadline - clock())))


def stable_boards(boards, previous, threshold):
    current = {b['uuid'] for b in boards if b['free'] >= threshold}
    return [b for b in boards if b['uuid'] in current & previous], current


def recovery_config(original):
    config = copy.deepcopy(original)
    config.update(id='stage20_gacr_list_distill_toys_recovery_v3', config_path=CONFIG, output=OUTPUT)
    config['resource_revision'] = {
        'previous_config': ORIGINAL, 'previous_config_sha256': digest(ROOT / ORIGINAL),
        'previous_output': original['output'], 'authorization': '2026-09-15 user requested recovery after failure diagnosis',
        'launch_margin_mib': 1024, 'stable_observations': 2,
        'post_cuda_admission': 'Wait in the same process for original minimum; no optimizer retry',
        'resource_wait_budget': 'One hour per task including scheduler and initialized-worker waits',
        'scientific_protocol_unchanged': True,
    }
    for path in ('experiment/phase20/gacr_list_distill_recovery.py',
                 'experiment/phase20/run_gacr_list_distill_recovery.sh',
                 'experiment/phase20/test_gacr_list_distill_recovery.py'):
        config['source_sha256'][path] = digest(ROOT / path)
    return config


def prepare():
    original = read(ROOT / ORIGINAL)
    verify_sources(original)
    old = ROOT / original['output']
    status = read(old / 'status.json')
    assert status['state'] == 'FAILED'
    identities = [status['controller_identity']]
    identities += [read(p)['identity'] for p in (old / 'parallel_v2').glob('*.launch.json')]
    assert not any(same_process(p) for p in identities), 'Old attempt still has live processes'
    assert not list(old.glob('*/train/*.pt')), 'Unexpected checkpoints; recovery must be reviewed'
    assert read(old / 'profile/profile.json')['state'] == 'PASSED'
    manifest = read(old / 'cache/manifest.json')
    assert manifest['state'] == 'COMPLETED' and manifest['cohort'] == original['frozen_cohort']
    for name, key in [('distillation_cache.npz', 'distillation_cache_sha256'),
                      ('training_order.npy', 'training_order_sha256')]:
        assert digest(old / 'cache' / name) == manifest[key]
    config = recovery_config(original)
    assert not (ROOT / CONFIG).exists(), 'Recovery config already exists'
    out = ROOT / OUTPUT
    out.mkdir(exist_ok=False)
    for name in ('profile', 'cache'):
        shutil.copytree(old / name, out / name)
    # Preserve the exact failure report before the shared report becomes live.
    shutil.copy2(ROOT / original['report'], out / 'previous_failure_report.md')
    config['input_sha256'][ORIGINAL] = digest(ROOT / ORIGINAL)
    for path in [out / 'profile/profile.json', out / 'cache/manifest.json',
                 out / 'cache/distillation_cache.npz', out / 'cache/training_order.npy']:
        config['input_sha256'][str(path.relative_to(ROOT))] = digest(path)
    write(ROOT / CONFIG, config)
    write(out / 'recovery_receipt.json', {
        'prepared_at': time.strftime('%FT%T%z'), 'source_output': original['output'],
        'source_status_sha256': digest(old / 'status.json'), 'source_formal_optimizer_steps': status['new_optimizer_steps'],
        'discarded_previous_training_steps': 1, 'new_training_steps': 513,
        'cache_copied_and_verified': True, 'cache_recomputed': False,
        'scientific_protocol_unchanged': True, 'test_read': False,
    })
    print('Prepared recovery:', CONFIG)


def run_worker(config, mode, arm, resource_deadline):
    from experiment.phase20 import gacr_list_distill as frozen
    job = frozen.Work(config, mode, arm)

    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError('Resource or execution time budget exceeded')
        raise frozen.Interrupted(f'SIGNAL_{signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    try:
        frozen.torch.set_num_threads(config['cpu_threads'])
        if mode != 'aggregate':
            signal.alarm(max(1, math.ceil(resource_deadline - time.monotonic())))
            job.status('WAITING_FOR_RESOURCE', stage='cuda_initialization')
            frozen.torch.cuda.set_device(0)

            def observe(free, admitted, remaining):
                value = {'free_mib': free, 'minimum_free_mib': config['minimum_free_mib'],
                         'gpu_uuid': config['gpu_uuid'], 'admitted': admitted,
                         'resource_wait_remaining_seconds': round(remaining, 1),
                         'checked_at': time.strftime('%FT%T%z'), 'new_optimizer_steps': 0}
                write(job.path / 'resource_admission.json', value)
                job.status('RUNNING' if admitted else 'WAITING_FOR_RESOURCE',
                           stage='resource_admitted' if admitted else 'waiting_for_gpu_memory', **value)

            wait_for_memory(lambda: frozen.torch.cuda.mem_get_info()[0] / 2**20,
                            config['minimum_free_mib'], resource_deadline, observe)
            frozen.torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
        signal.alarm(config[mode + '_budget_seconds'])
        # Everything below uses the unchanged scientific worker, including seed,
        # full source/input verification, decoder updates and real fresh beams.
        frozen.seed_everything(config['seed'])
        job.verify()
        job.snapshot()
        assert read(job.output / 'profile/profile.json')['state'] == 'PASSED'
        if mode == 'train':
            job.train_arm()
        elif mode == 'evaluate':
            job.evaluate()
        else:
            job.aggregate()
    except BaseException as error:
        job.status('STOPPED' if isinstance(error, frozen.Interrupted) else 'FAILED',
                   error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)


class Controller:
    def __init__(self, config_path=CONFIG, poll_seconds=30):
        self.config_path = config_path
        self.config = read(ROOT / config_path)
        self.out = ROOT / self.config['output']
        self.lock = (self.out / 'queue.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (self.out / 'status.json').exists(), 'Existing recovery attempt; no overwrite/retry'
        self.identity = process(os.getpid())
        self.started = time.monotonic()
        self.config_hash = digest(ROOT / config_path)
        self.tasks = [dict(id=m, mode=m, arm=None, state='COMPLETED', reused=True) for m in ('profile', 'cache')]
        self.tasks += [dict(id=a+'_'+m, mode=m, arm=a, state='PENDING') for m in ('train', 'evaluate') for a in ARMS]
        self.tasks += [dict(id='aggregate', mode='aggregate', arm=None, state='PENDING')]
        self.bindings, self.workers, self.active, self.wait_since = {}, {}, {}, {}
        self.previous_ready, self.latest_boards = set(), []
        self.stopping, self.poll_seconds = False, poll_seconds
        (self.out / 'workers').mkdir(exist_ok=False)

    def stop(self, *_):
        self.stopping = True

    def boards(self):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free',
                                       '--format=csv,noheader,nounits'], text=True, timeout=20)
        result = []
        for line in raw.strip().splitlines():
            index, uuid, name, free = [v.strip() for v in line.split(',')]
            result.append(dict(index=int(index), uuid=uuid, name=name, free=int(free)))
        return result

    def command(self, task, config_path, deadline):
        command = [sys.executable, '-u', '-m', MODULE, '--worker', '--config', config_path,
                   '--mode', task['mode'], '--resource-deadline', str(deadline)]
        return command + (['--arm', task['arm']] if task['arm'] else [])

    def launch(self, task, board, deadline):
        assert digest(ROOT / self.config_path) == self.config_hash
        verify_sources(self.config)
        assert not child_path(self.out, task).exists(), 'Existing task output; no overwrite'
        arm = task['arm']
        if arm and arm not in self.bindings:
            self.bindings[arm] = board
            worker = copy.deepcopy(self.config)
            path = str((self.out / 'workers' / (arm + '.json')).relative_to(ROOT))
            worker.update(config_path=path, gpu_uuid=board['uuid'])
            worker['resource_revision']['gpu_index'] = board['index']
            write(ROOT / path, worker)
            self.workers[arm] = path
        config_path = self.workers[arm] if arm else self.config_path
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=board['uuid'] if board else '', OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = self.command(task, config_path, deadline)
        log = (self.out / (task['id'] + '.log')).open('x')
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        running = {'child': child, 'identity': process(child.pid), 'log': log, 'board': board,
                   'deadline': (deadline if arm else time.monotonic()) + self.config[task['mode']+'_budget_seconds'] + 60}
        self.active[task['id']] = running
        task.update(state='RUNNING', pid=child.pid, gpu_index=board['index'] if board else None)
        write(self.out / 'workers' / (task['id']+'.launch.json'), {
            'identity': running['identity'], 'command': command, 'board': board,
            'launched_at': time.strftime('%FT%T%z'), 'config_sha256': digest(ROOT / config_path),
            'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'], 'test_read': False})

    def save(self, state, reason=None):
        details = {t['id']: read(child_path(self.out, t) / 'status.json') for t in self.tasks}
        for task in self.tasks:
            task['stage'] = details[task['id']].get('stage', task.get('stage', ''))
        profile = read(self.out / 'profile/profile.json')
        remaining = []
        for arm in ARMS:
            seconds = 0.
            for mode in ('train', 'evaluate'):
                task = next(t for t in self.tasks if t['id'] == arm+'_'+mode)
                if task['state'] in ('PENDING', 'RUNNING'):
                    estimate = profile['estimated_training_seconds'][arm] if mode == 'train' else profile['estimated_evaluation_seconds_per_arm']
                    detail = details[task['id']]
                    seconds += max(0., detail.get('stage_eta_seconds', estimate))
            remaining.append(seconds)
        eta = 0 if state == 'COMPLETED' else None if state in TERMINAL else round(max(remaining)+300, 1)
        value = dict(state=state, updated_at=time.strftime('%FT%T%z'), pid=os.getpid(),
                     pid_start_ticks=self.identity['start_ticks'], controller_identity=self.identity,
                     elapsed_seconds=round(time.monotonic()-self.started, 1), heartbeat_interval_seconds=self.poll_seconds,
                     tasks=self.tasks, task_status=details, arm_gpu_bindings=self.bindings,
                     active_children={k: {'pid': r['child'].pid, 'gpu': r['board']} for k, r in self.active.items()},
                     current_task=','.join(self.active) or (None if state in TERMINAL else 'waiting_for_gpu'),
                     completed_tasks=sum(t['state']=='COMPLETED' for t in self.tasks), total_tasks=len(self.tasks),
                     new_optimizer_steps=sum(d.get('new_optimizer_steps', 0) for d in details.values()),
                     planned_new_optimizer_steps=513, projected_remaining_seconds=eta,
                     eta_assumption='Per-arm observed stage progress or original profile; excludes future resource waits and load changes',
                     minimum_free_mib=self.config['minimum_free_mib'],
                     launch_margin_mib=self.config['resource_revision']['launch_margin_mib'],
                     latest_gpu_observations=self.latest_boards, reason=reason,
                     cache_recomputed=False, previous_attempt=self.config['resource_revision']['previous_output'],
                     test_read=False, automatic_retry=False, automatic_next_experiment=False)
        write(self.out / 'status.json', value)
        self.render(value)

    def render(self, value):
        display = copy.deepcopy(self.config)
        display['gpu_uuid'] = ', '.join(f'{a}=GPU{b["index"]}' for a, b in self.bindings.items()) or '等待分配'
        report(display, value)
        path = ROOT / self.config['report']
        content = path.read_text()
        notice = '\n**2026-09-15恢复尝试 v3：** 复用已校验的profile与共享缓存；三臂分卡从原C1重新开始，各171步。旧失败记录保留在`'+value['previous_attempt']+'`。\n\n'
        notice += '选卡连续两次满足31,024 MiB；CUDA初始化后不足30,000 MiB时原进程等待，累计等资源最多1小时。真实计算失败仍停止队列，不自动重试。\n\n'
        if value['state'] in TERMINAL:
            notice += f'队列已结束：**{value["state"]}**。原因：{value["reason"]}。\n\n'
        else:
            notice += f'当前工作量估计约{value["projected_remaining_seconds"]/3600:.2f}小时，不含后续等卡和负载变化。\n\n'
        content = content.replace('## Material Passport', notice+'## Material Passport', 1)
        content = content.replace('通过后依次生成父保持缓存、训练三臂、对三臂各完整生成', '本次复用已完成的profile和父保持缓存，三臂分卡训练后各完整生成')
        content = content.replace('profile后队列约', '原单卡串行工作量约')
        temp = path.with_suffix('.recovery.tmp.md')
        temp.write_text(content)
        temp.replace(path)

    def execute(self):
        verify_sources(self.config)
        self.save('STARTING', 'Verified cache reused; three new matched training runs')
        while True:
            if self.stopping:
                raise InterruptedError('User stopped recovery controller')
            for task_id, running in list(self.active.items()):
                code = running['child'].poll()
                if code is None:
                    if time.monotonic() > running['deadline']:
                        raise TimeoutError('Task exceeded resource plus execution budget: '+task_id)
                    continue
                running['log'].close()
                task = next(t for t in self.tasks if t['id']==task_id)
                detail = read(child_path(self.out, task) / 'status.json')
                task.update(state='COMPLETED' if code==0 and detail.get('state')=='COMPLETED' else 'FAILED', exit_code=code)
                del self.active[task_id]
                if task['state']=='FAILED':
                    raise RuntimeError(f'{task_id}: exit={code}, stage={detail.get("stage")}, error={detail.get("error")}')
            if all(t['state']=='COMPLETED' for t in self.tasks):
                self.save('COMPLETED', 'All three arms trained, evaluated and aggregated')
                return
            self.latest_boards = self.boards()
            boards, self.previous_ready = stable_boards(self.latest_boards, self.previous_ready,
                self.config['minimum_free_mib'] + self.config['resource_revision']['launch_margin_mib'])
            for task in ready_tasks(self.tasks):
                start = self.wait_since.setdefault(task['id'], time.monotonic())
                deadline = start + self.config['resource_wait_seconds']
                board = choose_gpu(task['arm'], self.bindings, boards,
                    self.config['minimum_free_mib'] + self.config['resource_revision']['launch_margin_mib']) if task['arm'] else None
                if task['arm'] and time.monotonic() >= deadline:
                    raise TimeoutError('Resource wait exceeded original 1h budget: '+task['id'])
                if task['arm'] and board is None:
                    task.update(stage='waiting_for_gpu_memory', resource_wait_seconds=round(time.monotonic()-start, 1))
                    continue
                self.launch(task, board, deadline)
            self.save('RUNNING' if self.active else 'WAITING_FOR_RESOURCE')
            time.sleep(self.poll_seconds)

    def cleanup(self, error):
        for running in self.active.values():
            if running['child'].poll() is None and same_process(running['identity']):
                checked_signal(running['identity'], signal.SIGTERM, group=True)
        deadline = time.monotonic()+30
        for running in self.active.values():
            try:
                running['child'].wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                if same_process(running['identity']):
                    checked_signal(running['identity'], signal.SIGKILL, group=True)
                running['child'].wait(timeout=5)
            running['log'].close()
        self.active.clear()
        for task in self.tasks:
            if task['state']=='RUNNING':
                task['state']='STOPPED'
        self.save('STOPPED' if isinstance(error, InterruptedError) else 'FAILED', str(error))
        write(self.out / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--mode', choices=('train', 'evaluate', 'aggregate'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--resource-deadline', type=float, default=0.)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    config = read(ROOT / args.config)
    if args.worker:
        assert args.mode and ((args.arm is not None) == (args.mode in ('train', 'evaluate')))
        run_worker(config, args.mode, args.arm, args.resource_deadline)
        return
    if args.stop:
        status = read(ROOT / config['output'] / 'status.json')
        assert status['state'] not in TERMINAL
        checked_signal(status['controller_identity'], signal.SIGTERM)
        return
    controller = Controller(args.config)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, controller.stop)
    try:
        controller.execute()
    except BaseException as error:
        controller.cleanup(error)
        raise


if __name__ == '__main__':
    main()
