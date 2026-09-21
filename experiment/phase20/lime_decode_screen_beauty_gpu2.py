"""Queue the unchanged Beauty screen behind Toys on the faster shared GPU."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20 import lime_decode_screen_pilot as pilot

OLD = pilot.OUT
OUT = OLD / 'beauty_gpu2'
MODULE = 'experiment.phase20.lime_decode_screen_beauty_gpu2'
base.OUT = pilot.OUT = OUT


def prepare():
    assert base.read(OLD / 'Beauty/status.json')['state'] == 'FAILED'
    assert not (OLD / 'Beauty/tuning.jsonl').exists()
    OUT.mkdir(exist_ok=False)
    (OUT / 'Beauty').mkdir()
    for name in ('signals.npz', 'samples.json', 'data_config.json', 'preflight.json'):
        os.link(OLD / 'Beauty' / name, OUT / 'Beauty' / name)
    config = base.read(OLD / 'config.json')
    # Absolute deadline measured from this task's first GPU screen launch.
    first = base.read(OLD.parent / 'launch.json')['started_at']
    deadline = datetime.strptime(first, '%Y-%m-%dT%H:%M:%S%z').timestamp() + 7200
    config.update(domains=['Beauty'], gpus={'Beauty': base.GPU['Toys']},
                  deadline_epoch=deadline, worker_timeout_seconds=7200,
                  scheduling_revision='Same signals/cohorts/alphas; wait for Toys to release GPU 2, then profile.')
    base.write(OUT / 'config.json', config)
    manifest = base.read(OLD / 'manifest.json')
    for p in (Path(__file__).resolve(), OUT / 'config.json'):
        manifest['sha256'][str(p.relative_to(base.ROOT))] = base.sha(p)
    base.write(OUT / 'manifest.json', manifest)
    (OUT / 'protocol.md').write_text((OLD / 'protocol.md').read_text() +
        '\n\n## Beauty资源调度补充\n\nGPU3短测完成但成本不通过，未读取Beauty选参/评价结果。'
        '在Toys仍选参、未查看效果时决定：等待Toys释放GPU2，再用同一模型、名单、分数、权重搜索及停止条件短测。'
        '样本档位继续只依工程成本确定。运行绝对截止时间为本任务首次GPU启动后两小时；等待与重试不重置这一时间。'
        '若成本仍不通过则停止，不能记为科学负结果。\n')
    manifest['sha256'][str((OUT / 'protocol.md').relative_to(base.ROOT))] = base.sha(OUT / 'protocol.md')
    base.write(OUT / 'manifest.json', manifest)
    base.write(OUT / 'status.json', dict(state='PREPARED', updated_at=base.stamp()))
    print(str(OUT), flush=True)


def supervise():
    config = base.read(OUT / 'config.json')
    while True:
        remaining = config['deadline_epoch'] - time.time()
        if remaining < 600:
            base.write(OUT / 'status.json', dict(state='RESOURCE_TIMEOUT', updated_at=base.stamp()))
            return
        toy = base.read(OLD / 'Toys/status.json')['state']
        if toy in ('COMPLETED', 'FAILED') and base.gpu_free(base.GPU['Toys']) >= config['minimum_free_mib']:
            break
        base.write(OUT / 'status.json', dict(state='WAITING_FOR_TOYS_GPU2', updated_at=base.stamp(),
                                           remaining_seconds=remaining))
        time.sleep(5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=base.GPU['Toys'], PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    with (OUT / 'Beauty/worker.log').open('x') as log:
        p = subprocess.Popen([sys.executable, '-m', MODULE, 'worker'], cwd=base.ROOT, env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while p.poll() is None:
        if time.time() > config['deadline_epoch']:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
            base.write(OUT / 'status.json', dict(state='TIMEOUT', updated_at=base.stamp()))
            return
        base.write(OUT / 'status.json', dict(state='RUNNING', updated_at=base.stamp(), worker_pid=p.pid))
        time.sleep(5)
    if p.returncode == 0:
        for path, digest in base.read(OUT / 'manifest.json')['sha256'].items():
            assert base.sha(base.ROOT / path) == digest, path
        base.write(OUT / 'verification.json', dict(input_hashes_unchanged=True, completed_at=base.stamp(),
                                                  independent_metrics_audit_pending=True))
    base.write(OUT / 'status.json', dict(state='COMPLETED' if p.returncode == 0 else 'INCOMPLETE',
                                        updated_at=base.stamp(), returncode=p.returncode))


class Worker(pilot.PilotWorker):
    def __init__(self):
        super().__init__('Beauty')
        self.config['worker_timeout_seconds'] = self.config['deadline_epoch'] - time.time() + time.monotonic() - self.started
        base.write(self.folder / 'runtime_budget.json', dict(worker_timeout_seconds=self.config['worker_timeout_seconds'],
                                                           absolute_deadline=self.config['deadline_epoch']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'launch', 'supervise', 'worker'])
    action = p.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'launch':
        assert base.read(OUT / 'status.json')['state'] == 'PREPARED'
        with (OUT / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', MODULE, 'supervise'], cwd=base.ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base.write(OUT / 'launch.json', dict(pid=child.pid, started_at=base.stamp()))
        print(dict(pid=child.pid, status=str(OUT / 'status.json')), flush=True)
    elif action == 'supervise':
        supervise()
    else:
        try:
            Worker().run()
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            previous = base.read(OUT / 'Beauty/status.json') if (OUT / 'Beauty/status.json').exists() else {}
            base.write(OUT / 'Beauty/status.json', dict(previous, state='FAILED', updated_at=base.stamp(), error=detail))
            raise


if __name__ == '__main__':
    main()
