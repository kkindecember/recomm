"""User-authorized full cohorts, parallel GPUs, no wall-clock cost cutoff."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20 import lime_decode_screen_pilot as pilot
from experiment.phase20 import finalize_lime_decode_screen as finalizer

PARENT = pilot.OUT
TOP = PARENT.parent
OUT = TOP / 'full_parallel_v4'
MODULE = 'experiment.phase20.lime_decode_screen_full'
GPUS = {'Toys': base.GPU['Toys'], 'Beauty': 'GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09'}
base.OUT = pilot.OUT = OUT
finalizer.RUN = finalizer.BEAUTY = OUT


def read_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        j = row['sample_index']
        assert j not in rows, ('duplicate cached sample', j)
        rows[j] = row
    return rows


def cache_matches(row, sample, choices, split, selection=None):
    for key in ('user_id', 'target', 'history', 'split'):
        assert row[key] == sample[key], ('cached sample mismatch', key)
    assert row['split'] == split
    if split == 'evaluation':
        return selection is not None and all(
            list(choices[s]) == [selection[s]] for s in base.SIGNALS)
    return all(f'{s}_{a}' in row['arms'] and f'{s}_{a}_late' in row['arms']
               for s, alphas in choices.items() for a in alphas)


def prepare():
    OUT.mkdir(exist_ok=False)
    config = base.read(PARENT / 'config.json')
    config.update(gpus=GPUS, tuning_users=128, evaluation_users=512,
                  worker_timeout_seconds=None, resource_wait_seconds=None,
                  minimum_free_mib=12288, cost_gate_enabled=False,
                  user_authorization='GPU contention and longer Beauty runtime accepted; run properly to completion.',
                  takeover_policy='Keep active Toys pilot; cancel only obsolete Beauty queue and status coordinator.')
    base.write(OUT / 'config.json', config)
    for domain in base.DOMAINS:
        folder = OUT / domain
        folder.mkdir()
        for name in ('signals.npz', 'samples.json', 'data_config.json', 'preflight.json'):
            os.link(PARENT / domain / name, folder / name)
        samples = base.read(folder / 'samples.json')
        assert len(samples) == 643
        assert all(s['split'] == 'tuning' for s in samples[3:131])
        assert all(s['split'] == 'evaluation' for s in samples[131:643])
        assert set(s['user_id'] for s in samples[3:131]).isdisjoint(s['user_id'] for s in samples[131:643])
        base.write(folder / 'status.json', dict(state='PENDING', stage='pending', domain=domain,
                                               updated_at=base.stamp(), tuning_users=128, evaluation_users=512))
        base.write(folder / 'budget.json', dict(tuning_users=128, evaluation_users=512,
            selection_basis='Original frozen full cohorts restored by user authorization, not by observed evaluation effects.',
            wall_clock_timeout_seconds=None, selected_at=base.stamp()))
    (OUT / 'protocol.md').write_text((PARENT / 'protocol.md').read_text() + '''

## 用户授权恢复完整规模与分卡并行

用户明确表示“抢算力无所谓呀 beauty时间长一点也无所谓呀 要好好跑”。因此撤销之前由助手自行设置的两小时、13:20截止和按耗时缩减样本的规则。Toys使用GPU2，Beauty使用GPU4，允许与现有训练共享算力；原有两个基础训练继续。

两域均恢复最初预先按哈希排列的128名选参用户与512名评价用户，不重新抽人。模型、分数、引导公式、alpha搜索、评价和继续标准不变；仍为旧开发集诊断，不是正式test。

当前Toys的32/128人小规模任务先完成；完整任务复用其已完成的32名选参记录，再补齐原选参集合中剩余96名用户。完整128名选参完毕后才选择权重；旧128名评价结果仅在所选两个权重均相同时复用，否则重新生成。其余评价用户补齐到原定512人。不会根据小规模评价效果决定是否扩样、重抽用户或调整参数。

所有复用输入及结果记录来源与SHA256；结束后独立重算候选排序、指标、权重选择及命中得失。模型与数值一致性检查继续保留。耗时仅用于状态和预计时间，不再触发缩样或停止。内存不足等待资源，运行错误如实报告，不自动搜索新配方。只撤销旧Beauty排队及旧状态汇总进程，不终止基础训练或当前Toys工作进程。
''')
    manifest = base.read(PARENT / 'manifest.json')
    sources = [Path(__file__).resolve(), OUT / 'config.json', OUT / 'protocol.md',
               base.ROOT / 'experiment/phase20/finalize_lime_decode_screen.py',
               base.ROOT / 'experiment/phase20/audit_lime_decode_available.py',
               base.ROOT / 'experiment/phase20/run_lime_decode_screen_full.sh']
    for path in sources:
        manifest['sha256'][str(path.relative_to(base.ROOT))] = base.sha(path)
    manifest['revision'] = 'User-authorized full sample sizes and parallel GPUs; no wall-clock cutoff.'
    base.write(OUT / 'manifest.json', manifest)
    base.write(OUT / 'status.json', dict(state='PREPARED', updated_at=base.stamp()))
    print(json.dumps(dict(prepared=str(OUT), frozen_inputs=len(manifest['sha256']))), flush=True)


class FullWorker(pilot.PilotWorker):
    def __init__(self, domain):
        super().__init__(domain)
        # The base decoder has a numeric time comparison; infinity disables it
        # in memory only. Persisted config explicitly records no timeout (null).
        self.config['worker_timeout_seconds'] = float('inf')
        self.nt, self.ne = 128, 512
        self.cached_tuning = {}
        self.cached_evaluation = None
        self.parent_selection = None
        self.reused = {'tuning': 0, 'evaluation': 0}
        self.parent_hashes = {}
        if domain == 'Toys':
            parent = PARENT / domain
            for name in ('tuning.jsonl', 'evaluation.jsonl', 'selection.json'):
                path = parent / name
                if path.exists():
                    self.parent_hashes[str(path.relative_to(base.ROOT))] = base.sha(path)
            if (parent / 'tuning.jsonl').exists():
                self.cached_tuning = read_rows(parent / 'tuning.jsonl')
            if (parent / 'selection.json').exists():
                self.parent_selection = base.read(parent / 'selection.json')['selected_alpha']
            base.write(self.folder / 'resume_manifest.json', dict(sha256=self.parent_hashes,
                cached_tuning_users=len(self.cached_tuning), evaluation_reuse_requires_same_selection=True,
                frozen_at=base.stamp()))

    def smoke(self):
        import torch
        self.stage = 'consistency_smoke'
        self.status()
        checks, timings = [], []
        for j in range(3):
            ctx = self.context(j)
            self.model._diff_cross_cache_reuse = False
            native = self.decode(ctx, native_entry=True, record=False)
            self.model._diff_cross_cache_reuse = True
            zero = self.decode(ctx)
            timings.append(self.seconds[-1])
            assert zero['paths'] == native['paths']
            np.testing.assert_allclose(zero['scores'], native['scores'], atol=5e-6, rtol=5e-6)
            q = self.index.conditional_logs(self.signals['lime'][j])
            self.decode(ctx, q, .3)
            timings.append(self.seconds[-1])
            zero2 = self.decode(ctx, q, 0.)
            timings.append(self.seconds[-1])
            assert zero2['paths'] == zero['paths']
            np.testing.assert_array_equal(zero2['scores'], zero['scores'])
            checks.append(dict(user_id=self.samples[j]['user_id'], cross_cache_paths_exact=True,
                max_cross_cache_score_error=float(np.max(np.abs(zero['scores'] - native['scores']))),
                zero_strength_exact=True, path_scores_reconstructed=True))
            self.status(completed_smoke_users=j + 1)
            del ctx, native, zero, zero2
        longest = max(range(len(self.data)), key=lambda i: len(self.data.records[i]['history']))
        ctx = self.context(0, longest)
        self.decode(ctx)
        timings.append(self.seconds[-1])
        del ctx
        assert self.digest() == self.initial_digest
        base.write(self.folder / 'smoke.json', dict(checks=checks, max_history_checked=True,
            optimized_decode_seconds=timings, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            initial_digest=self.initial_digest, cost_gate_enabled=False))
        self.status(tuning_users=self.nt, evaluation_users=self.ne, cost_gate_enabled=False)

    def one(self, j, choices, split):
        if split == 'tuning':
            cached = self.cached_tuning
        else:
            # Read prior evaluation only after the full tuning selection is saved.
            assert (self.folder / 'selection.json').exists()
            if self.cached_evaluation is None:
                path = PARENT / self.domain / 'evaluation.jsonl'
                self.cached_evaluation = read_rows(path) if self.domain == 'Toys' and path.exists() else {}
            cached = self.cached_evaluation
        if j in cached and cache_matches(cached[j], self.samples[j], choices, split, self.parent_selection):
            self.reused[split] += 1
            return cached[j]
        return super().one(j, choices, split)

    def run(self):
        super().run()
        for path, digest in self.parent_hashes.items():
            assert base.sha(base.ROOT / path) == digest, ('changed parent cache', path)
        summary = base.read(self.folder / 'summary.json')
        summary.update(reused_parent_users=self.reused, wall_clock_timeout_seconds=None,
                       protocol_revision='User-requested original full cohorts restored.')
        base.write(self.folder / 'summary.json', summary)
        base.write(self.folder / 'cache_reuse.json', dict(reused_users=self.reused,
            parent_sha256=self.parent_hashes, checked_at=base.stamp()))


def live_owned(pid, module, action):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:
        return False
    return '-m' in args and args[args.index('-m') + 1:args.index('-m') + 3] == [module, action]


def stop_owned(launch_path, module, action):
    if not launch_path.exists():
        return None
    pid = base.read(launch_path)['pid']
    if not live_owned(pid, module, action):
        return dict(pid=pid, stopped=False, reason='not running with expected identity')
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not live_owned(pid, module, action):
            return dict(pid=pid, stopped=True)
        time.sleep(.1)
    raise RuntimeError(f'Old coordinator did not exit: {pid}')


def supervise():
    config = base.read(OUT / 'config.json')
    active, done, pending = {}, {}, set(base.DOMAINS)
    started = time.monotonic()
    while pending or active:
        for domain in list(pending):
            if domain == 'Toys':
                parent = base.read(PARENT / 'Toys/status.json')
                if parent['state'] not in ('COMPLETED', 'FAILED') or live_owned(
                        parent['pid'], 'experiment.phase20.lime_decode_screen_pilot', 'worker'):
                    continue
            if base.gpu_free(GPUS[domain]) < config['minimum_free_mib']:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[domain], PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            with (OUT / domain / 'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--domain', domain],
                    cwd=base.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            active[domain] = child
            pending.remove(domain)
        for domain, child in list(active.items()):
            if child.poll() is not None:
                done[domain] = child.returncode
                del active[domain]
                if child.returncode and base.read(OUT / domain / 'status.json')['state'] not in ('FAILED',):
                    previous = base.read(OUT / domain / 'status.json')
                    base.write(OUT / domain / 'status.json', dict(previous, state='FAILED',
                        updated_at=base.stamp(), error=f'Worker exited with code {child.returncode}'))
        domains = {d: finalizer.progress(OUT / d) for d in base.DOMAINS}
        for domain in pending:
            domains[domain]['state'] = 'WAITING_FOR_PRIOR_TOYS' if domain == 'Toys' else 'WAITING_FOR_MEMORY'
            if domain == 'Toys':
                domains[domain]['prior_pilot'] = finalizer.progress(PARENT / 'Toys')
        status = dict(state='RUNNING', updated_at=base.stamp(), background=True,
            supervisor_pid=os.getpid(), mode='full_parallel_v4', domains=domains, gpus=GPUS,
            active={d:p.pid for d,p in active.items()}, pending=sorted(pending), completed=done,
            elapsed_seconds=time.monotonic() - started, wall_clock_timeout_seconds=None,
            results='screen_summary.json', report=str(finalizer.REPORT.relative_to(base.ROOT)),
            test_read=False, automatic_new_experiments=False)
        if not pending and not active:
            try:
                if all(rc == 0 for rc in done.values()):
                    pilot.verify()
                result = finalizer.finish(domains)
                n = len(result['domains'])
                status.update(state='COMPLETED' if n == 2 else 'PARTIAL' if n else 'INCOMPLETE',
                              decision=result['decision'], independent_audit=result['independent_audit'])
            except Exception:
                status.update(state='AUDIT_FAILED', error=traceback.format_exc())
        base.write(OUT / 'status.json', status)
        base.write(TOP / 'status.json', status)
        if pending or active:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'launch', 'supervise', 'worker'])
    parser.add_argument('--domain', choices=base.DOMAINS)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'launch':
        assert base.read(OUT / 'status.json')['state'] == 'PREPARED'
        queue = PARENT / 'beauty_gpu2'
        assert base.read(queue / 'status.json')['state'] == 'WAITING_FOR_TOYS_GPU2'
        stopped = [stop_owned(queue / 'launch.json', 'experiment.phase20.lime_decode_screen_beauty_gpu2', 'supervise'),
                   stop_owned(TOP / 'coordinator_launch.json', 'experiment.phase20.finalize_lime_decode_screen', 'run')]
        base.write(TOP / 'pre_full_parallel_status.json', base.read(TOP / 'status.json'))
        base.write(queue / 'status.json', dict(state='SUPERSEDED', updated_at=base.stamp(),
            replacement=str(OUT.relative_to(base.ROOT)), reason='User authorized GPU sharing and full runs without the old deadline.'))
        with (OUT / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', MODULE, 'supervise'], cwd=base.ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base.write(OUT / 'launch.json', dict(pid=child.pid, started_at=base.stamp(), stopped_obsolete_schedulers=stopped))
        print(json.dumps(dict(pid=child.pid, status=str(TOP / 'status.json'))), flush=True)
    elif args.action == 'supervise':
        supervise()
    else:
        try:
            FullWorker(args.domain).run()
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            previous = base.read(OUT / args.domain / 'status.json')
            base.write(OUT / args.domain / 'status.json', dict(previous, state='FAILED',
                updated_at=base.stamp(), error=detail))
            raise


if __name__ == '__main__':
    main()
