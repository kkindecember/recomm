"""Resume resource-stopped expert checks without modifying frozen experiments."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from experiment.phase21 import text_contribution as runner

ROOT = runner.ROOT
OUT = ROOT / 'artifacts/phase21/expert_resource_recovery_20260923_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.3E_服务器变慢后的贡献检查恢复_20260923.md'
MODULE = 'experiment.phase21.expert_resource_recovery'
GROUPS = {
    'removal': ('other_expert_contribution', 'other_expert_contribution_20260923_v1'),
    'mechanism': ('expert_mechanism', 'expert_mechanism_20260923_v1'),
}
GPU_POOL = [
    'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253',
    'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2',
    'GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc',
    'GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059',
    'GPU-7952ad6e-2db7-fb2e-581b-34be56287640',
    'GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09',
    'GPU-07374397-b213-09c1-2988-718acb6ed231',
]
read, write, sha, stamp = runner.read, runner.write, runner.sha, runner.stamp


def original(group):
    return ROOT / 'artifacts/phase21' / GROUPS[group][1]


def prepare():
    OUT.mkdir(exist_ok=False)
    write(OUT / 'status.json', dict(state='PREPARING', updated_at=stamp(), pid=os.getpid()))
    config = dict(max_parallel=2, worker_timeout_seconds=14400, supervisor_timeout_seconds=86400,
                  minimum_free_mib=16384, gpu_pool=GPU_POOL, test_read=False,
                  forecast_policy='warning_only', stall_warning_seconds=1800)
    write(OUT / 'config.json', config)
    shutil.copy2(PLAN, OUT / 'protocol.md')
    frozen, jobs, reused = {}, [], []
    for group in GROUPS:
        old = original(group); group_out = OUT / group; group_out.mkdir()
        prior = read(old / 'manifest.json')['sha256']
        for rel, digest in prior.items():
            if rel in frozen: assert frozen[rel] == digest, rel
            frozen[rel] = digest
        old_cfg = read(old / 'config.json')
        cfg = dict(old_cfg, worker_timeout_seconds=config['worker_timeout_seconds'],
                   forecast_policy='warning_only', recovered_from=str(old.relative_to(ROOT)))
        write(group_out / 'config.json', cfg)
        (group_out / 'shared').symlink_to(old / 'shared', target_is_directory=True)
        for arm in old_cfg['arms']:
            arm_out = group_out / arm; arm_out.mkdir()
            write(arm_out / 'config.json', cfg)
            for domain in runner.DOMAINS:
                src = old / arm / domain; dst = arm_out / domain
                previous = read(src / 'status.json')
                if previous['state'] == 'COMPLETED':
                    assert read(src / 'summary.json')['model_unchanged']
                    dst.symlink_to(src, target_is_directory=True)
                    reused.append(dict(group=group, arm=arm, domain=domain, source=str(src)))
                    for name in ('status.json', 'summary.json', 'predictions.jsonl', 'ranks.npz'):
                        frozen[str((src / name).relative_to(ROOT))] = sha(src / name)
                    continue
                assert previous['state'] in ('RESOURCE_BUDGET_STOP', 'PREPARED'), previous
                assert not (src / 'predictions.jsonl').exists(), 'Partial generation requires explicit continuation'
                dst.mkdir()
                for name in ('samples.json', 'data_config.json', 'catalog.json'):
                    (dst / name).symlink_to(src / name)
                for name in ('gate.pt', 'fused.npy', 'gate_prediction.npz', 'fit_checks.json',
                             'signal_audit.json', 'mechanism_scores.npy'):
                    if (src / name).exists():
                        frozen[str((src / name).relative_to(ROOT))] = sha(src / name)
                write(dst / 'status.json', dict(state='QUEUED', group=group, arm=arm, domain=domain,
                                                updated_at=stamp(), previous_state=previous['state']))
                jobs.append(dict(group=group, arm=arm, domain=domain))
    # Finish both removal checks before starting the deeper controls.
    jobs.sort(key=lambda j: (j['group'] != 'removal', j['domain'] != 'Toys', j['arm']))
    for p in (Path(__file__), PLAN, ROOT / 'experiment/phase21/run_expert_resource_recovery.sh',
              ROOT / 'experiment/phase21/test_expert_resource_recovery.py', OUT / 'config.json'):
        frozen[str(p.relative_to(ROOT))] = sha(p)
    for p in OUT.glob('*/*/config.json'):
        frozen[str(p.relative_to(ROOT))] = sha(p)
    write(OUT / 'jobs.json', dict(jobs=jobs, reused_completed=reused))
    frozen[str((OUT / 'jobs.json').relative_to(ROOT))] = sha(OUT / 'jobs.json')
    # Verify one deduplicated set; progress remains visible during slow storage reads.
    for count, (rel, digest) in enumerate(frozen.items(), 1):
        assert sha(ROOT / rel) == digest, rel
        if count % 10 == 0:
            write(OUT / 'status.json', dict(state='VERIFYING_INPUTS', verified=count, total=len(frozen),
                                           updated_at=stamp(), pid=os.getpid()))
    write(OUT / 'manifest.json', dict(sha256=frozen, created_at=stamp()))
    write(OUT / 'status.json', dict(state='PREPARED', jobs=len(jobs), reused_completed=len(reused),
                                   updated_at=stamp(), test_read=False))


class RecoveryMixin:
    def fit_signals(self):
        src = original(self.recovery_group) / self.arm / self.domain
        names = ('gate.pt', 'fused.npy', 'gate_prediction.npz', 'fit_checks.json')
        if not all((src / name).exists() for name in names):
            return super().fit_signals()
        import torch
        assert read(src / 'fit_checks.json')['state'] == 'PASSED'
        saved = torch.load(src / 'gate.pt', map_location='cpu')
        assert saved['arm'] == self.arm and not saved['development_labels_used']
        expected_users = sorted(s['user_id'] for s in self.samples if s['role'] == 'gate_calibration')
        assert saved['fit_users'] == expected_users
        self.fit_steps = saved['optimizer_steps']
        assert self.fit_steps == dict(Toys=12, Beauty=16)[self.domain]
        self.fused = np.load(src / 'fused.npy', mmap_mode='r')
        assert self.fused.shape == (len(self.samples), self.raw.shape[-1])
        for start in range(0, len(self.fused), 64): assert np.isfinite(self.fused[start:start+64]).all()
        for name in names + ('signal_audit.json', 'mechanism_scores.npy'):
            if (src / name).exists(): (self.folder / name).symlink_to(src / name)
        write(self.folder / 'cache_reuse.json', dict(source=str(src), fusion_optimizer_steps_this_attempt=0,
              prior_fusion_optimizer_steps=self.fit_steps, manifest_verified=True, updated_at=stamp()))
        self.stage = 'reused_frozen_fusion'; self.status()

    def smoke_control(self):
        self.stage = 'native_and_zero_smoke'; self.status()
        longest = max(self.development, key=lambda j: len(self.samples[j]['history']))
        js = list(dict.fromkeys([longest, *self.development]))[:3]
        checks, times = [], []
        for j in js:
            started = time.monotonic(); sample = self.samples[j]
            ctx = self.context(j); q = self.index.conditional_logs(self.fused[j])
            native = self.decode(ctx)
            native['guide'].rescore(native['paths'], q, .3)
            guided = self.decode(ctx, q, .3)
            times.append(time.monotonic() - started)
            plain = self.decode(ctx, native_entry=True, record=False)
            zero = self.decode(ctx, q, 0.)
            assert plain['paths'] == native['paths'] == zero['paths']
            np.testing.assert_array_equal(plain['scores'], native['scores'])
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            ref = self.references[sample['user_id']]
            assert native['paths'] == ref['paths']
            np.testing.assert_array_equal(native['scores'], ref['native'])
            checks.append(dict(user=sample['user_id'], native_and_zero_exact=True, old_native_exact=True))
            del ctx, plain, native, zero, guided
        forecast = float(np.median(times) * len(self.development) * 1.25)
        write(self.folder / 'smoke.json', dict(state='PASSED', checks=checks,
              formal_pair_seconds=times, estimated_formal_seconds_with_margin=forecast,
              margin=1.25, warning='SLOW_ESTIMATE' if forecast > 4500 else None,
              forecast_policy='warning_only', longest_history_checked=True))
        self.status(estimated_formal_seconds=forecast, slow_estimate=forecast > 4500)


def bind(group):
    mod = importlib.import_module('experiment.phase21.' + GROUPS[group][0])
    mod.OUT = OUT / group
    title = 'SASRec与ItemCF贡献检查恢复' if group == 'removal' else 'SASRec与ItemCF机制检查恢复'
    mod.REPORT = ROOT / f'report/第二十一阶段/{title}_20260923.md'
    mod.bind_runtime()
    return mod


def worker(group, arm, domain):
    folder = OUT / group / arm / domain
    quit_heartbeat = threading.Event()
    def heartbeat():
        while not quit_heartbeat.is_set():
            write(folder / 'heartbeat.json', dict(pid=os.getpid(), updated_at=stamp(),
                                                  liveness_only=True))
            quit_heartbeat.wait(30)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        manifest = read(OUT / 'manifest.json')['sha256']
        source = str((original(group) / arm / domain).relative_to(ROOT)) + '/'
        for rel, digest in manifest.items():
            if rel.startswith(('experiment/', 'GRAM/src/', 'third_party/', source)) or rel.endswith('/config.json'):
                assert sha(ROOT / rel) == digest, rel
        cfg = read(folder / 'data_config.json')
        assert sha(ROOT / cfg['parent_checkpoint']) == cfg['parent_sha256']
        bind(group)
        cls = type('RecoveredWorker', (RecoveryMixin, runner.Worker), {'recovery_group': group})
        cls(arm, domain).run()
    except BaseException as error:
        (folder / 'failure.txt').write_text(traceback.format_exc())
        state = 'RESOURCE_TIME_LIMIT' if isinstance(error, TimeoutError) else 'FAILED'
        write(folder / 'status.json', dict(state=state, error=repr(error), group=group, arm=arm,
              domain=domain, updated_at=stamp(), pid=os.getpid(), test_read=False))
        raise
    finally:
        quit_heartbeat.set(); thread.join(timeout=2)


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def snapshot(job):
    return read(OUT / job['group'] / job['arm'] / job['domain'] / 'status.json')


def supervise():
    cfg = read(OUT / 'config.json'); jobs = read(OUT / 'jobs.json')['jobs']
    pending = list(jobs); active = {}; failures = {}; done = []; summaries = {}; launches = []
    started = time.monotonic()
    try:
        while pending or active:
            now = time.monotonic()
            if now - started > cfg['supervisor_timeout_seconds']: raise TimeoutError('Queue hard limit: 24h')
            for key, rec in list(active.items()):
                process, job = rec['process'], rec['job']
                if process.poll() is None and now - rec['start'] > cfg['worker_timeout_seconds']:
                    stop_owned(process)
                    write(OUT / job['group'] / job['arm'] / job['domain'] / 'status.json',
                          dict(state='RESOURCE_TIME_LIMIT', reason='Actual wall time exceeded 4h', updated_at=stamp()))
                code = process.poll()
                if code is not None:
                    del active[key]
                    status = snapshot(job)
                    if code == 0 and status['state'] == 'COMPLETED': done.append(job)
                    else: failures[key] = dict(job=job, exit_code=code, status=status)
            resource_error = None
            if pending and len(active) < cfg['max_parallel']:
                try:
                    output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                        '--format=csv,noheader,nounits'], text=True, timeout=60)
                    free = {x.split(',')[0].strip(): int(x.split(',')[1]) for x in output.splitlines()}
                except (subprocess.SubprocessError, ValueError) as error:
                    free = {}; resource_error = repr(error)
                busy = {r['gpu'] for r in active.values()}
                available = sorted((g for g in cfg['gpu_pool'] if g not in busy and
                                    free.get(g, 0) >= cfg['minimum_free_mib']), key=lambda g: -free[g])
                for job in list(pending):
                    if len(active) >= cfg['max_parallel'] or not available: break
                    key = '/'.join(job[k] for k in ('group', 'arm', 'domain'))
                    if job['domain'] == 'Beauty':
                        dep = read(OUT / job['group'] / job['arm'] / 'Toys/status.json')['state']
                        if dep in ('FAILED', 'RESOURCE_TIME_LIMIT'):
                            write(OUT / key / 'status.json', dict(state='BLOCKED_BY_PREVIOUS_DOMAIN', updated_at=stamp()))
                            failures[key] = dict(job=job, reason='Previous domain incomplete'); pending.remove(job)
                            continue
                        if dep != 'COMPLETED': continue
                    gpu = available.pop(0); env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                    folder = OUT / key
                    write(folder / 'status.json', dict(state='STARTING', updated_at=stamp(), gpu=gpu))
                    with (folder / 'worker.log').open('x') as handle:
                        process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker',
                            '--group', job['group'], '--arm', job['arm'], '--domain', job['domain']],
                            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = dict(process=process, job=job, gpu=gpu, start=time.monotonic())
                    launches.append(dict(job=job, pid=process.pid, gpu=gpu, launched_at=stamp()))
                    write(OUT / 'launch.json', dict(supervisor_pid=os.getpid(), launches=launches))
                    pending.remove(job)
            for group in GROUPS:
                if group in summaries: continue
                relevant = [j for j in jobs if j['group'] == group]
                if all(snapshot(j)['state'] == 'COMPLETED' for j in relevant):
                    log = OUT / group / 'summarize.log'
                    with log.open('x') as handle:
                        result = subprocess.run([sys.executable, '-u', '-m', MODULE, 'summarize', '--group', group],
                                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=300)
                    summaries[group] = 'COMPLETED' if result.returncode == 0 else 'FAILED'
            active_status = {}
            for key, rec in active.items():
                s = snapshot(rec['job']); p = OUT / key / 'status.json'
                active_status[key] = dict(pid=rec['process'].pid, gpu=rec['gpu'], status=s,
                    warning='STALL_WARNING' if time.time() - p.stat().st_mtime > cfg['stall_warning_seconds'] else None)
            write(OUT / 'status.json', dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES',
                  pid=os.getpid(), updated_at=stamp(), active=active_status, pending=pending, completed=done,
                  failures=failures, summaries=summaries, resource_query_error=resource_error,
                  elapsed_seconds=now-started, test_read=False))
            if pending or active: time.sleep(15)
        terminal = 'COMPLETED'
        if failures or 'FAILED' in summaries.values():
            genuine = 'FAILED' in summaries.values() or any(
                f.get('status', {}).get('state', 'BLOCKED_BY_PREVIOUS_DOMAIN' if 'reason' in f else 'UNKNOWN')
                not in ('RESOURCE_TIME_LIMIT', 'RESOURCE_BUDGET_STOP', 'BLOCKED_BY_PREVIOUS_DOMAIN')
                for f in failures.values())
            terminal = 'FAILED' if genuine else 'RESOURCE_STOPPED'
        write(OUT / 'status.json', dict(state=terminal, completed=done, failures=failures, summaries=summaries,
                                       updated_at=stamp(), test_read=False))
    except BaseException as error:
        for rec in active.values():
            stop_owned(rec['process'])
            job = rec['job']
            write(OUT / job['group'] / job['arm'] / job['domain'] / 'status.json',
                  dict(state='RESOURCE_TIME_LIMIT' if isinstance(error, TimeoutError) else 'SUPERVISOR_STOPPED',
                       reason=repr(error), updated_at=stamp()))
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        write(OUT / 'status.json', dict(state='RESOURCE_TIME_LIMIT' if isinstance(error, TimeoutError) else 'FAILED',
                                       error=repr(error), updated_at=stamp(), test_read=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'prepare', 'supervise', 'worker', 'summarize'])
    parser.add_argument('--group', choices=GROUPS); parser.add_argument('--arm'); parser.add_argument('--domain')
    args = parser.parse_args()
    if args.action in ('prepare', 'start'): prepare()
    if args.action in ('supervise', 'start'): supervise()
    if args.action == 'worker': worker(args.group, args.arm, args.domain)
    if args.action == 'summarize':
        result = bind(args.group).summarize()
        write(OUT / args.group / 'status.json', dict(state='COMPLETED', updated_at=stamp(), test_read=False))
        print(json.dumps(dict(state=result['state'], group=args.group)), flush=True)


if __name__ == '__main__': main()
