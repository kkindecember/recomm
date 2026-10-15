"""Resume the timed-out cache without changing the frozen scientific code."""
import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from experiment.phase20 import lime_partial_state as original
from experiment.phase20.lime_decode_screen import arrays, read, write, sha, stamp
from experiment.phase20.lime_generation_trace import json_safe
from experiment.phase20.lime_partial_state_core import prefix_features

ROOT, OUT = original.ROOT, original.OUT
RECOVERY = OUT / 'restarts' / 'timeout_resume1'
MODULE = 'experiment.phase20.resume_lime_partial_state'
DOMAINS = ('Toys', 'Beauty')
CACHE_TIMEOUT = 6 * 3600
CPU_TIMEOUT = 2 * 3600


def verify_frozen():
    for name in ('manifest.json', 'analysis_code_frozen.json'):
        for rel, digest in read(OUT / name)['sha256'].items():
            assert sha(ROOT / rel) == digest, rel


def prepare():
    assert read(OUT / 'status.json')['state'] == 'FAILED'
    assert read(OUT / 'cpu_pipeline_status.json')['state'] == 'FAILED'
    verify_frozen()
    for domain in DOMAINS:
        assert read(OUT / domain / 'status.json')['state'] == 'FAILED'
        assert not (OUT / domain / 'heads').exists()
        assert not (OUT / domain / 'features.npz').exists()
        assert len(read(OUT / domain / 'smoke.json')['checks']) == 3
    RECOVERY.mkdir(parents=True, exist_ok=False)
    saved = {}
    for parent in ('', *DOMAINS):
        folder = RECOVERY / 'previous' / parent
        folder.mkdir(parents=True, exist_ok=True)
        names = ('status.json', 'failure.txt', 'cpu_pipeline_status.json',
                 'cpu_pipeline_failure.txt', 'config.json', 'launch.json',
                 'cpu_background_launch.json', 'smoke.json', 'traces.jsonl.gz')
        for name in names:
            src = OUT / parent / name
            if src.exists():
                dst = folder / name
                shutil.copy2(src, dst)
                assert sha(src) == sha(dst)
                saved[str(dst.relative_to(ROOT))] = sha(dst)
    for domain in DOMAINS:
        (RECOVERY / domain).mkdir()
    write(RECOVERY / 'config.json', dict(created_at=stamp(),
        reason='User requested restart after the 45-minute timeout.',
        cache_timeout_seconds=CACHE_TIMEOUT, cpu_step_timeout_seconds=CPU_TIMEOUT,
        gpus=original.GPUS, scientific_protocol_unchanged=True,
        raw_records_reused=True, automatic_retry=False, test_read=False,
        sha256=saved, runner_sha256=sha(Path(__file__))))
    write(RECOVERY / 'status.json', dict(state='PREPARED', updated_at=stamp()))
    print(json.dumps(dict(state='PREPARED', recovery=str(RECOVERY))), flush=True)


class ResumeWorker(original.Worker):
    def status(self, state='RUNNING', **kw):
        super().status(state, users_reused=getattr(self, 'reused', 0),
            cache_timeout_seconds=CACHE_TIMEOUT, recovery='timeout_resume1', **kw)

    def mapped(self, row, steps, features):
        sample_ix = int(self.old['sample_indices'][row])
        q = self.index.conditional_logs(self.fused[sample_ix])
        counts = {}
        for depth in (2, 3):
            values = prefix_features(steps, q, depth)
            keys = [tuple(self.catalog['decoder_paths'][int(item)-1][:depth+1])
                    for item in self.old['ids'][row]]
            assert all(key in values for key in keys)
            features[depth][row] = np.stack([values[key] for key in keys])
            counts[str(depth)] = dict(all_expansions=len(values),
                selected_distinct_prefixes=len(set(keys)))
        return counts

    def checkpoint(self, features):
        tmp = self.folder / 'features.partial.tmp.npz'
        n = self.users_completed
        np.savez_compressed(tmp, completed=n, users=self.old['users'][:n],
            **{f'prefix{d}': v[:n] for d, v in features.items()})
        tmp.replace(self.folder / 'features.partial.npz')
        self.status()

    def run(self):
        from experiment.phase20.review_lime_partial_state import restore
        self.config = dict(self.config, worker_timeout_seconds=CACHE_TIMEOUT)
        self.stage = 'restore_saved_traces'
        self.status()
        n = len(self.old['users'])
        features = {d: np.full((n, 50, 4), np.nan, dtype='float32') for d in (2, 3)}
        last = None
        with gzip.open(RECOVERY / 'previous' / self.domain / 'traces.jsonl.gz', 'rt') as f:
            for row, line in enumerate(f):
                rec = restore(json.loads(line))
                assert rec['row'] == row
                assert rec['sample_index'] == int(self.old['sample_indices'][row])
                assert rec['user_id'] == self.old['users'][row]
                assert self.samples[rec['sample_index']]['role'] != 'gate_calibration'
                assert self.mapped(row, rec['steps'], features) == rec['counts']
                self.users_completed = self.reused = row + 1
                last = rec
                if self.users_completed % 100 == 0:
                    self.checkpoint(features)
                if time.monotonic() - self.started > CACHE_TIMEOUT:
                    raise TimeoutError('Six-hour recovery cache budget')
        assert last is not None and self.users_completed <= n
        self.checkpoint(features)
        # Check the resumed runtime against an already-saved actual trajectory.
        self.stage = 'resume_parity_smoke'
        self.status()
        actual = self.observed(self.context(last['sample_index']))
        assert json_safe(actual['steps']) == json_safe(last['steps'])
        write(self.folder / 'resume_smoke.json', dict(state='VERIFIED',
            user_id=last['user_id'], saved_steps_exact=True, reused_users=self.reused))
        self.stage = 'prefix_cache'
        self.status()
        with gzip.open(self.folder / 'traces.jsonl.gz', 'at', compresslevel=1) as log:
            for row in range(self.reused, n):
                if time.monotonic() - self.started > CACHE_TIMEOUT:
                    raise TimeoutError('Six-hour recovery cache budget')
                sample_ix = int(self.old['sample_indices'][row])
                sample = self.samples[sample_ix]
                assert sample['user_id'] == self.old['users'][row]
                assert sample['role'] != 'gate_calibration'
                partial = self.observed(self.context(sample_ix))
                counts = self.mapped(row, partial['steps'], features)
                log.write(json.dumps(json_safe(dict(row=row, user_id=sample['user_id'],
                    sample_index=sample_ix, steps=partial['steps'], checks=partial['checks'],
                    counts=counts)), allow_nan=False) + '\n')
                log.flush()
                self.users_completed = row + 1
                if self.users_completed % 25 == 0:
                    self.checkpoint(features)
        assert all(np.isfinite(v).all() for v in features.values())
        np.savez_compressed(self.folder / 'features.npz', users=self.old['users'],
            ids=self.old['ids'], sample_indices=self.old['sample_indices'],
            **{f'prefix{d}': v for d, v in features.items()})
        assert self.digest() == self.initial_digest
        elapsed = time.monotonic() - self.started
        old_start = datetime.fromisoformat(read(RECOVERY / 'previous' / 'launch.json')['started_at'])
        old_end = datetime.fromisoformat(read(RECOVERY / 'previous' / self.domain / 'status.json')['updated_at'])
        old_seconds = (old_end - old_start).total_seconds()
        write(self.folder / 'cache_summary.json', dict(state='COMPLETED',
            users=n, partial_decodes=n, full_smoke_decodes=3,
            resume_parity_decodes=1, users_reused=self.reused, users_new=n-self.reused,
            seconds=old_seconds+elapsed, original_attempt_seconds=old_seconds,
            resumed_seconds=elapsed, model_unchanged=True,
            model_digest=self.initial_digest, test_read=False, upstream_optimizer_steps=0))
        names = ('features.npz', 'traces.jsonl.gz', 'smoke.json',
                 'resume_smoke.json', 'cache_summary.json')
        write(self.folder / 'cache_manifest.json', dict(sha256={
            name: sha(self.folder / name) for name in names}))
        self.stage = 'prefix_cache_completed'
        self.status(state='COMPLETED')


def worker(domain):
    try:
        ResumeWorker(domain).run()
    except BaseException as error:
        (RECOVERY / domain / 'resume_failure.txt').write_text(traceback.format_exc())
        previous = read(OUT / domain / 'status.json')
        write(OUT / domain / 'status.json', dict(previous, state='FAILED',
            error=repr(error), updated_at=stamp()))
        raise


def env():
    return dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')


def supervise():
    verify_frozen()
    cfg = read(RECOVERY / 'config.json')
    assert sha(Path(__file__)) == cfg['runner_sha256']
    for rel, digest in cfg['sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for domain in DOMAINS:
        assert sha(OUT / domain / 'traces.jsonl.gz') == sha(
            RECOVERY / 'previous' / domain / 'traces.jsonl.gz')
    resource = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in resource.splitlines()}
    for domain, gpu in original.GPUS.items():
        assert free[gpu] >= 12288, (domain, free[gpu])
    processes = {}
    started = time.monotonic()
    for domain, gpu in original.GPUS.items():
        with (RECOVERY / domain / 'worker.log').open('x') as log:
            processes[domain] = subprocess.Popen([sys.executable, '-u', '-m', MODULE,
                'worker', '--domain', domain], cwd=ROOT,
                env=dict(env(), CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
    write(RECOVERY / 'workers.json', dict(pid=os.getpid(), started_at=stamp(),
        workers={d: p.pid for d, p in processes.items()}, resources=resource))
    while any(p.poll() is None for p in processes.values()):
        domains = {d: original.safe_read(OUT / d / 'status.json') for d in DOMAINS}
        progress = dict(state='RUNNING', stage='partial_state_cache_recovery',
            pid=os.getpid(), updated_at=stamp(), elapsed_seconds=time.monotonic()-started,
            domains=domains, alive={d: p.poll() is None for d, p in processes.items()},
            recovery='timeout_resume1', cache_timeout_seconds=CACHE_TIMEOUT)
        write(OUT / 'status.json', progress)
        write(RECOVERY / 'status.json', progress)
        if time.monotonic()-started > CACHE_TIMEOUT:
            for p in processes.values():
                if p.poll() is None:
                    p.terminate()
            for p in processes.values():
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise TimeoutError('Six-hour recovery hard timeout')
        time.sleep(15)
    codes = {d: p.wait() for d, p in processes.items()}
    assert all(code == 0 for code in codes.values()), codes
    write(OUT / 'status.json', dict(state='CACHE_COMPLETED', updated_at=stamp(),
        recovery='timeout_resume1', exit_codes=codes))
    phases = [(f'features_{d}', ['experiment.phase20.review_lime_partial_state',
        'features', '--domain', d]) for d in DOMAINS]
    phases += [('analysis', ['experiment.phase20.analyze_lime_partial_state']),
               ('fit_review', ['experiment.phase20.review_lime_partial_state', 'fits']),
               ('report', ['experiment.phase20.summarize_lime_partial_state'])]
    for stage, args in phases:
        state = dict(state='RUNNING', stage=stage, pid=os.getpid(), updated_at=stamp())
        write(OUT / 'cpu_pipeline_status.json', state)
        # report hashes recovery/status.json, so freeze it before that phase.
        if stage != 'report':
            write(RECOVERY / 'status.json', state)
        with (RECOVERY / f'{stage}.log').open('x') as log:
            p = subprocess.Popen([sys.executable, '-u', '-m', *args], cwd=ROOT,
                env=env(), stdout=log, stderr=subprocess.STDOUT)
            phase_start = time.monotonic()
            while p.poll() is None:
                write(OUT / 'cpu_pipeline_status.json', dict(state, child_pid=p.pid,
                    updated_at=stamp(), phase_seconds=time.monotonic()-phase_start))
                if time.monotonic()-phase_start > CPU_TIMEOUT:
                    p.terminate()
                    try:
                        p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    raise TimeoutError(f'Two-hour CPU stage budget: {stage}')
                time.sleep(15)
            assert p.returncode == 0, (stage, p.returncode)
        if stage == 'fit_review':
            write(RECOVERY / 'status.json', dict(state='COMPLETED', verified=True,
                updated_at=stamp(), report_pending=True))
    write(OUT / 'cpu_pipeline_status.json', dict(state='COMPLETED',
        pid=os.getpid(), updated_at=stamp(), seconds=time.monotonic()-started))


def launch():
    assert read(RECOVERY / 'status.json')['state'] == 'PREPARED'
    assert read(OUT / 'status.json')['state'] == 'FAILED'
    assert not (RECOVERY / 'background_launch.json').exists()
    assert not (RECOVERY / 'supervisor.log').exists()
    for domain in DOMAINS:
        write(OUT / domain / 'status.json', dict(state='PREPARED', users_completed=0,
            users_total=len(arrays(original.SOURCE / domain / 'data.npz')['users']),
            updated_at=stamp(), recovery='timeout_resume1'))
    write(OUT / 'status.json', dict(state='STARTING', recovery='timeout_resume1', updated_at=stamp()))
    write(OUT / 'cpu_pipeline_status.json', dict(state='WAITING', stage='cache_recovery', updated_at=stamp()))
    with (RECOVERY / 'supervisor.log').open('x') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'supervise'],
            cwd=ROOT, env=env(), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(RECOVERY / 'background_launch.json', dict(pid=p.pid, started_at=stamp(), detached=True))
    print(json.dumps(dict(pid=p.pid, status=str(OUT / 'status.json'), detached=True)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch', 'supervise', 'worker'))
    parser.add_argument('--domain', choices=DOMAINS)
    args = parser.parse_args()
    if args.action == 'worker':
        assert args.domain
        worker(args.domain)
    elif args.action == 'supervise':
        try:
            supervise()
        except BaseException as error:
            (RECOVERY / 'failure.txt').write_text(traceback.format_exc())
            for path in (OUT / 'status.json', OUT / 'cpu_pipeline_status.json', RECOVERY / 'status.json'):
                write(path, dict(state='FAILED', updated_at=stamp(), error=repr(error), recovery='timeout_resume1'))
            raise
    elif args.action == 'prepare':
        prepare()
    else:
        launch()


if __name__ == '__main__':
    main()
