"""Recover completed prefix caches after the runtime timestamp failure.

Scientific collection, fitting, evaluation and review code remain frozen.
This coordinator uses CPU only and archives previous failure records first.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/lime_partial_state_20260922_v1'
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
OLD = OUT / 'restarts/timeout_resume1'
RECOVERY = OUT / 'restarts/metadata_recovery_20260923'
DOMAINS = ('Toys', 'Beauty')
PHASE_TIMEOUT = 7200


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def parse_stamp(value):
    # The recorded Python runtime rejects fromisoformat's +0800 offset.
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S%z')


def heartbeat(stage, **extra):
    write(OUT / 'cpu_pipeline_status.json', dict(
        state='RUNNING', stage=stage, updated_at=stamp(), pid=os.getpid(),
        recovery='metadata_recovery_20260923', **extra))


def prepare():
    import numpy as np
    assert read(OUT / 'status.json')['state'] == 'FAILED'
    assert read(OUT / 'cpu_pipeline_status.json')['state'] == 'FAILED'
    assert not (OUT / 'analysis_manifest.json').exists()
    assert not (OUT / 'summary.json').exists()
    for domain in DOMAINS:
        assert not (OUT / domain / 'heads').exists()
        assert not (OUT / domain / 'cache_manifest.json').exists()
        assert read(OUT / domain / 'status.json')['state'] == 'FAILED'
    RECOVERY.mkdir(exist_ok=False)
    archived = {}
    for parent in ('', *DOMAINS):
        dest = RECOVERY / 'previous' / parent
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('status.json', 'failure.txt', 'cpu_pipeline_status.json',
                     'cpu_pipeline_failure.txt', 'cpu_background_launch.json'):
            source = OUT / parent / name
            if source.exists():
                target = dest / name
                shutil.copy2(source, target)
                assert sha(source) == sha(target)
                archived[str(target.relative_to(ROOT))] = sha(target)
    write(RECOVERY / 'config.json', dict(
        created_at=stamp(), reason='User authorized continuing the method search and recovering this interrupted check.',
        scientific_protocol_unchanged=True, cpu_only=True, new_generation=False,
        automatic_retry=False, test_read=False, phase_timeout_seconds=PHASE_TIMEOUT,
        runner_sha256=sha(Path(__file__)), archived_sha256=archived))
    heartbeat('verify_frozen_inputs')
    for name in ('manifest.json', 'analysis_code_frozen.json'):
        for rel, digest in read(OUT / name)['sha256'].items():
            assert sha(ROOT / rel) == digest, rel
    assert sha(ROOT / 'experiment/phase20/resume_lime_partial_state.py') == read(OLD / 'config.json')['runner_sha256']
    resumed_at = parse_stamp(read(OLD / 'workers.json')['started_at'])
    old_started_at = parse_stamp(read(OLD / 'previous/launch.json')['started_at'])
    receipts = {}
    for domain in DOMAINS:
        heartbeat('finalize_cached_metadata', domain=domain)
        folder = OUT / domain
        failure = (OLD / domain / 'resume_failure.txt').read_text()
        assert 'line 153, in run' in failure and 'datetime.fromisoformat' in failure
        assert "ValueError: Invalid isoformat string:" in failure
        # In the frozen runner, finite-value and model-digest assertions precede
        # the failed timestamp statement. Do not claim a fresh GPU digest check.
        with np.load(folder / 'features.npz', allow_pickle=False) as cached, np.load(SOURCE / domain / 'data.npz', allow_pickle=False) as original:
            for key in ('users', 'ids', 'sample_indices'):
                np.testing.assert_array_equal(cached[key], original[key])
            n = len(cached['users'])
            for depth in (2, 3):
                value = cached['prefix%d' % depth]
                assert value.shape == (n, 50, 4) and np.isfinite(value).all()
        smoke = read(folder / 'smoke.json')['checks']
        assert len(smoke) == 3 and all(x['partial_steps_exact'] and x['final_cache_exact'] for x in smoke)
        resume = read(folder / 'resume_smoke.json')
        assert resume['state'] == 'VERIFIED' and resume['saved_steps_exact']
        previous = read(RECOVERY / 'previous' / domain / 'status.json')
        original_seconds = (parse_stamp(read(OLD / 'previous' / domain / 'status.json')['updated_at']) - old_started_at).total_seconds()
        resumed_seconds = (parse_stamp(previous['updated_at']) - resumed_at).total_seconds()
        assert original_seconds >= 0 and resumed_seconds >= 0
        summary = dict(state='COMPLETED', users=n, partial_decodes=n, full_smoke_decodes=3,
            resume_parity_decodes=1, users_reused=resume['reused_users'], users_new=n-resume['reused_users'],
            seconds=original_seconds+resumed_seconds, original_attempt_seconds=original_seconds,
            resumed_seconds=resumed_seconds, timing_basis='wall timestamps including initialization and failure handling',
            model_unchanged=True, model_digest=read(SOURCE / domain / 'cache_summary.json')['model_digest'],
            digest_evidence='Prior frozen runner passed its post-cache digest assertion before timestamp failure; no new GPU check.',
            test_read=False, upstream_optimizer_steps=0, metadata_recovered_at=stamp())
        write(folder / 'cache_summary.json', summary)
        names = ('features.npz', 'traces.jsonl.gz', 'smoke.json', 'resume_smoke.json', 'cache_summary.json')
        hashes = {name: sha(folder / name) for name in names}
        write(folder / 'cache_manifest.json', dict(sha256=hashes))
        write(folder / 'status.json', dict(state='COMPLETED', domain=domain, stage='cache_metadata_recovered',
            users_completed=n, users_total=n, pending_feature_verification=True,
            updated_at=stamp(), recovery='metadata_recovery_20260923', test_read=False))
        receipts[domain] = dict(users=n, sha256=hashes, failure_sha256=sha(OLD / domain / 'resume_failure.txt'))
    write(RECOVERY / 'cache_recovery_receipt.json', dict(state='CACHE_RECOVERED', domains=receipts,
        pending_full_trace_replay=True, science_unchanged=True, new_generation=False))
    write(OUT / 'status.json', dict(state='CACHE_COMPLETED', updated_at=stamp(),
        recovery='metadata_recovery_20260923', pending_feature_verification=True, test_read=False))


def run_phase(stage, module_args):
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
    with (RECOVERY / (stage + '.log')).open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', '-m'] + module_args,
                                cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            heartbeat(stage, child_pid=proc.pid, child_alive=True, phase_seconds=elapsed)
            if elapsed > PHASE_TIMEOUT:
                print('Stopping only this recovery child at the declared phase timeout: ' + stage, flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                raise TimeoutError('Two-hour recovery CPU phase budget: ' + stage)
            time.sleep(15)
        assert proc.returncode == 0, (stage, proc.returncode)
    print(json.dumps(dict(stage=stage, state='COMPLETED', seconds=time.monotonic()-started)), flush=True)


def main():
    started = time.monotonic()
    prepare()
    phases = [('features_' + d, ['experiment.phase20.review_lime_partial_state', 'features', '--domain', d]) for d in DOMAINS]
    phases += [('analysis', ['experiment.phase20.analyze_lime_partial_state']),
               ('fit_review', ['experiment.phase20.review_lime_partial_state', 'fits'])]
    for stage, args in phases:
        run_phase(stage, args)
    # Keep recovery files immutable while the frozen report hashes them.
    write(RECOVERY / 'status.json', dict(state='COMPLETED', verified=True, updated_at=stamp(), report_pending=True))
    run_phase('report', ['experiment.phase20.summarize_lime_partial_state'])
    write(OUT / 'cpu_pipeline_status.json', dict(state='COMPLETED', updated_at=stamp(),
        pid=os.getpid(), recovery='metadata_recovery_20260923', seconds=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if RECOVERY.exists():
            (RECOVERY / 'failure.txt').write_text(traceback.format_exc())
            write(OUT / 'cpu_pipeline_status.json', dict(state='FAILED', updated_at=stamp(), error=repr(error),
                  recovery='metadata_recovery_20260923', no_automatic_retry=True))
        raise
