"""Resume the failed public download; reuse all six completed SASRec trainings."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now, gpu_candidates
from experiment.phase20.lime_comparison_job import stop_child, finish

ORIGINAL = ROOT/'artifacts/phase20/lime_comparison_20260920_v1'
OUT = ORIGINAL/'recovery_v1'
CONFIG = ROOT/'experiment/phase20/lime_comparison_20260920_v1.json'


def resume_asset_source():
    # Each attempt is immutable. Reuse the latest saved prefix, never overwrite it.
    return (ORIGINAL/'recovery_v1' if OUT.name == 'recovery_v2' else ORIGINAL)/'assets'


def validate_range_response(headers, size, offset, end, total):
    matches = re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$', headers)
    assert matches, 'Server did not return a byte-range response; do not append'
    first, last, count = map(int, matches[-1])
    assert (first, last, count) == (offset, end, total), (first, last, count, offset, end, total)
    assert 0 < size <= end-offset+1, size


def check_file(path, entry):
    assert path.stat().st_size == entry['size'], path.name
    digest = sha(path)
    if 'lfs' in entry:
        assert digest == entry['lfs']['oid'], 'Full model SHA256 mismatch'
    else:
        data = path.read_bytes()
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == entry['oid'], path.name
    return digest


def download():
    c = read(CONFIG)
    out = OUT/'assets'; out.mkdir()
    source = resume_asset_source()
    model = out/'bge-base-en-v1.5'; model.mkdir()
    shutil.copy2(source/'huggingface_tree.json', out/'huggingface_tree.json')
    tree = {r['path']: r for r in read(out/'huggingface_tree.json')}
    hashes = {}
    started = time.monotonic()
    child = None
    def status(state, **extra):
        write(out/'status.json', dict(state=state, updated_at=now(), pid=os.getpid(),
            elapsed_seconds=round(time.monotonic()-started, 1), **extra))
    try:
        status('VERIFYING', stage='校验并复用已下载的小文件')
        for path in (source/'bge-base-en-v1.5').rglob('*'):
            if not path.is_file() or path.name.endswith('.part'): continue
            name = str(path.relative_to(source/'bge-base-en-v1.5'))
            hashes[name] = check_file(path, tree[name])
            target = model/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        partial = model/'pytorch_model.bin.part'
        shutil.copy2(source/'bge-base-en-v1.5/pytorch_model.bin.part', partial)
        total = tree['pytorch_model.bin']['size']
        initial = partial.stat().st_size
        assert 0 < initial < total
        no_progress = 0
        for attempt in range(1, 193):
            offset = partial.stat().st_size
            if offset == total: break
            assert time.monotonic()-started < c['download_timeout_seconds'], 'Recovery download timeout'
            end = min(offset+4*1024**2, total)-1
            body = out/f'segment_{attempt:02d}.part'
            header = out/f'segment_{attempt:02d}.headers'
            # A distinct public URL per offset avoids cached redirects with stale ranges.
            url = f'https://huggingface.co/{c["encoder"]}/resolve/{c["encoder_revision"]}/pytorch_model.bin?download=true&range_start={offset}'
            with (out/f'segment_{attempt:02d}.log').open('x') as log:
                child = subprocess.Popen(['curl', '-L', '--fail', '--connect-timeout', '30',
                    '--max-time', '180', '--range', f'{offset}-{end}', '--dump-header', str(header),
                    '--output', str(body), url], stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    downloaded = body.stat().st_size if body.exists() else 0
                    elapsed = time.monotonic()-started
                    progressed = offset+downloaded-initial
                    status('DOWNLOADING', stage='分段断点续传BGE，完成后校验官方SHA256',
                        attempt=attempt, bytes=offset+downloaded, total_bytes=total,
                        reused_bytes=initial, segment_start=offset,
                        estimated_remaining_seconds=round((total-offset-downloaded)*elapsed/progressed) if progressed > 0 else None)
                    time.sleep(5)
            size = body.stat().st_size if body.exists() else 0
            accepted = False
            if size and child.returncode in (0, 18, 28, 56):
                validate_range_response(header.read_text(), size, offset, end, total)
                with partial.open('ab') as target, body.open('rb') as part:
                    shutil.copyfileobj(part, target)
                accepted = True
                no_progress = 0
            else:
                no_progress += 1
            with (out/'attempts.jsonl').open('a') as f:
                import json
                f.write(json.dumps(dict(attempt=attempt, offset=offset, end=end, bytes=size,
                    curl_exit=child.returncode, range_verified_and_appended=accepted, time=now()))+'\n')
            assert no_progress < 12, 'Twelve download attempts made no progress; inspect segment logs'
            if no_progress:
                backoff = min(15 * 2**(no_progress-1), 180)
                until = time.monotonic()+backoff
                while time.monotonic() < until:
                    assert time.monotonic()-started < c['download_timeout_seconds'], 'Recovery download timeout'
                    status('RETRY_WAIT', stage='网络失败后退避，保留已验证的下载前缀',
                        attempt=attempt, bytes=partial.stat().st_size, total_bytes=total,
                        reused_bytes=initial, consecutive_failures=no_progress,
                        curl_exit=child.returncode, retry_in_seconds=round(until-time.monotonic()))
                    time.sleep(min(5, max(0, until-time.monotonic())))
        assert partial.stat().st_size == total, 'Download attempt budget exhausted'
        status('VERIFYING', stage='校验完整BGE权重', bytes=total, total_bytes=total)
        hashes['pytorch_model.bin'] = check_file(partial, tree['pytorch_model.bin'])
        partial.replace(model/'pytorch_model.bin')
        pool = read(model/'1_Pooling/config.json')
        assert pool['pooling_mode_cls_token'] is True
        assert not any(v for k, v in pool.items() if k.startswith('pooling_mode_') and k != 'pooling_mode_cls_token')
        modules = read(model/'modules.json')
        assert [m['type'] for m in modules] == ['sentence_transformers.models.Transformer',
            'sentence_transformers.models.Pooling', 'sentence_transformers.models.Normalize']
        assert read(model/'sentence_bert_config.json')['max_seq_length'] == 512
        write(out/'manifest.json', dict(model=c['encoder'], revision=c['encoder_revision'], hashes=hashes,
            verified_lfs_and_git_objects=True, pooling=pool, modules=modules,
            resumed_bytes=initial, resumed_from=str(source.relative_to(ROOT)),
            original_failure='curl18 partial transfer; recovery_v1 SSL connection failures'))
        status('COMPLETED', stage='BGE下载与官方哈希校验完成', bytes=total, total_bytes=total)
    except BaseException as e:
        if child is not None and child.poll() is None: child.terminate()
        (out/'failure.txt').write_text(traceback.format_exc())
        status('FAILED', stage='下载恢复失败，已训练模型仍保留', error=repr(e))
        raise


def supervise():
    c = read(CONFIG)
    started = time.monotonic()
    jobs = {}
    states = {d: dict(state='WAITING_ASSET', stage='三个SASRec已完成，等待文本权重') for d in c['domains']}
    asset_log = (OUT/'assets_supervisor.log').open('x')
    asset = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_recovery', 'download', '--attempt', OUT.name],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=asset_log, stderr=subprocess.STDOUT, start_new_session=True)
    def status(state, **extra):
        detail = read(OUT/'assets/status.json') if (OUT/'assets/status.json').exists() else {}
        row = dict(state=state, updated_at=now(), supervisor_pid=os.getpid(), attempt=OUT.name,
            elapsed_seconds=round(time.monotonic()-started, 1), domains=states, assets=detail,
            active_workers=len(jobs), reused_completed_sasrec=6, new_sasrec_optimizer_steps=0,
            output=str(OUT.relative_to(ROOT)), test_read=False, **extra)
        write(OUT/'status.json', row)
        write(ORIGINAL/'status.json', row)
    def interrupted(signum, frame): raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        manifest = {}
        for p in ROOT.glob('experiment/phase20/*lime_comparison*'):
            if not p.is_file(): continue
            key = str(p.relative_to(ROOT)); manifest[key] = sha(p)
            target = OUT/'sources'/key; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
        write(OUT/'source_manifest.json', manifest)
        shutil.copy2(ORIGINAL/'protocol.md', OUT/'original_protocol.md')
        write(OUT/'recovery_protocol.json', dict(cause='Public BGE partial transfer, followed by SSL connection failures',
            reuse='Six complete SASRec checkpoints and both ItemCF score matrices; verify hashes and score probes',
            changed_scientific_settings=False, new_sasrec_training=False,
            download='4MiB HTTP ranges; verify Content-Range and final official SHA256; at most 192 requests/4 hours; 12 consecutive failures with 15-180 second backoff',
            resume_assets_from=str(resume_asset_source().relative_to(ROOT)),
            result_location=str(OUT.relative_to(ROOT)), original_failure_preserved=True))
        waiting_start = None
        while True:
            detail = read(OUT/'assets/status.json') if (OUT/'assets/status.json').exists() else {}
            if asset.poll() is not None and (asset.returncode != 0 or detail.get('state') != 'COMPLETED'):
                raise RuntimeError(f'Asset recovery failed; {detail}')
            if time.monotonic()-started > c['download_timeout_seconds'] and asset.poll() is None:
                stop_child(asset); raise TimeoutError('Download exceeded four hours')
            if detail.get('state') == 'COMPLETED':
                if waiting_start is None: waiting_start = time.monotonic()
                for s in states.values():
                    if s['state'] == 'WAITING_ASSET': s.update(state='QUEUED', stage='下载校验通过，分配GPU')
            for d, job in list(jobs.items()):
                path = OUT/d/'status.json'
                ds = read(path) if path.exists() else {}
                states[d].update(worker=ds, stage=ds.get('stage', '核对已训练模型'))
                if job['child'].poll() is None and time.monotonic()-job['started'] > c['domain_timeout_seconds']:
                    stop_child(job['child']); states[d]['error'] = 'Recovery worker timed out after 12 hours'
                code = job['child'].poll()
                if code is None: continue
                states[d].update(state='COMPLETED' if code == 0 and ds.get('state') == 'COMPLETED' else 'FAILED', worker_alive=False)
                if states[d]['state'] == 'FAILED': states[d]['error'] = ds.get('error', f'exit {code}')
                job['log'].close(); del jobs[d]
            pending = [d for d, s in states.items() if s['state'] == 'QUEUED']
            free = gpu_candidates() if pending else []
            used = {job['gpu']['uuid'] for job in jobs.values()}
            free = [g for g in free if g['free_mib'] >= c['minimum_free_mib'] and g['uuid'] not in used]
            for d in pending:
                if not free or len(jobs) >= c['max_parallel_workers']: break
                gpu = free.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                    PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log = (OUT/f'{d}.worker.log').open('x')
                child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_worker',
                    '--domain', d, '--resume-from', str(ORIGINAL/d), '--recovery-attempt', OUT.name], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                jobs[d] = dict(child=child, log=log, started=time.monotonic(), gpu=gpu)
                states[d].update(state='RUNNING', worker_pid=child.pid, worker_alive=True, gpu=gpu)
            if waiting_start is not None:
                for d, s in states.items():
                    if s['state'] == 'QUEUED' and time.monotonic()-waiting_start > c['resource_wait_seconds']:
                        s.update(state='FAILED', error='GPU wait exceeded one hour')
            if all(s['state'] in ('COMPLETED', 'FAILED') for s in states.values()): break
            status('RUNNING' if jobs else ('DOWNLOADING' if detail.get('state') != 'COMPLETED' else 'WAITING_GPU'),
                stage='恢复后半段：文本编码、融合与比较' if jobs else '下载恢复中，六个模型已保存')
            time.sleep(5)
        finish(OUT, states)
        for p, digest in manifest.items(): assert sha(ROOT/p) == digest, f'Source changed during recovery: {p}'
        status('COMPLETED' if all(s['state'] == 'COMPLETED' for s in states.values()) else 'FINISHED_WITH_FAILURES', stage=f'恢复任务结束，查看{OUT.name}/report.md')
    except BaseException as e:
        for d, job in jobs.items():
            stop_child(job['child']); job['log'].close(); states[d].update(state='STOPPED', worker_alive=False)
        stop_child(asset)
        (OUT/'failure.txt').write_text(traceback.format_exc())
        status('FAILED', stage='恢复异常；此前六个检查点保留', error=repr(e))
        raise
    finally:
        asset_log.close()


def main():
    global OUT
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'supervise', 'download'])
    p.add_argument('--attempt', choices=['recovery_v1', 'recovery_v2'], default='recovery_v2')
    args = p.parse_args()
    OUT = ORIGINAL/args.attempt
    if args.action == 'launch':
        assert read(ORIGINAL/'status.json')['state'] in ('FINISHED_WITH_FAILURES', 'FAILED')
        assert read(resume_asset_source()/'status.json')['state'] == 'FAILED'
        assert read(ORIGINAL/'recovery_tests.json')['state'] == 'PASSED'
        OUT.mkdir(exist_ok=False)
        shutil.copy2(ORIGINAL/'status.json', OUT/'original_failed_status.json')
        with (OUT/'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_recovery', 'supervise', '--attempt', OUT.name],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(OUT/'launch.json', dict(pid=child.pid, created_at=now()))
        print(f'Recovery PID {child.pid}; same status path: {ORIGINAL}/status.json', flush=True)
    elif args.action == 'download': download()
    else: supervise()


if __name__ == '__main__':
    main()
