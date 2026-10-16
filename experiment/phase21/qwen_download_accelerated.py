"""Transport-only Qwen resume: verified ModelScope weights, no proxy.

Uses exactly the frozen Hugging Face revisions/content hashes. Existing partial
files remain in Hugging Face's cache layout; no encoder or experiment change.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from urllib.parse import quote
from huggingface_hub._local_folder import get_local_download_paths, write_download_metadata
from experiment.phase21.qwen_text_encoder import OUT, ROOT, config, write, sha

TRANSPORT = OUT/'transport_20261015'


def matches(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    if entry.get('lfs'):
        return sha(path) == entry['lfs']['sha256']
    digest = hashlib.sha1(f'blob {entry["size"]}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest() == entry['blobId']


def main():
    cfg = config()
    assert json.loads((TRANSPORT/'handoff.json').read_text())['old_download_stopped']
    # Applies only to this transport process and its child curl commands.
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        os.environ.pop(key, None)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    tasks = []
    for variant, spec in cfg['qwen_models'].items():
        metadata = json.loads((TRANSPORT/(variant+'_hf.json')).read_text())
        assert metadata['sha'] == spec['revision']
        ms = json.loads((TRANSPORT/(variant+'_modelscope.json')).read_text())
        ms_files = {entry['Path']: entry for entry in ms['Data']['Files']}
        folder = OUT/'assets'/variant
        for entry in metadata['siblings']:
            name = entry['rfilename']
            if not name.endswith(('.json', '.txt', '.safetensors')):
                continue
            assert not name.startswith('/') and '..' not in Path(name).parts
            etag = entry['lfs']['sha256'] if entry.get('lfs') else entry['blobId']
            paths = get_local_download_paths(folder, name)
            if name.endswith('.safetensors'):
                source = ms_files[name]
                assert source['Sha256'] == etag and source['Size'] == entry['size']
                url = f'https://modelscope.cn/models/{spec["repo"]}/resolve/{source["Revision"]}/{quote(name)}'
                route = 'modelscope_direct_no_proxy'
            else:
                url = f'https://hf-mirror.com/{spec["repo"]}/resolve/{spec["revision"]}/{quote(name)}'
                route = 'hf_mirror_direct_no_proxy'
            tasks.append(dict(variant=variant, folder=folder, entry=entry, etag=etag, paths=paths,
                partial=paths.incomplete_path(etag), url=url, route=route, revision=spec['revision'], state='PENDING'))
    tasks.sort(key=lambda t: -t['entry']['size'])
    def downloaded(task):
        path = task['paths'].file_path
        partial = task['partial']
        try:
            return min(task['entry']['size'], path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0)
        except FileNotFoundError:  # Atomic final-file rename between stat calls.
            return task['entry']['size'] if path.exists() else 0
    total = sum(task['entry']['size'] for task in tasks)
    initial = sum(map(downloaded, tasks))
    begin = time.monotonic()
    write(TRANSPORT/'plan.json', dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        total_bytes=total, retained_bytes=initial, concurrency=3, proxy=False,
        scientific_config_sha256=sha(OUT/'config.json'), script_sha256=sha(Path(__file__)),
        files=[dict(model=t['variant'], name=t['entry']['rfilename'], bytes=t['entry']['size'],
                    expected_hash=t['etag'], route=t['route'], pinned_url=t['url']) for t in tasks]))

    def fetch(task):
        entry, paths, partial = task['entry'], task['paths'], task['partial']
        name = entry['rfilename']
        try:
            task['state'] = 'VERIFYING_EXISTING'
            if paths.file_path.exists():
                assert matches(paths.file_path, entry), 'Existing file hash mismatch: '+name
            else:
                assert not partial.exists() or partial.stat().st_size <= entry['size']
                log_path = TRANSPORT/(task['variant']+'_'+name.replace('/', '_')+'.log')
                with log_path.open('a') as log:
                    for attempt in range(1, 4):
                        if partial.exists() and partial.stat().st_size == entry['size']:
                            break
                        task['state'], task['attempt'] = 'DOWNLOADING', attempt
                        command = ['curl', '-L', '--noproxy', '*', '--fail', '--silent', '--show-error',
                            '--connect-timeout', '15', '--max-time', '3600', '--speed-limit', '1024',
                            '--speed-time', '60', '--continue-at', '-', '--output', str(partial), task['url']]
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode == 0:
                            break
                        if attempt == 3:
                            raise RuntimeError(f'{name}: curl exited {result.returncode}')
                        time.sleep(2)
                task['state'] = 'HASHING'
                assert matches(partial, entry), 'Downloaded file hash mismatch: '+name
                partial.replace(paths.file_path)
            write_download_metadata(task['folder'], name, task['revision'], task['etag'])
            task['state'] = 'VERIFIED'
            print(json.dumps(dict(model=task['variant'], file=name, state='VERIFIED', route=task['route'],
                                  bytes=entry['size'])), flush=True)
        except BaseException as exc:
            task['state'], task['error'] = 'FAILED', repr(exc)
            raise

    previous_bytes, previous_time = initial, begin
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, task) for task in tasks]
        while not all(f.done() for f in futures):
            now = time.monotonic()
            done = sum(map(downloaded, tasks))
            speed = max(0, done-previous_bytes)/max(1e-6, now-previous_time)
            elapsed = now-begin
            average = max(0, done-initial)/max(1e-6, elapsed)
            write(OUT/'status.json', dict(state='DOWNLOADING_MODELS', transport='modelscope_and_hf_mirror',
                proxy=False, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                total_bytes=total, bytes_on_disk=done, retained_bytes=initial,
                downloaded_this_run_bytes=done-initial, elapsed_seconds=elapsed,
                recent_speed_MBps=speed/1e6, average_speed_MBps=average/1e6,
                estimated_remaining_download_seconds=(total-done)/average if average > 0 else None,
                files=[dict(model=t['variant'], name=t['entry']['rfilename'], state=t['state'],
                            bytes=downloaded(t), total=t['entry']['size'], route=t['route'],
                            error=t.get('error')) for t in tasks], note='Download only; experiment resumes automatically after all hashes pass'))
            previous_bytes, previous_time = done, now
            time.sleep(5)
        for f in futures:
            f.result()
    write(TRANSPORT/'verification.json', dict(state='VERIFIED', files=len(tasks),
        total_bytes=total, retained_bytes=initial, elapsed_seconds=time.monotonic()-begin,
        all_original_hf_hashes_matched=True, original_revisions_unchanged=True,
        proxy=False, source_sha256=sha(Path(__file__))))
    write(OUT/'status.json', dict(state='TRANSPORT_VERIFIED', updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        next='Original downloader verifies cache and writes model manifests, then original supervisor runs',
        total_bytes=total, elapsed_seconds=time.monotonic()-begin))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(OUT/'status.json', dict(state='DOWNLOAD_FAILED', error=repr(exc), traceback=traceback.format_exc(),
              updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), partial_files_preserved=True))
        raise
