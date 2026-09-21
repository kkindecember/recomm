"""Pinned public model download with size/hash checks and a durable heartbeat."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now

CONFIG = ROOT/'experiment/phase20/lime_comparison_20260920_v1.json'


def main():
    c = read(CONFIG)
    out = ROOT/c['output']/'assets'
    out.mkdir(exist_ok=True)
    model = out/'bge-base-en-v1.5'
    model.mkdir(exist_ok=True)
    tree = {r['path']: r for r in read(out/'huggingface_tree.json')}
    files = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.txt', 'tokenizer.json', 'modules.json', '1_Pooling/config.json',
             'sentence_bert_config.json', 'pytorch_model.bin']
    started = time.monotonic()
    hashes = {}
    child = None
    try:
        for name in files:
            target = model/name
            target.parent.mkdir(exist_ok=True, parents=True)
            assert not target.exists(), f'Refusing to overwrite {target}'
            partial = target.with_name(target.name+'.part')
            url = f'https://huggingface.co/{c["encoder"]}/resolve/{c["encoder_revision"]}/{name}'
            with (out/'download.log').open('a') as log:
                child = subprocess.Popen(['curl', '-L', '--fail', '--connect-timeout', '30',
                    '--max-time', str(c['download_timeout_seconds']), '--output', str(partial), url],
                    stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    write(out/'status.json', dict(state='DOWNLOADING', updated_at=now(), pid=os.getpid(),
                        file=name, bytes=partial.stat().st_size if partial.exists() else 0,
                        total_bytes=tree[name]['size'], elapsed_seconds=round(time.monotonic()-started, 1)))
                    if time.monotonic()-started > c['download_timeout_seconds']:
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        raise TimeoutError('Public model download exceeded four hours')
                    time.sleep(5)
                assert child.returncode == 0, f'curl exit {child.returncode} for {name}; see download.log'
            assert partial.stat().st_size == tree[name]['size'], name
            digest = sha(partial)
            if 'lfs' in tree[name]:
                assert digest == tree[name]['lfs']['oid'], name
            else:
                content = partial.read_bytes()
                assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() == tree[name]['oid'], name
            partial.replace(target)
            hashes[name] = digest
        pooling = read(model/'1_Pooling/config.json')
        assert pooling['pooling_mode_cls_token'] is True
        assert not any(v for k, v in pooling.items() if k.startswith('pooling_mode_') and k != 'pooling_mode_cls_token')
        modules = read(model/'modules.json')
        assert [m['type'] for m in modules] == ['sentence_transformers.models.Transformer',
            'sentence_transformers.models.Pooling', 'sentence_transformers.models.Normalize']
        assert read(model/'sentence_bert_config.json')['max_seq_length'] == 512
        write(out/'manifest.json', dict(model=c['encoder'], revision=c['encoder_revision'], hashes=hashes,
            verified_lfs_and_git_objects=True, pooling=pooling, modules=modules))
        write(out/'status.json', dict(state='COMPLETED', updated_at=now(), pid=os.getpid(),
            elapsed_seconds=round(time.monotonic()-started, 1), files=len(hashes)))
    except BaseException as e:
        if child is not None and child.poll() is None:
            child.terminate()
        (out/'failure.txt').write_text(traceback.format_exc())
        write(out/'status.json', dict(state='FAILED', updated_at=now(), pid=os.getpid(), error=repr(e)))
        raise


if __name__ == '__main__':
    main()
