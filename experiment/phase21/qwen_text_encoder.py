"""Isolated Transformers 4.51 Qwen encoder; writes vectors for the old GRAM runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase21/qwen_text_expert_20261015_v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def config():
    return json.loads((OUT/'config.json').read_text())


def download():
    from huggingface_hub import snapshot_download
    cfg = config()
    start = time.monotonic()
    for variant, spec in cfg['qwen_models'].items():
        folder = OUT/'assets'/variant
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(snapshot_download, repo_id=spec['repo'], revision=spec['revision'],
                               local_dir=str(folder), max_workers=2,
                               allow_patterns=['*.json', '*.txt', '*.safetensors'])
            while not task.done():
                files = list(folder.rglob('*')) if folder.exists() else []
                size = 0
                for p in files:
                    try:
                        if p.is_file():
                            size += p.stat().st_size
                    except FileNotFoundError:
                        pass  # The downloader just atomically renamed a completed file.
                write(OUT/'status.json', dict(state='DOWNLOADING_MODELS', model=variant,
                    bytes_on_disk=size, elapsed_seconds=time.monotonic()-start,
                    updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
                    note='Download and environment only; no GPU allocated yet.'))
                time.sleep(5)
            task.result()
        write(OUT/'status.json', dict(state='HASHING_MODEL', model=variant, pid=os.getpid(),
                                     updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        files = [p for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts]
        write(OUT/'assets'/(variant+'_manifest.json'), dict(**spec,
            sha256={str(p.relative_to(ROOT)): sha(p) for p in files}))
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'transformers', 'tokenizers', 'huggingface-hub', 'numpy', 'safetensors']}
    write(OUT/'encoder_environment.json', dict(versions=versions,
          python=os.sys.executable, separate_from_gram=True))
    write(OUT/'status.json', dict(state='PREPARED', models_downloaded=True,
          updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def encode(domain, lease_pid):
    import gc
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    cfg = config()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(cfg['allocation_limit_gib']*2**30 /
                                             torch.cuda.get_device_properties(0).total_memory)
    folder = OUT/domain
    info = json.loads((folder/'encoder_texts.json').read_text())
    texts, item_ids = info['texts'], info['item_ids']
    for variant, spec in cfg['qwen_models'].items():
        os.kill(lease_pid, 0)
        started = time.monotonic()
        def progress(stage, **extra):
            os.kill(lease_pid, 0)
            row = dict(state='RUNNING', stage=stage, variant=variant, domain=domain,
                pid=os.getpid(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                elapsed_seconds=time.monotonic()-started,
                updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)
            write(folder/'status.json', row)
            print(json.dumps(row), flush=True)
        progress('verify_encoder_weights')
        manifest = json.loads((OUT/'assets'/(variant+'_manifest.json')).read_text())
        assert manifest['revision'] == spec['revision']
        for relative, expected in manifest['sha256'].items():
            assert sha(ROOT/relative) == expected, relative
        model_dir = OUT/'assets'/variant
        tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left', local_files_only=True)
        tokenized = tokenizer(texts, add_special_tokens=True, truncation=False)
        lengths = np.asarray([len(v) for v in tokenized['input_ids']])
        assert lengths.max() <= cfg['qwen_max_length'], 'Common input needs a larger declared cap; do not silently truncate'
        progress('load_encoder', input_tokens=int(lengths.sum()), maximum_tokens=int(lengths.max()))
        model = AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16,
                                         attn_implementation='sdpa').eval().requires_grad_(False).cuda()

        def vectors(documents):
            batch = tokenizer(documents, padding=True, truncation=False, return_tensors='pt').to('cuda:0')
            assert batch['attention_mask'][:, -1].all()  # Last actual token under left padding.
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state[:, -1].float()
                result = torch.nn.functional.normalize(hidden, p=2, dim=1)
            assert torch.isfinite(result).all()
            return result.cpu().numpy()

        probe = list(dict.fromkeys([0, int(lengths.argmin()), int(lengths.argmax()), len(texts)-1]))
        mixed = vectors([texts[j] for j in probe])
        separate = np.concatenate([vectors([texts[j]]) for j in probe])
        cos = (mixed*separate).sum(1)
        assert float(cos.min()) > .999, ('padding/batch drift', cos)
        dim = spec['dimension']
        assert mixed.shape == (len(probe), dim)
        result = np.empty((len(texts), dim), dtype=np.float32)
        # Sort only for padded batch efficiency; restore original catalog row order.
        order = np.argsort(lengths, kind='stable')
        begin = time.monotonic()
        for start in range(0, len(texts), spec['batch_size']):
            ix = order[start:start+spec['batch_size']]
            result[ix] = vectors([texts[j] for j in ix])
            done = min(start+spec['batch_size'], len(texts))
            if start == 0 or done % 128 == 0 or done == len(texts):
                elapsed = time.monotonic()-begin
                progress('encode_items', items_completed=done, items_total=len(texts),
                         remaining_encoding_seconds_estimate=elapsed/done*(len(texts)-done),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1., atol=3e-6)
        destination = folder/(variant+'_embeddings.npz')
        np.savez(destination, item_ids=np.asarray(item_ids), embeddings=result)
        write(folder/(variant+'_encoding.json'), dict(state='PASSED', **spec,
            dtype='float16 model / float32 normalized output', attention='torch SDPA',
            pooling='last actual token; left padding', instructions=None,
            instruction_reason='Symmetric item-document vectors; no separately encoded user query.',
            texts_sha256=sha(folder/'encoder_texts.json'), embeddings_sha256=sha(destination),
            items=len(texts), token_count=int(lengths.sum()), max_tokens=int(lengths.max()),
            qwen_truncated_items=0, input_text_matched_to_bge=True,
            batch_vs_single_min_cosine=float(cos.min()), probe_indices=probe,
            encoding_seconds=time.monotonic()-begin, total_seconds=time.monotonic()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            targets_accessed=False))
        del model, tokenizer, tokenized, result
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['download', 'encode'])
    parser.add_argument('--domain', choices=['Beauty', 'Toys'])
    parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'download':
            download()
        else:
            assert args.domain and args.lease_pid
            encode(args.domain, args.lease_pid)
    except BaseException as exc:
        destination = OUT/args.domain if args.domain else OUT
        write(destination/'status.json', dict(state='FAILED', stage='qwen_'+args.mode,
              error=repr(exc), traceback=traceback.format_exc(),
              updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        raise
