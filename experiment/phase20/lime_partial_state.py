"""Collect only the first three actual beam steps from frozen clean GRAM."""
import argparse
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
from experiment.phase20 import lime_decode_screen as base
from experiment.phase20 import lime_clean_benefit as clean
from experiment.phase20.lime_generation_trace_core import BeamObserver, validate_steps
from experiment.phase20.lime_generation_trace import json_safe
from experiment.phase20.lime_partial_state_core import prefix_features

ROOT = base.ROOT
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
EARLY = ROOT / 'artifacts/phase20/lime_early_correction_20260922_v1'
OUT = ROOT / 'artifacts/phase20/lime_partial_state_20260922_v1'
PROTOCOL = ROOT / 'plan/第二十阶段/LIME当前生成状态替代最终信息_20260922.md'
MODULE = 'experiment.phase20.lime_partial_state'
GPUS = {'Toys': 'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2',
        'Beauty': 'GPU-07374397-b213-09c1-2988-718acb6ed231'}


def prepare():
    assert base.read(SOURCE / 'status.json')['verified'] and base.read(EARLY / 'status.json')['verified']
    OUT.mkdir(exist_ok=False)
    manifest = {}

    def use(p):
        p = Path(p); key = str(p.relative_to(ROOT))
        manifest[key] = base.sha(p)
        return p

    for old in (SOURCE, EARLY):
        for rel, digest in base.read(use(old / 'completion_manifest.json'))['sha256'].items():
            assert base.sha(ROOT / rel) == digest, rel
    base.write(OUT / 'config.json', dict(gpus=GPUS, worker_timeout_seconds=2700,
        minimum_free_mib=12288, allocation_limit_gib=11, cpu_threads=2, beam_size=50,
        snapshots=[2, 3], alpha=.3, test_read=False, upstream_optimizer_steps=0))
    shutil.copy2(PROTOCOL, OUT / 'protocol.md')
    for domain in GPUS:
        folder = OUT / domain; folder.mkdir()
        for name in ('catalog.json', 'samples.json', 'data_config.json'):
            shutil.copy2(use(SOURCE / domain / name), folder / name); use(folder / name)
        cfg = base.read(folder / 'data_config.json')
        assert base.sha(use(ROOT / cfg['parent_checkpoint'])) == cfg['parent_sha256']
        for name in ('data.npz', 'lime.npy', 'cache_summary.json', 'candidates.jsonl'):
            use(SOURCE / domain / name)
        for name in ('oof_predictions.npz', 'development_predictions.npz', 'ranks.npz'):
            use(EARLY / domain / name)
        base.write(folder / 'status.json', dict(state='PREPARED', domain=domain,
                                               users_completed=0, updated_at=base.stamp()))
    names = ('lime_partial_state.py', 'lime_partial_state_core.py', 'test_lime_partial_state.py',
             'lime_decode_screen.py', 'lime_decode_screen_core.py', 'lime_clean_benefit.py',
             'lime_clean_upstream_core.py', 'lime_clean_gram_core.py', 'guided_decoding_core.py',
             'lime_generation_trace_core.py', 'lime_generation_trace.py',
             'lime_early_correction_core.py', 'lime_clean_benefit_core.py',
             'source_competition_core.py', 'catalog_scope_core.py', 'run_lime_partial_state.sh')
    for name in names:
        use(ROOT / 'experiment/phase20' / name)
    for p in (PROTOCOL, OUT / 'protocol.md', OUT / 'config.json'):
        use(p)
    # Hash model implementation files too; tokenization/weights retain prior provenance.
    for p in (ROOT / 'GRAM/src').rglob('*.py'):
        use(p)
    base.write(OUT / 'manifest.json', dict(sha256=manifest, created_at=base.stamp()))
    base.write(OUT / 'status.json', dict(state='PREPARED', updated_at=base.stamp(), test_read=False))
    print(json.dumps(dict(state='PREPARED', input_hashes=len(manifest))), flush=True)


class Worker(clean.Worker):
    def __init__(self, domain):
        clean.OUT = OUT
        super().__init__(domain)
        assert self.initial_digest == base.read(SOURCE / domain / 'cache_summary.json')['model_digest']
        assert self.model.config.forced_eos_token_id is None
        self.old = base.arrays(SOURCE / domain / 'data.npz')
        self.fused = np.load(SOURCE / domain / 'lime.npy', mmap_mode='r')
        self.reference = clean.rows(SOURCE / domain / 'candidates.jsonl')
        assert len(self.reference) == len(self.old['users'])

    def observed(self, ctx, full=False):
        import torch
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        from experiment.phase20.lime_decode_screen_core import ProbabilityGuide
        begin = time.monotonic()
        guide = ProbabilityGuide(self.index)
        observer = BeamObserver(guide)
        with torch.no_grad(), observer.installed():
            result = super(GRAM, self.model).generate(input_ids=ctx['ids'].reshape(1, -1),
                attention_mask=ctx['mask'].reshape(1, -1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden']),
                max_length=self.max_length if full else 4, prefix_allowed_tokens_fn=self.prefix_fn,
                num_beams=50, num_return_sequences=50, return_dict_in_generate=True,
                output_scores=True, length_penalty=1., logits_processor=LogitsProcessorList([guide, observer]))
        checks = validate_steps(observer.steps)
        output = dict(steps=observer.steps, checks=checks)
        if full:
            paths = [p[:p.index(1)+1] for p in result.sequences.cpu().tolist()]
            output.update(paths=paths, ids=[self.index.path_to_item[tuple(p)] for p in paths],
                          scores=result.sequences_scores.cpu().double().numpy())
        else:
            assert len(observer.steps) == 3 and all(not s['finished'] for s in observer.steps)
        del result
        torch.cuda.synchronize()
        self.completed += 1; self.seconds.append(time.monotonic() - begin)
        return output

    def run(self):
        smoke = []; features = {depth: [] for depth in (2, 3)}
        self.stage = 'prefix_cache'; self.status()
        with gzip.open(self.folder / 'traces.jsonl.gz', 'wt', compresslevel=1) as log:
            for row, sample_ix in enumerate(self.old['sample_indices']):
                if time.monotonic() - self.started > self.config['worker_timeout_seconds']:
                    raise TimeoutError('Fixed 45 minute prefix-cache budget')
                sample_ix = int(sample_ix); sample = self.samples[sample_ix]
                assert sample['user_id'] == self.old['users'][row] and sample['role'] != 'gate_calibration'
                ctx = self.context(sample_ix)
                partial = self.observed(ctx)
                if row < 3:
                    assert sample['role'] == 'ranker_fit'
                    complete = self.observed(ctx, full=True)
                    assert complete['steps'][:3] == partial['steps']
                    assert complete['ids'] == self.reference[row]['ids']
                    assert complete['paths'] == self.reference[row]['paths']
                    np.testing.assert_array_equal(complete['scores'], self.old['native'][row])
                    smoke.append(dict(user=sample['user_id'], partial_steps_exact=True, final_cache_exact=True))
                    base.write(self.folder / 'smoke.json', dict(checks=smoke))
                q = self.index.conditional_logs(self.fused[sample_ix])
                stats = {}
                for depth in (2, 3):
                    mapped = prefix_features(partial['steps'], q, depth)
                    keys = [tuple(self.catalog['decoder_paths'][int(item)-1][:depth+1]) for item in self.old['ids'][row]]
                    assert all(key in mapped for key in keys), 'Frozen survivor prefix absent from actual live state'
                    features[depth].append(np.stack([mapped[key] for key in keys]))
                    stats[str(depth)] = dict(all_expansions=len(mapped), selected_distinct_prefixes=len(set(keys)))
                log.write(json.dumps(json_safe(dict(row=row, user_id=sample['user_id'], sample_index=sample_ix,
                    steps=partial['steps'], checks=partial['checks'], counts=stats)), allow_nan=False) + '\n')
                log.flush()
                self.users_completed += 1
                if self.users_completed == 1 or self.users_completed % 25 == 0:
                    self.status()
        np.savez_compressed(self.folder / 'features.npz', users=self.old['users'], ids=self.old['ids'],
                            sample_indices=self.old['sample_indices'], **{f'prefix{d}': np.stack(v) for d, v in features.items()})
        assert self.digest() == self.initial_digest
        base.write(self.folder / 'cache_summary.json', dict(state='COMPLETED', users=self.users_completed,
            partial_decodes=self.users_completed, full_smoke_decodes=3, seconds=time.monotonic()-self.started,
            model_unchanged=True, model_digest=self.initial_digest, test_read=False, upstream_optimizer_steps=0))
        base.write(self.folder / 'cache_manifest.json', dict(sha256={name: base.sha(self.folder / name)
            for name in ('features.npz', 'traces.jsonl.gz', 'smoke.json', 'cache_summary.json')}))
        self.stage = 'prefix_cache_completed'; self.status(state='COMPLETED')


def worker(domain):
    try:
        Worker(domain).run()
    except BaseException as e:
        (OUT / domain / 'failure.txt').write_text(traceback.format_exc())
        base.write(OUT / domain / 'status.json', dict(state='FAILED', error=repr(e), updated_at=base.stamp()))
        raise


def safe_read(p):
    for attempt in range(5):
        try:
            return base.read(p)
        except json.JSONDecodeError:
            time.sleep(.05)
    raise RuntimeError(f'Cannot read status {p}')


def supervise():
    cfg = base.read(OUT / 'config.json')
    for rel, digest in base.read(OUT / 'manifest.json')['sha256'].items():
        assert base.sha(ROOT / rel) == digest, rel
    resource = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in resource.splitlines()}
    for domain, gpu in GPUS.items():
        assert free[gpu] >= cfg['minimum_free_mib'], (domain, free[gpu])
    processes = {}; started = time.monotonic()
    for domain, gpu in GPUS.items():
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        with (OUT / domain / 'worker.log').open('x') as log:
            processes[domain] = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--domain', domain],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base.write(OUT / 'launch.json', dict(pid=os.getpid(), workers={d: p.pid for d, p in processes.items()},
                                       resources=resource, started_at=base.stamp()))
    while any(p.poll() is None for p in processes.values()):
        base.write(OUT / 'status.json', dict(state='RUNNING', stage='partial_state_cache', pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started, updated_at=base.stamp(),
            domains={d: safe_read(OUT / d / 'status.json') for d in GPUS},
            alive={d: p.poll() is None for d, p in processes.items()}))
        if time.monotonic() - started > cfg['worker_timeout_seconds']:
            for p in processes.values():
                if p.poll() is None:
                    p.terminate()
            raise TimeoutError('Prefix cache hard timeout')
        time.sleep(15)
    codes = {d: p.wait() for d, p in processes.items()}
    assert all(code == 0 for code in codes.values()), codes
    base.write(OUT / 'status.json', dict(state='CACHE_COMPLETED', exit_codes=codes, updated_at=base.stamp()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'launch', 'supervise', 'worker'))
    p.add_argument('--domain', choices=list(GPUS)); args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'worker':
        worker(args.domain)
    elif args.action == 'supervise':
        try:
            supervise()
        except BaseException as e:
            (OUT / 'failure.txt').write_text(traceback.format_exc())
            base.write(OUT / 'status.json', dict(state='FAILED', error=repr(e), updated_at=base.stamp()))
            raise
    else:
        assert base.read(OUT / 'status.json')['state'] == 'PREPARED'
        with (OUT / 'supervisor.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-m', MODULE, 'supervise'], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, output=str(OUT))), flush=True)


if __name__ == '__main__':
    main()
