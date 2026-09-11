"""Evaluate a frozen saved SPRec SFT checkpoint, without any training."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase20.run_preference import ROOT, Run, Stopped, seed_everything


def load_parent_reference(run, directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['data_manifest']['prepared_files'] != run.data_manifest['prepared_files']
            or manifest['data_manifest']['parent_sha256'] != run.data_config['parent_sha256']
            or manifest['config']['decoding'] != run.config['decoding']
            or manifest['precision'] != 'float32' or manifest['tf32'] is not False
            or manifest['pcrf_checkpoint_sha256'] != sha256(ROOT / run.config['pcrf']['checkpoint'])
            or manifest['config']['pcrf'] != run.config['pcrf']):
        raise AssertionError('Matched parent/PCRF protocol mismatch')
    path = directory / 'matched_parent_reference.json'
    reference = json.loads(path.read_text())
    if set(reference) != set(run.reference):
        raise AssertionError('Parent reference users mismatch')
    run.reference = reference
    write_json(run.output / 'shared_parent_reference_manifest.json', {
        'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
        'producer_manifest_sha256': sha256(directory / 'manifest.json')})


def gpu_setup(config):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    if free / 2**20 < config['minimum_free_mib']:
        raise RuntimeError(f'Insufficient free memory: {free / 2**20:.0f} MiB')
    torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Existing attempt; no automatic retry or overwrite')
    run = None
    try:
        gpu_setup(config)
        seed_everything(config['seed'])
        run = Run(config, config_path, output, 'run')
        def stop(signum, frame):
            run.stop_reason = f'SIGNAL_{signum}'
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]:
            signal.signal(sig, stop)
        signal.alarm(config['max_wall_seconds'])
        path = ROOT / config['evaluation_checkpoint']
        if sha256(path) != config['evaluation_checkpoint_sha256']:
            raise AssertionError('Saved SFT checkpoint SHA mismatch')
        payload = torch.load(path, map_location='cpu')
        if (payload['round'], payload['stage'], payload['step']) != (1, 'sft', 855):
            raise AssertionError('Unexpected checkpoint training stage')
        for key in ['dataset', 'seed', 'data_config', 'decoding', 'pcrf']:
            if payload['config'][key] != config[key]:
                raise AssertionError(f'Checkpoint protocol mismatch: {key}')
        run.model = run.new_model()
        run.model.load_state_dict(payload['model'], strict=True)
        run.model.eval().requires_grad_(False)
        del payload
        run.snapshot()
        source = ROOT / config['shared_parent_output']
        load_parent_reference(run, source)
        indices = json.loads((source / 'selection_indices.json').read_text())[:16]
        smoke = run.evaluate('round1_sft_parity16', indices)
        cached = {r['user_id']: r for r in map(json.loads, (source / 'round1_sft.predictions.jsonl').read_text().splitlines())}
        with (source / 'round1_sft.ranks.tsv').open() as handle:
            cached_ranks = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
        with (output / 'round1_sft_parity16.ranks.tsv').open() as handle:
            actual_ranks = list(csv.DictReader(handle, delimiter='\t'))
        actual = list(map(json.loads, (output / 'round1_sft_parity16.predictions.jsonl').read_text().splitlines()))
        for row, ranks in zip(actual, actual_ranks):
            old = cached[row['user_id']]
            assert row['ranked_item_ids'] == old['ranked_item_ids']
            np.testing.assert_allclose(row['scores'], old['scores'], rtol=0, atol=1e-6)
            assert ranks == cached_ranks[row['user_id']]
        write_json(output / 'smoke.json', {'state': 'PASSED', 'matched_saved_predictions': 16,
            'raw_and_pcrf_ranks_exact': True, 'score_atol': 1e-6, 'optimizer_steps': 0})
        run.event('checkpoint_parity_passed', users=16)
        full = run.evaluate('round1_sft_full', list(range(len(run.validation))))
        a = full['raw']['ndcg@10'] - full['original_gram']['ndcg@10']
        b = full['pcrf']['ndcg@10'] - full['original_gram_pcrf']['ndcg@10']
        write_json(output / 'summary.json', {'state': 'COMPLETED', 'evaluation_checkpoint': config['evaluation_checkpoint'],
            'checkpoint_sha256': config['evaluation_checkpoint_sha256'], 'full_validation': full,
            'raw_ndcg_delta_vs_gram': a, 'combined_ndcg_delta_vs_original_pcrf': b,
            'positive_signal': a > 0 or b > 0, 'new_optimizer_steps': 0, 'test_read': False,
            'scope': 'Saved first-round SFT checkpoint; not a training-budget-matched control for three-round SPRec.'})
        signal.alarm(0)
        write_json(output / 'status.json', {'state': 'COMPLETED', 'dataset': config['dataset'], 'pid': os.getpid(),
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'elapsed_seconds': time.time()-run.started,
            'test_read': False, 'optimizer_steps': 0})
    except BaseException as error:
        write_json(output / 'status.json', {'state': 'STOPPED' if isinstance(error, Stopped) else 'FAILED',
            'error': repr(error), 'traceback': traceback.format_exc(), 'pid': os.getpid(),
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'test_read': False, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
