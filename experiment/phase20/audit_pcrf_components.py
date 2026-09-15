"""Fixed CPU ablation of PCRF on the matched parent validation beam cache."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase9.eval_cf0_b3_beamfusion import score_item_head, standardize
from experiment.phase9.eval_cf0_b4_reliability import rank_matrix

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'experiment/phase20/pcrf_component_audit.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def metric(ranks):
    result = {'users': len(ranks)}
    for k in (5, 10, 20, 50):
        result[f'hit@{k}'] = float(np.mean(ranks <= k))
        result[f'ndcg@{k}'] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
    return result


def paired(candidate, baseline, config):
    result = {'users': len(baseline),
              'hit10_gained': int(((candidate <= 10) & (baseline > 10)).sum()),
              'hit10_lost': int(((candidate > 10) & (baseline <= 10)).sum())}
    values = np.stack([(candidate <= 10).astype(float) - (baseline <= 10),
                       np.where(candidate <= 10, 1 / np.log2(candidate + 1), 0)
                       - np.where(baseline <= 10, 1 / np.log2(baseline + 1), 0)], axis=1)
    rng = np.random.default_rng(config['bootstrap_seed'])
    samples = []
    for start in range(0, config['bootstrap_repeats'], 32):
        count = min(32, config['bootstrap_repeats'] - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(values[indices].mean(1))
    ci = np.quantile(np.concatenate(samples), [0.025, 0.975], axis=0)
    for j, name in enumerate(('hit@10', 'ndcg@10')):
        result[name] = {'delta': float(values[:, j].mean()), 'ci95': ci[:, j].tolist()}
    return result


def main():
    started = time.time()
    config = json.loads(CONFIG.read_text())
    torch.set_num_threads(config['cpu_threads'])
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=False)

    def status(state, **extra):
        value = {'state': state, 'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 'elapsed_seconds': round(time.time() - started, 2), **extra}
        (output / 'status.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    try:
        status('RUNNING', stage='load_and_hash')
        inputs = {k: ROOT / config[k] for k in ('train', 'validation', 'predictions', 'reference', 'item_head')}
        inputs.update({'config': CONFIG, 'code': Path(__file__),
                       'item_head_scoring_code': ROOT / 'experiment/phase9/eval_cf0_b3_beamfusion.py',
                       'pcrf_code': ROOT / 'experiment/phase9/eval_cf0_b4_reliability.py',
                       'frequency_code': ROOT / 'experiment/phase18/core/raserec_data.py'})
        hashes = {k: digest(p) for k, p in inputs.items()}
        (output / 'protocol.json').write_text(json.dumps({'config': config, 'input_sha256': hashes}, indent=2) + '\n')
        train = read_rows(inputs['train'])
        validation = read_rows(inputs['validation'])
        lookup = {r['user_id']: r for r in validation}
        preds = read_rows(inputs['predictions'])
        reference = json.loads(inputs['reference'].read_text())
        assert len(lookup) == len(validation) == len(preds) == config['users']
        assert len({r['user_id'] for r in preds}) == config['users']
        assert set(lookup) == set(reference) == {r['user_id'] for r in preds}
        frequencies = training_frequencies(train, config['catalog_size_with_padding'])
        records = []
        historical_tail_metadata_mismatches = 0
        for pred in preds:
            row = lookup[pred['user_id']]
            candidates = pred['ranked_item_ids']
            assert pred['split'] == 'validation' and pred['gold_item_id'] == row['target']
            assert len(candidates) == len(set(candidates)) == len(pred['scores']) == 50
            assert all(0 < i < len(frequencies) for i in candidates)
            assert np.isfinite(pred['scores']).all()
            ref = reference[row['user_id']]
            assert int(ref['target_frequency']) == frequencies[row['target']]
            freq = frequencies[candidates]
            tail = float((freq[:10] <= config['q1']).mean())
            # The matched reference refreshes ranks only; its tail_mass metadata
            # still describes historical beams. Recompute from the actual cache.
            historical_tail_metadata_mismatches += int(abs(tail - float(ref['tail_mass'])) >= 1e-12)
            records.append({'user': row['user_id'], 'history': row['history'][-20:],
                            'candidate_ids': candidates, 'seq': np.asarray(pred['scores']),
                            'target_position': candidates.index(row['target']) if row['target'] in candidates else -1,
                            'frequencies': freq, 'target_frequency': frequencies[row['target']], 'tail_mass': tail})
        status('RUNNING', stage='frozen_item_head_cpu_scoring', users=len(records))
        score_item_head(records, inputs['item_head'], 256)
        seq = np.stack([standardize(r['seq']) for r in records])
        cf = np.stack([standardize(r['cf']) for r in records])
        pop = np.stack([standardize(np.log1p(r['frequencies'])) for r in records])
        assert np.isfinite(cf).all()
        tail = np.asarray([r['tail_mass'] for r in records])
        positions = np.asarray([r['target_position'] for r in records])
        ranks = {name: rank_matrix(seq, cf, pop, tail, positions, tuple(params))[0]
                 for name, params in config['arms'].items()}
        ranks['cf_only'] = rank_matrix(np.zeros_like(seq), cf, pop, tail, positions, (1., 0., 0.))[0]
        for name, key in [('gram', 'baseline_rank'), ('pcrf', 'pcrf_rank')]:
            expected = np.asarray([int(reference[r['user']][key]) for r in records])
            assert np.array_equal(ranks[name], expected), f'{name} reference rank mismatch'
        assert all(np.array_equal(r <= 50, ranks['gram'] <= 50) for r in ranks.values())
        status('RUNNING', stage='paired_ablation_analysis', exact_reference_users=len(records))
        masks = {'overall': np.ones(len(records), dtype=bool),
                 'tail': np.asarray([r['target_frequency'] <= config['q1'] for r in records]),
                 'non_tail': np.asarray([r['target_frequency'] > config['q1'] for r in records])}
        groups = {group: {name: metric(r[mask]) for name, r in ranks.items()} for group, mask in masks.items()}
        comparisons = {}
        for baseline in ('gram', 'plain_fusion_100', 'plain_fusion_075', 'popularity_only', 'reliability_only'):
            comparisons['pcrf_minus_' + baseline] = {
                group: paired(ranks['pcrf'][mask], ranks[baseline][mask], config)
                for group, mask in masks.items()}
        with (output / 'per_user.tsv').open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['user_id', *ranks])
            for i, row in enumerate(records):
                writer.writerow([row['user'], *(int(r[i]) for r in ranks.values())])
        assert hashes == {k: digest(p) for k, p in inputs.items()}
        summary = {'state': 'COMPLETED', 'verification_status': 'ANALYZED', 'config': config,
                   'input_sha256': hashes, 'groups': groups, 'paired_comparisons': comparisons,
                   'exact_reference_users': len(records), 'candidate_coverage_identical': True,
                   'historical_tail_metadata_mismatches': historical_tail_metadata_mismatches,
                   'test_read': False, 'new_training_steps': 0, 'gpu_used': False,
                   'elapsed_seconds': round(time.time() - started, 2),
                   'limitations': 'Reused validation; fixed parent and item-head seed. Descriptive paired intervals; no multiple-comparison correction or independent confirmation.'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        status('COMPLETED', summary=str(output / 'summary.json'))
    except BaseException as error:
        status('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
