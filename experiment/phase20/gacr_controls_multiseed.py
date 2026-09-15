"""Frozen full-validation controls and residual seeds, with observable progress.

No backbone loading, training, test data, beam generation or parameter selection.
Historical implementations and output directories remain immutable.
"""
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

from experiment.phase20.audit_pcrf_components import digest, read_rows, metric, paired
from experiment.phase20.gacr_score_controls import residual_orders, reciprocal_scores, pcrf_ranks
from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase9.eval_cf0_b3_beamfusion import CF0B2ItemHead

ROOT = Path(__file__).resolve().parents[2]
BASE_METHODS = ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'gacr_v6', 'gacr_v6_pcrf')
CONTROL_PAIRS = (
    ('gacr_v6_pcrf', 'c1_reciprocal_pcrf'),
    ('gacr_v6_pcrf', 'original_gram_reciprocal_pcrf'),
    ('gacr_v6_pcrf', 'c1_fixed_candidates_residual_pcrf'),
    ('c1_fixed_candidates_residual_pcrf', 'c1_reciprocal_pcrf'),
    ('original_gram_reciprocal_pcrf', 'original_pcrf'),
    ('c1_reciprocal_pcrf', 'original_pcrf'),
    ('c1_fixed_candidates_residual_pcrf', 'original_pcrf'),
)


def write_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def group_masks(users, lookup, frequencies, fit_users, q1):
    target_freq = np.asarray([frequencies[lookup[u]['target']] for u in users])
    return {
        'all_validation': np.ones(len(users), dtype=bool),
        f'target_frequency_le{q1}': target_freq <= q1,
        f'target_frequency_gt{q1}': target_freq > q1,
        'outside_c1_training_users': np.asarray([u not in fit_users for u in users]),
        'inside_c1_training_users': np.asarray([u in fit_users for u in users]),
    }


@torch.no_grad()
def score_with_progress(records, checkpoint_path, batch_size, progress):
    """Same 50-candidate CPU arithmetic and batching as historical item scoring."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['model_config']
    model = CF0B2ItemHead(num_items=cfg['num_items'], max_history=cfg['max_history'],
                         d_model=cfg['d_model'], num_layers=cfg['num_layers'],
                         num_heads=cfg['num_heads'], dropout=cfg['dropout'],
                         temperature=cfg['temperature_initial'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    items = torch.nn.functional.normalize(model.item_embedding.weight[1:], dim=-1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        histories = torch.zeros(len(batch), cfg['max_history'], dtype=torch.long)
        masks = torch.zeros_like(histories, dtype=torch.bool)
        for i, record in enumerate(batch):
            history = record['history'][-cfg['max_history']:]
            histories[i, :len(history)] = torch.tensor(history)
            masks[i, :len(history)] = True
        users = model.encode(histories, masks)
        ids = torch.tensor([r['candidate_ids'] for r in batch]) - 1
        scores = scale * torch.einsum('bd,bkd->bk', users, items[ids])
        for record, values in zip(batch, scores.numpy()):
            record['cf'] = values.astype(np.float64)
        progress(min(start + batch_size, len(records)), len(records))


def remap_fixed_records(c1_records, predictions):
    result = []
    for original, pred in zip(c1_records, predictions):
        ids = pred['ranked_item_ids']
        assert pred['user_id'] == original['user']
        assert len(ids) == len(set(ids)) == 50 and set(ids) == set(original['candidate_ids'])
        order = [original['candidate_ids'].index(i) for i in ids]
        target = original['candidate_ids'][original['target_position']] if original['target_position'] >= 0 else None
        result.append({**original, 'candidate_ids': ids, 'seq': np.asarray(pred['scores'], dtype=np.float64),
                       'cf': original['cf'][order], 'target_position': ids.index(target) if target is not None else -1})
    return result


class Evaluation:
    def __init__(self, config, dataset, mode, attempt):
        self.config, self.domain = config, config['domains'][dataset]
        self.dataset, self.mode = dataset, mode
        suffix = f'{dataset}_smoke_{attempt}' if mode == 'smoke' else dataset
        self.output = ROOT / config['output'] / suffix
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = self.phase_started = time.monotonic()
        self.phase, self.stage = 'initialization', ''
        self.completed = []
        self.stage_started = self.started
        self.paths = {k: ROOT / v for k, v in self.domain['inputs'].items()}
        self.hashes = {}

    def status(self, state='RUNNING', stage=None, **extra):
        if stage is not None and stage != self.stage:
            self.stage, self.stage_started = stage, time.monotonic()
        row = {'state': state, 'dataset': self.dataset, 'mode': self.mode, 'phase': self.phase,
               'stage': self.stage, 'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'elapsed_seconds': round(time.monotonic() - self.started, 1),
               'phase_elapsed_seconds': round(time.monotonic() - self.phase_started, 1),
               'phase_budget_seconds': self.config['phase_budget_seconds'], 'completed_phases': self.completed,
               'new_optimizer_steps': 0, 'new_beam_inference': False, 'test_read': False, **extra}
        write_json(self.output / 'status.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def progress(self, done, total):
        elapsed = time.monotonic() - self.stage_started
        self.status(progress={'completed': done, 'total': total, 'unit': 'users',
                              'percent': round(100 * done / total, 2)},
                    stage_eta_seconds=round(elapsed * (total - done) / done, 1) if done else None)

    def begin(self, phase):
        self.phase, self.phase_started = phase, time.monotonic()
        signal.alarm(self.config['smoke_budget_seconds'] if self.mode == 'smoke' else self.config['phase_budget_seconds'])
        self.status(stage='begin_phase')

    def load(self):
        self.status(stage='verify_frozen_input_hashes')
        for i, (key, path) in enumerate(self.paths.items(), 1):
            self.hashes[key] = digest(path)
            assert self.hashes[key] == self.domain['expected_sha256'][key], ('input_hash_mismatch', key)
            self.status(hash_progress={'completed': i, 'total': len(self.paths), 'input': key})
        for path, expected in self.config['source_sha256'].items():
            assert digest(ROOT / path) == expected, ('source_hash_mismatch', path)
        write_json(self.output / 'protocol.json', {'config': self.config, 'dataset': self.dataset,
            'input_sha256': self.hashes, 'config_sha256': digest(ROOT / self.config['config_path']),
            'mode': self.mode, 'new_optimizer_steps': 0, 'test_read': False})
        validation = read_rows(self.paths['validation'])
        self.lookup = {r['user_id']: r for r in validation}
        assert len(self.lookup) == len(validation) == self.domain['users']
        self.users = list(self.lookup)
        if self.mode == 'smoke':
            self.users = self.users[:self.config['smoke_users']]
        self.frequencies = training_frequencies(read_rows(self.paths['train']), self.domain['catalog_size_with_padding'])
        with self.paths['reference_ranks'].open() as handle:
            self.reference = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
        assert set(self.reference) == set(self.lookup)
        self.masks = group_masks(self.users, self.lookup, self.frequencies,
                                set(self.paths['c1_training_users'].read_text().splitlines()), self.domain['q1'])
        self.caches = {}
        for name in ('parent', 'c1', 'gacr'):
            rows = read_rows(self.paths[name])
            cache = {r['user_id']: r for r in rows}
            assert len(cache) == len(rows) == len(self.lookup) and set(cache) == set(self.lookup)
            self.caches[name] = [cache[u] for u in self.users]
            for r in self.caches[name]:
                assert r['split'] == 'validation' and r['gold_item_id'] == self.lookup[r['user_id']]['target']
                assert len(r['ranked_item_ids']) == len(set(r['ranked_item_ids'])) == len(r['scores']) == 50
                assert all(0 < i < len(self.frequencies) for i in r['ranked_item_ids'])
                assert np.isfinite(r['scores']).all()
        self.status(stage='load_union_feature_cache')
        selected = set(self.users)
        user_indices = {u: i for i, u in enumerate(self.users)}
        features = {}
        with self.paths['features'].open() as handle:
            for line in handle:
                row = json.loads(line)
                u = row['user_id']
                assert u not in features
                if u not in selected:
                    continue
                assert row['union_ids'][:50] == self.caches['c1'][user_indices[u]]['ranked_item_ids']
                np.testing.assert_array_equal(np.asarray(row['base'][:50]), reciprocal_scores())
                assert np.count_nonzero(row['base'][50:]) == 0
                features[u] = {'union_ids': np.asarray(row['union_ids'], dtype=np.int64),
                               'base': np.asarray(row['base'], dtype=np.float32),
                               'features': np.asarray(row['features'], dtype=np.float32)}
                if len(features) % 256 == 0:
                    self.progress(len(features), len(self.users))
                if self.mode == 'smoke' and len(features) == len(self.users):
                    break
        assert set(features) == selected
        self.features = [features[u] for u in self.users]
        self.values = {}

    def residual(self, seed):
        self.status(stage=f'residual_seed{seed}')
        ranker = BoundedResidualRanker(6, 16, .2).eval().requires_grad_(False)
        ranker.load_state_dict(torch.load(self.paths[f'residual_{seed}'], map_location='cpu'), strict=True)
        ranker.to('cuda:0')
        full, fixed = [], []
        for i, (u, features) in enumerate(zip(self.users, self.features)):
            prediction, fixed_prediction = residual_orders(features, ranker)
            if seed == 2023:
                assert prediction[0].tolist() == self.caches['gacr'][i]['ranked_item_ids'], ('candidate_parity', u)
                np.testing.assert_array_equal(prediction[1].astype(np.float64), self.caches['gacr'][i]['scores'])
            for destination, (ids, scores) in ((full, prediction), (fixed, fixed_prediction)):
                destination.append({'user_id': u, 'split': 'validation', 'gold_item_id': self.lookup[u]['target'],
                                    'ranked_item_ids': ids.tolist(), 'scores': scores.astype(float).tolist()})
            assert set(fixed[-1]['ranked_item_ids']) == set(self.caches['c1'][i]['ranked_item_ids'])
            if (i + 1) % 256 == 0 or i + 1 == len(self.users):
                self.progress(i + 1, len(self.users))
        return full, fixed

    def records(self, predictions):
        return [{'user': r['user_id'], 'history': self.lookup[r['user_id']]['history'][-20:],
                 'candidate_ids': r['ranked_item_ids'], 'seq': np.asarray(r['scores'], dtype=np.float64),
                 'target_position': r['ranked_item_ids'].index(r['gold_item_id']) if r['gold_item_id'] in r['ranked_item_ids'] else -1}
                for r in predictions]

    def cf(self, records, name):
        self.status(stage='cf_scoring_' + name)
        score_with_progress(records, self.paths['item_head'], self.config['cf_batch_size'], self.progress)
        return records

    def ranks(self, records):
        return pcrf_ranks(records, self.frequencies, {**self.config['pcrf'], 'q1': self.domain['q1']})

    def summarize(self, destination, values, comparisons):
        groups = {g: {name: metric(rank[mask]) for name, rank in values.items()}
                  for g, mask in self.masks.items() if mask.any()}
        with (destination / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', *values])
            for i, u in enumerate(self.users):
                writer.writerow([u, *[int(r[i]) for r in values.values()]])
        results = {}
        for a, b in comparisons:
            name = a + '_minus_' + b
            results[name] = {}
            for group, mask in self.masks.items():
                if not mask.any():
                    continue
                self.status(stage='descriptive_bootstrap', comparison=name, group=group)
                point = paired(values[a][mask], values[b][mask], self.config)
                for k in (5, 20, 50):
                    point[f'hit{k}_gained'] = int(((values[a][mask] <= k) & (values[b][mask] > k)).sum())
                    point[f'hit{k}_lost'] = int(((values[b][mask] <= k) & (values[a][mask] > k)).sum())
                results[name][group] = point
        summary = {'state': 'COMPLETED', 'dataset': self.dataset, 'users': len(self.users),
                   'groups': groups, 'comparisons': results, 'phase': self.phase,
                   'verification_status': 'ANALYZED', 'bootstrap_repeats': self.config['bootstrap_repeats'],
                   'bootstrap_seed': self.config['bootstrap_seed'], 'test_read': False, 'new_optimizer_steps': 0,
                   'new_beam_inference': False, 'ci_caveat': 'Reused development validation; paired user uncertainty, '
                   'not seed uncertainty or independent confirmation; no multiplicity correction.'}
        write_json(destination / 'summary.json', summary)
        return summary

    @staticmethod
    def save_predictions(path, predictions):
        with path.open('x') as handle:
            for row in predictions:
                handle.write(json.dumps(row) + '\n')

    def controls(self):
        full, fixed = self.residual(2023)
        self.c1_records = None
        for source, raw, combo in (('parent', 'original_gram', 'original_pcrf'),
                                   ('c1', 'c1', 'c1_pcrf'), ('gacr', 'gacr_v6', 'gacr_v6_pcrf')):
            records = self.cf(self.records(self.caches[source]), source)
            self.values[raw] = np.asarray([r['target_position'] + 1 if r['target_position'] >= 0 else 51 for r in records])
            self.values[combo] = self.ranks(records)
            for name in (raw, combo):
                np.testing.assert_array_equal(self.values[name], [int(self.reference[u][name]) for u in self.users])
            if source == 'parent':
                parent_records = records
            if source == 'c1':
                self.c1_records = records
        write_json(self.output / 'parity.json', {'state': 'PASSED', 'users': len(self.users),
            'seed2023_union_order_and_scores_exact': True, 'six_original_arms_rank_exact': True})
        for name, records in (('original_gram_reciprocal_pcrf', parent_records), ('c1_reciprocal_pcrf', self.c1_records)):
            self.values[name] = self.ranks([{**r, 'seq': reciprocal_scores()} for r in records])
        self.values['c1_fixed_candidates_residual_pcrf'] = self.ranks(remap_fixed_records(self.c1_records, fixed))
        if 'controls_reference' in self.paths:
            with self.paths['controls_reference'].open() as handle:
                old = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
            for name in self.values:
                np.testing.assert_array_equal(self.values[name], [int(old[u][name]) for u in self.users])
        if self.mode == 'smoke':
            # No performance report or new outcomes from an engineering subset.
            for seed in (2024, 2025):
                self.residual(seed)
            return
        directory = self.output / 'controls_seed2023'
        directory.mkdir()
        self.save_predictions(directory / 'fixed.predictions.jsonl', fixed)
        # Explicit candidate alignment for later reuse; full-union CF is not inferred from this cache.
        np.savez(directory / 'c1_cf.npz', users=np.asarray(self.users),
                 candidate_ids=np.asarray([r['candidate_ids'] for r in self.c1_records]),
                 scores=np.stack([r['cf'] for r in self.c1_records]))
        result = self.summarize(directory, self.values, CONTROL_PAIRS)
        self.seed_metrics = {'2023': {'full': result['groups']['all_validation']['gacr_v6_pcrf'],
                                     'fixed': result['groups']['all_validation']['c1_fixed_candidates_residual_pcrf']}}
        self.completed.append(self.phase)

    def new_seed(self, seed):
        self.begin(f'seed{seed}')
        directory = self.output / f'seed{seed}'
        directory.mkdir()
        full, fixed = self.residual(seed)
        records = self.cf(self.records(full), f'full_seed{seed}')
        full_ranks, fixed_ranks = self.ranks(records), self.ranks(remap_fixed_records(self.c1_records, fixed))
        np.testing.assert_array_equal(fixed_ranks <= 50, self.values['c1'] <= 50)
        raw = np.asarray([r['target_position'] + 1 if r['target_position'] >= 0 else 51 for r in records])
        np.testing.assert_array_equal(full_ranks <= 50, raw <= 50)
        self.save_predictions(directory / 'full.predictions.jsonl', full)
        self.save_predictions(directory / 'fixed.predictions.jsonl', fixed)
        values = {'original_pcrf': self.values['original_pcrf'], 'full': full_ranks, 'fixed': fixed_ranks,
                  'gacr_raw': raw, 'seed2023_full': self.values['gacr_v6_pcrf'],
                  'seed2023_fixed': self.values['c1_fixed_candidates_residual_pcrf']}
        result = self.summarize(directory, values, (('full', 'original_pcrf'), ('fixed', 'original_pcrf'), ('full', 'fixed')))
        self.seed_metrics[str(seed)] = {name: result['groups']['all_validation'][name] for name in ('full', 'fixed')}
        self.completed.append(self.phase)

    def run(self):
        self.begin('controls_seed2023')
        self.load()
        self.controls()
        if self.mode == 'smoke':
            write_json(self.output / 'smoke.json', {'state': 'PASSED', 'dataset': self.dataset,
                'users': len(self.users), 'six_arm_rank_parity': True, 'seed2023_score_order_exact': True,
                'all_three_residuals_finite_fixed_candidate_sets_preserved': True,
                'toys_nine_arm_parity': 'controls_reference' in self.paths, 'new_outcomes_reported': False,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20, 'test_read': False})
            self.status('PASSED', 'smoke_finished')
            signal.alarm(0)
            return
        for seed in (2024, 2025):
            self.new_seed(seed)
        self.status(stage='final_input_integrity')
        for key, path in self.paths.items():
            assert digest(path) == self.hashes[key], ('end_input_hash_mismatch', key)
        for path, expected in self.config['source_sha256'].items():
            assert digest(ROOT / path) == expected, ('end_source_hash_mismatch', path)
        aggregate = {}
        baseline = metric(self.values['original_pcrf'])
        for method in ('full', 'fixed'):
            aggregate[method] = {}
            for key in baseline:
                if key == 'users':
                    continue
                points = np.asarray([self.seed_metrics[str(seed)][method][key] for seed in (2023, 2024, 2025)])
                aggregate[method][key] = {'mean': float(points.mean()), 'min': float(points.min()),
                    'max': float(points.max()), 'sample_sd_across_residual_seeds': float(points.std(ddof=1)),
                    'mean_delta_vs_original_pcrf': float(points.mean() - baseline[key]),
                    'positive_seed_count_vs_original_pcrf': int((points > baseline[key]).sum())}
        write_json(self.output / 'summary.json', {'state': 'COMPLETED', 'dataset': self.dataset,
            'users': len(self.users), 'seed_metrics': self.seed_metrics, 'aggregate': aggregate,
            'baseline': baseline, 'input_sha256': self.hashes, 'config_path': self.config['config_path'],
            'verification_status': 'ANALYZED', 'seed_scope': 'Residual only; C1 and item head fixed; no best-seed selection.',
            'completed_phases': self.completed, 'test_read': False, 'new_optimizer_steps': 0,
            'elapsed_seconds': round(time.monotonic() - self.started, 1)})
        self.status('COMPLETED', 'finished')
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', choices=('Beauty', 'Toys'), required=True)
    parser.add_argument('--mode', choices=('smoke', 'run'), required=True)
    parser.add_argument('--attempt', default='v1')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    evaluation = Evaluation(config, args.dataset, args.mode, args.attempt)

    def stop(signum, frame):
        raise TimeoutError('Phase hard budget exceeded' if signum == signal.SIGALRM else f'Stop signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    try:
        evaluation.status(stage='initialize_gpu')
        torch.set_num_threads(config['cpu_threads'])
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
        evaluation.run()
    except BaseException as error:
        signal.alarm(0)
        evaluation.status('STOPPED' if isinstance(error, TimeoutError) else 'FAILED', 'aborted', error=repr(error))
        (evaluation.output / 'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
