"""B1 continuation from residual seeds 2024/2025, reusing the frozen 2023 cohort.

Only Work.train/ranker/status/snapshot are reused; no CandidateBuilder or CF head
is instantiated. Old experiment sources and outputs remain immutable.
"""
import argparse
import csv
import gc
import json
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.gacr_fusion_train import Work, ROOT, read, write, json_rows, Interrupted
from experiment.phase20.gacr_fusion_core import training_scores, pairwise_losses, numpy_ranks
from experiment.phase20.gacr_controls_multiseed import group_masks
from experiment.phase20.audit_pcrf_components import metric, paired
from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase18.core.diff_data import sha256


def load_ranks(path):
    with path.open() as handle:
        records = list(csv.DictReader(handle, delimiter='\t'))
    assert len(records) == len({r['user_id'] for r in records})
    return {r['user_id']: {k: int(v) for k, v in r.items() if k != 'user_id'} for r in records}


def validate_cache(arrays, manifest):
    assert all(np.isfinite(v).all() for v in arrays.values())
    n = manifest['rows']
    assert arrays['features'].shape == (n, 50, 6)
    assert arrays['base'].shape == arrays['candidate_ids'].shape == arrays['cf'].shape == (n, 50)
    assert arrays['fit'].dtype == arrays['head'].dtype == np.dtype(bool)
    assert int(arrays['fit'].sum()) == manifest['cohort']['fit_samples']
    assert int((~arrays['fit']).sum()) == manifest['cohort']['calibration_samples'] == 256
    assert np.all((arrays['positions'] >= -1) & (arrays['positions'] < 50))
    covered = arrays['positions'] >= 0
    assert int((arrays['fit'] & covered).sum()) == manifest['fit_covered']
    assert int((arrays['fit'] & ~covered).sum()) == manifest['fit_uncovered']
    assert int((~arrays['fit'] & covered).sum()) == manifest['calibration_covered']
    assert manifest['forced_positive_insertions'] == 0 and not manifest['test_read']
    assert manifest['cohort']['fit_calibration_user_overlap'] == 0
    assert all(len(set(ids)) == 50 for ids in arrays['candidate_ids'])


class CachedWork(Work):
    def __init__(self, config, dataset, seed, mode):
        self.config, self.dataset, self.mode, self.seed = config, dataset, mode, seed
        self.domain = dict(config['domains'][dataset])
        self.domain['residual'] = self.domain['residuals'][str(seed)]
        self.path = ROOT / config['output'] / dataset / ('profile' if mode == 'profile' else f'seed{seed}')
        self.path.mkdir(parents=True, exist_ok=False)
        self.device = torch.device('cuda:0')
        self.started = self.stage_started = time.monotonic()
        self.stage, self.steps, self.snapshot_hashes = 'initialization', 0, {}
        self.candidate_progress = None

    def load_cache(self):
        self.status(stage='load_frozen_training_cache')
        manifest = read(ROOT / self.domain['cache_manifest'])
        assert manifest['cohort'] == self.domain['frozen_cohort']
        assert manifest['cache_sha256'] == self.domain['input_sha256'][self.domain['training_cache']]
        with np.load(ROOT / self.domain['training_cache']) as stored:
            arrays = {k: stored[k] for k in stored.files}
        validate_cache(arrays, manifest)
        write(self.path / 'cache_reuse.json', {'training_cache': self.domain['training_cache'],
            'cache_sha256': manifest['cache_sha256'], 'cohort': manifest['cohort'],
            'cohort_seed': 2023, 'residual_initialization_seed': self.seed,
            'new_candidate_generation': False, 'new_CF_scoring': False, 'test_read': False})
        return arrays

    def validation_cache(self):
        self.status(stage='load_frozen_validation_cache')
        rows = list(json_rows(ROOT / self.domain['validation']))
        lookup, users = {r['user_id']: r for r in rows}, [r['user_id'] for r in rows]
        assert len(lookup) == len(users) == self.domain['users']
        catalog = read(ROOT / self.domain['catalog'])
        frequencies = training_frequencies(list(json_rows(ROOT / self.domain['train'])), len(catalog['items']))
        with np.load(ROOT / self.domain['c1_cf']) as stored:
            assert stored['users'].tolist() == users
            ids, cf = stored['candidate_ids'], stored['scores']
        c1 = {r['user_id']: r['ranked_item_ids'] for r in json_rows(ROOT / self.domain['c1'])}
        features = {r['user_id']: r for r in json_rows(ROOT / self.domain['features'])}
        assert set(c1) == set(features) == set(users)
        x, base = [], []
        for i, u in enumerate(users):
            row = features[u]
            assert row['union_ids'][:50] == ids[i].tolist() == c1[u]
            assert len(set(ids[i])) == 50
            x.append(np.asarray(row['features'], dtype=np.float32))
            base.append(np.asarray(row['base'], dtype=np.float32))
        assert np.isfinite(cf).all() and all(np.isfinite(v).all() for v in x + base)
        positions = np.array([list(ids[i]).index(lookup[u]['target']) if lookup[u]['target'] in ids[i] else -1 for i, u in enumerate(users)])
        fit_users = set((ROOT / self.domain['c1_training_users']).read_text().splitlines())
        return {'users': users, 'ids': ids, 'cf': cf, 'frequencies': frequencies[ids],
                'positions': positions, 'features': x, 'base': base,
                'masks': group_masks(users, lookup, frequencies, fit_users, self.domain['q1'])}

    def score(self, cache, checkpoint, label, save=True):
        self.status(stage=f'validation_{label}')
        ranker, scores = self.ranker(checkpoint), []
        started = time.monotonic()
        with torch.no_grad():
            for i, (features, base) in enumerate(zip(cache['features'], cache['base'])):
                # Same per-user full-union FP32 GEMM shape as the frozen B1 run.
                x = torch.tensor(features, dtype=torch.float32, device=self.device)
                b = torch.tensor(base, dtype=torch.float32, device=self.device)
                scores.append((b + ranker(x))[:50].cpu().numpy().astype(np.float64))
                if (i + 1) % 256 == 0 or i + 1 == len(cache['users']):
                    self.progress(i + 1, len(cache['users']))
        g = np.stack(scores)
        raw, final, reliability = numpy_ranks(g, cache['cf'], cache['frequencies'], cache['positions'], self.domain['q1'])
        assert np.array_equal(final <= 50, cache['positions'] >= 0)
        assert np.array_equal(raw <= 50, cache['positions'] >= 0)
        if save:
            np.savez(self.path / f'{label}.validation_scores.npz', users=np.asarray(cache['users']),
                     candidate_ids=cache['ids'], residual_scores=g, reliability=reliability)
        del ranker
        return raw, final, g, time.monotonic() - started

    def profile(self):
        arrays = self.load_cache()
        cache = self.validation_cache()
        reference = load_ranks(ROOT / self.domain['known_ranks'])
        timings = {}
        # Reproduce both already-published 2023 arms, not accuracy on new arms.
        for label, checkpoint in [('B0', self.domain['residuals']['2023']), ('B1', self.domain['known_B1'])]:
            raw, final, g, seconds = self.score(cache, ROOT / checkpoint, label, save=False)
            np.testing.assert_array_equal(raw, [reference[u][label + '_raw'] for u in cache['users']])
            np.testing.assert_array_equal(final, [reference[u][label] for u in cache['users']])
            with np.load(ROOT / self.domain['known_scores'][label]) as old:
                np.testing.assert_array_equal(g, old['residual_scores'])
            timings[label] = seconds
        self.status(stage='full_fit_gradient_profile_no_optimizer')
        ranker = self.ranker(ROOT / self.domain['residuals']['2024'])
        tensors = {k: torch.as_tensor(arrays[k], device=self.device) for k in ('features', 'base', 'cf', 'frequencies', 'positions')}
        valid = arrays['fit'] & (arrays['positions'] >= 0)
        torch.cuda.synchronize(); started = time.monotonic()
        for head in (True, False):
            selected = np.flatnonzero(valid & (arrays['head'] == head))
            assert len(selected)
            for start in range(0, len(selected), self.config['training_chunk_size']):
                indices = torch.tensor(selected[start:start+self.config['training_chunk_size']], device=self.device)
                scores = training_scores(ranker, *[tensors[k][indices] for k in ('features', 'base', 'cf', 'frequencies')], self.domain['q1'], 'B1')
                loss = pairwise_losses(scores, tensors['positions'][indices]).sum() / (2 * len(selected))
                assert torch.isfinite(loss)
                loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10., error_if_nonfinite=True))
        assert gradient > 0
        torch.cuda.synchronize(); gradient_seconds = time.monotonic() - started
        assert self.steps == 0
        del ranker, tensors
        self.status(stage='bootstrap_runtime_profile_known_ranks')
        started = time.monotonic()
        paired(np.array([reference[u]['B1'] for u in cache['users']]),
               np.array([reference[u]['B0'] for u in cache['users']]), self.config)
        bootstrap_seconds = time.monotonic() - started
        # Four comparisons over five groups; full+frequency partition+user partition=3N.
        estimate = 30 * gradient_seconds + sum(timings.values()) + 12 * bootstrap_seconds + 90
        result = {'state': 'PASSED', 'dataset': self.dataset, 'seed2023_score_rank_exact_parity': True,
                  'validation_users': len(cache['users']), 'cache_manifest_verified': True,
                  'full_fit_gradient_seconds': gradient_seconds, 'gradient_norm': gradient,
                  'validation_seconds_per_arm': timings, 'bootstrap_full_population_seconds': bootstrap_seconds,
                  'estimated_seconds_per_new_seed': estimate,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                  'new_optimizer_steps': 0, 'new_beam_inference': False, 'test_read': False}
        assert estimate < self.config['task_budget_seconds'], ('projected_task_exceeds_budget', estimate)
        self.verify()
        write(self.path / 'profile.json', result)
        self.status('PASSED', stage='finished', profile=result)

    def run(self):
        assert self.seed in (2024, 2025)
        arrays = self.load_cache()
        cache = self.validation_cache()
        old = load_ranks(ROOT / self.domain['reference_ranks'][str(self.seed)])
        known = load_ranks(ROOT / self.domain['known_ranks'])
        assert set(old) == set(known) == set(cache['users'])
        raw0, final0, _, _ = self.score(cache, ROOT / self.domain['residual'], 'B0')
        np.testing.assert_array_equal(final0, [old[u]['fixed'] for u in cache['users']])
        np.testing.assert_array_equal(final0 <= 50, [known[u]['c1'] <= 50 for u in cache['users']])
        self.verify()  # Exact matched baseline must pass before any optimizer update.
        training = self.train(arrays, 'B1')
        del arrays
        gc.collect(); torch.cuda.empty_cache()
        raw1, final1, _, _ = self.score(cache, self.path / 'B1.pt', 'B1')
        values = {name: np.array([known[u][name] for u in cache['users']])
                  for name in ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'gacr_v6_pcrf')}
        np.testing.assert_array_equal(values['original_pcrf'], [old[u]['original_pcrf'] for u in cache['users']])
        values.update(seed_matched_full=np.array([old[u]['full'] for u in cache['users']]),
                      B0_raw=raw0, B0=final0, B1_raw=raw1, B1=final1)
        groups = {group: {a: metric(r[mask]) for a, r in values.items()} for group, mask in cache['masks'].items()}
        with (self.path / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', *values])
            for i, user in enumerate(cache['users']):
                writer.writerow([user, *[int(r[i]) for r in values.values()]])
        self.status(stage='paired_statistics')
        comparisons, done = {}, 0
        for a, b in [('B1', 'B0'), ('B1', 'original_pcrf'), ('B1', 'seed_matched_full'), ('B1_raw', 'B0_raw')]:
            comparison = comparisons[a + '_minus_' + b] = {}
            for group, mask in cache['masks'].items():
                point = paired(values[a][mask], values[b][mask], self.config)
                for k in (5, 20, 50):
                    point[f'hit{k}_gained'] = int(((values[a][mask] <= k) & (values[b][mask] > k)).sum())
                    point[f'hit{k}_lost'] = int(((values[b][mask] <= k) & (values[a][mask] > k)).sum())
                comparison[group] = point
                done += 1; self.progress(done, 20)
        self.verify()
        result = {'state': 'COMPLETED', 'dataset': self.dataset, 'residual_seed': self.seed,
                  'cohort_seed': 2023, 'groups': groups, 'comparisons': comparisons, 'training': training,
                  'B0_rank_parity': True, 'fixed_candidate_membership_and_hit50_preserved': True,
                  'new_optimizer_steps': self.steps, 'backbone_optimizer_steps': 0,
                  'new_beam_inference': False, 'new_CF_scoring': False, 'test_read': False,
                  'verification_status': 'ANALYZED', 'elapsed_seconds': time.monotonic() - self.started,
                  'ci_caveat': 'Repeated development validation; user-paired intervals are uncorrected and do not include training-seed uncertainty.'}
        write(self.path / 'summary.json', result)
        self.status('COMPLETED', stage='finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True, choices=('Toys', 'Beauty'))
    parser.add_argument('--seed', type=int, required=True, choices=(2023, 2024, 2025))
    parser.add_argument('--mode', choices=('profile', 'run'), required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    job = CachedWork(config, args.dataset, args.seed, args.mode)
    def stop(signum, frame):
        raise Interrupted(f'SIGNAL_{signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(config['profile_budget_seconds'] if args.mode == 'profile' else config['task_budget_seconds'])
    try:
        torch.set_num_threads(config['cpu_threads'])
        torch.cuda.set_device(0)
        free, _ = torch.cuda.mem_get_info()
        assert free / 2**20 >= config['minimum_free_mib']
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(2023)  # Cohort fixed; stochastic initialization overwritten by checkpoint.
        job.verify(); job.snapshot()
        if args.mode == 'profile':
            job.profile()
        else:
            assert all(read(ROOT / config['output'] / d / 'profile/profile.json')['state'] == 'PASSED' for d in config['execution_order'])
            job.run()
        signal.alarm(0)
    except BaseException as error:
        job.status('STOPPED' if isinstance(error, Interrupted) else 'FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
