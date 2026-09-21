"""Isolated CPU-only Toys four-arm method screen, with a bounded supervisor."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiment.phase20.conditional_reference_core import (
    Correction, arm_features, background_scores, category_vectors, frozen_fusion,
    history_features, matched_neighbors, ranks, visible_key)
from experiment.phase20.gacr_fusion_core import pairwise_losses, standardize_tensor
from experiment.phase20.personal_popularity_core import reconstruct_training

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment/phase9'))
from train_cf0_b2_item_head import CF0B2ItemHead  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def metrics(values):
    return {'users': len(values), **{f'{name}@{k}': float(np.mean(
        (values <= k) if name == 'hit' else np.where(values <= k, 1 / np.log2(values + 1), 0)))
        for k in (5, 10, 20, 50) for name in ('ndcg', 'hit')}}


def paired(a, b, config):
    n = len(a)
    delta = np.stack([(a <= 10).astype(float) - (b <= 10),
                      np.where(a <= 10, 1 / np.log2(a + 1), 0)
                      - np.where(b <= 10, 1 / np.log2(b + 1), 0)], axis=1)
    generator = np.random.default_rng(config['bootstrap_seed'])
    samples = []
    for start in range(0, config['bootstrap_repeats'], 32):
        count = min(32, config['bootstrap_repeats'] - start)
        indices = generator.integers(0, n, size=(count, n))
        samples.append(delta[indices].mean(1))
    interval = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    positive = np.sort(delta[delta[:, 1] > 0, 1])[::-1]
    return {'users': n, 'hit10_gained': int(((a <= 10) & (b > 10)).sum()),
            'hit10_lost': int(((b <= 10) & (a > 10)).sum()),
            'ndcg10_improved_users': int((delta[:, 1] > 0).sum()),
            'ndcg10_worsened_users': int((delta[:, 1] < 0).sum()),
            'top5_share_of_positive_ndcg_change': float(positive[:5].sum() / positive.sum()) if len(positive) else 0.,
            'hit@10': {'delta': float(delta[:, 0].mean()), 'ci95': interval[:, 0].tolist()},
            'ndcg@10': {'delta': float(delta[:, 1].mean()), 'ci95': interval[:, 1].tolist()}}


class Screen:
    def __init__(self, config_path, mode):
        self.config_path = config_path
        self.config = read(config_path)
        self.mode = mode
        self.path = ROOT / self.config['output'] / mode
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.steps = 0
        self.stage = 'initialization'
        self.budget = self.config['profile_budget_seconds'] if mode == 'profile' else self.config['run_budget_seconds']
        torch.set_num_threads(self.config['cpu_threads'])
        torch.set_num_interop_threads(1)
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        self.snapshot = {'config_sha256': sha(config_path),
                         'input_sha256': self.config['input_sha256'],
                         'source_sha256': self.config['source_sha256']}

    def status(self, state='RUNNING', stage=None, **extra):
        if stage:
            self.stage = stage
        elapsed = time.monotonic() - self.started
        if state == 'RUNNING' and elapsed > self.budget:
            raise TimeoutError('Predeclared worker wall-time budget reached')
        value = {'state': state, 'stage': self.stage, 'pid': os.getpid(),
                 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 'elapsed_seconds': elapsed, 'optimizer_steps': self.steps,
                 'gpu_used': False, **extra}
        write(self.path / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def progress(self, done, total):
        self.status(completed=int(done), total=int(total))

    def verify(self):
        assert sha(self.config_path) == self.snapshot['config_sha256']
        for group in ('input_sha256', 'source_sha256'):
            for name, expected in self.snapshot[group].items():
                assert sha(ROOT / name) == expected, ('Frozen input changed', name)

    def load(self):
        self.status(stage='verify_and_load_frozen_inputs')
        self.verify()
        write(self.path / 'protocol.json', self.config)
        d = self.config['data']
        self.catalog = read(ROOT / d['catalog'])
        train = list(rows(ROOT / d['train']))
        users, sequences, _, _ = reconstruct_training(train)
        frequencies = np.bincount([item for history in sequences.values() for item in history],
                                  minlength=len(self.catalog['items']))
        assert len(users) == self.config['validation_users']
        excluded = set((ROOT / d['c1_training_users']).read_text().splitlines())
        self.bank_users = [u for u in users if u not in excluded]
        self.bank_histories = [sequences[u][-20:] for u in self.bank_users]
        assert len(self.bank_users) == self.config['reference_users']
        with np.load(ROOT / d['training_cache']) as stored:
            a = {k: stored[k] for k in stored.files}
        manifest = read(ROOT / d['cache_manifest'])
        assert manifest['cache_sha256'] == sha(ROOT / d['training_cache'])
        assert manifest['candidate_jsonl_sha256'] == sha(ROOT / d['training_candidates'])
        metadata = []
        for i, record in enumerate(rows(ROOT / d['training_candidates'])):
            source = train[record['train_index']]
            assert source['user_id'] == record['user_id']
            assert source['history'] == record['history'] and source['target'] == record['target']
            assert record['union_ids'][:50] == a['candidate_ids'][i].tolist()
            assert record['target_position'] == int(a['positions'][i])
            assert (record['partition'] == 'fit') == bool(a['fit'][i])
            assert (record['target_group'] == 'head') == bool(a['head'][i])
            np.testing.assert_array_equal(np.asarray(record['features'][:50], dtype=np.float32), a['features'][i])
            metadata.append({k: record[k] for k in ('user_id', 'history', 'target', 'sample_key')})
        self.ntrain = len(metadata)
        assert self.ntrain == len(a['fit']) == 22095
        assert int(a['fit'].sum()) == 21839 and int((a['fit'] & (a['positions'] >= 0)).sum()) == 12399
        assert {r['user_id'] for r, fit in zip(metadata, a['fit']) if fit}.isdisjoint(
            {r['user_id'] for r, fit in zip(metadata, a['fit']) if not fit})
        assert all(r['user_id'] in excluded for r in metadata)
        np.testing.assert_array_equal(frequencies[a['candidate_ids']], a['frequencies'])
        self.validation = list(rows(ROOT / d['validation']))
        validation_users = [r['user_id'] for r in self.validation]
        assert len(set(validation_users)) == len(self.validation) == self.config['validation_users']
        assert all(r['history'] == sequences[r['user_id']][-20:] for r in self.validation)
        with np.load(ROOT / d['validation_cf']) as stored:
            assert stored['users'].tolist() == validation_users
            validation_ids, validation_cf = stored['candidate_ids'], stored['scores']
        with np.load(ROOT / d['validation_b1_scores']) as stored:
            assert stored['users'].tolist() == validation_users
            np.testing.assert_array_equal(stored['candidate_ids'], validation_ids)
            validation_g = stored['residual_scores']
            old_reliability = stored['reliability']
        with (ROOT / d['validation_ranks']).open() as handle:
            known = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
        self.known_ranks = np.asarray([int(known[u]['B1']) for u in validation_users])
        old_ranker = Correction(bound=.2).eval()
        old_ranker.load_state_dict(torch.load(ROOT / d['b1_checkpoint'], map_location='cpu'))
        train_g = []
        with torch.no_grad():
            for start in range(0, self.ntrain, 128):
                x = torch.from_numpy(a['features'][start:start + 128])
                base = torch.from_numpy(a['base'][start:start + 128])
                train_g.append((base + old_ranker(x)).numpy().astype(np.float64))
        train_g = np.concatenate(train_g)
        # CPU/GPU GEMM roundoff is measured on a fixed existing reference subset;
        # deployment uses the frozen saved B1 values, so its ranks stay exact.
        lookup = {u: i for i, u in enumerate(validation_users[:64])}
        errors = []
        with torch.no_grad():
            for r in rows(ROOT / d['validation_features']):
                if r['user_id'] not in lookup:
                    continue
                i = lookup[r['user_id']]
                assert r['union_ids'][:50] == validation_ids[i].tolist()
                computed = (torch.tensor(r['base'], dtype=torch.float32)
                            + old_ranker(torch.tensor(r['features'], dtype=torch.float32)))[:50].numpy()
                errors.append(float(np.max(np.abs(computed - validation_g[i]))))
                if len(errors) == 64:
                    break
        assert len(errors) == 64 and max(errors) <= 1e-6
        validation_positions = np.asarray([list(ids).index(r['target']) if r['target'] in ids else -1
                                           for ids, r in zip(validation_ids, self.validation)])
        validation_base = np.broadcast_to(1. / np.arange(1, 51), validation_ids.shape)
        self.ids = np.concatenate([a['candidate_ids'], validation_ids])
        self.cf = np.concatenate([a['cf'], validation_cf])
        self.users = [r['user_id'] for r in metadata] + validation_users
        self.histories = [r['history'] for r in metadata] + [r['history'] for r in self.validation]
        self.keys = [visible_key(u, h) for u, h in zip(self.users, self.histories)]
        g = np.concatenate([train_g, validation_g])
        self.score, self.common, self.positions, self.order, reliability = frozen_fusion(
            g, self.cf, frequencies[self.ids], np.r_[a['positions'], validation_positions],
            np.concatenate([a['base'], validation_base]), self.config['q1'])
        np.testing.assert_array_equal(reliability[self.ntrain:], old_reliability)
        np.testing.assert_array_equal(ranks(self.score[self.ntrain:], self.positions[self.ntrain:]), self.known_ranks)
        self.fit, self.head = a['fit'], a['head']
        self.target_frequencies = np.asarray([frequencies[r['target']] for r in self.validation])
        self.c1_user_mask = np.asarray([u in excluded for u in validation_users])
        self.frequencies = frequencies
        item_categories, idf = category_vectors(self.catalog['attribute_tables'][0], self.catalog['attribute_sizes'][0])
        self.qfeatures = history_features(self.histories, item_categories, frequencies, self.config['q1'])
        self.bfeatures = history_features(self.bank_histories, item_categories, frequencies, self.config['q1'])
        write(self.path / 'input_checks.json', {
            'frozen_B1_validation_rank_exact': True, 'validation_users': len(validation_users),
            'B1_cpu_reference_score_max_abs_error_64': max(errors),
            'training_g_precision': 'CPU FP32 B1 checkpoint, chunks128; identical fixed scores across arms',
            'validation_g_precision': 'Unmodified frozen FP64 copy of original FP32 B1 scores',
            'reference_users': len(self.bank_users), 'reference_disjoint_from_C1_fit_and_calibration_users': True,
            'training_rows_verified': self.ntrain, 'no_forced_positive_insertion': True,
            'category_vocabulary': len(idf), 'zero_weight_categories': int((idf == 0).sum()),
            'train_fit_covered': int((self.fit & (self.positions[:self.ntrain] >= 0)).sum()),
            'test_read': False})
        del train, a, self.catalog
        self.status(stage='frozen_reference_verified')

    def load_cf(self):
        checkpoint = torch.load(ROOT / self.config['data']['cf_checkpoint'], map_location='cpu')
        c = checkpoint['model_config']
        model = CF0B2ItemHead(c['num_items'], c['max_history'], c['d_model'], c['num_layers'],
                             c['num_heads'], c['dropout'], c['temperature_initial'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        items = torch.nn.functional.normalize(model.item_embedding.weight, dim=-1).detach().numpy().copy()
        return model, items, float(model.logit_scale.exp().clamp(max=100))

    def encode(self, model, histories):
        vectors = []
        with torch.no_grad():
            for start in range(0, len(histories), self.config['cf_batch_size']):
                batch = histories[start:start + self.config['cf_batch_size']]
                tensor = torch.zeros((len(batch), 20), dtype=torch.long)
                for i, h in enumerate(batch):
                    tensor[i, :len(h)] = torch.tensor(h)
                vectors.append(model.encode(tensor, tensor.ne(0)).numpy().copy())
                self.progress(min(start + len(batch), len(histories)), len(histories))
        return np.concatenate(vectors)

    def neighbors(self, indices=None):
        if indices is None:
            indices = np.arange(len(self.users))
        return matched_neighbors([self.users[i] for i in indices], [self.keys[i] for i in indices],
            tuple(a[indices] for a in self.qfeatures), self.bank_users, self.bfeatures,
            neighbors=self.config['neighbors'], block_width=self.config['matching_block_width'],
            batch_size=self.config['matching_batch_size'], seed=self.config['seed'], progress=self.progress)

    def train(self, features, arm, profile=False):
        torch.manual_seed(self.config['seed'])
        model = Correction(self.config['correction_bound'])
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['learning_rate'],
                                      weight_decay=self.config['weight_decay'])
        x = torch.from_numpy(features[:self.ntrain])
        base = torch.from_numpy(self.score[:self.ntrain])
        positions = torch.from_numpy(self.positions[:self.ntrain])
        valid = self.fit & (self.positions[:self.ntrain] >= 0)
        groups = [np.flatnonzero(valid & (self.head == head)) for head in (True, False)]
        curve = []
        start_time = time.monotonic()
        for step in range(1, (1 if profile else self.config['training_steps']) + 1):
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for group in groups:
                assert len(group)
                for start in range(0, len(group), self.config['training_batch_size']):
                    index = torch.from_numpy(group[start:start + self.config['training_batch_size']])
                    score = base[index] + model(x[index]).to(torch.float64)
                    loss = pairwise_losses(standardize_tensor(score), positions[index]).sum() / (2 * len(group))
                    assert torch.isfinite(loss)
                    loss.backward()
                    value += float(loss.detach())
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'], error_if_nonfinite=True))
            assert gradient > 0
            if profile:
                assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
                return time.monotonic() - start_time, gradient
            optimizer.step()
            self.steps += 1
            curve.append({'step': step, 'loss_before_step': value, 'gradient_before_clip': gradient})
            write(self.path / f'{arm}.training.json', {'arm': arm, 'curve': curve, 'selection': 'fixed_last_step'})
            self.progress(step, self.config['training_steps'])
        change = sum(float((v - initial[k]).square().sum()) for k, v in model.state_dict().items()) ** .5
        assert change > 0 and sum(p.numel() for p in model.parameters()) == 129
        torch.save(model.state_dict(), self.path / f'{arm}.pt')
        torch.save(optimizer.state_dict(), self.path / f'{arm}.optimizer.pt')
        scores = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(features), self.config['training_batch_size']):
                stop = start + self.config['training_batch_size']
                correction = model(torch.from_numpy(features[start:stop])).numpy().astype(np.float64)
                assert np.abs(correction).max() <= self.config['correction_bound']
                scores.append(self.score[start:stop] + correction)
        return np.concatenate(scores), {'steps': len(curve), 'parameter_change_l2': change,
                                        'seconds': time.monotonic() - start_time, 'curve': curve}

    def profile(self):
        self.load()
        self.status(stage='profile_matching')
        selected = []
        for b in range(3):
            pool = np.flatnonzero(self.qfeatures[2] == b)
            selected.extend(pool[np.linspace(0, len(pool) - 1, min(len(pool), 64)).astype(int)])
        selected = np.asarray(selected)
        t = time.monotonic()
        _, _, diagnostics = self.neighbors(selected)
        match_seconds = time.monotonic() - t
        assert diagnostics[:, 0].mean() >= diagnostics[:, 1].mean()
        self.status(stage='profile_cf_encoding')
        model, items, scale = self.load_cf()
        t = time.monotonic()
        # 512 longest reference histories conservatively cover Transformer cost.
        longest = sorted(range(len(self.bank_users)), key=lambda i: -len(self.bank_histories[i]))[:512]
        self.encode(model, [self.bank_histories[i] for i in longest])
        encode_seconds = time.monotonic() - t
        self.status(stage='profile_cf_cache_parity')
        vectors = self.encode(model, self.histories[self.ntrain:self.ntrain + 64])
        got = scale * np.einsum('bd,bkd->bk', vectors,
            items[self.ids[self.ntrain:self.ntrain + 64]], optimize=False)
        max_cf_error = float(np.max(np.abs(got - self.cf[self.ntrain:self.ntrain + 64])))
        assert max_cf_error <= 2e-5, max_cf_error
        self.status(stage='profile_full_fit_gradient_no_optimizer')
        x = arm_features(self.common, self.cf, self.order)
        gradient_seconds, gradient = self.train(x, 'B_plain', profile=True)
        self.verify()
        estimate = 1.5 * (match_seconds * len(self.users) / len(selected)
                          + encode_seconds * len(self.bank_users) / len(longest)
                          + gradient_seconds * self.config['training_steps'] * 3) + 300
        result = {'state': 'PASSED' if estimate <= self.config['run_budget_seconds'] else 'BUDGET_REJECTED',
                  'estimated_run_seconds': estimate, 'matching_seconds': match_seconds,
                  'matching_profile_queries': len(selected), 'cf_encode_longest512_seconds': encode_seconds,
                  'full_fit_gradient_seconds': gradient_seconds, 'gradient_norm': gradient,
                  'cf_cached_score_max_abs_error_64': max_cf_error,
                  'new_optimizer_steps': 0, 'new_method_accuracy_computed': False,
                  'B1_validation_rank_exact': True, 'gpu_used': False,
                  'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        write(self.path / 'profile.json', result)
        assert result['state'] == 'PASSED', result
        self.status('PASSED', stage='finished', profile=result)

    def run(self):
        profile = read(ROOT / self.config['output'] / 'profile/profile.json')
        assert profile['state'] == 'PASSED'
        old_protocol = read(ROOT / self.config['output'] / 'profile/protocol.json')
        assert old_protocol == self.config, 'Profile and run configuration differ'
        self.load()
        self.status(stage='match_reference_histories')
        c_ids, d_ids, diagnostics = self.neighbors()
        np.savez(self.path / 'neighbors.npz', semantic_ids=c_ids, simple_ids=d_ids,
                 diagnostics=diagnostics, bank_users=np.asarray(self.bank_users), query_keys=np.asarray(self.keys))
        self.status(stage='encode_frozen_CF_reference_bank')
        model, item_vectors, scale = self.load_cf()
        bank_vectors = self.encode(model, self.bank_histories)
        del model
        np.savez(self.path / 'reference_vectors.npz', users=np.asarray(self.bank_users), vectors=bank_vectors,
                 item_vectors=item_vectors, scale=np.asarray(scale))
        self.status(stage='score_semantic_reference')
        mc = background_scores(c_ids, bank_vectors, item_vectors, self.ids, scale, progress=self.progress)
        self.status(stage='score_simple_reference')
        md = background_scores(d_ids, bank_vectors, item_vectors, self.ids, scale, progress=self.progress)
        features = {'B_plain': arm_features(self.common, self.cf, self.order),
                    'C_semantic': arm_features(self.common, self.cf, self.order, mc),
                    'D_simple': arm_features(self.common, self.cf, self.order, md)}
        del mc, md
        np.savez(self.path / 'feature_cache.npz', **features, base_scores=self.score,
                 candidate_ids=np.take_along_axis(self.ids, self.order, axis=1), positions=self.positions,
                 fit=self.fit, head=self.head, users=np.asarray(self.users), ntrain=np.asarray(self.ntrain))
        # Initial all-arm scores must be exactly A, including every rank.
        for name, feature in features.items():
            torch.manual_seed(self.config['seed'])
            zero = Correction(self.config['correction_bound']).eval()
            with torch.no_grad():
                assert torch.count_nonzero(zero(torch.from_numpy(feature[:128]))).item() == 0
        scores = {'A_frozen': self.score}
        training = {}
        for arm in self.config['execution_order']:
            self.status(stage=f'train_{arm}')
            scores[arm], training[arm] = self.train(features[arm], arm)
        self.status(stage='full_validation_and_paired_comparisons')
        result_ranks = {name: ranks(value[self.ntrain:], self.positions[self.ntrain:]) for name, value in scores.items()}
        np.testing.assert_array_equal(result_ranks['A_frozen'], self.known_ranks)
        assert all(np.array_equal(r <= 50, self.known_ranks <= 50) for r in result_ranks.values())
        masks = {'all': np.ones(len(self.validation), dtype=bool),
                 'low_frequency': self.target_frequencies <= self.config['q1'],
                 'higher_frequency': self.target_frequencies > self.config['q1'],
                 'C1_training_users': self.c1_user_mask, 'outside_C1_training_users': ~self.c1_user_mask}
        groups = {group: {arm: metrics(r[mask]) for arm, r in result_ranks.items()} for group, mask in masks.items()}
        comparisons = {}
        for a, b in [('B_plain', 'A_frozen'), ('C_semantic', 'A_frozen'), ('D_simple', 'A_frozen'),
                     ('C_semantic', 'B_plain'), ('C_semantic', 'D_simple')]:
            comparisons[a + '_minus_' + b] = {group: paired(result_ranks[a][mask], result_ranks[b][mask], self.config)
                                               for group, mask in masks.items()}
            self.status(comparison=a + '_minus_' + b)
        all_points = groups['all']
        improvements = {b: all_points['C_semantic']['ndcg@10'] - all_points[b]['ndcg@10']
                        for b in ('A_frozen', 'B_plain', 'D_simple')}
        # This is a screen, not an automatic claim of practical/statistical gain.
        outcome = 'REVIEW_BEFORE_BEAUTY' if all(d > 0 for d in improvements.values()) else 'STOP_SEMANTIC_EXPANSION'
        with (self.path / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', *result_ranks])
            for i, r in enumerate(self.validation):
                writer.writerow([r['user_id'], *[int(v[i]) for v in result_ranks.values()]])
        np.savez(self.path / 'final_scores.npz', **scores)
        diagnostics_by_split = {}
        for name, selection in [('fit', np.flatnonzero(self.fit)),
                                 ('calibration', np.flatnonzero(~self.fit)),
                                 ('validation', np.arange(self.ntrain, len(self.users)))]:
            diagnostics_by_split[name] = {
                'rows': len(selection), 'mean_category_cosine_C': float(diagnostics[selection, 0].mean()),
                'mean_category_cosine_D': float(diagnostics[selection, 1].mean()),
                'same_reference_fraction': float(diagnostics[selection, 2].mean()),
                'zero_category_profile_fraction': float(diagnostics[selection, 3].mean()),
                'mean_absolute_standardized_nuisance_gap_C': diagnostics[selection, 4:8].mean(0).tolist(),
                'mean_absolute_standardized_nuisance_gap_D': diagnostics[selection, 8:12].mean(0).tolist()}
        self.verify()
        result = {'state': 'COMPLETED', 'screen_outcome': outcome, 'groups': groups,
                  'comparisons': comparisons, 'training': training,
                  'reference_diagnostics': diagnostics_by_split,
                  'new_optimizer_steps': self.steps, 'backbone_updates': 0, 'CF_updates': 0,
                  'new_beam_generation': False, 'test_read': False, 'gpu_used': False,
                  'automatic_Beauty': False, 'automatic_retry': False,
                  'elapsed_seconds': time.monotonic() - self.started,
                  'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  'fixed_B1_ranks_exact': True, 'fixed_candidate_hit50_preserved': True,
                  'inference_cost_note': 'Cached prototype; cold encoding and online reference lookup are not benchmarked end-to-end.',
                  'statistical_scope': 'Repeated development validation; descriptive paired-user intervals, no multiplicity correction or seed uncertainty; no causal identification.'}
        write(self.path / 'summary.json', result)
        write(self.path / 'output_manifest.json', {p.name: sha(p) for p in self.path.iterdir()
              if p.is_file() and p.name not in ('status.json', 'output_manifest.json')})
        self.report(result)
        self.status('COMPLETED', stage='finished', outcome=outcome)

    def report(self, result):
        target = ROOT / self.config['report']
        points = result['groups']['all']
        text = ['# Stage20 Toys 条件参照四臂筛选：完成结果', '',
                'Origin Skill：academic-research-suite / experiment-agent；状态 COMPLETED，描述性开发筛选。', '',
                f"首轮判定：**{result['screen_outcome']}**。该判定不自动启动Beauty或其他训练。", '',
                '| 臂 | NDCG@10 | Hit@10 | Hit@50 |', '|---|---:|---:|---:|']
        for arm, m in points.items():
            text.append(f"| {arm} | {m['ndcg@10']:.8f} | {m['hit@10']:.8f} | {m['hit@50']:.8f} |")
        text += ['', '| C语义参照相对 | ΔN10 | 描述性95%区间 | H10新增/丢失 | N10改善/变差用户 |',
                 '|---|---:|---|---|---|']
        for b in ('A_frozen', 'B_plain', 'D_simple'):
            p = result['comparisons']['C_semantic_minus_' + b]['all']
            text.append(f"| {b} | {p['ndcg@10']['delta']:+.8f} | {p['ndcg@10']['ci95']} | "
                        f"{p['hit10_gained']}/{p['hit10_lost']} | {p['ndcg10_improved_users']}/{p['ndcg10_worsened_users']} |")
        text += ['', f"CPU四线程；总计{result['elapsed_seconds']/60:.2f}分钟，三臂共{self.steps}次小模型更新，峰值RSS {result['peak_rss_mib']:.1f} MiB。", '',
                 '固定B1基线19412用户rank逐项一致，所有臂共享候选，H50保持。新模型只用合法FIT前缀训练，calibration不选步数；固定末步30。', '',
                 'C/D共用长度分层及流行度距离近邻池，在相同的30个距离块各取1名参照；C按类别相似度，D按固定随机优先级。匹配为近似匹配，剩余长度/流行度差异见summary，不能解释为识别出的因果效应。', '',
                 'validation已长期用于开发；区间不包含训练seed与选型历史的不确定性。三臂都从零修正初始化，只有129个可训练参数。该首轮不验证完整语义表示、全部邻居规则或通用方法无效。', '',
                 f"[冻结配置](../../{self.config_path.relative_to(ROOT)})、[汇总](../../{self.config['output']}/run/summary.json)、[逐用户结果](../../{self.config['output']}/run/per_user.tsv)。", '',
                 '完整八指标、低频与C1训练用户内外分组、匹配诊断、训练曲线和分数保存在对应artifacts目录；未读取test，未新增beam，未更新生成器或CF。', '']
        target.write_text('\n'.join(text))


def supervise(config_path):
    config = read(config_path)
    root = ROOT / config['output']
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'run').exists():
        raise FileExistsError('This attempt already has a run directory; no automatic retry')
    assert read(root / 'profile/profile.json')['state'] == 'PASSED'
    with (root / 'run.log').open('x') as handle:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'experiment.phase20.conditional_reference_screen',
                                  '--config', str(config_path), '--mode', 'run'],
                                 cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        while child.poll() is None:
            elapsed = time.monotonic() - started
            write(root / 'queue_status.json', {'state': 'RUNNING', 'supervisor_pid': os.getpid(),
                  'worker_pid': child.pid, 'elapsed_seconds': elapsed,
                  'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'gpu_used': False})
            if elapsed > config['run_budget_seconds']:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                write(root / 'queue_status.json', {'state': 'TIMEOUT', 'worker_pid': child.pid,
                      'elapsed_seconds': elapsed, 'automatic_retry': False})
                return 1
            time.sleep(5)
        state = 'COMPLETED' if child.returncode == 0 else 'FAILED'
        write(root / 'queue_status.json', {'state': state, 'exit_code': child.returncode,
              'worker_pid': child.pid, 'elapsed_seconds': time.monotonic() - started,
              'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'automatic_retry': False})
        return child.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase20/conditional_reference_toys_v1.json')
    parser.add_argument('--mode', choices=('profile', 'run', 'supervise'), required=True)
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if args.mode == 'supervise':
        raise SystemExit(supervise(path))
    job = Screen(path, args.mode)
    try:
        job.profile() if args.mode == 'profile' else job.run()
    except BaseException as error:
        job.status('FAILED', stage=job.stage, error=repr(error))
        (job.path / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
