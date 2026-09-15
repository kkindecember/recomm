"""Matched B1/B2 training on fixed C1 candidates, with durable train-only caches."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch

from experiment.phase18.core.diff_data import sha256
from experiment.phase18.protocol.s18_diff_gram import load_inputs, CatalogDecoder, seed_everything
from experiment.phase4.gcdh_p0 import CatalogDualHead
from experiment.phase4.gacr_s0 import BoundedResidualRanker, build_candidate_record, gt
from experiment.phase20.run_preference import Run
from experiment.phase20.gacr_features import vector_features
from experiment.phase20.gacr_controls_multiseed import score_with_progress, group_masks
from experiment.phase20.audit_pcrf_components import metric, paired
from experiment.phase20.gacr_fusion_core import cohort, training_scores, pairwise_losses, numpy_ranks

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def json_rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def write(path, payload):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Interrupted(Exception):
    pass


class CandidateBuilder:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.data_config = read(ROOT / config['data_config'])
        self.catalog, self.train, self.validation, self.manifest, self.tokenizer, self.collator = load_inputs(self.data_config)
        self.model = Run.new_model(self)
        self.decoder = CatalogDecoder(self.catalog, {'training': config['decoding']})
        self.numeric = {x: i for i, x in enumerate(self.catalog['items']) if i}
        lexical = dict(line.split(' ', 1) for line in (ROOT / config['c1_catalog_file']).read_text().splitlines())
        names = list(lexical)
        assert set(names) == set(self.numeric)
        assert all(lexical[n] == self.catalog['lexical_ids'][self.numeric[n]] for n in names)
        state = torch.load(ROOT / config['c1_checkpoint'], map_location='cpu')
        self.dual = CatalogDualHead(self.model, self.tokenizer, names, lexical, state['log_counts']).to(device)
        self.dual.load_state_dict(state, strict=True)
        self.dual.eval().requires_grad_(False)
        del state
        paths = [self.decoder.paths[self.numeric[n] - 1] for n in names]
        self.prepared = {'model': self.dual, 'collator': self.collator, 'catalog': names,
                        'item2lexid': lexical, 'encoded_candidates': paths,
                        'sequence_to_item': {tuple(p): n for p, n in zip(paths, names)}}
        self.trie = gt.Trie(paths)
        self.max_length = max(map(len, paths))
        assert not any(p.requires_grad for p in self.dual.parameters())

    def sample(self, index, split):
        data = self.train if split == 'train' else self.validation
        row = data.records[index]
        return {**data[index], 'positive_item': self.catalog['items'][row['target']],
                'sample_key': f'{row["user_id"]}:{split}:{index}',
                'history_items': [self.catalog['items'][i] for i in row['history'][-20:]]}

    @torch.no_grad()
    def candidate(self, index, split='train', sample_override=None):
        sample = sample_override if sample_override is not None else self.sample(index, split)
        inference = dict(sample, output=self.prepared['item2lexid'][self.prepared['catalog'][0]])
        batch = self.collator([inference])
        ids, mask = batch['item_text_ids'].to(self.device), batch['item_text_masks'].to(self.device)
        output = self.model.generate(input_ids=ids, attention_mask=mask, max_length=self.max_length,
            prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(self.trie), num_beams=50,
            num_return_sequences=50, return_dict_in_generate=True, output_scores=True, length_penalty=1.)
        sequences = output.sequences.detach().cpu().tolist()
        generation_scores = output.sequences_scores.detach().cpu().numpy()
        gram = [self.prepared['sequence_to_item'][tuple(s[:s.index(1) + 1])] for s in sequences]
        assert len(gram) == len(set(gram)) == 50 and np.isfinite(generation_scores).all()
        del output
        self.model.encoder.n_passages = ids.size(1)
        hidden = self.model.encoder(input_ids=ids.view(1, -1), attention_mask=mask.view(1, -1), return_dict=True)[0]
        pooled = self.dual.pool_coarse(hidden, mask, ids.shape[-1])[0]
        logits = self.dual.catalog_head(pooled)
        seen = [self.dual.item_to_index[x] for x in sample['history_items'] if x in self.dual.item_to_index]
        logits[seen] = -torch.inf
        catalog = [self.dual.catalog[i] for i in torch.topk(logits, k=50).indices.cpu().tolist()]
        union = list(dict.fromkeys(gram + catalog))
        gr, cr = {x: i + 1 for i, x in enumerate(gram)}, {x: i + 1 for i, x in enumerate(catalog)}
        indices = torch.tensor([self.dual.item_to_index[x] for x in union], device=self.device)
        features = vector_features(logits, pooled, self.dual.catalog_head.weight, indices,
                                   [gr.get(x, 0) for x in union], [cr.get(x, 0) for x in union])
        return {'union_ids': [self.numeric[x] for x in union], 'features': features.cpu().numpy(),
                'base': np.asarray([1 / gr[x] if x in gr else 0 for x in union], dtype=np.float32),
                'generation_scores': generation_scores}


class Work:
    def __init__(self, config, dataset, mode):
        self.config, self.dataset, self.mode = config, dataset, mode
        self.domain = config['domains'][dataset]
        self.path = ROOT / config['output'] / (f'{dataset}_profile_v1' if mode == 'profile' else dataset)
        self.path.mkdir(parents=True, exist_ok=False)
        self.device = torch.device('cuda:0')
        self.started = self.stage_started = time.monotonic()
        self.stage = 'initialization'
        self.steps, self.snapshot_hashes = 0, {}
        self.candidate_progress = None

    def status(self, state='RUNNING', stage=None, **extra):
        if stage is not None and stage != self.stage:
            self.stage, self.stage_started = stage, time.monotonic()
        row = {'state': state, 'dataset': self.dataset, 'mode': self.mode, 'stage': self.stage,
               'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'elapsed_seconds': round(time.monotonic() - self.started, 1),
               'stage_elapsed_seconds': round(time.monotonic() - self.stage_started, 1),
               'new_optimizer_steps': self.steps, 'backbone_optimizer_steps': 0,
               'test_read': False, **extra}
        if self.candidate_progress is not None:
            row['candidate_cache_progress'] = self.candidate_progress
        write(self.path / 'status.json', row)
        return row

    def progress(self, done, total):
        elapsed = time.monotonic() - self.stage_started
        self.status(progress={'completed': done, 'total': total},
                    stage_eta_seconds=round(elapsed * (total - done) / done, 1) if done else None)

    def verify(self):
        self.status(stage='verify_frozen_inputs')
        hashes = {**self.config['source_sha256'], **self.domain['input_sha256']}
        for path, expected in hashes.items():
            assert sha256(ROOT / path) == expected, ('changed_input', path)
        for path, expected in self.snapshot_hashes.items():
            assert sha256(ROOT / path) == expected, ('changed_loaded_source', path)

    def snapshot(self):
        paths = {ROOT / self.config['config_path']}
        for module in list(sys.modules.values()):
            raw = getattr(module, '__file__', '')
            if raw and raw.endswith('.py'):
                path = Path(raw).resolve()
                if ROOT in path.parents:
                    paths.add(path)
        for path in paths:
            relative = path.relative_to(ROOT)
            target = self.path / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            self.snapshot_hashes[str(relative)] = sha256(path)
        write(self.path / 'protocol.json', {'config': self.config, 'input_sha256': self.domain['input_sha256'],
            'loaded_source_sha256': self.snapshot_hashes, 'torch_version': torch.__version__,
            'precision': 'FP32 backbone/residual, FP64 PCRF standardization/loss', 'tf32': False,
            'test_read': False, 'backbone_optimizer_steps': 0})

    def prepare(self):
        self.status(stage='load_frozen_C1')
        transfer = read(ROOT / self.domain['transfer_config'])
        builder = CandidateBuilder(transfer, self.device)
        users = set((ROOT / transfer['c1_training_users']).read_text().splitlines())
        fit, calibration, frequencies, metadata = cohort(builder.train.records, builder.catalog, users, self.dataset)
        assert metadata['fit_samples'] == self.domain['fit_samples']
        assert metadata['calibration_samples'] == 256
        builder.prepared['heads'] = {builder.catalog['items'][i] for i in sorted(
            range(1, len(frequencies)), key=lambda i: (-frequencies[i], builder.catalog['items'][i]))[:int(np.ceil((frequencies[1:] > 0).sum() * .2))]}
        # The catalog head's saved popularity must equal prepared train-only counts.
        mapped = torch.tensor([frequencies[builder.numeric[n]] + 1 for n in builder.prepared['catalog']], dtype=torch.float32)
        torch.testing.assert_close(builder.dual.log_counts.detach().cpu(), mapped.to(self.device).log().cpu(), rtol=0, atol=0)
        write(self.path / 'cohort.json', metadata)
        self.snapshot()
        return builder, fit, calibration, frequencies, metadata

    def profile(self):
        builder, fit, calibration, frequencies, metadata = self.prepare()
        self.status(stage='candidate_engineering_parity')
        indices = list(dict.fromkeys([0, 1, 2, 3, max(range(len(builder.validation)), key=lambda i: len(builder.validation.records[i]['history']))]))
        selected = {builder.validation.records[i]['user_id']: i for i in indices}
        old_features = {r['user_id']: r for r in json_rows(ROOT / self.domain['features']) if r['user_id'] in selected}
        old_predictions = {r['user_id']: r for r in json_rows(ROOT / self.domain['c1']) if r['user_id'] in selected}
        for user, index in selected.items():
            new = builder.candidate(index, 'validation')
            old = old_features[user]
            assert new['union_ids'] == old['union_ids']
            assert new['union_ids'][:50] == old_predictions[user]['ranked_item_ids']
            np.testing.assert_allclose(new['features'], old['features'], atol=1e-6, rtol=0)
            np.testing.assert_array_equal(new['base'], np.asarray(old['base'], dtype=np.float32))
        # Exercise the final-score interface against existing B0 ranks before expensive caching.
        self.status(stage='cached_B0_full_validation_parity')
        old_cf = np.load(ROOT / self.domain['c1_cf'])
        users = [r['user_id'] for r in builder.validation.records]
        assert old_cf['users'].tolist() == users
        cf_ids, cf_scores = old_cf['candidate_ids'], old_cf['scores']
        position_by_user = {u: i for i, u in enumerate(users)}
        g = np.empty((len(users), 50), dtype=np.float64)
        ranker = self.ranker(ROOT / self.domain['residual'])
        with torch.no_grad():
            seen = set()
            for row in json_rows(ROOT / self.domain['features']):
                user = row['user_id']; i = position_by_user[user]
                assert user not in seen and row['union_ids'][:50] == cf_ids[i].tolist()
                seen.add(user)
                x = torch.tensor(row['features'], dtype=torch.float32, device=self.device)
                base = torch.tensor(row['base'], dtype=torch.float32, device=self.device)
                g[i] = (base + ranker(x))[:50].cpu().numpy().astype(np.float64)
                if len(seen) % 256 == 0:
                    self.progress(len(seen), len(users))
            assert seen == set(users)
        targets = [r['target'] for r in builder.validation.records]
        positions = np.asarray([list(cf_ids[i]).index(t) if t in cf_ids[i] else -1 for i, t in enumerate(targets)])
        _, final, _ = numpy_ranks(g, cf_scores, frequencies[cf_ids], positions, self.domain['q1'])
        with (ROOT / self.domain['control_ranks']).open() as handle:
            old_ranks = {r['user_id']: int(r['c1_fixed_candidates_residual_pcrf']) for r in csv.DictReader(handle, delimiter='\t')}
        np.testing.assert_array_equal(final, [old_ranks[u] for u in users])
        del ranker, g, old_cf
        self.status(stage='train_prefix_parity_and_profile')
        # Deterministic history-length quantiles, not selected on recommendation outcomes.
        ordered = sorted(fit, key=lambda r: (len(builder.train.records[r['index']]['history']), r['sample_key']))
        selected_fit = [ordered[i] for i in np.linspace(0, len(ordered) - 1, self.config['profile_samples'], dtype=int)]
        times, history_lengths = [], []
        for i, row in enumerate(selected_fit):
            torch.cuda.synchronize()
            start = time.monotonic()
            candidate = builder.candidate(row['index'])
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            history_lengths.append(len(builder.train.records[row['index']]['history']))
            if i == 0:
                sample = builder.sample(row['index'], 'train')
                original = build_candidate_record(sample, builder.prepared, builder.config, self.device)
                assert candidate['union_ids'] == [builder.numeric[n] for n in original['union']]
                np.testing.assert_allclose(candidate['features'], original['features'].cpu().numpy(), rtol=0, atol=1e-6)
                alternative = dict(sample, positive_item=builder.prepared['catalog'][0], output=builder.prepared['item2lexid'][builder.prepared['catalog'][0]])
                changed = builder.candidate(row['index'], sample_override=alternative)
                assert changed['union_ids'] == candidate['union_ids']
                np.testing.assert_array_equal(changed['features'], candidate['features'])
            self.progress(i + 1, len(selected_fit))
        estimate = float(np.mean(times)) * (len(fit) + len(calibration))
        # This is a resource projection only; no task/performance screening on these users.
        result = {'state': 'PASSED', 'dataset': self.dataset, 'validation_candidate_parity_users': len(indices),
                  'B0_full_validation_rank_parity_users': len(users),
                  'train_original_builder_parity': True, 'candidate_target_independence': True,
                  'history_lengths': history_lengths, 'candidate_seconds': times,
                  'candidate_estimated_seconds': estimate, 'candidate_budget_seconds': self.config['candidate_budget_seconds'],
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                  'cohort': metadata, 'new_optimizer_steps': 0, 'test_read': False,
                  'performance_metrics_reported': False}
        assert estimate <= self.config['candidate_budget_seconds'], 'Profile projection exceeds frozen candidate budget'
        self.verify()
        write(self.path / 'profile.json', result)
        self.status('PASSED', stage='finished', profile=result)

    def cache(self, builder, fit, calibration, frequencies):
        self.status(stage='train_candidate_cache')
        cohort_rows = [('fit', r) for r in fit] + [('calibration', r) for r in calibration]
        records, features, bases, positions, groups, split = [], [], [], [], [], []
        stage_start = time.monotonic()
        with (self.path / 'train_candidates.jsonl').open('x') as handle:
            for i, (partition, row) in enumerate(cohort_rows):
                if time.monotonic() - stage_start > self.config['candidate_budget_seconds']:
                    raise Interrupted('Candidate generation hard budget reached')
                result = builder.candidate(row['index'])
                source = builder.train.records[row['index']]
                ids = result['union_ids'][:50]
                position = ids.index(source['target']) if source['target'] in ids else -1
                record = {'user': source['user_id'], 'history': source['history'], 'candidate_ids': ids,
                          'sample_key': row['sample_key'], 'target_position': position}
                records.append(record)
                features.append(result['features'][:50]); bases.append(result['base'][:50])
                positions.append(position); groups.append(row['group'] == 'head'); split.append(partition == 'fit')
                handle.write(json.dumps({'split': 'train', 'partition': partition, 'train_index': row['index'],
                    'user_id': source['user_id'], 'sample_key': row['sample_key'], 'history': source['history'],
                    'target': source['target'], 'target_position': position, 'target_group': row['group'],
                    'union_ids': result['union_ids'], 'features': result['features'].tolist(),
                    'base': result['base'].tolist(), 'generation_scores': result['generation_scores'].tolist()}) + '\n')
                if i % 16 == 0 or i + 1 == len(cohort_rows):
                    handle.flush()
                    self.candidate_progress = {'completed': i + 1, 'total': len(cohort_rows)}
                    self.progress(i + 1, len(cohort_rows))
        return records, {'features': np.stack(features), 'base': np.stack(bases), 'positions': np.asarray(positions),
                         'head': np.asarray(groups), 'fit': np.asarray(split),
                         'candidate_ids': np.asarray([r['candidate_ids'] for r in records]),
                         'frequencies': np.asarray([frequencies[r['candidate_ids']] for r in records])}

    def ranker(self, path):
        model = BoundedResidualRanker(6, 16, .2).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device), strict=True)
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == 129
        return model

    def train(self, arrays, arm):
        self.status(stage=f'train_{arm}')
        ranker = self.ranker(ROOT / self.domain['residual'])
        initial = {k: v.detach().clone() for k, v in ranker.state_dict().items()}
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=.01, weight_decay=.01)
        tensors = {k: torch.as_tensor(arrays[k], device=self.device) for k in ('features', 'base', 'cf', 'frequencies', 'positions')}
        valid = arrays['fit'] & (arrays['positions'] >= 0)
        partitions = [np.flatnonzero(valid & (arrays['head'] == head)) for head in (True, False)]
        assert all(len(p) for p in partitions), 'No covered head/tail group'
        curve = []
        for step in range(1, self.config['training_steps'] + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for indices in partitions:
                for start in range(0, len(indices), self.config['training_chunk_size']):
                    index = torch.as_tensor(indices[start:start + self.config['training_chunk_size']], device=self.device)
                    args = [tensors[k][index] for k in ('features', 'base', 'cf', 'frequencies')]
                    scores = training_scores(ranker, *args, self.domain['q1'], arm)
                    loss = pairwise_losses(scores, tensors['positions'][index]).sum() / (2 * len(indices))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'{arm} nonfinite loss')
                    loss.backward()
                    loss_value += float(loss.detach())
            gradient = torch.nn.utils.clip_grad_norm_(ranker.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            self.steps += 1
            with torch.no_grad():
                change = sum(float((v - initial[k]).square().sum()) for k, v in ranker.state_dict().items()) ** .5
            point = {'step': step, 'loss': loss_value, 'gradient_norm_before_clip': float(gradient), 'parameter_change_l2': change}
            curve.append(point)
            write(self.path / f'{arm}.training.json', {'state': 'RUNNING', 'arm': arm, 'curve': curve,
                'covered_head': len(partitions[0]), 'covered_tail': len(partitions[1]), 'initialization': self.domain['residual']})
            self.progress(step, self.config['training_steps'])
        assert curve[-1]['parameter_change_l2'] > 0
        path = self.path / f'{arm}.pt'
        torch.save({k: v.detach().cpu() for k, v in ranker.state_dict().items()}, path)
        # Optimizer is separate from deployment weights, and all state tensors are CPU.
        state = optimizer.state_dict()
        for item in state['state'].values():
            for k, v in item.items():
                if torch.is_tensor(v):
                    item[k] = v.detach().cpu()
        torch.save({'optimizer': state, 'step': self.config['training_steps'], 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, self.path / f'{arm}.optimizer.pt')
        health = {'state': 'COMPLETED', 'arm': arm, 'curve': curve, 'optimizer_steps': self.config['training_steps'],
                  'covered_head': len(partitions[0]), 'covered_tail': len(partitions[1]),
                  'initialization_sha256': sha256(ROOT / self.domain['residual']), 'checkpoint_sha256': sha256(path),
                  'checkpoint_selection': 'fixed_last_step', 'test_read': False}
        calibration = np.flatnonzero(~arrays['fit'])
        with torch.no_grad():
            g = (tensors['base'][calibration] + ranker(tensors['features'][calibration])).cpu().numpy().astype(np.float64)
        raw, final, _ = numpy_ranks(g, arrays['cf'][calibration], arrays['frequencies'][calibration], arrays['positions'][calibration], self.domain['q1'])
        health['train_only_calibration'] = {'raw': metric(raw), 'pcrf': metric(final), 'used_for_selection': False}
        write(self.path / f'{arm}.training.json', health)
        del ranker, optimizer, tensors
        return health

    def evaluate(self, frequencies):
        self.status(stage='load_fixed_validation_cache')
        rows = list(json_rows(ROOT / self.domain['validation']))
        lookup = {r['user_id']: r for r in rows}
        users = [r['user_id'] for r in rows]
        with (ROOT / self.domain['control_ranks']).open() as handle:
            reference = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
        assert set(reference) == set(users) and len(users) == self.domain['users']
        old = np.load(ROOT / self.domain['c1_cf'])
        assert old['users'].tolist() == users
        ids, cf = old['candidate_ids'], old['scores']
        c1 = {r['user_id']: r for r in json_rows(ROOT / self.domain['c1'])}
        assert all(ids[i].tolist() == c1[u]['ranked_item_ids'] for i, u in enumerate(users))
        features = {r['user_id']: r for r in json_rows(ROOT / self.domain['features'])}
        positions = np.asarray([list(ids[i]).index(lookup[u]['target']) if lookup[u]['target'] in ids[i] else -1 for i, u in enumerate(users)])
        population = frequencies[ids]
        values = {name: np.asarray([int(reference[u][name]) for u in users]) for name in ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'gacr_v6_pcrf')}
        outputs = {}
        for arm in ('B0', 'B1', 'B2'):
            self.status(stage=f'validation_{arm}')
            ranker = self.ranker(ROOT / self.domain['residual'] if arm == 'B0' else self.path / f'{arm}.pt')
            g = []
            with torch.no_grad():
                for i, u in enumerate(users):
                    row = features[u]
                    assert row['union_ids'][:50] == ids[i].tolist()
                    # Preserve original full-union GEMM shape before retaining fixed50.
                    x = torch.tensor(row['features'], dtype=torch.float32, device=self.device)
                    base = torch.tensor(row['base'], dtype=torch.float32, device=self.device)
                    g.append((base + ranker(x))[:50].cpu().numpy().astype(np.float64))
                    if (i + 1) % 256 == 0 or i + 1 == len(users):
                        self.progress(i + 1, len(users))
            raw, final, reliability = numpy_ranks(np.stack(g), cf, population, positions, self.domain['q1'])
            assert np.array_equal(final <= 50, values['c1'] <= 50)
            if arm == 'B0':
                np.testing.assert_array_equal(final, [int(reference[u]['c1_fixed_candidates_residual_pcrf']) for u in users])
            values[arm + '_raw'], values[arm] = raw, final
            np.savez(self.path / f'{arm}.validation_scores.npz', users=np.asarray(users), candidate_ids=ids,
                     residual_scores=np.stack(g), reliability=reliability)
            outputs[arm] = {'raw': metric(raw), 'pcrf': metric(final)}
            del ranker
        self.status(stage='paired_statistics')
        fit_users = set((ROOT / self.domain['c1_training_users']).read_text().splitlines())
        masks = group_masks(users, lookup, frequencies, fit_users, self.domain['q1'])
        groups = {group: {name: metric(r[mask]) for name, r in values.items()} for group, mask in masks.items()}
        comparisons = {}
        pairs = [('B1', 'B0'), ('B2', 'B0'), ('B2', 'B1'), ('B1', 'original_pcrf'), ('B2', 'original_pcrf'),
                 ('B1', 'gacr_v6_pcrf'), ('B2', 'gacr_v6_pcrf'), ('B1_raw', 'B0_raw'), ('B2_raw', 'B0_raw')]
        for a, b in pairs:
            comparison = comparisons[a + '_minus_' + b] = {}
            for group, mask in masks.items():
                point = paired(values[a][mask], values[b][mask], self.config)
                for k in (5, 20, 50):
                    point[f'hit{k}_gained'] = int(((values[a][mask] <= k) & (values[b][mask] > k)).sum())
                    point[f'hit{k}_lost'] = int(((values[b][mask] <= k) & (values[a][mask] > k)).sum())
                comparison[group] = point
                self.status(comparison=a + '_minus_' + b, group=group)
        with (self.path / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', *values])
            for i, u in enumerate(users):
                writer.writerow([u, *[int(r[i]) for r in values.values()]])
        return {'groups': groups, 'comparisons': comparisons, 'outputs': outputs,
                'B0_rank_parity': True, 'fixed_candidate_membership_and_hit50_preserved': True,
                'ci_caveat': 'Repeated development validation, paired user uncertainty only; no multiplicity correction or independent training-seed confirmation.'}

    def run(self):
        builder, fit, calibration, frequencies, metadata = self.prepare()
        records, arrays = self.cache(builder, fit, calibration, frequencies)
        del builder
        gc.collect(); torch.cuda.empty_cache()
        self.status(stage='train_candidate_CF_scores')
        score_with_progress(records, ROOT / self.domain['item_head'], 256, self.progress)
        arrays['cf'] = np.stack([r['cf'] for r in records])
        assert np.isfinite(arrays['features']).all() and np.isfinite(arrays['cf']).all()
        np.savez(self.path / 'training_cache.npz', **arrays)
        cache_manifest = {'state': 'COMPLETED', 'rows': len(records), 'cohort': metadata,
            'fit_covered': int((arrays['fit'] & (arrays['positions'] >= 0)).sum()),
            'fit_uncovered': int((arrays['fit'] & (arrays['positions'] < 0)).sum()),
            'calibration_covered': int((~arrays['fit'] & (arrays['positions'] >= 0)).sum()),
            'candidate_count': 50, 'forced_positive_insertions': 0,
            'cache_sha256': sha256(self.path / 'training_cache.npz'),
            'candidate_jsonl_sha256': sha256(self.path / 'train_candidates.jsonl'),
            'test_read': False, 'training_targets_used_only_as_labels': True}
        write(self.path / 'cache_manifest.json', cache_manifest)
        # Candidate generation can take hours. Recheck sources/inputs before any optimizer.
        self.verify()
        training = {arm: self.train(arrays, arm) for arm in ('B1', 'B2')}
        del arrays, records
        gc.collect(); torch.cuda.empty_cache()
        result = self.evaluate(frequencies)
        self.verify()
        result.update(state='COMPLETED', dataset=self.dataset, training=training, cache=cache_manifest,
                      new_optimizer_steps=self.steps, backbone_optimizer_steps=0, test_read=False,
                      elapsed_seconds=time.monotonic() - self.started, verification_status='ANALYZED')
        write(self.path / 'summary.json', result)
        self.status('COMPLETED', stage='finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True, choices=('Toys', 'Beauty'))
    parser.add_argument('--mode', required=True, choices=('profile', 'run'))
    args = parser.parse_args()
    cfg = read(ROOT / args.config)
    job = Work(cfg, args.dataset, args.mode)
    def stop(signum, frame):
        raise Interrupted(f'SIGNAL_{signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(cfg['profile_budget_seconds'] if args.mode == 'profile' else cfg['domain_budget_seconds'])
    try:
        torch.set_num_threads(cfg['cpu_threads'])
        torch.cuda.set_device(0)
        free, _ = torch.cuda.mem_get_info()
        assert free / 2**20 >= cfg['minimum_free_mib']
        torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        seed_everything(2023)
        job.verify()
        if args.mode == 'profile':
            job.profile()
        else:
            assert read(ROOT / cfg['output'] / f'{args.dataset}_profile_v1/profile.json')['state'] == 'PASSED'
            job.run()
        signal.alarm(0)
    except BaseException as error:
        job.status('STOPPED' if isinstance(error, Interrupted) else 'FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
