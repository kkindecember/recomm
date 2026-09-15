"""Toys complete-list B1 -> C1 decoder distillation; immutable three-arm pilot."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from experiment.phase20.run_preference import Run
from experiment.phase18.protocol.s18_diff_gram import load_inputs, CatalogDecoder, seed_everything
from experiment.phase18.core.diff_data import sha256
from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase20.gacr_fusion_train import read, write, json_rows, Interrupted
from experiment.phase20.gacr_b1_multiseed import load_ranks
from experiment.phase20.gacr_controls_multiseed import group_masks, score_with_progress
from experiment.phase20.audit_pcrf_components import metric, paired
from experiment.phase9.eval_cf0_b3_beamfusion import standardize
from experiment.phase9.eval_cf0_b4_reliability import rank_matrix
from experiment.phase20.preference_core import path_labels
from experiment.phase20.gacr_list_distill_core import (
    teacher_distribution, freeze_encoder_and_embeddings, encode_history,
    all_scores, sample_backward,
)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('CE_anchor', 'true_teacher', 'shuffled_teacher')


def tensors_hash(items):
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Work:
    def __init__(self, config, mode, arm=None):
        self.config, self.mode, self.arm = config, mode, arm
        self.output = ROOT / config['output']
        self.path = self.output / (mode if mode in ('profile', 'cache', 'aggregate') else f'{arm}/{mode}')
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = self.stage_started = time.monotonic()
        self.stage, self.steps, self.engineering_steps = 'initialization', 0, 0
        self.device = torch.device('cuda:0')
        self.model, self.snapshot_hashes = None, {}

    def status(self, state='RUNNING', stage=None, **extra):
        if stage is not None and stage != self.stage:
            self.stage, self.stage_started = stage, time.monotonic()
        write(self.path / 'status.json', {'state': state, 'mode': self.mode, 'arm': self.arm,
              'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'elapsed_seconds': round(time.monotonic()-self.started, 1), 'stage': self.stage,
              'stage_elapsed_seconds': round(time.monotonic()-self.stage_started, 1),
              'new_optimizer_steps': self.steps, 'engineering_optimizer_steps': self.engineering_steps,
              'test_read': False, **extra})

    def progress(self, done, total):
        elapsed = time.monotonic()-self.stage_started
        self.status(progress={'completed': done, 'total': total},
                    stage_eta_seconds=round(elapsed*(total-done)/done, 1) if done else None)

    def verify(self):
        self.status(stage='verify_frozen_sources_inputs')
        for path, expected in {**self.config['source_sha256'], **self.config['input_sha256'], **self.snapshot_hashes}.items():
            assert sha256(ROOT / path) == expected, ('changed_input_or_source', path)

    def snapshot(self):
        sources = {ROOT / self.config['config_path']}
        for module in list(sys.modules.values()):
            name = getattr(module, '__file__', '')
            if name and name.endswith('.py'):
                p = Path(name).resolve()
                if ROOT in p.parents:
                    sources.add(p)
        for p in sources:
            relative = p.relative_to(ROOT)
            target = self.path / 'sources' / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target); self.snapshot_hashes[str(relative)] = sha256(p)
        write(self.path / 'protocol.json', {'config': self.config, 'loaded_source_sha256': self.snapshot_hashes,
            'torch_version': torch.__version__, 'numpy_version': np.__version__, 'precision': 'FP32 GRAM, FP64 list probability/loss',
            'tf32': False, 'dropout': False, 'test_read': False, 'student_initialization': self.config['c1_checkpoint']})

    def data(self):
        self.status(stage='load_train_validation_and_frozen_lists')
        self.data_config = read(ROOT / self.config['data_config'])
        self.catalog, self.train, self.validation, self.manifest, self.tokenizer, self.collator = load_inputs(self.data_config)
        self.decoder = CatalogDecoder(self.catalog, {'training': self.config['decoding']})
        self.frequency = training_frequencies(self.train.records, len(self.catalog['items']))
        self.records = list(json_rows(ROOT / self.config['train_candidate_jsonl']))
        with np.load(ROOT / self.config['training_cache']) as stored:
            self.arrays = {k: stored[k] for k in stored.files}
        cm = read(ROOT / self.config['cache_manifest'])
        assert cm['cohort'] == self.config['frozen_cohort'] and len(self.records) == cm['rows']
        assert sha256(ROOT / self.config['training_cache']) == cm['cache_sha256']
        assert sha256(ROOT / self.config['train_candidate_jsonl']) == cm['candidate_jsonl_sha256']
        for i, r in enumerate(self.records):
            source = self.train.records[r['train_index']]
            assert r['split'] == 'train' and all(r[k] == source[k] for k in ('user_id', 'history', 'target'))
            assert r['union_ids'][:50] == self.arrays['candidate_ids'][i].tolist()
            assert r['target_position'] == self.arrays['positions'][i]
            assert (r['partition'] == 'fit') == self.arrays['fit'][i]
        assert int(self.arrays['fit'].sum()) == self.config['fit_samples']
        assert np.array_equal(self.arrays['frequencies'], self.frequency[self.arrays['candidate_ids']])
        self.settings = dict(self.config['loss'], candidate_chunk=self.config['candidate_chunk'],
            covered_normalizer=self.config['fit_samples']/int((self.arrays['fit'] & (self.arrays['positions'] >= 0)).sum()))

    def load_model(self):
        self.status(stage='load_frozen_C1_and_decoder_parameters')
        # Reuse the parent constructor/config, then strictly replace every backbone
        # tensor with C1. The catalog head is not attached to the student.
        self.model = Run.new_model(self)
        state = torch.load(ROOT / self.config['c1_checkpoint'], map_location='cpu')
        backbone = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
        assert backbone
        self.model.load_state_dict(backbone, strict=True)
        del state, backbone
        self.trainable = freeze_encoder_and_embeddings(self.model)
        self.frozen_hash = tensors_hash((n, p) for n, p in self.model.named_parameters() if not p.requires_grad)
        write(self.path / 'parameter_contract.json', {'trainable_names': self.trainable,
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'total_parameters': sum(p.numel() for p in self.model.parameters()), 'frozen_parameters_sha256': self.frozen_hash,
            'encoder_frozen': True, 'shared_embeddings_and_lm_head_frozen': True,
            'C1_checkpoint_sha256': self.config['input_sha256'][self.config['c1_checkpoint']]})

    def frozen_unchanged(self):
        assert self.frozen_hash == tensors_hash((n, p) for n, p in self.model.named_parameters() if not p.requires_grad)
        assert all(p.grad is None for p in self.model.parameters() if not p.requires_grad)

    def batch(self, index):
        return self.collator([self.train[self.records[index]['train_index']]])

    def teacher(self, indices):
        ranker = BoundedResidualRanker(6, 16, .2).to(self.device).eval().requires_grad_(False)
        ranker.load_state_dict(torch.load(ROOT / self.config['teacher_B1'], map_location=self.device), strict=True)
        true, shuffled = [], []
        with torch.no_grad():
            for i in indices:
                r = self.records[i]
                x = torch.tensor(r['features'], dtype=torch.float32, device=self.device)
                b = torch.tensor(r['base'], dtype=torch.float32, device=self.device)
                g = (b + ranker(x))[:50].cpu().numpy().astype(np.float64)
                p, q = teacher_distribution(g, self.arrays['cf'][i], self.arrays['frequencies'][i],
                    r['sample_key'], self.config['pcrf']['q1'], self.settings['temperature'], self.config['shuffle_seed'])
                true.append(p); shuffled.append(q)
        del ranker
        return np.asarray(true), np.asarray(shuffled)

    def parent_scores(self, index):
        batch = self.batch(index)
        hidden, mask = encode_history(self.model, batch, self.device)
        labels = path_labels(self.decoder.paths, self.arrays['candidate_ids'][index].tolist(), self.device)
        scores = all_scores(self.model, hidden, mask, labels, self.config['candidate_chunk'])
        return scores.cpu().numpy()

    def save_parameters(self, path, **extra):
        state = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        torch.save({'decoder_parameters': state, 'trainable_names': self.trainable, 'step': self.steps,
                    'C1_checkpoint_sha256': self.config['input_sha256'][self.config['c1_checkpoint']],
                    'config_sha256': sha256(ROOT / self.config['config_path']), 'arm': self.arm,
                    'frozen_parameters_sha256': self.frozen_hash, **extra}, path)

    def load_parameters(self, path):
        payload = torch.load(path, map_location='cpu')
        assert payload['C1_checkpoint_sha256'] == self.config['input_sha256'][self.config['c1_checkpoint']]
        assert payload['config_sha256'] == sha256(ROOT / self.config['config_path'])
        assert payload['trainable_names'] == self.trainable
        assert set(payload['decoder_parameters']) == set(self.trainable)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(payload['decoder_parameters'][name])
        self.frozen_unchanged()
        return payload

    def profile(self):
        self.data(); self.load_model()
        fit = np.flatnonzero(self.arrays['fit'])
        ordered = sorted(fit.tolist(), key=lambda i: (len(self.records[i]['history']), self.records[i]['sample_key']))
        indices = [ordered[j] for j in np.linspace(0, len(ordered)-1, self.config['profile_samples']).astype(int)]
        assert max(len(self.records[i]['history']) for i in indices) == max(len(r['history']) for r in self.records)
        true, shuffled = self.teacher(indices)
        self.status(stage='parent_list_scoring_profile')
        parents, cache_times, errors = [], [], []
        for i in indices:
            torch.cuda.synchronize(); started = time.monotonic()
            scores = self.parent_scores(i)
            torch.cuda.synchronize(); cache_times.append(time.monotonic()-started); parents.append(scores)
            error = float(np.max(np.abs(scores - np.asarray(self.records[i]['generation_scores']))))
            assert error <= self.config['native_teacher_forcing_score_atol'], ('complete_score_normalization', i, error)
            errors.append(error)
        self.status(stage='frozen_validation_beam_parity')
        native = {r['user_id']: r for r in json_rows(ROOT / self.config['c1_predictions'])}
        order = sorted(range(len(self.validation)), key=lambda i: (len(self.validation.records[i]['history']), self.validation.records[i]['user_id']))
        val_indices = [order[j] for j in np.linspace(0, len(order)-1, self.config['profile_samples']).astype(int)]
        inference_times = []
        with torch.no_grad():
            for i in val_indices:
                batch = self.collator([self.validation[i]])
                torch.cuda.synchronize(); started = time.monotonic()
                generated = self.decoder.generate(self.model, {'input_ids': batch['item_text_ids'].to(self.device),
                    'attention_mask': batch['item_text_masks'].to(self.device)})[0]
                torch.cuda.synchronize(); inference_times.append(time.monotonic()-started)
                reference = native[self.validation.records[i]['user_id']]
                assert generated[0] == reference['ranked_item_ids']
                np.testing.assert_allclose(generated[1], reference['scores'], rtol=0, atol=1e-6)
        self.status(stage='exact_replay_gradient_and_optimizer_profile')
        timings, norms = {}, {}
        for arm in ARMS:
            times, magnitudes = [], []
            for j, i in enumerate(indices):
                self.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); started = time.monotonic()
                sample_backward(self.model, self.batch(i), self.decoder.paths, self.arrays['candidate_ids'][i].tolist(),
                    parents[j], shuffled[j] if arm == 'shuffled_teacher' else true[j], self.arrays['positions'][i] >= 0,
                    arm, self.settings, self.device)
                norm = float(torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
                assert norm > 0
                torch.cuda.synchronize(); times.append(time.monotonic()-started); magnitudes.append(norm)
                self.frozen_unchanged()
            timings[arm], norms[arm] = times, magnitudes
        initial = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.config['learning_rate'], weight_decay=.01)
        optimizer.step(); self.engineering_steps += 1
        self.frozen_unchanged()
        assert any(not torch.equal(initial[n], p.detach().cpu()) for n, p in self.model.named_parameters() if p.requires_grad)
        path = self.path / 'engineering_update.pt'
        self.save_parameters(path, discarded_engineering_update=True)
        self.load_parameters(path)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(initial[n])
        del optimizer, initial
        self.model.zero_grad(set_to_none=True); self.frozen_unchanged()
        training_seconds = {a: float(np.mean(xs))*self.config['fit_samples'] + 300 for a, xs in timings.items()}
        cache_seconds = float(np.mean(cache_times))*len(self.records) + 300
        evaluation_seconds = float(np.mean(inference_times))*len(self.validation) + self.config['evaluation_cpu_allowance_seconds']
        result = {'state': 'PASSED', 'dataset': self.config['dataset'], 'sampled_train_indices': indices,
            'sampled_validation_indices': val_indices, 'history_lengths': [len(self.records[i]['history']) for i in indices],
            'native_beam_parity_users': len(val_indices), 'native_beam_ids_exact': True, 'native_beam_score_atol': 1e-6,
            'teacher_forcing_vs_native_max_abs_errors': errors, 'teacher_forcing_native_exact_identity': False,
            'parent_cache_seconds_per_example': cache_times, 'training_seconds_per_example': timings, 'gradient_norms': norms,
            'estimated_cache_seconds': cache_seconds, 'estimated_training_seconds': training_seconds,
            'estimated_evaluation_seconds_per_arm': evaluation_seconds,
            'estimated_queue_seconds_after_profile': cache_seconds + sum(training_seconds.values()) + 3*evaluation_seconds,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
            'frozen_parameters_unchanged': True, 'engineering_optimizer_steps': 1,
            'formal_optimizer_steps': 0, 'discarded_engineering_update': True, 'test_read': False}
        violations = [name for name, estimate, bound in (
            ('cache', cache_seconds, self.config['cache_budget_seconds']),
            ('train', max(training_seconds.values()), self.config['train_budget_seconds']),
            ('evaluate', evaluation_seconds, self.config['evaluate_budget_seconds'])) if estimate >= bound]
        result['budget_violations'] = violations
        if violations:
            result['state'] = 'STOPPED'
        self.verify(); write(self.path / 'profile.json', result)
        assert not violations, ('measured_runtime_exceeds_frozen_budget', violations)
        self.status('PASSED', stage='finished', profile=result)

    def cache(self):
        self.data(); self.load_model()
        self.status(stage='frozen_B1_teacher_distributions')
        true, shuffled = self.teacher(range(len(self.records)))
        assert np.isfinite(true).all() and np.allclose(true.sum(1), 1.)
        assert np.array_equal(np.sort(true, axis=1), np.sort(shuffled, axis=1))
        parents = np.lib.format.open_memmap(self.path / 'parent_scores.partial.npy', mode='w+', dtype=np.float32, shape=(len(self.records), 50))
        self.status(stage='frozen_C1_complete_list_cache')
        for i in range(len(self.records)):
            parents[i] = self.parent_scores(i)
            if (i+1) % 32 == 0 or i+1 == len(self.records):
                parents.flush(); self.progress(i+1, len(self.records))
        np.savez(self.path / 'distillation_cache.npz', parent_scores=np.asarray(parents), true_probability=true,
                 shuffled_probability=shuffled, candidate_ids=self.arrays['candidate_ids'], fit=self.arrays['fit'],
                 positions=self.arrays['positions'], train_indices=np.array([r['train_index'] for r in self.records]))
        self.frozen_unchanged(); self.verify()
        fit = np.flatnonzero(self.arrays['fit'])
        order = np.random.RandomState(self.config['training_order_seed']).permutation(fit)
        np.save(self.path / 'training_order.npy', order)
        result = {'state': 'COMPLETED', 'rows': len(self.records), 'fit_rows': len(fit),
            'covered_fit_rows': int((self.arrays['fit'] & (self.arrays['positions'] >= 0)).sum()),
            'source_cache_sha256': sha256(ROOT / self.config['training_cache']), 'cohort': self.config['frozen_cohort'],
            'distillation_cache_sha256': sha256(self.path / 'distillation_cache.npz'),
            'training_order_sha256': sha256(self.path / 'training_order.npy'),
            'parent_C1_sha256': self.config['input_sha256'][self.config['c1_checkpoint']],
            'teacher_B1_sha256': self.config['input_sha256'][self.config['teacher_B1']],
            'parent_anchor_source': 'FP32 teacher-forced complete sequences, not cached native beam scores',
            'native_and_teacher_forced_exact_identity': False, 'new_beam_generation': False,
            'formal_optimizer_steps': 0, 'test_read': False}
        write(self.path / 'manifest.json', result); self.status('COMPLETED', stage='finished')

    def train_arm(self):
        self.data(); self.load_model()
        cache_path = self.output / 'cache'
        manifest = read(cache_path / 'manifest.json')
        assert manifest['state'] == 'COMPLETED' and manifest['cohort'] == self.config['frozen_cohort']
        assert sha256(cache_path / 'distillation_cache.npz') == manifest['distillation_cache_sha256']
        assert sha256(cache_path / 'training_order.npy') == manifest['training_order_sha256']
        with np.load(cache_path / 'distillation_cache.npz') as stored:
            cache = {k: stored[k] for k in stored.files}
        assert np.array_equal(cache['candidate_ids'], self.arrays['candidate_ids']) and np.array_equal(cache['fit'], self.arrays['fit'])
        order = np.load(cache_path / 'training_order.npy')
        assert sorted(order.tolist()) == np.flatnonzero(self.arrays['fit']).tolist()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.config['learning_rate'], weight_decay=.01, eps=1e-8)
        steps = math.ceil(len(order)/self.config['effective_batch_size'])
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, math.ceil(steps*.05)), steps)
        curve = []
        self.status(stage='decoder_training')
        for start in range(0, len(order), self.config['effective_batch_size']):
            batch_indices = order[start:start+self.config['effective_batch_size']]
            optimizer.zero_grad(set_to_none=True)
            totals = {k: 0. for k in ('loss', 'ce', 'anchor_kl', 'teacher_kl', 'weighted_teacher', 'covered')}
            for j, i in enumerate(batch_indices):
                probability = cache['shuffled_probability' if self.arm == 'shuffled_teacher' else 'true_probability'][i]
                values = sample_backward(self.model, self.batch(i), self.decoder.paths, cache['candidate_ids'][i].tolist(),
                    cache['parent_scores'][i], probability, cache['positions'][i] >= 0, self.arm, self.settings,
                    self.device, divisor=len(batch_indices))
                for key, value in values.items():
                    totals[key] += value/len(batch_indices)
                if (j+1) % 16 == 0:
                    self.progress(start+j+1, len(order))
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            learning_rate = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step(); self.steps += 1
            point = {'step': self.steps, 'examples': min(start+len(batch_indices), len(order)),
                     'gradient_norm_before_clip': gradient, 'learning_rate': learning_rate, **totals}
            curve.append(point)
            with (self.path / 'training_curve.jsonl').open('a') as handle:
                handle.write(json.dumps(point)+'\n')
            if self.steps % 50 == 0:
                self.save_parameters(self.path / 'last_progress.pt')
            self.progress(point['examples'], len(order))
        assert self.steps == steps == self.config['optimizer_steps_per_arm']
        self.frozen_unchanged(); self.verify()
        checkpoint = self.path / 'last.pt'; self.save_parameters(checkpoint)
        state = optimizer.state_dict()
        for item in state['state'].values():
            for k, v in item.items():
                if torch.is_tensor(v): item[k] = v.detach().cpu()
        torch.save({'optimizer': state, 'scheduler': scheduler.state_dict(), 'step': self.steps,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, self.path / 'optimizer.pt')
        result = {'state': 'COMPLETED', 'arm': self.arm, 'curve': curve, 'optimizer_steps': self.steps,
            'examples': len(order), 'epochs': 1, 'checkpoint_selection': 'fixed_last_step',
            'checkpoint_sha256': sha256(checkpoint), 'frozen_parameters_unchanged': True,
            'training_order_sha256': manifest['training_order_sha256'], 'test_read': False}
        write(self.path / 'summary.json', result); self.status('COMPLETED', stage='finished')

    def evaluate(self):
        self.data(); self.load_model()
        training = read(self.output / self.arm / 'train/summary.json')
        checkpoint = self.output / self.arm / 'train/last.pt'
        assert training['state'] == 'COMPLETED' and sha256(checkpoint) == training['checkpoint_sha256']
        self.load_parameters(checkpoint)
        self.status(stage='fresh_validation_beam50')
        records, raw_ranks, overlaps = [], [], []
        frozen_c1 = {r['user_id']: set(r['ranked_item_ids']) for r in json_rows(ROOT / self.config['c1_predictions'])}
        with (self.path / 'predictions.jsonl').open('x') as handle:
            with torch.no_grad():
                for i, source in enumerate(self.validation.records):
                    batch = self.collator([self.validation[i]])
                    ids, scores = self.decoder.generate(self.model, {'input_ids': batch['item_text_ids'].to(self.device),
                        'attention_mask': batch['item_text_masks'].to(self.device)})[0]
                    position = ids.index(source['target']) if source['target'] in ids else -1
                    raw_ranks.append(position+1 if position >= 0 else 51)
                    overlaps.append(len(set(ids) & frozen_c1[source['user_id']]))
                    record = {'user': source['user_id'], 'history': source['history'], 'candidate_ids': ids,
                              'seq': np.asarray(scores, dtype=np.float64), 'target_position': position}
                    records.append(record)
                    handle.write(json.dumps({'split': 'validation', 'user_id': source['user_id'], 'gold_item_id': source['target'],
                                             'ranked_item_ids': ids, 'scores': scores})+'\n')
                    if (i+1) % 16 == 0 or i+1 == len(self.validation):
                        handle.flush(); self.progress(i+1, len(self.validation))
        inference_elapsed = time.monotonic()-self.stage_started
        peak_allocated = torch.cuda.max_memory_allocated()/2**20
        self.frozen_unchanged(); del self.model; self.model = None; gc.collect(); torch.cuda.empty_cache()
        self.status(stage='frozen_PCRF_CF_scoring')
        score_with_progress(records, ROOT / self.config['pcrf']['checkpoint'], 256, self.progress)
        pop = np.array([self.frequency[r['candidate_ids']] for r in records])
        seq_z = np.stack([standardize(r['seq']) for r in records])
        cf_z = np.stack([standardize(r['cf']) for r in records])
        pop_z = np.stack([standardize(np.log1p(x)) for x in pop])
        positions = np.array([r['target_position'] for r in records])
        final, reliability = rank_matrix(seq_z, cf_z, pop_z, (pop[:, :10] <= self.config['pcrf']['q1']).mean(1),
            positions, (1., .5, 1.))
        raw = np.array(raw_ranks); assert np.array_equal(final <= 50, raw <= 50)
        users = [r['user'] for r in records]
        reference = load_ranks(ROOT / self.config['reference_ranks'])
        values = {name: np.array([reference[u][name] for u in users]) for name in ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'B1')}
        values.update(raw=raw, pcrf=final)
        np.savez(self.path / 'pcrf_scores.npz', users=np.asarray(users), candidate_ids=np.array([r['candidate_ids'] for r in records]),
                 seq=np.stack([r['seq'] for r in records]), cf=np.stack([r['cf'] for r in records]), reliability=reliability)
        masks = group_masks(users, {r['user_id']: r for r in self.validation.records}, self.frequency,
            set((ROOT / self.config['c1_training_users']).read_text().splitlines()), self.config['pcrf']['q1'])
        groups = {g: {a: metric(r[m]) for a, r in values.items()} for g, m in masks.items()}
        comparisons = {}; done = 0
        self.status(stage='paired_statistics')
        for a, b in [('raw', 'c1'), ('raw', 'B1'), ('pcrf', 'c1_pcrf'), ('pcrf', 'original_pcrf'), ('pcrf', 'B1')]:
            comparison = comparisons[a+'_minus_'+b] = {}
            for group, mask in masks.items():
                point = paired(values[a][mask], values[b][mask], self.config)
                for k in (5, 20, 50):
                    point[f'hit{k}_gained'] = int(((values[a][mask] <= k) & (values[b][mask] > k)).sum())
                    point[f'hit{k}_lost'] = int(((values[b][mask] <= k) & (values[a][mask] > k)).sum())
                comparison[group] = point; done += 1; self.progress(done, 25)
        with (self.path / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t'); writer.writerow(['user_id', *values])
            for i, user in enumerate(users):
                writer.writerow([user, *[int(r[i]) for r in values.values()]])
        self.verify()
        result = {'state': 'COMPLETED', 'arm': self.arm, 'groups': groups, 'comparisons': comparisons,
            'fresh_candidate_sets_changed_users': int(np.sum(np.array(overlaps) < 50)),
            'mean_overlap_items_vs_frozen_C1': float(np.mean(overlaps)),
            'fresh_beam_inference_seconds': inference_elapsed, 'peak_allocated_mib': peak_allocated,
            'checkpoint_sha256': training['checkpoint_sha256'], 'new_optimizer_steps': 0,
            'test_read': False, 'verification_status': 'ANALYZED',
            'ci_caveat': 'Single decoder training run per arm; repeated validation, uncorrected paired-user intervals.'}
        write(self.path / 'summary.json', result); self.status('COMPLETED', stage='finished')

    def aggregate(self):
        self.data()
        summaries, ranks = {}, {}
        users = [r['user_id'] for r in self.validation.records]
        for arm in ARMS:
            summaries[arm] = read(self.output / arm / 'evaluate/summary.json')
            assert summaries[arm]['state'] == 'COMPLETED'
            source = load_ranks(self.output / arm / 'evaluate/per_user.tsv')
            assert set(source) == set(users)
            ranks[arm] = {kind: np.array([source[u][kind] for u in users]) for kind in ('raw', 'pcrf')}
        masks = group_masks(users, {r['user_id']: r for r in self.validation.records}, self.frequency,
            set((ROOT / self.config['c1_training_users']).read_text().splitlines()), self.config['pcrf']['q1'])
        comparisons = {}; done = 0
        self.status(stage='matched_arm_paired_statistics')
        for other in ('CE_anchor', 'shuffled_teacher'):
            for kind in ('raw', 'pcrf'):
                by_group = comparisons[f'true_teacher_minus_{other}_{kind}'] = {}
                for group, mask in masks.items():
                    candidate, baseline = ranks['true_teacher'][kind][mask], ranks[other][kind][mask]
                    row = paired(candidate, baseline, self.config)
                    row['all_metric_deltas'] = {k: v-metric(baseline)[k] for k, v in metric(candidate).items() if k != 'users'}
                    for k in (5, 20, 50):
                        row[f'hit{k}_gained'] = int(((candidate <= k) & (baseline > k)).sum())
                        row[f'hit{k}_lost'] = int(((baseline <= k) & (candidate > k)).sum())
                    by_group[group] = row; done += 1; self.progress(done, 20)
        self.verify()
        result = {'state': 'COMPLETED', 'dataset': self.config['dataset'], 'arms': summaries,
            'matched_arm_comparisons': comparisons, 'new_optimizer_steps': 3*self.config['optimizer_steps_per_arm'],
            'engineering_optimizer_steps_discarded': 1, 'test_read': False, 'verification_status': 'ANALYZED',
            'fresh_validation_users_per_arm': len(users), 'automatic_next_experiment': False,
            'interpretation': 'Single fixed decoder run per arm. Compare true teacher with both matched controls and frozen B1; candidate change alone is not improvement.'}
        write(self.output / 'summary.json', result); write(self.path / 'summary.json', result)
        self.status('COMPLETED', stage='finished')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--mode', required=True, choices=('profile', 'cache', 'train', 'evaluate', 'aggregate'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args(); config = read(ROOT / args.config)
    assert (args.arm is not None) == (args.mode in ('train', 'evaluate'))
    job = Work(config, args.mode, args.arm)
    def stop(signum, frame):
        raise Interrupted(f'SIGNAL_{signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(config[args.mode+'_budget_seconds'])
    try:
        torch.set_num_threads(config['cpu_threads'])
        if args.mode != 'aggregate':
            torch.cuda.set_device(0)
            free, _ = torch.cuda.mem_get_info(); assert free/2**20 >= config['minimum_free_mib']
            torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
        seed_everything(config['seed']); job.verify(); job.snapshot()
        if args.mode != 'profile':
            assert read(job.output / 'profile/profile.json')['state'] == 'PASSED'
        if args.mode == 'train': job.train_arm()
        elif args.mode == 'evaluate': job.evaluate()
        else: getattr(job, args.mode)()
        signal.alarm(0)
    except BaseException as error:
        job.status('STOPPED' if isinstance(error, Interrupted) else 'FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
