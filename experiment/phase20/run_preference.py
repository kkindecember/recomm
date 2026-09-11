"""Bounded full-data self-play/SFT/DPO training on the original GRAM split.

Only prepared train/validation are loaded. Test is never constructed. The
reference is the SFT model at the start of each DPO phase, following SPRec's
released shell/SPRec.sh and train/dpo.py. Negative generation uses the previous
round's model and constrained ancestral sampling (paper-inspired adaptation).
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase18.protocol.s18_diff_gram import CatalogDecoder, load_inputs, seed_everything
from experiment.phase17.core.full_latte_gram_backend import create_fresh_gram_model
from experiment.phase20.preference_core import pair_logps, path_labels, preference_loss, sampling_prefix_function

ROOT = Path(__file__).resolve().parents[2]


def metrics(ranks):
    ranks = np.asarray(ranks)
    return {'users': len(ranks), **{f'{name}@{k}': float(np.mean(
        (ranks <= k) * (1. if name == 'hit' else 1. / np.log2(ranks + 1))))
        for k in (5, 10, 20, 50) for name in ('hit', 'ndcg')}}


def paired_delta(candidate, baseline, repeats=3000):
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    rng = np.random.RandomState(20260910)
    result = {}
    for name in ('hit', 'ndcg'):
        def score(r):
            return (r <= 10) * (1. if name == 'hit' else 1. / np.log2(r + 1))
        delta = score(candidate) - score(baseline)
        draws = np.empty(repeats)
        for start in range(0, repeats, 50):
            n = min(50, repeats - start)
            draws[start:start+n] = delta[rng.randint(len(delta), size=(n, len(delta)))].mean(1)
        result[f'{name}@10'] = {'delta': float(delta.mean()),
                                'paired_bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist()}
    return result


class Stopped(Exception):
    pass


class Run:
    def __init__(self, config, config_path, output, mode):
        self.config, self.config_path, self.output = config, config_path, output
        self.device = torch.device('cuda:0')
        self.mode, self.stage, self.round, self.step = mode, 'initializing', 0, 0
        self.started = time.time()
        self.stop_reason = None
        self.model, self.optimizer, self.scheduler = None, None, None
        self.last_checkpoint = None
        self.data_config = json.loads((ROOT / config['data_config']).read_text())
        self.catalog, self.train, self.validation, self.data_manifest, self.tokenizer, self.collator = load_inputs(self.data_config)
        self.decoder = CatalogDecoder(self.catalog, {'training': config['decoding']})
        self.frequencies = training_frequencies(self.train.records, len(self.catalog['items']))
        with (ROOT/config['pcrf']['reference']).open() as f:
            self.reference = {r['user_id']: r for r in csv.DictReader(f, delimiter='\t')}
        self.historical_reference = copy.deepcopy(self.reference)
        if set(self.reference) != {r['user_id'] for r in self.validation.records}:
            raise AssertionError('Validation reference user mismatch')
        self.outcomes = []
        self.best = {'raw': None, 'pcrf': None}

    def event(self, event, **values):
        row = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'event': event,
               'stage': self.stage, 'round': self.round, 'step': self.step,
               'elapsed_seconds': round(time.time()-self.started, 2), **values}
        with (self.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_json(self.output/'status.json', {'state': 'RUNNING', 'pid': os.getpid(),
            'mode': self.mode, 'dataset': self.config['dataset'], 'test_read': False,
            'last_checkpoint': self.last_checkpoint, **row})

    def check_stop(self):
        if self.stop_reason:
            raise Stopped(self.stop_reason)

    def new_model(self):
        cfg = self.data_config
        if sha256(ROOT/cfg['parent_checkpoint']) != cfg['parent_sha256']:
            raise AssertionError('Parent SHA256 mismatch')
        model = create_fresh_gram_model(ROOT, 'G0_GRAM_B0_FRESH', self.tokenizer,
                                        seed=self.config['seed'])
        weights = torch.load(ROOT/cfg['parent_checkpoint'], map_location='cpu')
        model.resize_token_embeddings(weights['shared.weight'].size(0))
        model.load_state_dict(weights, strict=True)
        del weights
        if model.config.cf0_enabled or model.config.hi_gram_enabled or model.config.s17_modules:
            raise AssertionError('Unexpected historical architecture enabled')
        return model.to(self.device)

    def snapshot(self):
        paths = {self.config_path, Path(__file__).resolve()}
        for module in list(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if name and name.endswith('.py'):
                path = Path(name).resolve()
                if ROOT in path.parents:
                    paths.add(path)
        hashes = {}
        for path in sorted(paths):
            relative = path.relative_to(ROOT)
            dst = self.output/'sources'/relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            hashes[str(relative)] = sha256(path)
        hashes[self.config['data_config']] = sha256(ROOT/self.config['data_config'])
        write_json(self.output/'manifest.json', {'config': self.config,
            'data_manifest': self.data_manifest, 'source_sha256': hashes,
            'torch': torch.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_name': torch.cuda.get_device_name(), 'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'pcrf_checkpoint_sha256': sha256(ROOT/self.config['pcrf']['checkpoint']),
            'reference_tsv_sha256': sha256(ROOT/self.config['pcrf']['reference']),
            'source_manifest_sha256': sha256(ROOT/'experiment/phase20/source_manifest.json'),
            'precision': 'float32', 'tf32': False, 'cross_attention_cache_patch': False,
            'test_read': False})

    def save_checkpoint(self, name, include_optimizer=False):
        path = self.output/f'{name}.pt'
        payload = {'model': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   'round': self.round, 'stage': self.stage, 'step': self.step,
                   'config': self.config, 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
                   'python_rng': random.getstate()}
        if include_optimizer and self.optimizer is not None:
            # CPU-copy nested optimizer state without allocating another GPU optimizer.
            def cpu(value):
                if torch.is_tensor(value):
                    return value.detach().cpu()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [cpu(v) for v in value]
                return value
            payload['optimizer'] = cpu(self.optimizer.state_dict())
            payload['scheduler'] = self.scheduler.state_dict()
        temporary = path.with_suffix('.tmp.pt')
        torch.save(payload, temporary)
        temporary.replace(path)
        self.last_checkpoint = str(path.relative_to(ROOT))
        return path

    def loader(self, dataset, indices, batch_size, shuffle=False):
        generator = torch.Generator().manual_seed(self.config['seed']+self.round*101+(1 if self.stage == 'dpo' else 0))
        # Indices, not user IDs, identify train prefixes. A user has multiple train rows.
        return DataLoader(indices, batch_size=batch_size, shuffle=shuffle, generator=generator,
                          num_workers=0, collate_fn=lambda ids: (ids, self.collator([dataset[i] for i in ids])))

    @torch.no_grad()
    def sample_negative(self, model, batch):
        sequences = model.generate(input_ids=batch['item_text_ids'].to(self.device),
            attention_mask=batch['item_text_masks'].to(self.device),
            max_length=max(map(len, self.decoder.paths)),
            prefix_allowed_tokens_fn=sampling_prefix_function(self.decoder.trie),
            do_sample=True, num_beams=1, num_return_sequences=1, top_k=0, top_p=1., temperature=1.)
        items = []
        for seq in sequences.cpu().tolist():
            if 1 not in seq:
                raise AssertionError('Sample lacks EOS')
            path = tuple(seq[:seq.index(1)+1])
            items.append(self.decoder.path_to_item[path])
        return items

    def mine(self):
        self.stage, self.step = 'negative_generation', 0
        self.model.eval()
        negatives = [None] * len(self.train)
        previous = time.time()
        path = self.output/f'round{self.round}_train_negatives.jsonl'
        with path.open('x') as f:
            for indices, batch in self.loader(self.train, list(range(len(self.train))), self.config['mining_microbatch']):
                self.check_stop()
                sampled = self.sample_negative(self.model, batch)
                for index, item in zip(indices, sampled):
                    negatives[index] = item
                    row = self.train.records[index]
                    f.write(json.dumps({'train_index': index, 'user_id': row['user_id'],
                        'chosen': row['target'], 'rejected': item, 'valid': item != row['target'],
                        'split': 'train'})+'\n')
                self.step += len(indices)
                if time.time()-previous >= 30:
                    self.event('mining_progress', examples=self.step, total=len(self.train))
                    f.flush()
                    previous = time.time()
        valid = [i for i, item in enumerate(negatives) if item != self.train.records[i]['target']]
        if not valid:
            raise ValueError('No non-colliding self-play negatives')
        self.event('mining_complete', examples=len(negatives), valid_pairs=len(valid),
                   collisions=len(negatives)-len(valid), negative_sha256=sha256(path))
        return negatives, valid

    def train_phase(self, phase, negatives=None, valid=None):
        self.stage, self.step = phase, 0
        # DPO log ratios must use deterministic dropout-free policy/reference scores.
        # eval() disables dropout but DOES NOT disable gradients or parameter updates.
        self.model.train(phase == 'sft')
        reference = None
        if phase == 'dpo':
            reference = copy.deepcopy(self.model).eval().requires_grad_(False)
        indices = list(range(len(self.train))) if phase == 'sft' else valid
        batch_size, effective = self.config['microbatch'], self.config['effective_batch_size']
        loader = self.loader(self.train, indices, batch_size, shuffle=True)
        accumulation = effective // batch_size
        steps = math.ceil(len(loader)/accumulation)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config[f'{phase}_learning_rate'],
                                           weight_decay=.01, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, min(20, max(1, steps//20)), steps)
        self.optimizer.zero_grad(set_to_none=True)
        previous, seen, totals = time.time(), 0, {'loss': 0., 'margin': 0., 'nll': 0.}
        self.event('training_started', examples=len(indices), optimizer_steps=steps, phase=phase)
        for index, (row_ids, batch) in enumerate(loader):
            if phase == 'sft':
                output = self.model(input_ids=batch['item_text_ids'].to(self.device),
                    attention_mask=batch['item_text_masks'].to(self.device),
                    labels=batch['target_ids'].to(self.device), use_cache=False, return_dict=True)
                loss, nll, margin = output.loss, output.loss.detach(), torch.tensor(0.)
            else:
                labels = path_labels(self.decoder.paths, [negatives[i] for i in row_ids], self.device)
                with torch.no_grad():
                    rp, rn, _ = pair_logps(reference, batch, labels, self.device)
                cp, cn, nll = pair_logps(self.model, batch, labels, self.device)
                losses, margins = preference_loss(cp, cn, rp, rn, self.config['beta'])
                loss, margin = losses.mean(), margins.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            # Exact example weighting, including the final partial accumulation group.
            group_start = (index // accumulation) * effective
            group_size = min(effective, len(indices)-group_start)
            (loss * (len(row_ids)/group_size)).backward()
            for key, value in [('loss', loss), ('margin', margin), ('nll', nll)]:
                totals[key] += float(value.detach())*len(row_ids)
            seen += len(row_ids)
            if (index+1) % accumulation == 0 or index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % 100 == 0 or self.stop_reason:
                    self.save_checkpoint('last_progress', include_optimizer=True)
                self.check_stop()
                if time.time()-previous >= 30:
                    self.event('training_progress', examples=seen, total=len(indices),
                        optimizer_steps=steps, **{k: v/seen for k, v in totals.items()},
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                    previous = time.time()
        path = self.save_checkpoint(f'round{self.round}_{phase}', include_optimizer=True)
        self.event('training_complete', examples=seen, checkpoint=str(path.relative_to(ROOT)),
                   **{k: v/seen for k, v in totals.items()})
        del reference
        self.optimizer, self.scheduler = None, None
        gc.collect()
        torch.cuda.empty_cache()

    def pcrf(self, predictions):
        from experiment.phase9.eval_cf0_b3_beamfusion import score_item_head, standardize
        from experiment.phase9.eval_cf0_b4_reliability import rank_matrix
        lookup = {r['user_id']: r for r in self.validation.records}
        records = []
        for r in predictions:
            source, items = lookup[r['user_id']], r['ranked_item_ids']
            records.append({'user': r['user_id'], 'history': source['history'][-20:],
                'candidate_ids': items, 'seq': np.asarray(r['scores'], dtype=np.float64),
                'target_position': items.index(source['target']) if source['target'] in items else -1})
        score_item_head(records, ROOT/self.config['pcrf']['checkpoint'], 256)
        pop = np.asarray([self.frequencies[r['candidate_ids']] for r in records])
        ranks, _ = rank_matrix(np.stack([standardize(r['seq']) for r in records]),
            np.stack([standardize(r['cf']) for r in records]),
            np.stack([standardize(np.log1p(p)) for p in pop]),
            (pop[:, :10] <= self.config['pcrf']['q1']).mean(1),
            np.asarray([r['target_position'] for r in records]),
            tuple(self.config['pcrf'][k] for k in ('lambda', 'beta', 'gamma')))
        return ranks.tolist()

    @torch.no_grad()
    def evaluate(self, tag, indices):
        self.stage = f'validation_{tag}'
        self.model.eval()
        predictions, ranks, previous = [], [], time.time()
        destination = self.output/f'{tag}.predictions.jsonl'
        with destination.open('x') as f:
            for _, batch in self.loader(self.validation, indices, self.config['validation_microbatch']):
                self.check_stop()
                generated = self.decoder.generate(self.model, {
                    'input_ids': batch['item_text_ids'].to(self.device),
                    'attention_mask': batch['item_text_masks'].to(self.device)})
                for user, target, (items, scores) in zip(batch['user_ids'], batch['target_item_ids'].tolist(), generated):
                    row = {'user_id': user, 'gold_item_id': target, 'ranked_item_ids': items,
                           'scores': scores, 'split': 'validation'}
                    predictions.append(row)
                    ranks.append(items.index(target)+1 if target in items else 51)
                    f.write(json.dumps(row)+'\n')
                if time.time()-previous >= 30:
                    self.event('validation_progress', tag=tag, examples=len(predictions), total=len(indices))
                    f.flush()
                    previous = time.time()
        reranked = self.pcrf(predictions)
        baseline = [int(self.reference[r['user_id']]['baseline_rank']) for r in predictions]
        baseline_pcrf = [int(self.reference[r['user_id']]['pcrf_rank']) for r in predictions]
        historical = [int(self.historical_reference[r['user_id']]['baseline_rank']) for r in predictions]
        historical_pcrf = [int(self.historical_reference[r['user_id']]['pcrf_rank']) for r in predictions]
        result = {'tag': tag, 'raw': metrics(ranks), 'pcrf': metrics(reranked),
            'original_gram': metrics(baseline), 'original_gram_pcrf': metrics(baseline_pcrf),
            'historical_gram': metrics(historical), 'historical_gram_pcrf': metrics(historical_pcrf),
            'comparator': 'matched_fp32_parent' if tag != 'parent_full' and self.mode == 'run' else 'historical_reference',
            'scope': 'full_validation' if len(indices) == len(self.validation) else 'selection_validation',
            'test_read': False}
        if result['scope'] == 'full_validation':
            result['descriptive_paired_ci'] = {
                'raw_minus_gram': paired_delta(ranks, baseline),
                'pcrf_minus_gram': paired_delta(reranked, baseline),
                'pcrf_minus_original_pcrf': paired_delta(reranked, baseline_pcrf)}
            result['ci_caveat'] = 'Validation reused across historical experiments; selected model; descriptive CI, not independent confirmation.'
        with (self.output/f'{tag}.ranks.tsv').open('w') as f:
            f.write('user_id\traw_rank\tpcrf_rank\tbaseline_rank\tbaseline_pcrf_rank\n')
            for r, a, b, c, d in zip(predictions, ranks, reranked, baseline, baseline_pcrf):
                f.write(f"{r['user_id']}\t{a}\t{b}\t{c}\t{d}\n")
        write_json(self.output/f'{tag}.metrics.json', result)
        self.event('validation_complete', **result)
        return result

    def smoke(self):
        self.stage = 'smoke'
        longest = sorted(range(len(self.train)), key=lambda i: len(self.train.records[i]['history']), reverse=True)
        batch = self.collator([self.train[i] for i in longest[:self.config['microbatch']]])
        targets = batch['target_item_ids'].tolist()
        labels = path_labels(self.decoder.paths, targets, self.device)
        torch.testing.assert_close(labels.cpu(), batch['target_ids'], atol=0, rtol=0)
        self.model.eval()
        sampled = self.sample_negative(self.model, batch)
        # Smoke uses deterministic non-colliding alternatives to exercise every gradient.
        negative = [1+(i % (len(self.catalog['items'])-1)) for i in targets]
        labels = path_labels(self.decoder.paths, negative, self.device)
        reference = copy.deepcopy(self.model).eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            rp, rn, _ = pair_logps(reference, batch, labels, self.device)
        cp, cn, _ = pair_logps(self.model, batch, labels, self.device)
        losses, margin = preference_loss(cp, cn, rp, rn, self.config['beta'])
        torch.testing.assert_close(margin, torch.zeros_like(margin), atol=1e-5, rtol=0)
        losses.mean().backward()
        groups = {}
        for prefix in ('encoder.', 'decoder.', 'shared.'):
            gradients = [p.grad for n, p in self.model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise AssertionError(f'Missing/nonfinite gradient in {prefix}')
            groups[prefix] = sum(float(g.abs().sum()) for g in gradients)
            if groups[prefix] <= 0:
                raise AssertionError(f'Zero gradient in {prefix}')
        # GRAM aliases the same position module under encoder and the root.
        # named_parameters() deduplicates it under encoder.position_embedding.
        position_gradient = self.model.position_embedding.weight.grad
        if position_gradient is None or not torch.isfinite(position_gradient).all():
            raise AssertionError('Missing/nonfinite shared position embedding gradient')
        groups['position_embedding'] = float(position_gradient.abs().sum())
        if groups['position_embedding'] <= 0:
            raise AssertionError('Zero position embedding gradient')
        if any(p.grad is not None for p in reference.parameters()):
            raise AssertionError('Frozen reference received a gradient')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['dpo_learning_rate'])
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.model.zero_grad(set_to_none=True)
        result = {'reference_identity_max_abs': float(margin.abs().max()), 'loss': float(losses.mean()),
                  'gradient_abs_sum': groups, 'sampled_legal_items': sampled,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        del reference, optimizer, cp, cn, losses
        gc.collect()
        torch.cuda.empty_cache()
        # Restore exact parent; an optimization smoke must never contaminate training.
        del self.model
        self.model = self.new_model()
        result['parent_validation_parity'] = self.evaluate('parent_smoke', list(range(16)))
        with (self.output/'parent_smoke.ranks.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        result['fresh_vs_historical_rank_mismatches'] = {
            'raw': sum(r['raw_rank'] != r['baseline_rank'] for r in rows),
            'pcrf': sum(r['pcrf_rank'] != r['baseline_pcrf_rank'] for r in rows)}
        # Historical predictions used the old runtime's precision. Verify the
        # PCRF adapter against exactly those cached candidates, separately from
        # fresh inference. Formal comparisons use a fresh, matched FP32 parent.
        from experiment.phase9.eval_cf0_b3_beamfusion import load_cached_beams, normalize_lexical_id
        beams, _ = load_cached_beams(ROOT/self.config['baseline_predictions'])
        lexical = {normalize_lexical_id(v): i for i, v in enumerate(self.catalog['lexical_ids']) if i}
        cached = []
        for row in self.validation.records[:64]:
            beam = beams[row['user_id']]
            if lexical[normalize_lexical_id(beam['gold'])] != row['target']:
                raise AssertionError('Cached gold / prepared target mismatch')
            cached.append({'user_id': row['user_id'],
                'ranked_item_ids': [lexical[normalize_lexical_id(v)] for v in beam['candidates']],
                'scores': beam['seq']})
        cached_ranks = self.pcrf(cached)
        if cached_ranks != [int(self.reference[r['user_id']]['pcrf_rank']) for r in cached]:
            raise AssertionError('PCRF adapter on historical beams differs from fixed reference')
        result['cached_pcrf_exact_rank_parity_users'] = len(cached)
        # Test the TF32 hypothesis on the same 16 cases, without selecting any
        # recommendation setting from their accuracy.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tf32 = self.evaluate('parent_tf32_diagnostic', list(range(16)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        fresh_on = [json.loads(s) for s in (self.output/'parent_tf32_diagnostic.predictions.jsonl').read_text().splitlines()]
        result['tf32_diagnostic'] = {'metrics': tf32,
            'exact_candidate_orders_vs_historical': sum(
                r['ranked_item_ids'] == [lexical[normalize_lexical_id(v)] for v in beams[r['user_id']]['candidates']]
                for r in fresh_on)}
        # Exercise the actual phase runner on eight train-only rows. This is an
        # engineering check, kept in a separate smoke directory and never selected.
        complete_train = self.train
        self.train = type(complete_train)(complete_train.records[:8], self.catalog)
        self.round = 1
        negatives, valid = self.mine()
        self.train_phase('sft')
        self.train_phase('dpo', negatives, valid)
        self.train = complete_train
        result['pipeline_smoke'] = {'training_rows': 8, 'sft_optimizer_steps': 1,
                                    'dpo_optimizer_steps': 1, 'valid_preference_rows': len(valid)}
        result.update(state='PASSED', test_read=False, scientific_efficacy_result=False)
        write_json(self.output/'smoke.json', result)
        self.event('smoke_passed', **result)

    def run(self):
        seed_everything(self.config['seed'])
        self.model = self.new_model()
        self.snapshot()
        if self.mode == 'smoke':
            self.smoke()
            return
        smoke = json.loads((ROOT/self.config['smoke_output']/'smoke.json').read_text())
        if smoke['state'] != 'PASSED':
            raise AssertionError('GPU smoke not passed')
        # A numeric parity check found that historical cached predictions differ
        # from fresh FP32 decoding. Re-evaluate the UNCHANGED parent once to avoid
        # attributing precision/decoding drift to preference training.
        self.evaluate('parent_full', list(range(len(self.validation))))
        with (self.output/'parent_full.ranks.tsv').open() as f:
            for r in csv.DictReader(f, delimiter='\t'):
                self.reference[r['user_id']]['baseline_rank'] = r['raw_rank']
                self.reference[r['user_id']]['pcrf_rank'] = r['pcrf_rank']
        write_json(self.output/'matched_parent_reference.json', self.reference)
        # Fixed subset selected once by seed, without reference ranks or targets.
        selection = sorted(random.Random(20260910).sample(range(len(self.validation)),
                            min(self.config['selection_users'], len(self.validation))))
        write_json(self.output/'selection_indices.json', selection)
        for round_index in range(1, self.config['rounds']+1):
            self.round = round_index
            negatives, valid = self.mine()
            self.train_phase('sft')
            sft = self.evaluate(f'round{round_index}_sft', selection)
            self.train_phase('dpo', negatives, valid)
            dpo = self.evaluate(f'round{round_index}_dpo', selection)
            self.outcomes.extend([sft, dpo])
            for name in self.best:
                if self.best[name] is None or dpo[name]['ndcg@10'] > self.best[name]['score']:
                    self.best[name] = {'tag': dpo['tag'], 'score': dpo[name]['ndcg@10']}
            write_json(self.output/'selection.json', {'outcomes': self.outcomes, 'best': self.best,
                'rule': 'max selection NDCG@10 among DPO rounds; raw and PCRF chosen separately; SFT descriptive only'})
        # Evaluate distinct selected checkpoints once, on all official validation users.
        full = {}
        for tag in sorted({r['tag'] for r in self.best.values()}):
            weights = torch.load(self.output/f'{tag}.pt', map_location='cpu')
            self.model.load_state_dict(weights['model'], strict=True)
            del weights
            full[tag] = self.evaluate(f'{tag}_full', list(range(len(self.validation))))
        raw = full[self.best['raw']['tag']]
        combo = full[self.best['pcrf']['tag']]
        raw_gain = raw['raw']['ndcg@10']-raw['original_gram']['ndcg@10']
        incremental = combo['pcrf']['ndcg@10']-combo['original_gram_pcrf']['ndcg@10']
        write_json(self.output/'summary.json', {'state': 'COMPLETED', 'best': self.best, 'full_validation': full,
            'raw_ndcg_delta_vs_gram': raw_gain, 'combined_ndcg_delta_vs_original_pcrf': incremental,
            'promising_for_beauty': raw_gain > 0 or incremental > 0,
            'decision': 'CROSS_DOMAIN_SCREEN' if raw_gain > 0 or incremental > 0 else 'NO_POSITIVE_SIGNAL_THIS_CONFIG',
            'not_a_convergence_claim': True, 'test_read': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['smoke', 'run'], required=True)
    args = parser.parse_args()
    config_path = (ROOT/args.config).resolve()
    config = json.loads(config_path.read_text())
    # Select the configured physical GPU before the first CUDA initialization.
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    if config['effective_batch_size'] % config['microbatch']:
        raise ValueError('Effective batch must be a multiple of microbatch')
    output = ROOT/config['smoke_output' if args.mode == 'smoke' else 'output']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Existing experiment: choose an explicit new attempt; no silent overwrite/retry')
    torch.set_num_threads(4)
    run = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(0)
        free, _ = torch.cuda.mem_get_info()
        if free/2**20 < config['minimum_free_mib']:
            raise RuntimeError(f'Insufficient free memory: {free/2**20:.0f} MiB')
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
        run = Run(config, config_path, output, args.mode)
        def stop(signum, frame):
            run.stop_reason = f'SIGNAL_{signum}'
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(config['max_wall_seconds'] if args.mode == 'run' else 900)
        run.run()
        signal.alarm(0)
        write_json(output/'status.json', {'state': 'COMPLETED', 'mode': args.mode,
            'dataset': config['dataset'], 'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'elapsed_seconds': time.time()-run.started, 'last_checkpoint': run.last_checkpoint, 'test_read': False})
    except BaseException as error:
        state = 'STOPPED' if isinstance(error, Stopped) else 'FAILED'
        write_json(output/'status.json', {'state': state, 'mode': args.mode, 'dataset': config['dataset'],
            'error': repr(error), 'traceback': traceback.format_exc(), 'pid': os.getpid(),
            'stage': run.stage if run else 'initializing', 'round': run.round if run else 0,
            'step': run.step if run else 0, 'last_checkpoint': run.last_checkpoint if run else None,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'test_read': False})
        raise


if __name__ == '__main__':
    main()
