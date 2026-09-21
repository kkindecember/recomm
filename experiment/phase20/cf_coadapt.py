"""Matched full-catalog, candidate-only and generator-conditioned CF adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, old_score, z, ranks, metrics, sha
from experiment.phase20.future_fusion_alignment import CF0B2ItemHead
from experiment.phase20.future_label_multiseed import write
from experiment.phase20.personal_popularity_core import reconstruct_training
from experiment.phase20.shared_cf_probe import read, arrays, rows
from experiment.phase20.cf_coadapt_core import install, trainable, adapter_state, restore, objective, checks

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'experiment/phase20/cf_coadapt_toys_v1.json'
OLD = ROOT / 'artifacts/phase20/gacr_fusion_train_v1/Toys'
TALOS = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'
FACTOR = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
PAIR = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
PREPARED = ROOT / 'artifacts/phase18/diff_gram/prepared_v2_categories'
CHECKPOINTS = [ROOT / 'artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt'] + [
    ROOT / f'artifacts/phase9/p9s_multiseed/Toys/seed{s}/item_head/best_item_head.pt' for s in (2024, 2025)]


def frozen_hash(models):
    digest = hashlib.sha256()
    for model in models:
        for name, value in model.named_parameters():
            if not value.requires_grad:
                digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Work:
    def __init__(self, mode):
        self.c = read(CONFIG); self.out = ROOT / self.c['output']; self.mode = mode
        self.started, self.steps, self.profile_steps = time.monotonic(), 0, 0
        torch.set_num_threads(self.c['cpu_threads']); torch.set_num_interop_threads(1)

    def status(self, stage, state='RUNNING', **kw):
        write(self.out / 'status.json', dict(state=state, mode=self.mode, stage=stage, pid=os.getpid(),
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), elapsed_seconds=time.monotonic() - self.started,
            adapter_updates=self.steps, discarded_profile_updates=self.profile_steps,
            original_CF_parameter_updates=0, GRAM_updates=0, new_beam=False, test_read=False, **kw))

    def hashes(self, paths):
        return {str(p.relative_to(ROOT)): sha(p) for p in paths}

    def verify_inputs(self):
        protocol = read(self.out / 'protocol.json'); assert self.c == protocol['config']
        for mapping in (protocol['sha256'], protocol['prepared_sha256']):
            for path, expected in mapping.items(): assert sha(ROOT / path) == expected, path
        return protocol

    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=False); self.status('prepare')
        health = checks()
        paths = {CONFIG, Path(__file__), ROOT / self.c['contract'], *CHECKPOINTS,
                 ROOT / 'experiment/phase20/cf_coadapt_core.py', ROOT / 'experiment/phase20/run_cf_coadapt.sh',
                 OLD / 'training_cache.npz', OLD / 'train_candidates.jsonl', OLD / 'B1.pt',
                 FACTOR / 'Toys.audit_validation_cache.npz', PAIR / 'Toys.cf_scores.npz',
                 PREPARED / 'train.jsonl', PREPARED / 'validation.jsonl', PREPARED / 'catalog.json',
                 TALOS / f'Toys.{self.c["parent_head_seed"]}.softmax.pt',
                 TALOS / f'Toys.{self.c["parent_head_seed"]}.predictions.npz'}
        trusted = read(TALOS / 'summary.json')['input_sha256']
        audit = read(TALOS / 'verification.json'); assert audit['state'] == 'VERIFIED'
        trusted.update(audit['output_sha256'])
        exposure_config = ROOT / 'experiment/phase20/cf_exposure_toys_v1.json'
        trusted.update(read(exposure_config)['input_sha256']); paths.add(exposure_config)
        paths.update((TALOS / 'summary.json', TALOS / 'verification.json'))
        a = arrays(OLD / 'training_cache.npz'); v = arrays(FACTOR / 'Toys.audit_validation_cache.npz')
        records, source, validation = rows(OLD / 'train_candidates.jsonl'), rows(PREPARED / 'train.jsonl'), rows(PREPARED / 'validation.jsonl')
        nfit = int(a['fit'].sum()); ntrain = len(records)
        assert nfit == 21839 and ntrain == 22095
        np.testing.assert_array_equal(np.flatnonzero(a['fit']), np.arange(nfit))
        fit_users, cal_users = set(), set()
        for j, row in enumerate(records):
            original = source[row['train_index']]
            for k in ('user_id', 'history', 'target'): assert row[k] == original[k]
            np.testing.assert_array_equal(row['union_ids'][:50], a['candidate_ids'][j])
            assert row['target_position'] == a['positions'][j]
            assert (row['partition'] == 'fit') == bool(a['fit'][j])
            (fit_users if j < nfit else cal_users).add(row['user_id'])
        assert fit_users.isdisjoint(cal_users)
        np.testing.assert_array_equal([r['user_id'] for r in validation], v['users'])
        for j, row in enumerate(validation):
            ids = v['candidate_ids'][j].tolist()
            assert v['positions'][j] == (ids.index(row['target']) if row['target'] in ids else -1)
        catalog = read(PREPARED / 'catalog.json')
        _, sequences, _, _ = reconstruct_training(source)
        freq = np.bincount([i for h in sequences.values() for i in h], minlength=len(catalog['items']))
        np.testing.assert_array_equal(freq[a['candidate_ids']], a['frequencies'])
        with torch.no_grad():
            b1 = old_score(torch.as_tensor(a['features']), torch.as_tensor(a['base']), torch.load(OLD / 'B1.pt', map_location='cpu'))
            model = Factor('next_only').eval().requires_grad_(False)
            model.load_state_dict(torch.load(TALOS / f'Toys.{self.c["parent_head_seed"]}.softmax.pt', map_location='cpu'))
            features = np.concatenate((a['features'], v['features']))
            prior = np.concatenate((z(b1) - np.log(49), v['prior']))
            raw = []
            # Match the original validation batching; do not let a train tail
            # change the FP32 GEMM shape of the final validation minibatch.
            for begin, end in ((0, ntrain), (ntrain, len(prior))):
                for off in range(begin, end, 256):
                    sl = slice(off, min(off + 256, end))
                    raw.append(model(torch.as_tensor(features[sl]), torch.as_tensor(prior[sl]))[0].numpy())
            base = z(np.concatenate(raw))
        ids = np.concatenate((a['candidate_ids'], v['candidate_ids']))
        frequencies = freq[ids]
        original_b1 = np.concatenate((b1, v['prior']))
        reliability = 1 - (np.take_along_axis(frequencies, np.argsort(-original_b1, axis=1, kind='stable')[:, :10], 1) <= 5).mean(1)
        history = np.zeros((len(ids), 20), np.int64)
        all_records = records + validation
        for j, row in enumerate(all_records):
            assert 0 < len(row['history']) <= 20
            history[j, :len(row['history'])] = row['history']
        pair = arrays(PAIR / 'Toys.cf_scores.npz')
        np.testing.assert_array_equal(pair['users'], v['users'])
        order = []
        for old_ids, new_ids in zip(pair['candidate_ids'], v['candidate_ids']):
            mapping = {int(i): k for k, i in enumerate(old_ids)}
            assert len(mapping) == 50 and set(mapping) == set(new_ids.tolist())
            order.append([mapping[int(i)] for i in new_ids])
        vcf = np.take_along_axis(pair['scores'], np.asarray(order)[None], axis=2)
        final = base[ntrain:] + reliability[ntrain:, None] * z(z(vcf) - .5 * z(np.log1p(frequencies[ntrain:]))[None]).mean(0)
        expected = arrays(TALOS / f'Toys.{self.c["parent_head_seed"]}.predictions.npz')
        expected_ranks = expected['ranks'][list(expected['rank_names']).index('softmax.mean')]
        np.testing.assert_array_equal(ranks(final, v['positions']), expected_ranks)
        np.testing.assert_allclose(base[ntrain:], expected['raw_scores'][list(expected['raw_names']).index('softmax')], atol=1e-11, rtol=0)
        np.savez_compressed(self.out / 'queries.npz', history=history, candidate_ids=ids,
            targets=np.asarray([r['target'] for r in all_records]), users=np.asarray([r['user_id'] for r in all_records]),
            positions=np.concatenate((a['positions'], v['positions'])), base=base,
            pop_z=z(np.log1p(frequencies)), reliability=reliability, head=a['head'],
            nfit=np.asarray(nfit), ntrain=np.asarray(ntrain), tail_mask=v['tail_mask'],
            validation_in_fit=np.isin(v['users'], list(fit_users)),
            old_train_cf=a['cf'], old_validation_cf=vcf, expected_validation_ranks=expected_ranks)
        write(self.out / 'record_identity.json', dict(fit=nfit, calibration=ntrain - nfit, validation=len(validation),
            fit_users=len(fit_users), calibration_users=len(cal_users), covered_fit=int((a['positions'][:nfit] >= 0).sum()),
            uncovered_fit=int((a['positions'][:nfit] < 0).sum()), fit_calibration_users_disjoint=True,
            all_histories_targets_and_candidates_checked=True, upstream_exposure_not_removed=True))
        for module in list(sys.modules.values()):
            raw_path = getattr(module, '__file__', '')
            if raw_path and raw_path.endswith('.py'):
                p = Path(raw_path).resolve()
                if ROOT in p.parents: paths.add(p)
        hashes = self.hashes(paths)
        for path, value in hashes.items():
            if path in trusted: assert value == trusted[path], path
            if Path(path).suffix in ('.py', '.sh') or path.startswith('experiment/') or path == self.c['contract']:
                dest = self.out / 'source_snapshot' / path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / path, dest)
        write(self.out / 'protocol.json', dict(config=self.c, checks=health, sha256=hashes,
            prepared_sha256=self.hashes([self.out / 'queries.npz', self.out / 'record_identity.json']),
            evidence_scope='One-seed fixed-upstream exploratory development screening; no test; no independent retraining claim.'))
        self.status('prepared', state='PREPARED', **read(self.out / 'record_identity.json'))

    def load(self, device):
        self.a = arrays(self.out / 'queries.npz'); self.nfit, self.ntrain = int(self.a['nfit']), int(self.a['ntrain'])
        self.device = device
        self.t = {k: torch.as_tensor(self.a[k], device=device) for k in ('history', 'candidate_ids', 'targets', 'positions', 'base', 'pop_z', 'reliability')}
        h = self.a['head'][:self.nfit]
        self.weights = torch.as_tensor(np.where(h, self.nfit / (2 * h.sum()), self.nfit / (2 * (~h).sum())), device=device)
        self.models, self.tables, self.scales = [], [], []
        torch.manual_seed(self.c['seed'])
        for path in CHECKPOINTS:
            checkpoint = torch.load(path, map_location='cpu'); c = checkpoint['model_config']
            model = CF0B2ItemHead(c['num_items'], c['max_history'], c['d_model'], c['num_layers'], c['num_heads'], c['dropout'], c['temperature_initial'])
            model.load_state_dict(checkpoint['model_state_dict']); install(model, self.c['rank']); model.to(device).eval()
            self.models.append(model)
            self.tables.append(F.normalize(model.item_embedding.weight.detach(), dim=-1))
            self.scales.append(model.logit_scale.detach().exp().clamp(max=100))
        self.params = [p for m in self.models for p in trainable(m)]
        assert len(self.params) == 6 and sum(p.numel() for p in self.params) == 122880
        self.initial = [adapter_state(m) for m in self.models]

    def reset(self, states=None):
        for model, state in zip(self.models, states or self.initial):
            restore(model, state); model.eval(); model.zero_grad(set_to_none=True)

    def scores(self, ix, full=False):
        history = self.t['history'][ix]; ids = self.t['candidate_ids'][ix]
        candidates, catalog = [], []
        for model, items, scale in zip(self.models, self.tables, self.scales):
            u = model.encode(history, history.ne(0))
            candidates.append(scale * (u[:, None] * items[ids]).sum(-1))
            if full: catalog.append(scale * F.linear(u, items[1:]))
        return torch.stack(candidates), catalog

    def optimizer(self):
        return torch.optim.AdamW(self.params, lr=self.c['learning_rate'], weight_decay=self.c['weight_decay'])

    def update(self, optimizer, ix, arm):
        optimizer.zero_grad(set_to_none=True)
        candidate, full = self.scores(ix, full=True)
        loss = objective(full, candidate, self.t['base'][ix], self.t['pop_z'][ix], self.t['reliability'][ix],
                         self.t['positions'][ix], self.t['targets'][ix], arm)
        loss = (loss * self.weights[ix]).mean(); assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.c['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()), float(norm)

    def predict(self, indices, stage):
        out = []
        with torch.no_grad():
            for off in range(0, len(indices), self.c['eval_batch_size']):
                ix = indices[off:off + self.c['eval_batch_size']]
                value, _ = self.scores(ix); assert torch.isfinite(value).all()
                out.append(value.cpu().numpy())
                if off % (8 * self.c['eval_batch_size']) == 0:
                    self.status(stage, completed=off + len(ix), total=len(indices))
        return np.concatenate(out, axis=1)

    def final_scores(self, cf, indices):
        cf = cf.astype(np.float64)
        adjusted = z(z(cf) - .5 * self.a['pop_z'][indices][None])
        return self.a['base'][indices] + self.a['reliability'][indices, None] * adjusted.mean(0)

    def run(self):
        assert not (self.out / 'run_started.json').exists(), 'Refusing overwrite or implicit retry'
        self.verify_inputs()
        assert torch.cuda.is_available(), 'CUDA unavailable'
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(self.c['cuda_memory_fraction'], 0)
        free, _ = torch.cuda.mem_get_info(); assert free / 2**20 >= self.c['minimum_free_mib']
        write(self.out / 'run_started.json', dict(pid=os.getpid(), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        def timeout(signum, frame): raise TimeoutError('Frozen one-hour budget exhausted')
        signal.signal(signal.SIGALRM, timeout); signal.alarm(self.c['max_run_seconds'])
        self.status('load_pretrained_cf'); self.load(torch.device('cuda:0'))
        before = frozen_hash(self.models); torch.save(self.initial, self.out / 'initial.pt')
        all_ix = np.arange(len(self.a['users']))
        frozen = self.predict(all_ix, 'initial_score_parity')
        train_error = float(np.abs(frozen[0, :self.ntrain] - self.a['old_train_cf']).max())
        validation_error = float(np.abs(frozen[:, self.ntrain:] - self.a['old_validation_cf']).max())
        assert max(train_error, validation_error) <= 1e-4, (train_error, validation_error)
        initial_score = self.final_scores(frozen, all_ix)
        initial_ranks = ranks(initial_score, self.a['positions'])
        np.testing.assert_array_equal(initial_ranks[self.ntrain:], self.a['expected_validation_ranks'])
        np.savez_compressed(self.out / 'frozen.predictions.npz', cf=frozen, scores=initial_score, ranks=initial_ranks)
        write(self.out / 'initial_parity.json', dict(train_cf_max_error=train_error, validation_cf_max_error=validation_error,
            all_validation_ranks_exact=True, trainable_parameters=sum(p.numel() for p in self.params), zero_adapters=True))
        # Deliberately include covered and uncovered long histories in the resource profile.
        lengths = (self.a['history'][:self.nfit] != 0).sum(1)
        covered = self.a['positions'][:self.nfit] >= 0
        half = self.c['batch_size'] // 2
        profile_ix = np.concatenate([np.flatnonzero(mask)[np.argsort(-lengths[mask], kind='stable')[:half]] for mask in (covered, ~covered)])
        profile = {}; steps_per_epoch = int(np.ceil(self.nfit / self.c['batch_size']))
        for arm in self.c['arms']:
            self.reset(); optimizer = self.optimizer(); losses = []; norms = []
            torch.cuda.synchronize(); start = time.monotonic()
            for _ in range(self.c['profile_steps_per_arm']):
                loss, norm = self.update(optimizer, profile_ix, arm); losses.append(loss); norms.append(norm); self.profile_steps += 1
            torch.cuda.synchronize(); elapsed = time.monotonic() - start
            assert all(p.grad is None for m in self.models for p in m.parameters() if not p.requires_grad)
            profile[arm] = dict(seconds_per_step=elapsed / len(losses), losses=losses, grad_norms=norms)
            self.status('profile', arm=arm, seconds_per_step=profile[arm]['seconds_per_step'])
        projected = sum(x['seconds_per_step'] for x in profile.values()) * steps_per_epoch * self.c['epochs']
        passed = projected <= self.c['max_projected_training_seconds']
        write(self.out / 'profile.json', dict(arms=profile, projected_training_seconds=projected, passed=passed,
            discarded_updates=self.profile_steps, cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20))
        self.reset(); assert frozen_hash(self.models) == before
        if not passed:
            write(self.out / 'summary.json', dict(state='COMPLETED', decision='STOP_RESOURCE_GATE', profile=profile,
                projected_training_seconds=projected, formal_updates=0, new_trained_validation_predictions=0))
            self.finish(); signal.alarm(0); self.status('finished', state='COMPLETED', decision='STOP_RESOURCE_GATE'); return
        training = {}; cal_ix = np.arange(self.nfit, self.ntrain)
        for arm in self.c['arms']:
            self.reset(); optimizer = self.optimizer(); rng = np.random.RandomState(self.c['seed']); curve = []
            for epoch in range(self.c['epochs']):
                order = rng.permutation(self.nfit); total = 0.; max_norm = 0.; clipped = 0
                started = time.monotonic()
                for off in range(0, self.nfit, self.c['batch_size']):
                    ix = order[off:off + self.c['batch_size']]
                    loss, norm = self.update(optimizer, ix, arm); self.steps += 1
                    total += loss * len(ix) / self.nfit; max_norm = max(max_norm, norm); clipped += norm > self.c['gradient_clip']
                    if off % (10 * self.c['batch_size']) == 0:
                        self.status('training', arm=arm, epoch=epoch + 1, completed=off + len(ix), total=self.nfit)
                cf_cal = self.predict(cal_ix, 'calibration_' + arm)
                cal = metrics(ranks(self.final_scores(cf_cal, cal_ix), self.a['positions'][cal_ix]))
                curve.append(dict(epoch=epoch + 1, loss=total, max_grad_norm=max_norm, clipped_steps=int(clipped),
                                  calibration=cal, seconds=time.monotonic() - started))
                write(self.out / f'{arm}.training.json', curve)
                self.status('training_epoch_finished', arm=arm, epoch=epoch + 1, total_epochs=self.c['epochs'])
            torch.save([adapter_state(m) for m in self.models], self.out / f'{arm}.pt')
            assert frozen_hash(self.models) == before
            training[arm] = curve
        # All arms are trained before any trained-model development ranking is produced.
        for arm in self.c['arms']:
            self.reset(torch.load(self.out / f'{arm}.pt', map_location='cpu'))
            cf = self.predict(all_ix, 'final_prediction_' + arm); score = self.final_scores(cf, all_ix)
            np.savez_compressed(self.out / f'{arm}.predictions.npz', cf=cf, scores=score, ranks=ranks(score, self.a['positions']))
        self.summarize(training)
        self.verify_inputs(); assert frozen_hash(self.models) == before
        self.finish(); signal.alarm(0)
        self.status('finished', state='COMPLETED', decision=read(self.out / 'summary.json')['decision'],
                    cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20)

    def masks(self):
        index = np.arange(len(self.a['users'])); validation = index >= self.ntrain
        return dict(fit=index < self.nfit,
            fit_covered=(index < self.nfit) & (self.a['positions'] >= 0),
            fit_uncovered=(index < self.nfit) & (self.a['positions'] < 0),
            calibration=(index >= self.nfit) & (index < self.ntrain), validation=validation,
            validation_tail=np.concatenate((np.zeros(self.ntrain, bool), self.a['tail_mask'])),
            validation_nonfit=np.concatenate((np.zeros(self.ntrain, bool), ~self.a['validation_in_fit'])))

    def summarize(self, training):
        masks = self.masks(); met = {}; rr = {}
        for arm in ('frozen', *self.c['arms']):
            data = arrays(self.out / f'{arm}.predictions.npz'); rr[arm] = data['ranks']
            met[arm] = {group: metrics(data['ranks'][mask]) for group, mask in masks.items()}
        differences = {}
        for control in ('frozen', 'full_catalog', 'candidate_cf'):
            difference = {}
            for group, mask in masks.items():
                a, b = rr['coadapt'][mask], rr[control][mask]
                difference[group] = dict(delta_N10=met['coadapt'][group]['N10'] - met[control][group]['N10'],
                    delta_H10=met['coadapt'][group]['H10'] - met[control][group]['H10'],
                    hit_gained=int(((a<=10)&(b>10)).sum()), hit_lost=int(((a>10)&(b<=10)).sum()))
            differences[control] = difference
        own, base = met['coadapt']['validation'], met['frozen']['validation']
        gain = own['N10']/base['N10'] - 1
        gate = dict(relative_N10=gain, practical_gain=gain >= self.c['min_relative_ndcg_gain'],
            beats_full_catalog=own['N10'] > met['full_catalog']['validation']['N10'],
            beats_candidate_cf=own['N10'] > met['candidate_cf']['validation']['N10'],
            hit_protection=own['H10'] >= base['H10'] - self.c['max_hit_loss'],
            tail_protection=met['coadapt']['validation_tail']['N10'] >= met['frozen']['validation_tail']['N10'] - self.c['max_tail_ndcg_loss'])
        passed = all(gate[k] for k in ('practical_gain', 'beats_full_catalog', 'beats_candidate_cf', 'hit_protection', 'tail_protection'))
        write(self.out / 'summary.json', dict(state='COMPLETED', decision='SUPPORTED_FOR_REPLICATION' if passed else 'NO_INCREMENT_ESTABLISHED_THIS_RECIPE',
            metrics=met, comparisons=differences, gate=gate, formal_updates=self.steps, discarded_profile_updates=self.profile_steps,
            trainable_parameters=sum(p.numel() for p in self.params), frozen_CF_unchanged=True,
            original_model_updates=0, test_read=False, development_reused=True, single_initialization=True,
            training_last_losses={a: t[-1]['loss'] for a, t in training.items()}))

    def finish(self):
        paths = [p for p in self.out.iterdir() if p.is_file() and p.name not in ('status.json', 'output_manifest.json', 'verification.json')]
        write(self.out / 'output_manifest.json', dict(sha256=self.hashes(paths)))

    def verify(self):
        self.verify_inputs()
        for path, expected in read(self.out / 'output_manifest.json')['sha256'].items(): assert sha(ROOT / path) == expected, path
        # GPU run is done. CPU replay samples from all three cohorts, then recompute
        # every saved recommendation rank independently of the experiment sorter.
        self.load(torch.device('cpu')); summary = read(self.out / 'summary.json')
        if summary['decision'] == 'STOP_RESOURCE_GATE':
            write(self.out / 'verification.json', dict(state='VERIFIED_RESOURCE_STOP', no_formal_training=True)); return
        rng = np.random.RandomState(20260919)
        sample = np.concatenate((rng.choice(self.nfit, 128, replace=False), np.arange(self.nfit, self.ntrain),
                                 self.ntrain + rng.choice(len(self.a['users']) - self.ntrain, 256, replace=False)))
        errors = {}; count = 0; masks = self.masks()
        for arm in ('frozen', *self.c['arms']):
            self.reset(self.initial if arm == 'frozen' else torch.load(self.out / f'{arm}.pt', map_location='cpu'))
            saved = arrays(self.out / f'{arm}.predictions.npz')
            # Do not call predict(): verification must not overwrite completed status.
            with torch.no_grad():
                recomputed = np.concatenate([self.scores(sample[off:off+128])[0].numpy() for off in range(0, len(sample), 128)], axis=1)
            errors[arm] = float(np.abs(recomputed - saved['cf'][:, sample]).max())
            np.testing.assert_allclose(recomputed, saved['cf'][:, sample], atol=1e-4, rtol=0)
            cf = saved['cf'].astype(np.float64)
            # NumPy formula replay of the differentiable training calibration.
            np.testing.assert_allclose(self.final_scores(cf, np.arange(len(self.a['users']))), saved['scores'], atol=1e-12, rtol=0)
            independent = np.asarray([sorted(range(50), key=lambda j: (-s[j], j)).index(int(p)) + 1 if p >= 0 else 51
                                      for s, p in zip(saved['scores'], self.a['positions'])])
            np.testing.assert_array_equal(independent, saved['ranks']); count += len(independent)
            for group, mask in masks.items():
                r = independent[mask]
                assert float(((r<=10)/np.log2(r+1)).mean()) == summary['metrics'][arm][group]['N10']
                assert float((r<=10).mean()) == summary['metrics'][arm][group]['H10']
        write(self.out / 'verification.json', dict(state='VERIFIED', input_output_hashes=True,
            independent_ranks=count, cpu_replay_queries_per_arm=len(sample), cf_score_max_errors=errors,
            all_group_H10_N10_recomputed=True, scope='Sampled CPU checkpoint forward, full saved-score formula and rank replay; not retraining.', test_read=False))
        print(json.dumps(read(self.out / 'verification.json')))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('check', 'prepare', 'run', 'verify')); args = parser.parse_args()
    if args.mode == 'check':
        torch.set_num_threads(4); print(json.dumps(checks())); return
    work = Work(args.mode)
    try: getattr(work, args.mode)()
    except BaseException as exc:
        if work.out.exists() and args.mode != 'verify': work.status('exception', state='FAILED', error=repr(exc))
        raise


if __name__ == '__main__': main()
