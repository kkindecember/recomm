"""Bounded Toys probe: reuse GRAM/C1 representations to replace mean-CF scoring."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, old_score, z, ranks, metrics, sha
from experiment.phase20.future_fusion_alignment import CF0B2ItemHead
from experiment.phase20.future_label_multiseed import write
from experiment.phase20.shared_cf_probe_core import SharedCFHead, fuse, teaching_loss, next_item_loss, agreement, checks

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'experiment/phase20/shared_cf_probe_toys_v1.json'
OLD = ROOT / 'artifacts/phase20/gacr_fusion_train_v1/Toys'
ALIGN = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'
TALOS = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'
VALID = ROOT / 'artifacts/phase20/future_factor_screen_20260917/Toys.audit_validation_cache.npz'


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Probe:
    def __init__(self, mode):
        self.c = read(CONFIG)
        self.out = ROOT / self.c['output']
        self.mode, self.start, self.steps = mode, time.monotonic(), 0
        torch.set_num_threads(self.c['cpu_threads']); torch.set_num_interop_threads(1)

    def status(self, stage, state='RUNNING', **kw):
        write(self.out / 'status.json', dict(state=state, mode=self.mode, stage=stage,
              pid=os.getpid(), elapsed_seconds=time.monotonic() - self.start,
              updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), optimizer_steps=self.steps,
              backbone_updates=0, new_beam=False, test_read=False, **kw))

    def hashes(self, paths):
        return {str(p.relative_to(ROOT)): sha(p) for p in paths}

    def verify_protocol(self):
        protocol = read(self.out / 'protocol.json')
        assert protocol['config'] == self.c
        for path, expected in protocol['sha256'].items():
            assert sha(ROOT / path) == expected, path
        for path, expected in protocol['prepared_sha256'].items():
            assert sha(ROOT / path) == expected, path
        return protocol

    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.status('input_lineage')
        health = checks()
        cfg_path = ROOT / 'experiment/phase20/gacr_fusion_train_v1.json'
        cfg = read(cfg_path)['domains']['Toys']
        paths = {CONFIG, Path(__file__), ROOT / self.c['contract'], cfg_path,
                 ROOT / 'experiment/phase20/shared_cf_probe_core.py',
                 ROOT / 'experiment/phase20/run_shared_cf_probe.sh',
                 OLD / 'training_cache.npz', OLD / 'train_candidates.jsonl', OLD / 'B1.pt', VALID,
                 ALIGN / 'Toys.training_context.npz', ROOT / cfg['validation'],
                 (ROOT / cfg['validation']).with_name('train.jsonl'),
                 TALOS / f'Toys.{self.c["teacher_seed"]}.softmax.pt',
                 TALOS / f'Toys.{self.c["teacher_seed"]}.predictions.npz'}
        trusted = dict(read(cfg_path)['domains']['Toys']['input_sha256'])
        for directory in (ALIGN, TALOS):
            verification = read(directory / 'verification.json')
            assert verification['state'] == 'VERIFIED'
            trusted.update(read(directory / 'summary.json')['input_sha256'])
            trusted.update(verification['output_sha256'])
            paths.add(directory / 'verification.json'); paths.add(directory / 'summary.json')
        transfer_path = ROOT / cfg['transfer_config']; paths.add(transfer_path)
        transfer = read(transfer_path)
        paths.update(ROOT / p for p in (transfer['c1_checkpoint'], transfer['data_config'], transfer['c1_catalog_file']))
        dc = read(ROOT / transfer['data_config'])
        paths.update(ROOT / p for p in (dc['parent_checkpoint'], dc['original_config']))
        prepared = ROOT / dc['prepared']; paths.update(prepared / f for f in ('catalog.json', 'manifest.json'))
        a, ctx, v = arrays(OLD / 'training_cache.npz'), arrays(ALIGN / 'Toys.training_context.npz'), arrays(VALID)
        records = rows(OLD / 'train_candidates.jsonl')
        fit_ix = ctx['fit_indices']; cal_ix = np.flatnonzero(~a['fit'])
        ix = np.concatenate((fit_ix, cal_ix)); nfit, ncal = len(fit_ix), len(cal_ix)
        assert nfit == 6884 and ncal == 256 and a['fit'][fit_ix].all()
        selected = [records[i] for i in ix]
        source = rows(prepared / 'train.jsonl'); validation = rows(prepared / 'validation.jsonl')
        for i, rec in zip(ix, selected):
            actual = source[rec['train_index']]
            for key in ('user_id', 'history', 'target'): assert actual[key] == rec[key]
            np.testing.assert_array_equal(rec['union_ids'][:50], a['candidate_ids'][i])
            assert rec['target_position'] == a['positions'][i]
        assert {r['user_id'] for r in selected[:nfit]}.isdisjoint(r['user_id'] for r in selected[nfit:])
        np.testing.assert_array_equal(ctx['candidate_ids'], a['candidate_ids'][fit_ix])
        np.testing.assert_array_equal(ctx['users'], [r['user_id'] for r in selected[:nfit]])
        np.testing.assert_array_equal(v['users'], [r['user_id'] for r in validation])
        for j, row in enumerate(validation):
            ids = v['candidate_ids'][j].tolist()
            assert v['positions'][j] == (ids.index(row['target']) if row['target'] in ids else -1)
        g_b1 = old_score(torch.as_tensor(a['features'][ix]), torch.as_tensor(a['base'][ix]),
                         torch.load(OLD / 'B1.pt', map_location='cpu'))
        prior = z(g_b1) - np.log(49)
        np.testing.assert_allclose(prior[:nfit], ctx['prior'], atol=1e-12, rtol=0)
        head = Factor('next_only').eval().requires_grad_(False)
        head.load_state_dict(torch.load(TALOS / f'Toys.{self.c["teacher_seed"]}.softmax.pt', map_location='cpu'))
        def head_scores(features, p):
            out = []
            with torch.no_grad():
                for off in range(0, len(p), 256):
                    sl = slice(off, off + 256)
                    out.append(head(torch.as_tensor(features[sl]), torch.as_tensor(p[sl]))[0].numpy())
            return z(np.concatenate(out))
        base = head_scores(a['features'][ix], prior)
        vbase = head_scores(v['features'], v['prior'])
        original = arrays(TALOS / f'Toys.{self.c["teacher_seed"]}.predictions.npz')
        np.testing.assert_array_equal(original['candidate_ids'], v['candidate_ids'])
        np.testing.assert_allclose(vbase, original['raw_scores'][list(original['raw_names']).index('softmax')], atol=1e-11, rtol=0)
        np.testing.assert_array_equal(ranks(vbase + v['cf_mean'], v['positions']),
            original['ranks'][list(original['rank_names']).index('softmax.mean')])
        frequencies = a['frequencies'][ix]
        rel = 1 - (np.take_along_axis(frequencies, np.argsort(-g_b1, axis=1, kind='stable')[:, :10], 1) <= cfg['q1']).mean(1)
        np.testing.assert_array_equal(rel[:nfit], ctx['reliability'])
        cal_hist = np.zeros((ncal, 20), np.int64)
        for j, record in enumerate(selected[nfit:]): cal_hist[j, :len(record['history'])] = record['history']
        cf_cal = []
        for seed in (2023, 2024, 2025):
            self.status('calibration_teacher', cf_seed=seed)
            checkpoint_path = ROOT / (cfg['item_head'] if seed == 2023 else f'artifacts/phase9/p9s_multiseed/Toys/seed{seed}/item_head/best_item_head.pt')
            paths.add(checkpoint_path); ck = torch.load(checkpoint_path, map_location='cpu'); mc = ck['model_config']
            model = CF0B2ItemHead(mc['num_items'], mc['max_history'], mc['d_model'], mc['num_layers'], mc['num_heads'], mc['dropout'], mc['temperature_initial'])
            model.load_state_dict(ck['model_state_dict']); model.eval().requires_grad_(False)
            with torch.no_grad():
                user = model.encode(torch.as_tensor(cal_hist), torch.as_tensor(cal_hist).ne(0))
                items = F.normalize(model.item_embedding.weight, dim=-1)[torch.as_tensor(a['candidate_ids'][cal_ix])]
                cf_cal.append((model.logit_scale.exp().clamp(max=100) * (user[:, None] * items).sum(-1)).numpy().astype(np.float64))
            del model, ck, items, user
        np.testing.assert_allclose(cf_cal[0], a['cf'][cal_ix], atol=1e-4, rtol=0)
        cf = np.concatenate((ctx['cf_scores'], np.stack(cf_cal)), axis=1)
        target = z(z(cf) - .5 * z(np.log1p(frequencies))[None]).mean(0)
        np.testing.assert_allclose(rel[:nfit, None] * target[:nfit], ctx['cf_mean'], atol=1e-12, rtol=0)
        # Validation reliability needs exact train-only item counts, not a new fit.
        catalog = read(prepared / 'catalog.json')
        from experiment.phase20.cf_exposure_screen import reconstruct_training
        _, sequences, _, _ = reconstruct_training(source)
        frequency = np.bincount([i for h in sequences.values() for i in h], minlength=len(catalog['items']))
        vfreq = frequency[v['candidate_ids']]
        vb1 = v['prior'] + np.log(49)
        vrel = 1 - (np.take_along_axis(vfreq, np.argsort(-vb1, axis=1, kind='stable')[:, :10], 1) <= cfg['q1']).mean(1)
        assert np.all(v['cf_mean'][vrel == 0] == 0)
        # Original mean-CF scores retain the pre-reliability target even for r=0.
        pair_path = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917/Toys.cf_scores.npz'
        paths.add(pair_path); vp = arrays(pair_path)
        np.testing.assert_array_equal(vp['users'], v['users'])
        # CF cache stores the same set in the original ordering; the audit cache
        # uses the frozen B1 ordering. Never join score arrays by column position.
        column_order = []
        for old_ids, new_ids in zip(vp['candidate_ids'], v['candidate_ids']):
            mapping = {int(item): j for j, item in enumerate(old_ids)}
            assert len(mapping) == 50 and set(mapping) == set(new_ids.tolist())
            column_order.append([mapping[int(item)] for item in new_ids])
        aligned_cf = np.take_along_axis(vp['scores'], np.asarray(column_order)[None], axis=2)
        vtarget = z(z(aligned_cf) - .5 * z(np.log1p(vfreq))[None]).mean(0)
        np.testing.assert_allclose(vrel[:, None] * vtarget, v['cf_mean'], atol=1e-11, rtol=0)
        rng = np.random.RandomState(self.c['seed']); seen = set(); tiny = []
        for j in rng.permutation(nfit):
            uid = selected[j]['user_id']
            if uid not in seen:
                tiny.append(j); seen.add(uid)
            if len(tiny) == self.c['tiny_users']: break
        data = dict(features=np.concatenate((a['features'][ix], v['features'])),
                    candidate_ids=np.concatenate((a['candidate_ids'][ix], v['candidate_ids'])),
                    positions=np.concatenate((a['positions'][ix], v['positions'])),
                    base=np.concatenate((base, vbase)), target=np.concatenate((target, vtarget)),
                    reliability=np.concatenate((rel, vrel)), head=a['head'][ix], tail_mask=v['tail_mask'],
                    users=np.asarray([r['user_id'] for r in selected] + [r['user_id'] for r in validation]),
                    nfit=np.asarray(nfit), ncal=np.asarray(ncal), tiny=np.asarray(tiny),
                    source_indices=ix, validation_in_fit=np.isin(v['users'], ctx['users']))
        np.savez_compressed(self.out / 'queries.npz', **data)
        metadata = [{k: r[k] for k in ('user_id', 'history', 'target', 'train_index')} for r in selected]
        metadata += [dict(r, validation_index=j) for j, r in enumerate(validation)]
        write(self.out / 'records.json', metadata)
        # Import encoding modules now so their exact loaded sources are frozen too.
        from experiment.phase20.gacr_fusion_train import CandidateBuilder  # noqa: F401
        for module in list(sys.modules.values()):
            raw = getattr(module, '__file__', '')
            if raw and raw.endswith('.py'):
                p = Path(raw).resolve()
                if ROOT in p.parents: paths.add(p)
        hashes = self.hashes(paths)
        for path, actual in hashes.items():
            if path in trusted: assert actual == trusted[path], path
        write(self.out / 'protocol.json', dict(config=self.c, checks=health, sha256=hashes,
            prepared_sha256=self.hashes([self.out / 'queries.npz', self.out / 'records.json']),
            fit=nfit, calibration=ncal, validation=len(validation), fit_cal_users_disjoint=True,
            teacher_replayed_exactly=True, new_head_holdout_only=True, development_reused=True,
            no_new_future_labels=True, teacher='softmax.mean', teacher_seed=self.c['teacher_seed'], test_read=False))
        self.status('prepared', state='PREPARED', fit=nfit, calibration=ncal, validation=len(validation))

    def data(self):
        self.a = arrays(self.out / 'queries.npz'); self.records = read(self.out / 'records.json')
        self.nfit, self.ncal = int(self.a['nfit']), int(self.a['ncal'])
        self.ntrain = self.nfit + self.ncal

    def run(self):
        assert not (self.out / 'run_started.json').exists(), 'Existing run; no implicit retry'
        self.verify_protocol(); self.data()
        assert torch.cuda.is_available(), 'CUDA required for bounded encoder extraction'
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(self.c['cuda_memory_fraction'], 0)
        free, _ = torch.cuda.mem_get_info(); assert free / 2**20 >= self.c['minimum_free_mib']
        self.device = torch.device('cuda:0')
        write(self.out / 'run_started.json', dict(pid=os.getpid(), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0), start=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        def timeout(signum, frame): raise TimeoutError('Frozen 1-hour run budget exhausted')
        signal.signal(signal.SIGALRM, timeout); signal.alarm(self.c['max_run_seconds'])
        self.status('load_frozen_C1')
        from experiment.phase20.gacr_fusion_train import CandidateBuilder
        transfer = read(ROOT / 'experiment/phase20/gacr_v6_pcrf_toys_v1.json')
        self.builder = CandidateBuilder(transfer, self.device)
        encoder = self.builder.model.encoder
        assert not encoder.cf0_enabled and not encoder.hi_gram_enabled and encoder.migration_runtime.feature_hooks.is_identity
        self.frozen_before = state_hash(self.builder.dual)
        dim = self.builder.model.config.d_model
        self.pooled = np.zeros((len(self.records), dim), np.float32)
        self.encoded = np.zeros(len(self.records), bool)
        table = torch.zeros(len(self.builder.catalog['items']), dim, device=self.device)
        for item, j in self.builder.dual.item_to_index.items(): table[self.builder.numeric[item]] = self.builder.dual.catalog_head.weight[j]
        self.table = table.detach()
        self.t = {k: torch.as_tensor(self.a[k], device=self.device) for k in
                  ('features', 'candidate_ids', 'positions', 'base', 'target', 'reliability')}
        # Complete-path vs coarse-only equality, and feature parity on fixed samples.
        parity = []
        for j in self.a['tiny'][:4]:
            sample = self.sample(int(j), coarse=False)
            batch = self.builder.collator([sample]); ids = batch['item_text_ids'].to(self.device); mask = batch['item_text_masks'].to(self.device)
            with torch.no_grad():
                encoder.n_passages = ids.shape[1]
                hidden = encoder(input_ids=ids.reshape(1, -1), attention_mask=mask.reshape(1, -1), return_dict=True)[0]
                full = self.builder.dual.pool_coarse(hidden, mask, ids.shape[-1])
                encoder.n_passages = 1
                hidden = encoder(input_ids=ids[:, 0], attention_mask=mask[:, 0], return_dict=True)[0]
                coarse = self.builder.dual.pool_coarse(hidden, mask[:, :1], ids.shape[-1])
                error = float((coarse - full).abs().max()); assert error <= 3e-5, error
                cos = F.cosine_similarity(coarse[:, None], self.table[self.t['candidate_ids'][j]][None], dim=-1)[0]
                feature_error = float((cos - self.t['features'][j, :, 5]).abs().max()); assert feature_error <= 3e-5, feature_error
                parity.append(dict(index=int(j), full_vs_coarse_max=error, cached_cosine_max=feature_error))
        del hidden, ids, mask, batch, full, coarse
        write(self.out / 'encoder_parity.json', parity)
        first = np.concatenate((self.a['tiny'], np.arange(self.nfit, self.ntrain)))
        self.encode(first, 'encode_tiny_and_calibration')
        model = self.new_head(dim)
        initial = copy.deepcopy(model.state_dict()); torch.save(initial, self.out / 'initial.pt')
        self.weights = torch.ones(len(self.records), device=self.device, dtype=torch.float64)
        tiny = self.a['tiny']; optimizer = self.optimizer(model)
        curve = []
        for step in range(self.c['tiny_steps']):
            loss, norm = self.update(model, optimizer, tiny, 'distill')
            if step % 25 == 0 or step + 1 == self.c['tiny_steps']:
                curve.append(dict(step=step + 1, loss=loss, grad_norm=norm))
                self.status('tiny_fit', completed=step + 1, total=self.c['tiny_steps'])
        pred = self.predict(model, tiny)
        result = self.diagnostics(pred, tiny)
        gate = (result['mse_reduction'] >= self.c['tiny_min_mse_reduction'] and
                result['teacher_top10_overlap'] >= self.c['tiny_min_top10_overlap'] and
                result['up_down_satisfaction'] is not None and result['up_down_satisfaction'] >= self.c['tiny_min_up_down'])
        cal_ix = np.arange(self.nfit, self.ntrain); cal_pred = self.predict(model, cal_ix)
        tiny_result = dict(fit=result, calibration=self.diagnostics(cal_pred, cal_ix), passed=bool(gate), curve=curve)
        write(self.out / 'tiny.json', tiny_result); torch.save(model.state_dict(), self.out / 'tiny.pt')
        np.savez_compressed(self.out / 'tiny_predictions.npz', indices=tiny, prediction=pred, cal_indices=cal_ix, cal_prediction=cal_pred)
        self.save_cache()
        if not gate:
            assert state_hash(self.builder.dual) == self.frozen_before
            summary = dict(state='COMPLETED', decision='STOP_TINY_LEARNABILITY_GATE', tiny=tiny_result,
                optimizer_steps=self.steps, full_training=False, new_validation_predictions=0,
                frozen_C1_unchanged=True, test_read=False)
            write(self.out / 'summary.json', summary); self.finish_manifest(); signal.alarm(0)
            self.status('finished', state='COMPLETED', decision=summary['decision']); return
        self.encode(np.arange(self.nfit), 'encode_remaining_fit')
        self.save_cache()
        head = self.a['head'][:self.nfit]
        self.weights[:self.nfit] = torch.as_tensor(np.where(head, self.nfit / (2 * head.sum()), self.nfit / (2 * (~head).sum())), device=self.device)
        trained = {}; training = {}
        for arm in ('ordinary_ce', 'distill'):
            model = self.new_head(dim); model.load_state_dict(initial); optimizer = self.optimizer(model)
            rng = np.random.RandomState(self.c['seed']); log = []
            for epoch in range(self.c['epochs']):
                order = rng.permutation(self.nfit); total_loss = 0.; norms = []
                for off in range(0, self.nfit, self.c['batch_size']):
                    index = order[off:off + self.c['batch_size']]
                    loss, norm = self.update(model, optimizer, index, arm)
                    total_loss += loss * len(index) / self.nfit; norms.append(norm)
                log.append(dict(epoch=epoch + 1, loss=total_loss, max_grad_norm=max(norms)))
                self.status('full_training', arm=arm, epoch=epoch + 1, total_epochs=self.c['epochs'])
            trained[arm] = model; training[arm] = log
            torch.save(model.state_dict(), self.out / f'{arm}.pt')
        write(self.out / 'training.json', training)
        self.encode(np.arange(self.ntrain, len(self.records)), 'encode_full_validation')
        self.save_cache()
        predictions = {arm: self.predict(model, np.arange(len(self.records))) for arm, model in trained.items()}
        self.evaluate(predictions)
        assert state_hash(self.builder.dual) == self.frozen_before
        self.verify_protocol(); self.finish_manifest(); signal.alarm(0)
        self.status('finished', state='COMPLETED', decision=read(self.out / 'summary.json')['decision'])

    def sample(self, j, coarse):
        row = self.records[j]
        sample = self.builder.sample(row['train_index'], 'train') if j < self.ntrain else self.builder.sample(row['validation_index'], 'validation')
        if coarse: sample = dict(sample, input=sample['input'][:1])
        # Encoder sees history only; use a constant output to prevent target dependence.
        return dict(sample, output=self.builder.catalog['lexical_ids'][1])

    def encode(self, indices, stage):
        indices = np.asarray(indices)[~self.encoded[indices]]
        encoder = self.builder.model.encoder; started = time.monotonic()
        with torch.no_grad():
            for off in range(0, len(indices), self.c['encoder_batch_size']):
                ix = indices[off:off + self.c['encoder_batch_size']]
                batch = self.builder.collator([self.sample(int(j), coarse=True) for j in ix])
                ids = batch['item_text_ids'][:, :1].to(self.device); mask = batch['item_text_masks'][:, :1].to(self.device)
                encoder.n_passages = 1
                hidden = encoder(input_ids=ids[:, 0], attention_mask=mask[:, 0], return_dict=True)[0]
                pooled = self.builder.dual.pool_coarse(hidden, mask, ids.shape[-1])
                assert torch.isfinite(pooled).all()
                self.pooled[ix] = pooled.cpu().numpy(); self.encoded[ix] = True
                if off == 0:
                    # Catch any discrepancy from collating only the first passage.
                    cosine = F.cosine_similarity(pooled[:, None], self.table[self.t['candidate_ids'][ix]], dim=-1)
                    torch.testing.assert_close(cosine, self.t['features'][ix, :, 5], atol=3e-5, rtol=0)
                completed = off + len(ix)
                estimate = (time.monotonic() - started) / completed * (len(indices) - completed)
                remaining = self.c['max_run_seconds'] - (time.monotonic() - self.start)
                assert estimate + 30 < remaining, 'Extraction projected beyond remaining hard budget'
                self.status(stage, completed=completed, total=len(indices), eta_seconds=estimate,
                            cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20)

    def new_head(self, dim):
        torch.manual_seed(self.c['seed'])
        return SharedCFHead(dim, self.c['projection'], self.c['hidden']).to(self.device)

    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.c['learning_rate'], weight_decay=self.c['weight_decay'])

    def forward(self, model, ix):
        assert self.encoded[ix].all()
        u = torch.as_tensor(self.pooled[ix], device=self.device)
        return model(u, self.table[self.t['candidate_ids'][ix]], self.t['features'][ix])

    def update(self, model, optimizer, ix, arm):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pred = self.forward(model, ix)
        if arm == 'distill': loss = teaching_loss(pred, self.t['target'][ix])
        else:
            assert (self.t['positions'][ix] >= 0).all()
            loss = next_item_loss(fuse(self.t['base'][ix], pred.double(), self.t['reliability'][ix]), self.t['positions'][ix])
        loss = (loss * self.weights[ix]).mean(); assert torch.isfinite(loss)
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.c['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); self.steps += 1
        return float(loss.detach()), float(norm)

    def predict(self, model, ix):
        model.eval(); out = []
        with torch.no_grad():
            for off in range(0, len(ix), self.c['batch_size']):
                out.append(self.forward(model, ix[off:off + self.c['batch_size']]).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def diagnostics(self, pred, ix):
        target, base, rel = (self.a[k][ix] for k in ('target', 'base', 'reliability'))
        s, t = fuse(base, pred, rel), fuse(base, target, rel)
        mse = float(np.mean((pred - target) ** 2)); zero_mse = float(np.mean(target ** 2))
        return dict(mse=mse, zero_mse=zero_mse, mse_reduction=1 - mse / zero_mse,
                    **agreement(s, t, base), student=metrics(ranks(s, self.a['positions'][ix])),
                    teacher=metrics(ranks(t, self.a['positions'][ix])), no_cf=metrics(ranks(base, self.a['positions'][ix])))

    def evaluate(self, predictions):
        score = {'no_cf': self.a['base'], 'teacher': fuse(self.a['base'], self.a['target'], self.a['reliability'])}
        score.update({arm: fuse(self.a['base'], p, self.a['reliability']) for arm, p in predictions.items()})
        rr = {arm: ranks(s, self.a['positions']) for arm, s in score.items()}
        masks = dict(fit=np.arange(len(self.records)) < self.nfit,
                     calibration=(np.arange(len(self.records)) >= self.nfit) & (np.arange(len(self.records)) < self.ntrain),
                     validation=np.arange(len(self.records)) >= self.ntrain)
        masks['validation_tail'] = np.concatenate((np.zeros(self.ntrain, bool), self.a['tail_mask']))
        masks['validation_nonfit'] = np.concatenate((np.zeros(self.ntrain, bool), ~self.a['validation_in_fit']))
        met = {arm: {group: metrics(r[mask]) for group, mask in masks.items()} for arm, r in rr.items()}
        diagnostics = {arm: {group: self.diagnostics(p[np.flatnonzero(mask)], np.flatnonzero(mask))
                      for group, mask in masks.items()} for arm, p in predictions.items()}
        teacher, base, student, ordinary = [met[a]['validation'] for a in ('teacher', 'no_cf', 'distill', 'ordinary_ce')]
        gain = teacher['N10'] - base['N10']
        retention = (student['N10'] - base['N10']) / gain if gain > 0 else None
        gate = dict(increment_retention=retention, retention=retention is not None and retention >= self.c['continue_min_increment_retention'],
                    beats_ordinary_ce=student['N10'] > ordinary['N10'],
                    hit_protection=student['H10'] >= teacher['H10'] - self.c['continue_max_hit_loss'],
                    tail_protection=met['distill']['validation_tail']['N10'] >= met['teacher']['validation_tail']['N10'] - self.c['continue_max_tail_ndcg_loss'])
        passed = all(gate[k] for k in ('retention', 'beats_ordinary_ce', 'hit_protection', 'tail_protection'))
        summary = dict(state='COMPLETED', decision='READY_FOR_COST_AND_REPLICATION_CHECK' if passed else 'STOP_THIS_SHARED_CF_RECIPE',
            metrics=met, diagnostics=diagnostics, gate=gate, optimizer_steps=self.steps,
            head_parameters=sum(p.numel() for p in self.new_head(self.pooled.shape[1]).parameters()),
            frozen_C1_unchanged=True, test_read=False, development_reused=True, full_validation_users=len(self.records)-self.ntrain,
            evidence='Single new-head seed, fixed upstream models, reused development set; no end-to-end speed or novelty claim.')
        np.savez_compressed(self.out / 'predictions.npz', arms=np.asarray(list(score)), scores=np.stack(list(score.values())),
                            ranks=np.stack(list(rr.values())), component_names=np.asarray(list(predictions)), components=np.stack(list(predictions.values())),
                            users=self.a['users'], candidate_ids=self.a['candidate_ids'], positions=self.a['positions'])
        write(self.out / 'summary.json', summary)

    def save_cache(self):
        np.savez_compressed(self.out / 'encoder_cache.npz', pooled=self.pooled, encoded=self.encoded)
        torch.save(self.table.detach().cpu(), self.out / 'item_table.pt')

    def finish_manifest(self):
        paths = [p for p in self.out.iterdir() if p.is_file() and p.name not in ('status.json', 'output_manifest.json', 'verification.json')]
        write(self.out / 'output_manifest.json', dict(sha256=self.hashes(paths)))

    def verify(self):
        self.verify_protocol(); self.data()
        for path, expected in read(self.out / 'output_manifest.json')['sha256'].items(): assert sha(ROOT / path) == expected, path
        cache = arrays(self.out / 'encoder_cache.npz'); table = torch.load(self.out / 'item_table.pt', map_location='cpu')
        tiny = arrays(self.out / 'tiny_predictions.npz'); summary = read(self.out / 'summary.json')
        # Fresh CPU forward, then independently construct ranks via stable sorted lists.
        def replay(checkpoint, indices, expected):
            model = SharedCFHead(cache['pooled'].shape[1], self.c['projection'], self.c['hidden'])
            model.load_state_dict(torch.load(self.out / checkpoint, map_location='cpu')); model.eval()
            values = []
            with torch.no_grad():
                for off in range(0, len(indices), 128):
                    ix = indices[off:off + 128]; assert cache['encoded'][ix].all()
                    values.append(model(torch.as_tensor(cache['pooled'][ix]), table[torch.as_tensor(self.a['candidate_ids'][ix])],
                                        torch.as_tensor(self.a['features'][ix])).numpy())
            result = np.concatenate(values)
            np.testing.assert_allclose(result, expected, atol=3e-5, rtol=1e-5)
            return float(np.max(np.abs(result - expected)))
        errors = {'tiny': replay('tiny.pt', tiny['indices'], tiny['prediction'])}
        actual = self.diagnostics(tiny['prediction'], tiny['indices'])
        assert actual == read(self.out / 'tiny.json')['fit']
        count = 0
        if (self.out / 'predictions.npz').exists():
            p = arrays(self.out / 'predictions.npz')
            for name, component in zip(p['component_names'], p['components']):
                errors[str(name)] = replay(str(name) + '.pt', np.arange(len(self.records)), component)
            for name, score, cached in zip(p['arms'], p['scores'], p['ranks']):
                independent = []
                for row, pos in zip(score, p['positions']):
                    order = sorted(range(len(row)), key=lambda j: (-row[j], j))
                    independent.append(order.index(int(pos)) + 1 if pos >= 0 else 51)
                np.testing.assert_array_equal(independent, cached); count += len(cached)
                np.testing.assert_allclose(metrics(cached[self.ntrain:])['N10'], summary['metrics'][str(name)]['validation']['N10'], atol=0, rtol=0)
        write(self.out / 'verification.json', dict(state='VERIFIED', frozen_inputs_and_outputs=True,
            CPU_checkpoint_score_max_errors=errors, independently_recomputed_ranks=count, tiny_diagnostics_recomputed=True,
            scope='Cached inference replay and ranks; not independent retraining.', test_read=False))
        print(json.dumps(read(self.out / 'verification.json')))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('check', 'prepare', 'run', 'verify')); args = parser.parse_args()
    if args.mode == 'check':
        torch.set_num_threads(4); print(json.dumps(checks())); return
    probe = Probe(args.mode)
    try: getattr(probe, args.mode)()
    except BaseException as exc:
        if probe.out.exists() and args.mode != 'verify': probe.status('exception', state='FAILED', error=repr(exc))
        raise


if __name__ == '__main__': main()
