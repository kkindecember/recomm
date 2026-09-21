"""Frozen encoder/CF extraction, calibration-only selection, and scope evaluation."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.catalog_scope_core import (
    split_users, eligible_mask, masked_z, mixed_scores, restricted, ranks,
    topk, metrics, select_weight, contrast, utilities)
from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now, cf_paths
from experiment.phase20.future_fusion_alignment import CF0B2ItemHead
from experiment.phase20.personal_popularity_core import reconstruct_training

CONFIG = ROOT / 'experiment/phase20/catalog_scope_20260919_v1.json'


def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in (f.files if keys is None else keys)}


class Worker:
    def __init__(self, domain, preflight=False):
        self.domain = domain
        self.c = read(CONFIG)
        self.root = ROOT/(self.c['output']+'_preflight' if preflight else self.c['output'])
        self.out = self.root/domain
        if preflight:
            self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.inputs = {}
        self.expected = {}
        self.parity = {}
        self.status('核对输入与旧结果')
        torch.set_num_threads(self.c['cpu_threads'])
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(self.c['selection_seed'])
        np.random.seed(self.c['selection_seed'])

    def status(self, stage, state='RUNNING', **extra):
        write(self.out/'status.json', dict(state=state, stage=stage, domain=self.domain,
            updated_at=now(), pid=os.getpid(), elapsed_seconds=round(time.monotonic()-self.started, 1),
            new_optimizer_steps=0, new_beam_decodes=0, test_read=False,
            peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1) if torch.cuda.is_initialized() else 0,
            **extra))

    def use(self, path):
        path = Path(path)
        key = str(path.relative_to(ROOT))
        assert 'test.json' not in key and '/test/' not in key
        if key not in self.inputs:
            actual = sha(path)
            if key in self.expected:
                assert actual == self.expected[key], ('changed_legacy_input', key)
            self.inputs[key] = actual
        return path

    def load(self):
        self.cfg = read(self.use(ROOT/'experiment/phase20/gacr_fusion_train_v1.json'))['domains'][self.domain]
        self.expected.update(self.cfg['input_sha256'])
        for directory in ('talos_candidate_screen_20260917', 'future_factor_screen_20260917', 'pair_uncertainty_screen_20260917', 'future_fusion_alignment_20260917'):
            folder = ROOT/'artifacts/phase20'/directory
            verification = read(self.use(folder/'verification.json'))
            assert verification['state'] == 'VERIFIED'
            self.expected.update(read(self.use(folder/'summary.json'))['input_sha256'])
            self.expected.update(verification.get('output_sha256', {}))
        self.transfer = read(self.use(ROOT/self.cfg['transfer_config']))
        dc = read(self.use(ROOT/self.transfer['data_config']))
        prepared = ROOT/dc['prepared']
        self.status('核对训练频次和验证用户')
        self.catalog = read(self.use(prepared/'catalog.json'))
        self.records = [json.loads(line) for line in self.use(prepared/'validation.jsonl').open()]
        train = [json.loads(line) for line in self.use(prepared/'train.jsonl').open()]
        _, sequences, _, _ = reconstruct_training(train)
        self.n, self.items = len(self.records), len(self.catalog['items'])
        self.frequency = np.bincount([i for h in sequences.values() for i in h], minlength=self.items)
        del train, sequences
        self.users = np.asarray([r['user_id'] for r in self.records])
        self.targets = np.asarray([r['target'] for r in self.records], dtype=np.int64)
        self.history = np.zeros((self.n, 20), dtype=np.int64)
        for j, row in enumerate(self.records):
            h = row['history'][-20:]
            assert row['target'] not in h
            self.history[j, :len(h)] = h
        self.cal, self.evaluation = split_users(self.users, self.domain, self.c['selection_seed'], self.c['calibration_users'])
        assert self.n == self.cfg['users']
        self.v = arrays(self.use(ROOT/f'artifacts/phase20/future_factor_screen_20260917/{self.domain}.audit_validation_cache.npz'))
        np.testing.assert_array_equal(self.v['users'], self.users)
        self.ids = self.v['candidate_ids']
        assert self.ids.shape == (self.n, 50)
        assert all(len(set(row)) == 50 for row in self.ids)
        actual_positions = np.where((self.ids == self.targets[:, None]).any(1), (self.ids == self.targets[:, None]).argmax(1), -1)
        np.testing.assert_array_equal(actual_positions, self.v['positions'])
        cf = arrays(self.use(ROOT/f'artifacts/phase20/pair_uncertainty_screen_20260917/{self.domain}.cf_scores.npz'))
        np.testing.assert_array_equal(cf['users'], self.users)
        column_orders = []
        for old, new in zip(cf['candidate_ids'], self.ids):
            mapping = {int(item): j for j, item in enumerate(old)}
            column_orders.append([mapping[int(item)] for item in new])
        ix = np.asarray(column_orders)
        self.cached_cf = np.take_along_axis(cf['scores'], ix[None], axis=2)
        self.a_scores = {}
        self.legacy_ranks = {}
        for seed in self.c['head_seeds']:
            self.status('核对当前方案旧得分和名次', head_seed=seed)
            old = arrays(self.use(ROOT/f'artifacts/phase20/talos_candidate_screen_20260917/{self.domain}.{seed}.predictions.npz'))
            np.testing.assert_array_equal(old['users'], self.users)
            np.testing.assert_array_equal(old['candidate_ids'], self.ids)
            score = old['raw_scores'][old['raw_names'].tolist().index('softmax')] + self.v['cf_mean']
            positions = self.v['positions']
            target = score[np.arange(self.n), np.maximum(positions, 0)]
            before = (score > target[:, None]) | ((score == target[:, None]) & (np.arange(50)[None] < positions[:, None]))
            old_ranks = np.where(positions >= 0, 1+before.sum(1), 51)
            np.testing.assert_array_equal(old_ranks, old['ranks'][old['rank_names'].tolist().index('softmax.mean')])
            self.a_scores[seed] = score
            self.legacy_ranks[seed] = old_ranks
        del old, cf
        write(self.out/'cohorts.json', dict(calibration_indices=self.cal.tolist(), evaluation_indices=self.evaluation.tolist(),
            calibration_users=self.users[self.cal].tolist(), evaluation_users=self.users[self.evaluation].tolist(),
            historical_validation_reused=True))
        np.savez_compressed(self.out/'queries.npz', users=self.users, targets=self.targets, history=self.history,
            candidate_ids=self.ids, calibration_indices=self.cal, evaluation_indices=self.evaluation,
            frequencies=self.frequency, tail_mask=self.frequency[self.targets] <= self.cfg['q1'])
        np.savez_compressed(self.out/'A_scores.npz', seeds=np.asarray(self.c['head_seeds']),
            scores=np.stack([self.a_scores[s] for s in self.c['head_seeds']]),
            original_ranks=np.stack([self.legacy_ranks[s] for s in self.c['head_seeds']]))
        # Freeze model/data inputs before computing new recommendations.
        self.status('冻结检查点与输入文件哈希')
        for path in cf_paths(self.domain): self.use(ROOT/path)
        self.use(ROOT/self.transfer['c1_checkpoint'])
        self.use(ROOT/self.transfer['c1_catalog_file'])
        self.use(CONFIG)
        self.use(ROOT/self.c['contract'])
        for path in (prepared/'manifest.json', ROOT/dc['parent_checkpoint'], ROOT/dc['original_config']): self.use(path)
        self.parity['A_legacy_ranks_checked'] = self.n*len(self.c['head_seeds'])

    def encode(self):
        assert torch.cuda.is_available(), 'CUDA must be available in the host worker'
        self.device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(self.c['cuda_memory_fraction'], 0)
        self.status('读取目录模型和编码表示')
        numeric = {x: i for i, x in enumerate(self.catalog['items']) if i}
        names = [line.split(' ', 1)[0] for line in (ROOT/self.transfer['c1_catalog_file']).read_text().splitlines()]
        state = torch.load(ROOT/self.transfer['c1_checkpoint'], map_location='cpu')
        self.table = torch.zeros((self.items, state['catalog_head.weight'].shape[1]))
        self.bias = torch.zeros(self.items)
        mapping = torch.tensor([numeric[x] for x in names])
        self.table[mapping] = state['catalog_head.weight']
        self.bias[mapping] = state['catalog_head.bias']
        # The C1 buffer was constructed with CUDA log, whose rounding differs
        # from CPU log by a few FP32 ULPs. Reproduce that path without loosening
        # the exact count/mapping check.
        counts = torch.tensor([self.frequency[numeric[x]]+1 for x in names], dtype=torch.float32, device=self.device).log().cpu()
        torch.testing.assert_close(state['log_counts'].cpu(), counts, atol=0, rtol=0)
        del state
        if self.domain == 'Toys':
            folder = ROOT/'artifacts/phase20/shared_cf_probe_toys_v1'
            assert read(self.use(folder/'verification.json'))['state'] == 'VERIFIED'
            self.expected.update(read(self.use(folder/'output_manifest.json'))['sha256'])
            records = read(self.use(folder/'records.json'))
            start = len(records)-self.n
            for old, new in zip(records[start:], self.records):
                for key in ('user_id', 'history', 'target'): assert old[key] == new[key]
            cache = arrays(self.use(folder/'encoder_cache.npz'))
            assert cache['encoded'][start:].all()
            self.pooled = cache['pooled'][start:].copy()
            old_table = torch.load(self.use(folder/'item_table.pt'), map_location='cpu')
            torch.testing.assert_close(old_table, self.table, atol=0, rtol=0)
            self.parity['reused_encoder_queries'] = self.n
            del cache, old_table
        else:
            from experiment.phase20.gacr_fusion_train import CandidateBuilder
            b = CandidateBuilder(self.transfer, self.device)
            assert [r['user_id'] for r in b.validation.records] == self.users.tolist()
            encoder = b.model.encoder
            assert not encoder.cf0_enabled and not encoder.hi_gram_enabled and encoder.migration_runtime.feature_hooks.is_identity
            # The scorer only needs passage 0. Check against the full frozen encoder.
            indices = [0, self.n-1, max(range(self.n), key=lambda i: len(self.records[i]['history']))]
            errors = []
            def sample(i, coarse):
                x = dict(b.sample(i, 'validation'), output=b.catalog['lexical_ids'][1])
                if coarse: x['input'] = x['input'][:1]
                return x
            with torch.no_grad():
                for i in dict.fromkeys(indices):
                    batch = b.collator([sample(i, False)])
                    ids = batch['item_text_ids'].to(self.device)
                    mask = batch['item_text_masks'].to(self.device)
                    encoder.n_passages = ids.shape[1]
                    hidden = encoder(input_ids=ids.reshape(1, -1), attention_mask=mask.reshape(1, -1), return_dict=True)[0]
                    full = b.dual.pool_coarse(hidden, mask, ids.shape[-1])
                    encoder.n_passages = 1
                    hidden = encoder(input_ids=ids[:, 0], attention_mask=mask[:, 0], return_dict=True)[0]
                    coarse = b.dual.pool_coarse(hidden, mask[:, :1], ids.shape[-1])
                    err = float((full-coarse).abs().max())
                    assert err < 3e-5, err
                    errors.append(err)
                self.parity['full_vs_first_passage_max_error'] = max(errors)
                self.pooled = np.empty((self.n, self.table.shape[1]), dtype=np.float32)
                started = time.monotonic()
                for off in range(0, self.n, self.c['encoder_batch_size']):
                    end = min(off+self.c['encoder_batch_size'], self.n)
                    batch = b.collator([sample(i, True) for i in range(off, end)])
                    ids = batch['item_text_ids'][:, :1].to(self.device)
                    mask = batch['item_text_masks'][:, :1].to(self.device)
                    encoder.n_passages = 1
                    hidden = encoder(input_ids=ids[:, 0], attention_mask=mask[:, 0], return_dict=True)[0]
                    self.pooled[off:end] = b.dual.pool_coarse(hidden, mask, ids.shape[-1]).cpu().numpy()
                    self.status('提取Beauty历史编码，无beam', completed=end, total=self.n,
                        stage_eta_seconds=(time.monotonic()-started)*(self.n-end)/end)
            assert all(not p.requires_grad for p in b.dual.parameters())
            del b, encoder, hidden, full, coarse, batch, ids, mask
            gc.collect()
            torch.cuda.empty_cache()
        assert np.isfinite(self.pooled).all()
        np.save(self.out/'pooled.npy', self.pooled)
        self.table = self.table.to(self.device)
        self.bias = self.bias.to(self.device)

    def extract(self):
        models = []
        for path in cf_paths(self.domain):
            checkpoint = torch.load(ROOT/path, map_location='cpu')
            c = checkpoint['model_config']
            model = CF0B2ItemHead(c['num_items'], c['max_history'], c['d_model'], c['num_layers'], c['num_heads'], c['dropout'], c['temperature_initial'])
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval().requires_grad_(False).to(self.device)
            models.append((model, F.normalize(model.item_embedding.weight, dim=-1)))
        del checkpoint
        directory = np.lib.format.open_memmap(self.out/'directory_z.npy', mode='w+', dtype=np.float32, shape=(self.n, self.items))
        collaborative = np.lib.format.open_memmap(self.out/'collaborative.npy', mode='w+', dtype=np.float32, shape=(self.n, self.items))
        self.parity.update(directory_feature_max_error=0., cosine_feature_max_error=0., cf_candidate_max_error=[0., 0., 0.])
        started = time.monotonic()
        with torch.no_grad():
            for off in range(0, self.n, self.c['scoring_batch_size']):
                end = min(off+self.c['scoring_batch_size'], self.n)
                sl = slice(off, end)
                pooled = torch.as_tensor(self.pooled[sl], device=self.device)
                raw = F.linear(pooled, self.table, self.bias).cpu().numpy()
                mask = eligible_mask(self.history[sl], self.items)
                d = masked_z(raw, mask)
                p = masked_z(np.broadcast_to(np.log1p(self.frequency), raw.shape), mask)
                directory[sl] = d.astype(np.float32)
                ids = self.ids[sl]
                old_d = self.v['features'][sl, :, 0]
                expected = np.where(restricted(mask, ids), np.clip(restricted(d, ids), -10, 10), -10)
                err = float(np.abs(expected-old_d).max())
                self.parity['directory_feature_max_error'] = max(self.parity['directory_feature_max_error'], err)
                cosine = F.cosine_similarity(pooled[:, None], self.table[torch.as_tensor(ids, device=self.device)], dim=-1).cpu().numpy()
                err = float(np.abs(cosine-self.v['features'][sl, :, 5]).max())
                self.parity['cosine_feature_max_error'] = max(self.parity['cosine_feature_max_error'], err)
                h = torch.as_tensor(self.history[sl], device=self.device)
                components = []
                for j, (model, table) in enumerate(models):
                    user = model.encode(h, h.ne(0))
                    raw_cf = (model.logit_scale.exp().clamp(max=100)*F.linear(user, table)).cpu().numpy()
                    err = float(np.abs(restricted(raw_cf, ids)-self.cached_cf[j, sl]).max())
                    self.parity['cf_candidate_max_error'][j] = max(self.parity['cf_candidate_max_error'][j], err)
                    components.append(masked_z(masked_z(raw_cf, mask)-self.c['frequency_coefficient']*p, mask))
                collaborative[sl] = np.mean(components, axis=0).astype(np.float32)
                assert self.parity['directory_feature_max_error'] < 1e-4, self.parity
                assert self.parity['cosine_feature_max_error'] < 3e-5, self.parity
                assert max(self.parity['cf_candidate_max_error']) < 1e-4, self.parity
                self.status('批量计算全目录分数，无训练', completed=end, total=self.n,
                    stage_eta_seconds=(time.monotonic()-started)*(self.n-end)/end)
        directory.flush(); collaborative.flush()
        self.directory = directory
        self.collaborative = collaborative
        self.parity['cached_score_dtype'] = 'float32; mixing/ranking float64'
        write(self.out/'engineering_checks.json', self.parity)
        del models, self.table, self.bias
        gc.collect()
        torch.cuda.empty_cache()

    def snapshot(self):
        for module in list(sys.modules.values()):
            raw = getattr(module, '__file__', '')
            if raw and raw.endswith('.py'):
                path = Path(raw).resolve()
                if ROOT in path.parents:
                    self.use(path)
                    target = self.out/'sources'/path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        write(self.out/'manifest.json', dict(created_at=now(), sha256=self.inputs, config=self.c,
            optimizer_steps=0, new_beam_decodes=0, evaluation_metrics_computed=False, historical_validation_reused=True))

    def scores(self, indices, weight):
        mask = eligible_mask(self.history[indices], self.items)
        score = mixed_scores(self.directory[indices], self.collaborative[indices], weight, mask)
        return score, restricted(score, self.ids[indices])

    def calibrate(self):
        table = []
        for weight in self.c['weights']:
            br, cr = [], []
            for off in range(0, len(self.cal), self.c['evaluation_batch_size']):
                ix = self.cal[off:off+self.c['evaluation_batch_size']]
                full, local = self.scores(ix, weight)
                br.extend(ranks(full, np.arange(self.items), self.targets[ix]))
                cr.extend(ranks(local, self.ids[ix], self.targets[ix]))
            table.append(dict(weight=weight, B=metrics(br), C=metrics(cr)))
            self.status('仅在校准用户上选融合权重', completed=len(table), total=len(self.c['weights']))
        chosen = select_weight(table, 'B')
        sensitivity = select_weight(table, 'C')
        selection = dict(created_at=now(), primary_weight=chosen, C_selected_sensitivity_weight=sensitivity,
            equal_weight=.5, calibration=table, evaluation_metrics_computed=0,
            primary_rule=self.c['primary_selection'], cohorts_sha256=sha(self.out/'cohorts.json'))
        write(self.out/'selection.json', selection)
        return chosen, sensitivity

    def evaluate(self, primary, sensitivity):
        weights = sorted(set((primary, sensitivity, .5)))
        variants = {float(w): {k: [] for k in ('B', 'C', 'directory50', 'CF50')} for w in weights}
        a_ranks = {s: [] for s in self.c['head_seeds']}
        lists = {k: [] for k in ('A', 'B', 'C', 'directory50', 'CF50')}
        a_history_removed = a_tie_changed = 0
        started = time.monotonic()
        for off in range(0, len(self.evaluation), self.c['evaluation_batch_size']):
            ix = self.evaluation[off:off+self.c['evaluation_batch_size']]
            ids, targets = self.ids[ix], self.targets[ix]
            mask = eligible_mask(self.history[ix], self.items)
            local_mask = restricted(mask, ids)
            a_history_removed += int((~local_mask).sum())
            for seed in self.c['head_seeds']:
                raw = self.a_scores[seed][ix]
                unfiltered = ranks(raw, ids, targets)
                old = self.legacy_ranks[seed][ix]
                a_tie_changed += int((np.minimum(unfiltered, 51) != old).sum())
                score = np.where(local_mask, raw, -np.inf)
                a_ranks[seed].extend(ranks(score, ids, targets))
                if seed == self.c['primary_head_seed']: lists['A'].append(topk(score, ids))
            d50 = topk(np.where(mask, self.directory[ix], -np.inf), np.arange(self.items))
            cf50 = topk(np.where(mask, self.collaborative[ix], -np.inf), np.arange(self.items))
            for weight in weights:
                full, local = self.scores(ix, weight)
                rr = dict(B=ranks(full, np.arange(self.items), targets), C=ranks(local, ids, targets),
                    directory50=ranks(restricted(full, d50), d50, targets),
                    CF50=ranks(restricted(full, cf50), cf50, targets))
                covered = (ids == targets[:, None]).any(1)
                assert (rr['C'][covered] <= rr['B'][covered]).all()
                assert np.array_equal(local, restricted(full, ids))
                for arm, r in rr.items(): variants[weight][arm].extend(r)
                if weight == primary:
                    lists['B'].append(topk(full, np.arange(self.items)))
                    lists['C'].append(topk(local, ids))
                    lists['directory50'].append(topk(restricted(full, d50), d50))
                    lists['CF50'].append(topk(restricted(full, cf50), cf50))
            end = off+len(ix)
            self.status('评价剩余用户，A/B/C共用合法目录', completed=end, total=len(self.evaluation),
                stage_eta_seconds=(time.monotonic()-started)*(len(self.evaluation)-end)/end)
        a_ranks = {s: np.asarray(r, dtype=np.int64) for s, r in a_ranks.items()}
        variants = {w: {a: np.asarray(r, dtype=np.int64) for a, r in arms.items()} for w, arms in variants.items()}
        rr = dict(A=a_ranks[self.c['primary_head_seed']], **variants[primary])
        top_lists = {a: np.concatenate(v) for a, v in lists.items()}
        for arm in rr:
            hit = top_lists[arm] == self.targets[self.evaluation, None]
            top_rank = np.where(hit.any(1), hit.argmax(1)+1, 51)
            np.testing.assert_array_equal(top_rank, np.minimum(rr[arm], 51))
        names = list(rr)
        np.savez_compressed(self.out/'evaluation.npz', users=self.users[self.evaluation],
            indices=self.evaluation, targets=self.targets[self.evaluation], arm_names=np.asarray(names),
            ranks=np.stack([rr[a] for a in names]), top50=np.stack([top_lists[a] for a in names]),
            sensitivity_weights=np.asarray(weights), sensitivity_arm_names=np.asarray(list(variants[primary])),
            sensitivity_ranks=np.stack([np.stack(list(variants[w].values())) for w in weights]),
            A_seeds=np.asarray(list(a_ranks)), A_ranks=np.stack(list(a_ranks.values())))
        tail = self.frequency[self.targets[self.evaluation]] <= self.cfg['q1']
        groups = {'all': np.ones(len(tail), bool), 'tail': tail, 'non_tail': ~tail}
        met = {a: {g: metrics(r[m]) for g, m in groups.items()} for a, r in rr.items()}
        comparisons = {a+'__vs__'+b: contrast(rr[a], rr[b], self.c['bootstrap_repeats'], self.c['selection_seed'])
                       for a, b in (('C', 'B'), ('A', 'B'), ('A', 'C'), ('C', 'directory50'), ('C', 'CF50'))}
        covered = (self.ids[self.evaluation] == self.targets[self.evaluation, None]).any(1)
        decomposition = {}
        for group, mask in (('target_in_generated50', covered), ('target_outside_generated50', ~covered)):
            decomposition[group] = dict(users=int(mask.sum()),
                contribution_to_C_minus_B_N10=float((utilities(rr['C'])-utilities(rr['B']))[mask].sum()/len(covered)),
                B_top10=int((rr['B'][mask] <= 10).sum()), C_top10=int((rr['C'][mask] <= 10).sum()))
        summary = dict(state='ANALYZED', domain=self.domain, calibration_users=len(self.cal), evaluation_users=len(self.evaluation),
            primary_weight=primary, metrics=met, comparisons=comparisons, coverage_decomposition=decomposition,
            A_head_seed_metrics={s: metrics(r) for s, r in a_ranks.items()},
            A_historical_unfiltered_metrics={s: metrics(r[self.evaluation]) for s, r in self.legacy_ranks.items()},
            A_history_items_filtered=a_history_removed, A_rank_tie_changes_total=a_tie_changed,
            sensitivity_metrics={str(w): {a: metrics(r) for a, r in arms.items()} for w, arms in variants.items()},
            group_users={g: int(m.sum()) for g, m in groups.items()}, engineering_checks=self.parity,
            no_new_optimizer_steps=True, no_new_beam=True, test_read=False,
            limitation='Reused development validation; scalar selected on separate users in this run. B still uses the frozen generative encoder/catalog head. No independent test or novelty claim.')
        write(self.out/'summary.json', summary)
        return summary

    def verify(self):
        self.status('独立重放主要排名与Top50')
        saved = arrays(self.out/'evaluation.npz')
        primary = read(self.out/'selection.json')['primary_weight']
        assert read(self.out/'selection.json')['evaluation_metrics_computed'] == 0
        arm_names = saved['arm_names'].tolist()
        probes = set(np.linspace(0, len(self.evaluation)-1, 128, dtype=int).tolist())
        checked = 0
        for off in range(0, len(self.evaluation), self.c['evaluation_batch_size']):
            ix = self.evaluation[off:off+self.c['evaluation_batch_size']]
            full, local = self.scores(ix, primary)
            for j, index in enumerate(ix):
                for name, score, ids in (('B', full[j], np.arange(self.items)), ('C', local[j], self.ids[index])):
                    target = self.targets[index]
                    matches = np.flatnonzero(ids == target)
                    if len(matches) and np.isfinite(score[matches[0]]):
                        value = score[matches[0]]
                        independent = 1+np.count_nonzero(score > value)+np.count_nonzero((score == value) & (ids < target))
                    else: independent = 2**30
                    a = arm_names.index(name)
                    assert independent == saved['ranks'][a, off+j]
                    if off+j in probes or name == 'C':
                        valid = np.flatnonzero(np.isfinite(score))
                        ordering = np.lexsort((ids[valid], -score[valid]))
                        ordered = ids[valid[ordering[:50]]]
                        expected = np.pad(ordered, (0, 50-len(ordered)))
                        np.testing.assert_array_equal(expected, saved['top50'][a, off+j])
                    checked += 1
            self.status('独立重放主要排名与Top50', completed=min(off+len(ix), len(self.evaluation)), total=len(self.evaluation))
        summary = read(self.out/'summary.json')
        for a, name in enumerate(arm_names):
            hits = saved['top50'][a] == saved['targets'][:, None]
            for k in (5, 10, 20, 50):
                utility = sum((1/np.log2(j+2))*hits[:, j].sum() for j in range(k))/len(hits)
                rate = hits[:, :k].any(1).mean()
                assert abs(utility-summary['metrics'][name]['all'][f'N{k}']) < 1e-12
                assert abs(rate-summary['metrics'][name]['all'][f'H{k}']) < 1e-12
        for path, expected in self.inputs.items():
            assert sha(ROOT/path) == expected, path
        outputs = {str(p.relative_to(self.out)): sha(p) for p in self.out.iterdir()
                   if p.is_file() and p.name not in ('status.json', 'worker.log', 'verification.json')}
        write(self.out/'verification.json', dict(state='VERIFIED_CACHED_PRIMARY_RANKS',
            checked_primary_ranks=checked, full_catalog_sorted_probes=len(probes),
            all_C_lists_independently_sorted=True, all_primary_metrics_recomputed_from_top50=True,
            input_hashes_unchanged=True, output_sha256=outputs, test_read=False,
            scope='CPU score-cache replay; model-forward parity against prior candidate caches checked during extraction; no backbone retraining.'))

    def run(self):
        self.load()
        self.encode()
        self.extract()
        self.snapshot()
        primary, sensitivity = self.calibrate()
        self.evaluate(primary, sensitivity)
        self.verify()
        self.status('本域完成，结果与独立排名复核通过', state='COMPLETED')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=('Toys', 'Beauty'), required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    worker = Worker(args.domain, args.preflight)
    try:
        if args.preflight:
            worker.load()
            worker.status('实际缓存身份和三小头旧排名核对通过', state='PRECHECK_PASSED',
                          users=worker.n, calibration_users=len(worker.cal), evaluation_users=len(worker.evaluation))
        else:
            worker.run()
    except BaseException as error:
        (worker.out/'failure.txt').write_text(traceback.format_exc())
        worker.status('失败，保留现场', state='FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
