"""Build legal train-prefix and development caches for the frozen experiment."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiment.phase20.guided_decoding_job import ROOT, read, write
from experiment.phase20.catalog_scope_core import ranks, topk
from experiment.phase20.lime_comparison_core import left_pad, semantic_users
from experiment.phase20.source_competition_core import candidate_union, make_features, target_positions

sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
from lime_rec.models import SASRec
from lime_rec.experts import ItemCFExpert
from lime_rec.evaluation import normalize_score_rows
from scripts.evaluation.run_repeat_aware_gate import features, RepeatAwareGate

LIME = ROOT/'artifacts/phase20/lime_comparison_20260920_v1'
RECOVERY = LIME/'recovery_v2'
NEAR = ROOT/'artifacts/phase20/nearest_comparison_20260918'
CATALOG = ROOT/'artifacts/phase20/catalog_scope_20260919_v1'
OLD = ROOT/'artifacts/phase20/gacr_fusion_train_v1'


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


class DataBuilder:
    def __init__(self, domain, config, use, status):
        self.domain, self.c, self.use, self.status = domain, config, use, status
        self.path = RECOVERY/domain
        self.q = arrays(use(CATALOG/domain/'queries.npz'))
        self.bank = arrays(use(NEAR/f'{domain}.train_sequences.npz'))
        self.train_index = arrays(use(NEAR/f'{domain}.train.npz'))
        self.user_index = {str(u): i for i, u in enumerate(self.bank['users'])}
        self.v = len(self.q['frequencies'])-1
        emb = arrays(use(self.path/'embeddings.npz'))
        self.emb = emb['embeddings']; self.source_ids = emb['item_ids'].tolist()
        assert len(self.source_ids) == self.v == len(self.emb)
        records = [json.loads(l) for l in use(NEAR/f'{domain}.items.jsonl').open()]
        assert [r['source_item_id'] for r in records] == self.source_ids
        self.cf = ItemCFExpert(str(use(self.path/'itemcf.json')), self.source_ids)
        seed = self.c['sasrec_seed']
        state = torch.load(use(self.path/f'sasrec_seed{seed}.pt'), map_location='cpu')
        assert state['train_complete'] and state['seed'] == seed
        self.sas = SASRec(self.v, 64, 50, 2, 2, 0., architecture='legacy')
        self.sas.load_state_dict(state['state_dict']); self.sas.eval()
        self.gate_state = torch.load(use(self.path/f'gate_seed{seed}_cal2048.pt'), map_location='cpu')
        np.testing.assert_array_equal(np.sort(self.gate_state['fit_indices']), self.q['calibration_indices'])
        assert not set(self.gate_state['fit_indices']) & set(self.q['evaluation_indices'])
        gc = self.gate_state['config']
        self.gate = RepeatAwareGate(14, gc['initial_weights'], gc['max_penalty'])
        self.gate.load_state_dict(self.gate_state['state_dict']); self.gate.eval()
        self.gate_device = torch.device(config.get('gate_feature_device', 'cpu'))
        if self.gate_device.type == 'cuda':
            if torch.cuda.is_initialized():
                assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gate_gpu_uuid']
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = config['gate_gpu_uuid']
                assert torch.cuda.is_available()
                torch.cuda.set_per_process_memory_fraction(config['gate_cuda_memory_fraction'])
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            self.gate.to(self.gate_device)
        for model in (self.sas, self.gate):
            for p in model.parameters(): p.requires_grad_(False)

    def history(self, user, position=None):
        i = self.user_index[str(user)]
        seq = self.bank['items'][self.bank['offsets'][i]:self.bank['offsets'][i+1]]
        return seq.tolist() if position is None else seq[:position].tolist()

    def fuse(self, raw, histories, cached_weights=None, cached_penalty=None):
        normalized = np.stack([normalize_score_rows(s) for s in raw], axis=1)
        scores = torch.as_tensor(normalized, device=self.gate_device)
        seen = torch.zeros((len(histories), self.v), dtype=torch.bool, device=self.gate_device)
        for row, h in enumerate(histories): seen[row, np.asarray(h)-1] = True
        with torch.no_grad():
            w, p = self.gate(features(scores, torch.tensor(list(map(len, histories)), dtype=torch.float32, device=self.gate_device), seen))
            if cached_weights is not None:
                np.testing.assert_allclose(w.cpu(), cached_weights, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(p.cpu(), cached_penalty, rtol=1e-6, atol=1e-7)
                w = torch.tensor(cached_weights, dtype=torch.float32, device=self.gate_device)
                p = torch.tensor(cached_penalty, dtype=torch.float32, device=self.gate_device)
            fused = ((w[:, :, None]*scores).sum(1)-p[:, None]*seen).cpu().numpy()
        for i, h in enumerate(histories): fused[i, np.asarray(h[-20:])-1] = -np.inf
        return normalized, fused

    def build_rows(self, gram, histories, raw, weights=None, penalty=None):
        normalized, fused = self.fuse(raw, histories, weights, penalty)
        lime = topk(fused, np.arange(1, self.v+1))
        ids, ing, inl = candidate_union(gram, lime, histories)
        # All gathered candidates are eligible; zero padding is explicitly reset.
        x, prior = make_features(ids, ing, inl, gram, normalized, fused,
            self.q['frequencies'], np.asarray(list(map(len, histories))))
        return dict(ids=ids, ing=ing, inl=inl, x=x, prior=prior), lime, fused

    def train(self):
        cached = arrays(self.use(OLD/self.domain/'training_cache.npz'))
        excluded_users = set(map(str, self.q['users'][self.q['calibration_indices']]))
        rows, histories, indices = [], [], []
        total_fit = int(cached['fit'].sum())
        with self.use(OLD/self.domain/'train_candidates.jsonl').open() as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                assert row['split'] == 'train'
                assert (row['partition'] == 'fit') == bool(cached['fit'][i])
                if not cached['fit'][i] or row['user_id'] in excluded_users: continue
                t = row['train_index']; ui = self.train_index['user_index'][t]
                assert str(self.bank['users'][ui]) == row['user_id']
                assert self.train_index['target'][t] == row['target']
                assert self.train_index['source_row'][t] == t
                h = self.history(row['user_id'], self.train_index['position'][t])
                assert h and h[-20:] == row['history']
                np.testing.assert_array_equal(cached['candidate_ids'][i], row['union_ids'][:50])
                rows.append(row); histories.append(h); indices.append(i)
        gram = cached['candidate_ids'][indices]
        n = len(rows); chunks = []; times = []
        for start in range(0, n, self.c['score_batch']):
            tic = time.monotonic(); sl = slice(start, start+self.c['score_batch']); hs = histories[sl]
            with torch.no_grad(): sequential = self.sas.predict(torch.from_numpy(left_pad(hs)))[:, 1:].numpy()
            cf = np.stack([self.cf.score('', [self.source_ids[j-1] for j in h]) for h in hs])
            semantic = semantic_users(hs, self.emb) @ self.emb.T
            a, _, _ = self.build_rows(gram[sl], hs, (sequential, cf, semantic))
            chunks.append(a); times.append(time.monotonic()-tic)
            if start % 2048 == 0 or start+len(hs) == n:
                self.status('准备训练候选', domain=self.domain, done=start+len(hs), total=n,
                    seconds_per_batch=float(np.mean(times[-8:])))
        data = {k: np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
        data.update(users=np.asarray([r['user_id'] for r in rows]),
            targets=np.asarray([r['target'] for r in rows]), source_cache_indices=np.asarray(indices),
            train_indices=np.asarray([r['train_index'] for r in rows]))
        data['positions'] = target_positions(data['ids'], data['targets'])
        data['target_in_gram'] = ((gram == data['targets'][:, None]) &
            np.array([r['target'] not in h[-20:] for r, h in zip(rows, histories)])[:, None]).any(1)
        data['tail'] = self.q['frequencies'][data['targets']] <= np.quantile(self.q['frequencies'][1:], .25)
        assert not set(data['users']) & excluded_users
        meta = dict(original_fit_prefixes=total_fit, prefixes=n, excluded_calibration_prefixes=total_fit-n,
            users=len(set(data['users'])), gram_covered=int(data['target_in_gram'].sum()),
            union_covered=int((data['positions'] >= 0).sum()),
            external_only=int(((data['positions'] >= 0) & ~data['target_in_gram']).sum()),
            uncovered=int((data['positions'] < 0).sum()), all_train_boundaries_checked=True,
            mean_candidate_count=float((data['ids'] > 0).sum(1).mean()), seconds=float(sum(times)))
        return data, meta

    def development(self):
        seed = self.c['sasrec_seed']
        old = arrays(self.use(self.path/f'evaluation_seed{seed}_cal2048.npz'))
        crossfit = arrays(self.use(self.path/f'evaluation_seed{seed}_crossfit.npz'))
        ix = self.q['evaluation_indices']; users = self.q['users'][ix]; targets = self.q['targets'][ix]
        for a in (old, crossfit):
            np.testing.assert_array_equal(a['users'], users)
            np.testing.assert_array_equal(a['targets'], targets)
        experts = [np.load(self.use(self.path/f'sasrec_scores_seed{seed}.npy'), mmap_mode='r'),
            np.load(self.use(LIME/self.domain/'itemcf_scores.npy'), mmap_mode='r'),
            np.load(self.use(self.path/'semantic_scores.npy'), mmap_mode='r')]
        histories = [self.history(u) for u in users]
        names = old['arm_names'].tolist(); chunks = []; baseline = []
        for start in range(0, len(ix), self.c['score_batch']):
            sl = slice(start, start+self.c['score_batch']); indices = ix[sl]; hs = histories[sl]
            gram = self.q['candidate_ids'][indices]
            for i, h in zip(indices, hs): assert h[-20:] == self.q['history'][i][self.q['history'][i] > 0].tolist()
            a, lime, fused = self.build_rows(gram, hs, [np.asarray(e[indices]) for e in experts],
                old['weights'][sl], old['penalty'][sl])
            np.testing.assert_array_equal(lime, old['top50'][names.index('LIME'), sl])
            full_rank = ranks(fused, np.arange(1, self.v+1), targets[sl])
            gen_values = np.take_along_axis(fused, np.maximum(gram-1, 0), 1)
            gen_rank = ranks(np.where(gram > 0, gen_values, -np.inf), gram, targets[sl])
            np.testing.assert_array_equal(np.minimum(full_rank, 51), np.minimum(old['ranks'][names.index('LIME'), sl], 51))
            np.testing.assert_array_equal(gen_rank, old['ranks'][names.index('LIME_scores_G50'), sl])
            chunks.append(a); baseline.append(np.stack([full_rank, gen_rank]))
            if start % 2048 == 0 or start+len(indices) == len(ix):
                self.status('复现固定评价与准备并集', domain=self.domain, done=start+len(indices), total=len(ix))
        data = {k: np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
        rr = np.concatenate(baseline, axis=1)
        data.update(users=users, targets=targets, tail=self.q['tail_mask'][ix],
            positions=target_positions(data['ids'], targets),
            target_in_gram=(self.q['candidate_ids'][ix] == targets[:, None]).any(1),
            full_lime_ranks=rr[0], gram_lime_ranks=rr[1], A_ranks=old['A_ranks'],
            crossfit_gram_lime_ranks=crossfit['ranks'][crossfit['arm_names'].tolist().index('LIME_scores_G50')])
        return data, dict(users=len(ix), original_rank_parity=True, original_top50_parity=True,
            evaluation_labels_used_for_fit=False, gate='frozen_cal2048',
            mean_candidate_count=float((data['ids'] > 0).sum(1).mean()))
