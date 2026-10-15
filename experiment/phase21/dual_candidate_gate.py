"""Matched dual-expert calibration screen, stage 21.22.

SASRec/BGE expert implementations and the frozen shared gate originate in
LIME-Rec. This experiment changes the calibration granularity, not the experts.
The frozen gate is algebraically projected to its two retained outputs/features.
No ItemCF values, proposals, statistics, or old three-expert fusion are used.
Generation is deliberately a separate stage: these are item-ranking results.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiment.phase21.candidate_gate_core import compressed_loss, fold, cohort, ranking, utilities
from experiment.phase21.candidate_gate_screen import ROOT, SOURCE, FULL, write, sha

OUT = ROOT / 'artifacts/phase21/dual_candidate_gate_20261014_v1'
DUAL = ROOT / 'artifacts/phase21/expert_resource_recovery_20260923_v1/removal/without_itemcf'
PLAN = ROOT / 'plan/第二十一阶段/21.22_双专家候选级权重探索_20261014.md'
REPORT = ROOT / 'report/第二十一阶段/双专家候选级权重探索_20261014.md'
DOMAINS = ('Toys', 'Beauty')
ARMS = ('user', 'candidate', 'scalar')
KEPT = [0, 2]
COLS = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
DU, DC, WIDTH, M = 13, 14, 27, 150
USER_NAMES = ['log_history_length'] + [f'{s}_{e}' for s in ('confidence', 'margin', 'std', 'contamination') for e in ('sas', 'text')] + ['history_ratio', 'w_sas', 'w_text', 'penalty']
CAND_NAMES = [f'{s}_{e}' for s in ('score', 'margin_to_reference', 'reciprocal_rank', 'top10') for e in ('sas', 'text')] + ['fused', 'fused_margin', 'fused_rr', 'gap10_11', 'gap1_10', 'log_frequency']
CFG = dict(seeds=[0, 1, 2], epochs=20, checkpoints=[0, 4, 8, 12, 20],
           batch_size=64, lr=.002, weight_decay=.001, eta=1., kl=.2,
           score_scale=20., threads_per_worker=2, timeout_seconds=14400,
           bootstrap_repeats=2000, bootstrap_seed=20261014,
           experts=['SASRec', 'BGE'], arms=list(ARMS), candidate_topk=50,
           candidate_maximum=M, width=WIDTH, hidden_width=32,
           user_names=USER_NAMES, candidate_names=CAND_NAMES,
           anchor='frozen previously fitted without_itemcf gate',
           head_training='ranker_fit only; internal hash 80/20 selection then all ranker_fit refit',
           candidate_scope='same retained-expert + frozen-dual top50 union for all arms',
           user_scope='one weight vector per user inside common candidate union; frozen anchor outside',
           test_read=False, true_generation=False, upstream_optimizer_steps=0,
           validation='reused exploratory validation; not independent confirmation',
           scalar='eta*tanh((z_sas-z_text)/2); same network/parameter count as gates')


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(path.read_text())


def retained_scores(raw, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return np.array(raw[rows[:, None], np.asarray(KEPT)[None, :], :], copy=True)


def event(domain, stage, **values):
    write(OUT/domain/'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(),
          updated_at=stamp(), test_read=False, true_generation=False, **values))
    print(domain, stage, values, flush=True)


class Anchor(torch.nn.Module):
    """Equivalent two-output gate, with no third-expert runtime dependency."""
    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint['state_dict']
        self.register_buffer('weight', state['linear.weight'][[0, 2, 3]][:, COLS].clone())
        self.register_buffer('bias', state['linear.bias'][[0, 2, 3]].clone())

    def forward(self, scores, histories):
        n, k, v = scores.shape
        assert k == 2
        history = torch.zeros(n, v, dtype=torch.bool)
        for j, h in enumerate(histories):
            history[j, np.asarray(h, int)-1] = True
        lengths = torch.tensor([len(h) for h in histories], dtype=torch.float32)
        top = scores.topk(2, dim=2).values
        x = torch.cat([lengths.log1p()[:, None]/5,
                       top[:, :, 0]-scores.mean(2), top[:, :, 0]-top[:, :, 1], scores.std(2),
                       history[:, None, :].expand_as(scores).gather(2, scores.topk(5, dim=2).indices).float().mean(2),
                       history.float().mean(1, keepdim=True)], 1)
        z = x @ self.weight.T + self.bias
        w, penalty = z[:, :2].softmax(1), .1*z[:, 2].sigmoid()
        fused = (scores*w[:, :, None]).sum(1)-penalty[:, None]*history
        return fused, torch.cat([x, w, penalty[:, None]], 1), w, penalty


def candidate_features(experts, fused, history, counts):
    """Target-free candidates/features, accepting only the two retained experts."""
    assert experts.shape[0] == 2
    eligible = np.ones(len(fused), bool)
    eligible[np.asarray(history[-20:], int)-1] = False
    orders, ranks = zip(*(ranking(x, eligible) for x in [*experts, fused]))
    ids = np.unique(np.concatenate([o[:50] for o in orders]))
    ref = orders[-1][9]
    s = experts[:, ids].T
    rr = np.stack([r[ids] for r in ranks[:-1]], axis=1)
    x = np.column_stack([s, s-experts[:, ref], 1./rr, rr <= 10,
                         fused[ids], fused[ids]-fused[ref], 1./ranks[-1][ids],
                         np.full(len(ids), fused[ref]-fused[orders[-1][10]]),
                         np.full(len(ids), fused[orders[-1][0]]-fused[ref]),
                         np.log1p(counts[ids+1])]).astype(np.float32)
    assert len(ids) <= M and x.shape == (len(ids), DC) and eligible[ids].all()
    return ids, x, ranks[-1]


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(WIDTH, 32)
        self.output = torch.nn.Linear(32, 2)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x, w0, arm):
        z = self.output(torch.relu(self.input(x)))
        if arm == 'scalar':
            return CFG['eta']*torch.tanh((z[..., 0]-z[..., 1])/2)
        base = w0.log().softmax(-1)
        return CFG['eta']*((w0.log()+z).softmax(-1)-base)


def check_anchor(domain, anchor):
    """Check every historical calibration/development row against saved outputs."""
    folder = DUAL/domain
    samples = read(folder/'samples.json')
    raw = np.load(SOURCE/domain/'experts.npy', mmap_mode='r')
    saved_fused = np.load(folder/'fused.npy', mmap_mode='r')
    saved = np.load(folder/'gate_prediction.npz')
    maxima = dict(fused=0., features=0., weights=0., penalty=0.)
    rank_checks = 0; tail_rank_differences = []
    with torch.no_grad():
        for start in range(0, len(samples), 64):
            ix = np.arange(start, min(start+64, len(samples)))
            ss = [samples[i] for i in ix]
            scores = torch.from_numpy(retained_scores(raw, [s['source_sample_index'] for s in ss]))
            f, x, w, p = anchor(scores, [s['full_history'] for s in ss])
            pairs = {'fused': (f.numpy(), saved_fused[ix]),
                     'features': (x[:, :10].numpy(), saved['features'][ix][:, COLS]),
                     'weights': (w.numpy(), saved['weights'][ix][:, KEPT]),
                     'penalty': (p.numpy(), saved['penalty'][ix])}
            for name, (actual, expected) in pairs.items():
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
                maxima[name] = max(maxima[name], float(np.abs(actual-expected).max()))
            for j, s in enumerate(ss):
                if s['role'] != 'development':
                    continue
                eligible = np.ones(raw.shape[-1], bool)
                eligible[np.asarray(s['history'])-1] = False
                y = s['target']-1
                actual_rank = int(ranking(f[j].numpy(), eligible)[1][y])
                saved_rank = int(ranking(saved_fused[ix[j]], eligible)[1][y])
                if actual_rank != saved_rank:
                    # CPU/projection roundoff can resolve an extreme-tail near-tie.
                    # Reject any change affecting the reported top10/top50 metrics.
                    assert min(actual_rank, saved_rank) > 50, (s['user_id'], actual_rank, saved_rank)
                    tail_rank_differences.append(dict(user_id=s['user_id'], actual=actual_rank, saved=saved_rank))
                rank_checks += 1
    assert rank_checks == 512
    return dict(state='PASSED', rows=len(samples), development_target_ranks_checked=rank_checks,
                exact_development_target_ranks=rank_checks-len(tail_rank_differences),
                all_development_N10_H10_H50_exact=True, tail_rank_differences=tail_rank_differences, max_abs=maxima)


class Data:
    def __init__(self, domain, training, anchor, limit=None):
        self.domain, self.training = domain, training
        origin = (SOURCE if training else FULL)/domain
        rows = read(origin/'samples.json')
        self.source_ix = np.array([i for i, s in enumerate(rows) if not training or s['role'] == 'ranker_fit'])
        if limit is not None:
            self.source_ix = self.source_ix[:limit]
        self.samples = [rows[i] for i in self.source_ix]
        self.n = len(self.samples)
        self.targets = np.array([s['target']-1 for s in self.samples], np.int64)
        self.folds = np.array([fold(s['user_id'], domain) for s in self.samples])
        self.cohorts = np.array(['train' if training else cohort(s) for s in self.samples])
        self.experts = np.load(origin/'experts.npy', mmap_mode='r')
        self.v = self.experts.shape[-1]
        counts = np.load(SOURCE/domain/'frequencies.npy')
        self.folder = OUT/domain/('train' if training else 'validation')
        self.folder.mkdir(parents=True, exist_ok=False)
        self.a = {}

        def arr(key, shape, dtype='float32'):
            self.a[key] = np.lib.format.open_memmap(self.folder/(key+'.npy'), mode='w+', dtype=dtype, shape=shape)
            return self.a[key]

        arr('user', (self.n, DU)); arr('candidate', (self.n, M, DC))
        arr('ids', (self.n, M), 'int64'); arr('mask', (self.n, M), 'bool')
        arr('scores', (self.n, M, 2)); arr('weights', (self.n, 2))
        arr('base', (self.n, M)); arr('outside', (self.n,))
        arr('target_base', (self.n,)); arr('target_pos', (self.n,), 'int64')
        arr('baseline_rank', (self.n,), 'int64'); arr('fused', (self.n, self.v))
        for a in self.a.values():
            a[:] = 0
        self.a['target_pos'][:] = -1
        begin = time.monotonic()
        with torch.no_grad():
            for start in range(0, self.n, 64):
                ix = np.arange(start, min(start+64, self.n))
                e = retained_scores(self.experts, self.source_ix[ix])
                histories = [self.samples[j]['full_history'] for j in ix]
                fused, user, w, _ = anchor(torch.from_numpy(e), histories)
                self.a['fused'][ix] = fused.numpy()
                self.a['user'][ix] = user.numpy()
                self.a['weights'][ix] = w.numpy()
                for j, k in enumerate(ix):
                    s, f = self.samples[k], fused[j].numpy()
                    assert s['history'] == s['full_history'][-20:]
                    ids, cx, rr = candidate_features(e[j], f, histories[j], counts)
                    m = len(ids)
                    self.a['ids'][k, :m] = ids
                    self.a['mask'][k, :m] = True
                    self.a['candidate'][k, :m] = cx
                    self.a['scores'][k, :m] = e[j, :, ids]
                    z = f.astype(np.float64)*20
                    z -= z.max()
                    self.a['base'][k, :m] = z[ids]
                    outside = np.ones(self.v, bool); outside[ids] = False
                    self.a['outside'][k] = np.log(np.exp(z[outside]).sum())
                    self.a['target_base'][k] = z[self.targets[k]]
                    match = np.flatnonzero(ids == self.targets[k])
                    if len(match):
                        self.a['target_pos'][k] = match[0]
                    self.a['baseline_rank'][k] = rr[self.targets[k]]
                done = int(ix[-1])+1
                if start == 0 or done % 1024 == 0 or done == self.n:
                    elapsed = time.monotonic()-begin
                    event(domain, 'features', split=self.folder.name, completed=done, total=self.n,
                          stage_elapsed_seconds=elapsed, stage_eta_seconds=elapsed*(self.n-done)/done)
        for a in self.a.values():
            a.flush()
        self.info = dict(users=self.n, target_in_candidates=int((self.a['target_pos'] >= 0).sum()),
                         mean_candidates=float(self.a['mask'].sum(1).mean()),
                         cohorts={c: int((self.cohorts == c).sum()) for c in sorted(set(self.cohorts))})
        write(self.folder/'metadata.json', self.info)
        write(self.folder/'users.json', [dict(user_id=s['user_id'], source_index=int(src), target=int(self.targets[j]),
              fold=int(self.folds[j]), cohort=str(self.cohorts[j])) for j, (src, s) in enumerate(zip(self.source_ix, self.samples))])

    def inputs(self, ix, arm):
        m = 1 if arm == 'user' else M
        x = np.zeros((len(ix), m, WIDTH), np.float32)
        x[:, :, :DU] = self.a['user'][ix, None, :]
        if arm != 'user':
            x[:, :, DU:] = self.a['candidate'][ix]
        return x

    def norm(self, ix, arm):
        total = np.zeros(WIDTH); square = total.copy(); n = 0
        for start in range(0, len(ix), 128):
            jj = ix[start:start+128]; x = self.inputs(jj, arm)
            mask = np.ones(x.shape[:2], bool) if arm == 'user' else self.a['mask'][jj]
            v = x[mask].astype(np.float64)
            total += v.sum(0); square += (v*v).sum(0); n += len(v)
        mean = total/n; std = np.sqrt(np.maximum(square/n-mean**2, 0))
        std[std < 1e-5] = 1.
        return mean.astype(np.float32), std.astype(np.float32)

    def delta(self, model, ix, arm, norm):
        x = np.clip((self.inputs(ix, arm)-norm[0])/norm[1], -10, 10)
        change = model(torch.from_numpy(x), torch.from_numpy(np.array(self.a['weights'][ix, None, :], copy=True)), arm)
        if arm == 'scalar':
            return change, None
        delta = (change*torch.from_numpy(np.array(self.a['scores'][ix], copy=True))).sum(2)
        return delta, change

    def objective(self, model, ix, arm, norm):
        delta, _ = self.delta(model, ix, arm, norm)
        values = [torch.from_numpy(np.array(self.a[k][ix], copy=True)) for k in ('base', 'mask', 'outside', 'target_base', 'target_pos')]
        return compressed_loss(values[0], delta, *values[1:], lam=CFG['kl'])[0]


def fit(data, arm, seed):
    fi, hi = np.flatnonzero(data.folds != 0), np.flatnonzero(data.folds == 0)
    norm = data.norm(fi, arm)

    def initialize():
        torch.manual_seed(seed)
        return Head()

    model = initialize()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    rng = np.random.default_rng(seed)

    def heldout():
        with torch.no_grad():
            return sum(float(data.objective(model, hi[s:s+64], arm, norm))*len(hi[s:s+64]) for s in range(0, len(hi), 64))/len(hi)

    curve = [dict(epoch=0, loss=heldout())]
    for epoch in range(1, CFG['epochs']+1):
        order = rng.permutation(fi)
        for start in range(0, len(order), 64):
            opt.zero_grad()
            loss = data.objective(model, order[start:start+64], arm, norm)
            assert torch.isfinite(loss)
            loss.backward(); opt.step()
        if epoch in CFG['checkpoints']:
            curve.append(dict(epoch=epoch, loss=heldout()))
            event(data.domain, 'head_selection', arm=arm, seed=seed, epoch=epoch, total=CFG['epochs'], heldout_loss=curve[-1]['loss'])
    selected = min(curve, key=lambda r: (r['loss'], r['epoch']))['epoch']
    norm = data.norm(np.arange(data.n), arm); model = initialize()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    rng = np.random.default_rng(seed)
    for epoch in range(selected):
        order = rng.permutation(data.n)
        for start in range(0, len(order), 64):
            opt.zero_grad(); loss = data.objective(model, order[start:start+64], arm, norm)
            assert torch.isfinite(loss)
            loss.backward(); opt.step()
        if (epoch+1) % 4 == 0:
            event(data.domain, 'head_refit', arm=arm, seed=seed, epoch=epoch+1, total=selected)
    info = dict(selected_epochs=selected, curve=curve, parameters=sum(p.numel() for p in model.parameters()),
                active_inputs=DU if arm == 'user' else WIDTH, selection_fit_users=len(fi), internal_holdout_users=len(hi))
    torch.save(dict(state_dict=model.state_dict(), mean=norm[0], std=norm[1], config=CFG,
                    arm=arm, seed=seed, fit_users=[s['user_id'] for s in data.samples], **info), OUT/data.domain/f'{arm}_{seed}.pt')
    return model, norm, info


def numpy_delta(data, model, norm, arm, k):
    """Independent NumPy forward, compared with all candidates for selected users."""
    x = np.clip((data.inputs(np.array([k]), arm)-norm[0])/norm[1], -10, 10)
    sd = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    z = np.maximum(x @ sd['input.weight'].T+sd['input.bias'], 0)
    z = z @ sd['output.weight'].T+sd['output.bias']
    if arm == 'scalar':
        return (CFG['eta']*np.tanh((z[..., 0]-z[..., 1])/2))[0]
    logw = np.log(data.a['weights'][k])

    def softmax(a):
        p = np.exp(a-a.max(-1, keepdims=True))
        return p/p.sum(-1, keepdims=True)

    dw = CFG['eta']*(softmax(logw+z)-softmax(logw))
    return (dw[0]*data.a['scores'][k]).sum(-1)


def evaluate(data, model, norm, arm, seed):
    ranks = np.empty(data.n, np.int64); spreads = []; checks = []; maximum = 0.
    check_ix = set(np.linspace(0, data.n-1, min(21, data.n), dtype=int).tolist())
    gen_rows = read(ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'/data.domain/'samples.json')
    chosen = {s['user_id']: j for j, s in enumerate(gen_rows)}
    signals = None; exported = set()
    if seed == 0:
        signals = np.lib.format.open_memmap(OUT/data.domain/f'{arm}_generation_signals.npy', mode='w+', dtype='float32', shape=(len(gen_rows), data.v))
    with torch.no_grad():
        for start in range(0, data.n, 128):
            ix = np.arange(start, min(start+128, data.n))
            delta, dw = data.delta(model, ix, arm, norm)
            delta = delta.numpy()
            if dw is not None:
                dw = dw.numpy()
            for j, k in enumerate(ix):
                mask = data.a['mask'][k]; ids = data.a['ids'][k, mask]
                f = np.array(data.a['fused'][k], copy=True)
                f[ids] += delta[j, mask]
                maximum = max(maximum, float(np.abs(delta[j, mask]).max()))
                if dw is not None and arm == 'candidate':
                    spreads.append(float(dw[j, mask].std(0).mean()))
                s = data.samples[k]
                if signals is not None and s['user_id'] in chosen:
                    gen_j = chosen[s['user_id']]
                    assert gen_rows[gen_j]['target'] == s['target'] and gen_rows[gen_j]['history'] == s['history']
                    signals[gen_j] = f; exported.add(gen_j)
                if k in check_ix:
                    np.testing.assert_allclose(delta[j, mask], numpy_delta(data, model, norm, arm, k)[mask], atol=3e-6, rtol=2e-5)
                    outside = np.ones(data.v, bool); outside[ids] = False
                    np.testing.assert_array_equal(f[outside], data.a['fused'][k, outside])
                f[np.asarray(s['history'])-1] = -np.inf
                y = data.targets[k]; t = f[y]
                ranks[k] = 1+np.count_nonzero(f > t)+np.count_nonzero((f == t) & (np.arange(data.v) < y)) if np.isfinite(t) else data.v+1
                if k in check_ix:
                    eligible = np.isfinite(f)
                    assert ranking(f, eligible)[1][y] == ranks[k]
                    checks.append(int(k))
            if start % 4096 == 0:
                event(data.domain, 'evaluate', arm=arm, seed=seed, completed=int(ix[-1])+1, total=data.n)
    assert maximum <= 1.+2e-6
    if signals is not None:
        assert len(exported) == len(gen_rows)
        signals.flush()
    return ranks, dict(numpy_forward_and_sort_checked=checks, outside_candidate_exact=True,
                       maximum_score_change=maximum, mean_candidate_weight_std=float(np.mean(spreads)) if spreads else 0.,
                       generation_signals_exported=len(exported))


def summarize(data, rr):
    result = {}
    contrasts = [('candidate', 'user'), ('candidate', 'scalar'), ('candidate', 'baseline'), ('user', 'baseline'), ('scalar', 'baseline')]
    for name in ('primary', 'previous512', 'all'):
        mask = np.ones(data.n, bool) if name == 'all' else data.cohorts == name
        metrics = {k: dict(N10=float(utilities(v[mask]).mean()), H10=float((v[mask] <= 10).mean()), H50=float((v[mask] <= 50).mean())) for k, v in rr.items()}
        u = {'baseline': utilities(rr['baseline'][mask])}
        for arm in ARMS:
            u[arm] = np.mean([utilities(rr[f'{arm}_{s}'][mask]) for s in CFG['seeds']], axis=0)
        diff = np.stack([u[a]-u[b] for a, b in contrasts], 1)
        rng = np.random.default_rng(CFG['bootstrap_seed']); boots = []
        for _ in range(CFG['bootstrap_repeats']//50):
            boots.append(diff[rng.integers(0, len(diff), size=(50, len(diff)))].mean(1))
        ci = np.quantile(np.concatenate(boots), [.025, .975], axis=0)
        comparisons = {}
        for j, (a, b) in enumerate(contrasts):
            changes = []
            for seed in CFG['seeds']:
                ra = rr[f'{a}_{seed}'][mask]
                rb = rr['baseline' if b == 'baseline' else f'{b}_{seed}'][mask]
                changes.append(dict(seed=seed, delta_N10=float((utilities(ra)-utilities(rb)).mean()),
                                    top10_gained=int(((ra <= 10) & (rb > 10)).sum()),
                                    top10_lost=int(((ra > 10) & (rb <= 10)).sum())))
            comparisons[f'{a}_vs_{b}'] = dict(delta_N10=float(diff[:, j].mean()), ci95_descriptive=ci[:, j].tolist(), seeds=changes)
        result[name] = dict(users=int(mask.sum()), metrics=metrics,
                            seed_mean_N10={k: float(v.mean()) for k, v in u.items()}, comparisons=comparisons)
    return result


def tests():
    from experiment.phase21.candidate_gate_core import tests as compressed_tests
    compressed_tests()
    torch.manual_seed(23)
    model = Head(); w = torch.rand(3, 1, 2).softmax(-1); x = torch.randn(3, M, WIDTH)
    for arm in ARMS:
        assert torch.count_nonzero(model(x, w, arm)) == 0
    with torch.no_grad():
        model.output.weight.normal_()
        dw = model(x, w, 'candidate')
        expected_w = w.log().softmax(-1)+dw
        assert expected_w.min() >= -2e-7 and expected_w.max() <= 1.+2e-7
        torch.testing.assert_close(expected_w.sum(-1), torch.ones(3, M), atol=2e-7, rtol=1e-6)
        assert model(x, w, 'scalar').abs().max() <= 1
    rng = np.random.default_rng(3)
    raw = rng.random((3, 213), dtype=np.float32)
    retained = raw[KEPT]; f = retained.mean(0)
    a = candidate_features(retained, f, [1, 3, 7], np.ones(214))
    raw[1] = rng.normal(100, 50, 213)
    b = candidate_features(raw[KEPT], f, [1, 3, 7], np.ones(214))
    for av, bv in zip(a, b):
        np.testing.assert_array_equal(av, bv)
    assert not np.isin(a[0], [0, 2, 6]).any()
    return dict(exact_zero_initialization=True, normalized_two_expert_weights=True,
                scalar_bound=True, removed_expert_invariance=True, candidate_history_filter=True,
                compressed_full_catalog_loss_and_gradient=True, outside_targets_retained=True)


def source_paths(domain):
    paths = [SOURCE/domain/n for n in ('samples.json', 'experts.npy', 'frequencies.npy', 'data_config.json', 'catalog.json')]
    paths += [FULL/domain/n for n in ('samples.json', 'experts.npy', 'score_cache_manifest.json')]
    paths += [DUAL/domain/n for n in ('samples.json', 'gate.pt', 'gate_prediction.npz', 'fused.npy', 'fit_checks.json')]
    paths += [ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'/domain/'samples.json']
    return paths


def worker(domain):
    torch.set_num_threads(CFG['threads_per_worker']); torch.set_num_interop_threads(1)
    begin = time.monotonic(); event(domain, 'verify_inputs')
    cfg = read(OUT/'config.json'); assert cfg == CFG
    for rel, digest in read(OUT/'manifest.json')['sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    paths = source_paths(domain)
    original_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    manifest = {}
    for p in paths:
        manifest[str(p.relative_to(ROOT))] = sha(p)
    expected = read(FULL/domain/'score_cache_manifest.json')['sha256']
    assert manifest[str((FULL/domain/'experts.npy').relative_to(ROOT))] == expected['experts.npy']
    write(OUT/domain/'input_manifest.json', dict(sha256=manifest, created_at=stamp()))
    ck = torch.load(DUAL/domain/'gate.pt', map_location='cpu')
    assert ck['arm'] == 'without_itemcf' and not ck['development_labels_used']
    anchor = Anchor(ck).eval()
    event(domain, 'verify_anchor')
    anchor_checks = check_anchor(domain, anchor)
    write(OUT/domain/'anchor_checks.json', anchor_checks)
    torch.save(dict(state_dict=anchor.state_dict(), source_sha256=sha(DUAL/domain/'gate.pt'), fit_users=ck['fit_users']), OUT/domain/'anchor.pt')
    train = Data(domain, True, anchor)
    train_users = {s['user_id'] for s in train.samples}
    assert len(train_users) == train.n and train_users.isdisjoint(ck['fit_users'])
    models, fits = {}, {}
    # All new head/checkpoint choices precede evaluation of the new validation predictions.
    for arm in ARMS:
        for seed in CFG['seeds']:
            model, norm, info = fit(train, arm, seed)
            key = f'{arm}_{seed}'; models[key] = (model, norm, arm, seed); fits[key] = info
    valid = Data(domain, False, anchor)
    primary = {s['user_id'] for s in valid.samples if cohort(s) == 'primary'}
    assert primary.isdisjoint(train_users | set(ck['fit_users']))
    assert len(primary) == dict(Toys=15406, Beauty=17766)[domain]
    rr = {'baseline': np.array(valid.a['baseline_rank'])}; checks = {}
    for key, (model, norm, arm, seed) in models.items():
        rr[key], checks[key] = evaluate(valid, model, norm, arm, seed)
        np.save(OUT/domain/(key+'_ranks.npy'), rr[key])
        np.savez_compressed(OUT/domain/'ranks.npz', **rr)
    event(domain, 'summarize')
    summary = dict(domain=domain, train=train.info, validation=valid.info, anchor_checks=anchor_checks,
                   fits=fits, diagnostics=checks, results=summarize(valid, rr), true_generation=False,
                   interpretation='Exploratory item ranking only. CIs resample users after averaging heads; exclude upstream and study-selection uncertainty.')
    write(OUT/domain/'summary.json', summary)
    loaded = np.load(OUT/domain/'ranks.npz')
    metric_checks = 0
    for name, result in summary['results'].items():
        mask = np.ones(valid.n, bool) if name == 'all' else valid.cohorts == name
        for key, ms in result['metrics'].items():
            r = loaded[key][mask]
            expected_ms = dict(N10=float(utilities(r).mean()), H10=float((r <= 10).mean()), H50=float((r <= 50).mean()))
            assert ms == expected_ms
            metric_checks += 3
    for p in paths:
        assert original_stats[str(p)] == (p.stat().st_size, p.stat().st_mtime_ns), str(p)
    write(OUT/domain/'verification.json', dict(state='VERIFIED_SCREEN', test_read=False, true_generation=False,
          aggregate_metrics_recomputed=metric_checks, numpy_forward_and_rank_checks=sum(len(v['numpy_forward_and_sort_checked']) for v in checks.values()),
          primary_disjoint_from_head_and_anchor_fit=True, input_files_unchanged=True, rows_dropped=0))
    outputs = list((OUT/domain).glob('*.pt')) + list((OUT/domain).glob('*ranks*')) + list((OUT/domain).glob('*signals.npy')) + [OUT/domain/'summary.json', OUT/domain/'verification.json']
    write(OUT/domain/'output_manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs}))
    write(OUT/domain/'status.json', dict(state='COMPLETED', stage='screen_complete', elapsed_seconds=time.monotonic()-begin,
          updated_at=stamp(), test_read=False, true_generation=False, next_step='Review screen before a separate real-generation comparison'))


def prepare():
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json', CFG)
    shutil.copy2(PLAN, OUT/'protocol.md')
    paths = [Path(__file__), PLAN, ROOT/'experiment/phase21/run_dual_candidate_gate.sh',
             ROOT/'experiment/phase21/test_dual_candidate_gate.py',
             ROOT/'experiment/phase21/candidate_gate_core.py', ROOT/'experiment/phase21/candidate_gate_screen.py']
    for p in paths:
        dest = OUT/'sources'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    write(OUT/'manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}))
    write(OUT/'tests.json', tests())
    for domain in DOMAINS:
        (OUT/domain).mkdir()
        write(OUT/domain/'status.json', dict(state='PREPARED', updated_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False, true_generation=False))
    REPORT.write_text('# 双专家候选级权重探索\n\n状态：PREPARED。双专家小头筛选，尚无新效果结论。\n\n协议：`plan/第二十一阶段/21.22_双专家候选级权重探索_20261014.md`。状态与结果：`artifacts/phase21/dual_candidate_gate_20261014_v1`。\n')


def final_report():
    summaries = {d: read(OUT/d/'summary.json') for d in DOMAINS}
    write(OUT/'summary.json', dict(domains=summaries, test_read=False, true_generation=False))
    lines = ['# 双专家候选级权重探索', '',
             '状态：COMPLETED。仅为专家分数的全目录物品排名筛选；未运行真实生成，不能把这些值与512用户的生成指标直接比较。', '',
             'Origin Skill: academic-research-suite，内联执行。SASRec/BGE 与共享融合基线沿用已有 LIME-Rec 派生实现；本轮不读取 ItemCF 通道。', '',
             '| 域 | 双专家基线 | 用户级调整 | 候选级调整 | 直接分数调整 |', '|---|---:|---:|---:|---:|']
    for d, result in summaries.items():
        m = result['results']['primary']['seed_mean_N10']
        lines.append(f'| {d} | {m["baseline"]:.6f} | {m["user"]:.6f} | {m["candidate"]:.6f} | {m["scalar"]:.6f} |')
    lines += ['', '表中为三个小头种子的平均 NDCG@10；每域只有一个冻结上游种子。用户级和候选级调整使用同一候选集合、相同网络参数量、损失、训练用户和预算。', '',
              '| 域 | 比较 | ΔN10 | 描述性95%区间 | 各种子ΔN10 |', '|---|---|---:|---|---|']
    for d, result in summaries.items():
        for name, c in result['results']['primary']['comparisons'].items():
            ci = c['ci95_descriptive']; seeds = ', '.join(f'{v["delta_N10"]:+.6f}' for v in c['seeds'])
            lines.append(f'| {d} | {name} | {c["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {seeds} |')
    lines += ['', 'H10/H50、每种子结果和前十新增/丢失用户均保存在 summary.json。', '',
              '判断边界：候选级高于用户级才支持候选粒度；高于共享基线才支持实际排序增益；若直接分数修正相当或更好，不能把收益唯一归于专家权重。区间跨0时保留不确定性，所有区间均为重复使用的探索池上的描述量，不作新颖性、因果性或最终论文确认。', '',
              '下一步由本轮结果决定是否值得做双专家真实生成。旧三专家真实生成续跑和历史依据分支保持暂停。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    active, finished = {}, {}; begin = time.monotonic()
    try:
        for domain in DOMAINS:
            log = (OUT/domain/'worker.log').open('x')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
            process = subprocess.Popen([sys.executable, '-u', '-m', 'experiment.phase21.dual_candidate_gate', 'worker', '--domain', domain],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[domain] = dict(process=process, log=log, started=time.monotonic())
        while active:
            for domain, job in list(active.items()):
                code = job['process'].poll()
                if code is None and time.monotonic()-job['started'] > CFG['timeout_seconds']:
                    job['process'].terminate()
                    try:
                        code = job['process'].wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        job['process'].kill(); code = job['process'].wait()
                    old = read(OUT/domain/'status.json')
                    write(OUT/domain/'status.json', dict(state='TIME_LIMIT', previous=old, updated_at=stamp()))
                if code is not None:
                    st = read(OUT/domain/'status.json')
                    if st['state'] not in ('COMPLETED', 'FAILED', 'TIME_LIMIT'):
                        write(OUT/domain/'status.json', dict(state='CRASHED', exit_code=code, previous=st, updated_at=stamp()))
                        st = read(OUT/domain/'status.json')
                    job['log'].close(); finished[domain] = dict(exit_code=code, state=st['state'])
                    del active[domain]
            write(OUT/'status.json', dict(state='RUNNING' if active else 'FINALIZING', pid=os.getpid(),
                  updated_at=stamp(), elapsed_seconds=time.monotonic()-begin, resource='CPU only; two workers, two threads each',
                  active={d: dict(pid=j['process'].pid, worker=read(OUT/d/'status.json')) for d, j in active.items()},
                  finished=finished, test_read=False, true_generation=False, timeout_seconds=CFG['timeout_seconds']))
            if active:
                time.sleep(15)
        complete = len(finished) == 2 and all(v == dict(exit_code=0, state='COMPLETED') for v in finished.values())
        if complete:
            final_report()
        write(OUT/'status.json', dict(state='COMPLETED' if complete else 'INCOMPLETE', finished=finished,
              updated_at=stamp(), elapsed_seconds=time.monotonic()-begin, test_read=False, true_generation=False,
              report=str(REPORT.relative_to(ROOT)), automatic_generation_scheduled=False))
    finally:
        for job in active.values():
            if job['process'].poll() is None:
                job['process'].terminate()
            job['log'].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'supervise', 'worker', 'test'])
    parser.add_argument('--domain', choices=DOMAINS)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError('Received termination; partial outputs retained')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            torch.set_num_threads(2)
            print(json.dumps(tests()))
        elif args.mode == 'prepare':
            prepare()
        elif args.mode == 'worker':
            assert args.domain
            worker(args.domain)
        else:
            supervise()
    except BaseException as exc:
        folder = OUT/args.domain if args.mode == 'worker' else OUT
        if folder.exists():
            previous = read(folder/'status.json') if (folder/'status.json').exists() else None
            write(folder/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                  previous=previous, updated_at=stamp(), test_read=False))
        raise


if __name__ == '__main__':
    main()
