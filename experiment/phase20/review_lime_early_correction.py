"""Standalone numerical replay; no import of this experiment's fitting code."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from experiment.phase20.review_lime_clean_benefit import arrays, sha, read, rr, metrics

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT / 'artifacts/phase20/lime_early_correction_20260922_v1'
COLS = {'full': tuple(range(10)), 'early': (2, 3, 4, 5, 8, 9), 'late': (0, 1, 6, 7)}
NAMES = tuple(f'{g}_seed{s}' for g in COLS for s in range(3))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def subset(d, ix):
    return {k: v[ix] for k, v in d.items()}


def replay(saved, selected):
    z = np.zeros((*selected.shape[:-1], 10), dtype='float32')
    z[..., list(saved['columns'])] = (selected - saved['mean']) / saved['std']
    weights = saved['state_dict']
    with torch.no_grad():
        h = torch.tanh(torch.nn.functional.linear(torch.from_numpy(z), weights['network.0.weight'], weights['network.0.bias']))
        v = torch.nn.functional.linear(h, weights['network.2.weight'], weights['network.2.bias'])
        return (2 * torch.tanh(v.squeeze(-1))).numpy()


def utility(r):
    return np.array([1 / math.log2(int(x) + 1) if x <= 10 else 0. for x in r])


def audit(domain):
    folder = OUT / domain
    d = arrays(SOURCE / domain / 'data.npz')
    samples = read(SOURCE / domain / 'samples.json')
    train = subset(d, np.flatnonzero(d['roles'] == 'ranker_fit'))
    dev = subset(d, np.flatnonzero(d['roles'] == 'development'))
    folds = np.array([int.from_bytes(hashlib.sha256(
        f'20260922|clean-benefit-fold|{domain}|{u}'.encode()).digest()[:8], 'big') % 5 for u in train['users']])
    gate = torch.load(SOURCE / domain / 'gate.pt', map_location='cpu')
    assert not set(gate['fit_users']) & set(d['users'])
    assert not set(train['users']) & set(dev['users'])
    pred = {label: arrays(folder / f'{label}_predictions.npz') for label in ('oof', 'development')}
    delta = {label: arrays(folder / f'{label}_corrections.npz') for label in ('oof', 'development')}
    np.testing.assert_array_equal(folds, pred['oof']['folds'])
    for label, src in [('oof', train), ('development', dev)]:
        np.testing.assert_array_equal(pred[label]['users'], src['users'])
        np.testing.assert_array_equal(delta[label]['users'], src['users'])
    # Rebuild early inputs from catalog-wide pre-generation caches, no late scores.
    experts = np.load(SOURCE / domain / 'experts.npy', mmap_mode='r')
    lime = np.load(SOURCE / domain / 'lime.npy', mmap_mode='r')
    counts = np.load(SOURCE / domain / 'frequencies.npy')
    for row, sample_ix in enumerate(d['sample_indices']):
        sample = samples[int(sample_ix)]
        items = d['ids'][row]
        assert sample['user_id'] == d['users'][row]
        early = np.column_stack((experts[sample_ix][:, items - 1].T, lime[sample_ix, items - 1],
                                 np.log1p(counts[items]), np.full(50, np.log1p(len(sample['full_history']))))).astype('float32')
        np.testing.assert_array_equal(early, d['x'][row][..., list(COLS['early'])])
    heads = 0; maximum_error = 0.; exact_old_heads = 0
    for f in list(range(5)) + ['final']:
        tag = f'fold{f}' if f != 'final' else 'final'
        label = 'oof' if f != 'final' else 'development'
        src = train if f != 'final' else dev
        fit_ix = np.flatnonzero(folds != f) if f != 'final' else np.arange(len(train['users']))
        ix = np.flatnonzero(folds == f) if f != 'final' else np.arange(len(dev['users']))
        used = np.array([i for i in fit_ix if np.any((train['ids'][i] == train['targets'][i]) & train['valid'][i])])
        for name in NAMES:
            c = torch.load(folder / 'heads' / f'{tag}.{name}.pt', map_location='cpu')
            group, seed = name.split('_seed'); columns = COLS[group]
            assert c['group'] == group and c['seed'] == int(seed) and tuple(c['columns']) == columns
            np.testing.assert_array_equal(c['fit_indices'], fit_ix)
            np.testing.assert_array_equal(c['used'], used)
            np.testing.assert_array_equal(c['fit_users'], train['users'][fit_ix])
            np.testing.assert_array_equal(c['used_users'], train['users'][used])
            assert not set(c['fit_users']) & set(src['users'][ix])
            raw = train['x'][used][..., list(columns)][train['valid'][used]]
            np.testing.assert_array_equal(c['mean'], raw.mean(0, dtype=float).astype('float32'))
            np.testing.assert_array_equal(c['std'], np.maximum(raw.std(0, dtype=float), 1e-5).astype('float32'))
            assert c['optimizer_steps'] == len(c['curve']) == 30
            actual = replay(c, src['x'][ix][..., list(columns)])
            maximum_error = max(maximum_error, float(np.max(abs(actual - delta[label][name][ix]))))
            np.testing.assert_allclose(actual, delta[label][name][ix], atol=1e-6, rtol=1e-6)
            score = (10 * src['mixed'][ix]).astype('float32') + actual
            np.testing.assert_allclose(score, pred[label][name][ix], atol=2e-6, rtol=1e-6)
            np.testing.assert_array_equal(rr(subset(src, ix), score), rr(subset(src, ix), pred[label][name][ix]))
            np.testing.assert_array_equal((10 * src['mixed'][ix]).astype('float32') + delta[label][name][ix], pred[label][name][ix])
            if group == 'full':
                old = torch.load(SOURCE / domain / 'heads' / f'{tag}.mlp_seed{seed}.pt', map_location='cpu')
                for key in c['state_dict']:
                    assert torch.equal(c['state_dict'][key], old['state_dict'][key])
                exact_old_heads += 1
            if group == 'early':
                selected = src['x'][ix[:3]][..., list(columns)]
                reference = replay(c, selected)
                np.testing.assert_allclose(replay(c, selected[:, ::-1]), reference[:, ::-1], atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(replay(c, selected[:, :1]), reference[:, :1], atol=1e-6, rtol=1e-6)
            heads += 1
    # A final early model can score the entire catalog before selecting candidates.
    catalog_examples = 0
    for seed in range(3):
        c = torch.load(folder / 'heads' / f'final.early_seed{seed}.pt', map_location='cpu')
        for j in range(3):
            sample_ix = dev['sample_indices'][j]; sample = samples[int(sample_ix)]
            early = np.column_stack((experts[sample_ix].T, lime[sample_ix], np.log1p(counts[1:]),
                                     np.full(experts.shape[2], np.log1p(len(sample['full_history']))))).astype('float32')
            all_corrections = replay(c, early[None])[0]
            np.testing.assert_allclose(all_corrections[dev['ids'][j] - 1], delta['development'][f'early_seed{seed}'][j], atol=1e-6, rtol=1e-6)
            catalog_examples += 1
    ranks = arrays(folder / 'ranks.npz'); report = read(folder / 'analysis.json')
    checked = 0
    for label, src in [('oof', train), ('development', dev)]:
        bootstrap = np.random.RandomState(20260922).randint(0, len(src['users']), (1000, len(src['users'])))
        for name in ('GRAM', 'fixed_late') + NAMES:
            r = rr(src, pred[label][name]); np.testing.assert_array_equal(r, ranks[f'{label}_{name}'])
            for k, value in metrics(r).items():
                assert abs(value - report[label]['metrics'][name][k]) < 1e-12
            if label == 'oof':
                for f in range(5):
                    for k, value in metrics(r[folds == f]).items():
                        assert abs(value - report['folds'][str(f)][name][k]) < 1e-12
            for ref, comparison in report[label]['contrasts'].get(name, {}).items():
                ref_r = ranks[f'{label}_{ref}']; dn = utility(r) - utility(ref_r)
                dh = (r <= 10).astype(float) - (ref_r <= 10).astype(float)
                assert abs(dn.mean() - comparison['delta_N10']) < 1e-12
                assert abs(dh.mean() - comparison['delta_H10']) < 1e-12
                assert comparison['top10_gained'] == int(((r <= 10) & (ref_r > 10)).sum())
                assert comparison['top10_lost'] == int(((r > 10) & (ref_r <= 10)).sum())
                for values, key in [(dn, 'N10'), (dh, 'H10')]:
                    ci = np.percentile(values[bootstrap].mean(1), [2.5, 97.5])
                    np.testing.assert_allclose(ci, comparison[f'descriptive_95pct_{key}'], atol=1e-12, rtol=1e-12)
            checked += len(r)
        if label == 'oof':
            assert report[label]['covered'] == int((ranks['oof_GRAM'] < 2**30).sum())
    for name in NAMES:
        wins = sum(all(metrics(ranks[f'oof_{name}'][folds == f])['N10'] > metrics(ranks[f'oof_{ref}'][folds == f])['N10']
                       for ref in ('GRAM', 'fixed_late')) for f in range(5))
        dm = {a: metrics(ranks['development_' + a]) for a in (name, 'GRAM', 'fixed_late')}
        om = {a: metrics(ranks['oof_' + a]) for a in (name, 'GRAM', 'fixed_late')}
        dp = all(dm[name]['N10'] >= 1.01 * dm[a]['N10'] and dm[name]['H10'] >= dm[a]['H10'] for a in ('GRAM', 'fixed_late'))
        op = wins >= 4 and all(om[name]['N10'] > om[a]['N10'] for a in ('GRAM', 'fixed_late'))
        assert report['checks'][name] == dict(development_pass=dp, oof_pass=op, positive_folds_vs_both=wins, passed=dp and op)
    return dict(state='VERIFIED', heads=heads, exact_previous_heads=exact_old_heads,
                users=len(d['users']), checked_ranks=checked, max_correction_error=maximum_error,
                catalog_wide_examples=catalog_examples, early_features_rebuilt=True,
                pointwise_invariance=True, fit_users_disjoint=True, independent_sorting=True)


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    assert read(OUT / 'status.json')['state'] == 'ANALYZED'
    manifest = read(OUT / 'manifest.json')['sha256']
    for rel, digest in manifest.items():
        assert sha(ROOT / rel) == digest, rel
    results = {domain: audit(domain) for domain in ('Toys', 'Beauty')}
    summary = read(OUT / 'summary.json')
    expected = {name: all(read(OUT / d / 'analysis.json')['checks'][name]['passed'] for d in results) for name in NAMES}
    expected['early_all_three_seeds'] = all(expected[f'early_seed{s}'] for s in range(3))
    assert summary['decision'] == expected
    result = dict(state='VERIFIED', domains=results, input_hashes=len(manifest),
                  seconds=time.monotonic() - started,
                  note='Independent numerical implementation in the same assistant session; no external reviewer.')
    write(OUT / 'verification.json', result)
    summary.update(state='COMPLETED', pending_verification=False, verification=result)
    write(OUT / 'summary.json', summary)
    s = read(OUT / 'status.json'); s.update(state='COMPLETED', verified=True, pending_verification=False,
                                         updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write(OUT / 'status.json', s)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
