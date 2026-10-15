"""Independent raw-snapshot reconstruction and saved-head/rank replay."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from experiment.phase20.review_lime_clean_benefit import arrays, sha, read, rr, metrics
from experiment.phase20.guided_decoding_core import PrefixIndex

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
EARLY = ROOT / 'artifacts/phase20/lime_early_correction_20260922_v1'
OUT = ROOT / 'artifacts/phase20/lime_partial_state_20260922_v1'
NEW = tuple(f'{v}_seed{s}' for v in ('prefix2', 'prefix3', 'state2', 'state3') for s in range(3))
OLD = tuple(f'{v}_seed{s}' for v in ('full', 'early') for s in range(3))
ALL = ('GRAM', 'fixed_late') + OLD + NEW
COLUMNS = [0, 1, 6, 7]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def subset(d, ix):
    return {k: v[ix] for k, v in d.items()}


def restore(x):
    if x is None: return -math.inf
    if isinstance(x, list): return [restore(v) for v in x]
    if isinstance(x, dict): return {k: restore(v) for k, v in x.items()}
    return x


def independent_features(snapshots, masses, index, depth):
    # Read only observations no later than the requested decision.
    past = {}
    for step in snapshots[:depth]:
        for prefix, score, row in zip(step['prefixes'], step['parent_scores'], step['rows']):
            if score > -1e8:
                past[tuple(prefix)] = row

    def q(prefix, token):
        node = index.lookup[prefix]; child = index.children[node][token]
        return float(np.log(masses[child] / masses[node]))

    def cumulative(prefix):
        result = 0.
        for length in range(1, len(prefix)):
            parent = prefix[:length]; token = prefix[length]; row = past[parent]
            native = row['native'][row['tokens'].index(token)]
            result += float(np.logaddexp(math.log(.7) + native,
                            math.log(.3) + row['legal_log_mass'] + q(parent, token)))
        return result

    candidates = {}
    step = snapshots[depth - 1]
    for raw_prefix, parent, row in zip(step['prefixes'], step['parent_scores'], step['rows']):
        if parent <= -1e8: continue
        prefix = tuple(raw_prefix); mix_parent = cumulative(prefix)
        child_nodes = np.array([index.children[index.lookup[prefix]][t] for t in row['tokens']])
        expert = np.log(masses[child_nodes] / masses[index.lookup[prefix]])
        native = np.array(row['native'])
        mixed = mix_parent + np.logaddexp(math.log(.7) + native, math.log(.3) + row['legal_log_mass'] + expert)
        total = (np.float32(parent) + native.astype('float32')).astype(float)
        for token, n, m in zip(row['tokens'], total, mixed):
            candidates[prefix + (token,)] = (float(n), float(m))
    ordered = sorted(candidates.items(), key=lambda z: (-z[1][0], z[0]))
    mixed_sorted = sorted(v[1] for v in candidates.values())
    boundary = mixed_sorted[-min(50, len(mixed_sorted))]
    return {key: np.array([n/depth, (m-n)/depth, 1/(rank+1), (m-boundary)/depth], dtype='float32')
            for rank, (key, (n, m)) in enumerate(ordered)}


def feature_audit(domain):
    started = time.monotonic(); folder = OUT / domain
    assert read(folder / 'status.json')['state'] == 'COMPLETED'
    for name, digest in read(folder / 'cache_manifest.json')['sha256'].items():
        assert sha(folder / name) == digest, name
    cached = arrays(folder / 'features.npz'); old = arrays(SOURCE / domain / 'data.npz')
    for key in ('users', 'ids', 'sample_indices'):
        np.testing.assert_array_equal(cached[key], old[key])
    catalog = read(folder / 'catalog.json'); paths = catalog['decoder_paths']
    index = PrefixIndex(paths, range(1, len(paths)+1))
    lime = np.load(SOURCE / domain / 'lime.npy', mmap_mode='r')
    samples = read(folder / 'samples.json')
    checked = 0; steps_checked = 0; max_error = 0.; stats = {'2': [], '3': []}
    with gzip.open(folder / 'traces.jsonl.gz', 'rt') as f:
        for row_number, line in enumerate(f):
            rec = restore(json.loads(line)); j = int(old['sample_indices'][row_number])
            assert rec['row'] == row_number and rec['sample_index'] == j
            assert rec['user_id'] == old['users'][row_number] == samples[j]['user_id']
            steps = rec['steps']; assert len(steps) == 3
            for n, step in enumerate(steps):
                assert step['depth'] == n+1 and not step['finished']
                for row in step['rows']:
                    np.testing.assert_array_equal(row['native'], row['used'])
                if n:
                    prev = steps[n-1]; selected = prev['selected']
                    expected = [prev['prefixes'][p] + [t] for p, t in zip(selected['next_beam_indices'], selected['next_beam_tokens'])]
                    assert expected == step['prefixes']
                    assert selected['next_beam_scores'] == step['parent_scores']
                selected = step['selected']
                for parent, token, total in zip(selected['next_beam_indices'], selected['next_beam_tokens'], selected['next_beam_scores']):
                    if math.isfinite(total):
                        r = step['rows'][parent]; local = r['native'][r['tokens'].index(token)]
                        assert total == float(np.float32(step['parent_scores'][parent]) + np.float32(local))
                steps_checked += 1
            values = lime[j].astype('float64'); weights = np.exp(20 * (values-values.max()))
            masses = np.bincount(index.node_items, weights=weights[index.item_refs-1], minlength=len(index.nodes))
            assert np.all(masses > 0)
            for depth in (2, 3):
                mapped = independent_features(steps, masses, index, depth)
                keys = [tuple(paths[int(item)-1][:depth+1]) for item in old['ids'][row_number]]
                actual = np.stack([mapped[key] for key in keys]); expected = cached[f'prefix{depth}'][row_number]
                max_error = max(max_error, float(abs(actual-expected).max()))
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                assert rec['counts'][str(depth)] == dict(all_expansions=len(mapped), selected_distinct_prefixes=len(set(keys)))
                stats[str(depth)].append([len(mapped), len(set(keys))]); checked += actual.size
            if (row_number+1) % 500 == 0:
                print(json.dumps(dict(stage='feature_review', domain=domain, users=row_number+1)), flush=True)
    assert row_number+1 == len(old['users'])
    summary = read(folder / 'cache_summary.json'); assert summary['model_unchanged']
    assert summary['model_digest'] == read(SOURCE / domain / 'cache_summary.json')['model_digest']
    smoke = read(folder / 'smoke.json')['checks']; assert len(smoke) == 3
    assert all(v['partial_steps_exact'] and v['final_cache_exact'] for v in smoke)
    result = dict(state='VERIFIED', users=len(old['users']), checked_feature_values=checked,
        checked_beam_steps=steps_checked, max_feature_error=max_error, seconds=time.monotonic()-started,
        expansion_means={k: np.mean(v, axis=0).tolist() for k, v in stats.items()},
        code_sha256=sha(Path(__file__)), target_free_features=True, all_active_expansions=True)
    write(folder / 'feature_verification.json', result); print(json.dumps(result), flush=True)


def replay(c, x):
    z = np.zeros(x.shape, dtype='float32'); cols = list(c['columns'])
    z[..., cols] = (x[..., cols]-c['mean'])/c['std']; w = c['state_dict']
    with torch.no_grad():
        h = torch.tanh(torch.nn.functional.linear(torch.from_numpy(z), w['network.0.weight'], w['network.0.bias']))
        out = torch.nn.functional.linear(h, w['network.2.weight'], w['network.2.bias'])
        return (2*torch.tanh(out.squeeze(-1))).numpy()


def utility(r):
    return np.array([1/math.log2(int(x)+1) if x <= 10 else 0. for x in r])


def fit_audit(domain):
    folder = OUT / domain; data = arrays(SOURCE / domain / 'data.npz'); cached = arrays(folder / 'features.npz')
    tr = np.flatnonzero(data['roles'] == 'ranker_fit'); dv = np.flatnonzero(data['roles'] == 'development')
    train, dev = subset(data, tr), subset(data, dv)
    assert not set(train['users']) & set(dev['users'])
    folds = np.array([int.from_bytes(hashlib.sha256(f'20260922|clean-benefit-fold|{domain}|{u}'.encode()).digest()[:8], 'big')%5 for u in train['users']])
    pred = {label: arrays(folder/f'{label}_predictions.npz') for label in ('oof', 'development')}
    delta = {label: arrays(folder/f'{label}_corrections.npz') for label in ('oof', 'development')}
    np.testing.assert_array_equal(folds, pred['oof']['folds'])
    for label, src in [('oof', train), ('development', dev)]:
        old = arrays(EARLY/domain/f'{label}_predictions.npz')
        np.testing.assert_array_equal(pred[label]['users'], src['users'])
        np.testing.assert_array_equal(delta[label]['users'], src['users'])
        for name in ('GRAM', 'fixed_late') + OLD:
            np.testing.assert_array_equal(pred[label][name], old[name])
    count = 0; error = 0.
    for variant in ('prefix2', 'prefix3', 'state2', 'state3'):
        depth = int(variant[-1]); cols = list(range(10)) if variant.startswith('prefix') else COLUMNS
        x = data['x'].copy(); x[..., COLUMNS] = cached[f'prefix{depth}']
        for f in list(range(5)) + ['final']:
            tag = f'fold{f}' if f != 'final' else 'final'; label = 'oof' if f != 'final' else 'development'
            src = train if f != 'final' else dev; source_rows = tr if f != 'final' else dv
            fit_ix = np.flatnonzero(folds != f) if f != 'final' else np.arange(len(tr))
            ix = np.flatnonzero(folds == f) if f != 'final' else np.arange(len(dv))
            covered = np.array([i for i in fit_ix if np.any((train['ids'][i] == train['targets'][i]) & train['valid'][i])])
            for seed in range(3):
                name = f'{variant}_seed{seed}'; c = torch.load(folder/'heads'/f'{tag}.{name}.pt', map_location='cpu')
                assert c['variant'] == variant and c['depth'] == depth and c['seed'] == seed
                assert list(c['columns']) == cols and c['optimizer_steps'] == len(c['curve']) == 30
                np.testing.assert_array_equal(c['fit_indices'], fit_ix); np.testing.assert_array_equal(c['used'], covered)
                np.testing.assert_array_equal(c['fit_users'], train['users'][fit_ix])
                np.testing.assert_array_equal(c['used_users'], train['users'][covered])
                assert not set(c['fit_users']) & set(src['users'][ix])
                raw = x[tr[covered]][..., cols][train['valid'][covered]]
                np.testing.assert_array_equal(c['mean'], raw.mean(0, dtype=float).astype('float32'))
                np.testing.assert_array_equal(c['std'], np.maximum(raw.std(0, dtype=float), 1e-5).astype('float32'))
                rebuilt = replay(c, x[source_rows[ix]])
                error = max(error, float(abs(rebuilt-delta[label][name][ix]).max()))
                np.testing.assert_allclose(rebuilt, delta[label][name][ix], atol=1e-6, rtol=1e-6)
                score = (10*src['mixed'][ix]).astype('float32') + rebuilt
                np.testing.assert_allclose(score, pred[label][name][ix], atol=2e-6, rtol=1e-6)
                np.testing.assert_array_equal(rr(subset(src, ix), score), rr(subset(src, ix), pred[label][name][ix])); count += 1
    ranks = arrays(folder/'ranks.npz'); report = read(folder/'analysis.json'); rank_count = 0
    for label, src in [('oof', train), ('development', dev)]:
        boot = np.random.RandomState(20260922).randint(0, len(src['users']), (1000, len(src['users'])))
        for name in ALL:
            r = rr(src, pred[label][name]); np.testing.assert_array_equal(r, ranks[f'{label}_{name}'])
            for k, v in metrics(r).items(): assert abs(v-report[label]['metrics'][name][k]) < 1e-12
            if label == 'oof':
                for f in range(5):
                    for k, v in metrics(r[folds == f]).items(): assert abs(v-report['folds'][str(f)][name][k]) < 1e-12
            for ref, c in report[label]['contrasts'].get(name, {}).items():
                other = ranks[f'{label}_{ref}']; dn = utility(r)-utility(other)
                dh = (r <= 10).astype(float)-(other <= 10).astype(float)
                assert abs(dn.mean()-c['delta_N10']) < 1e-12 and abs(dh.mean()-c['delta_H10']) < 1e-12
                assert c['top10_gained'] == int(((r <= 10) & (other > 10)).sum())
                assert c['top10_lost'] == int(((r > 10) & (other <= 10)).sum())
                for values, key in [(dn, 'N10'), (dh, 'H10')]:
                    np.testing.assert_allclose(np.percentile(values[boot].mean(1), [2.5, 97.5]), c[f'descriptive_95pct_{key}'], atol=1e-12, rtol=1e-12)
            rank_count += len(r)
    for name in NEW:
        dm = {a: metrics(ranks['development_'+a]) for a in (name, 'GRAM', 'fixed_late')}
        om = {a: metrics(ranks['oof_'+a]) for a in (name, 'GRAM', 'fixed_late')}
        wins = sum(all(metrics(ranks['oof_'+name][folds == f])['N10'] > metrics(ranks['oof_'+a][folds == f])['N10'] for a in ('GRAM', 'fixed_late')) for f in range(5))
        dp = all(dm[name]['N10'] >= 1.01*dm[a]['N10'] and dm[name]['H10'] >= dm[a]['H10'] for a in ('GRAM', 'fixed_late'))
        op = wins >= 4 and all(om[name]['N10'] > om[a]['N10'] for a in ('GRAM', 'fixed_late'))
        assert report['checks'][name] == dict(development_pass=dp, oof_pass=op, positive_folds_vs_both=wins, passed=dp and op)
    return dict(state='VERIFIED', heads=count, checked_ranks=rank_count, max_correction_error=error,
                previous_predictions_exact=True, user_isolation=True, independent_sorting=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('features', 'fits')); p.add_argument('--domain', choices=('Toys', 'Beauty'))
    args = p.parse_args(); torch.set_num_threads(2)
    if args.action == 'features':
        assert args.domain; feature_audit(args.domain); return
    assert read(OUT/'status.json')['state'] == 'ANALYZED'; started = time.monotonic()
    for name in ('manifest.json', 'analysis_manifest.json'):
        for rel, digest in read(OUT/name)['sha256'].items(): assert sha(ROOT/rel) == digest, rel
    result = {d: fit_audit(d) for d in ('Toys', 'Beauty')}
    summary = read(OUT/'summary.json')
    expected = {name: all(read(OUT/d/'analysis.json')['checks'][name]['passed'] for d in result) for name in NEW}
    for depth in (2, 3): expected[f'prefix{depth}_all_seeds'] = all(expected[f'prefix{depth}_seed{s}'] for s in range(3))
    assert summary['decision'] == expected
    receipt = dict(state='VERIFIED', domains=result, seconds=time.monotonic()-started,
        reviewer_sha256=sha(Path(__file__)), note='Independent numerical code in the same assistant session; not an external reviewer.')
    write(OUT/'verification.json', receipt)
    summary.update(state='COMPLETED', pending_verification=False, verification=receipt); write(OUT/'summary.json', summary)
    status = read(OUT/'status.json'); status.update(state='COMPLETED', verified=True, pending_verification=False,
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')); write(OUT/'status.json', status)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
