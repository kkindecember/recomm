"""Replay saved predictions with an independent sort and check fold boundaries."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/relation_direction_20260922_v1'
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'


def read(path):
    return json.loads(path.read_text())


def independent_rank(values, ids, valid, targets):
    result = np.full(len(ids), 2**30, dtype=np.int64)
    for i in range(len(ids)):
        legal = [(float(s), int(item)) for s, item, ok in zip(values[i], ids[i], valid[i]) if ok]
        ordered = [item for score, item in sorted(legal, key=lambda p: (-p[0], p[1]))]
        if int(targets[i]) in ordered:
            result[i] = ordered.index(int(targets[i])) + 1
    return result


def metric(rr):
    return {'N10': float(np.mean([1 / np.log2(r + 1) if r <= 10 else 0 for r in rr])),
            'H10': float(np.mean(rr <= 10))}


def main():
    summary = read(OUT / 'summary.json')
    assert summary['state'] == 'COMPLETED'
    verification = {}
    for domain, result in summary['domains'].items():
        folder = OUT / domain
        data = dict(np.load(SOURCE / domain / 'data.npz'))
        features = dict(np.load(folder / 'relation_features.npz'))
        predictions = dict(np.load(folder / 'predictions.npz'))
        saved_ranks = dict(np.load(folder / 'ranks.npz'))
        alignment = dict(np.load(folder / 'alignment.npz'))
        np.testing.assert_array_equal(alignment['users'], data['users'])
        folds = np.array([int.from_bytes(hashlib.sha256(
            f'20260922|clean-benefit-fold|{domain}|{u}'.encode()).digest()[:8], 'big') % 5 for u in data['users']])
        np.testing.assert_array_equal(folds, alignment['folds'])
        prior = 10 * data['mixed']
        checks = 0; max_error = 0.
        for arm, values in predictions.items():
            x = data['x'].astype(np.float64)
            if arm != 'base':
                x = np.concatenate((x, features[arm][:, :, None]), axis=2)
            for fold in range(6):
                head = dict(np.load(folder / f'head_{arm}_{fold}.npz'))
                train = np.flatnonzero(data['roles'] == 'ranker_fit')
                expected_fit = train if fold == 5 else train[folds[train] != fold]
                evaluation = np.flatnonzero(data['roles'] == 'development') if fold == 5 else train[folds[train] == fold]
                np.testing.assert_array_equal(head['fit_indices'], expected_fit)
                assert not set(expected_fit) & set(evaluation)
                used = [i for i in expected_fit if data['targets'][i] in data['ids'][i][data['valid'][i]]]
                np.testing.assert_array_equal(head['used'], used)
                flat = x[used][data['valid'][used]]
                np.testing.assert_allclose(head['mean'], flat.mean(0), atol=1e-12)
                np.testing.assert_allclose(head['std'], np.maximum(flat.std(0), 1e-5), atol=1e-12)
                z = (x[evaluation] - head['mean']) / head['std']
                replay = prior[evaluation] + np.sum(z * head['coef'], axis=2)
                error = float(np.max(np.abs(replay - values[evaluation])))
                max_error = max(max_error, error)
                np.testing.assert_allclose(replay, values[evaluation], atol=1e-10, rtol=1e-10)
                checks += 1
            rr = independent_rank(values, data['ids'], data['valid'], data['targets'])
            np.testing.assert_array_equal(rr, saved_ranks[arm])
        for arm, values in [('gram', data['native']), ('fixed_late', prior)]:
            np.testing.assert_array_equal(independent_rank(values, data['ids'], data['valid'], data['targets']), saved_ranks[arm])
        for role, selector in [('oof', data['roles'] == 'ranker_fit'), ('development', data['roles'] == 'development')]:
            for arm, rr in saved_ranks.items():
                for key, value in metric(rr[selector]).items():
                    assert abs(value - result['groups'][role]['ranking'][arm][key]) < 1e-12
        tops = dict(np.load(folder / 'catalogue_top50.npz'))
        cat_ranks = dict(np.load(folder / 'catalogue_ranks.npz'))
        samples = read(SOURCE / domain / 'samples.json')
        with (SOURCE / domain / 'candidates.jsonl').open() as stream:
            candidates = [json.loads(line) for line in stream if line.strip()]
        for arm, top in tops.items():
            np.testing.assert_array_equal(np.any(top == data['targets'][:, None], axis=1), cat_ranks[arm] <= 50)
            for i, sample_index in enumerate(data['sample_indices']):
                assert not set(top[i][top[i] > 0]) & set(samples[sample_index]['history'])
                assert len(set(top[i][top[i] > 0])) == int((top[i] > 0).sum())
            for role, label in [('oof', 'ranker_fit'), ('development', 'development')]:
                new_native = 0; new_all = 0; union25 = 0; hits = 0
                for i in np.flatnonzero(data['roles'] == label):
                    history = set(samples[data['sample_indices'][i]]['history'])
                    native = set(data['ids'][i][data['valid'][i]])
                    existing = set(native)
                    for baseline in ('sasrec', 'itemcf', 'semantic', 'lime'):
                        existing.update(tops[baseline][i])
                    existing.update(item for item in candidates[i].get('guided', {}).get('ids', []) if item not in history)
                    y = data['targets'][i]; hit = y in top[i]
                    hits += int(hit); new_native += int(hit and y not in native)
                    new_all += int(hit and y not in existing)
                    ordered = sorted(zip(data['native'][i], data['ids'][i]), key=lambda p: (-p[0], p[1]))
                    native25 = [item for score, item in ordered if item not in history][:25]
                    union25 += int(y in set(native25).union(top[i, :25]))
                expected = result['groups'][role]['coverage'][arm]
                assert (hits, new_native, new_all, union25) == (expected['top50_hits'], expected['new_vs_gram50'],
                       expected['new_vs_all_existing'], expected['gram25_plus_source25_hits'])
        controls = ('base', 'symmetric_recent', 'reverse_recent')
        d = result['groups']['development']['ranking']; o = result['groups']['oof']['ranking']
        wins = 0
        for fold in range(5):
            ix = (data['roles'] == 'ranker_fit') & (folds == fold)
            t = {arm: metric(rr[ix]) for arm, rr in saved_ranks.items()}
            wins += all(t['forward_recent']['N10'] > t[c]['N10'] for c in controls)
        gates = dict(development_relative_N10=all(d['forward_recent']['N10'] >= 1.01*d[c]['N10'] for c in controls),
                     development_H10=d['forward_recent']['H10'] >= d['base']['H10'],
                     oof_N10=all(o['forward_recent']['N10'] > o[c]['N10'] for c in controls),
                     fold_consistency=wins >= 4,
                     new_opportunities=result['groups']['development']['coverage']['forward_recent']['new_vs_all_existing'] >= 3)
        assert gates == result['gates'] and all(gates.values()) == result['passes']
        verification[domain] = dict(heads_replayed=checks, ranked_rows=len(data['users'])*len(saved_ranks),
                                    max_prediction_error=max_error, fold_boundary_verified=True,
                                    all_ranks_exact=True, all_gates_exact=True, catalogue_lists_legal=True,
                                    coverage_counts_rebuilt=True)
    assert summary['passes'] == all(s['passes'] for s in summary['domains'].values())
    destination = OUT / 'independent_verification.json'
    assert not destination.exists()
    destination.write_text(json.dumps(dict(state='VERIFIED', domains=verification,
        note='Same assistant, separate numerical replay; not an external reviewer.'), indent=2)+'\n')
    print(destination.read_text())


if __name__ == '__main__':
    main()
