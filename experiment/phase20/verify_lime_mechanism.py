"""Independent item ranking, selection, cohort and input-integrity audit.

Does not import the runner, scoring policy, or its metric/selection functions.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/lime_mechanism_screen_20260921_v1'
PARENT = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1/full_parallel_v4'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def calculate(ranks):
    ranks = np.asarray(ranks)
    return {f'{name}{k}': float(np.mean(ranks <= k) if name == 'H'
                              else np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
            for k in (5, 10, 20, 50) for name in ('N', 'H')}


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = np.where(a <= 10, 1 / np.log2(a + 1), 0) - np.where(b <= 10, 1 / np.log2(b + 1), 0)
    base = calculate(b)['N10']
    bootstrap = delta[np.random.RandomState(20260921).randint(len(a), size=(2000, len(a)))].mean(axis=1)
    return dict(delta_N10=float(delta.mean()), relative_N10=float(delta.mean() / base) if base else None,
                descriptive_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
                gained_H10=int(((a <= 10) & (b > 10)).sum()), lost_H10=int(((a > 10) & (b <= 10)).sum()),
                net_H10=int((a <= 10).sum() - (b <= 10).sum()),
                gained_H50=int(((a <= 50) & (b > 50)).sum()), lost_H50=int(((a > 50) & (b <= 50)).sum()))


def main():
    frozen = read(OUT / 'manifest.json')['sha256']
    for path, expected in frozen.items():
        assert digest(ROOT / path) == expected, ('Changed input', path)
    verified = {}
    for domain in ('Toys', 'Beauty'):
        folder = OUT / domain
        samples = read(folder / 'samples.json')
        previous_samples = read(PARENT / domain / 'samples.json')
        assert samples[:131] == previous_samples[:131]
        assert len(samples) == 643 and len({r['user_id'] for r in samples}) == 643
        assert {r['user_id'] for r in samples[131:]}.isdisjoint(r['user_id'] for r in previous_samples)
        recovery = ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        with np.load(recovery / 'evaluation_seed0_cal2048.npz', allow_pickle=False) as old:
            users = old['users'].copy()
        excluded = {r['user_id'] for r in previous_samples}
        expected_users = sorted([str(u) for u in users if str(u) not in excluded],
            key=lambda u: hashlib.sha256(f'20260921|lime-mechanism-v1|{domain}|{u}'.encode()).digest())[:512]
        assert [r['user_id'] for r in samples[131:]] == expected_users
        with np.load(ROOT / f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz', allow_pickle=False) as q:
            assert {r['query_index'] for r in samples}.isdisjoint(q['calibration_indices'].tolist())
        signals = np.load(folder / 'signals.npz', allow_pickle=False)['lime']
        tuning, evaluation = rows(folder / 'tuning.jsonl'), rows(folder / 'evaluation.jsonl')
        selection, summary = read(folder / 'selection.json'), read(folder / 'summary.json')
        assert len(tuning) == 128 and len(evaluation) == 512
        assert selection['evaluation_users_read'] == 0
        assert digest(folder / 'selection.json') == summary['selection_sha256']
        assert summary['model_unchanged'] and not summary['test_read'] and summary['optimizer_steps'] == 0
        checked = 0
        for split, data, indices in [('tuning', tuning, range(3, 131)), ('evaluation', evaluation, range(131, 643))]:
            assert [r['sample_index'] for r in data] == list(indices)
            for row in data:
                j = row['sample_index']; sample = samples[j]
                assert row['split'] == split
                assert all(row[key] == sample[key] for key in ('user_id', 'target', 'history', 'split'))
                history = set(row['history'])
                for name, arm in row['arms'].items():
                    if name == 'LIME':
                        ids = np.arange(1, signals.shape[1] + 1)
                        scores = signals[j]
                    else:
                        values = row['candidate_scores'][name]
                        ids, scores = np.asarray(values['ids']), np.asarray(values['scores'])
                        assert len(ids) == 50 and len(set(ids.tolist())) == 50
                    assert len(ids) == len(scores) and np.isfinite(scores).all()
                    keep = np.asarray([int(i) not in history for i in ids])
                    ids, scores = ids[keep], scores[keep]
                    ordered = ids[np.lexsort((ids, -scores))][:50].tolist()
                    rank = ordered.index(row['target']) + 1 if row['target'] in ordered else 51
                    assert ordered == arm['items'] and rank == arm['rank'], (domain, split, j, name)
                    checked += 1
        for name in ('sasrec', 'lime', 'delayed_branch', 'agreement'):
            table = {str(a): calculate([r['arms']['GRAM' if a == 0 else f'{name}_{a}']['rank'] for r in tuning])['N10']
                     for a in (0., .1, .3)}
            chosen = max((0., .1, .3), key=lambda a: (table[str(a)], -a))
            assert selection['selected_alpha'][name] == chosen == summary['selected_alpha'][name]
            for alpha in table:
                np.testing.assert_allclose(table[alpha], selection['tuning_N10'][name][alpha], atol=1e-14, rtol=0)
        ranks = {n: [r['arms'][n]['rank'] for r in evaluation] for n in evaluation[0]['arms']}
        assert set(ranks) == set(summary['metrics'])
        for name, rr in ranks.items():
            for key, value in calculate(rr).items():
                np.testing.assert_allclose(value, summary['metrics'][name][key], atol=1e-14, rtol=0)
        pairs = {n + '_vs_GRAM': (n, 'GRAM') for n in ranks if n != 'GRAM'}
        for n in ('sasrec', 'lime', 'delayed_branch', 'agreement'):
            pairs[n + '_vs_late'] = (n, n + '_late')
        for n in ('delayed_branch', 'agreement'):
            pairs[n + '_vs_lime'] = (n, 'lime')
            pairs[n + '_vs_sasrec'] = (n, 'sasrec')
            pairs[n + '_vs_direct_LIME_rerank'] = (n, 'LIME_after_GRAM')
        assert set(pairs) == set(summary['comparisons'])
        for name, (a, b) in pairs.items():
            actual = compare(ranks[a], ranks[b])
            for key, value in actual.items():
                np.testing.assert_allclose(value, summary['comparisons'][name][key], atol=1e-12, rtol=0)
        verified[domain] = dict(tuning_users=128, evaluation_users=512, predictions_checked=checked,
            evaluation_user_hash_order_verified=True, old_decode_cohort_disjoint=True,
            gate_users_excluded=True, metrics_and_bootstrap_recomputed=True,
            summary_sha256=digest(folder / 'summary.json'))
    result = dict(state='VERIFIED', verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        verified_domains=verified, input_hashes_verified=len(frozen),
        independent_of_runner_scoring_code=True, test_read=False)
    temporary = OUT / f'verification.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUT / 'verification.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
