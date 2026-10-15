"""Independent saved-output audit of the full-catalog single-fusion pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment.phase21 import single_fusion_core as C

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
PRIOR = ROOT/'artifacts/phase21/dual_candidate_generation_20261014_v1'
SOURCE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
ARMS = ('mean', 'user', 'candidate')
KEYS = ('native', 'baseline', 'three', 'residual_candidate') + tuple(a+s for a in ARMS for s in ('_late', '_guided'))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def rank(ids, scores, target, history, missing):
    ids, scores = np.asarray(ids), np.asarray(scores)
    keep = ~np.isin(ids, history)
    ids, scores = ids[keep], scores[keep]
    order = ids[np.lexsort((ids, -scores))]
    pos = np.flatnonzero(order == target)
    return int(pos[0])+1 if len(pos) else missing


def utility(r, k=10):
    return np.where(r <= k, 1/np.log2(r+1), 0.)


def main():
    torch.set_num_threads(2)
    status = read(OUT/'status.json'); assert status['state'] == 'COMPLETED'
    manifest = read(OUT/'manifest.json')['sha256']; hashes = {}

    def verify(path, expected=None):
        name = str(path.relative_to(ROOT))
        actual = sha(path)
        assert actual == (manifest[name] if expected is None else expected), name
        hashes[name] = actual

    records = arrays = metric_count = comparison_count = pure_count = 0
    results = {}
    for domain in ('Toys', 'Beauty'):
        folder = OUT/domain
        assert read(folder/'status.json')['state'] == 'COMPLETED'
        assert read(folder/'lease_status.json')['state'] == 'released'
        runtime = read(folder/'runtime_checks.json')
        assert runtime['gram_model_digest_unchanged'] and runtime['native_reproduced_users'] == 512
        for file in ('output_manifest.json', 'training_manifest.json'):
            for name, digest in read(folder/file)['sha256'].items():
                verify(ROOT/name, digest)
        for name in ('samples.json', 'data_config.json', 'catalog.json', 'references.json'):
            verify(folder/name)
        samples = read(folder/'samples.json')
        predictions = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        refs = read(folder/'references.json')
        users = [s['user_id'] for s in samples]
        assert len(samples) == len(predictions) == len(refs) == len(set(users)) == 512
        saved = np.load(folder/'ranks.npz')
        np.testing.assert_array_equal(saved['users'], users)
        summary = read(folder/'summary.json')
        ranks = {key: [] for key in KEYS}
        for sample, row, reference in zip(samples, predictions, refs):
            for key in ('user_id', 'target', 'history'):
                assert sample[key] == row[key]
            assert reference['user_id'] == sample['user_id']
            for key in ('native', 'baseline', 'three', 'residual_candidate'):
                assert row[key]['ids'] == reference[key]['ids']
            for arm in ARMS:
                assert set(row[arm+'_late']['ids']) == set(row['native']['ids'])
            for key in KEYS:
                p = row[key]
                assert len(p['ids']) == len(set(p['ids'])) == len(p['scores']) == 50
                assert np.isfinite(p['scores']).all()
                ranks[key].append(rank(p['ids'], p['scores'], sample['target'], sample['history'], 2**30))
                records += 1
        ranks = {key: np.asarray(value) for key, value in ranks.items()}
        for key, r in ranks.items():
            np.testing.assert_array_equal(r, saved[key]); arrays += 1
            for k in (5, 10, 20, 50):
                assert abs(float(utility(r, k).mean()) - summary['metrics'][key]['N'+str(k)]) < 1e-14
                assert abs(float((r <= k).mean()) - summary['metrics'][key]['H'+str(k)]) < 1e-14
                metric_count += 2
        prior = np.load(PRIOR/domain/'ranks.npz')
        np.testing.assert_array_equal(prior['users'], users)
        for key, old in (('native', 'native'), ('baseline', 'baseline_guided'), ('three', 'three'), ('residual_candidate', 'candidate_guided')):
            np.testing.assert_array_equal(ranks[key], prior[old])
        for name, comparison in summary['comparisons'].items():
            a, b = name.split('_vs_'); x, y = ranks[a], ranks[b]
            assert abs(float((utility(x)-utility(y)).mean()) - comparison['delta_N10']) < 1e-14
            assert int(((x <= 10) & (y > 10)).sum()) == comparison['top10_gained']
            assert int(((x > 10) & (y <= 10)).sum()) == comparison['top10_lost']
            comparison_count += 1

        training = [s for s in read(SOURCE/domain/'samples.json') if s['role'] == 'ranker_fit']
        expected_fit_users = [s['user_id'] for s in training]
        assert set(expected_fit_users).isdisjoint(users)
        item = np.load(folder/'validation/item.npy', mmap_mode='r')
        user = np.load(folder/'validation/user.npy', mmap_mode='r')
        signals = np.load(folder/'signals.npz')
        pure = np.load(folder/'pure_ranks.npz')
        np.testing.assert_array_equal(pure['users'], users)
        np.testing.assert_array_equal(signals['mean'], item[..., :2].mean(-1))
        model_checks = {}
        for arm in ('user', 'candidate'):
            ck = torch.load(folder/(arm+'.pt'), map_location='cpu')
            assert ck['fit_users'] == expected_fit_users and ck['config'] == read(OUT/'config.json')
            assert ck['seed'] == 0 and ck['old_fusion_used'] is False
            assert ck['selected_epochs'] == min(ck['selection_curve'], key=lambda r: (r['loss'], r['epoch']))['epoch']
            checked = [0, 127, 255, 383, 511]
            for j in checked:
                scores, weights = C.numpy_score(ck['state_dict'], user[j], item[j], ck['mean'], ck['std'], arm)
                np.testing.assert_allclose(scores, signals[arm][j], atol=3e-6, rtol=2e-5)
                np.testing.assert_allclose(weights.sum(-1), 1., atol=2e-6)
                assert (scores >= item[j, :, :2].min(-1)-2e-6).all()
                assert (scores <= item[j, :, :2].max(-1)+2e-6).all()
            model_checks[arm] = dict(checked_full_catalog_users=checked, selected_epochs=ck['selected_epochs'], fit_users=len(expected_fit_users))
        for arm in ARMS:
            rr = []
            for j, sample in enumerate(samples):
                values = signals[arm][j]; v = len(values)
                rr.append(rank(np.arange(1, v+1), values, sample['target'], sample['history'], v+1))
                pure_count += 1
            np.testing.assert_array_equal(rr, pure[arm])
            for k in (5, 10, 20, 50):
                r = np.asarray(rr)
                assert abs(float(utility(r, k).mean()) - summary['training']['pure_metrics'][arm]['N'+str(k)]) < 1e-14
                assert abs(float((r <= k).mean()) - summary['training']['pure_metrics'][arm]['H'+str(k)]) < 1e-14
                metric_count += 2

        m = summary['metrics']
        relative = {a: {b: float(m[a+'_guided']['N10']/m[b]['N10']-1) for b in ('native', 'baseline', 'three', 'mean_guided')}
                    for a in ('user', 'candidate')}
        relative['candidate']['user_guided'] = m['candidate_guided']['N10']/m['user_guided']['N10']-1
        results[domain] = dict(relative_N10=relative, model_checks=model_checks,
            hits={key: {str(k): int((r <= k).sum()) for k in (10, 50)} for key, r in ranks.items()},
            candidate_minus_user_by_stage=dict(
                pure=float(utility(pure['candidate']).mean()-utility(pure['user']).mean()),
                late=m['candidate_late']['N10']-m['user_late']['N10'],
                guided=m['candidate_guided']['N10']-m['user_guided']['N10']),
            coverage={a: dict(new_top10_outside_native50=int(((ranks[a+'_guided'] <= 10) & (ranks['native'] > 50)).sum()),
                top50_gained=int(((ranks[a+'_guided'] <= 50) & (ranks['native'] > 50)).sum()),
                top50_lost=int(((ranks[a+'_guided'] > 50) & (ranks['native'] <= 50)).sum())) for a in ARMS})
    for p in (ROOT/'experiment/phase21/single_fusion.py', ROOT/'experiment/phase21/single_fusion_core.py', OUT/'config.json'):
        verify(p)
    review = OUT/'review'; review.mkdir(exist_ok=False)
    write(review/'diagnostic.json', results)
    write(review/'manifest.json', dict(sha256=hashes, reviewer_sha256=sha(Path(__file__))))
    write(review/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS',
        prediction_records_recomputed=records, generated_rank_arrays_recomputed=arrays,
        pure_ranking_records_recomputed=pure_count, aggregate_metrics_rechecked=metric_count,
        contrasts_rechecked=comparison_count, source_and_output_hashes_rechecked=len(hashes),
        all_expected_users_present=True, fit_development_users_disjoint=True,
        epoch_selection_from_internal_holdout_only=True, model_numpy_forward_checks=20,
        new_training=False, new_generation=False, test_read=False))
    scan = {
        'Simpson_paradox': 'Separate domains: candidate-minus-user direction differs; do not pool away Toys regression.',
        'ecological_fallacy': 'Paired ranks and hit counts; average improvements do not imply universal user benefit.',
        'Berkson_selection': '512 repeatedly used development users; exploratory rather than independent confirmation.',
        'collider_bias': 'No rows excluded by outcomes; coverage groups are explicitly descriptive outcome-defined groups.',
        'base_rate_neglect': 'Absolute N10 and H10/H50 counts retained alongside relative gains.',
        'regression_to_mean': 'Seed0 fixed; prior branch selection and exploratory reuse remain limitations.',
        'survivorship_bias': 'All 1024 planned user records and all planned variants completed.',
        'look_elsewhere': 'All variants and both domains reported; unadjusted intervals, no definitive significance claims.',
        'forking_paths': 'Internal CE selects epochs; neither development N10 nor pure ranking selected seeds or variants for generation.',
        'causal_language': 'Multiple structural changes vs previous baseline; cannot attribute gain solely to removing residuals or expanding scope.',
        'reverse_causality': 'Feature API excludes target; stored targets only enter supervised ranker_fit loss and outcome evaluation.'}
    write(review/'interpretation_checks.json', dict(coverage='11/11', checks=scan,
        decision='Single fusion is a viable provisional simplification; candidate-specific benefit is domain-dependent and expert necessity remains untested here.'))
    print(json.dumps(read(review/'verification.json'), indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
