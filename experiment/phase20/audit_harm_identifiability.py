"""Independent checks of saved nested predictions, routing and metrics."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def independent_ranks(scores, positions):
    result = np.full(len(scores), 51, dtype=np.int64)
    for i, p in enumerate(positions):
        if p >= 0:
            result[i] = 1 + np.count_nonzero(scores[i] > scores[i, p]) + np.count_nonzero(scores[i, :p] == scores[i, p])
    return result


def main():
    path = ROOT / 'artifacts/phase20/harm_identifiability_toys_v1'
    config_path = ROOT / 'experiment/phase20/harm_identifiability_toys_v1.json'
    cfg, summary = read(config_path), read(path / 'summary.json')
    assert sha(config_path) == summary['config_sha256']
    for group in ('input_sha256', 'source_sha256'):
        for name, expected in cfg[group].items():
            assert sha(ROOT / name) == expected, name
    manifest = read(path / 'manifest.json')
    for name, expected in manifest.items():
        assert sha(path / name) == expected, name
    data = load(path / 'dataset.npz')
    predictions, decisions, ranks = [load(path / name) for name in
        ('oof_predictions.npz', 'oof_decisions.npz', 'oof_ranks.npz')]
    q = load(ROOT / cfg['data']['queries'])
    source_scores = load(ROOT / cfg['data']['proposal_scores'])
    n = int(q['ntrain'])
    baseline = independent_ranks(source_scores['baseline'], q['positions'][n:])
    proposal = independent_ranks(source_scores['excluded'], q['positions'][n:])
    np.testing.assert_array_equal(baseline, data['rank_baseline'])
    np.testing.assert_array_equal(proposal, data['rank_proposal'])
    nd = lambda r: np.where(r <= 10, 1. / np.log2(r + 1), 0.)
    y = nd(proposal) - nd(baseline)
    yh = (proposal <= 10).astype(float) - (baseline <= 10)
    np.testing.assert_array_equal(y, data['delta_ndcg'])
    records = read(path / 'fold_results.json')
    visited = np.zeros(len(baseline), dtype=int)
    for fold, record in enumerate(records):
        ix = load(path / f'fold{fold}.indices.npz')
        fit, cal, ev = ix['fit'], ix['calibration'], ix['evaluation']
        assert not (set(fit) & set(cal) or set(fit) & set(ev) or set(cal) & set(ev))
        assert len(set(fit) | set(cal) | set(ev)) == len(baseline)
        assert (data['outer'][ev] == fold).all()
        visited[ev] += 1
        for arm in ('popularity', 'full', 'shuffled'):
            model = joblib.load(path / f'fold{fold}.{arm}.joblib')
            saved = load(path / f'fold{fold}.{arm}.predictions.npz')
            for name, ids in [('fit', fit), ('calibration', cal), ('evaluation', ev)]:
                computed = model.predict(data['features'][ids][:, saved['columns']])
                np.testing.assert_array_equal(computed, saved[name])
            np.testing.assert_array_equal(predictions[arm][ev], saved['evaluation'])
            cy, ch = y[cal], yh[cal]
            if arm == 'shuffled':
                perm = load(path / f'fold{fold}.shuffles.npz')['calibration']
                np.testing.assert_array_equal(np.sort(perm), np.arange(len(cal)))
                cy, ch = cy[perm], ch[perm]
            best_threshold, best_gain, best_count = 'never', 0., 0
            for candidate in record['arms'][arm]['threshold_candidates']:
                t = candidate['threshold']
                mask = np.zeros(len(cal), dtype=bool) if t == 'never' else np.ones(len(cal), dtype=bool) if t == 'always' else saved['calibration'] > t
                gain, hit = float(np.mean(cy * mask)), float(np.mean(ch * mask))
                changed = int(np.count_nonzero(mask & (cy != 0)))
                eligible = gain >= cfg['minimum_calibration_ndcg'] and hit >= 0 and changed >= cfg['minimum_calibration_changed_users']
                assert abs(gain - candidate['ndcg_gain']) < 1e-14
                assert bool(eligible) == candidate['eligible']
                if eligible and (gain > best_gain or (gain == best_gain and mask.sum() < best_count)):
                    best_threshold, best_gain, best_count = t, gain, int(mask.sum())
            assert best_threshold == record['arms'][arm]['threshold']
            t = best_threshold
            chosen = np.zeros(len(ev), dtype=bool) if t == 'never' else np.ones(len(ev), dtype=bool) if t == 'always' else saved['evaluation'] > t
            np.testing.assert_array_equal(chosen, decisions[arm][ev])
    np.testing.assert_array_equal(visited, 1)
    for arm in decisions:
        np.testing.assert_array_equal(ranks[arm], np.where(decisions[arm], proposal, baseline))
        changed = y != 0
        yy, pp = y[changed] > 0, predictions[arm][changed]
        np_, nn_ = int(yy.sum()), int((~yy).sum())
        auc = (rankdata(pp)[yy].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_)
        assert abs(auc - summary['diagnostics'][arm]['auc_changed']) < 1e-12
    for group, mask in [('all', np.ones(len(baseline), dtype=bool)), ('low_frequency', data['low_frequency']),
                        ('higher_frequency', ~data['low_frequency'])]:
        for arm, rank in ranks.items():
            r = rank[mask]
            for k in (5, 10, 20, 50):
                actual = summary['groups'][group][arm]
                assert abs(np.mean(r <= k) - actual[f'hit@{k}']) < 1e-12
                dcg = sum(1. / np.log2(int(x) + 1) for x in r if x <= k) / len(r)
                assert abs(dcg - actual[f'ndcg@{k}']) < 1e-12
    assert summary['validation_outcomes_used_for_training'] and not summary['train_only_method']
    result = dict(state='PASSED', manifest_files=len(manifest), nested_folds=5,
        saved_models_repredicted=15, evaluation_users=len(baseline), model_predictions_exact=True,
        threshold_selection_reproduced_without_outer_labels=True, independent_ranks_and_metrics=True,
        independent_changed_user_auc=True, development_training_disclosed=True, test_read=False,
        config_sha256=sha(config_path), audit_source_sha256=sha(Path(__file__)))
    with (path / 'independent_audit.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
