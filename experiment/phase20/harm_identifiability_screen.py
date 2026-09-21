"""Nested development-user probe for identifying harmful frozen corrections.

Explicitly learns from development-validation outcomes; never a train-only
method score. Fixed proposal and outer-user exclusion are checked and saved.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiment.phase20.harm_identifiability_core import (
    accept, features, fold_ids, ndcg10, partitions, select_threshold, target_ranks,
)

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def metrics(r):
    return dict(users=len(r), **{f'{name}@{k}': float(np.mean(
        (r <= k) if name == 'hit' else np.where(r <= k, 1. / np.log2(r + 1), 0.)))
        for k in (5, 10, 20, 50) for name in ('hit', 'ndcg')})


def paired(a, b, config):
    delta = np.stack([ndcg10(a) - ndcg10(b), (a <= 10).astype(float) - (b <= 10)], axis=1)
    rng = np.random.default_rng(config['seed'])
    samples = []
    for start in range(0, config['bootstrap_repeats'], 32):
        count = min(32, config['bootstrap_repeats'] - start)
        ix = rng.integers(0, len(a), size=(count, len(a)))
        samples.append(delta[ix].mean(1))
    ci = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    positive = np.sort(delta[delta[:, 0] > 0, 0])[::-1]
    return dict(ndcg10_delta=float(delta[:, 0].mean()), ndcg10_ci95=ci[:, 0].tolist(),
        hit10_delta=float(delta[:, 1].mean()), hit10_ci95=ci[:, 1].tolist(),
        ndcg10_improved=int((delta[:, 0] > 0).sum()), ndcg10_harmed=int((delta[:, 0] < 0).sum()),
        hit10_gained=int(((a <= 10) & (b > 10)).sum()), hit10_lost=int(((a > 10) & (b <= 10)).sum()),
        top5_positive_gain_share=float(positive[:5].sum() / positive.sum()) if len(positive) else 0.)


class Probe:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = read(config_path)
        self.path = ROOT / self.config['output']
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.config_hash = sha(config_path)

    def status(self, state='RUNNING', stage='', **extra):
        elapsed = time.monotonic() - self.started
        if state == 'RUNNING' and elapsed > self.config['budget_seconds']:
            raise TimeoutError('Predeclared CPU screen budget exceeded')
        value = dict(state=state, stage=stage, pid=os.getpid(), elapsed_seconds=elapsed,
                     updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)
        write(self.path / 'status.json', value)
        with (self.path / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def verify(self):
        assert sha(self.config_path) == self.config_hash
        for group in ('input_sha256', 'source_sha256'):
            for path, expected in self.config[group].items():
                assert sha(ROOT / path) == expected, path

    def prepare(self):
        self.status(stage='verify_and_prepare')
        self.verify()
        d = self.config['data']
        q = load_npz(ROOT / d['queries'])
        saved = load_npz(ROOT / d['proposal_scores'])
        n = int(q['ntrain'])
        users = q['users'][n:]
        assert len(set(users)) == len(users) == 19412
        baseline, proposal = saved['baseline'], saved['excluded']
        np.testing.assert_array_equal(baseline, q['original_base'][n:])
        rank_base, rank_proposal = [target_ranks(s, q['positions'][n:]) for s in (baseline, proposal)]
        np.testing.assert_array_equal(rank_base, q['known_ranks'])
        assert np.array_equal(rank_base <= 50, rank_proposal <= 50)
        # Frequencies reconstructed from prepared train only, including each
        # user's initial item once; validation targets never enter this prior.
        frequency = np.zeros(int(q['num_items']) + 1, dtype=np.int64)
        first = set()
        with (ROOT / d['train']).open() as handle:
            for line in handle:
                r = json.loads(line)
                if r['user_id'] not in first:
                    assert len(r['history']) == 1
                    frequency[r['history'][0]] += 1
                    first.add(r['user_id'])
                frequency[r['target']] += 1
        np.testing.assert_array_equal(frequency[q['candidate_ids']], q['frequencies'])
        f = q['original_features'][n:]
        inputs = dict(baseline=baseline, proposal=proposal, generation=f[:, :, 0], collaborative=f[:, :, 5],
            candidate_frequencies=np.take_along_axis(q['frequencies'][n:], q['order'][n:], axis=1),
            history_frequencies=frequency[q['histories'][n:]], history_mask=q['histories'][n:] > 0)
        x, names, simple = features(**inputs)
        assert names == self.config['feature_names']
        outer = fold_ids(users, self.config['seed'])
        y = ndcg10(rank_proposal) - ndcg10(rank_base)
        yh = (rank_proposal <= 10).astype(float) - (rank_base <= 10)
        np.savez(self.path / 'dataset.npz', features=x, users=users, outer=outer,
                 rank_baseline=rank_base, rank_proposal=rank_proposal, delta_ndcg=y, delta_hit=yh,
                 low_frequency=q['target_frequencies'][n:] <= 5, simple_columns=simple)
        write(self.path / 'feature_contract.json', dict(names=names, simple_columns=simple.tolist(),
            target_free_feature_function=True, identity_only_used_for_partition=True,
            validation_outcomes_used_for_probe_training=True, train_only_method=False))
        return x, users, outer, y, yh, rank_base, rank_proposal, q['target_frequencies'][n:] <= 5, simple

    def run(self):
        x, users, outer, y, yh, rb, rp, low, simple = self.prepare()
        predictions = {a: np.full(len(x), np.nan) for a in ('popularity', 'full', 'shuffled')}
        decisions = {a: np.zeros(len(x), dtype=bool) for a in predictions}
        assignment_count = np.zeros(len(x), dtype=int)
        folds = []
        for fold in range(5):
            fit, cal, ev = partitions(users, self.config['seed'], fold)
            assert set(fit).isdisjoint(cal) and set(fit).isdisjoint(ev) and set(cal).isdisjoint(ev)
            assert len(fit) + len(cal) + len(ev) == len(x)
            assignment_count[ev] += 1
            record = dict(fold=fold, fit=len(fit), calibration=len(cal), evaluation=len(ev), arms={})
            np.savez(self.path / f'fold{fold}.indices.npz', fit=fit, calibration=cal, evaluation=ev)
            for arm in predictions:
                self.status(stage='fit', fold=fold, arm=arm)
                columns = simple if arm == 'popularity' else np.arange(x.shape[1])
                fit_y, cal_y, cal_h = y[fit], y[cal], yh[cal]
                if arm == 'shuffled':
                    rng = np.random.default_rng(self.config['seed'] + 1000 + fold)
                    fit_permutation, cal_permutation = rng.permutation(len(fit)), rng.permutation(len(cal))
                    fit_y = fit_y[fit_permutation]
                    cal_y, cal_h = cal_y[cal_permutation], cal_h[cal_permutation]
                    np.savez(self.path / f'fold{fold}.shuffles.npz', fit=fit_permutation, calibration=cal_permutation)
                model = HistGradientBoostingRegressor(random_state=self.config['seed'], **self.config['model'])
                model.fit(x[fit][:, columns], fit_y)
                assert model.n_iter_ == self.config['model']['max_iter'] and not model.do_early_stopping_
                fit_prediction, cal_prediction, ev_prediction = [model.predict(x[ix][:, columns]) for ix in (fit, cal, ev)]
                threshold, candidates = select_threshold(fit_prediction, cal_prediction, cal_y, cal_h, self.config)
                predictions[arm][ev] = ev_prediction
                decisions[arm][ev] = accept(ev_prediction, threshold)
                joblib.dump(model, self.path / f'fold{fold}.{arm}.joblib')
                np.savez(self.path / f'fold{fold}.{arm}.predictions.npz', fit=fit_prediction, calibration=cal_prediction,
                         evaluation=ev_prediction, columns=columns)
                dr = np.where(decisions[arm][ev], rp[ev], rb[ev])
                record['arms'][arm] = dict(threshold='never' if threshold is None else 'always' if threshold == -np.inf else float(threshold),
                    threshold_candidates=candidates, evaluation_accepted=int(decisions[arm][ev].sum()),
                    evaluation_ndcg_delta=float((ndcg10(dr) - ndcg10(rb[ev])).mean()),
                    evaluation_hit_delta=float(((dr <= 10).astype(float) - (rb[ev] <= 10)).mean()))
            folds.append(record)
            write(self.path / 'fold_results.json', folds)
            self.status(stage='fold_complete', fold=fold)
        assert (assignment_count == 1).all()
        assert all(np.isfinite(p).all() for p in predictions.values())
        result_ranks = dict(baseline=rb, always=rp, **{a: np.where(decisions[a], rp, rb) for a in predictions})
        assert all(np.array_equal(r <= 50, rb <= 50) for r in result_ranks.values())
        masks = dict(all=np.ones(len(x), dtype=bool), low_frequency=low, higher_frequency=~low)
        groups = {g: {a: metrics(r[m]) for a, r in result_ranks.items()} for g, m in masks.items()}
        comparisons = {}
        for a, b in [('always', 'baseline'), ('popularity', 'baseline'), ('full', 'baseline'),
                     ('shuffled', 'baseline'), ('full', 'popularity'), ('full', 'shuffled')]:
            comparisons[a + '_minus_' + b] = {g: paired(result_ranks[a][m], result_ranks[b][m], self.config)
                                              for g, m in masks.items()}
            self.status(stage='statistics', comparison=a + '_minus_' + b)
        changed = y != 0
        diagnostic = {}
        for arm, prediction in predictions.items():
            take = decisions[arm]
            diagnostic[arm] = dict(auc_changed=float(roc_auc_score(y[changed] > 0, prediction[changed])),
                accepted_users=int(take.sum()), accepted_gain_users=int((take & (y > 0)).sum()),
                accepted_harm_users=int((take & (y < 0)).sum()), rejected_gain_users=int((~take & (y > 0)).sum()),
                rejected_harm_users=int((~take & (y < 0)).sum()),
                positive_folds=sum(f['arms'][arm]['evaluation_ndcg_delta'] > 0 for f in folds),
                accepted_gain_mass=float(y[take & (y > 0)].sum()), accepted_harm_mass=float(-y[take & (y < 0)].sum()))
        points = groups['all']
        checks = dict(ndcg_practical_gain=points['full']['ndcg@10'] - points['baseline']['ndcg@10'] >= .0004,
            beats_popularity=points['full']['ndcg@10'] > points['popularity']['ndcg@10'],
            beats_shuffled=points['full']['ndcg@10'] > points['shuffled']['ndcg@10'],
            preserves_hit10=points['full']['hit@10'] >= points['baseline']['hit@10'],
            positive_four_folds=diagnostic['full']['positive_folds'] >= 4,
            auc_above_working_bar=diagnostic['full']['auc_changed'] >= .55)
        np.savez(self.path / 'oof_predictions.npz', **predictions)
        np.savez(self.path / 'oof_decisions.npz', **decisions)
        np.savez(self.path / 'oof_ranks.npz', **result_ranks)
        with (self.path / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', 'outer_fold', *result_ranks])
            for i, u in enumerate(users):
                writer.writerow([u, int(outer[i]), *[int(r[i]) for r in result_ranks.values()]])
        self.verify()
        result = dict(state='COMPLETED', outcome='INVESTIGATE_TRAIN_ONLY_TRANSFER' if all(checks.values()) else 'STOP_THIS_IDENTIFICATION_RECIPE',
            decision_checks=checks, groups=groups, comparisons=comparisons, diagnostics=diagnostic, folds=folds,
            oracle_diagnostic=dict(ndcg10=float(ndcg10(rb).mean() + np.maximum(y, 0).mean()),
                                   gain_users=int((y > 0).sum()), harm_users=int((y < 0).sum()),
                                   deployable=False, used_for_selection=False),
            config_sha256=self.config_hash, elapsed_seconds=time.monotonic() - self.started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            sklearn_version=sklearn.__version__, validation_outcomes_used_for_training=True,
            train_only_method=False, new_CF_or_backbone_updates=0, gpu_used=False, test_read=False,
            statistical_scope='Nested predictions on repeatedly used development validation. Conditional descriptive user bootstrap, no seed/fold fitting or adaptive development uncertainty; not confirmatory.')
        write(self.path / 'summary.json', result)
        self.status('COMPLETED', 'finished', outcome=result['outcome'])
        write(self.path / 'manifest.json', {p.name: sha(p) for p in self.path.iterdir() if p.is_file() and p.name != 'manifest.json'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase20/harm_identifiability_toys_v1.json')
    args = parser.parse_args()
    job = Probe(ROOT / args.config)
    try:
        with threadpool_limits(limits=4):
            job.run()
    except BaseException as error:
        job.status('FAILED', 'exception', error=repr(error))
        (job.path / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
