"""Post-outcome descriptive train/development difficulty audit; no fitting or selection."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'
INPUTS = {}


def read(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data: return {k: data[k] for k in data.files}


def z(x):
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-6)


def profile(score, pos, tail, weights):
    assert (pos >= 0).all()
    n = len(pos); idx = np.arange(n); gold = score[idx, pos]
    ranks = 1 + (score > gold[:, None]).sum(1) + ((score == gold[:, None]) & (np.arange(50)[None] < pos[:, None])).sum(1)
    alternative = 1 + np.argsort(np.argsort(-score, axis=1, kind='stable'), axis=1)[idx, pos]
    np.testing.assert_array_equal(ranks, alternative)
    beta = np.partition(score, 40, axis=1)[:, 40]
    exp = np.exp(score - score.max(1, keepdims=True)); prob = exp / exp.sum(1, keepdims=True)
    neg = exp.copy(); neg[idx, pos] = 0; neg /= neg.sum(1, keepdims=True)
    ce_grad = prob.copy(); ce_grad[idx, pos] -= 1
    neg_grad = neg.copy(); neg_grad[idx, pos] -= 1
    identity_error = float(np.abs(ce_grad - (1 - prob[idx, pos])[:, None] * neg_grad).max())
    assert identity_error < 1e-12
    positive_gradient = dict(softmax=1 - prob[idx, pos], negative_softmax=np.ones(n),
                            talos=np.exp(-np.logaddexp(0, gold - beta)))
    assert abs(weights.sum() - 1) < 1e-12
    bins = {'rank1': ranks == 1, 'rank2_5': (ranks >= 2) & (ranks <= 5), 'rank6_10': (ranks >= 6) & (ranks <= 10),
            'rank11_20': (ranks >= 11) & (ranks <= 20), 'rank21_50': ranks > 20}
    groups = dict(all=np.ones(n, bool), tail=tail, non_tail=~tail)
    return {'rows': n, 'top10_fraction_unweighted': float((ranks <= 10).mean()),
            'top10_fraction_weighted': float(weights @ (ranks <= 10)),
            'positive_probability_mean': float(prob[idx, pos].mean()),
            'gold_minus_threshold_quantiles': dict(zip(['q00', 'q25', 'q50', 'q75', 'q100'], np.quantile(gold - beta, [0, .25, .5, .75, 1]).tolist())),
            'ce_negative_gradient_identity_error': identity_error,
            'groups': {name: {'rows': int(mask.sum()), 'top10_fraction_unweighted': float((ranks[mask] <= 10).mean())} for name, mask in groups.items()},
            'rank_bins': {name: {'rows': int(mask.sum()), 'fraction_unweighted': float(mask.mean()), 'sample_weight_share': float(weights[mask].sum()),
                'positive_score_gradient_mass_share': {arm: float((weights[mask] * grad[mask]).sum() / (weights * grad).sum()) for arm, grad in positive_gradient.items()}}
                for name, mask in bins.items()}}


def main():
    started = time.monotonic(); domains = {}
    audit = json.loads((OUT / 'verification.json').read_text()); assert audit['state'] == 'VERIFIED'
    for domain in ('Toys', 'Beauty'):
        ctx = read(ROOT / f'artifacts/phase20/future_fusion_alignment_20260917/{domain}.training_context.npz')
        a = read(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/training_cache.npz')
        ix = ctx['fit_indices']; pos = a['positions'][ix]; tail = ~a['head'][ix]
        weights = np.where(tail, .5 / tail.sum(), .5 / (~tail).sum())
        train = profile(z(ctx['prior']) + ctx['cf_mean'], pos, tail, weights)
        v = read(ROOT / f'artifacts/phase20/future_factor_screen_20260917/{domain}.audit_validation_cache.npz')
        covered = v['positions'] >= 0
        valid = profile(z(v['prior'][covered]) + v['cf_mean'][covered], v['positions'][covered], v['tail_mask'][covered], np.full(covered.sum(), 1 / covered.sum()))
        nonfit = covered & ~np.isin(v['users'], np.unique(ctx['users']))
        valid_nonfit = profile(z(v['prior'][nonfit]) + v['cf_mean'][nonfit], v['positions'][nonfit], v['tail_mask'][nonfit], np.full(nonfit.sum(), 1 / nonfit.sum()))
        domains[domain] = dict(train_fit_covered_complete_window=train, development_covered=valid, development_nonfit_covered=valid_nonfit,
            development_users=len(covered), development_covered_users=int(covered.sum()), development_uncovered_users=int((~covered).sum()))
    result = dict(state='CHECKED', analysis_timing='post_outcome_exploratory', training_steps=0, test_read=False,
        note='Initial frozen scores only. Different sample-selection conditions. Positive-score derivative mass is not network-parameter gradient attribution. Development labels used for diagnosis only.',
        domains=domains, input_sha256=INPUTS, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), elapsed_seconds=time.monotonic()-started)
    (OUT / 'initial_difficulty.json').write_text(json.dumps(result, indent=2) + '\n')
    for d, v in domains.items():
        print(d, {k: {'rows': v[k]['rows'], 'top10': v[k]['top10_fraction_unweighted'], 'median_margin': v[k]['gold_minus_threshold_quantiles']['q50']} for k in ('train_fit_covered_complete_window', 'development_covered', 'development_nonfit_covered')})
        print('Train outside-Top10 positive-score gradient shares:', {arm: sum(v['train_fit_covered_complete_window']['rank_bins'][b]['positive_score_gradient_mass_share'][arm] for b in ('rank11_20','rank21_50')) for arm in ('softmax','negative_softmax','talos')})
    print('seconds', result['elapsed_seconds'])


if __name__ == '__main__': main()
