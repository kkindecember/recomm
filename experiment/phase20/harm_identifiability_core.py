"""Outcome-free features and calibration-only decision selection.

This is an identifiability probe, not a train-only recommender method.
"""
import hashlib
import numpy as np


def fold_ids(users, seed, salt='outer', count=5):
    return np.asarray([int.from_bytes(hashlib.sha256(
        f'{seed}|{salt}|{u}'.encode()).digest()[:8], 'big') % count for u in users])


def partitions(users, seed, fold):
    outer = fold_ids(users, seed)
    calibration = fold_ids(users, seed, f'calibration-{fold}') == 0
    evaluation = outer == fold
    return np.flatnonzero(~evaluation & ~calibration), np.flatnonzero(~evaluation & calibration), np.flatnonzero(evaluation)


def item_ranks(scores):
    return np.argsort(np.argsort(-scores, axis=1, kind='stable'), axis=1) + 1


def target_ranks(scores, positions):
    p = np.asarray(positions)
    return np.where(p >= 0, item_ranks(scores)[np.arange(len(p)), np.maximum(p, 0)], 51)


def ndcg10(r):
    return np.where(r <= 10, 1. / np.log2(r + 1), 0.)


def features(*, baseline, proposal, generation, collaborative, candidate_frequencies,
             history_frequencies, history_mask):
    """No target, position, coverage, or identity accepted by this function."""
    baseline, proposal, generation, collaborative = [np.asarray(x, dtype=np.float64)
        for x in (baseline, proposal, generation, collaborative)]
    assert baseline.shape[1] == 50
    assert all(x.shape == baseline.shape for x in (proposal, generation, collaborative, candidate_frequencies))
    rb, rp, rg, rc = [item_ranks(x) for x in (baseline, proposal, generation, collaborative)]
    db, dp = [np.where(r <= 10, 1. / np.log2(r + 1), 0.) for r in (rb, rp)]
    movement = dp - db
    logf = np.log1p(candidate_frequencies)
    tail = np.asarray(candidate_frequencies) <= 5
    length = history_mask.sum(1)
    assert (length > 0).all()
    hlog = np.log1p(history_frequencies)
    hmean = (hlog * history_mask).sum(1) / length
    hstd = np.sqrt(((hlog - hmean[:, None]) ** 2 * history_mask).sum(1) / length)
    cols, names = [], []
    def add(name, value):
        assert value.shape == (len(baseline),)
        names.append(name)
        cols.append(value)
    def topmean(values, r, k=10):
        return (values * (r <= k)).sum(1) / k
    def topscore(values, r):
        return (values * (r == 1)).sum(1)
    # Simple control gets exactly these popularity features, no outcomes.
    add('pop_history_mean_log_frequency', hmean)
    add('pop_history_std_log_frequency', hstd)
    add('pop_history_tail_fraction', ((history_frequencies <= 5) * history_mask).sum(1) / length)
    add('pop_baseline_top10_mean_log_frequency', topmean(logf, rb))
    add('pop_baseline_top10_tail_fraction', topmean(tail, rb))
    add('pop_proposal_top10_log_frequency_change', topmean(logf, rp) - topmean(logf, rb))
    add('pop_proposal_top10_tail_fraction_change', topmean(tail, rp) - topmean(tail, rb))
    add('pop_top1_log_frequency_change', topscore(logf, rp) - topscore(logf, rb))
    add('pop_discount_weighted_frequency_change', (movement * logf).sum(1))
    add('history_log_length', np.log1p(length))
    ordered = np.sort(baseline, axis=1)[:, ::-1]
    for k in (1, 3, 5, 10, 20):
        add(f'baseline_gap_{k}_{k+1}', ordered[:, k - 1] - ordered[:, k])
    centered = baseline - baseline.max(1, keepdims=True)
    prob = np.exp(centered) / np.exp(centered).sum(1, keepdims=True)
    add('baseline_score_entropy', -(prob * np.log(np.maximum(prob, 1e-15))).sum(1))
    for name, r1, r2 in [('generation_cf', rg, rc), ('baseline_generation', rb, rg), ('baseline_cf', rb, rc)]:
        add(name + '_top10_overlap', ((r1 <= 10) & (r2 <= 10)).sum(1) / 10.)
    add('generation_cf_rank_correlation', ((rg - 25.5) * (rc - 25.5)).mean(1) / ((50**2 - 1) / 12.))
    add('edit_top1_changed', np.any((rb == 1) != (rp == 1), axis=1).astype(float))
    for k in (5, 10, 20):
        add(f'edit_top{k}_turnover', ((rp <= k) & (rb > k)).sum(1) / k)
    add('edit_top10_mean_abs_rank_change', topmean(np.abs(rp - rb), rb))
    add('edit_discount_movement_l1', np.abs(movement).sum(1))
    add('edit_positive_discount_count', (movement > 0).sum(1).astype(float))
    add('edit_negative_discount_count', (movement < 0).sum(1).astype(float))
    correction = proposal - baseline
    add('edit_score_correction_abs_mean', np.abs(correction).mean(1))
    add('edit_score_correction_abs_max', np.abs(correction).max(1))
    for name, values in [('generation', generation), ('cf', collaborative), ('baseline', baseline)]:
        add(f'edit_discount_weighted_{name}_support', (movement * values).sum(1))
        add(f'edit_top10_{name}_change', topmean(values, rp) - topmean(values, rb))
    result = np.stack(cols, axis=1)
    assert np.isfinite(result).all()
    return result, names, np.asarray([i for i, name in enumerate(names) if name.startswith('pop_')])


def threshold_grid(fit_predictions, quantiles):
    # None means never apply; -inf is represented as a named sentinel in logs.
    return [None, -np.inf, 0., *np.quantile(fit_predictions, quantiles).tolist()]


def accept(predictions, threshold):
    return np.zeros(len(predictions), dtype=bool) if threshold is None else predictions > threshold


def select_threshold(fit_predictions, calibration_predictions, delta_ndcg, delta_hit, config):
    best, best_gain, best_count = None, 0., 0
    records = []
    for threshold in threshold_grid(fit_predictions, config['threshold_quantiles']):
        mask = accept(calibration_predictions, threshold)
        gain = float(np.mean(delta_ndcg * mask))
        hit = float(np.mean(delta_hit * mask))
        count = int(mask.sum())
        changed = int(((delta_ndcg != 0) & mask).sum())
        eligible = (gain >= config['minimum_calibration_ndcg'] and hit >= 0 and
                    changed >= config['minimum_calibration_changed_users'])
        records.append(dict(threshold='never' if threshold is None else 'always' if threshold == -np.inf else float(threshold),
            ndcg_gain=gain, hit_gain=hit, accepted=count, changed_accepted=changed, eligible=eligible))
        if eligible and (gain > best_gain or (gain == best_gain and count < best_count)):
            best, best_gain, best_count = threshold, gain, count
    return best, records
