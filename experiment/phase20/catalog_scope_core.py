"""Shared full-catalog scoring and exact candidate-scope comparisons."""
import hashlib
import numpy as np


def split_users(users, domain, seed, calibration_size):
    assert len(set(users)) == len(users) and 0 < calibration_size < len(users)
    order = sorted(range(len(users)), key=lambda i: hashlib.sha256(
        f'{seed}|catalog_scope|{domain}|{users[i]}'.encode()).digest())
    return np.sort(order[:calibration_size]), np.sort(order[calibration_size:])


def eligible_mask(histories, num_items):
    mask = np.ones((len(histories), num_items), dtype=bool)
    mask[:, 0] = False
    for i, history in enumerate(histories):
        mask[i, np.asarray(history, dtype=np.int64)] = False
    return mask


def masked_z(values, mask):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == mask.shape and mask.sum(1).min() > 1
    assert np.isfinite(values[mask]).all()
    n = mask.sum(1, keepdims=True)
    mean = np.where(mask, values, 0.).sum(1, keepdims=True)/n
    centered = values-mean
    variance = np.where(mask, centered*centered, 0.).sum(1, keepdims=True)/n
    result = centered/np.maximum(np.sqrt(variance), 1e-6)
    return np.where(mask, result, 0.)


def mixed_scores(directory, collaborative, weight, mask):
    assert 0 <= weight <= 1
    # Both scopes use these exact full-catalog scores; never renormalize the beam.
    result = (1-weight)*np.asarray(directory, dtype=np.float64)+weight*np.asarray(collaborative, dtype=np.float64)
    return np.where(mask, result, -np.inf)


def restricted(scores, candidate_ids):
    return np.take_along_axis(scores, candidate_ids, axis=1)


def ranks(scores, ids, targets):
    """Rank ties by ascending numeric item ID; absent/ineligible target is infinity."""
    ids = np.broadcast_to(ids, scores.shape)
    matches = ids == np.asarray(targets)[:, None]
    positions = matches.argmax(1)
    target_scores = scores[np.arange(len(scores)), positions]
    present = matches.any(1) & np.isfinite(target_scores)
    above = (scores > target_scores[:, None]) | ((scores == target_scores[:, None]) & (ids < np.asarray(targets)[:, None]))
    return np.where(present, 1+above.sum(1), 2**30).astype(np.int64)


def topk(scores, ids, k=50):
    ids = np.broadcast_to(ids, scores.shape)
    result = np.zeros((len(scores), k), dtype=np.int64)
    for j, (row, items) in enumerate(zip(scores, ids)):
        # Partition first; include all boundary ties before deterministic sorting.
        finite = np.flatnonzero(np.isfinite(row))
        if len(finite) > k:
            boundary = np.partition(row[finite], len(finite)-k)[len(finite)-k]
            finite = finite[row[finite] >= boundary]
        order = np.lexsort((items[finite], -row[finite]))
        chosen = items[finite[order[:k]]]
        result[j, :len(chosen)] = chosen
    return result


def utilities(rr, k=10):
    rr = np.asarray(rr)
    return np.where(rr <= k, 1/np.log2(rr+1), 0.)


def metrics(rr):
    rr = np.asarray(rr)
    return {f'{m}{k}': float((rr <= k).mean() if m == 'H' else utilities(rr, k).mean())
            for k in (5, 10, 20, 50) for m in ('H', 'N')}


def select_weight(table, scope):
    # Explicit, frozen tie-break: N10, H10, closest to equal fusion, smaller weight.
    return max(table, key=lambda row: (row[scope]['N10'], row[scope]['H10'],
                                      -abs(row['weight']-.5), -row['weight']))['weight']


def contrast(a, b, repeats=2000, seed=20260919):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and len(a)
    dn = utilities(a)-utilities(b)
    dh = (a <= 10).astype(float)-(b <= 10).astype(float)
    rng = np.random.RandomState(seed)
    means = np.empty((repeats, 2))
    for i in range(repeats):
        ix = rng.randint(0, len(a), len(a))
        means[i] = dn[ix].mean(), dh[ix].mean()
    return dict(delta_N10=float(dn.mean()), delta_H10=float(dh.mean()),
                descriptive_95pct_N10=np.percentile(means[:, 0], [2.5, 97.5]).tolist(),
                descriptive_95pct_H10=np.percentile(means[:, 1], [2.5, 97.5]).tolist(),
                top10_gained=int(((a <= 10) & (b > 10)).sum()),
                top10_lost=int(((b <= 10) & (a > 10)).sum()))
