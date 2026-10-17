"""Target-free text leave-self-out and fixed history-score penalties.

The semantic prototype and score normalization are inherited from LIME-derived
scorers. This ablation adds no trainable parameters and makes no novelty claim.
leave_self_out implements residual prototypes directly, without an extra cap
at the original historical score. In exact arithmetic the self contribution
already increases its own cosine, so that cap would normally be redundant.
"""
import numpy as np

from experiment.phase20.lime_comparison_core import semantic_users, unit_rows

PAIR_ARMS = ('pair_raw', 'pair_loo', 'pair_raw_penalty', 'pair_loo_penalty')
CONTROL_ARMS = ('uniform', 'uniform_penalty')
ARMS = PAIR_ARMS + CONTROL_ARMS
FRESH_ARMS = tuple(a for a in ARMS if a != 'pair_raw')


def leave_self_out(histories, embeddings, reference):
    """Remove every occurrence of candidate i from its own text prototype.

    Only visible historical candidates are revised. Preserve the original
    full-catalog min/max transform, clip to [0, 1], and copy all nonhistorical
    scores exactly. Empty residual evidence receives support zero. No target,
    outcome, or GRAM score is an input.
    """
    embeddings = np.asarray(embeddings, np.float32)
    baseline = np.asarray(reference, np.float32)
    candidates = unit_rows(embeddings)
    raw = semantic_users(histories, embeddings) @ candidates.T
    low, span = raw.min(1), np.ptp(raw, axis=1)
    np.testing.assert_allclose(baseline, (raw-low[:, None])/(span[:, None]+1e-8),
                               atol=8e-6, rtol=3e-5)
    output = baseline.copy()
    for j, history in enumerate(histories):
        ids = np.asarray(history, np.int64)-1
        assert len(ids) and ids.min() >= 0 and ids.max() < len(embeddings)
        weights = np.exp(-.1*np.arange(len(ids)-1, -1, -1))
        weighted = embeddings[ids].astype(np.float64)*weights[:, None]
        for item in np.unique(ids):
            keep = ids != item
            if not keep.any() or span[j] <= 1e-7:
                value = 0.
            else:
                prototype = weighted[keep].sum(0)/weights[keep].sum()
                prototype /= np.linalg.norm(prototype)+1e-8
                cosine = prototype @ candidates[item].astype(np.float64)
                value = (cosine-float(low[j]))/(float(span[j])+1e-8)
            output[j, item] = np.clip(value, 0., 1.)
    return output


def arm_scores(sas, raw_text, loo_text, full_history, penalty=.05):
    sas, raw_text, loo_text = [np.asarray(a, np.float32) for a in (sas, raw_text, loo_text)]
    assert sas.ndim == 1 and sas.shape == raw_text.shape == loo_text.shape
    assert np.isfinite(np.stack([sas, raw_text, loo_text])).all()
    ids = np.unique(np.asarray(full_history, np.int64))-1
    assert len(ids) and ids.min() >= 0 and ids.max() < len(sas)
    result = dict(pair_raw=np.float32(.5)*sas + np.float32(.5)*raw_text,
                  pair_loo=np.float32(.5)*sas + np.float32(.5)*loo_text,
                  uniform=np.zeros_like(sas))
    for key in ('pair_raw', 'pair_loo', 'uniform'):
        value = result[key].copy()
        value[ids] -= np.float32(penalty)
        result[key+'_penalty'] = value
    return {key: result[key] for key in ARMS}


def tests():
    rng = np.random.default_rng(2150)
    embedding = unit_rows(rng.normal(size=(29, 7)).astype(np.float32))
    histories = [[1, 5, 8], [3], [2, 2, 9, 2], [7, 4]]
    raw = semantic_users(histories, embedding) @ unit_rows(embedding).T
    baseline = (raw-raw.min(1, keepdims=True))/(np.ptp(raw, axis=1)[:, None]+1e-8)
    revised = leave_self_out(histories, embedding, baseline)
    for j, history in enumerate(histories):
        mask = np.isin(np.arange(1, 30), history)
        np.testing.assert_array_equal(revised[j, ~mask], baseline[j, ~mask])
        weights = np.exp(-.1*np.arange(len(history)-1, -1, -1))
        for item in set(history):
            retained = [(k, x) for k, x in enumerate(history) if x != item]
            if retained:
                vector = sum(weights[k]*embedding[x-1].astype(float) for k, x in retained)
                vector /= sum(weights[k] for k, _ in retained)
                vector /= np.linalg.norm(vector)+1e-8
                expected = np.clip((vector @ unit_rows(embedding)[item-1]-raw[j].min())/
                                   (np.ptp(raw[j])+1e-8), 0., 1.)
            else:
                expected = 0.
            np.testing.assert_allclose(revised[j, item-1], expected, atol=3e-6)
    assert revised[1, 2] == 0.
    # Positive self contribution cannot reduce its own cosine in exact arithmetic.
    for j, history in enumerate(histories):
        ids = np.unique(history)-1
        assert np.all(revised[j, ids] <= baseline[j, ids]+3e-6)
    values = arm_scores(baseline[0], baseline[0], revised[0], [1, 29])
    mask = np.isin(np.arange(1, 30), [1, 29])
    for name in ('pair_raw', 'pair_loo', 'uniform'):
        np.testing.assert_array_equal(values[name][~mask], values[name+'_penalty'][~mask])
        np.testing.assert_allclose(values[name][mask]-values[name+'_penalty'][mask], .05, atol=3e-8)
    zero = arm_scores(baseline[0], baseline[0], revised[0], [1, 29], penalty=0.)
    for name in ('pair_raw', 'pair_loo', 'uniform'):
        np.testing.assert_array_equal(zero[name], zero[name+'_penalty'])
    # Independent branch probability check: historical/nonhistorical odds x exp(-1).
    p = np.exp(20*values['uniform_penalty'])
    np.testing.assert_allclose(p[0]/p[1], np.exp(-1), rtol=1e-6)
    return dict(state='PASSED', independent_residual_prototypes=True,
        duplicate_occurrences_removed=True, singleton_zero_support=True,
        nonhistorical_text_scores_exact=True, no_monotonic_cap=True,
        full_history_penalty_only=True, zero_penalty_exact=True,
        penalty_odds_formula=True, targets_absent_from_scoring_api=True)
