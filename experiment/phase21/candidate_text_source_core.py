"""Matched candidate-text controls; all score functions are target-free."""
import numpy as np
from experiment.phase20.lime_comparison_core import semantic_users, unit_rows


def normalize_rows(scores):
    scores = np.asarray(scores, dtype=np.float32)
    assert scores.ndim == 2 and np.isfinite(scores).all()
    lo, hi = scores.min(1, keepdims=True), scores.max(1, keepdims=True)
    return (scores-lo)/(hi-lo+1e-8)


def semantic_scores(histories, history_embeddings, candidate_embeddings):
    """Same recency aggregation and candidate normalization as cached BGE."""
    return normalize_rows(semantic_users(histories, history_embeddings) @ unit_rows(candidate_embeddings).T)


def tests():
    rng = np.random.default_rng(12)
    full = unit_rows(rng.normal(size=(23, 8)).astype(np.float32))
    ids = unit_rows(rng.normal(size=(23, 8)).astype(np.float32))
    histories = [[1, 4, 2], [6], [3, 3, 5, 7]]
    for htable, ctable in ((full, full), (full, ids), (ids, ids)):
        actual = semantic_scores(histories, htable, ctable)
        raw = []
        for h in histories:
            w = np.exp(-.1*np.arange(len(h)-1, -1, -1)).astype(np.float32)
            u = sum(htable[i-1]*weight for i, weight in zip(h, w))/w.sum()
            u = u/(np.linalg.norm(u)+1e-8)
            raw.append([float(u @ (e/(np.linalg.norm(e)+1e-8))) for e in ctable])
        expected = np.asarray(raw)
        expected = (expected-expected.min(1, keepdims=True))/(np.ptp(expected, axis=1, keepdims=True)+1e-8)
        np.testing.assert_allclose(actual, expected, atol=4e-7)
    np.testing.assert_allclose(semantic_scores(histories, full, full), semantic_scores(histories, full.copy(), full.copy()))
    assert not np.allclose(semantic_scores(histories, full, ids), semantic_scores(histories, ids, ids))
    np.testing.assert_array_equal(normalize_rows(np.ones((2, 7))), np.zeros((2, 7)))
    return dict(independent_scalar_cosines=True, candidate_only_vs_both_distinct=True,
                same_representation_identity=True, constant_scores_finite=True,
                targets_absent_from_score_interface=True)
