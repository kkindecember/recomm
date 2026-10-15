"""Matched real / removed / misassigned semantic-score contribution controls."""
import hashlib

import numpy as np
import torch

ARMS = ('real', 'without_text', 'shuffled')
SEMANTIC_FEATURE_COLUMNS = (3, 6, 9, 12)


def semantic_donors(item_ids, counts, domain, seed=20260923):
    """Fixed derangement inside training-frequency buckets, without targets."""
    counts = np.asarray(counts)
    assert len(item_ids) == len(counts) and (counts >= 0).all()
    buckets = np.floor(np.log2(np.maximum(counts, 1))).astype(int)
    donors = np.arange(len(item_ids))
    for bucket in np.unique(buckets):
        indices = np.flatnonzero(buckets == bucket).tolist()
        indices.sort(key=lambda i: hashlib.sha256(
            f'{seed}|semantic-control|{domain}|{item_ids[i]}'.encode()).digest())
        if len(indices) > 1:
            donors[indices] = np.roll(indices, 1)
    assert np.array_equal(np.sort(donors), np.arange(len(donors)))
    assert np.array_equal(buckets, buckets[donors])
    return donors, dict(changed=int((donors != np.arange(len(donors))).sum()),
                        total=len(donors), singleton_buckets=int(sum((buckets == b).sum() == 1 for b in np.unique(buckets))))


def transform_scores(raw, arm, donors):
    assert arm in ARMS
    scores = np.array(raw, dtype=np.float32, copy=True)
    assert scores.ndim == 3 and scores.shape[1] == 3
    if arm == 'without_text':
        scores[:, 2] = 0.
    elif arm == 'shuffled':
        scores[:, 2] = scores[:, 2, donors]
    return scores


def gate_features(scores, lengths, history, arm):
    from scripts.evaluation.run_repeat_aware_gate import features
    x = features(scores, lengths, history)
    if arm == 'without_text':
        # Remove the score summaries too: no residual semantic-information path.
        x = x.clone()
        x[:, SEMANTIC_FEATURE_COLUMNS] = 0.
    return x


def fuse(gate, scores, lengths, history, arm):
    x = gate_features(scores, lengths, history, arm)
    weights, penalty = gate(x)
    if arm == 'without_text':
        weights = weights * weights.new_tensor([1., 1., 0.])
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    result = (weights[:, :, None] * scores).sum(1) - penalty[:, None] * history
    return result, x, weights, penalty
