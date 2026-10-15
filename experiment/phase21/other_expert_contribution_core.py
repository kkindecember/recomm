"""Removal controls for external sequence and co-occurrence expert scores."""
import numpy as np

ARMS = ('without_sasrec', 'without_itemcf')
REMOVED = {'without_sasrec': 0, 'without_itemcf': 1}


def transform_scores(raw, arm, donors=None):
    scores = np.array(raw, dtype=np.float32, copy=True)
    assert scores.ndim == 3 and scores.shape[1] == 3
    scores[:, REMOVED[arm]] = 0.
    return scores


def fuse(gate, scores, lengths, history, arm):
    from scripts.evaluation.run_repeat_aware_gate import features
    removed = REMOVED[arm]
    x = features(scores, lengths, history).clone()
    x[:, [1 + removed, 4 + removed, 7 + removed, 10 + removed]] = 0.
    weights, penalty = gate(x)
    mask = weights.new_ones(3); mask[removed] = 0.
    weights = weights * mask
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    result = (weights[:, :, None] * scores).sum(1) - penalty[:, None] * history
    return result, x, weights, penalty
