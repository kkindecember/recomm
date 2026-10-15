"""Bounded score-correspondence, order-sensitivity and CF-strength controls."""
import hashlib

import numpy as np

ARMS = ('sasrec_misassigned', 'itemcf_misassigned', 'sasrec_order', 'itemcf_flat')
CHANNEL = dict(sasrec_misassigned=0, itemcf_misassigned=1, sasrec_order=0, itemcf_flat=1)


def transform_scores(raw, arm, donors):
    assert arm in ARMS
    scores = np.array(raw, dtype=np.float32, copy=True)
    if arm.endswith('_misassigned'):
        c = CHANNEL[arm]
        scores[:, c] = scores[:, c, donors]
    return scores


def fuse(gate, scores, lengths, history, arm):
    from scripts.evaluation.run_repeat_aware_gate import features
    assert arm in ARMS
    x = features(scores, lengths, history)
    weights, penalty = gate(x)
    return (weights[:, :, None] * scores).sum(1) - penalty[:, None] * history, x, weights, penalty


def shuffle_older_history(history, user, domain, width=50):
    """Keep the scoring window, multiset and latest event; change older order only.

    No target is accepted. This is sensitivity of a frozen sequence model, not a
    retrained order-free model or a separation of recency from transitions.
    """
    tail = list(history[-width:])
    if len(tail) <= 2:
        return tail
    older = tail[:-1]
    positions = sorted(range(len(older)), key=lambda j: hashlib.sha256(
        f'20260923|sasrec-order|{domain}|{user}|{j}'.encode()).digest())
    shuffled = [older[j] for j in positions]
    if shuffled == older and len(set(older)) > 1:
        shuffled = older[1:] + older[:1]
    return shuffled + tail[-1:]


def flat_neighbor_scores(values):
    """Remove within-source strength differences, preserving total source mass."""
    a = np.asarray(values, dtype=np.float32)
    assert a.ndim == 1 and len(a) and np.isfinite(a).all() and (a > 0).all()
    return np.full_like(a, np.mean(a, dtype=np.float64))


class ScoreOverride:
    """Map original source-row indices to the allowed subset's new channel."""
    def __init__(self, original, source_indices, replacement, channel):
        self.original, self.replacement, self.channel = original, replacement, channel
        self.shape = original.shape
        self.lookup = np.full(original.shape[0], -1, dtype=np.int64)
        source_indices = np.asarray(source_indices)
        assert len(np.unique(source_indices)) == len(source_indices) == len(replacement)
        assert replacement.shape[1] == original.shape[-1]
        self.lookup[source_indices] = np.arange(len(source_indices))

    def __getitem__(self, indices):
        local = self.lookup[indices]
        assert (local >= 0).all(), 'Unprepared source row requested'
        value = np.array(self.original[indices], dtype=np.float32, copy=True)
        value[:, self.channel] = self.replacement[local]
        return value
