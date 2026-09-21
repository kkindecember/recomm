"""Matched ordinary / hard-negative / cross-source ranking objectives.

Candidate construction and prediction never consume target labels. Labels are
used only for training losses and retrospective evaluation partitions.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('ordinary_ce', 'hard_negative', 'cross_source')
FEATURES = ('sasrec', 'itemcf', 'semantic', 'lime', 'in_gram', 'in_lime',
            'gram_reciprocal_rank', 'lime_minus_gram_top10', 'log_frequency', 'log_history_length')


def candidate_union(gram, lime, histories):
    ids = np.sort(np.concatenate((gram, lime), axis=1), axis=1)
    duplicate = np.concatenate((np.zeros((len(ids), 1), bool), ids[:, 1:] == ids[:, :-1]), axis=1)
    ids[duplicate] = 0
    for i, history in enumerate(histories):
        ids[i, np.isin(ids[i], history[-20:])] = 0
    valid = ids > 0
    ing = (ids[:, :, None] == gram[:, None, :]).any(2) & valid
    inl = (ids[:, :, None] == lime[:, None, :]).any(2) & valid
    return ids, ing, inl


def make_features(ids, ing, inl, gram, scores, fused, frequencies, lengths):
    safe = np.maximum(ids - 1, 0)
    rows = np.arange(len(ids))[:, None]
    expert = np.stack([s[rows, safe] for s in scores.transpose(1, 0, 2)], axis=2)
    fs = fused[rows, safe]
    match = (ids[:, :, None] == gram[:, None, :]) & ing[:, :, None]
    grank = np.where(ing, 1. / (match.argmax(2) + 1), 0.)
    gvalues = np.where(gram > 0, fused[rows, np.maximum(gram-1, 0)], -np.inf)
    boundary = np.sort(gvalues, axis=1)[:, -10]
    assert np.isfinite(boundary).all()
    x = np.concatenate((expert, fs[:, :, None], ing[:, :, None], inl[:, :, None],
        grank[:, :, None], (fs-boundary[:, None])[:, :, None],
        np.log1p(frequencies[ids])[:, :, None],
        np.broadcast_to(np.log1p(lengths)[:, None, None], (*ids.shape, 1))), axis=2).astype(np.float32)
    x[ids == 0] = 0
    prior = np.where(ids > 0, 20. * fs, 0.).astype(np.float32)
    assert x.shape[-1] == len(FEATURES) and np.isfinite(x).all() and np.isfinite(prior).all()
    return x, prior


def target_positions(ids, targets):
    hit = (ids == targets[:, None]) & (ids > 0)
    return np.where(hit.any(1), hit.argmax(1), -1)


def pair_masks(prior, valid, ing, positions, max_pairs=5):
    """Freeze top competitors under the base score; both arms get equal counts.

    For a GRAM-covered training target, cross-source negatives are external.
    For an external target, they are GRAM candidates. The control picks the same
    number of hardest negatives from the entire union, regardless of source.
    """
    assert (positions >= 0).all()
    neg = valid.copy()
    neg[np.arange(len(neg)), positions] = False
    target_g = ing[np.arange(len(ing)), positions]
    eligible = neg & np.where(target_g[:, None], ~ing, ing)
    counts = np.minimum(eligible.sum(1), max_pairs)
    masks = []
    for allowed in (neg, eligible):
        order = np.argsort(-np.where(allowed, prior, -np.inf), axis=1, kind='stable')
        chosen = np.zeros_like(allowed)
        np.put_along_axis(chosen, order, np.arange(prior.shape[1])[None] < counts[:, None], axis=1)
        assert not (chosen & ~allowed).any()
        masks.append(chosen)
    np.testing.assert_array_equal(masks[0].sum(1), masks[1].sum(1))
    return dict(hard_negative=masks[0], cross_source=masks[1])


class ResidualRanker(nn.Module):
    def __init__(self, n_features=len(FEATURES), hidden=32, bound=2.):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.bound = bound
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, prior):
        return prior + self.bound * torch.tanh(self.network(x).squeeze(-1))


def row_losses(scores, valid, positions, pairs=None, pair_weight=.25):
    safe = scores.masked_fill(~valid, -1.e9)
    ce = F.cross_entropy(safe, positions, reduction='none')
    if pairs is None:
        return ce
    positive = scores.gather(1, positions[:, None])
    extra = (F.softplus(scores-positive) * pairs).sum(1) / pairs.sum(1).clamp(min=1)
    return ce + pair_weight * extra


def independent_ranks(scores, ids, targets):
    """Full lexicographic sorting, independent of the counting rank routine."""
    ordered = np.lexsort((ids, -scores), axis=1)
    ranked = np.take_along_axis(ids, ordered, 1)
    finite = np.isfinite(np.take_along_axis(scores, ordered, 1))
    found = (ranked == targets[:, None]) & finite
    return np.where(found.any(1), found.argmax(1)+1, 2**30)
