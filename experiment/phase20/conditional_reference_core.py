"""Label-free matched reference construction for the Stage20 Toys screen.

All functions used to build neighborhoods accept histories, never targets.
The three learned arms share a six-input, 129-parameter residual model.
"""
import hashlib

import numpy as np
import torch
from torch import nn


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(-1, keepdims=True)) / np.maximum(
        values.std(-1, keepdims=True), 1e-6)


def ranks(scores, positions):
    order = np.argsort(-scores, axis=1, kind='stable')
    inverse = np.argsort(order, axis=1)
    p = np.asarray(positions)
    return np.where(p >= 0, inverse[np.arange(len(p)), np.maximum(p, 0)] + 1, 51)


def category_vectors(table, vocabulary_size):
    """Catalog-only category IDF; ubiquitous categories have weight zero."""
    table = np.asarray(table, dtype=np.int64)
    matrix = np.zeros((len(table), vocabulary_size), dtype=np.float64)
    for i, row in enumerate(table):
        ids = np.unique(row[row > 0])
        matrix[i, ids] = 1.
    df = matrix[1:].sum(0)
    weights = np.log(max(len(table) - 1, 1) / np.maximum(df, 1))
    weights[0] = 0
    matrix *= weights
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix.astype(np.float32), weights


def history_features(histories, item_categories, frequencies, q1):
    semantic = np.zeros((len(histories), item_categories.shape[1]), dtype=np.float32)
    nuisance = np.zeros((len(histories), 4), dtype=np.float64)
    lengths = np.zeros(len(histories), dtype=np.int64)
    for i, history in enumerate(histories):
        ids = np.asarray(history, dtype=np.int64)
        if not 0 < len(ids) <= 20 or (ids <= 0).any():
            raise ValueError('Invalid visible history')
        semantic[i] = item_categories[ids].mean(0)
        f = frequencies[ids]
        logf = np.log1p(f)
        nuisance[i] = [np.log1p(len(ids)), logf.mean(), logf.std(), (f <= q1).mean()]
        lengths[i] = len(ids)
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
    bins = np.minimum((lengths - 1) // 5, 2)
    return semantic, nuisance, bins


def stable_nearest(distance, count):
    """Ascending distance, then bank order, including ties at the boundary."""
    if np.isfinite(distance).sum() < count:
        raise ValueError('Insufficient non-self reference histories')
    threshold = np.partition(distance, count - 1)[count - 1]
    chosen = np.r_[np.flatnonzero(distance < threshold),
                   np.flatnonzero(distance == threshold)[:count - int((distance < threshold).sum())]]
    return chosen[np.lexsort((chosen, distance[chosen]))]


def key64(value):
    return np.uint64(int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little'))


def visible_key(user, history):
    """Never use historical sample_key: it contains the next target item ID."""
    return user + '|' + ','.join(str(int(item)) for item in history)


def priorities(bank_keys, query_key):
    # SplitMix64 is only a deterministic tie breaker / reference sampler.
    with np.errstate(over='ignore'):
        z = bank_keys ^ query_key
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        return z ^ (z >> np.uint64(31))


def matched_neighbors(query_users, query_keys, query_features, bank_users, bank_features,
                      neighbors=30, block_width=10, batch_size=64, seed=20260916,
                      progress=None):
    """C and D take one reference from each identical nuisance-distance block.

    Hard-match visible-history length bin. Within it, take the 300 nearest
    histories in four standardized nuisance features, exclude the query user,
    and split the distance ordering into 30 blocks of 10. C maximizes category
    cosine within each block; D samples without using categories. Ties use the
    same label-free priorities. Matching is approximate, not causal balancing.
    """
    qs, qn, qb = query_features
    bs, bn, bb = bank_features
    scale = np.maximum(bn.std(0), 1e-6)
    center = bn.mean(0)
    qn, bn = (qn - center) / scale, (bn - center) / scale
    n = len(query_users)
    semantic_ids = np.empty((n, neighbors), dtype=np.int32)
    simple_ids = np.empty_like(semantic_ids)
    diagnostics = np.empty((n, 12), dtype=np.float64)
    lookup = {u: i for i, u in enumerate(bank_users)}
    bank_keys = np.asarray([key64(f'{seed}|bank|{u}') for u in bank_users], dtype=np.uint64)
    pools = {b: np.flatnonzero(bb == b) for b in range(3)}
    total = neighbors * block_width
    completed = 0
    for b, pool in pools.items():
        queries = np.flatnonzero(qb == b)
        if not len(queries):
            continue
        if len(pool) <= total:
            raise ValueError('Reference length stratum too small')
        pool_vectors = bn[pool]
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            distances = ((qn[batch] ** 2).sum(1)[:, None]
                         + (pool_vectors ** 2).sum(1)[None, :]
                         - 2 * qn[batch] @ pool_vectors.T)
            distances = np.maximum(distances, 0)
            for local, row in enumerate(batch):
                if query_users[row] in lookup:
                    distances[local, pool == lookup[query_users[row]]] = np.inf
                near = stable_nearest(distances[local], total)
                blocks = pool[near].reshape(neighbors, block_width)
                tie = priorities(bank_keys[blocks], key64(f'{seed}|query|{query_keys[row]}'))
                similarity = np.einsum('bkd,d->bk', bs[blocks], qs[row], optimize=False)
                d_slots = np.argmin(tie, axis=1)
                max_sim = similarity.max(1, keepdims=True)
                masked_tie = np.where(similarity == max_sim, tie, np.iinfo(np.uint64).max)
                c_slots = np.argmin(masked_tie, axis=1)
                c_ids = blocks[np.arange(neighbors), c_slots]
                d_ids = blocks[np.arange(neighbors), d_slots]
                semantic_ids[row], simple_ids[row] = c_ids, d_ids
                c_gap = np.abs(bn[c_ids] - qn[row]).mean(0)
                d_gap = np.abs(bn[d_ids] - qn[row]).mean(0)
                diagnostics[row] = [similarity[np.arange(neighbors), c_slots].mean(),
                                    similarity[np.arange(neighbors), d_slots].mean(),
                                    np.mean(c_ids == d_ids), float(np.linalg.norm(qs[row]) == 0),
                                    *c_gap, *d_gap]
                assert len(set(c_ids)) == len(set(d_ids)) == neighbors
                assert all(bank_users[j] != query_users[row] for j in np.r_[c_ids, d_ids])
            completed += len(batch)
            if progress:
                progress(completed, n)
    assert np.isfinite(diagnostics).all()
    return semantic_ids, simple_ids, diagnostics


def background_scores(neighbor_ids, bank_vectors, item_vectors, candidates, scale,
                      batch_size=128, progress=None):
    result = np.empty(candidates.shape, dtype=np.float64)
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        # Do not renormalize the mean: exact mean of CF logits is required.
        background = bank_vectors[neighbor_ids[start:stop]].mean(1, dtype=np.float64)
        result[start:stop] = scale * np.einsum('bd,bkd->bk', background,
                                               item_vectors[candidates[start:stop]], optimize=False)
        if progress:
            progress(stop, len(candidates))
    return result


def frozen_fusion(g, cf, frequencies, positions, base_rank, q1):
    """Canonical tie order is the historical residual order, not CF order."""
    order = np.argsort(-g, axis=1, kind='stable')
    inverse = np.argsort(order, axis=1)
    take = lambda a: np.take_along_axis(a, order, 1)
    g, cf, frequencies, base_rank = [take(a) for a in (g, cf, frequencies, base_rank)]
    p = np.where(positions >= 0, inverse[np.arange(len(g)), np.maximum(positions, 0)], -1)
    gz, cz, pz = standardize(g), standardize(cf), standardize(np.log1p(frequencies))
    adjusted = standardize(cz - .5 * pz)
    reliability = 1. - (frequencies[:, :10] <= q1).mean(1)
    score = gz + reliability[:, None] * adjusted
    common = np.stack([gz, adjusted, pz, base_rank,
                       np.broadcast_to(reliability[:, None], score.shape)], axis=-1).astype(np.float32)
    return score, common, p, order, reliability


def arm_features(common, cf, order, background=None):
    signal = cf if background is None else cf - background
    signal = standardize(np.take_along_axis(signal, order, axis=1)).astype(np.float32)
    return np.concatenate([common, signal[:, :, None]], axis=-1)


class Correction(nn.Module):
    def __init__(self, bound=.5):
        super().__init__()
        self.bound = float(bound)
        self.network = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.bound * torch.tanh(self.network(features).squeeze(-1))
