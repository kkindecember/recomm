"""Train-only user neighborhoods and label-separated PP ranking utilities."""
import hashlib

import numpy as np
from scipy import sparse


def reconstruct_training(rows, max_history=20):
    sequences = {}
    for row in rows:
        user = row['user_id']
        history = list(row['history'])
        if not history or any(int(x) <= 0 for x in history + [row['target']]):
            raise ValueError('Invalid training interaction')
        if user not in sequences:
            if len(history) != 1:
                raise ValueError('Missing initial training transition')
            sequences[user] = history.copy()
        if sequences[user][-max_history:] != history:
            raise ValueError('Training transitions reordered or incomplete')
        sequences[user].append(int(row['target']))
    users = sorted(sequences)
    recent = [sorted(set(sequences[u][-max_history:])) for u in users]
    complete = [sorted(set(sequences[u])) for u in users]
    return users, sequences, recent, complete


def incidence(sets, num_items):
    lengths = np.asarray([len(x) for x in sets], dtype=np.int64)
    columns = np.asarray([x for row in sets for x in row], dtype=np.int64)
    if len(columns) == 0 or columns.min() <= 0 or columns.max() >= num_items:
        raise ValueError('Invalid catalog index')
    return sparse.csr_matrix((np.ones(len(columns), dtype=np.int32), columns,
                              np.r_[0, np.cumsum(lengths)]), shape=(len(sets), num_items))


def stable_top_neighbors(similarities, count, self_index):
    values = np.asarray(similarities, dtype=np.float64).copy()
    if len(values) <= count or not np.isfinite(values).all():
        raise ValueError('Insufficient or invalid neighborhood')
    values[self_index] = -1.0
    threshold = np.partition(values, len(values) - count)[-count]
    above = np.flatnonzero(values > threshold)
    equal = np.flatnonzero(values == threshold)[:count - len(above)]
    chosen = np.r_[above, equal]
    chosen = chosen[np.lexsort((chosen, -values[chosen]))]
    assert len(chosen) == count and self_index not in chosen
    return chosen, values[chosen]


def neighborhoods(bank, query_indices, count, block_size=128, progress=None):
    """The bank is in lexicographic user-ID order; ties follow that order."""
    sizes = np.asarray(bank.sum(axis=1)).ravel()
    neighbors, scores, positive_counts = [], [], []
    for start in range(0, len(query_indices), block_size):
        indices = np.asarray(query_indices[start:start + block_size])
        intersections = (bank[indices] @ bank.T).toarray().astype(np.float64)
        union = sizes[indices, None] + sizes[None, :] - intersections
        similarities = intersections / np.maximum(union, 1)
        for row, index in zip(similarities, indices):
            positive_counts.append(int((row > 0).sum()) - 1)
            chosen, values = stable_top_neighbors(row, count, int(index))
            neighbors.append(chosen)
            scores.append(values)
        if progress:
            progress(min(start + block_size, len(query_indices)), len(query_indices))
    return np.asarray(neighbors), np.asarray(scores), np.asarray(positive_counts)


def shuffled_users(users, seed):
    """A label-free deterministic derangement, kept fixed for all coefficients."""
    if len(users) < 2:
        raise ValueError('At least two users required')
    order = sorted(range(len(users)), key=lambda i: (
        hashlib.sha256(f'pp-shuffle:{seed}:{users[i]}'.encode()).digest(), users[i]))
    mapping = np.empty(len(users), dtype=np.int64)
    for source, donor in zip(order, order[1:] + order[:1]):
        mapping[source] = donor
    assert np.all(mapping != np.arange(len(users)))
    return mapping


def donor_neighbors(all_neighbors, donor_indices, query_indices, k):
    """Also exclude the actual query user from its shuffled donor's bank."""
    result = []
    for donor, query in zip(donor_indices, query_indices):
        row = all_neighbors[int(donor)]
        filtered = row[row != query][:k]
        if len(filtered) != k:
            raise ValueError('Need k+1 cached neighbors for self exclusion')
        result.append(filtered)
    return np.asarray(result)


def candidate_popularity(bank, neighbor_ids, candidate_ids, block_size=128, progress=None):
    """Score the actual query's candidate item IDs, never donor rank positions."""
    candidate_ids = np.asarray(candidate_ids)
    k = neighbor_ids.shape[1]
    result = np.empty(candidate_ids.shape, dtype=np.float64)
    for start in range(0, len(candidate_ids), block_size):
        end = min(start + block_size, len(candidate_ids))
        local = neighbor_ids[start:end]
        edges = sparse.csr_matrix((np.ones(local.size, dtype=np.int32),
                                    (np.repeat(np.arange(end-start), k), local.ravel())),
                                   shape=(end-start, bank.shape[0]))
        counts = edges @ bank
        picked = counts[np.arange(end-start)[:, None], candidate_ids[start:end]]
        if sparse.issparse(picked):
            picked = picked.toarray()
        result[start:end] = np.asarray(picked, dtype=np.float64) / k
        if progress:
            progress(end, len(candidate_ids))
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    return result


def standardize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-6)


def ranks_from_scores(scores, target_positions):
    order = np.argsort(-np.asarray(scores), axis=1, kind='stable')
    inverse = np.argsort(order, axis=1)
    positions = np.asarray(target_positions)
    ranks = inverse[np.arange(len(order)), np.maximum(positions, 0)] + 1
    return np.where(positions >= 0, ranks, order.shape[1]+1).astype(np.int64)


def ndcg10(ranks):
    ranks = np.asarray(ranks)
    return np.where(ranks <= 10, 1.0 / np.log2(ranks+1), 0.0)


def user_folds(users, seed, count=5):
    order = sorted(range(len(users)), key=lambda i: (
        hashlib.sha256(f'pp-fold:{seed}:{users[i]}'.encode()).digest(), users[i]))
    result = np.empty(len(users), dtype=np.int64)
    result[order] = np.arange(len(users)) % count
    return result


def choose_eta(rank_grid, indices, etas):
    values = np.asarray([ndcg10(r[indices]).mean() for r in rank_grid])
    best = values.max()
    # Sorted eta grid means ties (within numerical roundoff) favor less change.
    chosen = int(np.flatnonzero(values >= best - 1e-15)[0])
    return chosen, {'eta': float(etas[chosen]), 'development_ndcg10': values.tolist()}


def crossfit(rank_grid, folds, etas):
    output = np.empty(len(folds), dtype=np.int64)
    selected = []
    for fold in sorted(set(folds.tolist())):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        index, info = choose_eta(rank_grid, train, etas)
        output[held] = rank_grid[index][held]
        selected.append({'fold': int(fold), 'selection_users': len(train),
                         'evaluation_users': len(held), **info})
    _, full_selection = choose_eta(rank_grid, np.arange(len(folds)), etas)
    return output, selected, full_selection
