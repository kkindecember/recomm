"""Target-free decomposition of existing ItemCF edges by visible GRAM hints.

Catalog histories/hints use 1-based numeric IDs; adjacency rows/columns use
0-based offsets. ItemCF scores follow the LIME-Rec-derived implementation.
"""
import numpy as np
from experiment.phase21.candidate_text_source_core import normalize_rows

GRAPH_ARMS = ('cf_full', 'cf_prompt', 'cf_matched', 'cf_outside')


def graphs(model, items, hints):
    v = len(items)
    assert hints.shape[0] == v and hints.min() >= 1 and hints.max() <= v
    index = {item: j for j, item in enumerate(items)}
    assert len(index) == v and model['training_only'] and model['scoring_history_length'] == 20
    result = {arm: [] for arm in GRAPH_ARMS}
    counts, masses, overlap = [], [], []
    for i, item in enumerate(items):
        rows = model['neighbors'].get(item, [])
        ids = np.asarray([index[r['item_id']] for r in rows], dtype=np.int64)
        w = np.asarray([r['score'] for r in rows], dtype=np.float32)
        assert len(ids) == len(set(ids)) and np.isfinite(w).all() and (w > 0).all()
        inside = np.isin(ids+1, hints[i])
        n = int(inside.sum())
        top = np.lexsort((ids, -w))[:n]
        matched = w[top].copy()
        if n:
            matched *= float(w[inside].sum(dtype=np.float64)/matched.sum(dtype=np.float64))
        for arm, cols, weights in (
            ('cf_full', ids, w), ('cf_prompt', ids[inside], w[inside]),
            ('cf_matched', ids[top], matched), ('cf_outside', ids[~inside], w[~inside])):
            result[arm].append((cols.copy(), weights.copy()))
        assert len(top) == n
        np.testing.assert_allclose(matched.sum(dtype=np.float64), w[inside].sum(dtype=np.float64), atol=1e-7, rtol=2e-7)
        counts.append([len(ids), n]); masses.append([float(w.sum(dtype=np.float64)), float(w[inside].sum(dtype=np.float64))])
        overlap.append(len(set(ids[top]) & set(ids[inside])))
    counts, masses = np.asarray(counts), np.asarray(masses)
    nonempty = counts[:, 0] > 0
    stats = dict(catalog_items=v, cf_edges=int(counts[:, 0].sum()), prompt_cf_edges=int(counts[:, 1].sum()),
        nonempty_cf_sources=int(nonempty.sum()), nonempty_sources_without_prompt_cf=int((nonempty & (counts[:, 1] == 0)).sum()),
        mean_prompt_mass_fraction=float(np.mean(masses[nonempty, 1]/masses[nonempty, 0])),
        matched_edges_shared_with_prompt=int(sum(overlap)), matched_edge_count_equal=True,
        matched_per_source_mass_equal=True, source_graph_metadata={k: v for k, v in model.items() if k != 'neighbors'})
    return result, stats


def score(history, graph):
    result = np.zeros(len(graph), dtype=np.float32)
    window = history[-20:]
    for offset, item in enumerate(window, 1):
        assert 1 <= item <= len(graph)
        cols, weights = graph[item-1]
        result[cols] += (offset/len(window))*weights
    return result


def normalize(scores):
    return normalize_rows(scores[None])[0]


def save(path, graph):
    np.savez(path, indptr=np.r_[0, np.cumsum([len(cols) for cols, _ in graph])],
        indices=np.concatenate([cols for cols, _ in graph]), weights=np.concatenate([w for _, w in graph]))


def load(path):
    with np.load(path) as a:
        ptr, ids, weights = a['indptr'], a['indices'], a['weights']
    return [(ids[start:stop], weights[start:stop]) for start, stop in zip(ptr[:-1], ptr[1:])]


def tests():
    items = ['a', 'b', 'c', 'd', 'e']
    model = dict(training_only=True, scoring_history_length=20, neighbors={
        'a': [dict(item_id='b', score=.2), dict(item_id='c', score=.6), dict(item_id='e', score=.6)],
        'b': [dict(item_id='a', score=.3)], 'c': [dict(item_id='d', score=.4)]})
    hints = np.asarray([[2], [3], [4], [1], [1]])
    g, stats = graphs(model, items, hints)
    assert g['cf_prompt'][0][0].tolist() == [1]  # catches row/value off-by-one
    assert g['cf_matched'][0][0].tolist() == [2]  # tied .6 -> smaller catalog ID
    np.testing.assert_allclose(g['cf_matched'][0][1], [.2])
    assert not len(g['cf_prompt'][1][0]) and not len(g['cf_matched'][1][0])
    history = [5]*3+[1, 2, 1, 3]*6
    for arm, graph in g.items():
        actual = score(history, graph)
        expected = np.zeros(len(items), dtype=np.float64)
        for j, source in enumerate(history[-20:], 1):
            for col, w in zip(*graph[source-1]):
                expected[col] += j/min(len(history), 20)*float(w)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(score(history, g['cf_full']), score(history, g['cf_prompt'])+score(history, g['cf_outside']), atol=1e-6)
    np.testing.assert_array_equal(normalize(score([4, 5], g['cf_full'])), np.zeros(5))
    assert stats['prompt_cf_edges'] == 2 and stats['cf_edges'] == 5
    return dict(id_offsets_checked=True, target_free_interface=True, count_and_source_mass_matched=True,
        deterministic_ties=True, empty_rows_finite=True, independent_scalar_history_score=True,
        repeated_history_and_window_checked=True, raw_decomposition=True)
