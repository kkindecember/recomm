"""Frozen prefix lookahead with an exactly specified completed-item score.

This is a heuristic experiment, not an implementation of RGD's learned reward
model or its theoretical guarantee. No target is accepted by the search API.
"""
import hashlib

import numpy as np
import torch


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(-1, keepdims=True)) / np.maximum(x.std(-1, keepdims=True), 1e-6)


def rewards(cf_scores, frequencies, user, seed=20260919):
    """Catalog-wide statistics exclude padding; no held-out labels are used."""
    freq = np.asarray(frequencies)[1:]
    cf = np.asarray(cf_scores, dtype=np.float64)[:, 1:]
    calibrated = standardize(standardize(cf) - .5 * standardize(np.log1p(freq)))
    actual = np.tanh(calibrated.mean(0) / 2.)
    popular = np.tanh(standardize(np.log1p(freq)) / 2.)
    # Shuffle within fixed logarithmic frequency buckets, preserving each
    # bucket's score multiset and the coarse popularity association.
    buckets = np.floor(np.log2(np.maximum(freq, 1))).astype(np.int64)
    shuffled = actual.copy()
    rng = np.random.RandomState(int(hashlib.sha256(f'{seed}|{user}'.encode()).hexdigest()[:8], 16))
    permutation = np.arange(len(freq))
    for bucket in np.unique(buckets):
        indices = np.flatnonzero(buckets == bucket)
        permutation[indices] = rng.permutation(indices)
        shuffled[indices] = actual[permutation[indices]]
    result = {name: np.r_[0., value] for name, value in
              [('real_cf', actual), ('shuffled_cf', shuffled), ('popularity', popular)]}
    assert all(np.isfinite(x).all() and np.max(np.abs(x)) <= 1 for x in result.values())
    return result, permutation


class PrefixIndex:
    def __init__(self, paths, item_ids, eos=1):
        self.eos = eos
        self.nodes, self.lookup, self.children = [], {}, []
        node_items, item_refs = [], []
        self.path_to_item = {}
        for path, item in zip(paths, item_ids):
            path = tuple(map(int, path))
            assert path[0] == 0 and path[-1] == eos and eos not in path[:-1]
            assert path not in self.path_to_item, 'Ambiguous lexical item path'
            self.path_to_item[path] = int(item)
            parent = None
            for length in range(1, len(path) + 1):
                prefix = path[:length]
                if prefix not in self.lookup:
                    self.lookup[prefix] = len(self.nodes)
                    self.nodes.append(prefix)
                    self.children.append({})
                node = self.lookup[prefix]
                if parent is not None:
                    self.children[parent][path[length - 1]] = node
                node_items.append(node)
                item_refs.append(int(item))
                parent = node
        self.node_items = np.asarray(node_items)
        self.item_refs = np.asarray(item_refs)
        self.lengths = np.asarray([len(p) - int(p[-1] == eos) for p in self.nodes])
        self.root = self.lookup[(0,)]

    def potentials(self, item_rewards):
        best = np.full(len(self.nodes), -np.inf)
        np.maximum.at(best, self.node_items, np.asarray(item_rewards)[self.item_refs])
        assert np.isfinite(best).all()
        potential = self.lengths * best
        potential[self.root] = 0.
        return potential


class PrefixGuidance:
    """Add differences of prefix potentials AFTER native trie masking.

For an unfinished prefix p, Phi(p)=len(p)*max_descendant r(item).
Phi(start)=0. For a complete path, Phi=length_before_EOS*r(item),
matching transformers 4.26 BeamHypotheses' length denominator.
The accumulated bonus telescopes. The completed normalized score is
native_sequence_score + strength*r(item), not one reward per token.
Native length penalty and early stopping are retained; this remains a
heuristic beam search with no optimal-search guarantee.
"""
    def __init__(self, index, item_rewards, strength, trace=False):
        self.index = index
        self.values = index.potentials(item_rewards)
        self.strength = float(strength)
        self.calls = 0
        self.trace = [] if trace else None

    def __call__(self, input_ids, scores):
        self.calls += 1
        prefixes = input_ids.detach().cpu().tolist()
        if self.trace is not None:
            self.trace.append(prefixes)
        if self.strength == 0:
            return scores
        row_ids, tokens, changes = [], [], []
        for row, prefix in enumerate(prefixes):
            node = self.index.lookup.get(tuple(prefix))
            if node is None:
                # HF may pad the active set with impossible -inf branches when
                # fewer than beam_size continuations survive. Never revive them.
                assert torch.isneginf(scores[row]).all(), 'Unmasked noncatalog prefix'
                continue
            for token, child in self.index.children[node].items():
                row_ids.append(row)
                tokens.append(token)
                changes.append(self.strength * (self.values[child] - self.values[node]))
        if not row_ids:
            return scores
        row_ids = torch.tensor(row_ids, dtype=torch.long, device=scores.device)
        tokens = torch.tensor(tokens, dtype=torch.long, device=scores.device)
        change = torch.tensor(changes, dtype=scores.dtype, device=scores.device)
        assert torch.isfinite(scores[row_ids, tokens]).all(), 'Legal path masked unexpectedly'
        scores[row_ids, tokens] += change
        return scores


def rank_of(ids, scores, target):
    ids = np.asarray(ids)
    at = np.flatnonzero(ids == target)
    if not len(at):
        return 51
    order = np.argsort(-np.asarray(scores), kind='stable')
    return int(np.flatnonzero(order == at[0])[0]) + 1


def summarize_ranks(ranks):
    ranks = np.asarray(ranks)
    return {f'{name}{k}': float(np.mean((ranks <= k) if name == 'H' else
            (ranks <= k) / np.log2(ranks + 1))) for k in (5, 10, 20, 50) for name in ('N', 'H')}


def paired_ranks(current, reference):
    a, b = np.asarray(current), np.asarray(reference)
    na, nb = (a <= 10) / np.log2(a + 1), (b <= 10) / np.log2(b + 1)
    return dict(delta_N10=float(np.mean(na - nb)), delta_H10=float(np.mean((a <= 10).astype(float) - (b <= 10))),
                hit_gained=int(np.sum((a <= 10) & (b > 10))), hit_lost=int(np.sum((a > 10) & (b <= 10))),
                coverage_gained=int(np.sum((a <= 50) & (b > 50))), coverage_lost=int(np.sum((a > 50) & (b <= 50))))
