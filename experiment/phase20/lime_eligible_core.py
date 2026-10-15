"""User-specific eligibility for the next controlled decoding experiment.

This module is not installed into existing/frozen experiments. Excluded IDs
must be exactly the existing final-filter history, not all historical items.
"""
import numpy as np


class EligibleTrie:
    def __init__(self, index, excluded_items):
        self.original = index
        self.nodes, self.lookup = index.nodes, index.lookup
        self.root, self.eos = index.root, index.eos
        self.path_to_item = index.path_to_item
        self.node_items, self.item_refs = index.node_items, index.item_refs
        self.item_count = int(index.item_refs.max())
        excluded = np.unique(np.asarray(list(excluded_items), dtype=np.int64))
        assert ((excluded >= 1) & (excluded <= self.item_count)).all()
        self.eligible = np.ones(self.item_count, dtype=bool)
        self.eligible[excluded - 1] = False
        assert self.eligible.any(), 'No eligible catalog item remains'
        self.counts = np.bincount(self.node_items,
            weights=self.eligible[self.item_refs - 1].astype(float), minlength=len(self.nodes))
        self.children = [{token: child for token, child in cs.items() if self.counts[child] > 0}
                         for cs in index.children]

    def allowed_tokens(self, batch_id, sentence):
        prefix = tuple(sentence.tolist() if hasattr(sentence, 'tolist') else sentence)
        node = self.lookup.get(prefix)
        return [] if node is None else list(self.children[node])

    def conditional_logs(self, scores, temperature_scale=20.):
        """Aggregate only eligible item mass at every prefix, including shared ones."""
        scores = np.asarray(scores, dtype=np.float64)
        assert scores.shape == self.eligible.shape and np.isfinite(scores).all()
        weights = np.zeros(self.item_count)
        s = scores[self.eligible]
        weights[self.eligible] = np.exp(temperature_scale * (s - s.max()))
        masses = np.bincount(self.node_items, weights=weights[self.item_refs - 1],
                             minlength=len(self.nodes))
        logs = {}
        for node, cs in enumerate(self.children):
            if cs:
                assert masses[node] > 0 and all(masses[child] > 0 for child in cs.values())
                logs[self.nodes[node]] = {t: float(np.log(masses[c] / masses[node])) for t, c in cs.items()}
        return logs

    def condition_original_logs(self, original_logs):
        """Plain filtering control: mask illegal children and renormalize locally.

Unlike conditional_logs(), this keeps the original expert mass associated with
excluded descendants hidden inside still-eligible shared child prefixes.
"""
        logs = {}
        for node, cs in enumerate(self.children):
            if cs:
                prefix = self.nodes[node]
                values = np.asarray([original_logs[prefix][t] for t in cs], dtype=np.float64)
                normalizer = np.logaddexp.reduce(values)
                logs[prefix] = dict(zip(cs, (values - normalizer).tolist()))
        return logs
