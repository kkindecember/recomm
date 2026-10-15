"""Bounded prefix-dependent expert weighting; no labels or model fitting.

Keep score-level fusion followed by item-to-token aggregation unchanged.
Local JS divergence from the descendant-count prior is a heuristic for
discrimination, NOT an estimate of expert accuracy or calibrated confidence.
"""
import numpy as np


class Descendants:
    def __init__(self, index):
        self.index = index
        order = np.argsort(index.node_items, kind='stable')
        self.items = index.item_refs[order] - 1
        self.counts = np.bincount(index.node_items, minlength=len(index.nodes))
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.groups = {}

    def ids(self, node):
        return self.items[self.offsets[node]:self.offsets[node + 1]]

    def children(self, node):
        if node not in self.groups:
            children = self.index.children[node]
            tokens = list(children)
            counts = self.counts[list(children.values())]
            ids = np.concatenate([self.ids(c) for c in children.values()])
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            assert len(ids) == self.counts[node] and len(np.unique(ids)) == len(ids)
            self.groups[node] = (tokens, ids, starts, counts / counts.sum())
        return self.groups[node]


def js_divergence(probabilities, reference):
    p = np.asarray(probabilities, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    m = (p + r) / 2
    # All probabilities come from finite score softmaxes and positive counts.
    assert (p > 0).all() and (r > 0).all()
    result = .5 * (np.sum(p * np.log(p / m), axis=-1)
                   + np.sum(r * np.log(r / m), axis=-1)) / np.log(2.)
    return np.clip(result, 0., 1.)


class LogView:
    def __init__(self, provider, calibrated):
        self.provider, self.calibrated = provider, calibrated

    def __getitem__(self, prefix):
        return self.provider.logs(prefix)[int(self.calibrated)]


class PrefixExperts:
    def __init__(self, structure, scores, prior, history, penalty,
                 coefficient=1., scale=20.):
        self.structure, self.index = structure, structure.index
        self.scores = np.asarray(scores, dtype=np.float32)
        self.prior = np.asarray(prior, dtype=np.float64)
        assert self.scores.ndim == 2 and self.scores.shape[0] == len(self.prior) == 3
        assert np.isfinite(self.scores).all() and (self.prior >= 0).all()
        assert abs(self.prior.sum() - 1) < 1e-6 and 0 <= coefficient <= 1
        self.prior = self.prior / self.prior.sum()
        self.coefficient, self.scale = float(coefficient), float(scale)
        self.penalty = np.float32(penalty)
        self.seen = np.zeros(self.scores.shape[1], dtype=bool)
        self.seen[np.asarray(history, dtype=np.int64) - 1] = True
        values = np.exp(self.scale * (self.scores.astype(np.float64)
                                     - self.scores.max(axis=1)[:, None]))
        self.masses = np.stack([np.bincount(self.index.node_items,
            weights=row[self.index.item_refs - 1], minlength=len(self.index.nodes)) for row in values])
        self.cache, self.weight_cache, self.evidence_cache = {}, {}, {}
        self.views = {False: LogView(self, False), True: LogView(self, True)}

    def weights(self, prefix):
        node = self.index.lookup[prefix]
        if node not in self.weight_cache:
            children = list(self.index.children[node].values())
            assert children
            q = self.masses[:, children] / self.masses[:, node, None]
            counts = self.structure.counts[children]
            evidence = js_divergence(q, counts / counts.sum())
            raw = self.prior * (1 + self.coefficient * evidence)
            self.weight_cache[node] = raw / raw.sum()
            self.evidence_cache[node] = evidence
        return self.weight_cache[node]

    def logs(self, prefix):
        node = self.index.lookup[prefix]
        if node in self.cache:
            return self.cache[node]
        tokens, ids, starts, _ = self.structure.children(node)
        if len(tokens) == 1:
            result = ({tokens[0]: 0.}, {tokens[0]: 0.})
        else:
            weights = self.weights(prefix).astype(np.float32)
            # Three FP32 terms in the original expert order (SASRec, ItemCF, BGE).
            fused = (weights[:, None] * self.scores[:, ids]).sum(axis=0)
            calibrated = fused.copy()
            calibrated[self.seen[ids]] -= self.penalty
            values = []
            for scores in (fused, calibrated):
                scores = scores.astype(np.float64)
                masses = np.add.reduceat(np.exp(self.scale * (scores - scores.max())), starts)
                values.append(dict(zip(tokens, np.log(masses / masses.sum()).tolist())))
            result = tuple(values)
        self.cache[node] = result
        return result

    def diagnostics(self):
        if not self.weight_cache:
            return dict(branch_nodes=0, max_weight_change=0.)
        ws = np.stack(list(self.weight_cache.values()))
        es = np.stack(list(self.evidence_cache.values()))
        return dict(branch_nodes=len(ws), max_weight_change=float(abs(ws - self.prior).max()),
                    mean_absolute_weight_change=float(abs(ws - self.prior).mean()),
                    mean_evidence=es.mean(axis=0).tolist(),
                    max_evidence=float(es.max()), prior=self.prior.tolist())
