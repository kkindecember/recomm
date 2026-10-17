"""Repeat/explore leaf mixtures; shared training and beam-search semantics.

Expert scoring is inherited from LIME-Rec and probability guidance from the
project's AGRec-inspired decoder. No evaluation-time history filtering.
"""
import math
import numpy as np
import torch


def consumed_mask(n, history):
    ids = np.asarray(sorted(set(history)), dtype=np.int64)
    assert np.all((ids >= 1) & (ids <= n))
    mask = np.zeros(n, bool)
    mask[ids - 1] = True
    return mask


def components(scores, history, tau=20.):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 1 and np.isfinite(scores).all()
    seen = consumed_mask(len(scores), history)
    weight = np.exp(tau * (scores - scores.max()))
    result = np.zeros((2, len(scores)), np.float64)
    for group, mask in enumerate((~seen, seen)):
        if mask.any():
            # Separate shifts prevent underflow for a low-scoring group.
            z = tau * (scores[mask] - scores[mask].max())
            result[group, mask] = np.exp(z) / np.exp(z).sum()
    return result, float(weight[seen].sum() / weight.sum())


def item_mass(scores, history, rho, tau=20.):
    assert 0 <= rho <= 1
    groups, _ = components(scores, history, tau)
    if groups[0].sum() == 0:
        return groups[1]
    if groups[1].sum() == 0:
        return groups[0]
    return (1-rho)*groups[0] + rho*groups[1]


def node_masses(index, leaves):
    return np.bincount(index.node_items, weights=leaves[index.item_refs-1],
                       minlength=len(index.nodes))


def conditional_logs(index, scores, history, rho, tau=20.):
    masses = node_masses(index, item_mass(scores, history, rho, tau))
    return {index.nodes[node]: (None if masses[node] <= 0 else {
        token: math.log(masses[child]/masses[node]) if masses[child] > 0 else -math.inf
        for token, child in children.items()})
        for node, children in enumerate(index.children) if children}


def novel_counts(index, history):
    return node_masses(index, (~consumed_mask(int(index.item_refs.max()), history)).astype(float))


def path_components(index, scores, history, path, depth):
    """Log probabilities of target parent/child in each normalized leaf group.

    Final dimension is explore/repeat. Padded steps are benign zero logs and
    excluded with valid. Neither target nor path changes the leaf distribution.
    """
    groups, mass = components(scores, history)
    nodes = [node_masses(index, g) for g in groups]
    parent, child = np.zeros((depth, 2)), np.zeros((depth, 2))
    valid = np.zeros(depth, bool)
    for d in range(1, len(path)):
        a, b = index.lookup[tuple(path[:d])], index.lookup[tuple(path[:d+1])]
        for group in (0, 1):
            parent[d-1, group] = math.log(nodes[group][a]) if nodes[group][a] > 0 else -math.inf
            child[d-1, group] = math.log(nodes[group][b]) if nodes[group][b] > 0 else -math.inf
        valid[d-1] = True
    return parent, child, valid, mass


def path_logs(parent, child, theta):
    """Differentiable q_rho conditionals, with safe zero-support fallback."""
    lg = torch.stack((torch.nn.functional.logsigmoid(-theta),
                      torch.nn.functional.logsigmoid(theta)))
    supported = torch.isfinite(parent).any(-1)
    # logsumexp([-inf,-inf]) has undefined gradients. Mask before reduction.
    ps = torch.where(supported[..., None], parent, torch.zeros_like(parent))
    csupport = torch.isfinite(child).any(-1)
    cs = torch.where(csupport[..., None], child, torch.zeros_like(child))
    p = torch.logsumexp(ps + lg, -1)
    c = torch.logsumexp(cs + lg, -1)
    q = torch.where(csupport, c-p, torch.full_like(p, -torch.inf))
    return q, supported


def training_nll(native, legal, parent, child, valid, theta, alpha=.3):
    q, supported = path_logs(parent, child, theta)
    mixed = torch.logaddexp(native+math.log1p(-alpha), legal+q+math.log(alpha))
    mixed = torch.where(supported, mixed, native)
    return -(torch.where(valid, mixed, torch.zeros_like(mixed)).sum(-1) / valid.sum(-1)).mean()


class SplitGuide:
    """Keep native legal mass, skip expert mixture on zero-support prefixes."""
    def __init__(self, index, expert_logs=None, alpha=0., counts=None):
        assert 0 <= alpha < 1
        self.index, self.expert_logs, self.alpha = index, expert_logs, float(alpha)
        self.counts = counts
        self.trace, self.calls = {}, 0

    def __call__(self, input_ids, scores):
        self.calls += 1
        prefixes = [tuple(p) for p in input_ids.detach().cpu().tolist()]
        row_ids, tokens, qlogs, rows = [], [], [], []
        for row, prefix in enumerate(prefixes):
            node = self.index.lookup.get(prefix)
            if node is None:
                assert torch.isneginf(scores[row]).all()
                continue
            children = self.index.children[node]
            allowed = list(children)
            if self.counts is not None:
                forbidden = [t for t, c in children.items() if self.counts[c] == 0]
                if forbidden:
                    scores[row, forbidden] = -torch.inf
                allowed = [t for t, c in children.items() if self.counts[c] > 0]
            if not allowed:
                continue
            start = len(row_ids)
            q = self.expert_logs[prefix] if self.alpha else None
            for token in allowed:
                row_ids.append(row); tokens.append(token)
                qlogs.append(q[token] if q is not None else -math.inf)
            rows.append((row, prefix, start, len(row_ids), q is not None))
        if not row_ids:
            return scores
        rr = torch.tensor(row_ids, device=scores.device, dtype=torch.long)
        tt = torch.tensor(tokens, device=scores.device, dtype=torch.long)
        native = scores[rr, tt]
        assert torch.isfinite(native).all()
        masses = torch.logsumexp(scores, -1)
        cn, cm = native.detach().cpu().double().numpy(), masses.detach().cpu().double().numpy()
        enabled = np.zeros(len(rr), bool)
        for row, prefix, start, stop, active in rows:
            if prefix not in self.trace:
                self.trace[prefix] = (tokens[start:stop], cn[start:stop].copy(), float(cm[row]))
            enabled[start:stop] = active
        if self.alpha and enabled.any():
            use = torch.from_numpy(enabled).to(scores.device)
            q = torch.tensor(qlogs, device=scores.device, dtype=scores.dtype)
            mixed = torch.logaddexp(native+math.log1p(-self.alpha),
                                   q+masses[rr]+math.log(self.alpha))
            scores[rr[use], tt[use]] = mixed[use]
        return scores

    def rescore(self, paths, expert_logs=None, alpha=0.):
        values = []
        for path in paths:
            total = 0.
            for d in range(1, len(path)):
                prefix, token = tuple(path[:d]), path[d]
                tokens, native, mass = self.trace[prefix]
                value = native[tokens.index(token)]
                q = expert_logs[prefix] if alpha else None
                if q is not None:
                    value = np.logaddexp(math.log1p(-alpha)+value,
                                         math.log(alpha)+mass+q[token])
                total += float(value)
            values.append(total/(len(path)-1))
        return np.asarray(values)


def ordered(record):
    ids, scores = np.asarray(record['ids']), np.asarray(record['scores'])
    assert len(ids) == 50 and len(set(ids.tolist())) == 50 and np.isfinite(scores).all()
    return ids[np.lexsort((ids, -scores))].tolist()


def rank(record, target):
    items = ordered(record)
    return items.index(int(target))+1 if int(target) in items else 51
