"""Frozen expert probability guidance on the original GRAM lexical trie.

AGRec-inspired probability mixing, not an AGRec reproduction. Preserve the
native probability mass of legal tokens so alpha=0 is exactly original GRAM.
"""
import math

import numpy as np
import torch

from experiment.phase20.guided_decoding_core import PrefixIndex


class ExpertTrie(PrefixIndex):
    def conditional_logs(self, scores, temperature_scale=20.):
        scores = np.asarray(scores, dtype=np.float64)
        assert scores.ndim == 1 and np.isfinite(scores).all()
        weights = np.exp(temperature_scale * (scores - scores.max()))
        masses = np.bincount(self.node_items, weights=weights[self.item_refs-1],
                             minlength=len(self.nodes))
        assert np.all(masses > 0)
        logs = {}
        for node, children in enumerate(self.children):
            if children:
                values = {token: float(np.log(masses[child]/masses[node]))
                          for token, child in children.items()}
                assert abs(sum(math.exp(v) for v in values.values()) - 1.) < 1e-10
                logs[self.nodes[node]] = values
        return logs


def mixed_log_values(native, legal_log_mass, expert_log_prob, alpha):
    native = np.asarray(native, dtype=np.float64)
    if alpha == 0:
        return native.copy()
    return np.logaddexp(math.log1p(-alpha) + native,
                        math.log(alpha) + legal_log_mass + np.asarray(expert_log_prob))


class ProbabilityGuide:
    """Called AFTER native trie masking and BEFORE cumulative beam scores.

Let m be native total legal-token mass, and p be its conditional distribution.
Return log(m * ((1-alpha)*p + alpha*q)). Keeping m avoids silently changing
the baseline by renormalizing each parent's legal token probabilities.
The trace permits exact same-rule rescoring of native completed paths.
"""
    def __init__(self, index, expert_logs=None, alpha=0.):
        assert 0 <= alpha < 1
        self.index, self.expert_logs, self.alpha = index, expert_logs, float(alpha)
        self.trace, self.calls = {}, 0

    def __call__(self, input_ids, scores):
        self.calls += 1
        prefixes = [tuple(p) for p in input_ids.detach().cpu().tolist()]
        row_ids, tokens, qlogs, rows = [], [], [], []
        for row, prefix in enumerate(prefixes):
            node = self.index.lookup.get(prefix)
            if node is None:
                assert torch.isneginf(scores[row]).all(), 'Unexpected unmasked prefix'
                continue
            children = self.index.children[node]
            start = len(row_ids)
            for token in children:
                row_ids.append(row); tokens.append(token)
                if self.alpha:
                    qlogs.append(self.expert_logs[prefix][token])
            rows.append((row, prefix, start, len(row_ids)))
        if not row_ids:
            return scores
        rr = torch.tensor(row_ids, device=scores.device, dtype=torch.long)
        tt = torch.tensor(tokens, device=scores.device, dtype=torch.long)
        native = scores[rr, tt]
        assert torch.isfinite(native).all()
        masses = torch.logsumexp(scores, dim=-1)
        cpu_native = native.detach().cpu().double().numpy()
        cpu_masses = masses.detach().cpu().double().numpy()
        for row, prefix, start, stop in rows:
            if prefix not in self.trace:
                self.trace[prefix] = (tokens[start:stop], cpu_native[start:stop].copy(), float(cpu_masses[row]))
        if self.alpha:
            q = torch.tensor(qlogs, device=scores.device, dtype=scores.dtype)
            scores[rr, tt] = torch.logaddexp(native + math.log1p(-self.alpha),
                q + masses[rr] + math.log(self.alpha))
        return scores

    def rescore(self, paths, expert_logs=None, alpha=0.):
        values = []
        for path in paths:
            total = 0.
            for step in range(1, len(path)):
                prefix, token = tuple(path[:step]), int(path[step])
                tokens, native, mass = self.trace[prefix]
                position = tokens.index(token)
                q = 0. if alpha == 0 else expert_logs[prefix][token]
                total += float(mixed_log_values(native[position], mass, q, alpha))
            # Transformers 4.26 BeamHypotheses scores the sequence before EOS,
            # including its decoder start token, with length_penalty=1.
            values.append(total/(len(path)-1))
        return np.asarray(values)


def ordered(ids, scores, history=()):
    ids, scores = np.asarray(ids), np.asarray(scores)
    keep = ~np.isin(ids, np.asarray(history, dtype=np.int64))
    ids, scores = ids[keep], scores[keep]
    return ids[np.lexsort((ids, -scores))].tolist()


def target_rank(items, target):
    return items.index(int(target))+1 if int(target) in items else 51


def metrics(rr):
    rr = np.asarray(rr)
    return {f'{name}{k}': float(np.mean((rr <= k) if name == 'H' else
            (rr <= k)/np.log2(rr+1))) for k in (5, 10, 20, 50) for name in ('N', 'H')}


def contrast(a, b, seed=20260921):
    a, b = np.asarray(a), np.asarray(b)
    diff = (a <= 10)/np.log2(a+1) - (b <= 10)/np.log2(b+1)
    rng = np.random.RandomState(seed)
    boot = diff[rng.randint(len(a), size=(2000, len(a)))].mean(1)
    base = metrics(b)['N10']
    return dict(delta_N10=float(diff.mean()), relative_N10=float(diff.mean()/base) if base else None,
        descriptive_95_interval=np.quantile(boot, [.025, .975]).tolist(),
        gained_H10=int(((a<=10)&(b>10)).sum()), lost_H10=int(((a>10)&(b<=10)).sum()),
        net_H10=int((a<=10).sum()-(b<=10).sum()),
        gained_H50=int(((a<=50)&(b>50)).sum()), lost_H50=int(((a>50)&(b<=50)).sum()))
