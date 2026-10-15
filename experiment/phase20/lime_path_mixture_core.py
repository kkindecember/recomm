"""Path-conditioned GRAM/expert mixture, preserving native legal-token mass."""
import math
import numpy as np
import torch
from experiment.phase20.lime_decode_screen_core import ProbabilityGuide


def log_mix(logp, logq, weight):
    if weight == 0.:
        return float(logp)
    if weight == 1.:
        return float(logq)
    return float(np.logaddexp(math.log1p(-weight) + logp, math.log(weight) + logq))


def next_weight(weight, logp, logq, alpha, rho):
    if rho == 0.:
        return alpha
    posterior = 0. if weight == 0. else (1. if weight == 1. else
        math.exp(math.log(weight) + logq - log_mix(logp, logq, weight)))
    return (1. - rho) * alpha + rho * min(1., max(0., posterior))


def rescore_paths(trace, paths, expert_logs=None, alpha=0., rho=0.):
    values = []
    for path in paths:
        total, weight = 0., alpha
        for step in range(1, len(path)):
            prefix, token = tuple(path[:step]), int(path[step])
            tokens, native, mass = trace[prefix]
            n = float(native[tokens.index(token)])
            q = 0. if alpha == 0. else expert_logs[prefix][token]
            # This form matches the existing raw-legal-mass scoring convention.
            total += log_mix(n, mass + q, weight)
            weight = next_weight(weight, n - mass, q, alpha, rho)
        values.append(total / (len(path) - 1))
    return np.asarray(values)


class PathMixtureGuide:
    def __init__(self, index, expert_logs=None, alpha=0., rho=0.):
        assert 0 <= alpha < 1 and 0 <= rho <= 1
        self.index, self.q, self.alpha, self.rho = index, expert_logs, alpha, rho
        self.recorder = ProbabilityGuide(index, expert_logs, alpha if rho == 0 else 0.)
        self.trace = self.recorder.trace
        self.weights = {}

    @property
    def calls(self):
        return self.recorder.calls

    def weight(self, prefix):
        if prefix in self.weights:
            return self.weights[prefix]
        if len(prefix) == 1 or self.rho == 0. or self.alpha == 0.:
            value = self.alpha
        else:
            parent, token = prefix[:-1], prefix[-1]
            tokens, native, mass = self.trace[parent]
            value = next_weight(self.weight(parent), native[tokens.index(token)] - mass,
                                self.q[parent][token], self.alpha, self.rho)
        self.weights[prefix] = value
        return value

    def __call__(self, input_ids, scores):
        scores = self.recorder(input_ids, scores)
        if self.rho == 0. or self.alpha == 0.:
            return scores
        rr, tt, qq, ww = [], [], [], []
        for row, raw in enumerate(input_ids.detach().cpu().tolist()):
            prefix = tuple(raw)
            node = self.index.lookup.get(prefix)
            if node is None or not self.index.children[node]:
                continue
            weight = self.weight(prefix)
            for token in self.index.children[node]:
                rr.append(row); tt.append(token)
                qq.append(self.q[prefix][token]); ww.append(weight)
        if rr:
            rows = torch.tensor(rr, device=scores.device, dtype=torch.long)
            tokens = torch.tensor(tt, device=scores.device, dtype=torch.long)
            q = torch.tensor(qq, device=scores.device, dtype=scores.dtype)
            w = torch.tensor(ww, device=scores.device, dtype=scores.dtype)
            mass = torch.logsumexp(scores, dim=-1)
            scores[rows, tokens] = torch.logaddexp(scores[rows, tokens] + torch.log1p(-w),
                                                  mass[rows] + q + torch.log(w))
        return scores

    def rescore(self, paths):
        return rescore_paths(self.trace, paths, self.q, self.alpha, self.rho)

    def diagnostics(self):
        weights = [self.weight(p) for p in self.trace
                   if len(self.index.children[self.index.lookup[p]]) > 1]
        return dict(branch_prefixes=len(weights),
                    mean_weight=float(np.mean(weights)) if weights else self.alpha,
                    max_change=max((abs(w-self.alpha) for w in weights), default=0.),
                    saturated_prefixes=sum(w < .01 or w > .99 for w in weights),
                    decoder_steps=self.calls)
