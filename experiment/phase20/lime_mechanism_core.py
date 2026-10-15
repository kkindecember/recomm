"""Two frozen, prefix-dependent probability-mixing hypotheses.

Agreement is distribution overlap, not an estimated probability of correctness.
No target labels, user outcomes, or candidate ranks enter either policy.
"""
import numpy as np
import torch

from experiment.phase20.lime_decode_screen_core import ProbabilityGuide, mixed_log_values


POLICIES = ('constant', 'delayed_branch', 'agreement')


class ScheduledGuide(ProbabilityGuide):
    def __init__(self, index, expert_logs, alpha, policy):
        super().__init__(index, expert_logs, alpha)
        assert policy in POLICIES
        self.policy = policy
        # Count real branching decisions already traversed, not word positions.
        if not hasattr(index, '_mechanism_prior_branches'):
            index._mechanism_prior_branches = {
                prefix: sum(len(index.children[index.lookup[prefix[:k]]]) > 1
                            for k in range(1, len(prefix)))
                for prefix in index.nodes}
        self.prior_branches = index._mechanism_prior_branches
        self.alpha_count = 0
        self.alpha_sum = 0.
        self.branch_alpha_count = 0
        self.branch_alpha_sum = 0.

    def effective_alpha(self, prefix, native, mass, q):
        if not self.alpha:
            return 0.
        if self.policy == 'delayed_branch':
            return self.alpha if self.prior_branches[prefix] else 0.
        if self.policy == 'agreement':
            p = np.exp(np.asarray(native, dtype=np.float64) - mass)
            # Re-normalize only for measuring overlap; mixing still preserves m.
            p /= p.sum()
            expert = np.exp(np.asarray(q, dtype=np.float64))
            expert /= expert.sum()
            overlap = float(np.minimum(p, expert).sum())
            return self.alpha * float(np.clip(overlap, 0., 1.))
        return self.alpha

    def __call__(self, input_ids, scores):
        if self.policy == 'constant' or not self.alpha:
            return super().__call__(input_ids, scores)
        self.calls += 1
        prefixes = [tuple(p) for p in input_ids.detach().cpu().tolist()]
        rows, tokens, qlogs, groups = [], [], [], []
        for row, prefix in enumerate(prefixes):
            node = self.index.lookup.get(prefix)
            if node is None:
                assert torch.isneginf(scores[row]).all(), 'Unmasked noncatalog prefix'
                continue
            legal = list(self.index.children[node])
            if not legal:
                continue
            start = len(tokens)
            rows.extend([row] * len(legal))
            tokens.extend(legal)
            qlogs.extend(self.expert_logs[prefix][t] for t in legal)
            groups.append((row, prefix, start, len(tokens)))
        if not rows:
            return scores
        rr = torch.tensor(rows, dtype=torch.long, device=scores.device)
        tt = torch.tensor(tokens, dtype=torch.long, device=scores.device)
        native_tensor = scores[rr, tt]
        masses = torch.logsumexp(scores, dim=-1)
        cpu_native = native_tensor.detach().cpu().double().numpy()
        cpu_masses = masses.detach().cpu().double().numpy()
        alphas = np.zeros(len(tokens), dtype=np.float64)
        for row, prefix, start, stop in groups:
            legal, native = tokens[start:stop], cpu_native[start:stop]
            mass, q = cpu_masses[row], qlogs[start:stop]
            assert np.isfinite(native).all()
            if prefix not in self.trace:
                self.trace[prefix] = (legal, native.copy(), mass)
            a = self.effective_alpha(prefix, native, mass, q)
            self.alpha_count += 1
            self.alpha_sum += a
            if len(legal) > 1:
                self.branch_alpha_count += 1
                self.branch_alpha_sum += a
            alphas[start:stop] = a
        # Leave zero-strength branches bit-for-bit unchanged.
        nonzero = np.flatnonzero(alphas)
        if len(nonzero):
            keep = torch.tensor(nonzero, dtype=torch.long, device=scores.device)
            rr, tt = rr[keep], tt[keep]
            q = torch.tensor(np.asarray(qlogs)[nonzero], dtype=scores.dtype, device=scores.device)
            a = torch.tensor(alphas[nonzero], dtype=scores.dtype, device=scores.device)
            scores[rr, tt] = torch.logaddexp(
                scores[rr, tt] + torch.log1p(-a),
                q + masses[rr] + torch.log(a))
        return scores

    def rescore(self, paths, expert_logs=None, alpha=None):
        qlogs = self.expert_logs if expert_logs is None else expert_logs
        if alpha is not None:
            assert alpha == self.alpha
        if self.policy == 'constant' or not self.alpha:
            return super().rescore(paths, qlogs, self.alpha)
        result = []
        for path in paths:
            total = 0.
            for step in range(1, len(path)):
                prefix, token = tuple(path[:step]), path[step]
                legal, native, mass = self.trace[prefix]
                q = [qlogs[prefix][t] for t in legal]
                a = self.effective_alpha(prefix, native, mass, q)
                position = legal.index(token)
                total += float(mixed_log_values(native[position], mass, q[position], a))
            result.append(total / (len(path) - 1))
        return np.asarray(result)

    def diagnostics(self):
        return dict(policy=self.policy, alpha_ceiling=self.alpha,
                    visited_branch_rows=self.branch_alpha_count,
                    mean_branch_alpha=(self.branch_alpha_sum / self.branch_alpha_count
                                       if self.branch_alpha_count else None))


def select_alpha(rows, name, choices=(0., .1, .3)):
    """Selection uses tuning records only; exact ties favor smaller alpha."""
    values = {}
    for alpha in choices:
        arm = 'GRAM' if alpha == 0 else f'{name}_{alpha}'
        ranks = np.asarray([r['arms'][arm]['rank'] for r in rows])
        values[str(alpha)] = float(np.mean((ranks <= 10) / np.log2(ranks + 1)))
    chosen = max(choices, key=lambda a: (values[str(a)], -a))
    return chosen, values
