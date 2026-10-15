"""Frozen text vectors; only the history pooling query differs between A/B/C/D."""
import numpy as np
import torch

ARMS = ('B', 'C', 'D')


def unit_rows(x):
    return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-8)


def pool_inputs(embeddings, history):
    assert history and min(history) >= 1 and max(history) <= len(embeddings)
    h = embeddings[torch.tensor(history, device=embeddings.device) - 1]
    w = np.asarray([np.exp(-0.1 * (len(history)-1-t)) for t in range(len(history))], dtype=np.float32)
    weights = torch.from_numpy(w).to(device=embeddings.device, dtype=embeddings.dtype)
    mean = (h * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8)
    return h, weights, mean


def query_weights(queries, history_vectors, recency, tau=.1):
    assert tau > 0
    logits = unit_rows(queries) @ unit_rows(history_vectors).T / tau
    # Multiplying exp(logits) by the original float32 recency preserves its
    # treatment of very old, numerically zero weights.
    weights = torch.exp(logits - logits.max(-1, keepdim=True).values) * recency
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)


def score_user(embeddings, normalized_items, history, arm, tau=.1, block=1024, query_map=None):
    """No target, validation label, or decoder state enters the scoring API."""
    assert arm in ('A', *ARMS) and embeddings.ndim == 2 and block > 0
    h, recency, mean = pool_inputs(embeddings, history)
    if arm == 'A': return normalized_items @ unit_rows(mean)
    if arm == 'B':
        weights = query_weights(mean[None], h, recency, tau)
        pooled = weights @ h
        return (normalized_items @ unit_rows(pooled)[0])
    if arm == 'D':
        assert query_map is not None and len(query_map) == len(embeddings)
        assert query_map.min() >= 0 and query_map.max() < len(embeddings)
    result = torch.empty(len(embeddings), device=embeddings.device, dtype=embeddings.dtype)
    for start in range(0, len(embeddings), block):
        end = min(start + block, len(embeddings))
        queries = embeddings[start:end] if arm == 'C' else embeddings[query_map[start:end]]
        weights = query_weights(queries, h, recency, tau)
        pooled = unit_rows(weights @ h)
        # D uses the wrong item only to query history, not as the scored item.
        result[start:end] = (normalized_items[start:end] * pooled).sum(-1)
    return result


def transform_scores(raw, arm, donors):
    assert arm in ARMS
    return np.array(raw, dtype=np.float32, copy=True)


def fuse(gate, scores, lengths, history, arm):
    assert arm in ARMS
    from experiment.phase21.text_contribution_core import fuse as old_fuse
    return old_fuse(gate, scores, lengths, history, 'real')


def screen_rule(domains):
    """Predeclared two-domain gate on full-catalog fused metrics, not fixed-50."""
    over_a, over_b = {}, {}
    for domain, summary in domains.items():
        mm = {arm: summary['arms'][arm]['metrics']['fusion_catalog'] for arm in ('A', 'B', 'C')}
        over_a[domain] = bool(mm['C']['N10'] > mm['A']['N10'] and mm['C']['H10'] >= mm['A']['H10'])
        over_b[domain] = float(mm['C']['N10'] - mm['B']['N10'])
    passed = set(domains) == {'Toys', 'Beauty'} and all(over_a.values()) and all(v >= 0 for v in over_b.values()) and any(v > 0 for v in over_b.values())
    return dict(pass_for_actual_generation=bool(passed), C_over_A=over_a, C_minus_B_N10=over_b)
