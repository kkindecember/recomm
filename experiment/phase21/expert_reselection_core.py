"""Target-free self-excluded text scoring and six contextual gate inputs.

Self-exclusion changes evidence used to score visible historical items. It does
not remove any item from the recommendation catalog. A history-mass-matched
uniform calibration is included to separate self-exclusion from generic reuse
suppression. This file makes no claim that either operation is novel.
"""
import math
import numpy as np
import torch
from experiment.phase20.lime_comparison_core import semantic_users, unit_rows

SCALE = 20.
NAMES = ['log_history_length', 'history_mass_behavior', 'history_mass_text',
         'novel_entropy_behavior', 'novel_entropy_text', 'novel_expert_js']


def probabilities(scores):
    x = np.asarray(scores, dtype=np.float64)
    assert np.isfinite(x).all()
    x = np.exp(SCALE*(x-x.max(-1, keepdims=True)))
    return x/x.sum(-1, keepdims=True)


def self_excluded_scores(histories, embeddings, reference=None):
    """Preserve baseline minmax scale; revise ONLY historical candidate scores.

    A historical candidate cannot use any occurrence of its own text in its
    user prototype. Other historical texts remain. Empty residual evidence gets
    minimum support 0. Revised scores stay in [0, baseline historical score];
    nonhistorical values are copied exactly from the original score reference.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    candidate = unit_rows(embeddings)
    raw = semantic_users(histories, embeddings) @ candidate.T
    lo, span = raw.min(1), np.ptp(raw, axis=1)
    computed = (raw-lo[:, None])/(span[:, None]+1e-8)
    if reference is None:
        baseline = computed
    else:
        baseline = np.asarray(reference, dtype=np.float32)
        assert baseline.shape == computed.shape
        np.testing.assert_allclose(baseline, computed, atol=8e-6, rtol=3e-5)
    corrected = baseline.copy()
    for j, history in enumerate(histories):
        ids = np.asarray(history, dtype=np.int64)-1
        assert len(ids) and ids.min() >= 0 and ids.max() < len(embeddings)
        weights = np.exp(-.1*np.arange(len(ids)-1, -1, -1)).astype(np.float64)
        weighted = embeddings[ids].astype(np.float64)*weights[:, None]
        total = weighted.sum(0)
        for item in np.unique(ids):
            retained_weight = weights[ids != item].sum()
            if retained_weight <= 0 or span[j] <= 1e-7:
                value = 0.
            else:
                remainder = (total-weighted[ids == item].sum(0))/retained_weight
                remainder /= np.linalg.norm(remainder)+1e-8
                cosine = float(remainder @ candidate[item].astype(np.float64))
                value = (cosine-float(lo[j]))/(float(span[j])+1e-8)
            corrected[j, item] = np.clip(value, 0., baseline[j, item])
    return baseline.astype(np.float32), corrected.astype(np.float32)


def history_mass(scores, history):
    ids = np.unique(np.asarray(history, dtype=np.int64))-1
    return float(probabilities(scores)[ids].sum())


def uniform_mass_control(original, corrected, history):
    """Uniform logit decrement with the same total historical probability.

    Historical items stay legal; no final ranking filter. Unlike self-exclusion,
    this control preserves their relative ordering within the expert.
    """
    original = np.asarray(original, dtype=np.float64)
    ids = np.unique(np.asarray(history, dtype=np.int64))-1
    a, b = history_mass(original, history), history_mass(corrected, history)
    assert b <= a+1e-7
    output = original.copy()
    if len(ids) and len(ids) < len(original):
        a, b = np.clip([a, b], 1e-15, 1.-1e-15)
        logit = lambda x:np.log(x)-np.log1p(-x)
        decrement = max(0., float(logit(a)-logit(b)))/SCALE
        output[ids] -= decrement
    np.testing.assert_allclose(history_mass(output, history), history_mass(corrected, history), atol=2e-8, rtol=2e-6)
    return output.astype(np.float32)


def features(experts, history):
    """No target, GRAM distribution, TopK or top1-top2 margin is an input."""
    experts = np.asarray(experts)
    assert experts.ndim == 2 and experts.shape[0] == 2
    p = probabilities(experts)
    mask = np.zeros(experts.shape[1], bool)
    mask[np.unique(np.asarray(history, dtype=np.int64))-1] = True
    masses = p[:, mask].sum(1)
    count = int((~mask).sum())
    if count:
        q = p[:, ~mask]; q /= q.sum(1, keepdims=True)
        entropy = -(q*np.log(np.maximum(q, 1e-300))).sum(1)/np.log(count) if count > 1 else np.zeros(2)
        middle = q.mean(0)
        js = .5*(q*(np.log(np.maximum(q, 1e-300))-np.log(np.maximum(middle, 1e-300)))).sum()/np.log(2)
    else:
        entropy, js = np.zeros(2), 0.
    result = np.r_[np.log1p(len(history)), masses, entropy, js].astype(np.float32)
    assert result.shape == (6,) and np.isfinite(result).all()
    return result


class Fusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(6, 32)
        self.beta = torch.nn.Linear(32, 2)
        self.strength = torch.nn.Linear(32, 1)
        torch.nn.init.zeros_(self.beta.weight); torch.nn.init.zeros_(self.beta.bias)
        torch.nn.init.zeros_(self.strength.weight)
        torch.nn.init.constant_(self.strength.bias, math.log(.3/.7))

    def forward(self, x):
        hidden = torch.relu(self.input(x))
        return self.beta(hidden).softmax(-1), self.strength(hidden).squeeze(-1).sigmoid().clamp(1e-6, 1.-1e-6)


def tests():
    rng = np.random.default_rng(1042)
    embedding = unit_rows(rng.normal(size=(23, 11)).astype(np.float32))
    histories = [[1, 5, 8], [3], [2, 2, 9, 2], [7, 4]]
    original, corrected = self_excluded_scores(histories, embedding)
    _, replay = self_excluded_scores(histories, embedding, original)
    np.testing.assert_array_equal(replay, corrected)
    for j, history in enumerate(histories):
        mask = np.isin(np.arange(1, 24), history)
        np.testing.assert_array_equal(original[j, ~mask], corrected[j, ~mask])
        assert (corrected[j, mask] <= original[j, mask]).all()
        # Independent residual-prototype calculation, including duplicate IDs.
        w = np.exp(-.1*np.arange(len(history)-1, -1, -1))
        raw = semantic_users([history], embedding)[0] @ unit_rows(embedding).T
        for item in set(history):
            remaining = [k for k, value in enumerate(history) if value != item]
            if remaining:
                vector = sum(w[k]*embedding[history[k]-1].astype(np.float64) for k in remaining)/sum(w[k] for k in remaining)
                vector /= np.linalg.norm(vector)+1e-8
                score = (vector @ unit_rows(embedding)[item-1]-raw.min())/(np.ptp(raw)+1e-8)
                expected = np.clip(score, 0., original[j, item-1])
            else:
                expected = 0.
            np.testing.assert_allclose(corrected[j, item-1], expected, atol=3e-6, rtol=2e-5)
        uniform = uniform_mass_control(original[j], corrected[j], history)
        np.testing.assert_array_equal(uniform[~mask], original[j, ~mask])
        np.testing.assert_allclose(probabilities(uniform)[~mask], probabilities(corrected[j])[~mask], atol=2e-7, rtol=5e-6)
    assert corrected[1, 2] == 0
    _, constant = self_excluded_scores([[1, 2], [4]], np.ones((5, 3), np.float32))
    assert np.isfinite(constant).all()
    experts = rng.random((2, 23))
    value = features(experts, histories[0]); swapped = features(experts[::-1], histories[0])
    np.testing.assert_allclose(swapped, value[[0, 2, 1, 4, 3, 5]], atol=1e-7)
    np.testing.assert_allclose(features(experts+3., histories[0]), value, atol=2e-6)
    p = probabilities(experts); ids = np.asarray(histories[0])-1
    np.testing.assert_allclose(value[1:3], p[:, ids].sum(1), atol=3e-8)
    q = p[:, np.setdiff1d(np.arange(23), ids)]; q /= q.sum(1, keepdims=True)
    np.testing.assert_allclose(value[3:5], -(q*np.log(q)).sum(1)/np.log(q.shape[1]), atol=1e-7)
    assert np.isfinite(features(experts, [])).all() and np.isfinite(features(experts, list(range(1, 24)))).all()
    torch.manual_seed(0); model = Fusion(); beta, alpha = model(torch.tensor(np.stack([value]*3)))
    torch.testing.assert_close(beta, torch.full((3, 2), .5))
    torch.testing.assert_close(alpha, torch.full((3,), .3))
    assert sum(p.numel() for p in model.parameters()) == 323
    return dict(self_exclusion_matches_independent_prototypes=True, duplicate_history_occurrences_removed=True,
        nonhistorical_scores_exact=True, historical_scores_nonincreasing=True, singleton_and_constant_cases=True,
        uniform_control_matches_history_probability=True, feature_swap_equivariance=True,
        six_feature_mass_entropy_formulas=True, score_shift_invariance=True,
        empty_and_all_history_feature_cases=True, target_absent_from_feature_api=True, parameters=323)
