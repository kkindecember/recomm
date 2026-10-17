"""Reverse-KL anchored user guidance strength over the SASRec/BGE expert pair.

Reuses the frozen CE expert mixture and the AGRec-inspired decoder rule through
the existing AlphaOnly head. The only new quantity is an anchor on the candidate
pool: it penalises removing mass from candidates GRAM already ranked well, while
leaving low-probability candidates free to be raised. That asymmetry is the point
of the reverse direction; the forward direction is carried as a control arm.

This module changes a training objective only. It does not add experts, features,
or decoder rules, and it does not claim the anchor improves NDCG.
"""
import numpy as np
import torch

from experiment.phase21 import qwen4_alpha_rank_core as A


def candidate_logits(s_gram, s_mix, valid, scale=20.):
    """Masked candidate logits for the native and the mixed path distributions."""
    assert s_gram.shape == s_mix.shape == valid.shape
    assert valid.dtype == torch.bool
    return (scale*s_gram).masked_fill(~valid, -torch.inf), (scale*s_mix).masked_fill(~valid, -torch.inf)


def kl_divergence(s_gram, s_mix, valid, scale=20., forward=False):
    """KL over the candidate pool. Exactly zero when s_mix equals s_gram.

    Reverse (default) is KL(p_gram || p_mix): finite penalty only where p_gram is
    positive, so candidates GRAM considers near-impossible stay unconstrained.
    Forward is KL(p_mix || p_gram): it penalises raising such candidates, which is
    the opposite behaviour and is therefore a direction control, not a tuning knob.
    """
    z_gram, z_mix = candidate_logits(s_gram, s_mix, valid, scale)
    log_gram = z_gram.log_softmax(-1)
    log_mix = z_mix.log_softmax(-1)
    p_gram, p_mix = log_gram.exp(), log_mix.exp()
    # KL(p||q) = sum p (log p - log q); the forward arm simply swaps the roles.
    # Masked slots are -inf in both log-probabilities, and -inf - -inf is nan, so
    # those positions are replaced by zero before the weighted sum.
    if forward:
        weight = p_mix
        difference = torch.where(valid, log_mix - log_gram, torch.zeros_like(log_gram))
    else:
        weight = p_gram
        difference = torch.where(valid, log_gram - log_mix, torch.zeros_like(log_gram))
    return (weight*difference).sum(-1)


def anchored_objective(scores, s_gram, s_mix, valid, kind, weight):
    """Target path NLL plus the pool anchor. Returns both terms separately.

    `scores[:, 0]` is the realised target path, so the NLL is exactly the one the
    unanchored arm minimises; the anchor is the only added term.
    """
    nll = -scores[:, 0].mean()
    if kind == 'nll':
        return nll, torch.zeros((), dtype=nll.dtype, device=nll.device)
    assert kind in ('kl', 'kl_fwd')
    penalty = kl_divergence(s_gram, s_mix, valid, forward=(kind == 'kl_fwd')).mean()
    return nll + weight*penalty, penalty


def mechanical_weight(nll_low, nll_high, kl_low, kl_high):
    """Next power of ten at or above the NLL/KL exchange rate.

    Chosen before any outcome is inspected: it makes the anchor comparable in
    magnitude to the NLL gain the unanchored loss was chasing between the two
    prescribed strengths. No arm is selected on the basis of this number.
    """
    gain = float(nll_low - nll_high)
    cost = float(kl_high - kl_low)
    if not np.isfinite(gain) or not np.isfinite(cost) or gain <= 0 or cost <= 0:
        return None
    ratio = gain/cost
    return float(10.**np.ceil(np.log10(ratio)))


def tests():
    torch.manual_seed(1729)
    n, k = 5, 31
    s = torch.randn(n, k, dtype=torch.double)
    valid = torch.ones(n, k, dtype=torch.bool)
    valid[:, -3:] = False
    # Identical distributions give exactly zero divergence in both directions.
    for forward in (False, True):
        value = kl_divergence(s, s.clone(), valid, forward=forward)
        torch.testing.assert_close(value, torch.zeros(n, dtype=torch.double), rtol=0, atol=0)
    # A finite, non-negative penalty whenever the mixed distribution moves away.
    moved = s.clone(); moved[:, 0] += 3.
    for value in (kl_divergence(s, moved, valid), kl_divergence(s, moved, valid, forward=True)):
        assert torch.isfinite(value).all() and bool((value >= -1e-12).all())
    # The two directions are not interchangeable. Promoting a candidate GRAM
    # treated as near-impossible is what the forward anchor punishes; demoting a
    # candidate GRAM ranked first is what the reverse anchor punishes. A gentle
    # scale is used so the comparison is not reduced to two one-hot rows.
    ok = torch.ones(1, 3, dtype=torch.bool)
    base = torch.tensor([[0., -3., -3.]], dtype=torch.double)
    promote = torch.tensor([[0., -3., 0.]], dtype=torch.double)
    demote = torch.tensor([[0., -20., -20.]], dtype=torch.double)
    assert float(kl_divergence(base, promote, ok, scale=1., forward=True)) > \
           float(kl_divergence(base, promote, ok, scale=1.))
    assert float(kl_divergence(base, demote, ok, scale=1.)) > \
           float(kl_divergence(base, demote, ok, scale=1., forward=True))
    # Gradient flows to the strength only, and is finite.
    alpha = torch.tensor([.3], dtype=torch.double, requires_grad=True)
    mix = s[:1] + alpha[:, None]*.5
    kl_divergence(s[:1], mix, valid[:1]).sum().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum() > 0
    # Mechanical weight: exchange rate 100 -> 100, 101 -> 1000, invalid -> None.
    assert mechanical_weight(1.0, 0.0, 0.0, 0.01) == 100.
    assert mechanical_weight(1.01, 0.0, 0.0, 0.01) == 1000.
    assert mechanical_weight(0.0, 1.0, 0.0, 0.01) is None
    return dict(zero_at_identical_distributions=True, finite_penalties=True,
                direction_is_material=True, reverse_targets_demotion=True,
                gradient_reaches_strength_only=True, mechanical_weight_rule=True)


if __name__ == '__main__':
    import json
    print(json.dumps(tests()))
