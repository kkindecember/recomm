"""LOHRec Eq.14 item-order objective adapted to fixed GRAM lists.

Only the objective is borrowed: no LOHRec tokenizer, hierarchy distillation,
gold-first candidate correction, or claim of complete benchmark reproduction.
"""
import numpy as np
import torch
import torch.nn.functional as F

from experiment.phase20.gacr_list_distill_core import (
    encode_history, all_scores, decode_scores, list_loss,
)
from experiment.phase20.preference_core import path_labels


def order_loss(scores, teacher_probability, margin):
    """Mean of all n(n-1)/2 distance-margin hinges; no score standardization."""
    assert scores.ndim == 1 and len(scores) >= 2 and margin >= 0
    probability = torch.as_tensor(teacher_probability).detach().cpu().numpy()
    assert probability.shape == tuple(scores.shape) and np.isfinite(probability).all()
    order = torch.as_tensor(np.argsort(-probability, kind='stable').copy(), device=scores.device)
    ordered = scores.to(torch.float64)[order]
    i, j = torch.triu_indices(len(scores), len(scores), 1, device=scores.device)
    violations = ordered[j] - ordered[i] + (j-i).to(ordered.dtype)*margin
    return F.relu(violations).mean()


def auxiliary_loss(scores, parent, probability, covered, arm, settings):
    assert arm in ('CE_anchor', 'order_teacher')
    _, anchor, _ = list_loss(scores, torch.as_tensor(parent, device=scores.device),
                            torch.as_tensor(probability, device=scores.device), 1., 0., 1.)
    rank = order_loss(scores, probability, settings['margin'])
    weight = settings['order_weight']*settings['covered_normalizer'] if covered and arm == 'order_teacher' else 0.
    return settings['anchor_weight']*anchor + weight*rank, anchor, rank, weight


def sample_backward(model, batch, paths, candidates, parent, probability,
                    covered, arm, settings, device, divisor=1):
    assert len(candidates) == len(set(candidates)) == 50 and divisor > 0
    hidden, mask = encode_history(model, batch, device)
    labels = path_labels(paths, candidates, device)
    scores = all_scores(model, hidden, mask, labels, settings['candidate_chunk'])
    leaf = scores.detach().requires_grad_(True)
    auxiliary, anchor, rank, weight = auxiliary_loss(leaf, parent, probability, covered, arm, settings)
    derivative = torch.autograd.grad(auxiliary/divisor, leaf)[0].detach()
    ce = -decode_scores(model, hidden, mask, batch['target_ids'].to(device)).mean()
    assert torch.isfinite(ce+auxiliary) and torch.isfinite(derivative).all()
    (settings['ce_weight']*ce/divisor).backward()
    for start in range(0, 50, settings['candidate_chunk']):
        stop = min(start+settings['candidate_chunk'], 50)
        replay = decode_scores(model, hidden, mask, labels[start:stop])
        torch.testing.assert_close(replay.detach(), scores[start:stop], rtol=0, atol=1e-6)
        (replay*derivative[start:stop]).sum().backward()
    return dict(loss=float(settings['ce_weight']*ce.detach()+auxiliary.detach()),
                ce=float(ce.detach()), anchor_kl=float(anchor.detach()),
                order_hinge=float(rank.detach()), weighted_order=float(rank.detach())*weight,
                covered=int(covered))


def flat_gradient(model):
    return torch.cat([p.grad.detach().cpu().reshape(-1) if p.grad is not None else torch.zeros(p.numel())
                      for p in model.parameters() if p.requires_grad])


def calibration_gradients(model, batch, paths, candidates, probability, settings, device):
    """Separate CE and normalized order gradients, before any optimizer update."""
    hidden, mask = encode_history(model, batch, device)
    labels = path_labels(paths, candidates, device)
    scores = all_scores(model, hidden, mask, labels, settings['candidate_chunk'])
    leaf = scores.detach().requires_grad_(True)
    loss = settings['covered_normalizer']*order_loss(leaf, probability, settings['margin'])
    derivative = torch.autograd.grad(loss, leaf)[0].detach()
    model.zero_grad(set_to_none=True)
    ce = -decode_scores(model, hidden, mask, batch['target_ids'].to(device)).mean()
    (settings['ce_weight']*ce).backward()
    ce_gradient = flat_gradient(model)
    model.zero_grad(set_to_none=True)
    for start in range(0, 50, settings['candidate_chunk']):
        stop = min(start+settings['candidate_chunk'], 50)
        replay = decode_scores(model, hidden, mask, labels[start:stop])
        torch.testing.assert_close(replay.detach(), scores[start:stop], rtol=0, atol=1e-6)
        (replay*derivative[start:stop]).sum().backward()
    rank_gradient = flat_gradient(model)
    model.zero_grad(set_to_none=True)
    assert torch.isfinite(ce_gradient).all() and torch.isfinite(rank_gradient).all()
    return ce_gradient, rank_gradient
