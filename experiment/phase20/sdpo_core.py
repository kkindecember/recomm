"""S-DPO Eq.(11), Chen et al., NeurIPS2024, arXiv:2406.09215v3.

Independent GRAM adaptation: raw response log-probability sums, a frozen
reference, and four hard negatives mined on legal train prefixes.
"""
import torch
from experiment.phase20.preference_core import path_labels, sequence_logps


def softmax_preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    if rejected.ndim != 2 or rejected.shape[0] != chosen.shape[0] or rejected.shape[1] < 1:
        raise ValueError('Expected one or more negatives for every example')
    if reference_rejected.shape != rejected.shape or reference_chosen.shape != chosen.shape or beta <= 0:
        raise ValueError('Invalid reference shape or beta')
    pos_reward = beta * (chosen - reference_chosen.detach())
    neg_reward = beta * (rejected - reference_rejected.detach())
    logits = neg_reward - pos_reward[:, None]
    zeros = torch.zeros_like(pos_reward[:, None])
    return torch.logsumexp(torch.cat([zeros, logits], dim=1), dim=1), (-logits).detach()


def multi_logps(model, batch, negatives, paths, device):
    """One history encoding; one positive and K complete decoder responses."""
    labels = batch['target_ids'].to(device)
    mask = batch['item_text_masks'].to(device)
    positive = model(input_ids=batch['item_text_ids'].to(device), attention_mask=mask,
                     labels=labels, use_cache=False, return_dict=True)
    chosen = sequence_logps(positive.logits, labels)
    nll = positive.loss
    scores = []
    for j in range(len(negatives[0])):
        neg_labels = path_labels(paths, [row[j] for row in negatives], device)
        negative = model(input_ids=None, attention_mask=mask,
            encoder_outputs=(positive.encoder_last_hidden_state,), labels=neg_labels,
            use_cache=False, return_dict=True)
        scores.append(sequence_logps(negative.logits, neg_labels))
    return chosen, torch.stack(scores, dim=1), nll


def choose_hard_negatives(candidates, target, count):
    result = [item for item in candidates if item != target][:count]
    if len(result) != count or len(set(result)) != count or target in result:
        raise ValueError('Not enough distinct non-target hard negatives')
    return result
