"""Sequence preference primitives for the SPRec -> GRAM adaptation.

Attribution: Gao et al., WWW 2025, https://arxiv.org/abs/2412.09243.
This is an independent encoder-decoder implementation, not the authors' Llama
reproduction. DPO uses summed response log probabilities, including EOS.
"""
import torch
from torch.nn import functional as F


def sequence_logps(logits, labels):
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit/label shape mismatch")
    mask = labels.ne(-100)
    if not mask.any(1).all():
        raise ValueError("empty response")
    safe = labels.masked_fill(~mask, 0)
    token = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token * mask).sum(-1)


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = beta * ((chosen - rejected) -
                     (reference_chosen.detach() - reference_rejected.detach()))
    return -F.logsigmoid(margin), margin.detach()


def path_labels(paths, item_ids, device):
    # paths include decoder start (0) and EOS (1); the start is never a label.
    rows = [paths[int(i) - 1][1:] for i in item_ids]
    if any(not row or row[-1] != 1 or 0 in row for row in rows):
        raise ValueError("invalid lexical response path")
    labels = torch.full((len(rows), max(map(len, rows))), -100, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        labels[i, :len(row)] = torch.tensor(row, device=device)
    return labels


def sampling_prefix_function(trie):
    allowed = trie.prefix_allowed_tokens_fn()
    def prefix(batch_id, input_ids):
        # HF's batched ancestral sampler still computes probabilities for rows
        # that already emitted EOS while other rows continue. An empty terminal
        # trie branch would make every logit -inf and multinomial receive NaNs.
        if 1 in input_ids.tolist():
            return [0]
        tokens = allowed(batch_id, input_ids)
        if not tokens:
            raise ValueError('Unfinished sample reached an empty lexical trie branch')
        return tokens
    return prefix


def pair_logps(model, batch, rejected_labels, device):
    """Encode history once and backpropagate both responses through that encoder."""
    labels = batch['target_ids'].to(device)
    mask = batch['item_text_masks'].to(device)
    positive = model(input_ids=batch['item_text_ids'].to(device), attention_mask=mask,
                     labels=labels, use_cache=False, return_dict=True)
    chosen = sequence_logps(positive.logits, labels)
    negative = model(input_ids=None, attention_mask=mask,
                     encoder_outputs=(positive.encoder_last_hidden_state,),
                     labels=rejected_labels, use_cache=False, return_dict=True)
    rejected = sequence_logps(negative.logits, rejected_labels)
    return chosen, rejected, positive.loss
