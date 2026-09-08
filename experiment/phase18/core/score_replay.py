"""Read-only numerical comparisons on a frozen beam frontier."""
from contextlib import contextmanager

import torch
from transformers.modeling_outputs import BaseModelOutput


@contextmanager
def precision(tf32):
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def parent_indices(step, paths, depth):
    indices = []
    for path in paths:
        matches = [i for i, prefix in enumerate(step['input_prefixes'])
                   if step['input_valid'][i] and prefix == list(path[:depth])]
        if len(matches) != 1:
            raise RuntimeError('fixed prefix must have exactly one valid parent')
        indices.append(matches[0])
    return indices


def select_frontier(logits, step, paths, depth):
    """Normalize before slicing, preserving the original batch shape/strides."""
    logp = logits.log_softmax(-1)
    indices = parent_indices(step, paths, depth)
    return logits[indices].detach().cpu(), logp[indices].detach().cpu()


def pack(logits, logp, paths, device):
    # logits/logp: [paths, prediction depths, vocabulary].
    if logits.shape != logp.shape or logits.shape[:2] != (len(paths), len(paths[0])):
        raise RuntimeError('path/logit dimensions disagree')
    if not torch.isfinite(logits).all() or not torch.isfinite(logp).all():
        raise FloatingPointError('nonfinite numerical replay')
    tokens = torch.tensor(paths, dtype=torch.long)
    chosen = logp.gather(-1, tokens[..., None]).squeeze(-1)
    chosen_gpu = chosen.to(device)
    sequential = torch.zeros(len(paths), dtype=chosen_gpu.dtype, device=device)
    for d in range(chosen_gpu.shape[1]):
        sequential = sequential + chosen_gpu[:, d]
    summary = {'tokens': [list(p) for p in paths], 'selected_log_probabilities': chosen.tolist(),
               'sequential_float32_scores': sequential.cpu().tolist(),
               'sum_float32_scores': chosen_gpu.sum(-1).cpu().tolist(),
               'sum_float64_scores': chosen.double().sum(-1).tolist()}
    summary['selected_logits'] = logits.gather(-1, tokens[..., None]).squeeze(-1).tolist()
    summary['summation_max_absolute_difference'] = float((sequential - chosen_gpu.sum(-1)).abs().max())
    return {'logits': logits, 'logp': logp, 'chosen': chosen, 'summary': summary}


def compare(left, right):
    a = torch.tensor(left['summary']['sequential_float32_scores'], dtype=torch.float64)
    b = torch.tensor(right['summary']['sequential_float32_scores'], dtype=torch.float64)
    return {'max_absolute_logit_difference': float((left['logits'] - right['logits']).abs().max()),
            'max_absolute_log_probability_difference': float((left['logp'] - right['logp']).abs().max()),
            'max_absolute_selected_log_probability_difference': float((left['chosen'] - right['chosen']).abs().max()),
            'score_differences': (a - b).tolist(),
            'max_absolute_score_difference': float((a - b).abs().max()),
            'margin_difference': float((a[0] - a[1]) - (b[0] - b[1])),
            'logits_exact': torch.equal(left['logits'], right['logits']),
            'log_probabilities_exact': torch.equal(left['logp'], right['logp'])}


def encode(model, batch, device, copies=1):
    ids = batch['item_text_ids'].to(device).repeat(copies, 1, 1)
    mask = batch['item_text_masks'].to(device).repeat(copies, 1, 1)
    model.encoder.n_passages = ids.shape[1]
    model.encoder.set_migration_context(batch['history_item_ids'].to(device).repeat(copies, 1),
                                       batch['history_item_mask'].to(device).repeat(copies, 1))
    return model.encoder(input_ids=ids.reshape(copies, -1),
                         attention_mask=mask.reshape(copies, -1), return_dict=True)[0]


def fixed_frontier(model, hidden, mask, steps, paths, device, use_cache):
    """Replay saved parent indices; never call topk or choose a new frontier."""
    width = len(steps[0]['input_prefixes'])
    encoder = BaseModelOutput(last_hidden_state=hidden.repeat(width, 1, 1))
    attention = mask.to(device).reshape(1, -1).repeat(width, 1)
    past, logits, logp = None, [], []
    for d in range(len(paths[0])):
        step = steps[d]
        ids = torch.tensor([[0] + p for p in step['input_prefixes']], device=device)
        prepared = model.prepare_inputs_for_generation(ids, past_key_values=past, attention_mask=attention,
                                                       encoder_outputs=encoder, use_cache=use_cache)
        output = model(**prepared, return_dict=True)
        a, b = select_frontier(output.logits[:, -1, :], step, paths, d)
        logits.append(a)
        logp.append(b)
        if use_cache and d + 1 < len(paths[0]):
            order = torch.tensor([c['parent_index'] for c in step['retained']], device=device)
            reconstructed = torch.cat([ids[order], torch.tensor([c['token'] for c in step['retained']], device=device)[:, None]], -1)
            expected = torch.tensor([[0] + p for p in steps[d + 1]['input_prefixes']], device=device)
            if not torch.equal(reconstructed, expected):
                raise RuntimeError('saved frontier cannot be reconstructed')
            past = model._reorder_cache(output.past_key_values, order)
        del output
    return pack(torch.stack(logits, 1), torch.stack(logp, 1), paths, device)


def path_full(model, hidden, mask, paths, device):
    if hidden.shape[0] == 1:
        hidden = hidden.repeat(len(paths), 1, 1)
    labels = torch.tensor(paths, device=device)
    output = model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                   attention_mask=mask.to(device).reshape(1, -1).repeat(len(paths), 1),
                   labels=labels, use_cache=False, return_dict=True)
    logits = output.logits
    return pack(logits.detach().cpu(), logits.log_softmax(-1).detach().cpu(), paths, device)


def path_cached(model, hidden, mask, paths, device):
    count = len(paths)
    encoder = BaseModelOutput(last_hidden_state=hidden.repeat(count, 1, 1))
    attention = mask.to(device).reshape(1, -1).repeat(count, 1)
    logits, logp, past = [], [], None
    for d in range(len(paths[0])):
        ids = torch.tensor([[0] + list(p[:d]) for p in paths], device=device)
        output = model(**model.prepare_inputs_for_generation(ids, past_key_values=past,
                    encoder_outputs=encoder, attention_mask=attention, use_cache=True), return_dict=True)
        value = output.logits[:, -1, :]
        logits.append(value.detach().cpu())
        logp.append(value.log_softmax(-1).detach().cpu())
        past = output.past_key_values
    return pack(torch.stack(logits, 1), torch.stack(logp, 1), paths, device)


def path_single(model, hidden, mask, paths, device):
    singles = [path_full(model, hidden, mask, [p], device) for p in paths]
    return pack(torch.cat([p['logits'] for p in singles]), torch.cat([p['logp'] for p in singles]), paths, device)
