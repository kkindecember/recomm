"""Complete-item list distillation with exact chunked gradient replay.

The historical encoder and shared embeddings are frozen. No local sibling or
token-boundary surrogate is used. Candidate labels include EOS, never BOS.
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from experiment.phase20.preference_core import sequence_logps, path_labels
from experiment.phase20.gacr_fusion_core import standardize_tensor


def teacher_distribution(g, cf, frequencies, sample_key, q1=5, temperature=1., seed=20260913):
    """Return item-aligned true/shuffled distributions from frozen B1+PCRF."""
    def z(x):
        return (x - x.mean()) / max(float(x.std()), 1e-6)
    order = np.argsort(-g, kind='stable')
    g, cf, pop = np.asarray(g)[order], np.asarray(cf)[order], np.asarray(frequencies)[order]
    reliability = 1. - (pop[:10] <= q1).mean()
    joint = z(g) + reliability * z(z(cf) - .5 * z(np.log1p(pop)))
    score = z(joint) / temperature
    probability = np.exp(score - score.max()); probability /= probability.sum()
    aligned = probability[np.argsort(order)]
    shuffle_seed = int(hashlib.sha256(f'{seed}|{sample_key}|list-teacher'.encode()).hexdigest()[:8], 16)
    permutation = np.random.RandomState(shuffle_seed).permutation(len(aligned))
    return aligned, aligned[permutation]


def freeze_encoder_and_embeddings(model):
    shared = {id(p) for p in model.shared.parameters()}
    selected = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith('decoder.') and id(parameter) not in shared
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    assert selected and not any(p.requires_grad for p in model.encoder.parameters())
    assert not any(p.requires_grad for p in model.shared.parameters())
    assert not any(p.requires_grad for p in model.lm_head.parameters())
    model.eval()  # Dropout-free scoring/replay; gradients remain enabled.
    return selected


@torch.no_grad()
def encode_history(model, batch, device):
    ids, mask = batch['item_text_ids'].to(device), batch['item_text_masks'].to(device)
    assert ids.size(0) == 1 and ids.ndim == mask.ndim == 3
    model.encoder.n_passages = ids.size(1)
    model.encoder.set_migration_context(None, None)
    hidden = model.encoder(input_ids=ids.reshape(1, -1), attention_mask=mask.reshape(1, -1), return_dict=True)[0]
    assert not hidden.requires_grad
    return hidden, mask.reshape(1, -1)


def decode_scores(model, hidden, mask, labels):
    count = len(labels)
    output = model(input_ids=None, encoder_outputs=(hidden.expand(count, -1, -1),),
                   attention_mask=mask.expand(count, -1), labels=labels,
                   use_cache=False, return_dict=True)
    # For EOS-completed paths, HF beam length excluding appended EOS but including
    # decoder start equals this label count (non-BOS tokens including EOS).
    return sequence_logps(output.logits, labels) / labels.ne(-100).sum(-1)


def list_loss(scores, parent_scores, teacher_probability, anchor_weight, teacher_weight, temperature):
    log_student = F.log_softmax(standardize_tensor(scores) / temperature, -1)
    with torch.no_grad():
        parent = F.softmax(standardize_tensor(parent_scores) / temperature, -1)
    anchor = F.kl_div(log_student, parent, reduction='sum') * temperature**2
    teacher = F.kl_div(log_student, teacher_probability.detach().to(torch.float64), reduction='sum') * temperature**2
    return anchor_weight * anchor + teacher_weight * teacher, anchor, teacher


@torch.no_grad()
def all_scores(model, hidden, mask, labels, chunk_size):
    return torch.cat([decode_scores(model, hidden, mask, labels[start:start+chunk_size])
                      for start in range(0, len(labels), chunk_size)])


def sample_backward(model, batch, paths, candidates, parent_scores, probability,
                    covered, arm, settings, device, divisor=1):
    """Differentiate the whole 50-way loss, then replay each decoder chunk.

    The first pass retains only 50 scalar scores; replay uses dL/d(score), so
    chunking does not replace the complete-list objective with chunk objectives.
    """
    assert len(candidates) == len(set(candidates)) == 50 and arm in ('CE_anchor', 'true_teacher', 'shuffled_teacher')
    hidden, mask = encode_history(model, batch, device)
    labels = path_labels(paths, candidates, device)
    scores = all_scores(model, hidden, mask, labels, settings['candidate_chunk'])
    leaf = scores.detach().requires_grad_(True)
    weight = settings['teacher_weight'] * settings['covered_normalizer'] if covered and arm != 'CE_anchor' else 0.
    auxiliary, anchor, teacher = list_loss(leaf, torch.as_tensor(parent_scores, device=device),
        torch.as_tensor(probability, device=device), settings['anchor_weight'], weight, settings['temperature'])
    derivative = torch.autograd.grad(auxiliary / divisor, leaf)[0].detach()
    gold = batch['target_ids'].to(device)
    ce = -decode_scores(model, hidden, mask, gold).mean()
    assert torch.isfinite(ce) and torch.isfinite(auxiliary) and torch.isfinite(derivative).all()
    (settings['ce_weight'] * ce / divisor).backward()
    for start in range(0, 50, settings['candidate_chunk']):
        stop = min(start + settings['candidate_chunk'], 50)
        replay = decode_scores(model, hidden, mask, labels[start:stop])
        torch.testing.assert_close(replay.detach(), scores[start:stop], rtol=0, atol=1e-6)
        (replay * derivative[start:stop]).sum().backward()
    return {'loss': float(settings['ce_weight'] * ce.detach() + auxiliary.detach()),
            'ce': float(ce.detach()), 'anchor_kl': float(anchor.detach()),
            'teacher_kl': float(teacher.detach()) if covered else 0.,
            'weighted_teacher': float(teacher.detach()) * weight,
            'covered': int(covered)}
