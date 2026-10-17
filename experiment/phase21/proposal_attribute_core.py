"""Behavior proposals with target-free, uncertain attribute compatibility.

BGE history representations follow the existing LIME-derived semantic scorer.
This module predicts catalogue attributes, not item scores, and only reranks
fixed proposals. It does not claim that predicted preferences are hard needs.
"""
from collections import Counter
import hashlib
import math

import numpy as np
import torch
from torch import nn

from experiment.phase21.attribute_retention_core import parse_fields, EMPTY
from experiment.phase20.lime_comparison_core import semantic_users, unit_rows

FIELDS = ('coarse_category', 'fine_category', 'brand')
ARMS = ('off', 'bge', 'attribute', 'attribute_no_confidence', 'attribute_prior',
        'attribute_history_shuffled', 'attribute_item_shuffled')


def categories(raw, domain):
    value = ' '.join(raw.lower().split())
    parts = tuple(x.strip() for x in value.split(','))
    root = 'toys games' if domain == 'Toys' else 'beauty'
    if (not value or value in EMPTY or parts[0] != root or len(parts) > 8 or
        len(value) > 220 or any(not x or len(x) > 60 for x in parts) or ':' in value or ';' in value):
        return '', ''
    return ', '.join(parts[:2]) if len(parts) >= 2 else '', ', '.join(parts) if len(parts) >= 3 else ''


def catalogue_schema(catalog, domain):
    values = {k: [''] for k in FIELDS}
    for passage in catalog['passages'][1:]:
        f = parse_fields(passage)
        coarse, fine = categories(f.get('categories', ''), domain)
        brand = ' '.join(f.get('brand', '').lower().split())
        if brand in EMPTY or len(brand) > 80 or ':' in brand or ';' in brand:
            brand = ''
        for key, value in zip(FIELDS, (coarse, fine, brand)):
            values[key].append(value)
    return values


def supported_schema(schema, fit_targets, minimum=3):
    """Vocabulary support comes only from the head-fitting training-side users."""
    result = []
    for field in FIELDS:
        values = schema[field]
        counts = Counter(values[int(t)] for t in fit_targets if values[int(t)])
        names = sorted(k for k, count in counts.items() if count >= minimum)
        if len(names) < 2:
            continue
        mapping = {name: j for j, name in enumerate(names)}
        labels = np.asarray([mapping.get(x, -1) for x in values], np.int64)
        frequency = np.asarray([counts[k] for k in names], float)
        prior = (frequency+.5)/(frequency.sum()+.5*len(names))
        result.append(dict(field=field, values=names, labels=labels, prior=prior,
                           fitting_target_counts=frequency.astype(int)))
    assert result, 'No usable attribute field'
    return result


def history_inputs(histories, embeddings):
    assert all(len(h) for h in histories)
    global_view = semantic_users(histories, embeddings)
    last_view = unit_rows(embeddings)[np.asarray([h[-1] for h in histories])-1]
    return np.concatenate([global_view, last_view], axis=1).astype(np.float32)


def fold(user, domain):
    return int.from_bytes(hashlib.sha256(f'proposal-attribute-20261017|{domain}|{user}'.encode()).digest()[:8], 'big') % 5


def history_donors(samples, domain):
    groups = {}
    for j, sample in enumerate(samples):
        n = len(sample['full_history'])
        bucket = 0 if n <= 10 else 1 if n <= 20 else 2 if n <= 50 else 3
        groups.setdefault(bucket, []).append(j)
    donors = np.arange(len(samples))
    for bucket, members in groups.items():
        if len(members) == 1:
            # Retain the singleton rather than silently crossing cohort boundaries.
            continue
        members.sort(key=lambda j: hashlib.sha256(f'attribute-donor|{domain}|{samples[j]["user_id"]}'.encode()).digest())
        donors[members] = members[1:]+members[:1]
    return donors


def shuffled_labels(specs, domain):
    """Permute candidate attributes; preserve per-field missingness and marginals."""
    result = []
    for spec in specs:
        labels = spec['labels'].copy()
        known = np.flatnonzero(labels >= 0).tolist()
        known.sort(key=lambda j: hashlib.sha256(f'attribute-item-placebo|{domain}|{spec["field"]}|{j}'.encode()).digest())
        if len(known) > 1:
            labels[known] = labels[known[1:]+known[:1]]
        result.append(labels)
    return result


class AttributeHead(nn.Module):
    def __init__(self, width, specs, hidden=64):
        super().__init__()
        self.input = nn.Linear(width, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, len(s['values'])) for s in specs])
        for head, spec in zip(self.heads, specs):
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(torch.tensor(np.log(spec['prior']), dtype=head.bias.dtype))

    def forward(self, inputs):
        hidden = torch.tanh(self.input(inputs))
        return [head(hidden) for head in self.heads]


def loss(logits, labels):
    pieces = []
    for values, target in zip(logits, labels.T):
        known = target >= 0
        if known.any():
            pieces.append(nn.functional.cross_entropy(values[known], target[known]))
    assert pieces
    return torch.stack(pieces).mean()


def softmax(logits, temperature=1.):
    x = np.asarray(logits, float)/temperature
    x = np.exp(x-x.max(1, keepdims=True))
    return x/x.sum(1, keepdims=True)


def confidence(probabilities):
    q = np.asarray(probabilities, float)
    assert q.shape[1] >= 2 and np.allclose(q.sum(1), 1.)
    entropy = -(q*np.log(np.maximum(q, 1e-300))).sum(1)
    # Entropy weight, not a guarantee of correctness or a calibrated reliability score.
    return np.clip(1-entropy/math.log(q.shape[1]), 0., 1.)


def attribute_penalty(probabilities, labels, candidate_ids, use_confidence=True):
    """No target input. Missing/unsupported attributes contribute zero penalty."""
    ids = np.asarray(candidate_ids, np.int64)
    assert ids.ndim == 2 and ids.min() >= 1
    output = np.zeros(ids.shape, float)
    for q, catalog_labels in zip(probabilities, labels):
        value = catalog_labels[ids]
        known = value >= 0
        positions = value.clip(min=0)
        support = q[np.arange(len(q))[:, None], positions]/q.max(1, keepdims=True)
        weight = confidence(q) if use_confidence else np.ones(len(q))
        output += weight[:, None]*np.where(known, 1-support, 0.)
    return output/len(probabilities)


def penalties(probabilities, specs, ids, bge, donors, placebo_labels):
    real = [s['labels'] for s in specs]
    prior = [np.broadcast_to(s['prior'], (len(ids), len(s['prior']))) for s in specs]
    return dict(off=np.zeros(ids.shape), bge=1-np.asarray(bge, float),
        attribute=attribute_penalty(probabilities, real, ids),
        attribute_no_confidence=attribute_penalty(probabilities, real, ids, False),
        attribute_prior=attribute_penalty(prior, real, ids),
        attribute_history_shuffled=attribute_penalty([q[donors] for q in probabilities], real, ids),
        attribute_item_shuffled=attribute_penalty(probabilities, placebo_labels, ids))


def corrected_scores(base_scores, penalty, strength):
    scores = np.asarray(base_scores, float)
    if strength == 0:
        return scores.copy()
    # Affine per-user standardization preserves baseline ranks, including ID ties.
    normalized = (scores-scores.max(1, keepdims=True))/np.maximum(scores.std(1, keepdims=True), 1e-3)
    return normalized-float(strength)*np.asarray(penalty, float)


def ranks(ids, scores, targets):
    from experiment.phase20.catalog_scope_core import ranks as rank_matrix
    return rank_matrix(np.asarray(scores), np.asarray(ids), np.asarray(targets))


def tests():
    from experiment.phase20.catalog_scope_core import metrics
    catalog = dict(passages=['', 'title: x; categories: toys games, games, cards; brand: na;',
        'title: y; categories: toys games, games, board; brand: na;',
        'title: z; categories: toys games; brand: na;'])
    schema = catalogue_schema(catalog, 'Toys')
    specs = supported_schema(schema, [1, 1, 2, 2], minimum=2)
    assert len(specs) == 1 and specs[0]['field'] == 'fine_category'
    assert categories('beauty, x; description: z', 'Beauty') == ('', '')
    ids = np.array([[1, 2, 3], [2, 1, 3]])
    q = np.array([[.8, .2], [.5, .5]])
    labels = [specs[0]['labels']]
    actual = attribute_penalty([q], labels, ids)
    expected = np.zeros((2, 3))
    for row in range(2):
        certainty = 1+sum(p*math.log(p) for p in q[row])/math.log(2)
        for col, item in enumerate(ids[row]):
            label = labels[0][item]
            expected[row, col] = certainty*(1-q[row, label]/max(q[row])) if label >= 0 else 0.
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert actual[0, 2] == 0 and np.max(np.abs(actual[1])) < 1e-15
    placebo = shuffled_labels(specs, 'Toys')
    np.testing.assert_array_equal(np.sort(placebo[0]), np.sort(labels[0]))
    np.testing.assert_array_equal(placebo[0] < 0, labels[0] < 0)
    base = np.array([[3., 3., 1.], [2., 1., 0.]])
    zero = corrected_scores(base, actual, 0.)
    np.testing.assert_array_equal(base, zero)
    np.testing.assert_array_equal(ranks(ids, base, [2, 3]), [2, 3])
    # An absent target stays absent; no target is appended and no historical mask exists.
    assert ranks(ids, base, [4, 4]).min() == 2**30
    assert metrics(ranks(ids, base, [4, 4]))['H50'] == 0.
    torch.manual_seed(0)
    head = AttributeHead(4, specs)
    x = torch.randn(3, 4)
    out = head(x)[0].detach().numpy()
    np.testing.assert_allclose(softmax(out), np.broadcast_to(specs[0]['prior'], (3, 2)), atol=1e-7)
    y = torch.tensor([[0], [1], [-1]])
    expected_loss = -(torch.log_softmax(head(x)[0], -1)[0, 0]+torch.log_softmax(head(x)[0], -1)[1, 1])/2
    torch.testing.assert_close(loss(head(x), y), expected_loss)
    return dict(state='PASSED', strict_field_parsing=True, training_supported_vocabulary=True,
        independent_penalty_formula=True, uniform_confidence_zero=True, missing_attribute_neutral=True,
        shuffled_marginals_and_missingness_exact=True, zero_strength_exact=True,
        deterministic_item_ties=True, outside_targets_retained=True, prior_initialization=True,
        masked_attribute_loss_independent=True, target_free_inference=True)
