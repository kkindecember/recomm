"""Target-free expert cards, matched evidence controls, and pre-decoder context."""
import hashlib
from collections import defaultdict

import numpy as np

from experiment.phase21.attribute_retention_core import parse_fields, token_ids

ARMS = ('control', 'numeric', 'semantic', 'shuffled')
EXPERTS = ('sequence', 'co-purchase', 'text similarity')


def select_candidates(scores, history, per_expert=3):
    """Each expert independently nominates items; repeats across experts are kept."""
    scores = np.asarray(scores)
    assert scores.ndim == 2 and scores.shape[0] == 3 and np.isfinite(scores).all()
    ids = np.arange(1, scores.shape[1] + 1)
    eligible = ~np.isin(ids, history)
    assert eligible.sum() >= per_expert
    return np.stack([ids[eligible][np.lexsort((ids[eligible], -row[eligible]))[:per_expert]]
                     for row in scores])


def shuffle_evidence(prefixes, tails, users, roles, experts):
    """Only evidence changes, within role/expert/exact token-length strata."""
    assert len(prefixes) == len(tails) == len(users) == len(roles) == len(experts)
    groups = defaultdict(list)
    for i, tail in enumerate(tails):
        groups[(roles[i], experts[i], len(tail))].append(i)
    donors = np.arange(len(tails))
    for group in groups.values():
        group.sort(key=lambda i: (hashlib.sha256(str(users[i]).encode()).digest(), i))
        if len({users[i] for i in group}) < 2:
            continue
        # Users have at most three cards per expert. Test cyclic shifts rather
        # than accidentally leaving same-user evidence in a placebo row.
        for shift in range(1, len(group)):
            proposed = group[shift:] + group[:shift]
            if all(users[a] != users[b] for a, b in zip(group, proposed)):
                donors[group] = proposed
                break
    cards = [list(p) + list(tails[d]) + [1] for p, d in zip(prefixes, donors)]
    for i, d in enumerate(donors):
        assert (roles[i], experts[i], len(tails[i])) == (roles[d], experts[d], len(tails[d]))
        assert cards[i][:len(prefixes[i])] == list(prefixes[i])
        if i != d:
            assert users[i] != users[d]
    return cards, donors


class CardBuilder:
    """Scores and evidence depend on past interactions and catalog metadata only."""
    def __init__(self, catalog, tokenizer, embeddings, itemcf, per_expert=3, width=128):
        self.catalog, self.tokenizer = catalog, tokenizer
        self.per_expert, self.width = per_expert, width
        self.fields = [parse_fields(p) for p in catalog['passages']]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb = np.asarray(embeddings / np.maximum(norms, 1e-12), dtype=np.float32)
        self.cf = {source: {r['item_id']: float(r['score']) for r in neighbors}
                   for source, neighbors in itemcf['neighbors'].items()}
        self.cache = {}

    def tokens(self, text, limit):
        key = (text, limit)
        if key not in self.cache:
            self.cache[key] = token_ids(self.tokenizer, text)[:limit]
        return self.cache[key]

    def item_tokens(self, item):
        # Preserve the complete candidate identity using its existing GRAM
        # vocabulary path, without re-tokenizing or truncating printed ID text.
        path = self.catalog['decoder_paths'][item - 1]
        assert path[0] == 0 and path[-1] == 1
        return list(path[1:-1])

    def evidence(self, history, item, expert):
        assert history and item > 0
        source = self.catalog['items']
        fields = self.fields[item]
        # Candidate metadata is real catalog text. SASRec is not presented as
        # producing a causal attribution to a particular historical item.
        head = list(self.tokens('title: ' + fields.get('title', 'na'), 14))
        if expert == 0:
            supports, values = [], []
            text = '; Recommended from ordered history. No item-level attribution.'
        elif expert == 1:
            recent = history[-20:]
            contributions = defaultdict(float)
            for j, h in enumerate(recent):
                contributions[h] += (j+1)/len(recent)*self.cf.get(source[h], {}).get(source[item], 0.)
            supports = sorted((h for h in contributions if contributions[h] > 0),
                              key=lambda h: (-contributions[h], h))[:2]
            values = [float(contributions[h]) for h in supports]
            text = '; Co-purchase supports: ' if supports else '; No positive co-purchase support in recent history.'
        else:
            unique = sorted(set(history))
            similarities = self.emb[np.asarray(unique) - 1] @ self.emb[item - 1]
            order = np.lexsort((unique, -similarities))[:2]
            supports = [unique[j] for j in order]
            values = [float(similarities[j]) for j in order]
            text = '; Closest text matches in observed history: '
        tail = head + self.tokens(text, 16)
        for support, value in zip(supports, values):
            tail += self.item_tokens(support)
            tail += self.tokens(' ' + self.fields[support].get('title', 'na'), 7)
            tail += self.tokens(' %.2f;' % value, 6)
        # Evidence identities precede optional metadata so truncation cannot
        # silently replace related older evidence with the latest purchase.
        tail += self.tokens('; categories: ' + fields.get('categories', 'na'), 12)
        tail += self.tokens('; brand: ' + fields.get('brand', 'na'), 8)
        return tail, dict(support_items=[int(h) for h in supports], support_values=values,
                          evidence_type=('sequence_ranking_only', 'co_purchase_contribution', 'closest_text')[expert])

    def build(self, history, scores):
        """No target, role, user identity or label argument is accepted."""
        nominations = select_candidates(scores, history[-20:], self.per_expert)
        prefixes, tails, facts = [], [], []
        for expert, candidates in enumerate(nominations):
            for rank, item in enumerate(candidates, 1):
                prefix = list(self.tokens(f'{EXPERTS[expert]} rank {rank}; item: ', 16))
                prefix += self.item_tokens(int(item))
                values = scores[:, item - 1]
                prefix += self.tokens('; SAS %.3f CF %.3f text %.3f;' % tuple(values), 999)
                assert len(prefix) <= 64, (expert, int(item), len(prefix))
                evidence, fact = self.evidence(history, int(item), expert)
                evidence = evidence[:self.width - 1 - len(prefix)]
                prefixes.append(prefix)
                tails.append(evidence)
                facts.append(dict(expert=expert, expert_rank=rank, item_id=int(item),
                                  scores=values.tolist(), **fact))
        return prefixes, tails, facts


def pack_cards(cards, rows, cards_per_row, width=128):
    assert len(cards) == rows * cards_per_row
    result = np.zeros((rows, cards_per_row, width), dtype=np.int64)
    for j, card in enumerate(cards):
        assert 0 < len(card) <= width and card[-1] == 1 and 0 not in card
        result[j // cards_per_row, j % cards_per_row, :len(card)] = card
    return result


def encode_context(model, input_ids, attention_mask, feedback_ids=None):
    """Original GRAM encoder plus separately encoded expert text, before decoder.

    Does not replace history passages, touch lm_head, or modify output scores.
    Extra cards have no historical-position embedding: their expert/rank is text.
    """
    import torch
    from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
    assert not any(p.requires_grad for p in model.encoder.parameters())
    batch = input_ids.shape[0]
    model.encoder.n_passages = input_ids.shape[1]
    model.encoder.set_migration_context(None, None)
    flat_ids, mask = input_ids.reshape(batch, -1), attention_mask.reshape(batch, -1)
    with torch.no_grad():
        hidden = model.encoder(input_ids=flat_ids, attention_mask=mask, return_dict=True)[0]
        if feedback_ids is not None:
            assert feedback_ids.ndim == 3 and feedback_ids.shape[0] == batch
            _, count, width = feedback_ids.shape
            extra = feedback_ids.reshape(batch * count, width)
            extra_mask = extra.ne(0)
            encoded = model.encoder.encoder(input_ids=extra, attention_mask=extra_mask, return_dict=True)[0]
            hidden = torch.cat((hidden, encoded.reshape(batch, count * width, -1)), dim=1)
            mask = torch.cat((mask, extra_mask.reshape(batch, -1)), dim=1)
            flat_ids = torch.cat((flat_ids, feedback_ids.reshape(batch, -1)), dim=1)
    assert hidden.shape[:2] == mask.shape == flat_ids.shape
    return flat_ids, mask, BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden)


def generate_from_context(model, context, **kwargs):
    ids, mask, encoded = context
    assert 'logits_processor' not in kwargs, 'This experiment never mixes expert output scores'
    # Bypass GRAM.generate's redundant encoder call; keep its native T5 decoder.
    return super(type(model), model).generate(input_ids=ids, attention_mask=mask,
                                             encoder_outputs=encoded, **kwargs)
