"""A decoder-state residual reading cached history-description representations.

No modification of GRAM's original input, encoder, or decoder cache is needed.
The frozen language-model output head receives an additional bounded vector.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment.phase21.attribute_retention_core import EMPTY, parse_fields, token_ids

ARMS = ('seen', 'attribute', 'shuffled')


def memory_token_banks(catalog, tokenizer, original_tokens, width=64):
    """Match description lengths, masks and missingness without user/target data."""
    count = len(catalog['items'])
    assert original_tokens.shape == (count, 128)
    banks = {name: np.zeros((count, width), dtype=np.int64) for name in ARMS}
    lengths = np.zeros(count, dtype=np.int64)
    usable = np.zeros(count, dtype=bool)
    buckets = {}
    for item in range(1, count):
        text = parse_fields(catalog['passages'][item]).get('description', '')
        if text.strip().lower() in EMPTY:
            continue
        content = token_ids(tokenizer, 'description: ' + text)[:width-1]
        if not content:
            continue
        n = len(content) + 1
        lengths[item] = n
        usable[item] = True
        banks['attribute'][item, :n] = content + [1]
        visible = [int(t) for t in original_tokens[item] if int(t) not in (0, 1)]
        # All Beauty original passages are 128 tokens. Fail rather than invent
        # unseen filler if a later domain cannot support this matched control.
        assert len(visible) >= n-1, item
        banks['seen'][item, :n] = visible[:n-1] + [1]
        buckets.setdefault(n, []).append(item)
    donors = np.arange(count, dtype=np.int64)
    banks['shuffled'][:] = banks['attribute']
    for n, items in buckets.items():
        items.sort(key=lambda i: hashlib.sha256(
            ('20260923|attribute-memory|'+str(catalog['items'][i])).encode()).digest())
        if len(items) > 1:
            for receiver, donor in zip(items, items[1:] + items[:1]):
                donors[receiver] = donor
                banks['shuffled'][receiver] = banks['attribute'][donor]
    for arm in ARMS:
        assert np.array_equal((banks[arm] != 0).sum(1), lengths)
    assert np.array_equal(lengths[donors], lengths)
    audit = dict(items=count-1, usable_items=int(usable.sum()), width=width,
                 changed_items=int(np.any(banks['attribute'] != banks['shuffled'], axis=1).sum()),
                 same_lengths=True, same_missingness=True, no_user_or_target_input=True,
                 description_is_first_chunk=True, entirely_new_information_not_claimed=True)
    return banks, lengths, usable, donors, audit


class AttributeMemoryResidual(nn.Module):
    def __init__(self, model_width, hidden_width=64):
        super().__init__()
        self.model_width = model_width
        self.hidden_width = hidden_width
        self.query = nn.Linear(model_width, hidden_width, bias=False)
        self.key = nn.Linear(model_width, hidden_width, bias=False)
        self.value = nn.Linear(model_width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, model_width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.memory = None
        self.valid = None

    def set_memory(self, memory, valid):
        assert memory.ndim == 3 and valid.shape == memory.shape[:2]
        assert memory.shape[-1] == self.model_width and memory.shape[1] > 0
        assert torch.isfinite(memory).all()
        # Cached encoder values are fixed evidence, never optimization variables.
        self.memory = memory.detach()
        self.valid = valid.to(dtype=torch.bool).detach()

    def clear_memory(self):
        self.memory = self.valid = None

    def forward(self, scaled_decoder_state):
        if self.memory is None:
            raise RuntimeError('Set history memory explicitly for every request/batch')
        batch = scaled_decoder_state.shape[0]
        if batch % self.memory.shape[0]:
            raise ValueError('Decoder batch is not an integer beam expansion of memory')
        repeats = batch // self.memory.shape[0]
        memory = self.memory.repeat_interleave(repeats, dim=0)
        valid = self.valid.repeat_interleave(repeats, dim=0)
        query = self.query(F.layer_norm(scaled_decoder_state, (self.model_width,)))
        normalized = F.layer_norm(memory, (self.model_width,))
        key, value = self.key(normalized), self.value(normalized)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_width)
        scores = scores.masked_fill(~valid[:, None, :], -1e4)
        attention = scores.softmax(-1) * valid[:, None, :].to(scores.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        update = torch.tanh(self.output(torch.matmul(attention, value))) / math.sqrt(self.model_width)
        return scaled_decoder_state + update

    def attach(self, lm_head):
        # Hook only the input to the output projection. Original backbone hidden
        # states and key/value caches retain their native numerical computation.
        def before_projection(_module, inputs):
            assert len(inputs) == 1
            return (self(inputs[0]),)
        return lm_head.register_forward_pre_hook(before_projection)
