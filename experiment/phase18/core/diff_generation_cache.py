"""Avoid copying identical encoder KV caches when beams change ancestry.

Standard deterministic encoder-decoder beam search never moves a beam between
examples, and encoder memory is the same for all beams of an example. Its cross
attention KV cache therefore remains identical after every legal beam reorder.
Only decoder self-attention caches depend on beam ancestry and need index_select.
This opt-in runtime optimization changes no model parameters or beam scores.
"""

from types import MethodType

import torch


def reorder_with_shared_cross_cache(past, beam_idx, beams):
    if past is None:
        return past
    destinations = torch.arange(beam_idx.numel(), device=beam_idx.device)
    if not torch.equal(beam_idx // beams, destinations // beams):
        raise ValueError("Beam ancestry crossed examples; cross-cache reuse is invalid")
    result = []
    for states in past:
        if len(states) != 4:
            raise ValueError("Expected T5 self K/V and cross K/V caches")
        result.append(tuple(state.index_select(0, beam_idx.to(state.device)) for state in states[:2]) + states[2:])
    return tuple(result)


def install_cross_cache_reuse(gram, beams):
    if beams <= 1:
        raise ValueError("Cache reuse is intended for beam search")
    if not hasattr(gram, "_diff_original_reorder_cache"):
        gram._diff_original_reorder_cache = gram._reorder_cache
    gram._diff_cross_cache_reuse = True

    def reorder(self, past, beam_idx):
        if self.training or not self._diff_cross_cache_reuse:
            return self._diff_original_reorder_cache(past, beam_idx)
        return reorder_with_shared_cross_cache(past, beam_idx, beams)

    gram._reorder_cache = MethodType(reorder, gram)
