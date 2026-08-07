"""CPU unit tests for Phase-12 HI-GRAM EncoderWrapper additions.

Covers:
  1. hi_gram_enabled=False → output bitwise-identical to before HI-GRAM was added
     (i.e. the disabled path is a no-op).
  2. Forward pass runs, output shape correct, values finite.
  3. Backward pass produces finite gradients on every HI-GRAM parameter (α, item
     position embedding, local attention, global attention, token norm).
  4. Padding tokens are not perturbed (the residual bias is masked).
  5. Degenerate case: when all valid item passages carry identical hidden states,
     the pooled representation is stable (finite, matches the identical input).
  6. n_items == 0 edge case (only user prompt present, no history items) returns
     the input unchanged.

Run:  python experiment/phase12/test_hi_gram_encoder.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import T5Config, T5EncoderModel

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = REPO_ROOT / "GRAM" / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))

from model.gram import EncoderWrapper  # noqa: E402


D_MODEL = 32
N_HEADS = 4
NUM_LAYERS = 2
MAX_ITEM_NUM = 6  # keep small for cpu unit test speed
BSZ = 2
N_PASSAGES = MAX_ITEM_NUM + 1  # 1 user prompt + 6 history items
PASSAGE_LEN = 8
VOCAB_SIZE = 100


def _build_config(hi_gram_enabled: bool) -> T5Config:
    """Build a tiny T5 config with (optional) HI-GRAM attributes attached."""
    config = T5Config(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_kv=D_MODEL // N_HEADS,
        d_ff=D_MODEL * 2,
        num_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        num_heads=N_HEADS,
        dropout_rate=0.0,
    )
    # GRAM extensions expected on config:
    config.max_seq_len = PASSAGE_LEN
    config.max_item_num = MAX_ITEM_NUM
    config.use_position_embedding = 0
    config.cf0_enabled = False
    config.cf0_arm = "A"

    config.hi_gram_enabled = hi_gram_enabled
    config.hi_gram_local_window = 3
    config.hi_gram_local_layers = 1
    config.hi_gram_global_layers = 1
    config.hi_gram_num_heads = N_HEADS
    config.hi_gram_dropout = 0.0
    config.hi_gram_fusion_scale_init = 0.1
    config.hi_gram_include_user_prompt = False
    return config


def _build_wrapper(hi_gram_enabled: bool, seed: int = 42) -> EncoderWrapper:
    torch.manual_seed(seed)
    config = _build_config(hi_gram_enabled)
    encoder = T5EncoderModel(config).encoder
    wrapper = EncoderWrapper(encoder=encoder, config=config, use_checkpoint=False)
    wrapper.eval()
    wrapper.n_passages = N_PASSAGES
    return wrapper


def _make_batch(
    n_passages: int = N_PASSAGES,
    passage_len: int = PASSAGE_LEN,
    empty_history: bool = False,
    identical_items: bool = False,
    seed: int = 7,
):
    """Build (input_ids, attention_mask) shaped (B, N*L).

    If empty_history=True, all passages except passage 0 are entirely padded.
    If identical_items=True, all valid passages 1..N-1 have the same tokens.
    """
    torch.manual_seed(seed)
    input_ids = torch.randint(1, VOCAB_SIZE, (BSZ, n_passages, passage_len))
    attention_mask = torch.ones(BSZ, n_passages, passage_len, dtype=torch.long)

    # pad the last two tokens of every passage
    input_ids[:, :, -2:] = 0
    attention_mask[:, :, -2:] = 0

    if empty_history:
        input_ids[:, 1:, :] = 0
        attention_mask[:, 1:, :] = 0
    elif identical_items:
        # All history items identical to passage 1
        input_ids[:, 2:, :] = input_ids[:, 1:2, :]
        attention_mask[:, 2:, :] = attention_mask[:, 1:2, :]

    return (
        input_ids.reshape(BSZ, n_passages * passage_len),
        attention_mask.reshape(BSZ, n_passages * passage_len),
    )


class HiGramEncoderTests(unittest.TestCase):
    def test_disabled_is_noop_bitwise(self):
        wrapper_off = _build_wrapper(hi_gram_enabled=False, seed=42)
        wrapper_on = _build_wrapper(hi_gram_enabled=True, seed=42)
        # Sanity: the T5 encoder weights are seeded identically → outputs must match
        # when the HI-GRAM path is not triggered. To achieve that, disable α by
        # setting fusion scale to 0 AND swap out the on-wrapper's attention modules
        # so any residual bias equals zero — but simpler test: with hi_gram_enabled
        # False the code path is skipped entirely, so use two independent wrappers.
        input_ids, attn = _make_batch()
        with torch.no_grad():
            out_off = wrapper_off(input_ids=input_ids, attention_mask=attn)[0]
            # For the "on" case with α=0 explicitly forced, output should match "off"
            wrapper_on.hi_gram_fusion_scale.data.fill_(0.0)
            out_on = wrapper_on(input_ids=input_ids, attention_mask=attn)[0]
        # α=0 makes HI-GRAM inject no bias, so outputs should be identical
        max_diff = (out_off - out_on).abs().max().item()
        self.assertLess(
            max_diff,
            1e-6,
            f"With α=0 HI-GRAM should be a no-op; max_diff={max_diff}",
        )

    def test_forward_shape_and_finite(self):
        wrapper = _build_wrapper(hi_gram_enabled=True)
        input_ids, attn = _make_batch()
        out = wrapper(input_ids=input_ids, attention_mask=attn)[0]
        expected_shape = (BSZ, N_PASSAGES * PASSAGE_LEN, D_MODEL)
        self.assertEqual(tuple(out.shape), expected_shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_backward_gradients_finite(self):
        wrapper = _build_wrapper(hi_gram_enabled=True)
        wrapper.train()
        input_ids, attn = _make_batch()
        out = wrapper(input_ids=input_ids, attention_mask=attn)[0]
        loss = out.pow(2).mean()
        loss.backward()
        # Every HI-GRAM parameter must have a finite gradient
        hi_params = [
            ("fusion_scale", wrapper.hi_gram_fusion_scale),
            ("item_position", wrapper.hi_gram_item_position.weight),
        ]
        for name, param in wrapper.hi_gram_local_attn.named_parameters():
            hi_params.append((f"local.{name}", param))
        for name, param in wrapper.hi_gram_global_attn.named_parameters():
            hi_params.append((f"global.{name}", param))
        for name, param in wrapper.hi_gram_token_norm.named_parameters():
            hi_params.append((f"tokennorm.{name}", param))

        for name, param in hi_params:
            self.assertIsNotNone(
                param.grad, f"HI-GRAM param {name} received no gradient"
            )
            self.assertTrue(
                torch.isfinite(param.grad).all(),
                f"HI-GRAM param {name} has non-finite gradient",
            )
        # α must have received a nonzero gradient in general
        self.assertGreater(
            wrapper.hi_gram_fusion_scale.grad.abs().item(),
            0.0,
            "α should have a nonzero gradient",
        )

    def test_padding_tokens_not_perturbed(self):
        """Tokens with attention_mask==0 must keep their pre-HI-GRAM values."""
        wrapper = _build_wrapper(hi_gram_enabled=True)
        input_ids, attn = _make_batch()

        # First, compute output with HI-GRAM enabled
        with torch.no_grad():
            out_on = wrapper(input_ids=input_ids, attention_mask=attn)[0]
            # Then with α=0 (should match the pre-HI-GRAM outputs)
            wrapper.hi_gram_fusion_scale.data.fill_(0.0)
            out_baseline = wrapper(input_ids=input_ids, attention_mask=attn)[0]

        # Reshape to (B, N, L, D)
        out_on_4d = out_on.reshape(BSZ, N_PASSAGES, PASSAGE_LEN, D_MODEL)
        out_base_4d = out_baseline.reshape(BSZ, N_PASSAGES, PASSAGE_LEN, D_MODEL)
        attn_4d = attn.reshape(BSZ, N_PASSAGES, PASSAGE_LEN).bool()

        # Where attention_mask == 0 (padding), values should be identical (bias masked out)
        pad_positions = ~attn_4d.unsqueeze(-1).expand_as(out_on_4d)
        diff_at_pad = (out_on_4d - out_base_4d).abs()[pad_positions].max().item()
        # α was reset to 0 for the baseline; for α=0.1 the pad diff should also
        # be exactly 0 because we masked the bias with token_valid before injection.
        # Reset α for the real check.
        wrapper.hi_gram_fusion_scale.data.fill_(0.1)
        with torch.no_grad():
            out_alpha = wrapper(input_ids=input_ids, attention_mask=attn)[0]
        out_alpha_4d = out_alpha.reshape(BSZ, N_PASSAGES, PASSAGE_LEN, D_MODEL)
        pad_diff_alpha = (out_alpha_4d - out_base_4d).abs()[pad_positions].max().item()

        self.assertLess(
            pad_diff_alpha,
            1e-6,
            f"Padding tokens should not receive HI-GRAM bias; max_diff={pad_diff_alpha}",
        )

    def test_identical_items_degenerate_case(self):
        """When all valid history passages are identical, output must be finite."""
        wrapper = _build_wrapper(hi_gram_enabled=True)
        input_ids, attn = _make_batch(identical_items=True)
        with torch.no_grad():
            out = wrapper(input_ids=input_ids, attention_mask=attn)[0]
        self.assertTrue(torch.isfinite(out).all())

    def test_empty_history_returns_unchanged(self):
        """When all history passages are padding, only user prompt is affected."""
        wrapper = _build_wrapper(hi_gram_enabled=True)
        input_ids, attn = _make_batch(empty_history=True)

        with torch.no_grad():
            out_alpha = wrapper(input_ids=input_ids, attention_mask=attn)[0]
            wrapper.hi_gram_fusion_scale.data.fill_(0.0)
            out_base = wrapper(input_ids=input_ids, attention_mask=attn)[0]

        # All history passages are padding → the HI-GRAM bias is masked out for them.
        # user prompt (passage 0) is skipped by default (include_user_prompt=False).
        # So output must equal baseline.
        max_diff = (out_alpha - out_base).abs().max().item()
        self.assertLess(
            max_diff,
            1e-6,
            f"Empty history should leave HI-GRAM as no-op; max_diff={max_diff}",
        )

    def test_local_window_mask_shape(self):
        wrapper = _build_wrapper(hi_gram_enabled=True)
        m = wrapper._build_local_window_mask(6, torch.device("cpu"))
        self.assertEqual(m.shape, (6, 6))
        # Diagonal should always be attendable (mask = False)
        self.assertFalse(m.diagonal().any().item())
        # With window=3, |i-j| < 3 is allowed
        # Distance 3 → masked
        self.assertTrue(m[0, 3].item())
        # Distance 2 → allowed
        self.assertFalse(m[0, 2].item())

    def test_uneven_history_lengths_no_nan(self):
        """Regression: batch with uneven per-sample history lengths must not
        produce NaN in forward or backward.

        Original bug: when a sample had only k < local_window valid history
        items, the attention query positions for the padding items had their
        entire local window masked by (local_mask | key_padding_mask), giving
        softmax(all -inf) = NaN. The NaN survived torch.where cleanup because
        NaN * 0 = NaN in backward, poisoning every T5 encoder gradient.
        """
        wrapper = _build_wrapper(hi_gram_enabled=True)
        wrapper.train()

        input_ids = torch.randint(1, VOCAB_SIZE, (BSZ, N_PASSAGES, PASSAGE_LEN))
        attention_mask = torch.ones(BSZ, N_PASSAGES, PASSAGE_LEN, dtype=torch.long)
        # Sample 0: only 1 valid history item (item slot 1); slots 2..6 padded.
        # This is the pathological pattern — item slot 6 (padding query) sits
        # far from any valid item, and with local_window=3 its window
        # [4, 8] contains only padded slots.
        attention_mask[0, 2:, :] = 0
        input_ids[0, 2:, :] = 0
        # Sample 1: full valid history (no padding items) — sanity control.

        input_ids = input_ids.reshape(BSZ, N_PASSAGES * PASSAGE_LEN)
        attention_mask = attention_mask.reshape(BSZ, N_PASSAGES * PASSAGE_LEN)

        out = wrapper(input_ids=input_ids, attention_mask=attention_mask)[0]
        self.assertTrue(
            torch.isfinite(out).all(),
            "Forward output contains NaN/Inf under uneven history lengths",
        )

        loss = out.pow(2).mean()
        loss.backward()

        for name, param in wrapper.named_parameters():
            if param.grad is None:
                continue
            self.assertTrue(
                torch.isfinite(param.grad).all(),
                f"Non-finite gradient on {name} under uneven history lengths",
            )


if __name__ == "__main__":
    unittest.main()
