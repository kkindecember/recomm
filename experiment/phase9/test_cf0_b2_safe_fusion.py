import sys
from pathlib import Path

import torch


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from train_cf0_b2_safe_fusion import (  # noqa: E402
    SafeFusionEncoder,
    ZeroInitSafeFusion,
    reverse_valid_prefix,
    summarize_nll,
)


def test_zero_init_is_exact_identity_and_nonzero_scale_changes_valid_tokens():
    torch.manual_seed(1)
    adapter = ZeroInitSafeFusion(8, max_residual_scale=0.2)
    hidden = torch.randn(2, 3, 8)
    cf_state = torch.randn(2, 8)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    initial = adapter(hidden, cf_state, mask)
    assert torch.equal(initial, hidden)
    with torch.no_grad():
        adapter.alpha.fill_(0.5)
    changed = adapter(hidden, cf_state, mask)
    assert not torch.equal(changed[0, 0], hidden[0, 0])
    assert torch.equal(changed[0, 2], hidden[0, 2])


def test_reverse_valid_prefix_preserves_padding():
    values = torch.tensor([[5, 4, 3, 0, 0], [9, 8, 0, 0, 0]])
    mask = values.ne(0)
    output = reverse_valid_prefix(values, mask)
    assert output.tolist() == [[3, 4, 5, 0, 0], [8, 9, 0, 0, 0]]


def test_nll_gate_requires_effect_and_confidence():
    baseline = torch.ones(128).numpy()
    fused = (torch.ones(128) - 0.01).numpy()
    passed = summarize_nll(baseline, fused, 200, 2023)
    assert passed["status"] == "passed"
    failed = summarize_nll(baseline, baseline.copy(), 200, 2023)
    assert failed["status"] == "failed"


class _BaseEncoder(torch.nn.Module):
    main_input_name = "input_ids"

    def __init__(self):
        super().__init__()
        self.n_passages = 2

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        batch = input_ids.size(0)
        return (torch.arange(batch * 6 * 4, dtype=torch.float32).reshape(batch, 6, 4),)


def test_encoder_only_modifies_prompt_passage():
    base = _BaseEncoder()
    adapter = ZeroInitSafeFusion(4)
    with torch.no_grad():
        adapter.alpha.fill_(0.4)
    wrapped = SafeFusionEncoder(base, adapter)
    wrapped.set_cf_state(torch.randn(1, 4))
    mask = torch.ones(1, 6, dtype=torch.bool)
    output = wrapped(input_ids=torch.ones(1, 6, dtype=torch.long), attention_mask=mask)[0]
    original = base(input_ids=torch.ones(1, 6, dtype=torch.long), attention_mask=mask)[0]
    assert not torch.equal(output[:, :3], original[:, :3])
    assert torch.equal(output[:, 3:], original[:, 3:])
