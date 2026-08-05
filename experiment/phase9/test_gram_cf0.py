import math
import sys
from pathlib import Path

import torch
from transformers import T5Config


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = REPO_ROOT / "GRAM" / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))

from model.gram import EncoderWrapper, GRAM  # noqa: E402
from cf0_diagnostic_metrics import item_metrics_from_ranks, rank_from_logits  # noqa: E402


def tiny_config(arm):
    config = T5Config(
        vocab_size=64,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    config.max_seq_len = 5
    config.max_item_num = 4
    config.use_position_embedding = True
    config.cf0_arm = arm
    config.cf0_enabled = arm in {"B", "C"}
    config.cf0_num_items = 12 if config.cf0_enabled else 0
    config.cf0_num_layers = 2
    config.cf0_num_heads = 4
    config.cf0_dropout = 0.0
    config.cf0_loss_weight = 0.1
    config.cf0_injection_scale = 0.1
    config.cf0_joint_score_weight = 0.25
    return config


def tiny_batch():
    torch.manual_seed(7)
    input_ids = torch.randint(2, 64, (2, 4, 5))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[:, 3] = False
    input_ids[:, 3] = 0
    history_item_ids = torch.tensor([[1, 2, 0], [3, 4, 0]])
    history_item_mask = history_item_ids.ne(0)
    target_item_ids = torch.tensor([5, 6])
    labels = torch.tensor([[7, 8, 1], [9, 10, 1]])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "history_item_ids": history_item_ids,
        "history_item_mask": history_item_mask,
        "target_item_ids": target_item_ids,
        "labels": labels,
    }


def test_reverse_valid_prefix_preserves_padding():
    values = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    mask = values.ne(0)
    result = EncoderWrapper._reverse_valid_prefix(values, mask)
    assert result.tolist() == [[3, 2, 1, 0, 0], [5, 4, 0, 0, 0]]


def test_arm_a_ignores_cf0_batch_fields():
    model = GRAM(tiny_config("A")).eval()
    batch = tiny_batch()
    plain = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_dict=False,
    )[0]
    with_extra_fields = model(return_dict=False, **batch)[0]
    assert torch.equal(plain, with_extra_fields)


def test_arm_b_has_finite_joint_loss_and_cf0_gradients():
    model = GRAM(tiny_config("B")).train()
    outputs = model(return_dict=False, **tiny_batch())
    loss = outputs[0]
    assert torch.isfinite(loss)
    assert model.last_loss_components is not None
    assert torch.isfinite(model.last_loss_components["cf0_item"])
    loss.backward()
    grad = model.encoder.cf0_item_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_arm_c_exposes_gate_and_candidate_scores():
    model = GRAM(tiny_config("C")).eval()
    with torch.no_grad():
        outputs = model(return_dict=False, **tiny_batch())
        scores = model.score_cf0_candidates(torch.tensor([[1, 2], [3, 4]]))
    assert torch.isfinite(outputs[0])
    assert scores.shape == (2, 2) and torch.isfinite(scores).all()
    assert model.encoder.last_cf0_gate_mean is not None
    assert 0.0 <= float(model.encoder.last_cf0_gate_mean) <= 1.0


def test_item_head_rank_excludes_padding_and_computes_metrics():
    logits = torch.tensor(
        [
            [100.0, 0.2, 0.9, 0.7],
            [100.0, 0.8, 0.3, 0.1],
        ]
    )
    targets = torch.tensor([3, 1])
    ranks = rank_from_logits(logits, targets)
    assert ranks.tolist() == [2, 1]
    metrics = item_metrics_from_ranks(ranks)
    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@5"] == 1.0
    assert math.isclose(metrics["MRR".lower()], 0.75)
