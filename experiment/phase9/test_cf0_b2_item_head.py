import sys
from collections import Counter
from pathlib import Path

import torch


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from train_cf0_b2_item_head import (  # noqa: E402
    CF0B2ItemHead,
    build_splits,
    collate_sequences,
    scientific_gate,
)


def test_split_uses_only_training_prefix_and_validation_target():
    users = ["u1"]
    sequences = [[1, 2, 3, 4, 5]]
    train, validation, frequencies = build_splits(users, sequences, max_history=2)
    assert train == [("u1", [1], 2), ("u1", [1, 2], 3)]
    assert validation == [("u1", [2, 3], 4)]
    assert frequencies == Counter({1: 1, 2: 1, 3: 1})


def test_item_head_has_finite_loss_and_gradients():
    torch.manual_seed(7)
    model = CF0B2ItemHead(
        num_items=8,
        max_history=4,
        d_model=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    batch = collate_sequences([("u1", [1, 2], 3), ("u2", [2], 4)])
    loss, logits = model(
        batch["history_item_ids"],
        batch["history_item_mask"],
        batch["target_item_ids"],
    )
    assert logits.shape == (2, 8)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.item_embedding.weight.grad is not None
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad is not None


def test_gate_requires_both_overall_and_nonhead_improvement():
    baseline = {"Recall@10": 0.10, "Recall@50": 0.20}
    metrics = {
        "overall": {"Recall@10": 0.13, "Recall@50": 0.25},
        "by_target_popularity": {
            "tail": {"count": 10, "Recall@50": 0.01},
            "middle": {"count": 10, "Recall@50": 0.01},
        },
    }
    assert scientific_gate(metrics, baseline, 0.2, 0.005)["status"] == "passed"
    metrics["by_target_popularity"]["tail"]["Recall@50"] = 0.0
    metrics["by_target_popularity"]["middle"]["Recall@50"] = 0.0
    assert scientific_gate(metrics, baseline, 0.2, 0.005)["status"] == "failed"
