import sys
from pathlib import Path

import numpy as np
import torch


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_b0_score_identity import (  # noqa: E402
    correlation,
    deterministic_users,
    scientific_gate,
    sequence_scores,
)


def test_sequence_score_includes_eos_and_length_normalizes():
    logits = torch.tensor([[[2.0, 1.0], [0.0, 3.0]]])
    ids = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, True]])
    expected = (torch.log_softmax(logits, -1)[0, 0, 0] + torch.log_softmax(logits, -1)[0, 1, 1]) / 2
    assert torch.allclose(sequence_scores(logits, ids, mask), expected[None])


def test_deterministic_users_is_order_invariant():
    assert deterministic_users(["c", "a", "b"], 2) == deterministic_users(["b", "c", "a"], 2)


def test_correlations_are_exact_for_affine_scores():
    result = correlation([1, 2, 3, 4], [3, 5, 7, 9])
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["spearman"], 1.0)


def test_gate_pass_and_fail():
    metrics = {
        "finite_fraction": 1.0, "pearson": 0.999, "spearman": 0.999,
        "mean_top10_set_overlap": 0.99, "recomputed_Hit@10": 0.1,
        "cached_Hit@10": 0.1, "peak_allocated_mib": 2000, "wall_time_seconds": 30,
    }
    assert scientific_gate(metrics)["status"] == "passed"
    metrics["pearson"] = 0.9
    assert scientific_gate(metrics)["status"] == "failed_score_identity_gate"

