import numpy as np
import torch

from experiment.phase5.cet_rank_r0g import (
    bootstrap_mean_interval,
    cosine_similarity,
    gradient_summary,
    jensen_shannon,
    length_normalized_scores,
    route_decision,
)


def test_jensen_shannon_identity_and_symmetry():
    left = torch.tensor([0.2, -0.4, 1.0], requires_grad=True)
    right = torch.tensor([-0.3, 0.1, 0.7], requires_grad=True)
    assert abs(float(jensen_shannon(left, left))) < 1e-8
    assert torch.allclose(jensen_shannon(left, right), jensen_shannon(right, left))


def test_length_normalized_scores_gather_through_eos():
    logits = torch.zeros(2, 3, 5)
    labels = torch.tensor([[2, 1, -100], [3, 4, 1]])
    logits[0, 0, 2] = 2.0
    logits[0, 1, 1] = 2.0
    logits[1, 0, 3] = 1.0
    logits[1, 1, 4] = 1.0
    logits[1, 2, 1] = 1.0
    scores = length_normalized_scores(logits, labels)
    logp = torch.log_softmax(logits, dim=-1)
    expected0 = (logp[0, 0, 2] + logp[0, 1, 1]) / 2
    expected1 = (logp[1, 0, 3] + logp[1, 1, 4] + logp[1, 2, 1]) / 3
    assert torch.allclose(scores, torch.stack([expected0, expected1]))


def test_gradient_cosine_and_summary():
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])) == 0.0
    rows = [
        {"local_gradient_norm": 1.0, "rank_gradient_norm": 2.0, "gradient_cosine": -0.5},
        {"local_gradient_norm": 2.0, "rank_gradient_norm": 2.0, "gradient_cosine": 0.5},
    ]
    summary = gradient_summary(rows)
    assert summary["negative_cosine_prevalence"] == 0.5
    assert summary["median_gradient_norm_ratio_rank_over_local"] == 1.5


def test_bootstrap_interval_is_reproducible():
    first = bootstrap_mean_interval([1.0, 2.0, 3.0], 100, 7)
    second = bootstrap_mean_interval([1.0, 2.0, 3.0], 100, 7)
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_route_decision_precedence():
    thresholds = {
        "minimum_masked_users_per_dataset": 24,
        "minimum_rank_loss_signal_coverage": 0.9,
        "minimum_rank_gradient_nonzero_coverage": 0.9,
        "distinct_median_cosine_max": 0.2,
        "distinct_bootstrap_ci_upper_max": 0.3,
        "redundant_median_cosine_min": 0.5,
        "redundant_negative_cosine_prevalence_max": 0.1,
    }
    usable_distinct = {
        d: {"masked_users": 30, "rank_loss_signal_coverage": 1.0,
            "rank_nonzero_gradient_coverage": 1.0, "median_gradient_cosine": 0.1,
            "mean_cosine_bootstrap_95ci": [-0.1, 0.25],
            "negative_cosine_prevalence": 0.4}
        for d in ("Toys", "Beauty")
    }
    assert route_decision(usable_distinct, thresholds, True) == "CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT"
    unusable = {d: dict(v, masked_users=20) for d, v in usable_distinct.items()}
    assert route_decision(unusable, thresholds, True) == "STOP_CET_RANK_NO_USABLE_GRADIENT"
    assert route_decision(usable_distinct, thresholds, False) == "INVALID_R0G_FIX_AND_EXACT_RERUN"
