import torch

from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase6.gacr_v5 import (
    evaluate_soft_weight,
    rows_identical,
    select_domain_alphas,
    soft_multiplier,
)


def _record(target_index=1, target_group="tail"):
    return {
        "sample_key": "u:validation:i",
        "target_group": target_group,
        "gram_rank": 2,
        "target_index": target_index,
        "base": torch.tensor([1.0, 0.5, 0.0]),
        "features": torch.tensor(
            [
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.1],
                [0.5, 0.5, 1.0, 1.0, 1.0, 0.2],
                [-0.5, 0.0, 0.5, 0.0, 1.0, -0.1],
            ]
        ),
    }


def _residual_state(bound=0.2):
    ranker = BoundedResidualRanker(6, 16, bound)
    with torch.no_grad():
        ranker.network[-1].weight.fill_(0.1)
        ranker.network[-1].bias.fill_(0.1)
    return ranker.state_dict()


def _gate_state(probability=0.5):
    return {
        "mean": torch.zeros(8),
        "std": torch.ones(8),
        "weight": torch.zeros(8),
        "bias": torch.logit(torch.tensor(probability)),
    }


def test_soft_multiplier_has_exact_identity_and_probability_endpoints():
    assert soft_multiplier(0.2, 1.0) == 1.0
    assert soft_multiplier(0.2, 0.0) == 0.2
    assert soft_multiplier(0.2, 0.5) == 0.6


def test_alpha_one_applies_full_residual():
    groups, rows = evaluate_soft_weight(
        [_record(), _record(target_group="head")],
        _residual_state(),
        _gate_state(0.01),
        0.2,
        1.0,
        torch.device("cpu"),
    )
    assert rows[0]["soft_multiplier"] == 1.0
    assert groups["overall"]["mean_soft_multiplier"] == 1.0


def test_rows_identity_checks_scientific_fields_only():
    _, first = evaluate_soft_weight(
        [_record(), _record(target_group="head")],
        _residual_state(),
        _gate_state(0.2),
        0.2,
        1.0,
        torch.device("cpu"),
    )
    second = [
        dict(row, gate_probability=0.9, soft_multiplier=0.5) for row in first
    ]
    assert rows_identical(first, second)


def _groups(ndcg, tail_ndcg, recall10=1.0, recall50=1.0, harm=0.0, multiplier=1.0):
    return {
        "overall": {
            "baseline_NDCG@10": 1.0,
            "candidate_NDCG@10": ndcg,
            "baseline_Recall@10": 1.0,
            "candidate_Recall@10": recall10,
            "baseline_Recall@50": 1.0,
            "candidate_Recall@50": recall50,
            "broad_harm_rate": harm,
            "mean_soft_multiplier": multiplier,
        },
        "tail": {
            "baseline_NDCG@10": 1.0,
            "candidate_NDCG@10": tail_ndcg,
            "baseline_Recall@50": 1.0,
            "candidate_Recall@50": recall50,
        },
    }


def _selection_config():
    return {
        "datasets": ["Toys", "Beauty"],
        "training_seeds": [1],
        "soft_weight": {"alpha_candidates": [0.0, 0.5, 1.0]},
        "calibration_safety": {"broad_harm_max": 0.01},
    }


def test_selection_prefers_safe_strict_increment():
    training = {
        dataset: {
            "seeds": {
                "1": {
                    "calibration": {
                        "0.0": _groups(1.01, 1.01, multiplier=0.5),
                        "0.5": _groups(1.03, 1.02, multiplier=0.75),
                        "1.0": _groups(1.02, 1.02),
                    }
                }
            }
        }
        for dataset in ("Toys", "Beauty")
    }
    selected, _ = select_domain_alphas(training, _selection_config())
    assert selected == {"Toys": 0.5, "Beauty": 0.5}


def test_selection_falls_back_on_tie_or_unsafe_increment():
    training = {
        dataset: {
            "seeds": {
                "1": {
                    "calibration": {
                        "0.0": _groups(1.02, 1.02),
                        "0.5": _groups(1.05, 0.99),
                        "1.0": _groups(1.02, 1.02),
                    }
                }
            }
        }
        for dataset in ("Toys", "Beauty")
    }
    selected, _ = select_domain_alphas(training, _selection_config())
    assert selected == {"Toys": 1.0, "Beauty": 1.0}
