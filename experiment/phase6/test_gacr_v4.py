import torch

from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase6.gacr_v4 import (
    evaluate_gate,
    select_domain_thresholds,
    target_free_gate_features,
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
        ranker.network[-1].bias.fill_(0.5)
    return ranker.state_dict()


def _gate_state(probability=0.5):
    logit = torch.logit(torch.tensor(probability))
    return {
        "mean": torch.zeros(8),
        "std": torch.ones(8),
        "weight": torch.zeros(8),
        "bias": logit,
    }


def test_gate_features_do_not_use_target_or_group():
    residual = torch.tensor([0.1, -0.1, 0.0])
    first = target_free_gate_features(_record(1, "tail"), residual)
    second = target_free_gate_features(_record(2, "head"), residual)
    assert torch.equal(first, second)


def test_threshold_zero_is_exact_v3_application_control():
    groups, rows = evaluate_gate(
        [_record(), _record(target_group="head")],
        _residual_state(),
        _gate_state(0.01),
        0.2,
        0.0,
        torch.device("cpu"),
    )
    assert rows[0]["gate_applied"] == 1
    assert groups["overall"]["gate_application_rate"] == 1.0


def test_threshold_above_one_is_exact_baseline_identity():
    groups, rows = evaluate_gate(
        [_record(), _record(target_group="head")],
        _residual_state(),
        _gate_state(0.99),
        0.2,
        1.1,
        torch.device("cpu"),
    )
    assert rows[0]["gate_applied"] == 0
    assert rows[0]["candidate_rank"] == rows[0]["baseline_rank"]
    assert groups["overall"]["changed_user_coverage"] == 0.0


def _groups(ndcg, tail_ndcg, recall10=1.0, recall50=1.0, harm=0.0, rate=1.0):
    return {
        "overall": {
            "baseline_NDCG@10": 1.0,
            "candidate_NDCG@10": ndcg,
            "baseline_Recall@10": 1.0,
            "candidate_Recall@10": recall10,
            "baseline_Recall@50": 1.0,
            "candidate_Recall@50": recall50,
            "broad_harm_rate": harm,
            "gate_application_rate": rate,
        },
        "tail": {
            "baseline_NDCG@10": 1.0,
            "candidate_NDCG@10": tail_ndcg,
            "baseline_Recall@50": 1.0,
            "candidate_Recall@50": recall50,
        },
    }


def test_threshold_selection_prefers_safe_increment():
    training = {
        dataset: {
            "seeds": {
                "1": {
                    "calibration": {
                        "0.0": _groups(1.01, 1.01),
                        "0.5": _groups(1.03, 1.02, rate=0.5),
                    }
                }
            }
        }
        for dataset in ("Toys", "Beauty")
    }
    config = {
        "datasets": ["Toys", "Beauty"],
        "training_seeds": [1],
        "gate": {"threshold_candidates": [0.0, 0.5]},
        "calibration_safety": {"broad_harm_max": 0.01},
    }
    selected, _ = select_domain_thresholds(training, config)
    assert selected == {"Toys": 0.5, "Beauty": 0.5}


def test_unsafe_increment_falls_back_to_v3_identity():
    training = {
        dataset: {
            "seeds": {
                "1": {
                    "calibration": {
                        "0.0": _groups(1.01, 1.01),
                        "0.5": _groups(1.05, 0.99),
                    }
                }
            }
        }
        for dataset in ("Toys", "Beauty")
    }
    config = {
        "datasets": ["Toys", "Beauty"],
        "training_seeds": [1],
        "gate": {"threshold_candidates": [0.0, 0.5]},
        "calibration_safety": {"broad_harm_max": 0.01},
    }
    selected, _ = select_domain_thresholds(training, config)
    assert selected == {"Toys": 0.0, "Beauty": 0.0}
