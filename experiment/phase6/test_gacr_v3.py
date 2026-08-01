import torch

from experiment.phase6.gacr_v3 import (
    evaluate_budget,
    residual_safety_multiplier,
    select_domain_budgets,
)


def _constant_state(bound=0.2):
    from experiment.phase4.gacr_s0 import BoundedResidualRanker

    ranker = BoundedResidualRanker(6, 16, bound)
    with torch.no_grad():
        ranker.network[-1].weight[0, 0] = 1.0
    return ranker.state_dict()


def _record():
    return {
        "sample_key": "u1",
        "target_group": "tail",
        "gram_rank": 2,
        "target_index": 1,
        "base": torch.tensor([1.0, 0.5]),
        "features": torch.tensor([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]]),
    }


def test_zero_budget_is_exact_identity():
    head = dict(_record())
    head["sample_key"] = "u2"
    head["target_group"] = "head"
    groups, rows = evaluate_budget(
        [_record(), head], _constant_state(), 0.2, 0.0, torch.device("cpu")
    )
    assert rows[0]["candidate_rank"] == rows[0]["baseline_rank"]
    assert rows[0]["safety_multiplier"] == 0.0
    assert groups["overall"]["changed_user_coverage"] == 0.0


def test_residual_budget_caps_pairwise_spread():
    residual = torch.tensor([-0.2, 0.1, 0.2])
    multiplier = residual_safety_multiplier(residual, 0.04)
    assert abs(multiplier - 0.1) < 1e-7
    capped = multiplier * residual
    assert float(capped.max() - capped.min()) <= 0.04000001


def test_safety_multiplier_is_target_independent():
    residual = torch.tensor([-0.1, 0.0, 0.2])
    first = residual_safety_multiplier(residual, 0.03)
    reordered = residual_safety_multiplier(residual[[2, 0, 1]], 0.03)
    assert first == reordered


def test_domain_budget_selection_fail_closes_to_identity():
    def groups(ndcg_gain, recall_gain, tail_gain, harm, changed, multiplier):
        return {
            "overall": {
                "baseline_NDCG@10": 1.0,
                "candidate_NDCG@10": 1.0 + ndcg_gain,
                "baseline_Recall@10": 1.0,
                "candidate_Recall@10": 1.0 + recall_gain,
                "broad_harm_rate": harm,
                "changed_user_coverage": changed,
                "mean_safety_multiplier": multiplier,
            },
            "tail": {
                "baseline_NDCG@10": 1.0,
                "candidate_NDCG@10": 1.0 + tail_gain,
            },
        }

    training = {}
    for dataset in ("Toys", "Beauty"):
        training[dataset] = {
            "seeds": {
                "1": {
                    "budget_calibration": {
                        "0.0": groups(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        "0.1": groups(0.1, -0.01, 0.1, 0.0, 0.2, 1.0),
                    }
                }
            }
        }
    config = {
        "datasets": ["Toys", "Beauty"],
        "training_seeds": [1],
        "safety_budget_candidates": [0.0, 0.1],
        "calibration_safety": {"broad_harm_max": 0.01},
    }
    selected, _ = select_domain_budgets(training, config)
    assert selected == {"Toys": 0.0, "Beauty": 0.0}


def test_domain_budget_selection_can_differ_by_domain():
    def groups(gain):
        return {
            "overall": {
                "baseline_NDCG@10": 1.0,
                "candidate_NDCG@10": 1.0 + gain,
                "baseline_Recall@10": 1.0,
                "candidate_Recall@10": 1.0,
                "broad_harm_rate": 0.0,
                "changed_user_coverage": gain,
                "mean_safety_multiplier": gain,
            },
            "tail": {
                "baseline_NDCG@10": 1.0,
                "candidate_NDCG@10": 1.0 + gain,
            },
        }

    training = {
        "Toys": {"seeds": {"1": {"budget_calibration": {"0.0": groups(0.0), "0.1": groups(0.01), "0.2": groups(0.02)}}}},
        "Beauty": {"seeds": {"1": {"budget_calibration": {"0.0": groups(0.0), "0.1": groups(0.03), "0.2": groups(0.01)}}}},
    }
    config = {
        "datasets": ["Toys", "Beauty"],
        "training_seeds": [1],
        "safety_budget_candidates": [0.0, 0.1, 0.2],
        "calibration_safety": {"broad_harm_max": 0.01},
    }
    selected, _ = select_domain_budgets(training, config)
    assert selected == {"Toys": 0.2, "Beauty": 0.1}
