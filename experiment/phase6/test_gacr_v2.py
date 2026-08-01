import torch

from experiment.phase6.gacr_v2 import (
    evaluate_scale,
    select_shared_scale,
    summarize_seed_stability,
    validate_checkpoint_lineage,
)


def _state_with_constant_residual(bound=0.2):
    from experiment.phase4.gacr_s0 import BoundedResidualRanker

    ranker = BoundedResidualRanker(6, 16, bound)
    with torch.no_grad():
        ranker.network[-1].bias.fill_(1.0)
    return ranker.state_dict()


def test_zero_scale_is_exact_baseline_identity():
    records = [
        {
            "sample_key": "u1",
            "target_group": "tail",
            "gram_rank": 2,
            "target_index": 1,
            "base": torch.tensor([1.0, 0.5]),
            "features": torch.zeros(2, 6),
        },
        {
            "sample_key": "u2",
            "target_group": "head",
            "gram_rank": 1,
            "target_index": 0,
            "base": torch.tensor([1.0, 0.5]),
            "features": torch.zeros(2, 6),
        }
    ]
    groups, rows = evaluate_scale(
        records, _state_with_constant_residual(), 0.2, 0.0, torch.device("cpu")
    )
    assert all(row["candidate_rank"] == row["baseline_rank"] for row in rows)
    assert groups["overall"]["changed_user_coverage"] == 0.0


def test_shared_scale_prefers_best_eligible_macro_gain():
    def groups(gain, changed):
        return {
            "overall": {
                "baseline_NDCG@10": 1.0,
                "candidate_NDCG@10": 1.0 + gain,
                "baseline_Recall@10": 1.0,
                "candidate_Recall@10": 1.0,
                "broad_harm_rate": 0.0,
                "changed_user_coverage": changed,
            },
            "tail": {
                "baseline_NDCG@10": 1.0,
                "candidate_NDCG@10": 1.0 + gain,
            },
        }

    training = {
        dataset: {
            "seeds": {
                "1": {"calibration": {"1.0": groups(0.01, 0.1), "1.5": groups(0.03, 0.2)}}
            }
        }
        for dataset in ("Toys", "Beauty")
    }
    config = {
        "datasets": ["Toys", "Beauty"],
        "training_seeds": [1],
        "deployment_scale_candidates": [1.0, 1.5],
        "calibration_safety": {"broad_harm_max": 0.01},
    }
    selected, _ = select_shared_scale(training, config)
    assert selected == 1.5


def test_seed_stability_uses_all_domain_seed_cells():
    validation = {
        dataset: {
            "seeds": {
                str(seed): {
                    "gacr_v2": {
                        "gains": {"overall_ndcg10_relative_gain": value}
                    }
                }
                for seed, value in ((1, 0.01), (2, -0.01))
            }
        }
        for dataset in ("Toys", "Beauty")
    }
    result = summarize_seed_stability(validation, "gacr_v2")
    assert result["domain_seed_cells"] == 4
    assert result["positive_cell_fraction"] == 0.5


def test_checkpoint_lineage_gate_rejects_missing_checkpoint(tmp_path):
    config = {
        "datasets": ["Toys"],
        "inputs": {
            "checkpoint_root": str(tmp_path),
            "expected_c1_checkpoint_sha256": {"Toys": "missing"},
        },
    }
    try:
        validate_checkpoint_lineage(config)
    except RuntimeError as error:
        assert "Toys:missing:" in str(error)
        return
    raise AssertionError("missing parent checkpoint must fail before GPU work")
