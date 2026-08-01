import torch

from experiment.phase6.gacr_v6 import (
    add_standard_metrics,
    compare_methods,
    rank_metric,
    train_full_batch_seed,
)


def test_rank_metric_standard_cutoffs():
    assert rank_metric(5, 5, False) == 1.0
    assert rank_metric(6, 5, False) == 0.0
    assert rank_metric(None, 10, True) == 0.0
    assert rank_metric(1, 10, True) == 1.0


def _rows(candidate_rank):
    return [
        {
            "sample_key": "a",
            "target_group": "head",
            "baseline_rank": 6,
            "candidate_rank": candidate_rank,
            "union_covered": 1,
            "baseline_Recall@10": 1.0,
            "baseline_NDCG@10": 0.0,
            "baseline_Recall@50": 1.0,
            "candidate_Recall@10": 1.0,
            "candidate_NDCG@10": 0.0,
            "candidate_Recall@50": 1.0,
            "changed": 1,
            "broad_harm": 0,
        },
        {
            "sample_key": "b",
            "target_group": "tail",
            "baseline_rank": 11,
            "candidate_rank": 10,
            "union_covered": 1,
            "baseline_Recall@10": 0.0,
            "baseline_NDCG@10": 0.0,
            "baseline_Recall@50": 1.0,
            "candidate_Recall@10": 1.0,
            "candidate_NDCG@10": 1.0,
            "candidate_Recall@50": 1.0,
            "changed": 1,
            "broad_harm": 0,
        },
    ]


def test_standard_metrics_and_aligned_comparison():
    _, first = add_standard_metrics({}, _rows(5))
    _, second = add_standard_metrics({}, _rows(4))
    comparison = compare_methods(first, second)
    assert comparison["Recall@5"]["v3"] == 0.5
    assert comparison["Recall@5"]["v6"] == 0.5
    assert comparison["NDCG@5"]["v6"] > comparison["NDCG@5"]["v3"]


def _record(group, target_index=1):
    return {
        "sample_key": group,
        "target_group": group,
        "gram_rank": 2,
        "target_index": target_index,
        "base": torch.tensor([1.0, 0.5, 0.0]),
        "features": torch.tensor(
            [[1.0] * 6, [0.5] * 6, [-0.5] * 6], dtype=torch.float32
        ),
    }


def test_chunked_full_batch_training_runs_fixed_steps():
    config = {
        "residual": {
            "hidden_dim": 16,
            "bound": 0.2,
            "margin": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "fixed_training_step": 2,
        },
        "deployment_scale": 1.0,
    }
    result = train_full_batch_seed(
        [_record("head"), _record("tail")],
        [_record("head"), _record("tail")],
        config,
        2023,
        torch.device("cpu"),
        chunk_size=1,
    )
    assert result["optimizer_steps"] == 2
    assert result["zero_residual_identity_rate"] == 1.0
    assert torch.isfinite(torch.tensor(result["last_loss"]))
