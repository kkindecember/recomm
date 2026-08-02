import pytest
import torch

from experiment.phase7.gcgd_p1_run import prefix_feature, summarize_rows


def _row(group, covered, baseline_rank, candidate_rank):
    def metrics(rank):
        return {
            "Recall@5": float(rank is not None and rank <= 5),
            "NDCG@5": 1.0 if rank == 1 else 0.0,
            "Recall@10": float(rank is not None and rank <= 10),
            "NDCG@10": 1.0 if rank == 1 else 0.0,
            "Recall@50": float(rank is not None and rank <= 50),
            "MRR": 0.0 if rank is None else 1.0 / rank,
        }

    baseline = metrics(baseline_rank)
    candidate = metrics(candidate_rank)
    return {
        "target_group": group,
        "graph_covered": int(covered),
        **{f"baseline_{key}": value for key, value in baseline.items()},
        **{f"candidate_{key}": value for key, value in candidate.items()},
        "target_in_baseline_beam50": int(baseline_rank is not None),
        "target_in_candidate_beam50": int(candidate_rank is not None),
        "new_hit_at10_outside_A_beam": int(baseline_rank is None and candidate_rank == 1),
        "changed": int(baseline_rank != candidate_rank),
        "broad_harm": int(baseline_rank == 1 and candidate_rank is None),
    }


def test_prefix_feature_is_target_free_bounded_and_depth_sensitive():
    shallow = prefix_feature(
        torch.tensor([2.0, 1.0]),
        torch.log_softmax(torch.tensor([0.3, 0.7]), dim=0),
        compatible_leaf_fraction=0.5,
        depth=1,
        maximum_depth=4,
    )
    deep = prefix_feature(
        torch.tensor([2.0, 1.0]),
        torch.log_softmax(torch.tensor([0.3, 0.7]), dim=0),
        compatible_leaf_fraction=0.5,
        depth=4,
        maximum_depth=4,
    )
    assert len(shallow) == 6
    assert all(0.0 <= value <= 1.0 for value in shallow + deep)
    assert shallow[-1] == 0.0 and deep[-1] == 0.75


def test_summary_reports_all_groups_and_relative_gain():
    result = summarize_rows([
        _row("head", True, 1, 1),
        _row("tail", False, None, 1),
    ])
    assert result["overall"]["baseline_NDCG@10"] == pytest.approx(0.5)
    assert result["overall"]["candidate_NDCG@10"] == pytest.approx(1.0)
    assert result["overall"]["relative_gain_NDCG@10"] == pytest.approx(1.0)
    assert result["graph_covered"]["n"] == 1
    assert result["graph_uncovered"]["n"] == 1
