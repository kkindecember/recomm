import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "protocol" / "tier1_resolver_checkpoint_trajectory.py"
)
SPEC = importlib.util.spec_from_file_location("tier1_resolver_checkpoint_trajectory", MODULE_PATH)
trajectory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trajectory)


def test_checkpoint_epochs_must_be_unique_and_increasing():
    assert trajectory.validate_checkpoint_epochs([12, 30, 60]) == [12, 30, 60]
    for invalid in ([30, 12], [12, 12], [0, 12], []):
        try:
            trajectory.validate_checkpoint_epochs(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid checkpoint list accepted: {invalid}")


def test_portfolio_candidate_filter_matches_frozen_interface():
    gram = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "cold_in_rank8", "g9"]
    resolver = ["warm", "g2", "cold_in_rank8", "cold_new_a", "cold_new_b"]
    cold = {"cold_in_rank8", "cold_new_a", "cold_new_b"}
    assert trajectory.portfolio_candidates(gram, resolver, cold) == [
        "cold_in_rank8",
        "cold_new_a",
        "cold_new_b",
    ]


def test_portfolio2_reproduces_p6_b1_anchor_and_stable_unique():
    gram = [f"g{i}" for i in range(1, 51)]
    resolver = ["c1", "c2", "r3"]
    ranking = trajectory.portfolio2_ranking(gram, resolver, ["c1", "c2"])
    assert ranking[:10] == [*gram[:8], "c1", "c2"]
    assert len(ranking) == len(set(ranking))


def test_rank_summary_reports_head_and_top50_recall():
    summary = trajectory.rank_summary([1, 3, 9, 40, None])
    assert summary["events_top50"] == 4
    assert summary["recall"]["@3"] == 0.4
    assert summary["recall"]["@50"] == 0.8
    assert summary["rank_buckets"]["absent_top50"] == 1
