import numpy as np

from experiment.phase5.cet_rank_r0 import (
    rankdata,
    spearman,
    union_rank_displacement,
)


def test_rankdata_uses_average_tie_ranks():
    values = rankdata(np.asarray([3.0, 1.0, 1.0, 2.0]))
    assert values.tolist() == [4.0, 1.5, 1.5, 3.0]


def test_spearman_detects_monotonic_relation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_union_rank_displacement_zero_for_identity():
    ranked = ["a", "b", "c", "d"]
    assert union_rank_displacement(ranked, ranked, 3, 5) == 0.0


def test_union_rank_displacement_positive_for_reordering():
    assert union_rank_displacement(
        ["a", "b", "c"], ["b", "a", "c"], 3, 4
    ) > 0.0
