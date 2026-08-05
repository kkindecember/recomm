import sys
from pathlib import Path

import numpy as np


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from eval_cf0_b4_reliability import (  # noqa: E402
    make_folds,
    rank_matrix,
)


def test_fold_assignment_is_exact_and_order_invariant():
    users = [f"u{index}" for index in range(12)]
    first = make_folds(users, num_folds=5, seed="2023")
    second = make_folds(list(reversed(users)), num_folds=5, seed="2023")
    assert first == second
    counts = sorted(list(first.values()).count(fold) for fold in range(5))
    assert counts == [2, 2, 2, 3, 3]


def test_popularity_calibration_demotes_popular_candidate():
    seq = np.array([[0.0, 0.0, 0.0]])
    cf = np.array([[1.0, 0.9, -1.0]])
    pop = np.array([[2.0, 0.0, 0.0]])
    tail_mass = np.array([0.0])
    targets = np.array([1])
    no_correction, _ = rank_matrix(seq, cf, pop, tail_mass, targets, (1.0, 0.0, 0.0))
    corrected, _ = rank_matrix(seq, cf, pop, tail_mass, targets, (1.0, 1.0, 0.0))
    assert no_correction.tolist() == [2]
    assert corrected.tolist() == [1]


def test_tail_heavy_candidate_set_shrinks_effective_weight():
    seq = np.array([[2.0, 1.0], [2.0, 1.0]])
    cf = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    pop = np.zeros_like(cf)
    tail_mass = np.array([0.0, 0.75])
    targets = np.array([0, 0])
    _, reliability = rank_matrix(seq, cf, pop, tail_mass, targets, (1.0, 0.0, 2.0))
    assert reliability.tolist() == [1.0, 0.0625]


def test_missing_target_stays_outside_beam():
    seq = np.array([[1.0, 0.0]])
    cf = np.array([[0.0, 1.0]])
    pop = np.zeros_like(cf)
    ranks, _ = rank_matrix(seq, cf, pop, np.array([0.0]), np.array([-1]), (0.75, 0.0, 1.0))
    assert ranks.tolist() == [51]
