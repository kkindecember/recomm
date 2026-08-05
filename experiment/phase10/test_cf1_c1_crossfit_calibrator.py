import sys
from pathlib import Path

import numpy as np


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_c1_crossfit_calibrator import listwise_loss_grad, rank_target


def test_listwise_gradient_matches_finite_difference():
    x = np.asarray([[1.0, 0.2], [0.1, 1.0], [0.3, -0.2], [0.4, 0.7], [-0.1, 0.2]])
    lengths = np.asarray([2, 3])
    gold = np.asarray([1, 2])
    weights = np.asarray([0.3, -0.4])
    loss, gradient = listwise_loss_grad(weights, x, lengths, gold, l2=0.01)
    eps = 1e-6
    numerical = []
    for index in range(weights.size):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical.append((listwise_loss_grad(plus, x, lengths, gold, l2=0.01)[0] -
                          listwise_loss_grad(minus, x, lengths, gold, l2=0.01)[0]) / (2 * eps))
    assert np.isfinite(loss)
    assert np.allclose(gradient, numerical, atol=1e-6)


def test_rank_target_stable_and_missing():
    assert rank_target(np.asarray([0.5, 0.5, 0.1]), 0) == 1
    assert rank_target(np.asarray([0.5, 0.5, 0.1]), 1) == 2
    assert rank_target(np.asarray([0.5, 0.5, 0.1]), -1) == 91
