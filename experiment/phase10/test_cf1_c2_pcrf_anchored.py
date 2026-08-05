import sys
from pathlib import Path

import numpy as np


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_c2_pcrf_anchored import (
    FEATURE_NAMES,
    RESIDUAL_CAP,
    anchored_pairwise_loss_grad,
    popularity_weights,
    prepare_pairs,
    standardize,
)


def test_anchor_standardization_preserves_order_and_floor_semantics():
    raw = np.asarray([0.4, -0.2, 1.1, 0.0])
    anchor_g = standardize(raw)
    assert np.array_equal(np.argsort(-raw, kind="stable"),
                          np.argsort(-anchor_g, kind="stable"))
    anchor = np.full(7, anchor_g.min())
    anchor[:4] = anchor_g
    assert np.all(anchor[4:] == anchor_g.min())


def test_popularity_weights_balance_groups_and_average_one():
    frequency = np.asarray([1, 2, 7, 20, 26, 100])
    weights, group_weights = popularity_weights(frequency)
    groups = [frequency <= 5, (frequency > 5) & (frequency < 26), frequency >= 26]
    weighted_counts = [weights[mask].sum() for mask in groups]
    assert np.allclose(weighted_counts, weighted_counts[0])
    assert np.isclose(weights.mean(), 1.0)
    assert np.all(group_weights > 0)


def test_pairwise_gradient_matches_finite_difference():
    x = np.asarray([
        [0.2, 0.1], [0.0, 0.5], [0.4, -0.2],
        [0.3, 0.2], [-0.1, 0.6], [0.5, 0.0],
        [0.1, -0.2], [0.2, 0.4], [-0.3, 0.3],
    ])
    anchor = np.asarray([0.5, 0.2, -0.1, 0.8, 0.0, -0.3, 0.4, 0.1, -0.2])
    lengths = np.asarray([3, 3, 3])
    gold = np.asarray([1, 2, 0])
    pairs, _ = prepare_pairs(
        lengths, gold, [anchor[:3], anchor[3:6], anchor[6:]], np.asarray([2, 30, 10])
    )
    weights = np.asarray([0.15, -0.25])
    loss, gradient = anchored_pairwise_loss_grad(weights, x, anchor, pairs, l2=0.01)
    eps = 1e-6
    numerical = []
    for index in range(weights.size):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical.append((
            anchored_pairwise_loss_grad(plus, x, anchor, pairs, l2=0.01)[0]
            - anchored_pairwise_loss_grad(minus, x, anchor, pairs, l2=0.01)[0]
        ) / (2 * eps))
    assert np.isfinite(loss)
    assert np.allclose(gradient, numerical, atol=1e-6)


def test_residual_is_bounded_and_inference_schema_has_no_target_fields():
    linear = np.asarray([-100.0, 0.0, 100.0])
    residual = RESIDUAL_CAP * np.tanh(linear)
    assert np.max(np.abs(residual)) <= RESIDUAL_CAP
    assert not any(
        forbidden in name for name in FEATURE_NAMES
        for forbidden in ("target", "gold", "label")
    )
