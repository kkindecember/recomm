import numpy as np

from eval_p9r_fresh_beam import baseline_ranks, correlation, deterministic_users


def test_deterministic_users_is_stable_and_bounded():
    users = ["u3", "u1", "u2", "u4"]
    assert deterministic_users(users, 3) == deterministic_users(reversed(users), 3)


def test_correlation_identity():
    result = correlation([1.0, 3.0, 2.0], [1.0, 3.0, 2.0])
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["spearman"], 1.0)


def test_baseline_rank_and_missing_target():
    records = [
        {"seq": np.asarray([0.1, 0.3, 0.2]), "target_position": 2},
        {"seq": np.asarray([0.1, 0.3, 0.2]), "target_position": -1},
    ]
    assert baseline_ranks(records).tolist() == [2, 51]
