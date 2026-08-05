from eval_bw1_candidate_ceiling import decide, deterministic_users


def test_deterministic_users_stable():
    users = ["u3", "u1", "u4", "u2"]
    assert deterministic_users(users, 3) == deterministic_users(reversed(users), 3)


def test_decide_computes_preregistered_headroom():
    rows = [
        {"width": 50, "candidate_recall": 0.20, "pcrf": {"Hit@10": 0.10}},
        {"width": 100, "candidate_recall": 0.23, "pcrf": {"Hit@10": 0.11}},
        {"width": 200, "candidate_recall": 0.25, "pcrf": {"Hit@10": 0.12}},
    ]
    result = decide(rows)
    assert abs(result["coverage_headroom"] - 0.05) < 1e-12
    assert abs(result["pcrf_headroom"] - 0.02) < 1e-12
