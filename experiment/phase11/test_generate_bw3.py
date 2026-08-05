from generate_bw3_pseudofuture_beams import deterministic_users


def test_formal_generator_cohort_is_deterministic():
    users = ["u4", "u1", "u3", "u2"]
    assert deterministic_users(users, 3, 2027) == deterministic_users(reversed(users), 3, 2027)
