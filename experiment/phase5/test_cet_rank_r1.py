import json

from experiment.phase5.cet_rank_r1 import ordered_file_users, route_decision


def test_ordered_file_users_preserves_order_and_rejects_duplicates(tmp_path):
    path = tmp_path / "users.txt"
    path.write_text("u3\nu1\nu2\n")
    assert ordered_file_users(path) == ["u3", "u1", "u2"]
    path.write_text("u1\nu1\n")
    try:
        ordered_file_users(path)
    except ValueError:
        return
    raise AssertionError("duplicates must fail")


def test_route_decision_precedence():
    passed = {
        dataset: {"integrity_pass": True, "optimization_pass": True}
        for dataset in ("Toys", "Beauty")
    }
    assert route_decision(passed) == "CET_R1_RANK_CONSISTENCY_PASS"
    failed = json.loads(json.dumps(passed))
    failed["Beauty"]["optimization_pass"] = False
    assert route_decision(failed) == "STOP_CET_RANK_NOT_OPTIMIZABLE"
    invalid = json.loads(json.dumps(passed))
    invalid["Toys"]["integrity_pass"] = False
    assert route_decision(invalid) == "INVALID_R1_FIX_AND_EXACT_RERUN"
