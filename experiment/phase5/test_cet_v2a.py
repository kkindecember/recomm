from experiment.phase5.cet_v2a import ordered_file_users


def test_ordered_file_users_preserves_order(tmp_path):
    path = tmp_path / "users.txt"
    path.write_text("u3\nu1\nu2\n")
    assert ordered_file_users(path) == ["u3", "u1", "u2"]


def test_ordered_file_users_rejects_duplicates(tmp_path):
    path = tmp_path / "users.txt"
    path.write_text("u1\nu1\n")
    try:
        ordered_file_users(path)
    except ValueError:
        return
    raise AssertionError("duplicate users should fail")
