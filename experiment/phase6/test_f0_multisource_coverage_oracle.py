from experiment.phase6.f0_multisource_coverage_oracle import dedup, hit, ndcg


def test_source_metrics_are_target_free_and_bounded():
    ranking = dedup(["a", "b", "a"])
    assert ranking == ["a", "b"]
    assert hit(ranking, "b", 50) == 1.0
    assert hit(ranking, "missing", 50) == 0.0
    assert ndcg(ranking, "a") == 1.0


def test_source_does_not_name_sports_or_test_paths():
    source = open("experiment/phase6/f0_multisource_coverage_oracle.py").read()
    assert '"Sports"' not in source
    assert "test_users.txt" not in source


def test_direct_script_adds_repository_root_to_module_path():
    source = open("experiment/phase6/f0_multisource_coverage_oracle.py").read()
    assert "REPO_ROOT = Path(__file__).resolve().parents[2]" in source
    assert "sys.path.insert(0, str(REPO_ROOT))" in source
