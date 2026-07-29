from pathlib import Path

import numpy as np

from experiment.phase5.cet_c2 import paired_bootstrap, select_validation_users


def test_select_validation_users_is_deterministic_and_disjoint():
    sequences = {str(index): ["a", "b", "c"] for index in range(20)}
    first = select_validation_users("Toys", sequences, {"1", "2"}, 8, "salt")
    second = select_validation_users("Toys", sequences, {"1", "2"}, 8, "salt")
    assert first == second
    assert not set(first).intersection({"1", "2"})


def test_paired_bootstrap_exact_constant_difference():
    baseline = np.array([0.0, 0.5, 1.0])
    candidate = baseline + 0.25
    result = paired_bootstrap(baseline, candidate, 2023, 100)
    assert result["mean_difference"] == 0.25
    assert result["ci95"] == [0.25, 0.25]
    assert result["changed_users"] == 3


def test_source_exists():
    assert Path("experiment/phase5/cet_c2.py").is_file()
