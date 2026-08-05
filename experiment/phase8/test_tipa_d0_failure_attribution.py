import math

import numpy as np

from experiment.phase8.tipa_d0_failure_attribution import bh_adjust, bootstrap_mean, wilson


def test_bootstrap_mean_is_seeded_and_contains_point_estimate():
    a = bootstrap_mean([1.0, 2.0, 3.0], np.random.default_rng(2023))
    b = bootstrap_mean([1.0, 2.0, 3.0], np.random.default_rng(2023))
    assert a == b and a[1] <= a[0] <= a[2]


def test_wilson_handles_zero_success_without_zero_upper_bound():
    lo, hi = wilson(0, 6)
    assert lo == 0.0 and 0.0 < hi < 1.0
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_bh_adjust_is_monotone_in_sorted_p_values():
    rows = [{"p_value": .001}, {"p_value": .02}, {"p_value": .2}]
    bh_adjust(rows)
    assert [r["bh_q_value"] for r in rows] == sorted(r["bh_q_value"] for r in rows)
    assert rows[0]["bh_fdr_0_05"] and rows[1]["bh_fdr_0_05"] and not rows[2]["bh_fdr_0_05"]
