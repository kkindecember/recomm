from pathlib import Path
import sys

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from route_admission import (  # noqa: E402
    FEATURE_NAMES,
    build_candidates_and_features,
    candidate_features,
    rank_candidates,
    stable_partition,
)


def fixture_inputs():
    routes = {
        "w1": ("a",),
        "w2": ("b",),
        "c1": ("a",),
        "c2": ("c",),
    }
    cold = {"c1", "c2"}
    row = {
        "v0_top50": ["w1", "w2", "c1"],
        "resolver_top50": ["c2", "c1", "w2"],
    }
    return routes, cold, row


def test_stable_partition_is_deterministic_and_has_two_values():
    users = [f"user-{i}" for i in range(100)]
    first = [stable_partition(uid) for uid in users]
    second = [stable_partition(uid) for uid in users]
    assert first == second
    assert set(first) == {"calibration", "audit"}


def test_candidate_features_use_candidate_state_not_target():
    routes, cold, row = fixture_inputs()
    a = candidate_features("c1", row["v0_top50"], row["resolver_top50"], routes, cold)
    b = candidate_features("c1", row["v0_top50"], row["resolver_top50"], routes, cold)
    assert a == b
    assert len(a) == len(FEATURE_NAMES)
    assert a[4] == 1.0
    assert a[0] > 0 and a[1] > 0 and a[2] > 0 and a[3] == 1.0


def test_union_is_unique_and_catalog_bounded():
    routes, cold, row = fixture_inputs()
    candidates, features = build_candidates_and_features(row, routes, cold)
    assert candidates == ["w1", "w2", "c1", "c2"]
    assert len(candidates) == len(set(candidates))
    assert features.shape == (4, len(FEATURE_NAMES))


def test_rank_candidates_is_deterministic_with_lexical_tie_break():
    candidates = ["z", "a", "m"]
    features = torch.zeros((3, len(FEATURE_NAMES)))
    ranked = rank_candidates(candidates, features, torch.zeros(len(FEATURE_NAMES)))
    assert ranked == ["a", "m", "z"]
