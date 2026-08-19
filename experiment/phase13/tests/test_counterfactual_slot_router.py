from pathlib import Path
import inspect
import sys


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from counterfactual_slot_router import (  # noqa: E402
    FEATURE_NAMES,
    choose_best_action,
    expected_action_utility,
    insert_candidate,
)


def test_insert_candidate_preserves_requested_anchor_and_uniqueness():
    gram = [f"g{i}" for i in range(1, 13)]
    resolver = ["cold", *gram]
    at7 = insert_candidate(gram, resolver, "cold", 7)
    at10 = insert_candidate(gram, resolver, "cold", 10)
    assert at7[:7] == [*gram[:6], "cold"]
    assert at10[:10] == [*gram[:9], "cold"]
    assert len(at7) == len(set(at7))
    assert len(at10) == len(set(at10))


def test_expected_utility_accounts_for_displacement_risk():
    gram = [f"g{i}" for i in range(1, 13)]
    resolver = ["cold", *gram]
    probabilities = {"cold": 0.2, "g7": 0.9, "g8": 0.0, "g9": 0.0, "g10": 0.0}
    utility7 = expected_action_utility(gram, resolver, "cold", probabilities, 7)
    utility10 = expected_action_utility(gram, resolver, "cold", probabilities, 10)
    assert utility10 > utility7


def test_best_action_can_choose_tail_slot_for_high_boundary_risk():
    row = {
        "v0_top50": [f"g{i}" for i in range(1, 13)],
        "resolver_top50": ["cold", *[f"g{i}" for i in range(1, 13)]],
        "proposed_cold_item": "cold",
        "modeled_items": ["cold", "g7", "g8", "g9", "g10"],
    }
    action, utility, utilities = choose_best_action(row, [0.2, 0.9, 0.0, 0.0, 0.0])
    assert action == "insert@10"
    assert utility == utilities[action]


def test_feature_schema_and_action_router_have_no_target_argument():
    assert "target" not in inspect.signature(choose_best_action).parameters
    assert len(FEATURE_NAMES) == 8
    assert all("target" not in name and "label" not in name for name in FEATURE_NAMES)
