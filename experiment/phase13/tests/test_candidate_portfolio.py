from pathlib import Path
import inspect
import sys


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from candidate_portfolio import (  # noqa: E402
    PORTFOLIO_SIZES,
    choose_best_portfolio_action,
    expected_portfolio_utility,
    portfolio_ranking,
)


def test_portfolio_actions_are_multi_candidate_only():
    assert PORTFOLIO_SIZES == (2, 3)
    assert 1 not in PORTFOLIO_SIZES


def test_portfolio_ranking_protects_anchor_and_is_unique():
    gram = [f"g{index}" for index in range(1, 21)]
    resolver = ["c1", "c2", "c3", *gram]
    candidates = resolver[:3]
    size2 = portfolio_ranking(gram, resolver, candidates, 2)
    size3 = portfolio_ranking(gram, resolver, candidates, 3)
    assert size2[:10] == [*gram[:8], "c1", "c2"]
    assert size3[:10] == [*gram[:7], "c1", "c2", "c3"]
    assert len(size2) == len(set(size2))
    assert len(size3) == len(set(size3))


def test_portfolio_utility_accounts_for_all_candidates_and_displacement():
    gram = [f"g{index}" for index in range(1, 21)]
    resolver = ["c1", "c2", "c3", *gram]
    candidates = resolver[:3]
    favorable = {"c1": 0.5, "c2": 0.5, "c3": 0.5, "g8": 0.0, "g9": 0.0, "g10": 0.0}
    risky = {**favorable, "g8": 1.0, "g9": 1.0, "g10": 1.0}
    assert expected_portfolio_utility(gram, resolver, candidates, favorable, 3) > 0
    assert expected_portfolio_utility(gram, resolver, candidates, risky, 3) < expected_portfolio_utility(
        gram, resolver, candidates, favorable, 3
    )


def test_action_router_is_target_free_and_can_choose_size_two():
    gram = [f"g{index}" for index in range(1, 21)]
    resolver = ["c1", "c2", "c3", *gram]
    row = {
        "v0_top50": gram,
        "resolver_top50": resolver,
        "portfolio_candidates": resolver[:3],
        "modeled_items": ["c1", "c2", "c3", "g8", "g9", "g10"],
    }
    action, utility, utilities = choose_best_portfolio_action(
        row, [0.5, 0.5, 0.0, 0.9, 0.0, 0.0]
    )
    assert action == "portfolio@2"
    assert utility == utilities[action]
    assert "target" not in inspect.signature(choose_best_portfolio_action).parameters
