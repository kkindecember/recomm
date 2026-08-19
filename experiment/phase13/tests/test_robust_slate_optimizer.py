from pathlib import Path
import inspect
import sys

import torch

PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from robust_slate_optimizer import (  # noqa: E402
    BETA_GRID,
    actions_from_scores,
    bootstrap_user_sample,
    robust_policy_scores,
)


def test_bootstrap_is_deterministic_and_user_level():
    first = bootstrap_user_sample([2, 4, 6, 8], 11)
    second = bootstrap_user_sample([2, 4, 6, 8], 11)
    assert first == second and len(first) == 4
    assert set(first) <= {2, 4, 6, 8}


def test_beta_grid_is_frozen_and_nonnegative():
    assert BETA_GRID == (0.0, 0.5, 1.0, 2.0)


def test_robust_score_abstains_when_lcb_is_negative():
    row = {
        "v0_top50": [f"g{i}" for i in range(1, 21)],
        "resolver_top50": ["c1", "c2", "c3", *[f"g{i}" for i in range(1, 21)]],
        "portfolio_candidates": ["c1", "c2", "c3"],
        "modeled_items": ["c1", "c2", "c3", "g8", "g9", "g10"],
        "sample_indices": list(range(6)),
    }
    members = [torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) for _ in range(3)]
    scores = robust_policy_scores([row], members, beta=2.0)
    assert actions_from_scores(scores) == ["abstain"]
    assert "target" not in inspect.signature(robust_policy_scores).parameters
