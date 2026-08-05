import sys
from pathlib import Path


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from eval_cf0_b5_pcrf_test import (  # noqa: E402
    FIXED_P9C_PARAMS,
    FROZEN_PARAMS,
    Q1,
    Q3,
    build_test_history_target,
)


def test_test_split_includes_validation_interaction_in_history():
    history, target = build_test_history_target([1, 2, 3, 4, 5], max_history=3)
    assert history == [2, 3, 4]
    assert target == 5


def test_test_history_truncates_to_twenty():
    sequence = list(range(30))
    history, target = build_test_history_target(sequence, max_history=20)
    assert history == list(range(9, 29))
    assert target == 29


def test_confirmation_parameters_are_exactly_frozen():
    assert FROZEN_PARAMS == (1.0, 0.5, 1.0)
    assert FIXED_P9C_PARAMS == (0.75, 0.0, 0.0)
    assert (Q1, Q3) == (5, 26)
