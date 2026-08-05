import sys
from pathlib import Path


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_a2_budgeted_union import (  # noqa: E402
    adaptive_history_slots,
    fill_cf_only,
    fixed_prefix_union,
    scientific_gate,
)


def test_fill_cf_only_uses_rank_order_and_fills_unique_slots():
    assert fill_cf_only([1, 2, 3], [3, 4, 2, 5, 6], 2) == [1, 2, 3, 4, 5]


def test_fixed_prefix_does_not_backfill_overlap():
    assert fixed_prefix_union([1, 2, 3], [3, 4, 5], 2) == [1, 2, 3, 4]


def test_adaptive_history_schedule_is_frozen():
    assert [adaptive_history_slots(x) for x in (1, 5, 6, 10, 11, 20)] == [25, 25, 30, 30, 40, 40]


def test_gate_requires_budget_and_retention():
    metrics = {
        "G50_coverage": 0.21193076447558212,
        "U50_coverage": 0.2666907067793118,
        "U50_tail_complementary_not_G50": 0.023449612403100777,
        "policies": {"fill_cf_only_40": {
            "coverage": 0.26,
            "tail_complementary_not_G50": 0.02,
            "fraction_le_90": 1.0,
        }},
    }
    assert scientific_gate(metrics, True)["status"] == "passed"
    metrics["policies"]["fill_cf_only_40"]["fraction_le_90"] = 0.99
    assert scientific_gate(metrics, True)["status"] == "failed_budgeted_union_gate"


def test_smoke_does_not_evaluate_gate():
    assert scientific_gate({}, False)["status"] == "not_evaluated_smoke"

