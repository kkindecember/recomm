import sys
from pathlib import Path


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_a_candidate_union import (  # noqa: E402
    coverage,
    scientific_gate,
    union_diagnostics,
)


def test_union_diagnostics_counts_overlap_and_new_candidates():
    result = union_diagnostics([1, 2, 3], [3, 4, 5])
    assert result["union_size"] == 5
    assert result["intersection_size"] == 1
    assert result["cf_only_size"] == 2
    assert result["jaccard"] == 0.2


def test_coverage_accepts_generators():
    assert coverage(value for value in [True, False, True, True]) == 0.75


def test_gate_requires_coverage_tail_budget_and_identity():
    metrics = {
        "coverage": {"G50": 0.20, "U50": 0.24, "C50": 0.17463424685761383},
        "union_size": {"fraction_le_90": 0.85},
        "stratified": {"target_tail": {"complementary_C50_not_G50": 0.03}},
    }
    assert scientific_gate(metrics, True)["status"] == "passed"
    metrics["union_size"]["fraction_le_90"] = 0.79
    assert scientific_gate(metrics, True)["status"] == "failed_candidate_union_gate"


def test_smoke_does_not_evaluate_scientific_gate():
    assert scientific_gate({}, False)["status"] == "not_evaluated_smoke"
