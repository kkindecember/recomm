import sys
from pathlib import Path


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_b1_arbitrary_score_pilot import scientific_gate  # noqa: E402


def passing_metrics():
    return {
        "valid_budget_fraction": 1.0,
        "legal_path_fraction": 1.0,
        "finite_fraction": 1.0,
        "G50_pearson": 0.999,
        "G50_spearman": 0.999,
        "G50_mean_top10_set_overlap": 0.99,
        "G50_recomputed_Hit@10": 0.1,
        "G50_cached_Hit@10": 0.1,
        "peak_allocated_mib": 2000,
        "wall_time_seconds": 100,
        "projected_full_validation_hours": 2.0,
    }


def test_gate_passes_all_frozen_checks():
    assert scientific_gate(passing_metrics())["status"] == "passed"


def test_gate_rejects_illegal_paths():
    metrics = passing_metrics()
    metrics["legal_path_fraction"] = 0.999
    assert scientific_gate(metrics)["status"] == "failed_arbitrary_score_resource_gate"


def test_gate_rejects_slow_full_projection():
    metrics = passing_metrics()
    metrics["projected_full_validation_hours"] = 4.01
    assert scientific_gate(metrics)["status"] == "failed_arbitrary_score_resource_gate"

