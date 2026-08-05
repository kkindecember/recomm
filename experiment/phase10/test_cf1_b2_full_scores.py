import sys
from pathlib import Path


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_b2_full_scores import scientific_gate  # noqa: E402


def valid_metrics():
    return {
        "users": 19412,
        "total_candidates": 1698905,
        "cf_only_candidates": 728305,
        "valid_budget_fraction": 1.0,
        "legal_path_fraction": 1.0,
        "finite_fraction": 1.0,
        "G50_pearson": 0.999,
        "G50_spearman": 0.999,
        "G50_mean_top10_set_overlap": 0.99,
        "G50_recomputed_Hit@10": 0.1,
        "G50_cached_Hit@10": 0.1,
        "peak_allocated_mib": 2000,
        "wall_time_seconds": 3600,
    }


def test_full_gate_passes_frozen_identities():
    assert scientific_gate(valid_metrics())["status"] == "passed"


def test_full_gate_rejects_candidate_identity_drift():
    metrics = valid_metrics()
    metrics["cf_only_candidates"] -= 1
    assert scientific_gate(metrics)["status"] == "failed_full_score_gate"


def test_full_gate_rejects_timeout():
    metrics = valid_metrics()
    metrics["wall_time_seconds"] = 14400.1
    assert scientific_gate(metrics)["status"] == "failed_full_score_gate"

