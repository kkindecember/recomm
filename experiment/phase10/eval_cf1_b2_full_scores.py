#!/usr/bin/env python3
"""Full-validation wrapper around the frozen CF1-B1 arbitrary path scorer."""

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE10 = REPO_ROOT / "experiment/phase10"
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

import eval_cf1_b1_arbitrary_score_pilot as b1  # noqa: E402


EXPECTED_USERS = 19412
EXPECTED_TOTAL_CANDIDATES = 1698905
EXPECTED_CF_ONLY_CANDIDATES = 728305


def scientific_gate(metrics):
    checks = {
        "users_exact_19412": metrics["users"] == EXPECTED_USERS,
        "total_candidates_identity": metrics["total_candidates"] == EXPECTED_TOTAL_CANDIDATES,
        "cf_only_candidates_identity": metrics["cf_only_candidates"] == EXPECTED_CF_ONLY_CANDIDATES,
        "all_users_valid_budget": metrics["valid_budget_fraction"] == 1.0,
        "all_paths_legal": metrics["legal_path_fraction"] == 1.0,
        "all_scores_finite": metrics["finite_fraction"] == 1.0,
        "G50_pearson_at_least_0.995": metrics["G50_pearson"] >= 0.995,
        "G50_spearman_at_least_0.995": metrics["G50_spearman"] >= 0.995,
        "G50_top10_overlap_at_least_0.98": metrics["G50_mean_top10_set_overlap"] >= 0.98,
        "G50_hit10_identity_within_0.001": abs(metrics["G50_recomputed_Hit@10"] - metrics["G50_cached_Hit@10"]) <= 0.001,
        "peak_allocated_mib_at_most_12000": metrics["peak_allocated_mib"] <= 12000,
        "wall_time_at_most_4h": metrics["wall_time_seconds"] <= 14400,
    }
    return {"status": "passed" if all(checks.values()) else "failed_full_score_gate", "checks": checks}


def main():
    b1.scientific_gate = scientific_gate
    b1.main()
    args = b1.parse_args()
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment_id"] = "GRAM_PHASE10_CF1_B2_TOYS_FULL_SCORE_V1"
    summary["full_validation_run"] = True
    summary["upstream_scorer"] = "CF1-B1 frozen arbitrary-candidate scorer"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

