import copy

from experiment.phase6.gacr_v8_recover_validation import (
    aggregate_seed_rows,
    completed_run_integrity_passes,
    direct_comparison_rows,
    recompute_qualified_arms,
)


def _integrity():
    return {
        "all_fit_records_used": True,
        "fit_calibration_user_disjoint": True,
        "parent_checkpoint_sha_unchanged_during_training": True,
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }


def test_typed_integrity_accepts_zero_steps_and_false_forbidden_reads():
    assert completed_run_integrity_passes(_integrity())
    for key, bad in (
        ("backbone_optimizer_steps", 1),
        ("test_data_read", True),
        ("sports_data_read", True),
    ):
        evidence = _integrity()
        evidence[key] = bad
        assert not completed_run_integrity_passes(evidence)


def test_recomputed_gate_qualifies_only_arm_passing_every_domain_seed():
    config = {"datasets": ["Toys", "Beauty"], "training_seeds": [2023, 2024, 2025]}
    summary = {"integrity": _integrity(), "training": {}}
    for dataset in config["datasets"]:
        summary["training"][dataset] = {"arms": {}}
        for arm in ("D", "E"):
            summary["training"][dataset]["arms"][arm] = {
                str(seed): {
                    "finite_checkpoint": True,
                    "calibration_noninferiority": {"eligible": True},
                }
                for seed in config["training_seeds"]
            }
    summary["training"]["Beauty"]["arms"]["D"]["2024"]["calibration_noninferiority"][
        "eligible"
    ] = False
    assert recompute_qualified_arms(summary, config) == {"D": False, "E": True}


def _row(key, rank, recall10, ndcg10, recall50, group="head"):
    return {
        "sample_key": key,
        "target_group": group,
        "candidate_rank": rank,
        "union_covered": 1,
        "candidate_Recall@10": recall10,
        "candidate_NDCG@10": ndcg10,
        "candidate_Recall@50": recall50,
    }


def test_direct_comparison_uses_incumbent_as_baseline():
    rows = direct_comparison_rows(
        [_row("u", 11, 0.0, 0.0, 1.0)],
        [_row("u", 5, 1.0, 0.4, 1.0)],
    )
    assert rows[0]["baseline_rank"] == 11
    assert rows[0]["candidate_rank"] == 5
    assert rows[0]["baseline_NDCG@10"] == 0.0
    assert rows[0]["candidate_NDCG@10"] == 0.4


def test_three_seed_aggregation_means_each_user_before_bootstrap():
    base = direct_comparison_rows(
        [_row("u", 11, 0.0, 0.0, 1.0)],
        [_row("u", 5, 1.0, 0.4, 1.0)],
    )[0]
    low = copy.deepcopy(base)
    low["candidate_NDCG@10"] = 0.1
    middle = copy.deepcopy(base)
    middle["candidate_NDCG@10"] = 0.2
    high = copy.deepcopy(base)
    high["candidate_NDCG@10"] = 0.3
    result = aggregate_seed_rows({"2023": [low], "2024": [middle], "2025": [high]})
    assert abs(result[0]["candidate_NDCG@10"] - 0.2) < 1e-12
