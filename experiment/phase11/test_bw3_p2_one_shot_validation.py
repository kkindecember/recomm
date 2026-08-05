import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eval_bw3_p2_one_shot_validation import (
    EXPERIMENT_ID,
    FEATURES,
    atomic_reveal,
    build_event,
    evaluate_events,
    load_config,
    p2_scientific_gate,
    score_and_admit,
    update_status,
    validate_gate,
    verify_files,
)


def synthetic_gate(dataset="Toys"):
    return {
        "experiment_id": "GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1",
        "dataset": dataset,
        "objective": "per_user_listwise_cross_entropy_with_fixed_reject_logit_zero",
        "feature_schema": FEATURES,
        "weight": [1.0] + [0.0] * (len(FEATURES) - 1),
        "bias": 0.0,
        "feature_mean": [0.0] * len(FEATURES),
        "feature_std": [1.0] * len(FEATURES),
        "selected_margin": 0.0,
        "max_admissions": 3,
    }


def model(weight=None, bias=0.0, margin=0.0):
    return {
        "weight": np.asarray(weight or [1.0] + [0.0] * (len(FEATURES) - 1)),
        "bias": bias,
        "mean": np.zeros(len(FEATURES)),
        "std": np.ones(len(FEATURES)),
        "margin": margin,
    }


def event(user="u", target=1, base_rank=1, expansion=None, target_frequency=1, q1=1):
    return {
        "user": user,
        "target": target,
        "target_frequency": target_frequency,
        "q1": q1,
        "base_top10": list(range(1, 11)),
        "base_rank": base_rank,
        "in_beam50": target <= 50,
        "in_beam200": target <= 200,
        "expansion": expansion or [],
        "reliability": 0.5,
    }


def candidate(candidate_id, first_feature):
    values = np.zeros(len(FEATURES), dtype=np.float64)
    values[0] = first_feature
    return {"candidate_id": candidate_id, "features": values}


def test_gate_schema_and_margin_are_frozen():
    parsed = validate_gate(synthetic_gate(), "Toys")
    assert parsed["margin"] == 0.0
    assert parsed["weight"].shape == (9,)
    altered = synthetic_gate()
    altered["selected_margin"] = 0.25
    with pytest.raises(ValueError, match="margin"):
        validate_gate(altered, "Toys")


def test_score_admission_uses_logit_threshold_cap_and_stable_id_tie_break():
    row = event(
        expansion=[candidate(30, 2.0), candidate(20, 2.0), candidate(40, 1.0), candidate(50, -1.0)]
    )
    result = score_and_admit(row, model())
    assert result["admitted"] == [20, 30, 40]
    assert result["final_top10"] == list(range(1, 8)) + [20, 30, 40]


def test_no_admission_is_exact_fallback():
    row = event(expansion=[candidate(20, -1.0)])
    result = score_and_admit(row, model())
    assert result["fallback"] is True
    assert result["final_top10"] == row["base_top10"]


def test_build_event_has_exact_feature_schema_and_second_standardized_pcrf_base():
    base = {
        "candidate_ids": list(range(1, 51)),
        "seq": np.linspace(0.0, 3.0, 50),
        "cf": np.linspace(3.0, 0.0, 50),
        "candidate_frequencies": np.arange(1, 51, dtype=np.float64),
    }
    wide = {
        "candidate_ids": list(range(1, 201)),
        "seq": np.linspace(0.0, 4.0, 200),
        "cf": np.linspace(4.0, 0.0, 200),
        "candidate_frequencies": np.arange(1, 201, dtype=np.float64),
    }
    row = build_event("u", 1, 1, 5, base, wide)
    assert len(row["expansion"]) == 150
    assert row["expansion"][0]["features"].shape == (len(FEATURES),)
    assert row["expansion"][0]["features"][6] == 51 / 200
    assert np.isfinite(np.stack([entry["features"] for entry in row["expansion"]])).all()
    assert len(row["base_top10"]) == 10


def test_evaluate_events_computes_promotion_regression_tail_and_fallback():
    rows = [
        event("promoted", target=20, base_rank=201, expansion=[candidate(20, 2.0)]),
        event("stable", target=1, base_rank=1, expansion=[candidate(30, -2.0)]),
    ]
    summary, per_user = evaluate_events("Toys", rows, model(), 20, 2023)
    assert summary["promotions"] == 1
    assert summary["regressions"] == 0
    assert summary["admissions"] == 1
    assert summary["fallback_users"] == 1
    assert summary["hit10_delta"] == 0.5
    assert len(per_user) == 2


def domain_summary(dataset, hit_delta=0.002, ndcg_delta=0.0, tail_delta=0.0):
    return {
        "dataset": dataset,
        "hit10_delta": hit_delta,
        "ndcg10_delta": ndcg_delta,
        "tail_hit10_delta": tail_delta,
        "admissions": 1,
        "promotions": 1,
        "regressions": 0,
        "integrity": {"a": True},
    }


def test_p2_gate_matches_preregistered_cross_domain_thresholds():
    passed = p2_scientific_gate([domain_summary("Toys"), domain_summary("Beauty", 0.0)])
    assert passed["status"] == "passed_scientific_gate_awaiting_resource_audit"
    failed = p2_scientific_gate([domain_summary("Toys", -0.001), domain_summary("Beauty", 0.01)])
    assert failed["status"] == "failed_scientific_gate"
    assert failed["checks"]["both_hit10_nondegrade"] is False


def test_status_state_records_consumption_before_reveal(tmp_path):
    status = tmp_path / "status.json"
    update_status(
        status,
        validation_access_started=True,
        validation_consumed=True,
        results_revealed=False,
    )
    payload = json.loads(status.read_text())
    assert payload == {
        "validation_access_started": True,
        "validation_consumed": True,
        "results_revealed": False,
    }
    update_status(status, results_revealed=True)
    assert json.loads(status.read_text())["validation_consumed"] is True


def test_atomic_reveal_writes_both_domains_together_and_refuses_overwrite(tmp_path):
    output = tmp_path / "scientific"
    summaries = [domain_summary("Toys"), domain_summary("Beauty")]
    payload = {"datasets": summaries}
    rows = {
        "Toys": [{"user": "u", "rank": 1}],
        "Beauty": [{"user": "v", "rank": 2}],
    }
    atomic_reveal(output, payload, rows)
    assert (output / "Toys/per_user.tsv").is_file()
    assert (output / "Beauty/per_user.tsv").is_file()
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        atomic_reveal(output, payload, rows)


def test_input_sha_lock_passes_and_fails_closed(tmp_path):
    locked = tmp_path / "locked.txt"
    locked.write_text("frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    assert verify_files(tmp_path, {"locked.txt": digest}) == {"locked.txt": digest}
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_files(tmp_path, {"locked.txt": "0" * 64})


def test_disabled_config_cannot_be_loaded_for_formal_execution(tmp_path):
    config = {
        "experiment_id": EXPERIMENT_ID,
        "execution_enabled": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(PermissionError, match="disabled"):
        load_config(path)


def test_synthetic_tests_never_reference_real_validation_artifacts():
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = ["fresh" + "_beams_w50.tsv", "bw1" + "_candidate_ceiling"]
    assert all(value not in source for value in forbidden)
