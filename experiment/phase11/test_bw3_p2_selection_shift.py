import numpy as np

from diagnose_bw3_p2_selection_shift import (
    FEATURES,
    analyze_split,
    candidate_logits,
    feature_shift,
    quantiles,
)


def model():
    return {
        "mean": np.zeros(len(FEATURES)),
        "std": np.ones(len(FEATURES)),
        "weight": np.asarray([1.0] + [0.0] * (len(FEATURES) - 1)),
        "bias": 0.0,
        "margin": 0.0,
    }


def candidate(candidate_id, score):
    feature = np.zeros(len(FEATURES))
    feature[0] = score
    return {"candidate_id": candidate_id, "features": feature}


def event(user, target, candidates):
    return {
        "user": user,
        "target": target,
        "expansion": candidates,
    }


def test_candidate_logits_are_stable_and_include_feature_contributions():
    rows = candidate_logits(event("u", 1, [candidate(3, 1.0), candidate(2, 1.0)]), model())
    assert [row["candidate_id"] for row in rows] == [2, 3]
    assert rows[0]["logit"] == 1.0
    assert rows[0]["contributions"].shape == (len(FEATURES),)


def test_analysis_separates_margin_failure_from_top3_competition():
    rows = [
        event("below", 10, [candidate(10, -1.0), candidate(11, 0.5)]),
        event(
            "competed",
            20,
            [candidate(20, 0.1), candidate(21, 4.0), candidate(22, 3.0), candidate(23, 2.0)],
        ),
        event("selected", 30, [candidate(30, 1.0), candidate(31, 0.5)]),
    ]
    summary, targets, z, contributions = analyze_split("Toys", "synthetic", rows, model())
    assert summary["expansion_target_users"] == 3
    assert summary["target_below_margin"] == 1
    assert summary["target_passes_but_competition_rejects"] == 1
    assert summary["target_selected_top3"] == 1
    assert len(targets) == 3
    assert z.shape == contributions.shape == (3, len(FEATURES))


def test_feature_shift_orders_most_negative_contribution_first():
    calibration_z = np.zeros((2, len(FEATURES)))
    validation_z = np.zeros((2, len(FEATURES)))
    calibration_contrib = np.zeros((2, len(FEATURES)))
    validation_contrib = np.zeros((2, len(FEATURES)))
    validation_contrib[:, 3] = -2.0
    validation_contrib[:, 1] = -1.0
    rows = feature_shift(calibration_z, validation_z, calibration_contrib, validation_contrib)
    assert rows[0]["feature"] == FEATURES[3]
    assert rows[0]["logit_contribution_shift"] == -2.0


def test_quantiles_empty_and_nonempty():
    assert quantiles([])["median"] is None
    assert quantiles([1.0, 3.0])["median"] == 2.0
