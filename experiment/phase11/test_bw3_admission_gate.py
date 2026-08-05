import numpy as np

from train_bw3_admission_gate import anchor_apply, evaluate_margin


def test_anchor_apply_uses_frozen_anchor():
    values = np.asarray([1.0, 2.0, 3.0])
    anchor = np.asarray([1.0, 2.0])
    assert np.allclose(anchor_apply(values, anchor), (values - 1.5) / 0.5)


def test_margin_fallback_identity_when_no_admission():
    event = {
        "q1": 1,
        "target": 2,
        "target_frequency": 1,
        "base_top10": list(range(1, 11)),
        "base_rank": 2,
        "expansion": [{"candidate_id": 20, "features": np.zeros(8), "label": 0}],
    }
    model = {"mean": np.zeros(8), "std": np.ones(8), "weight": np.zeros(8), "bias": -10.0}
    result = evaluate_margin([event], model, 0.0)
    assert result["hit10_delta"] == 0
    assert result["admissions"] == 0
