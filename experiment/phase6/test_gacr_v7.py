import math
from pathlib import Path

import torch

from experiment.phase6.gacr_v2 import build_validation_records
from experiment.phase6.gacr_v7 import (
    assess_calibration_noninferiority,
    cutoff_discount,
    metric_aligned_pairwise_loss,
    metric_pair_weights,
    train_metric_aligned_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cutoff_weights_match_preregistered_formula():
    base = torch.arange(60, 0, -1, dtype=torch.float32)
    weights = metric_pair_weights(base, target_index=9)  # stable rank 10
    expected_rank_11 = abs(cutoff_discount(10) - cutoff_discount(11))
    expected_rank_51 = expected_rank_11 + 0.25
    assert weights[9].item() == 0.0
    assert math.isclose(weights[10].item(), expected_rank_11, rel_tol=1e-6)
    assert math.isclose(weights[50].item(), expected_rank_51, rel_tol=1e-6)


def test_zero_weight_record_is_excluded():
    base = torch.tensor([1.0])
    residual = torch.zeros_like(base, requires_grad=True)
    assert metric_aligned_pairwise_loss(base, residual, 0) is None


def test_pairwise_loss_is_finite_and_prefers_target():
    base = torch.tensor([1.0, 0.5, 0.0])
    neutral = metric_aligned_pairwise_loss(base, torch.zeros(3), 1)
    favored = metric_aligned_pairwise_loss(
        base, torch.tensor([0.0, 1.0, 0.0]), 1
    )
    assert neutral is not None and favored is not None
    assert torch.isfinite(neutral)
    assert favored < neutral


def _groups(
    broad=0.01,
    overall_r10=-0.002,
    overall_r50=-0.002,
    tail_r50=-0.004,
    tail_ndcg=-0.0005,
):
    return {
        "overall": {
            "baseline_Recall@10": 0.20,
            "candidate_Recall@10": 0.20 + overall_r10,
            "baseline_Recall@50": 0.30,
            "candidate_Recall@50": 0.30 + overall_r50,
            "broad_harm_rate": broad,
        },
        "tail": {
            "baseline_Recall@50": 0.20,
            "candidate_Recall@50": 0.20 + tail_r50,
            "baseline_NDCG@10": 0.05,
            "candidate_NDCG@10": 0.05 + tail_ndcg,
        },
    }


def _gate_config():
    return {
        "calibration_noninferiority": {
            "broad_harm_max": 0.01,
            "overall_recall10_absolute_delta_min": -0.002,
            "overall_recall50_absolute_delta_min": -0.002,
            "tail_recall50_absolute_delta_min": -0.004,
            "tail_ndcg10_absolute_delta_min": -0.0005,
        }
    }


def test_calibration_noninferiority_accepts_boundaries_and_rejects_breach():
    assert assess_calibration_noninferiority(_groups(), _gate_config())["eligible"]
    failed = assess_calibration_noninferiority(
        _groups(overall_r50=-0.0021), _gate_config()
    )
    assert not failed["eligible"]
    assert not failed["checks"]["overall_recall50"]


def _record(group, target_index=1):
    return {
        "sample_key": group,
        "target_group": group,
        "gram_rank": 2,
        "target_index": target_index,
        "base": torch.tensor([1.0, 0.5, 0.0]),
        "features": torch.tensor(
            [[1.0] * 6, [0.5] * 6, [-0.5] * 6], dtype=torch.float32
        ),
    }


def test_metric_training_runs_fixed_steps_and_preserves_identity_at_init():
    config = {
        "residual": {
            "hidden_dim": 16,
            "bound": 0.2,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "fixed_training_step": 2,
        },
        "deployment_scale": 1.0,
    }
    result = train_metric_aligned_seed(
        [_record("head"), _record("tail")],
        [_record("head"), _record("tail")],
        config,
        2023,
        torch.device("cpu"),
        chunk_size=1,
    )
    assert result["optimizer_steps"] == 2
    assert result["effective_head_records"] == 1
    assert result["effective_tail_records"] == 1
    assert result["zero_residual_identity_rate"] == 1.0
    assert result["finite_checkpoint"]


def test_validation_excludes_all_prior_cohorts_without_test_or_sports(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "experiment.phase6.gacr_v2.prepare",
        lambda *args: {
            "model": type(
                "Model",
                (),
                {
                    "load_state_dict": lambda *args, **kwargs: None,
                    "eval": lambda self: None,
                },
            )(),
            "sequences": {"u": [1]},
            "item2input": {},
            "item2lexid": {},
        },
    )
    monkeypatch.setattr("experiment.phase6.gacr_v2.sha256", lambda *args: "sha")
    monkeypatch.setattr("experiment.phase6.gacr_v2.read_users", lambda *args: set())
    monkeypatch.setattr(
        "experiment.phase6.gacr_v2.select_fresh_validation_users",
        lambda all_users, exclusions, dataset, salt, count: captured.append(salt) or [],
    )
    monkeypatch.setattr(
        "experiment.phase6.gacr_v2.build_validation_samples", lambda *args: []
    )
    monkeypatch.setattr("experiment.phase6.gacr_v2.torch.load", lambda *args, **kwargs: {})
    config = {
        "inputs": {"checkpoint_root": "unused", "split_root": "unused"},
        "prior_validation_salts": [f"prior-{index}" for index in range(6)],
        "validation_salt": "fresh-v7",
        "validation_users_per_dataset": 1,
    }
    metadata, records = build_validation_records(
        "Toys", config, {}, torch.device("cpu")
    )
    assert records == []
    assert metadata["prior_validation_cohorts_excluded"] == 6
    assert captured == [f"prior-{index}" for index in range(6)] + ["fresh-v7"]


def test_v7_source_has_no_test_or_sports_input_paths():
    source = (REPO_ROOT / "experiment/phase6/gacr_v7.py").read_text()
    assert "test_users.txt" not in source
    assert '"Sports"' not in source
