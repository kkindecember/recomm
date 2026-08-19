from pathlib import Path
import inspect
import sys

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from confidence_abstention import (  # noqa: E402
    FEATURE_NAMES,
    auc_roc,
    extract_inference_features,
    stable_fold,
)


def test_stable_fold_is_deterministic_and_covers_all_folds():
    users = [f"user-{i}" for i in range(200)]
    first = [stable_fold(uid, 5) for uid in users]
    second = [stable_fold(uid, 5) for uid in users]
    assert first == second
    assert set(first) == set(range(5))


def test_auc_roc_handles_perfect_reverse_and_ties():
    labels = torch.tensor([0, 0, 1, 1])
    assert auc_roc(labels, torch.tensor([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_roc(labels, torch.tensor([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc_roc(labels, torch.ones(4)) == 0.5


def test_inference_feature_function_has_no_target_argument():
    parameters = inspect.signature(extract_inference_features).parameters
    assert "target" not in parameters
    assert len(FEATURE_NAMES) == 6


def test_feature_names_are_candidate_or_model_signals_only():
    forbidden = ("target", "label", "is_cold_user")
    for name in FEATURE_NAMES:
        assert not any(token in name for token in forbidden)
