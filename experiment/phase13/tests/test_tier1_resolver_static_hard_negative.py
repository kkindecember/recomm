import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).parents[1] / "protocol" / "tier1_resolver_static_hard_negative.py"
)
SPEC = importlib.util.spec_from_file_location("tier1_resolver_static_hard_negative", MODULE_PATH)
hardneg = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(hardneg)


def test_hard_negative_counts_require_zero_control_and_increasing_arms():
    assert hardneg.validate_hard_negative_counts([0, 8, 16, 32]) == [0, 8, 16, 32]
    for invalid in ([8, 16], [0], [0, 16, 8], [0, 8, 8], [0, -1, 8], []):
        try:
            hardneg.validate_hard_negative_counts(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid hard-negative counts accepted: {invalid}")


def test_static_lookup_is_warm_only_and_excludes_own_target():
    embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.7, 0.7],
            ]
        ),
        dim=1,
    )
    train_targets = torch.tensor([0, 0, 1, 2])
    warm_indices = torch.tensor([0, 1, 2, 3])
    lookup, audit = hardneg.build_static_warm_negative_lookup(
        train_targets,
        embeddings,
        warm_indices,
        max_count=2,
        device=torch.device("cpu"),
        chunk_size=2,
    )
    assert lookup[0, 0].item() == 1
    assert all(index in {0, 1, 2, 3} for index in lookup[train_targets].flatten().tolist())
    assert all(lookup[target].ne(target).all() for target in torch.unique(train_targets))
    assert audit["cold_negative_count"] == 0
    assert audit["self_negative_count"] == 0
    assert audit["all_hard_negatives_warm"] is True


def test_static_lookup_rejects_nonwarm_training_target():
    embeddings = F.normalize(torch.eye(3), dim=1)
    try:
        hardneg.build_static_warm_negative_lookup(
            torch.tensor([0, 2]),
            embeddings,
            torch.tensor([0, 1]),
            max_count=1,
            device=torch.device("cpu"),
            chunk_size=2,
        )
    except ValueError as error:
        assert "outside warm catalog" in str(error)
    else:
        raise AssertionError("nonwarm target was accepted")


def test_empty_hard_negative_loss_matches_inbatch_loss():
    user = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    target = user.clone()
    target_ids = torch.tensor([4, 7])
    expected = hardneg.rr.multi_positive_inbatch_loss(user, target, target_ids, 0.07)
    actual = hardneg.hard_negative_augmented_loss(user, target, target_ids, None, 0.07)
    assert torch.equal(actual, expected)


def test_hard_negative_addition_cannot_reduce_per_batch_loss():
    user = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    target = user.clone()
    target_ids = torch.tensor([4, 7])
    hard = F.normalize(torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]]), dim=2)
    baseline = hardneg.hard_negative_augmented_loss(user, target, target_ids, None, 0.07)
    augmented = hardneg.hard_negative_augmented_loss(user, target, target_ids, hard, 0.07)
    assert augmented > baseline


def test_bonferroni_bootstrap_interval_reports_requested_confidence():
    result = hardneg.paired_bootstrap_interval(
        [1.0] * 20,
        [0.0] * 20,
        resamples=200,
        seed=123,
        confidence=1.0 - 0.05 / 3,
    )
    assert result["confidence"] == 1.0 - 0.05 / 3
    assert result["ci"] == [1.0, 1.0]
    assert result["interpretation"] == "positive"
