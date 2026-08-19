from pathlib import Path
import inspect
import sys

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from setwise_candidate_selector import (  # noqa: E402
    SELECTOR_FEATURE_NAMES,
    SetwiseSelector,
    build_warm_transitions,
    is_pseudo_cold_item,
    selector_feature_tensor,
)


def test_item_partition_is_deterministic_and_non_degenerate():
    items = [f"item-{index}" for index in range(200)]
    first = [is_pseudo_cold_item(item) for item in items]
    second = [is_pseudo_cold_item(item) for item in items]
    assert first == second
    assert any(first) and not all(first)


def test_setwise_selector_is_permutation_equivariant():
    torch.manual_seed(1)
    model = SetwiseSelector(7, 8).eval()
    features = torch.randn(2, 5, 7)
    permutation = torch.tensor([2, 0, 4, 1, 3])
    original = model(features)
    permuted = model(features[:, permutation])
    assert torch.allclose(permuted, original[:, permutation], atol=1e-6)


def test_selector_features_are_target_free_and_shape_correct():
    projected = torch.tensor([[1.0, 0.0]])
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    candidates = torch.tensor([[0, 1]])
    histories = torch.tensor([[2]])
    mask = torch.tensor([[True]])
    features = selector_feature_tensor(projected, candidates, histories, mask, embeddings)
    assert features.shape == (1, 2, len(SELECTOR_FEATURE_NAMES))
    assert "target" not in inspect.signature(selector_feature_tensor).parameters


def test_warm_transition_builder_excludes_validation_and_test_targets():
    sequences = [("u", ["i0", "i1", "i2", "i3"])]
    item_to_idx = {f"i{index}": index for index in range(4)}
    rows = build_warm_transitions(sequences, item_to_idx, set(item_to_idx), max_history=20)
    assert rows == [([0], 1, "i1")]
