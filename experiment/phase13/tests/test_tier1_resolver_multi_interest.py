import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).parents[1] / "protocol" / "tier1_resolver_multi_interest.py"
SPEC = importlib.util.spec_from_file_location("tier1_resolver_multi_interest", MODULE_PATH)
multi = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(multi)


def test_interest_counts_require_single_vector_control_and_increasing_arms():
    assert multi.validate_interest_counts([1, 2, 4]) == [1, 2, 4]
    for invalid in ([2, 4], [1], [1, 4, 2], [1, 2, 2], [1, 0, 2], []):
        try:
            multi.validate_interest_counts(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid interest counts accepted: {invalid}")


def test_single_interest_is_bitwise_identical_to_frozen_recency_pooling():
    embeddings = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1)
    expected = multi.rr.recency_weighted_history([0, 1, 2], embeddings, 0.85)
    actual = multi.deterministic_interest_vectors([0, 1, 2], embeddings, 1, 0.85, 5)
    assert torch.equal(actual[0], expected)


def test_semantic_clustering_is_deterministic_and_separates_two_interests():
    embeddings = F.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=1
    )
    first = multi.deterministic_interest_vectors([0, 1, 2, 3], embeddings, 2, 0.85, 5)
    second = multi.deterministic_interest_vectors([0, 1, 2, 3], embeddings, 2, 0.85, 5)
    assert torch.equal(first, second)
    assert len(first) == 2
    assert (first @ torch.tensor([1.0, 0.0])).max() > 0.99
    assert (first @ torch.tensor([0.0, 1.0])).max() > 0.99


def test_effective_interest_count_is_capped_by_history_length():
    embeddings = F.normalize(torch.eye(3), dim=1)
    vectors = multi.deterministic_interest_vectors([0, 1], embeddings, 4, 0.85, 5)
    assert vectors.shape == (2, 3)
    assert torch.isfinite(vectors).all()


def test_validation_history_excludes_validation_and_test_positions():
    sequences = [("u1", ["a", "b", "c", "d", "e"])]
    histories, audit = multi.build_validation_histories(
        sequences, {item: index for index, item in enumerate("abcde")}, max_history=2
    )
    assert histories == {"u1": [1, 2]}
    assert audit["history_end_exclusive"] == -2
    assert audit["validation_target_position"] == -2
    assert audit["held_out_test_position"] == -1


def test_max_interest_score_uses_best_interest_for_each_item():
    projected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    catalog = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=1)
    scores = multi.max_interest_scores(projected, catalog)
    assert torch.equal(scores, torch.tensor([1.0, 1.0, 0.0]))
