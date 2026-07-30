import torch

from experiment.phase5.cet_c2_optimization_audit import (
    legal_child_symmetric_kl,
    ordered_calibration_samples,
)
from utils import generation_trie as gt


def test_ordered_calibration_samples_is_deterministic_and_disjoint():
    sequences = {
        f"u{index}": ["a", "b", "c", "d", "e"] for index in range(20)
    }
    item2input = {item: item for item in "abcde"}
    item2lexid = {item: item for item in "abcde"}
    first = ordered_calibration_samples(
        "Toys",
        sequences,
        item2input,
        item2lexid,
        {"u1", "u2"},
        8,
        2,
        "salt",
    )
    second = ordered_calibration_samples(
        "Toys",
        sequences,
        item2input,
        item2lexid,
        {"u1", "u2"},
        8,
        2,
        "salt",
    )
    assert first == second
    assert not {row["user_id"] for row in first} & {"u1", "u2"}


def test_symmetric_kl_is_zero_for_identical_logits():
    trie = gt.Trie([[0, 2, 4], [0, 3, 4]])
    logits = torch.randn(1, 2, 8)
    value, competitive, eligible = legal_child_symmetric_kl(
        logits, logits.clone(), [[0, 2, 4]], trie, 4, 1.0
    )
    assert float(value) == 0.0
    assert competitive == 1
    assert eligible == 1


def test_symmetric_kl_is_positive_for_changed_distribution():
    trie = gt.Trie([[0, 2, 4], [0, 3, 4]])
    clean = torch.zeros(1, 2, 8)
    perturbed = clean.clone()
    perturbed[0, 0, 2] = 2.0
    value, competitive, _ = legal_child_symmetric_kl(
        clean, perturbed, [[0, 2, 4]], trie, 4, 1.0
    )
    assert float(value) > 0.0
    assert competitive == 1
