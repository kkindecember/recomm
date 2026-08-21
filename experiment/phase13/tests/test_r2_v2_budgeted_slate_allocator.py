"""Contract tests for the frozen Phase-13 R²-v2 CBSA."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from experiment.phase13.protocol.r2_v2_budgeted_slate_allocator import (
    ACTIONS,
    BUDGETS,
    FEATURE_NAMES,
    FOLD_SALT,
    PRIMARY_BUDGET,
    BudgetConditionedAllocator,
    Standardizer,
    build_action_ranking,
    domain_balanced_batches,
    extract_features,
    guard_source_path,
    safe_argmax,
    stable_fold,
    summarize_source_gate,
    validate_static_contract,
)


def _rankings():
    gram = [f"g{index}" for index in range(1, 51)]
    resolver = ["r1", "r2", "r3", *gram]
    catalog = set(gram) | {"r1", "r2", "r3"}
    cold = {"r1", "r2", "r3"}
    return gram, resolver, catalog, cold


def test_action_definitions_match_preregistration():
    gram, resolver, catalog, cold = _rankings()
    action0, ranking0 = build_action_ranking(gram, resolver, "a0", catalog, cold)
    action2, ranking2 = build_action_ranking(gram, resolver, "a2", catalog, cold)
    action3, ranking3 = build_action_ranking(gram, resolver, "a3", catalog, cold)
    assert action0 == "a0" and ranking0 == gram
    assert action2 == "a2" and ranking2[:10] == [*gram[:8], "r1", "r2"]
    assert action3 == "a3" and ranking3[:10] == [*gram[:7], "r1", "r2", "r3"]
    for ranking in (ranking0, ranking2, ranking3):
        assert len(ranking) == 50
        assert len(ranking) == len(set(ranking))
        assert set(ranking) <= catalog


def test_action_degrades_deterministically_when_candidates_are_insufficient():
    gram = [f"g{index}" for index in range(1, 51)]
    catalog = set(gram) | {"r1", "r2"}
    cold = {"r1", "r2"}
    effective3, ranking3 = build_action_ranking(
        gram, ["r1", "r2"], "a3", catalog, cold
    )
    effective2, ranking2 = build_action_ranking(gram, ["r1"], "a2", catalog, cold)
    assert effective3 == "a2"
    assert ranking3[:10] == [*gram[:8], "r1", "r2"]
    assert effective2 == "a0"
    assert ranking2 == gram


def test_actions_use_the_frozen_b1_catalog_cold_candidate_pool():
    gram = [f"g{index}" for index in range(1, 51)]
    resolver = ["warm1", gram[7], "cold1", "cold2", "cold3", *gram]
    catalog = set(gram) | {"warm1", "cold1", "cold2", "cold3"}
    cold = {"cold1", "cold2", "cold3", gram[7]}

    action2, ranking2 = build_action_ranking(gram, resolver, "a2", catalog, cold)
    action3, ranking3 = build_action_ranking(gram, resolver, "a3", catalog, cold)

    # B1 protects top-7 while building one shared pool. gram[7] is eligible,
    # but stable exact-item de-duplication leaves it in its anchored rank.
    assert action2 == "a2"
    assert ranking2[:10] == [*gram[:8], "cold1", "g9"]
    assert action3 == "a3"
    assert ranking3[:10] == [*gram[:8], "cold1", "cold2"]


def test_warm_resolver_items_are_never_action_candidates():
    gram = [f"g{index}" for index in range(1, 51)]
    resolver = ["warm1", "cold1", "warm2", "cold2", "cold3", *gram]
    catalog = set(gram) | {"warm1", "warm2", "cold1", "cold2", "cold3"}
    cold = {"cold1", "cold2", "cold3"}
    _, ranking = build_action_ranking(gram, resolver, "a3", catalog, cold)
    assert ranking[:10] == [*gram[:7], "cold1", "cold2", "cold3"]
    assert "warm1" not in ranking[:10]


def test_target_leakage_contract_is_structural():
    validate_static_contract()
    parameters = inspect.signature(extract_features).parameters
    assert not {"target", "label", "reward", "is_cold"} & set(parameters)
    forbidden = ("target", "label", "reward", "oracle", "ndcg", "hit")
    assert all(not any(token in name for token in forbidden) for name in FEATURE_NAMES)


def test_feature_vector_has_frozen_schema_and_uses_catalog_state_only():
    gram, resolver, catalog, _cold = _rankings()
    cold = {"r1", "r2", "r3"}
    values = extract_features(
        gram,
        [float(50 - index) for index in range(50)],
        resolver[:50],
        [1.0 - index / 100 for index in range(50)],
        cold,
        catalog,
        ["g1", "r1", "g2"],
        [0.7, 0.6, 0.5],
    )
    assert len(values) == len(FEATURE_NAMES) == 36
    by_name = dict(zip(FEATURE_NAMES, values))
    assert by_name["history_length"] == 3
    assert by_name["history_cold_count"] == 1
    assert by_name["usable_unique_candidates_a3"] == 3


def test_standardizer_is_fit_on_training_fold_only_and_adds_missing_indicators():
    train = torch.zeros((4, len(FEATURE_NAMES)))
    train[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    train[0, 1] = torch.nan
    held = torch.zeros((1, len(FEATURE_NAMES)))
    held[0, 0] = 10_000.0
    standardizer = Standardizer.fit(train)
    transformed = standardizer.transform(held)
    assert standardizer.mean[0].item() == pytest.approx(1.5)
    assert transformed.shape[1] == 2 * len(FEATURE_NAMES)
    assert transformed[0, len(FEATURE_NAMES) + 1].item() == 0.0


def test_fold_assignment_is_domain_salted_and_deterministic():
    first = stable_fold("user-1", "Toys")
    assert first == stable_fold("user-1", "Toys")
    assert 0 <= first < 5
    assert FOLD_SALT in FOLD_SALT
    # Domain is part of the digest; compare many IDs to avoid asserting a
    # particular single hash collision cannot occur.
    assert any(
        stable_fold(f"user-{index}", "Toys") != stable_fold(f"user-{index}", "Beauty")
        for index in range(20)
    )


def test_sports_and_test_guards_fire_before_file_existence_checks(tmp_path):
    with pytest.raises(ValueError, match="Sports guard"):
        guard_source_path(tmp_path / "Sports_cold50" / "missing.jsonl")
    with pytest.raises(ValueError, match="Test guard"):
        guard_source_path(Path("/tmp/r2_v2_guard_case/predictions_test.tsv"))
    guard_source_path(Path("/tmp/r2_v2_guard_case/item_plain_text.txt"))


def test_safe_tie_break_prefers_a0_then_a2_then_a3():
    logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 2.0], [0.0, 1.0, 2.0]])
    assert safe_argmax(logits).tolist() == [0, 1, 2]


def test_budget_is_a_real_model_input_and_grid_is_frozen():
    assert ACTIONS == ("a0", "a2", "a3")
    assert BUDGETS == (0.93, 0.95, 0.97, 0.99)
    assert PRIMARY_BUDGET == 0.97
    torch.manual_seed(1)
    model = BudgetConditionedAllocator(2 * len(FEATURE_NAMES)).eval()
    features = torch.ones((2, 2 * len(FEATURE_NAMES)))
    with torch.no_grad():
        low = model(features, torch.full((2,), 0.93))
        high = model(features, torch.full((2,), 0.99))
    assert not torch.equal(low, high)
    assert model.net[0].in_features == 2 * len(FEATURE_NAMES) + 1
    assert model.net[0].out_features == 64
    assert model.net[3].out_features == 32
    assert model.net[-1].out_features == 3


class _Record:
    def __init__(self, domain: str):
        self.domain = domain


def test_domain_balanced_batches_have_equal_domain_counts():
    records = [_Record("Toys") for _ in range(7)] + [_Record("Beauty") for _ in range(11)]
    batches = list(domain_balanced_batches(records, 8, torch.Generator().manual_seed(3)))
    assert batches
    for batch in batches:
        domains = [records[index].domain for index in batch]
        assert domains.count("Toys") == domains.count("Beauty") == 4


def _gate_rows(events: int = 40):
    rows = []
    for domain in ("Toys", "Beauty"):
        for index in range(80):
            cold = index < 40
            baseline_hit50 = 1.0 if cold and index < events else 0.0
            rows.append({
                "domain": domain,
                "is_cold": cold,
                "selected_action": "a2" if index % 2 else "a0",
                "effective_action": "a2" if index % 2 else "a0",
                "fold_isolation": True,
                "catalog_unique": True,
                "portfolio2_ndcg10": 0.1,
                "cbsa_ndcg10": 0.11,
                "portfolio2_hit50": baseline_hit50,
                "cbsa_hit50": baseline_hit50,
            })
    return rows


def test_paired_domain_gate_passes_uniform_positive_utility_and_cold_tie():
    summary = summarize_source_gate(_gate_rows(events=40))
    assert summary["verdict"] == "PASS_TO_R2_V2_SPORTS_CONFIRMATION_DISCUSSION"
    assert summary["gate_states"] == {
        "overall_ndcg10": "PASS",
        "warm_ndcg10": "PASS",
        "cold_hit50_noninferiority": "PASS",
    }


def test_event_density_guard_forces_inconclusive():
    summary = summarize_source_gate(_gate_rows(events=10))
    assert summary["event_density_guard_triggered"] is True
    assert summary["verdict"] == "INCONCLUSIVE_STOP_R2_V2_SOURCE"
