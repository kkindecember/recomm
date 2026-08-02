import math

import pytest
import torch

from experiment.phase7.st_gcgd_v2 import (
    TemporalMultiRelationGraph,
    audit_arm_rows,
    build_temporal_graph,
    deterministic_negatives,
    evaluate_scores,
    loss_for_scores,
)


def graph():
    return build_temporal_graph(
        {
            "u1": ["a", "b", "c", "d", "hold1", "hold2"],
            "u2": ["b", "b", "a", "c", "hold1", "hold2"],
            "u3": ["c", "a", "b", "d", "hold1", "hold2"],
        },
        ["a", "b", "c", "d", "hold1", "hold2", "z"],
        seed=2023,
        calibration_fraction=0.5,
        recency_decay=0.8,
        skip_self_transitions=True,
    )


def test_pseudo_future_and_holdouts_never_enter_edges():
    value = graph()
    item = {name: index for index, name in enumerate(value.items)}
    assert all(record.target not in record.prefix for record in value.records if record.user_id in {"u1", "u3"})
    edge_items = set(value.ui_edges[1].tolist())
    assert item["hold1"] not in edge_items and item["hold2"] not in edge_items
    assert value.records[0].target == item["d"]


def test_transition_direction_self_loop_and_recency_are_deterministic():
    value = graph()
    transitions = {tuple(edge) for edge in value.transition_edges.t().tolist()}
    item = {name: index for index, name in enumerate(value.items)}
    assert (item["a"], item["b"]) in transitions
    assert (item["b"], item["a"]) in transitions
    assert (item["b"], item["b"]) not in transitions
    u1 = value.users.index("u1")
    weights = {tuple(edge): float(weight) for edge, weight in zip(value.ui_edges.t().tolist(), value.ui_weights)}
    assert weights[(u1, item["c"])] > weights[(u1, item["a"])]


@pytest.mark.parametrize("arm", ["static", "ui", "transition", "full"])
def test_all_relation_arms_have_finite_differentiable_objective(arm):
    value = graph()
    model = TemporalMultiRelationGraph(len(value.users), len(value.items), 8)
    records = list(value.records)
    scores = model.scores(value, records, arm)
    targets = torch.tensor([record.target for record in records])
    negatives = deterministic_negatives(value, records, 2, 2023, {})
    loss = loss_for_scores(scores, targets, negatives, 1.0, 1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_hard_negatives_are_used_first_and_forbidden_items_are_rejected():
    value = graph()
    record = value.records[0]
    hard_item = "z"
    negatives = deterministic_negatives(value, [record], 3, 7, {record.sample_key: [record.user_id, hard_item]})
    assert negatives[0, 0].item() == value.items.index(hard_item)
    assert record.target not in negatives[0].tolist()
    assert not set(record.prefix) & set(negatives[0].tolist())


def test_evaluation_is_finite_and_reports_empty_subgroup_without_warning():
    value = graph()
    records = [record for record in value.records if record.target_group == "head"]
    if not records:
        records = [value.records[0]]
    scores = torch.zeros((len(records), len(value.items)))
    result = evaluate_scores(scores, records)
    assert all(math.isfinite(result[key]) for key in ("Recall@10", "NDCG@10", "Recall@50", "MRR", "mean_target_margin"))
    assert result["tail"]["available"] is False or result["head"]["available"] is False


def test_arm_audit_fails_closed_on_duplicate_unmatched_and_nonfinite_rows():
    good = {"A": [{"sample_key": "x", "metric": 1.0}], "E": [{"sample_key": "x", "metric": 2.0}]}
    assert audit_arm_rows(good, ["A", "E"], 1)["passed"] is True
    with pytest.raises(ValueError, match="duplicate"):
        audit_arm_rows({"A": good["A"], "E": good["E"] * 2}, ["A", "E"], 2)
    with pytest.raises(ValueError, match="unmatched"):
        audit_arm_rows({"A": good["A"], "E": [{"sample_key": "y", "metric": 1.0}]}, ["A", "E"], 1)
    with pytest.raises(ValueError, match="non-finite"):
        audit_arm_rows({"A": [{"sample_key": "x", "metric": float("nan")}]}, ["A"], 1)
