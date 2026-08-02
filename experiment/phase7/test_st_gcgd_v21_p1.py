import pytest
import torch

from experiment.phase7.st_gcgd_v21_p1 import (
    AdvantageGate, ScalarGatePrefixProcessor, advantage_features,
    improvement_label, train_advantage_gate,
)
from experiment.phase7.st_gcgd_v2 import build_temporal_graph
from experiment.phase7.st_gcgd_v21 import TransitionFirstGraph


def test_advantage_gate_is_large_target_free_scalar_model():
    model = AdvantageGate(8)
    assert sum(p.numel() for p in model.parameters()) > 9000
    output = model(torch.rand(4, 8))
    assert output.shape == (4,) and torch.all((output > 0) & (output < 1))


def test_scalar_gate_fail_closes_below_threshold_and_changes_only_declared_tokens():
    scores = torch.zeros(1, 6)
    closed = ScalarGatePrefixProcessor({(0,): {2: -0.2}}, {(0,): 1.0}, alpha=.3, maximum_depth=4, scalar_gate=.4, threshold=.7)
    assert closed(torch.tensor([[0]]), scores).data_ptr() == scores.data_ptr()
    open_gate = ScalarGatePrefixProcessor({(0,): {2: -0.2, 4: -1.0}}, {(0,): 1.0}, alpha=.3, maximum_depth=4, scalar_gate=.9, threshold=.7)
    output = open_gate(torch.tensor([[0]]), scores)
    assert output[0, [0, 1, 3, 5]].tolist() == [0., 0., 0., 0.]
    assert output[0, 2] != 0 and output[0, 4] != 0


def test_improvement_label_requires_actual_better_candidate_rank():
    assert improvement_label("x", ["a", "x"], ["x", "a"]) == 1
    assert improvement_label("x", ["x", "a"], ["a", "x"]) == 0
    assert improvement_label("x", ["a"], ["x"]) == 1


def test_advantage_features_do_not_accept_or_use_target():
    values = {"a": 1., "b": 0., "c": -1.}
    feature = advantage_features(values, ["b", "b"], ["a", "c"], {1}, {"a": 0, "b": 1, "c": 2})
    assert len(feature) == 8
    assert all(0 <= value <= 1 for value in feature)


def test_gate_training_is_finite_with_imbalanced_labels():
    records = [{"feature": [float((i + j) % 3) / 2 for j in range(8)], "label": float(i == 0)} for i in range(10)]
    gate, history, diagnostics = train_advantage_gate(records, {"seed": 3, "learning_rate": .001, "weight_decay": .0001, "epochs": 3, "threshold": .7}, torch.device("cpu"))
    assert history and diagnostics["positive_labels"] == 1
    assert torch.isfinite(gate(torch.rand(2, 8))).all()


def test_cached_transition_logits_equal_full_forward():
    graph = build_temporal_graph({"u": ("a", "b", "c", "d")}, ("a", "b", "c", "d"), seed=2, calibration_fraction=.2, recency_decay=.9, skip_self_transitions=True)
    model = TransitionFirstGraph(1, 4, 8, .2, layers=1, dropout=0.)
    model.eval()
    sample = {"user_id": "u", "history_items": ("a", "b"), "sample_key": "u:k"}
    from experiment.phase7.st_gcgd_v21_p1 import transition_logits_for_sample
    direct = transition_logits_for_sample(model, graph, sample)
    cached = transition_logits_for_sample(model, graph, sample, model.propagate_deep(graph))
    assert cached == pytest.approx(direct)
