import math

import pytest
import torch

from experiment.phase7.st_gcgd_v2 import build_temporal_graph, deterministic_negatives, loss_for_scores
from experiment.phase7.st_gcgd_v21 import TransitionFirstGraph, evaluate_scores_v21


def graph():
    return build_temporal_graph(
        {
            "u1": ["a", "b", "c", "d", "h1", "h2"],
            "u2": ["b", "c", "a", "d", "h1", "h2"],
            "u3": ["c", "a", "b", "e", "h1", "h2"],
        },
        ["a", "b", "c", "d", "e", "h1", "h2", "z"],
        seed=17,
        calibration_fraction=0.5,
        recency_decay=0.8,
        skip_self_transitions=True,
    )


def test_deep_relation_arms_are_finite_and_distinct():
    value = graph()
    model = TransitionFirstGraph(len(value.users), len(value.items), 16, 0.2, layers=2, dropout=0.0)
    records = list(value.records)
    outputs = {arm: model.scores(value, records, arm) for arm in ("static", "ui", "transition", "full")}
    assert all(output.shape == (3, 8) and torch.isfinite(output).all() for output in outputs.values())
    assert not torch.equal(outputs["transition"], outputs["ui"])


def test_transition_first_gate_starts_nearly_closed():
    value = graph()
    model = TransitionFirstGraph(len(value.users), len(value.items), 8, 0.2, layers=1, dropout=0.0)
    records = list(value.records)
    transition = model.scores(value, records, "transition")
    full = model.scores(value, records, "full")
    relative = (full - transition).abs().mean() / transition.abs().mean().clamp_min(1e-8)
    assert float(relative) < 0.02


def test_deep_transition_objective_has_finite_gradients():
    value = graph()
    model = TransitionFirstGraph(len(value.users), len(value.items), 16, 0.2, layers=2, dropout=0.0)
    records = list(value.records)
    scores = model.scores(value, records, "transition")
    targets = torch.tensor([record.target for record in records])
    negatives = deterministic_negatives(value, records, 2, 2023, {})
    loss = loss_for_scores(scores, targets, negatives, 1.0, 1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.session_gru.weight_ih_l0.grad is not None
    assert model.transition_layers[0]["out"].weight.grad is not None


def test_z_separation_is_scale_invariant_and_finite():
    value = graph()
    records = list(value.records)
    scores = torch.randn(len(records), len(value.items))
    first = evaluate_scores_v21(scores, records)
    second = evaluate_scores_v21(scores * 10.0, records)
    assert first["mean_target_z_separation"] == pytest.approx(second["mean_target_z_separation"], abs=1e-6)
    assert all(math.isfinite(first[key]) for key in ("mean_target_z_separation", "mean_target_margin"))
