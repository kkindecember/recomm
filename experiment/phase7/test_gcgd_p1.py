import pytest
import torch

from experiment.phase7.gcgd_p1 import (
    AdaptiveGraphPrefixLogitsProcessor,
    adapter_training_loss,
    arm_metric_row,
    build_indexed_graph,
    graph_logits_for_user,
    graph_prefix_inputs,
    lightgcn_epoch_loss,
    metrics_for_rank,
    seeded_negative_items,
    summarize_metric_rows,
)
from experiment.phase7.gcgd_v1 import GraphReliabilityAdapter, LightGCN


def test_indexed_graph_deduplicates_repeated_visible_interactions():
    graph = build_indexed_graph({"u2": ("b",), "u1": ("a", "a", "b")}, ("a", "b", "c"))
    assert graph.users == ("u1", "u2")
    assert graph.edges.tolist() == [[0, 0, 1], [0, 1, 1]]


def test_seeded_negatives_are_deterministic_and_not_in_history():
    graph = build_indexed_graph({"u1": ("a",), "u2": ("b",)}, ("a", "b", "c"))
    first = seeded_negative_items(graph, graph.edges[0], seed=9)
    second = seeded_negative_items(graph, graph.edges[0], seed=9)
    assert torch.equal(first, second)
    for user, negative in zip(graph.edges[0].tolist(), first.tolist()):
        assert negative not in graph.user_history[user]


def test_lightgcn_epoch_loss_is_finite_and_differentiable():
    graph = build_indexed_graph({"u1": ("a",), "u2": ("b",)}, ("a", "b", "c"))
    model = LightGCN(2, 3, 4, 1)
    negatives = seeded_negative_items(graph, graph.edges[0], seed=3)
    loss = lightgcn_epoch_loss(model, graph.edges, negatives, batch_size=1, l2=1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embedding.weight.grad is not None


def test_lightgcn_training_rejects_nonpositive_objective_weight():
    from experiment.phase7.gcgd_p1 import train_lightgcn

    graph = build_indexed_graph({"u1": ("a",)}, ("a", "b"))
    config = {
        "seed": 1,
        "embedding_dim": 2,
        "layers": 0,
        "learning_rate": 0.001,
        "epochs": 1,
        "batch_size": 1,
        "l2": 0.0,
        "objective_weight": 0.0,
    }
    with pytest.raises(ValueError, match="objective_weight"):
        train_lightgcn(graph, config, torch.device("cpu"))


def test_graph_logits_mask_only_sample_time_visible_history():
    graph = build_indexed_graph({"u1": ("a", "b")}, ("a", "b", "c"))
    model = LightGCN(1, 3, 2, 0)
    logits = graph_logits_for_user(
        model, graph, "u1", visible_history_items=("a",), seen_item_sentinel=-10000.0
    )
    assert logits["a"] == -10000.0
    assert logits["b"] != -10000.0


def test_graph_logits_accept_locked_propagated_embedding_cache():
    graph = build_indexed_graph({"u1": ("a",)}, ("a", "b"))
    model = LightGCN(1, 2, 2, 0)
    cached = model.propagate(graph.edges)
    direct = graph_logits_for_user(model, graph, "u1", visible_history_items=())
    reused = graph_logits_for_user(
        model, graph, "u1", visible_history_items=(), propagated_embeddings=cached
    )
    assert reused == direct


def test_rank_metrics_and_summary_cover_all_preregistered_cutoffs():
    top = metrics_for_rank(3)
    miss = metrics_for_rank(None)
    assert top["Recall@5"] == 1.0 and top["MRR"] == pytest.approx(1 / 3)
    assert all(value == 0.0 for value in miss.values())
    summary = summarize_metric_rows([top, miss])
    assert summary["n"] == 2
    assert summary["Recall@10"] == 0.5


def test_graph_prefix_inputs_track_probability_and_compatible_leaf_fraction():
    scores, fractions = graph_prefix_inputs(
        {"a": (0, 2, 1), "b": (0, 3, 1)}, {"a": 0.0, "b": 0.0}
    )
    assert sum(torch.exp(torch.tensor(list(scores[(0,)].values())))).item() == pytest.approx(1.0)
    assert fractions[(0,)] == 1.0
    assert fractions[(0, 2)] == 0.5


def test_fixed_and_adaptive_processors_never_change_undeclared_tokens():
    prefix_scores = {(0,): {2: -0.2, 4: -1.0}}
    fractions = {(0,): 1.0}
    scores = torch.zeros((1, 6))
    fixed = AdaptiveGraphPrefixLogitsProcessor(
        prefix_scores, fractions, alpha=0.3, maximum_depth=4, adapter=None
    )
    fixed_output = fixed(torch.tensor([[0]]), scores)
    assert fixed_output[0, [0, 1, 3, 5]].tolist() == [0.0, 0.0, 0.0, 0.0]
    adapter = GraphReliabilityAdapter()
    adaptive = AdaptiveGraphPrefixLogitsProcessor(
        prefix_scores, fractions, alpha=0.5, maximum_depth=4, adapter=adapter
    )
    adaptive_output = adaptive(torch.tensor([[0]]), scores)
    assert adaptive_output[0, [0, 1, 3, 5]].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert 0.0 < adaptive.gates[0] < 1.0


def test_adaptive_processor_alpha_zero_is_bitwise_identity():
    scores = torch.randn(2, 5)
    processor = AdaptiveGraphPrefixLogitsProcessor(
        {(0,): {2: -0.2}}, {(0,): 1.0}, alpha=0.0, maximum_depth=3, adapter=GraphReliabilityAdapter()
    )
    output = processor(torch.tensor([[0], [0]]), scores)
    assert output.data_ptr() == scores.data_ptr()


def test_adapter_training_loss_has_finite_gradients_and_components():
    adapter = GraphReliabilityAdapter()
    features = torch.rand(3, 6)
    gram = torch.randn(3, 4)
    graph = torch.log_softmax(torch.randn(3, 4), dim=-1)
    total, components = adapter_training_loss(
        adapter,
        features,
        gram,
        graph,
        torch.tensor([0, 1, 2]),
        torch.tensor([1.0, 0.0, 1.0]),
        alpha=0.5,
        next_token_weight=1.0,
        reliability_weight=0.1,
    )
    assert torch.isfinite(total)
    assert set(components) == {"next_token_ce", "gate_reliability_bce", "temperature"}
    total.backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_arm_metric_row_reports_new_hits_changes_and_broad_harm():
    new_hit = arm_metric_row(
        sample_key="u:v:x",
        target="x",
        baseline_items=["a", "b"],
        candidate_items=["x", "b"],
        target_group="tail",
        graph_covered=True,
    )
    assert new_hit["new_hit_at10_outside_A_beam"] == 1
    assert new_hit["changed"] == 1
    harm = arm_metric_row(
        sample_key="u:v:a",
        target="a",
        baseline_items=["a"],
        candidate_items=["b"],
        target_group="head",
        graph_covered=True,
    )
    assert harm["broad_harm"] == 1
