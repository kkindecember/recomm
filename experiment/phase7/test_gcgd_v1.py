import math

import pytest
import torch

from experiment.phase7.gcgd_v1 import (
    GraphReliabilityAdapter,
    LightGCN,
    aggregate_graph_prefix_logits,
    fuse_token_logits,
    normalize_item_logits,
    reliability_features,
    validate_design_config,
)


def test_normalized_item_logits_conserve_probability_mass():
    result = normalize_item_logits({"a": 1.0, "b": 2.0, "c": -1.0})
    assert sum(math.exp(value) for value in result.values()) == pytest.approx(1.0)


def test_prefix_projection_aggregates_leaf_probability_mass():
    paths = {"a": (1, 7), "b": (1, 8), "c": (2, 9)}
    result = aggregate_graph_prefix_logits(paths, {"a": 0.0, "b": 0.0, "c": 0.0}, ())
    assert math.exp(result[1]) == pytest.approx(2.0 / 3.0)
    assert math.exp(result[2]) == pytest.approx(1.0 / 3.0)
    nested = aggregate_graph_prefix_logits(paths, {"a": 0.0, "b": 1.0, "c": 20.0}, (1,))
    assert sum(math.exp(value) for value in nested.values()) == pytest.approx(1.0)
    assert set(nested) == {7, 8}


def test_prefix_projection_abstains_without_compatible_leaf():
    assert aggregate_graph_prefix_logits({"a": (1, 7)}, {"a": 0.0}, (2,)) == {}


def test_zero_gate_is_exact_gram_identity():
    gram = {1: -0.2, 2: -1.4}
    assert fuse_token_logits(gram, {1: -0.3, 2: -1.3}, alpha=1.0, gate=0.0) == gram


def test_graph_cannot_add_token_outside_legal_trie_set():
    with pytest.raises(ValueError, match="outside the legal Trie"):
        fuse_token_logits({1: -0.2}, {2: -0.3}, alpha=1.0, gate=1.0)


def test_design_config_enforces_gpu0_30g_lease_and_codellama_lifecycle():
    config = {
        "execution_enabled": False,
        "scientific_workload_implemented": False,
        "execution": {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "total_gpu_lease_mib": 30720,
            "expected_workload_peak_mib": 24576,
            "sidecar_reservation_mib": 6144,
            "background_tmux_required": True,
            "codellama_must_be_running_before_start": True,
            "stop_codellama_before_workload": True,
            "restore_codellama_after_every_exit": True,
            "no_automatic_retry": True,
        },
        "integrity": {
            "test_predictions_forbidden": True,
            "sports_forbidden": True,
        },
    }
    assert validate_design_config(config) == []


def test_lightgcn_propagation_and_bpr_are_finite():
    torch.manual_seed(7)
    model = LightGCN(users=2, items=3, embedding_dim=4, layers=2)
    edges = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    users, items = model.propagate(edges)
    assert users.shape == (2, 4)
    assert items.shape == (3, 4)
    loss = model.bpr_loss(
        edges,
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([2, 1]),
        1e-4,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_lightgcn_rejects_out_of_range_graph_indices():
    model = LightGCN(users=1, items=1, embedding_dim=2, layers=1)
    with pytest.raises(ValueError, match="item index"):
        model.propagate(torch.tensor([[0], [1]], dtype=torch.long))


def test_reliability_adapter_is_target_free_bounded_and_differentiable():
    features = torch.tensor(
        [reliability_features(
            graph_coverage=1.0,
            normalized_entropy=0.4,
            top_margin=0.3,
            compatible_leaf_fraction=0.6,
            gram_graph_agreement=1.0,
            normalized_depth=0.5,
        )]
    )
    adapter = GraphReliabilityAdapter()
    gate, temperature = adapter(features)
    assert 0.0 < gate.item() < 1.0
    assert 0.5 < temperature.item() < 2.0
    (gate.mean() + temperature).backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_reliability_features_fail_closed_outside_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability_features(
            graph_coverage=1.1,
            normalized_entropy=0.4,
            top_margin=0.3,
            compatible_leaf_fraction=0.6,
            gram_graph_agreement=1.0,
            normalized_depth=0.5,
        )


def test_domain_specific_leases_each_sum_to_exactly_30g():
    config = {
        "execution_enabled": False,
        "scientific_workload_implemented": False,
        "execution": {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "total_gpu_lease_mib": 30720,
            "domain_gpu_lease_mib": {
                "Toys": {"expected_workload_peak_mib": 4608, "sidecar_reservation_mib": 26112},
                "Beauty": {"expected_workload_peak_mib": 1792, "sidecar_reservation_mib": 28928},
            },
            "background_tmux_required": True,
            "codellama_must_be_running_before_start": True,
            "stop_codellama_before_workload": True,
            "restore_codellama_after_every_exit": True,
            "no_automatic_retry": True,
        },
        "integrity": {"test_predictions_forbidden": True, "sports_forbidden": True},
    }
    assert validate_design_config(config) == []
