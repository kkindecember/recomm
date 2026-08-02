import math

import pytest
import torch

from experiment.phase7.gcgd_p0_gpu_smoke import (
    GraphPrefixLogitsProcessor,
    prefix_log_probabilities,
    validate_config,
)


def test_prefix_probabilities_conserve_mass_at_every_node():
    paths = {"a": (0, 3, 1), "b": (0, 4, 1), "c": (0, 4, 2, 1)}
    result = prefix_log_probabilities(paths, {"a": 0.0, "b": 1.0, "c": 2.0})
    for children in result.values():
        assert sum(math.exp(value) for value in children.values()) == pytest.approx(1.0)


def test_zero_mass_children_are_omitted_and_zero_only_prefix_abstains():
    paths = {"seen": (0, 2, 7, 1), "unseen": (0, 3, 8, 1)}
    result = prefix_log_probabilities(paths, {"seen": -10000.0, "unseen": 0.0})
    assert result[(0,)] == {3: 0.0}
    assert (0, 2) not in result
    assert (0, 2, 7) not in result
    assert result[(0, 3)] == {8: 0.0}


def test_zero_alpha_processor_is_bitwise_identity():
    scores = torch.tensor([[0.1, 0.2, 0.3]])
    processor = GraphPrefixLogitsProcessor({(0,): {2: -0.7}}, 0.0)
    output = processor(torch.tensor([[0]]), scores)
    assert output.data_ptr() == scores.data_ptr()
    assert torch.equal(output, scores)


def test_active_processor_only_changes_declared_legal_children():
    scores = torch.zeros((1, 5))
    processor = GraphPrefixLogitsProcessor({(0,): {2: -0.2, 4: -1.0}}, 0.5)
    output = processor(torch.tensor([[0]]), scores)
    assert output[0, 2].item() == pytest.approx(-0.1)
    assert output[0, 4].item() == pytest.approx(-0.5)
    assert output[0, [0, 1, 3]].tolist() == [0.0, 0.0, 0.0]
    assert processor.applied_rows == 1


def test_config_requires_gpu0_exact_30g_lease_and_no_holdout_reads():
    config = {
        "decision_status": "PREREGISTERED_FROZEN_READY_TO_RUN",
        "execution_enabled": True,
        "execution": {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "total_gpu_lease_mib": 30720,
            "expected_workload_peak_mib": 24576,
            "sidecar_reservation_mib": 6144,
        },
        "integrity": {
            "fresh_validation_read": False,
            "test_predictions_read": False,
            "sports_read": False,
        },
    }
    assert validate_config(config) == []
