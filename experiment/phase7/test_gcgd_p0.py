import math

import pytest

from experiment.phase7.gcgd_p0 import (
    encode_item_paths,
    graph_summary,
    parse_user_sequence,
    prefix_summary,
    validate_config,
)


class FakeTokenizer:
    values = {
        "alpha": [10, 11, 1],
        "beta": [10, 12, 1],
        "gamma": [20, 1820, 1],
    }

    def encode(self, value):
        return self.values[value]


def test_parse_user_sequence_requires_three_items():
    assert parse_user_sequence("u a b c\n") == ("u", ("a", "b", "c"))
    with pytest.raises(ValueError):
        parse_user_sequence("u a b")


def test_encode_item_paths_matches_split_trie_convention():
    paths = encode_item_paths(
        FakeTokenizer(), {"a": "alpha", "b": "beta", "c": "gamma"}, 0, [1820, 9175]
    )
    assert paths["a"] == (0, 10, 11, 1)
    assert paths["c"] == (0, 20, 1)


def test_duplicate_encoded_paths_fail_closed():
    with pytest.raises(ValueError, match="duplicate encoded lexical path"):
        encode_item_paths(FakeTokenizer(), {"a": "alpha", "b": "alpha"}, 0, [])


def test_graph_and_prefix_summary_use_catalog_and_conserve_mass():
    graph, degree = graph_summary({"u1": ("a", "b"), "u2": ("a", "a")}, {"a", "b", "c"})
    assert graph["train_interactions"] == 4
    assert graph["unique_user_item_edges"] == 3
    assert graph["cold_catalog_items"] == 1
    paths = {"a": (0, 10, 11, 1), "b": (0, 10, 12, 1), "c": (0, 20, 1)}
    prefix = prefix_summary(paths, degree)
    assert prefix["unique_encoded_item_paths"] == 3
    assert prefix["branching_by_depth"]["1"]["branching_max"] == 2
    assert prefix["pseudo_degree_probability_mass_max_abs_error"] <= 1e-12


def test_p0_config_requires_cpu_codellama_gpu0_and_no_holdout_consumption():
    config = {
        "decision_status": "PREREGISTERED_FROZEN_READY_TO_RUN",
        "execution_enabled": True,
        "execution": {
            "mode": "cpu_only",
            "cuda_visible_devices": "",
            "physical_gpu_reserved_by_codellama": 0,
            "codellama_reservation_mib": 30720,
            "background_tmux_required": True,
        },
        "graph": {"train_slice": "items[:-2]"},
        "integrity": {
            "checkpoint_loaded": False,
            "predictions_generated": False,
            "validation_or_test_target_values_consumed": False,
            "fresh_validation_read": False,
            "test_predictions_read": False,
            "sports_read": False,
        },
    }
    assert validate_config(config) == []
