from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_external_evaluator import (
    aggregate_metrics,
    compare_predictions,
    fp1_gate,
    fp2_gate,
    paired_bootstrap_delta,
    psid_collision_diagnostics,
    subgroup_assignments,
    subgroup_comparison,
    summarize_mechanisms,
    user_ranking_metrics,
    validate_prediction_rows,
)
from experiment.phase17.core.full_latte_external_inference import (
    _load_trusted_checkpoint,
)
from experiment.phase17.core.fullport_data import (
    APPROVED_D0_SUFFIX,
    FullportExternalExample,
    FullportTrainUser,
    materialize_external_evaluation_view,
)


def prediction_rows(arm: str, variant: str, rankings: list[list[str]]):
    examples = [
        FullportExternalExample("u1", ("a",), "x"),
        FullportExternalExample("u2", ("b", "c"), "y"),
        FullportExternalExample("u3", ("d", "e", "f", "g"), "z"),
    ]
    rows = []
    for example, ranking in zip(examples, rankings):
        try:
            target_rank = ranking.index(example.target) + 1
        except ValueError:
            target_rank = 0
        rows.append(
            {
                "arm_id": arm,
                "variant": variant,
                "user_id": example.user_id,
                "target": example.target,
                "ranking": ranking,
                "target_rank": target_rank,
            }
        )
    return examples, rows


class ExternalEvaluatorTests(unittest.TestCase):
    def test_checkpoint_loader_adapts_to_old_and_new_torch_signatures(self) -> None:
        calls = []

        class OldTorch:
            @staticmethod
            def load(path, map_location=None, **pickle_load_args):
                calls.append((path, map_location, pickle_load_args))
                return {"version": "old"}

        class NewTorch:
            @staticmethod
            def load(path, map_location=None, weights_only=True):
                calls.append((path, map_location, {"weights_only": weights_only}))
                return {"version": "new"}

        self.assertEqual(
            _load_trusted_checkpoint(OldTorch, Path("old.pt")), {"version": "old"}
        )
        self.assertEqual(
            _load_trusted_checkpoint(NewTorch, Path("new.pt")), {"version": "new"}
        )
        self.assertEqual(calls[0], (Path("old.pt"), "cpu", {}))
        self.assertEqual(
            calls[1], (Path("new.pt"), "cpu", {"weights_only": False})
        )

    def test_single_target_metrics_cover_frozen_outputs(self) -> None:
        first = user_ranking_metrics("x", ["x", "a"])
        self.assertEqual(first["hit@5"], 1.0)
        self.assertEqual(first["ndcg@10"], 1.0)
        self.assertEqual(first["mrr@10"], 1.0)
        missed = user_ranking_metrics("x", [str(index) for index in range(50)])
        self.assertEqual(missed["hit@50"], 0.0)
        with self.assertRaises(ValueError):
            user_ranking_metrics("x", ["a", "a"])

    def test_prediction_alignment_and_paired_effects(self) -> None:
        examples, treatment_rows = prediction_rows(
            "T", "beam500_agg_max", [["x"], ["y"], ["z"]]
        )
        _, control_rows = prediction_rows(
            "C", "beam500_identity", [["a"], ["b"], ["z"]]
        )
        treatment = validate_prediction_rows(treatment_rows, examples, expected_arm_id="T")
        control = validate_prediction_rows(control_rows, examples, expected_arm_id="C")
        report = compare_predictions(
            treatment,
            control,
            treatment_label="T",
            control_label="C",
            replicates=200,
            seed=7,
        )
        self.assertGreater(report["effects"]["ndcg@10"]["mean_delta"], 0)
        self.assertEqual(report["primary_user_outcomes"]["gain"], 2)
        self.assertEqual(report["primary_user_outcomes"]["tie"], 1)
        self.assertEqual(aggregate_metrics(treatment)["hit@10"], 1.0)

    def test_bootstrap_is_deterministic_and_chunk_independent(self) -> None:
        first = paired_bootstrap_delta(
            [1, 1, 0, 1], [0, 0, 0, 0], replicates=250, seed=9, chunk_size=7
        )
        second = paired_bootstrap_delta(
            [1, 1, 0, 1], [0, 0, 0, 0], replicates=250, seed=9, chunk_size=31
        )
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 0.75)

    def test_subgroups_use_train_prefix_only_frequency_and_memory(self) -> None:
        users = [
            FullportTrainUser("u1", ("x",)),
            FullportTrainUser("u2", ("a", "a", "y", "b")),
            FullportTrainUser("u3", tuple(["c"] * 10)),
        ]
        examples = [
            FullportExternalExample("u1", ("x",), "x"),
            FullportExternalExample("u2", ("a", "a", "y", "b"), "y"),
            FullportExternalExample("u3", tuple(["c"] * 10), "z"),
        ]
        assignments, thresholds = subgroup_assignments(users, examples)
        self.assertEqual(assignments["u1"]["history_length"], "short_le3")
        self.assertEqual(assignments["u2"]["history_length"], "medium_4_9")
        self.assertEqual(assignments["u3"]["history_length"], "long_ge10")
        self.assertEqual(assignments["u1"]["memory"], "memorization")
        self.assertEqual(assignments["u3"]["memory"], "generalization")
        self.assertIn("target_train_frequency_q1", thresholds)

    def test_mechanisms_and_gates_are_fail_closed(self) -> None:
        examples, rows = prediction_rows(
            "N1", "beam500_agg_max", [["x"], ["y"], ["z"]]
        )
        for row in rows:
            row["mechanism"] = {
                "latent_counts": {"1": 2, "2": 1},
                "latent_root_count": 2,
                "generated_path_count": 3,
                "valid_path_count": 3,
                "valid_path_rate": 1.0,
                "unique_item_count": 2,
                "duplicate_item_path_count": 1,
                "duplicate_path_rate": 1 / 3,
                "multi_path_item_rate": 0.5,
                "target_path_survived": 1.0,
                "pre_aggregation_target_rank": 1,
                "post_aggregation_target_rank": 1,
                "pre_aggregation_ndcg@10": 1.0,
                "post_aggregation_ndcg@10": 1.0,
                "aggregation_gain_ndcg@10": 0.0,
                "tree_distance_score_correlation": 0.2,
            }
        indexed = validate_prediction_rows(rows, examples)
        mechanisms = summarize_mechanisms(indexed)
        self.assertFalse(mechanisms["latent_collapsed"])
        positive = {
            "effects": {
                "ndcg@10": {"mean_delta": 0.01, "ci95_low": 0.001},
                "hit@10": {"mean_delta": 0.01},
            }
        }
        self.assertEqual(
            fp1_gate(
                positive, mechanisms, aggregate_item_valid=True, integrity_valid=True
            )["verdict"],
            "FP1_STRONG_PASS",
        )
        subgroup = {
            "history_length": {
                "short": {"users": 2, "delta_ndcg@10": 0.0}
            },
            "target_frequency": {
                "tail": {"users": 1, "delta_ndcg@10": -0.003}
            },
        }
        self.assertEqual(
            fp2_gate(
                positive,
                positive,
                subgroup,
                mechanisms,
                {"mean_tree_distance_score_correlation": 0.5},
                aggregate_item_valid=True,
                integrity_valid=True,
            )["verdict"],
            "FP2_NOT_STRONG_PASS",
        )

    def test_psid_collision_diagnostics_are_computed_from_frozen_shapes(self) -> None:
        import numpy as np

        resolved = {"a": (0, 0, 0), "b": (1, 1, 1)}
        raw = np.asarray([[0, 0, 0], [1, 0, 1]], dtype=np.int64)
        centroids = np.zeros((3, 2, 2), dtype=np.float64)
        centroids[1, 1] = (1.0, 0.0)
        result = psid_collision_diagnostics(resolved, raw, centroids)
        self.assertEqual(result["reassigned_items"], 1)
        self.assertEqual(result["collision_aliases_after"], 0)
        self.assertEqual(result["mean_reassigned_digit_hamming"], 1.0)
        self.assertEqual(result["mean_reconstruction_l2_distortion"], 1.0)

    def test_single_pass_materializer_denies_without_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / APPROVED_D0_SUFFIX
            path.parent.mkdir(parents=True)
            content = "u1 a x guard\n"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(PermissionError):
                materialize_external_evaluation_view(path, root=root)
            users, examples = materialize_external_evaluation_view(
                path,
                root=root,
                external_target_authorized=True,
                expected_sha256=hashlib.sha256(content.encode()).hexdigest(),
            )
            self.assertEqual(users[0].train_items, ("a",))
            self.assertEqual(examples[0].target, "x")
            with self.assertRaises(RuntimeError):
                materialize_external_evaluation_view(
                    path,
                    root=root,
                    external_target_authorized=True,
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
