from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiment.phase16.protocol.finalize_stage16_s4_toys import (
    exact_paired_binary_greater,
    holm_adjust,
    strictly_dominates,
)
from experiment.phase16.protocol.stage16_s4_toys_validation import (
    FORMAL_ARMS,
    finite_unseen_constrained_draft,
    official_finalize,
    reset_peak_memory_stats_compat,
    saux_embedding_views,
    summarize_rows,
    validate_config,
    verifier_score_lengths,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiment/phase16/configs/stage16_s4_toys_standalone_gpu4_a7.json"
FORMAL_RUNNER = ROOT / "experiment/phase16/run_stage16_s4_toys_standalone_gpu4_a7_inner.sh"
REPEAT_RUNNER = ROOT / "experiment/phase16/run_stage16_s4_toys_repeat_gpu4_a7_inner.sh"


class Stage16S4ToysValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_is_gpu4_fixed_and_formal_arms_are_complete(self) -> None:
        for arm in FORMAL_ARMS:
            validate_config(self.config, arm)
        self.assertEqual(self.config["physical_gpu"], 4)
        self.assertEqual(tuple(self.config["formal_arms"]), FORMAL_ARMS)

    def test_physical_gpu_metadata_is_portable_but_visible_gpu_remains_zero(self) -> None:
        gpu0 = copy.deepcopy(self.config)
        gpu0["physical_gpu"] = 0
        validate_config(gpu0, "S-AUX")

        for invalid_physical_gpu in (-1, True, "4"):
            drifted = copy.deepcopy(gpu0)
            drifted["physical_gpu"] = invalid_physical_gpu
            with self.assertRaisesRegex(ValueError, "seed/GPU identity"):
                validate_config(drifted, "S-AUX")

        drifted = copy.deepcopy(gpu0)
        drifted["visible_gpu"] = 4
        with self.assertRaisesRegex(ValueError, "seed/GPU identity"):
            validate_config(drifted, "S-AUX")

    def test_gpu4_a7_config_and_memory_only_runners_are_consistent(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        for arm in FORMAL_ARMS:
            validate_config(config, arm)
        self.assertEqual(config["physical_gpu"], 4)
        self.assertEqual(config["visible_gpu"], 0)
        self.assertEqual(config["resources"]["minimum_free_mib"], 19000)

        formal = FORMAL_RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU=4", formal)
        self.assertIn("MINIMUM_FREE=19000", formal)
        self.assertNotIn("MAXIMUM_UTIL", formal)

        repeat = REPEAT_RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU=4", repeat)
        self.assertIn("MINIMUM_FREE=19000", repeat)
        self.assertIn("--discard-output", repeat)
        self.assertNotIn("--output-dir", repeat)

    def test_saux_checkpoint_and_inductive_history_views_remain_distinct(self) -> None:
        embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        views = saux_embedding_views(
            item_ids=["c", "a", "e", "b", "d"],
            embeddings=embeddings,
            retained_items={"a", "b"},
            ordered_items=["a", "b", "c", "d", "e"],
        )
        self.assertEqual(tuple(views["train_embeddings"].shape), (3, 4))
        self.assertEqual(tuple(views["history_embeddings"].shape), (6, 4))
        self.assertEqual(tuple(views["candidate_embeddings"].shape), (5, 4))
        self.assertEqual(views["history_index"], {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        self.assertTrue(torch.equal(views["train_embeddings"][1], embeddings[1]))
        self.assertTrue(torch.equal(views["history_embeddings"][3], embeddings[0]))

    def test_strict_acceptance_and_live_beam_are_frozen(self) -> None:
        faithful = self.config["faithful_inference"]
        self.assertEqual(faithful["acceptance"], "strict_gt")
        self.assertEqual(faithful["threshold"], -1.8)
        self.assertEqual(faithful["guided_redraft"], "current_live_verifier_beam_prefixes")
        self.assertEqual(
            faithful["underfilled_live_round"],
            "draft_all_finite_unseen_then_advance_verifier_beam",
        )
        self.assertFalse(faithful["stage15_b2_reused_as_faithful"])

    def test_underfilled_live_round_drafts_only_finite_unseen_candidates(self) -> None:
        logits = torch.tensor([[5.0, float("-inf"), 3.0, float("-inf")]])
        selected = finite_unseen_constrained_draft(logits, 3)
        self.assertEqual(selected.shape, (1, 2))
        self.assertEqual(set(selected[0].tolist()), {0, 2})
        self.assertTrue(torch.isneginf(logits).all())

        exhausted = finite_unseen_constrained_draft(logits, 3)
        self.assertEqual(exhausted.shape, (1, 0))

    def test_non_strict_drift_fails_closed(self) -> None:
        drifted = copy.deepcopy(self.config)
        drifted["faithful_inference"]["acceptance"] = "gte"
        with self.assertRaisesRegex(ValueError, "strict SpecGR acceptance"):
            validate_config(drifted, "S-AUX")

    def test_official_aux_and_plus_prefix_length_policies_remain_distinct(self) -> None:
        paths = {"warm": (1, 2, 3, 4), "cold": (1, 9, 8, 7)}
        aux = verifier_score_lengths(paths, {"warm"}, {"cold"}, arm="S-AUX")
        plus = verifier_score_lengths(paths, {"warm"}, {"cold"}, arm="S-PLUS")
        self.assertEqual(aux, {"warm": 4, "cold": 2})
        self.assertEqual(plus, {"warm": 4, "cold": 3})

    def test_peak_reset_uses_torch_1_11_integer_cuda_index(self) -> None:
        with patch.object(torch.cuda, "init") as initialize, patch.object(
            torch.cuda, "reset_peak_memory_stats"
        ) as reset:
            self.assertEqual(reset_peak_memory_stats_compat(torch.device("cuda")), 0)
        initialize.assert_called_once_with()
        reset.assert_called_once_with(0)

    def test_official_finalize_truncates_accepted_in_draft_order_then_sorts(self) -> None:
        verified = [
            ("first", -3.0, True),
            ("second", -2.0, True),
            ("late-better", -1.0, True),
        ]
        result = official_finalize(verified, [], 2)
        self.assertEqual(result, ["second", "first"])
        self.assertNotIn("late-better", result)

    def test_official_fallback_excludes_already_drafted_beam_item(self) -> None:
        verified = [("accepted", -1.0, True), ("rejected", -3.0, False)]
        beam = [("rejected", 10.0), ("beam", -2.0)]
        self.assertEqual(official_finalize(verified, beam, 2), ["accepted", "beam"])

    def test_pareto_dominance_requires_all_weak_and_one_strict(self) -> None:
        left = {
            "cold_hit@50": 0.2,
            "warm_ndcg@10": 0.1,
            "update_seconds": 1.0,
            "inference_seconds": 2.0,
            "extra_state_bytes": 3.0,
        }
        right = {
            "cold_hit@50": 0.1,
            "warm_ndcg@10": 0.1,
            "update_seconds": 2.0,
            "inference_seconds": 2.0,
            "extra_state_bytes": 3.0,
        }
        self.assertTrue(strictly_dominates(left, right))
        worse_warm = dict(left)
        worse_warm["warm_ndcg@10"] = 0.0
        self.assertFalse(strictly_dominates(worse_warm, right))

    def test_exact_paired_binary_test_is_directional(self) -> None:
        events = [
            {
                "is_cold": True,
                "metrics": {
                    "treatment": {"hit@50": treatment},
                    "control": {"hit@50": control},
                },
            }
            for treatment, control in ((1, 0), (1, 0), (1, 0), (0, 0))
        ]
        result = exact_paired_binary_greater(events, "treatment", "control")
        self.assertEqual(result["treatment_only_hits"], 3)
        self.assertEqual(result["control_only_hits"], 0)
        self.assertEqual(result["discordant_pairs"], 3)
        self.assertEqual(result["raw_p_value"], 0.125)

    def test_holm_adjustment_is_monotone_and_familywise(self) -> None:
        result = holm_adjust(
            {"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5}, alpha=0.05
        )
        self.assertAlmostEqual(result["a"]["holm_adjusted_p_value"], 0.004)
        self.assertAlmostEqual(result["b"]["holm_adjusted_p_value"], 0.03)
        self.assertAlmostEqual(result["c"]["holm_adjusted_p_value"], 0.08)
        self.assertAlmostEqual(result["d"]["holm_adjusted_p_value"], 0.5)
        self.assertTrue(result["a"]["reject_at_alpha"])
        self.assertTrue(result["b"]["reject_at_alpha"])
        self.assertFalse(result["c"]["reject_at_alpha"])
        self.assertFalse(result["d"]["reject_at_alpha"])

    def test_finalizer_does_not_hardcode_an_obsolete_runtime_attempt(self) -> None:
        finalizer = (
            ROOT / "experiment/phase16/protocol/finalize_stage16_s4_toys.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("gpu0_a3_isolated_runtime_v1", finalizer)
        self.assertIn("Holm", finalizer)

    def test_repeat_is_discard_only_and_nonpromotional(self) -> None:
        repeat = self.config["post_terminal_repeat"]
        self.assertTrue(repeat["enabled_by_user"])
        self.assertTrue(repeat["discard_output"])
        self.assertTrue(repeat["formal_output_read_only"])
        self.assertFalse(repeat["affects_scientific_results"])
        self.assertFalse(repeat["promotion_eligible"])
        self.assertFalse(repeat["repeat_artifacts_saved"])

    def test_bounded_smoke_is_one_discard_only_event_per_arm(self) -> None:
        self.assertEqual(self.config["resources"]["bounded_smoke_events_per_arm"], 1)
        cold_only = [
            {"is_cold": True, "metrics": {"hit@50": 1, "ndcg@10": 0.5}}
        ]
        partial = summarize_rows(cold_only, require_all_subsets=False)
        self.assertEqual(set(partial), {"overall", "cold"})

    def test_formal_runner_is_gpu4_write_once_and_repeat_after_completed(self) -> None:
        runner = FORMAL_RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU=4", runner)
        self.assertIn("MINIMUM_FREE=19000", runner)
        self.assertNotIn("MAXIMUM_UTIL", runner)
        self.assertIn("--output-dir \"$OUTPUT/arms/$arm\"", runner)
        self.assertIn("runtime_snapshot_manifest.json", runner)
        self.assertLess(
            runner.index("terminal_status COMPLETED"),
            runner.index("tmux new-session -d -s \"$REPEAT_SESSION\""),
        )

    def test_repeat_runner_discards_outputs_and_only_signals_own_pid(self) -> None:
        runner = REPEAT_RUNNER.read_text(encoding="utf-8")
        self.assertIn("--discard-output", runner)
        self.assertNotIn("--output-dir", runner)
        self.assertIn(">/dev/null 2>&1", runner)
        self.assertNotIn("pkill", runner)
        self.assertNotIn("killall", runner)
        self.assertIn('kill -TERM "$WORKLOAD_PID"', runner)


if __name__ == "__main__":
    unittest.main()
