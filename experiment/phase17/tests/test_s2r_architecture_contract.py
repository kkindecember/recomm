from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUDGET_PATH = (
    ROOT / "experiment/phase17/config/s17_s2r_architecture_reselection_budget.json"
)
CARD_DIR = ROOT / "experiment/phase17/registry/architecture_cards"
PLAN_PATH = (
    ROOT
    / "plan/第十七阶段/GRAM_第十七阶段_S17-2R架构级候选重选与大改实验计划v0.1.md"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class S2RArchitectureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = load_json(BUDGET_PATH)
        self.cards = {
            entry["candidate_id"]: load_json(ROOT / entry["card"])
            for entry in self.budget["candidate_pool"]
        }

    def test_big_changes_are_explicitly_authorized(self) -> None:
        authorization = self.budget["authorization"]
        for key in (
            "architecture_level_changes_allowed",
            "tokenizer_replacement_allowed",
            "identifier_replacement_allowed",
            "decoder_replacement_allowed",
            "backbone_replacement_allowed",
            "training_from_scratch_allowed",
        ):
            self.assertTrue(authorization[key], key)
        self.assertTrue(authorization["s17_5_superseded_until_s2r_passes"])

    def test_scientific_data_boundaries_remain_sealed(self) -> None:
        protected = self.budget["protected_contracts"]
        self.assertTrue(protected["shadow_fold_causality"])
        self.assertTrue(protected["no_future_target_context"])
        self.assertTrue(protected["unified_item_level_evaluator"])
        self.assertFalse(protected["official_test_read"])
        self.assertFalse(protected["sports_read"])
        self.assertFalse(protected["d1_read_during_s2r"])

    def test_old_1k_probe_cannot_be_reused_as_matched_baseline(self) -> None:
        forbidden = self.budget["forbidden_selection_evidence"]
        self.assertIn("historical_s17_s0_1k_metric_as_matched_baseline", forbidden)
        self.assertIn("runtime_occupancy_repeat_metrics", forbidden)
        self.assertIn("unmatched_single_model_metric", forbidden)

    def test_candidate_pool_is_bounded_and_every_candidate_has_a_native_control(self) -> None:
        p0 = [
            card
            for card in self.cards.values()
            if str(card["priority"]).startswith("P0")
        ]
        self.assertLessEqual(len(p0), self.budget["caps"]["maximum_p0_families"])
        self.assertEqual(self.budget["caps"]["maximum_full_d0_finalists"], 2)
        self.assertEqual(
            self.budget["caps"]["maximum_diagnostic_revisions_per_family"], 1
        )
        for candidate_id, card in self.cards.items():
            with self.subTest(candidate_id=candidate_id):
                self.assertTrue(card["matched_native_control"].strip())
                self.assertTrue(card["mechanism_metrics"])
                self.assertTrue(card["source"]["paper_url"].startswith("https://"))
                self.assertIn("copy_policy", card)
                self.assertIn("prior_local_result_does_not_reject", card)

    def test_unlicensed_sources_are_not_copyable(self) -> None:
        for candidate_id in (
            "R2C_diffgrm",
            "R2D_setrec_full",
            "R2E_diger_jointsid",
        ):
            card = self.cards[candidate_id]
            self.assertNotIn("MAY_BE_ADAPTED", card["copy_policy"])
        self.assertEqual(self.cards["R2B_latte_full"]["source"]["license_status"], "MIT")

    def test_full_latte_and_setrec_are_not_the_rejected_local_hooks(self) -> None:
        latte = self.cards["R2B_latte_full"]
        setrec = self.cards["R2D_setrec_full"]
        self.assertIn(
            "replace_native_lexical_ids_with_trained_semantic_ids",
            latte["architecture_changes"],
        )
        self.assertIn("query_guided_simultaneous_token_generation", setrec["architecture_changes"])
        self.assertIn("did not train Semantic IDs", latte["prior_local_result_does_not_reject"])
        self.assertIn("did not replace AR decoding", setrec["prior_local_result_does_not_reject"])

    def test_diger_cannot_enter_p0_before_gradient_fix_validation(self) -> None:
        diger = self.cards["R2E_diger_jointsid"]
        self.assertEqual(diger["source"]["required_branch"], "gradient-fix")
        self.assertTrue(diger["priority"].startswith("P1_STATIC_ONLY"))
        self.assertIn("gradient reachability unit test", diger["promotion_blockers"])

    def test_gpu1_repeat_is_protected_from_preflight_and_smoke(self) -> None:
        resources = self.budget["resources"]
        self.assertTrue(resources["gpu1_reserved_for_non_scientific_repeat"])
        self.assertTrue(resources["gpu1_repeat_must_not_be_stopped_for_preflight"])
        self.assertTrue(resources["gpu1_scientific_use_requires_explicit_handoff_authorization"])
        self.assertEqual(self.budget["r1_smoke"]["exclude_physical_gpu_ids"], [1])

    def test_r2_gate_requires_matched_predictions_and_three_cohorts(self) -> None:
        r2 = self.budget["r2_screen"]
        self.assertTrue(r2["native_control_required"])
        self.assertTrue(r2["paired_uncertainty_required"])
        self.assertTrue(r2["convergence_or_preregistered_early_stop_required"])
        self.assertFalse(r2["fixed_one_epoch_for_all_architectures"])
        self.assertEqual(len(r2["evaluation_cohorts"]), 3)
        gate = r2["strong_promotion"]
        self.assertGreaterEqual(gate["minimum_mean_absolute_delta_ndcg_at_10"], 0.0015)
        self.assertGreaterEqual(gate["minimum_positive_cohorts"], 2)

    def test_plan_marks_s17_5_hold_and_no_gpu_science_started(self) -> None:
        text = PLAN_PATH.read_text(encoding="utf-8")
        self.assertIn("S17-5 暂时 `HOLD`", text)
        self.assertIn("GPU_NOT_STARTED", text)
        self.assertIn("GPU1 继续只运行", text)

    def test_status_schema_accepts_revision_step_id(self) -> None:
        schema = load_json(ROOT / "experiment/phase17/schemas/status.schema.json")
        pattern = schema["properties"]["step_id"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, "S17-2R"))
        self.assertIsNotNone(re.fullmatch(pattern, "S17-4"))
        self.assertIsNone(re.fullmatch(pattern, "S17-R2"))


if __name__ == "__main__":
    unittest.main()
