from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiment/phase17/config/s17_fp_resource_allocation.json"


class S17FPResourceAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_no_formal_launch_is_authorized_by_planning(self) -> None:
        self.assertFalse(self.config["formal_launch_authorized"])
        self.assertFalse(self.config["fp1_fp2_formal"]["launch_authorized"])
        self.assertFalse(self.config["automatic_process_termination"])

    def test_researcher_directed_formal_attempt_is_separate_from_planning(self) -> None:
        attempt = self.config["fp1_fp2_formal_attempt_001"]
        self.assertEqual(attempt["state"], "AUTHORIZED_CHECKPOINT_SELECTION_EXTERNAL_TARGET_STILL_SEALED")
        self.assertTrue(all(attempt["launch_authorized_by_arm"].values()))
        self.assertFalse(attempt["external_d0_target_materialized"])
        self.assertFalse(attempt["external_d0_evaluation_authorized"])
        self.assertFalse(attempt["automatic_retry"])

    def test_four_card_two_wave_mapping_preserves_gpu1(self) -> None:
        formal = self.config["fp1_fp2_formal"]
        self.assertEqual(
            formal["preferred_topology"],
            "four_cards_two_waves_gpu4_shared_for_native_arms",
        )
        self.assertEqual(formal["wave_1"]["gpu_1"], "G0_GRAM_B0_FRESH")
        self.assertEqual(formal["wave_1"]["gpu_4"], "N0_NATIVE_PSID")
        self.assertEqual(formal["wave_2"], {"gpu_4": "N1_NATIVE_LATTE"})
        repeat = self.config["gpu1_handoff_and_repeat"]
        self.assertTrue(repeat["repeat_starts_immediately_after_gpu1_science_terminal_state"])
        self.assertTrue(repeat["repeat_runs_until_next_researcher_handoff"])
        self.assertFalse(repeat["repeat_result_selection_eligible"])
        self.assertTrue(repeat["repeat_metrics_ignored"])
        self.assertFalse(repeat["repeat_affects_scientific_result"])

    def test_gpu1_and_stage16_guards_are_explicit(self) -> None:
        constraints = self.config["fixed_constraints"]
        self.assertTrue(constraints["gpu1_usable_for_formal_science"])
        self.assertTrue(constraints["gpu1_handoff_required"])
        self.assertTrue(constraints["gpu1_post_science_repeat_required"])
        self.assertTrue(constraints["gpu1_post_science_release_forbidden"])
        self.assertTrue(constraints["stage16_process_must_not_be_stopped_or_modified"])
        self.assertTrue(constraints["gpu4_shared_remaining_memory_authorized"])
        self.assertTrue(constraints["gpu4_preexisting_processes_must_be_preserved"])
        self.assertNotIn(4, self.config["fp1_fp2_formal"]["temporarily_excluded_gpu_ids"])

    def test_gpu4_share_has_a_memory_cap_and_fallback(self) -> None:
        formal = self.config["fp1_fp2_formal"]
        self.assertEqual(formal["gpu4_initial_profile_peak_reserved_cap_mib"], 16384)
        self.assertEqual(formal["gpu4_initial_profile_minimum_free_mib"], 19456)
        self.assertTrue(formal["gpu4_recheck_must_preserve_preexisting_processes"])
        self.assertIn("three_card_fallback_if_gpu4_native_arm_does_not_fit", formal)
        profiles = self.config["arm_specific_resource_profiles"]
        self.assertEqual(profiles["physical_gpu_by_arm"]["N0_NATIVE_PSID"], 4)
        self.assertEqual(profiles["physical_gpu_by_arm"]["N1_NATIVE_LATTE"], 4)
        self.assertTrue(profiles["gpu4_native_profiles_run_sequentially"])

    def test_resource_thresholds_follow_measured_profiles(self) -> None:
        evidence = self.config["measured_resource_evidence"]
        self.assertEqual(evidence["tokenizer_peak_reserved_mib"], 984)
        self.assertGreaterEqual(
            self.config["full_data_tokenizer"]["minimum_free_mib"],
            evidence["tokenizer_peak_reserved_mib"] + 4096,
        )
        self.assertTrue(evidence["fixed_30g_is_not_required_after_arm_specific_profile"])
        self.assertEqual(evidence["formal_target_peak_reserved_mib"], 20480)
        self.assertEqual(evidence["formal_postprofile_safety_margin_mib"], 3072)
        formal = self.config["fp1_fp2_formal"]
        self.assertEqual(formal["postprofile_target_minimum_free_mib"], 23552)
        self.assertEqual(formal["postprofile_target_requires_peak_reserved_mib_lte"], 20480)
        self.assertFalse(self.config["current_admission"]["fp1_fp2_formal_any_card"])

    def test_memory_reduction_preserves_scientific_protocol(self) -> None:
        profiles = self.config["arm_specific_resource_profiles"]
        self.assertTrue(profiles["profile_is_not_effect_experiment"])
        self.assertTrue(profiles["post_profile_gpu1_repeat_required"])
        self.assertFalse(profiles["formal_launch_authorized"])
        self.assertIn("change effective global batch", profiles["forbidden_adjustments"])
        self.assertIn("reduce beam or top-k", profiles["forbidden_adjustments"])


if __name__ == "__main__":
    unittest.main()
