from __future__ import annotations

import inspect
import unittest

from experiment.phase18.protocol import s18_s1_beauty_recovery as recovery


class S18S1BeautyRecoveryTests(unittest.TestCase):
    def test_authorization_is_checkpoint_only_and_beauty_only(self) -> None:
        _, _, authorization = recovery.load_contracts()
        scope = authorization["correction_scope"]
        self.assertTrue(scope["resource_only"])
        self.assertTrue(scope["checkpoint_only_diagnostic_recovery"])
        self.assertTrue(scope["beauty_only_execution"])
        self.assertTrue(scope["carry_completed_toys"])
        self.assertFalse(scope["parent_retraining"])
        self.assertFalse(scope["item_head_retraining"])
        self.assertFalse(scope["scientific_config_changes"])
        self.assertFalse(scope["protected_data_access"])
        self.assertFalse(scope["automatic_retry"])
        self.assertFalse(scope["automatic_s18_2"])
        self.assertEqual(authorization["runtime"]["beauty_units"], ["Beauty:I0", "Beauty:I-1"])

    def test_attempt_paths_do_not_overwrite_previous_runs(self) -> None:
        self.assertEqual(recovery.ATTEMPT_ID, "run-0006")
        self.assertTrue(str(recovery.OUTPUT).endswith("/run-0006"))
        for prior in range(1, 6):
            self.assertNotEqual(recovery.OUTPUT.name, f"run-{prior:04d}")

    def test_diagnostic_has_no_training_or_science_change(self) -> None:
        source = inspect.getsource(recovery.run_diagnostic)
        self.assertNotIn("train_parent(", source)
        self.assertNotIn("train_item_head(", source)
        self.assertIn("load_frozen_models(", source)
        self.assertIn("generation_use_cache=True", source)
        self.assertIn("cross_attention_cache=True", source)
        self.assertIn("release_cuda_cache_per_user=release_cuda_cache_per_user", source)
        self.assertIn("enable_two_gpu_decoder_parallel", source)

    def test_smoke_is_beauty_specific_and_non_scientific(self) -> None:
        _, _, authorization = recovery.load_contracts()
        self.assertEqual(authorization["runtime"]["smoke_users"], 128)
        source = inspect.getsource(recovery.smoke_worker)
        self.assertIn('"I0"', source)
        self.assertIn('"scientific_result_eligible": False', source)
        self.assertIn("compare_first_beauty_user", source)

    def test_master_carries_toys_and_executes_only_beauty(self) -> None:
        source = inspect.getsource(recovery.master)
        self.assertIn('carry_forward(\n            "Toys:I0"', source)
        self.assertIn('carry_forward(\n            "Toys:I-1"', source)
        self.assertIn('authorization["runtime"]["beauty_units"]', source)
        self.assertNotIn("automatic_s18_2=True", source)

    def test_canonical_status_is_synchronized(self) -> None:
        source = inspect.getsource(recovery.update_status)
        self.assertIn("CANONICAL_STATUS", source)
        launch_source = inspect.getsource(recovery.launch)
        self.assertIn("STATUS_ARCHIVE", launch_source)


if __name__ == "__main__":
    unittest.main()
