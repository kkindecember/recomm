from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from experiment.phase18.protocol import s18_s1_resource_recovery as recovery


class S18S1ResourceRecoveryTests(unittest.TestCase):
    def test_authorization_is_resource_only_and_gpu1_is_explicit(self) -> None:
        _, _, authorization = recovery.verify_authorization()
        scope = authorization["correction_scope"]
        self.assertTrue(scope["resource_only"])
        self.assertTrue(scope["generation_use_cache"])
        self.assertTrue(scope["cross_attention_cache"])
        self.assertTrue(scope["decoder_model_parallel"])
        self.assertTrue(scope["release_cuda_cache_per_user"])
        self.assertFalse(scope["scientific_config_changes"])
        self.assertFalse(scope["beam_changes"])
        self.assertFalse(scope["score_changes_allowed"])
        self.assertIn(1, authorization["runtime"]["candidate_physical_gpus"])

    def test_unit_is_checkpoint_only_and_cache_on(self) -> None:
        source = inspect.getsource(recovery.run_unit)
        self.assertNotIn("train_parent(", source)
        self.assertNotIn("train_item_head(", source)
        self.assertIn("load_frozen_models(", source)
        self.assertIn("generation_use_cache=True", source)
        self.assertIn("cross_attention_cache=True", source)
        self.assertIn("enable_two_gpu_decoder_parallel", source)
        self.assertIn("release_cuda_cache_per_user=True", source)

    def test_no_existing_attempt_is_overwritten(self) -> None:
        self.assertEqual(recovery.ATTEMPT_ID, "run-0003")
        self.assertNotEqual(recovery.OUTPUT, recovery.RUN2)
        source = inspect.getsource(recovery.run_unit)
        self.assertIn("retry forbidden", source)

    def test_master_waits_for_smoke_and_run2(self) -> None:
        source = inspect.getsource(recovery.wait_for_prerequisites)
        self.assertIn("validate_memory_smoke", source)
        self.assertIn("run2_is_terminal", source)

    def test_completed_run2_units_are_carried_not_recomputed(self) -> None:
        source = inspect.getsource(recovery.master)
        self.assertIn("completed_run2_units()", source)
        self.assertIn("carry_forward(label, source)", source)

    def test_units_are_serial_and_have_stable_admission(self) -> None:
        _, _, authorization = recovery.verify_authorization()
        self.assertTrue(authorization["runtime"]["serial_units"])
        source = inspect.getsource(recovery.wait_for_gpu_pair)
        self.assertIn("stable_snapshots_required", source)


if __name__ == "__main__":
    unittest.main()
