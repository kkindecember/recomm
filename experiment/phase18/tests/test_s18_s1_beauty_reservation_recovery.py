from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest import mock

from experiment.phase18.core.contracts import load_json, sha256
from experiment.phase18.protocol import s18_s1_beauty_recovery as engine
from experiment.phase18.protocol import s18_s1_beauty_reservation_recovery as run0007


class S18S1BeautyReservationRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authorization = load_json(
            run0007.ROOT
            / "experiment/phase18/config/s18_s1_beauty_reservation_authorization.json"
        )

    def test_attempt_is_named_and_does_not_overwrite_run0006(self) -> None:
        source = inspect.getsource(run0007.configure)
        self.assertIn('attempt_id="run-0007"', source)
        self.assertIn('run-0006"', source)
        self.assertIn('run-0007"', source)
        self.assertNotEqual(
            run0007.ROOT / "artifacts/phase18/s1_actionability/run-0007",
            engine.OUTPUT,
        )

    def test_scope_retains_allocator_cache_without_changing_science(self) -> None:
        scope = self.authorization["correction_scope"]
        self.assertTrue(scope["resource_only"])
        self.assertTrue(scope["release_unused_generation_tensors"])
        self.assertFalse(scope["release_cuda_cache_per_user"])
        self.assertTrue(scope["retain_allocator_cache_between_users"])
        self.assertTrue(scope["preclaim_allocator_reservation"])
        for field in (
            "parent_retraining",
            "item_head_retraining",
            "scientific_config_changes",
            "cohort_changes",
            "beam_changes",
            "score_changes_allowed",
            "protected_data_access",
            "automatic_retry",
            "automatic_s18_2",
        ):
            self.assertFalse(scope[field])

    def test_frozen_inputs_match(self) -> None:
        for record in self.authorization["frozen_inputs"].values():
            path = run0007.ROOT / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256(path), record["sha256"], path)

    def test_allocator_claim_is_cached_not_emptied(self) -> None:
        authorization = {
            "runtime": {"allocator_reservation_mib_by_gpu": {"7": 100, "6": 80}}
        }
        device_context = mock.MagicMock()
        device_context.__enter__.return_value = None
        device_context.__exit__.return_value = False
        with (
            mock.patch.object(engine.torch.cuda, "device", return_value=device_context),
            mock.patch.object(
                engine.torch.cuda,
                "mem_get_info",
                side_effect=[(200 * 1024**2, 500 * 1024**2)] * 2,
            ),
            mock.patch.object(engine.torch, "empty", side_effect=[object(), object()]) as empty,
            mock.patch.object(engine.torch.cuda, "synchronize"),
            mock.patch.object(
                engine.torch.cuda,
                "memory_reserved",
                side_effect=[100 * 1024**2, 80 * 1024**2],
            ),
            mock.patch.object(engine.torch.cuda, "empty_cache") as empty_cache,
        ):
            result = engine.prime_allocator_reservation([7, 6], authorization)
        self.assertEqual(empty.call_count, 2)
        empty_cache.assert_not_called()
        self.assertEqual(result["0"]["reserved_mib"], 100)
        self.assertEqual(result["1"]["reserved_mib"], 80)

    def test_formal_admission_includes_buffer(self) -> None:
        smoke = {
            "peak_by_visible_gpu": {
                "0": {"physical_gpu": 7, "reserved_mib": 17408.0},
                "1": {"physical_gpu": 5, "reserved_mib": 16384.0},
            }
        }
        self.assertEqual(
            engine.formal_required_free(smoke, self.authorization),
            {7: 21504, 5: 20480},
        )

    def test_every_unit_is_readmitted_immediately_before_spawn(self) -> None:
        source = inspect.getsource(engine.execute_unit)
        self.assertIn("validate_pair_free", source)
        self.assertLess(source.index("validate_pair_free"), source.index("subprocess.Popen"))
        self.assertIn("ENTRY_PATH", source)

    def test_diagnostic_uses_authorized_cache_policy(self) -> None:
        source = inspect.getsource(engine.run_diagnostic)
        self.assertIn("prime_allocator_reservation", source)
        self.assertIn("release_cuda_cache_per_user=release_cuda_cache_per_user", source)
        self.assertNotIn("release_cuda_cache_per_user=True", source)


if __name__ == "__main__":
    unittest.main()
