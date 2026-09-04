from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from experiment.phase18.protocol import s18_s1_recovery as recovery
from experiment.phase18.protocol import s18_s1_runtime as runtime


ROOT = Path(__file__).resolve().parents[3]


class S18S1RecoveryTests(unittest.TestCase):
    def test_numpy_values_are_serializable_at_artifact_boundary(self) -> None:
        encoded = json.dumps(
            {
                "finite": np.bool_(True),
                "count": np.int64(7),
                "score": np.float64(1.25),
                "values": np.asarray([1, 2]),
            },
            default=runtime.json_default,
        )
        self.assertEqual(
            json.loads(encoded),
            {"finite": True, "count": 7, "score": 1.25, "values": [1, 2]},
        )

    def test_recovery_is_checkpoint_only(self) -> None:
        source = inspect.getsource(recovery.run_unit)
        self.assertNotIn("train_parent(", source)
        self.assertNotIn("train_item_head(", source)
        self.assertIn("load_frozen_models(", source)

    def test_recovery_output_is_disjoint_from_failed_attempt(self) -> None:
        failed = ROOT / "artifacts/phase18/s1_actionability/run-0001"
        self.assertNotEqual(recovery.OUTPUT, failed)
        self.assertNotIn(failed.resolve(), recovery.OUTPUT.resolve().parents)
        self.assertNotIn(recovery.OUTPUT.resolve(), failed.resolve().parents)

    def test_authorization_freezes_science_and_disables_retry(self) -> None:
        _, authorization = recovery.verify_authorization(verify_checkpoint_hashes=False)
        scope = authorization["correction_scope"]
        self.assertTrue(scope["checkpoint_only_diagnostic_recovery"])
        self.assertFalse(scope["parent_retraining"])
        self.assertFalse(scope["item_head_retraining"])
        self.assertFalse(scope["scientific_config_changes"])
        self.assertFalse(scope["effect_results_read_before_correction"])
        self.assertFalse(scope["automatic_retry"])
        self.assertFalse(scope["automatic_s18_2"])
        self.assertEqual(authorization["runtime"]["excluded_physical_gpus"], [1])

    def test_mutable_canonical_status_resolves_to_exact_sha_archive(self) -> None:
        authorization = json.loads(recovery.AUTH_PATH.read_text(encoding="utf-8"))
        record = authorization["frozen_inputs"]["failed_status"]
        resolved = recovery.resolve_frozen_input(record)
        self.assertEqual(
            resolved,
            ROOT
            / "artifacts/phase18/status/history/s18_s1_actionability.run-0001.status.json",
        )
        self.assertNotEqual(resolved, ROOT / record["path"])

    def test_checkpoint_sources_are_epoch10_from_failed_attempt(self) -> None:
        authorization = json.loads(recovery.AUTH_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(authorization["checkpoints"]), {"Beauty:I0", "Beauty:I-1", "Toys:I0", "Toys:I-1"})
        for pair in authorization["checkpoints"].values():
            for record in pair.values():
                self.assertIn("/run-0001/units/", f"/{record['path']}")
                self.assertTrue(record["path"].endswith("epoch10.pt"))
                self.assertEqual(len(record["sha256"]), 64)

    def test_smoke_uses_only_one_frozen_cohort_user(self) -> None:
        source = inspect.getsource(recovery.smoke)
        self.assertIn("max_users=1", source)
        self.assertFalse(recovery.SMOKE == recovery.OUTPUT)


if __name__ == "__main__":
    unittest.main()
