from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.core.full_latte_arm_contracts import build_preregistered_matrix
from experiment.phase17.protocol.s17_fp12_readiness_runtime import inspect_readiness


def allocation() -> dict:
    return {
        "formal_launch_authorized": False,
        "arm_specific_resource_profiles": {
            "formal_launch_authorized": False,
            "physical_gpu_by_arm": {
                "G0_GRAM_B0_FRESH": 1,
                "G1_GRAM_PSID_FULL": 0,
                "G2_GRAM_LATTE_FULL": 7,
                "N0_NATIVE_PSID": 4,
                "N1_NATIVE_LATTE": 4,
            },
        },
        "fp1_fp2_formal": {"state": "WAITING"},
    }


class FP12ReadinessTests(unittest.TestCase):
    def _root(self, temporary: str, status: dict) -> Path:
        root = Path(temporary)
        matrix_path = root / "experiment/phase17/config/s17_fp12_latte_arm_matrix.json"
        matrix_path.parent.mkdir(parents=True)
        matrix_path.write_text(json.dumps(build_preregistered_matrix()), encoding="utf-8")
        allocation_path = root / "experiment/phase17/config/s17_fp_resource_allocation.json"
        allocation_path.write_text(json.dumps(allocation()), encoding="utf-8")
        status_path = root / "artifacts/phase17/status/s17_fp0_full_data_tokenizer.status.json"
        status_path.parent.mkdir(parents=True)
        status_path.write_text(json.dumps(status), encoding="utf-8")
        return root

    def test_current_preflight_state_blocks_all_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(
                temporary,
                {
                    "attempt_id": "attempt_001",
                    "scientific_state": "PREFLIGHT",
                    "execution_state": "PREFLIGHT",
                    "status_code": "READY_AUTHORIZATION_REQUIRED",
                },
            )
            result = inspect_readiness(root)
            self.assertEqual(result["state"], "BLOCKED_FULL_DATA_TOKENIZER_NOT_COMPLETE")
            self.assertEqual(
                result["next_action"], "authorize_and_complete_full_data_tokenizer_attempt_001"
            )
            self.assertFalse(result["writes_performed"])

    def test_completed_tokenizer_with_matching_manifest_opens_profile_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "tokenizer_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}\n", encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            root = self._root(
                temporary,
                {
                    "attempt_id": "attempt_001",
                    "scientific_state": "COMPLETED",
                    "execution_state": "SCIENTIFIC_COMPLETED",
                    "status_code": "PASS_S17_FP0_FULL_DATA_TOKENIZER",
                    "tokenizer_manifest_path": "tokenizer_manifest.json",
                    "tokenizer_manifest_sha256": digest,
                },
            )
            result = inspect_readiness(root)
            self.assertEqual(result["state"], "READY_TO_IMPLEMENT_AND_PREPARE_PROFILES")
            self.assertEqual(len(result["resource_profile_contracts"]), 5)

    def test_five_prepared_profiles_open_authorization_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "tokenizer_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}\n", encoding="utf-8")
            import hashlib

            root = self._root(
                temporary,
                {
                    "attempt_id": "attempt_001",
                    "scientific_state": "COMPLETED",
                    "execution_state": "SCIENTIFIC_COMPLETED",
                    "status_code": "PASS_S17_FP0_FULL_DATA_TOKENIZER",
                    "tokenizer_manifest_path": "tokenizer_manifest.json",
                    "tokenizer_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                },
            )
            for arm in build_preregistered_matrix()["arms"]:
                status_path = (
                    root
                    / f"artifacts/phase17/status/s17_fp12_profile_{arm.lower()}.status.json"
                )
                status_path.write_text(
                    json.dumps(
                        {
                            "scientific_state": "PREFLIGHT",
                            "execution_state": "PREFLIGHT",
                            "status_code": "S17_FP12_RESOURCE_PROFILE_READY_AUTHORIZATION_REQUIRED",
                            "launch_authorized": False,
                        }
                    ),
                    encoding="utf-8",
                )
            result = inspect_readiness(root)
            self.assertEqual(
                result["state"], "READY_FOR_ARM_SPECIFIC_PROFILE_AUTHORIZATION"
            )
            self.assertTrue(result["all_profiles_prepared"])
            self.assertEqual(len(result["prepared_profile_arms"]), 5)


if __name__ == "__main__":
    unittest.main()
