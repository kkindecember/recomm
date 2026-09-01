from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.protocol.s17_fp0_full_data_tokenizer_runtime import (
    ATTEMPT_ID,
    EXPERIMENT_ID,
    MINIMUM_FREE_MIB,
    TARGET_GPU_ID,
    frozen_config,
    paths,
    verify_launch_authorization,
)


class FullDataTokenizerRuntimeTests(unittest.TestCase):
    def _allocation(self, root: Path, *, authorized: bool) -> None:
        path = root / "experiment/phase17/config/s17_fp_resource_allocation.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "full_data_tokenizer": {
                        "physical_gpu": TARGET_GPU_ID,
                        "minimum_free_mib": MINIMUM_FREE_MIB,
                        "launch_authorized": authorized,
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_frozen_config_has_no_effect_or_external_target_access(self) -> None:
        config = frozen_config(Path("/tmp"))
        self.assertFalse(config["effect_experiment_started"])
        self.assertFalse(config["external_target_materialized"])
        self.assertEqual(config["fit_contract"]["pca_fit"], "train_prefix_mask_only")
        self.assertTrue(config["background_required"])

    def test_authorization_fails_closed_on_current_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._allocation(root, authorized=False)
            with self.assertRaises(PermissionError):
                verify_launch_authorization(root)

    def test_attempt_specific_authorization_must_match_gpu_and_no_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._allocation(root, authorized=True)
            authorization = paths(root)["authorization"]
            authorization.parent.mkdir(parents=True)
            authorization.write_text(
                json.dumps(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "attempt_id": ATTEMPT_ID,
                        "authorized": True,
                        "physical_gpu": TARGET_GPU_ID,
                        "automatic_process_termination": False,
                        "automatic_retry": False,
                        "researcher_direction": "unit-test authorization",
                    }
                ),
                encoding="utf-8",
            )
            result = verify_launch_authorization(root)
            self.assertEqual(result["authorization"]["physical_gpu"], TARGET_GPU_ID)
            self.assertFalse(result["authorization"]["automatic_retry"])


if __name__ == "__main__":
    unittest.main()
