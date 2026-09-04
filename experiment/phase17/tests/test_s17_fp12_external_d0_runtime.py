from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.protocol.s17_fp12_external_d0_runtime import (
    _parse_gpu_map,
    arm_paths,
    paths,
    source_paths,
    verify_readiness,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class ExternalD0RuntimeTests(unittest.TestCase):
    def test_frozen_config_is_unauthorized_and_complete(self) -> None:
        config = json.loads(
            (ROOT / "experiment/phase17/config/s17_fp12_external_d0.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(config["checkpoints"]), set(ARM_IDS))
        self.assertFalse(config["authorization"]["external_d0_evaluation_authorized"])
        self.assertFalse(config["authorization"]["gpu_execution_authorized"])
        self.assertEqual(config["statistics"]["paired_bootstrap_replicates"], 2000)
        self.assertEqual(config["inference"]["beam_sizes"], [50, 500])

    def test_readiness_does_not_call_target_materializer(self) -> None:
        with patch(
            "experiment.phase17.protocol.s17_fp12_external_d0_runtime."
            "materialize_external_evaluation_view",
            side_effect=AssertionError("preflight must not read D0 targets"),
        ):
            readiness = verify_readiness(ROOT)
        self.assertEqual(
            readiness["verdict"], "READY_AWAITING_EXPLICIT_D0_GPU_AUTHORIZATION"
        )
        self.assertFalse(readiness["external_projection_content_read"])
        self.assertEqual(set(readiness["checkpoint_evidence"]), set(ARM_IDS))

    def test_authorization_requires_exact_five_arm_gpu_map(self) -> None:
        values = [f"{arm}=0" for arm in ARM_IDS]
        self.assertEqual(set(_parse_gpu_map(values)), set(ARM_IDS))
        with self.assertRaises(ValueError):
            _parse_gpu_map(values[:-1])
        with self.assertRaises(ValueError):
            _parse_gpu_map(values + [f"{ARM_IDS[0]}=1"])

    def test_worker_commands_are_background_gpu_isolated(self) -> None:
        for index, arm_id in enumerate(ARM_IDS):
            command = worker_command(ROOT, arm_id, index)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={index}", command)
            self.assertIn(str(arm_paths(ROOT, arm_id)["snapshot_worker"]), command)
            self.assertIn("worker", command)

    def test_snapshot_sources_exist_and_no_result_has_been_materialized(self) -> None:
        for source in source_paths(ROOT):
            self.assertTrue(source.is_file(), source)
        resolved = paths(ROOT)
        if not resolved["result"].exists():
            self.assertFalse(resolved["bundle"].exists())
            self.assertFalse(resolved["seal"].exists())


if __name__ == "__main__":
    unittest.main()
