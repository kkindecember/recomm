from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "experiment" / "phase16" / "protocol"
sys.path.insert(0, str(PROTOCOL))

from stage16_s4_toys_frozen_preflight import (  # noqa: E402
    EXPECTED_CONTROLS,
    REQUIRED_ARMS,
    validate_config_contract,
)


CONFIG = ROOT / "experiment/phase16/configs/stage16_s4_toys_frozen_preflight.json"


class Stage16S4ToysFrozenPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_contract_is_valid(self) -> None:
        validate_config_contract(self.config)
        self.assertEqual(tuple(self.config["arms"]), REQUIRED_ARMS)
        self.assertEqual(
            {name: self.config["arms"][name]["control"] for name in REQUIRED_ARMS},
            EXPECTED_CONTROLS,
        )

    def test_stage15_pilots_and_blocked_gfull_are_excluded(self) -> None:
        excluded = self.config["excluded_arms"]
        self.assertIn("G-FULL", excluded)
        self.assertIn("Stage15-B2", excluded)
        self.assertIn("Stage15-B3", excluded)

    def test_non_strict_acceptance_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.config)
        drifted["arms"]["S-AUX"]["acceptance"] = "score_greater_than_or_equal_threshold"
        with self.assertRaisesRegex(ValueError, "strict > acceptance"):
            validate_config_contract(drifted)

    def test_historical_top_candidate_redraft_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.config)
        drifted["arms"]["S-PLUS"]["guided_redraft"] = "historical_top_candidate_prefixes"
        with self.assertRaisesRegex(ValueError, "live verifier beam prefixes"):
            validate_config_contract(drifted)

    def test_wrong_causal_control_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.config)
        drifted["arms"]["S-PLUS"]["control"] = "F0"
        with self.assertRaisesRegex(ValueError, "matched-control drift"):
            validate_config_contract(drifted)

    def test_cpu_freeze_cannot_unlock_gpu(self) -> None:
        drifted = copy.deepcopy(self.config)
        drifted["launch_contract"]["gpu_launch_ready_after_this_preflight"] = True
        with self.assertRaisesRegex(ValueError, "must not authorize GPU launch"):
            validate_config_contract(drifted)

    def test_test_and_automatic_retry_remain_sealed(self) -> None:
        self.assertFalse(self.config["evaluation_contract"]["test_read"])
        self.assertFalse(self.config["launch_contract"]["test_read"])
        self.assertFalse(self.config["launch_contract"]["automatic_retry"])


if __name__ == "__main__":
    unittest.main()
