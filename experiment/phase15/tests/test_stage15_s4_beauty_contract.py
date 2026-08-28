from __future__ import annotations

import copy
import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from stage15_s4_beauty_contract import (  # noqa: E402
    EXPECTED_B2,
    EXPECTED_B3,
    EXPECTED_COMMON,
    validate_contract_payload,
)


def frozen_payload() -> dict:
    return {
        "stage": "S15-4",
        "domain": "Beauty_cold50",
        "common": copy.deepcopy(EXPECTED_COMMON),
        "b2": copy.deepcopy(EXPECTED_B2),
        "b3": copy.deepcopy(EXPECTED_B3),
        "execution": {
            "b2_minimum_free_mib": 16384,
            "b3_minimum_free_mib": 15360,
            "hard_timeout_seconds": 86400,
        },
        "test_read": False,
        "automatic_retry": False,
    }


class TestStage15S4BeautyContract(unittest.TestCase):
    def test_accepts_frozen_eight_position_contract(self):
        validate_contract_payload(frozen_payload())

    def test_rejects_toys_depth_or_hyperparameter_drift(self):
        payload = frozen_payload()
        payload["common"]["lexical_positions"] = 6
        with self.assertRaisesRegex(ValueError, "common contract drift"):
            validate_contract_payload(payload)
        payload = frozen_payload()
        payload["b3"]["requests_per_position"] = 5
        with self.assertRaisesRegex(ValueError, "B3 contract drift"):
            validate_contract_payload(payload)

    def test_rejects_test_or_retry_flags(self):
        payload = frozen_payload()
        payload["test_read"] = True
        with self.assertRaisesRegex(ValueError, "safety flags drift"):
            validate_contract_payload(payload)


if __name__ == "__main__":
    unittest.main()
