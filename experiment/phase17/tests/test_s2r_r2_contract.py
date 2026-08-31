from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment.phase17.protocol.s2r_r2_contract import prepare_r2_contract


ROOT = Path(__file__).resolve().parents[3]


class S2RR2ContractTests(unittest.TestCase):
    def test_frozen_r2_contract_is_fold_safe_and_complete(self) -> None:
        payload = prepare_r2_contract()
        self.assertEqual(payload["selected_users"], 3000)
        self.assertEqual(payload["internal_early_stop_examples"], 300)
        self.assertEqual(payload["external_evaluation_examples"], 3000)
        self.assertEqual([row["users"] for row in payload["cohorts"]], [1000] * 3)
        self.assertTrue(payload["cohorts_disjoint"])
        self.assertTrue(payload["cohorts_partition_selected_users"])
        self.assertFalse(payload["external_target_read_during_early_stop"])
        self.assertFalse(payload["official_test_read"])
        self.assertFalse(payload["sports_read"])
        manifest = json.loads(
            (ROOT / "artifacts/phase17/s2r_preflight/r2_contract/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["early_stop_user_ids_sha256"], manifest["early_stop_user_ids_sha256"])


if __name__ == "__main__":
    unittest.main()
