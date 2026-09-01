from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_native_backend import (
    cpu_preflight_native_arm,
)


ROOT = Path(__file__).resolve().parents[3]


class FullLatteNativeBackendTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("datasets") is not None
        and importlib.util.find_spec("accelerate") is not None,
        "requires the pinned official LATTE environment",
    )
    def test_pinned_official_n0_n1_cpu_preflights(self) -> None:
        n0 = cpu_preflight_native_arm(ROOT, "N0_NATIVE_PSID")
        n1 = cpu_preflight_native_arm(ROOT, "N1_NATIVE_LATTE")
        self.assertEqual(n0["state"], "PASS_CPU_PREFLIGHT")
        self.assertEqual(n1["state"], "PASS_CPU_PREFLIGHT")
        self.assertEqual(n0["catalog_items"], 11924)
        self.assertEqual(n1["catalog_items"], 11924)
        self.assertEqual(n0["rolling_train_examples"], 56421)
        self.assertEqual(n1["rolling_train_examples"], 56421)
        self.assertEqual(n0["vocab_size"], 771)
        self.assertEqual(n1["vocab_size"], 779)
        self.assertEqual(n0["train_target_token_length"], 4)
        self.assertEqual(n1["train_target_token_length"], 5)
        self.assertEqual(n1["latent_token_ids_observed"], list(range(1, 9)))
        self.assertTrue(n0["official_model_module"].endswith("PSID.model"))
        self.assertTrue(n1["official_model_module"].endswith("Latte.model"))
        self.assertFalse(n0["external_target_materialized"])
        self.assertFalse(n1["external_target_materialized"])


if __name__ == "__main__":
    unittest.main()
