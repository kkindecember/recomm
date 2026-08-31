from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "artifacts/phase17/fullport/manifests"
CONFIG = ROOT / "artifacts/phase17/fullport/config"
SUMMARY = ROOT / "artifacts/phase17/fullport/fp0/attempt_001/summary.json"


class S17FP0FreezeTests(unittest.TestCase):
    def load(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_sources_are_commit_and_license_frozen(self) -> None:
        latte = self.load(MANIFESTS / "latte_source_manifest.json")
        setrec = self.load(MANIFESTS / "setrec_source_manifest.json")
        self.assertEqual(latte["commit"], "05e4e6d983225bcb7172f148a076890e80c524d1")
        self.assertEqual(latte["license_status"], "MIT")
        self.assertEqual(setrec["commit"], "2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7")
        self.assertEqual(setrec["license_status"], "NO_STANDARD_LICENSE_FILE")
        self.assertIn("clean-room", setrec["reuse_policy"])

    def test_latte_official_config_parity(self) -> None:
        payload = self.load(CONFIG / "latte_native_toys_d0.json")
        official = payload["official_config_primary"]
        self.assertEqual(official["epochs"], 150)
        self.assertEqual(official["patience"], 50)
        self.assertEqual(official["n_latent_tokens"], 8)
        self.assertEqual(official["vq_method"], "rqkmeans")
        self.assertEqual(official["vq_n_codebooks"], 3)
        self.assertEqual(official["vq_codebook_size"], 256)
        self.assertEqual(official["d_model"], 128)
        self.assertEqual(official["num_layers"], 4)
        self.assertTrue(all(payload["parity_checks"].values()))

    def test_setrec_official_config_and_mechanism_parity(self) -> None:
        payload = self.load(CONFIG / "setrec_native_toys_d0.json")
        official = payload["official_source_protocol"]
        self.assertEqual(official["epochs"], 30)
        self.assertEqual(official["global_batch_size"], 512)
        self.assertEqual(official["world_size"], 4)
        self.assertEqual(official["n_sem"], 4)
        self.assertEqual(official["n_query"], 5)
        self.assertEqual(official["alpha"], 0.7)
        self.assertEqual(official["max_history_items"], 50)
        self.assertTrue(all(payload["parity_checks"].values()))

    def test_existing_proxies_cannot_be_called_full(self) -> None:
        for name in ("latte_fidelity_matrix.json", "setrec_fidelity_matrix.json"):
            payload = self.load(MANIFESTS / name)
            self.assertFalse(payload["existing_s17_2r_is_full"])
            self.assertFalse(payload["full_name_allowed_before_components_pass"])
            self.assertFalse(payload["implementation_ready"])

    def test_data_gate_is_sealed_and_full_d0(self) -> None:
        data = self.load(MANIFESTS / "data_manifest.json")
        self.assertEqual(data["users"], 12833)
        self.assertFalse(data["external_d0_target_materialized"])
        self.assertFalse(data["official_test_read"])
        self.assertFalse(data["sports_read"])
        self.assertFalse(data["d1_read"])
        self.assertFalse(data["d2_read"])
        summary = self.load(SUMMARY)
        self.assertEqual(summary["verdict"], "PASS_S17_FP0_SOURCE_DATA_FIDELITY_FREEZE")
        self.assertFalse(summary["effect_experiment_started"])
        self.assertFalse(summary["gpu_used"])


if __name__ == "__main__":
    unittest.main()
