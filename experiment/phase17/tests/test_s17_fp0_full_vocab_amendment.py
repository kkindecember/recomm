from __future__ import annotations

import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_arm_contracts import full_semantic_vocabulary
from experiment.phase17.protocol.s17_fp0_full_vocab_amendment import build_amendment


ROOT = Path(__file__).resolve().parents[3]


class FullVocabularyAmendmentTests(unittest.TestCase):
    def test_real_amendment_contract_identifies_exactly_one_missing_token(self) -> None:
        result = build_amendment(ROOT, apply=False)
        self.assertEqual(result["source_observed_token_count"], 775)
        self.assertEqual(result["complete_token_count"], 776)
        self.assertEqual(len(result["missing_observed_tokens_added"]), 1)
        self.assertFalse(result["original_attempt_modified"])
        self.assertFalse(result["official_semantic_ids_modified"])

    def test_complete_vocabulary_is_position_complete(self) -> None:
        values = full_semantic_vocabulary()
        self.assertEqual(len(values), 776)
        for digit in range(3):
            for code in range(256):
                self.assertIn(f"<s17_sid{digit}_{code}>", values)
        for latent in range(8):
            self.assertIn(f"<s17_latent_{latent}>", values)


if __name__ == "__main__":
    unittest.main()
