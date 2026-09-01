from __future__ import annotations

import random
import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_arm_contracts import (
    build_preregistered_matrix,
    decoder_paths,
    full_semantic_vocabulary,
    gram_target_text,
    load_and_validate_arm_matrix,
    validate_preregistered_matrix,
)


ROOT = Path(__file__).resolve().parents[3]


class FullLatteArmContractTests(unittest.TestCase):
    def test_checked_in_matrix_equals_generated_contract(self) -> None:
        checked_in = load_and_validate_arm_matrix(
            ROOT / "experiment/phase17/config/s17_fp12_latte_arm_matrix.json"
        )
        generated = build_preregistered_matrix()
        self.assertEqual(checked_in, generated)
        validate_preregistered_matrix(checked_in)

    def test_g1_g2_share_full_added_vocabulary(self) -> None:
        vocabulary = full_semantic_vocabulary()
        self.assertEqual(len(vocabulary), 776)
        self.assertEqual(len(set(vocabulary)), 776)
        self.assertIn("<s17_sid2_255>", vocabulary)
        self.assertIn("<s17_latent_7>", vocabulary)

    def test_g2_samples_latent_per_exposure_and_keeps_sid(self) -> None:
        rng = random.Random(2023)
        semantic = {"item": (4, 5, 6)}
        outputs = [
            gram_target_text(
                "G2_GRAM_LATTE_FULL",
                "item",
                lexical_ids={"item": "lex"},
                semantic_codes=semantic,
                rng=rng,
            )
            for _ in range(1000)
        ]
        roots = {value.split()[0] for value in outputs}
        suffixes = {" ".join(value.split()[1:]) for value in outputs}
        self.assertEqual(len(roots), 8)
        self.assertEqual(suffixes, {"<s17_sid0_4> <s17_sid1_5> <s17_sid2_6>"})

    def test_g2_decoder_has_eight_paths_per_item(self) -> None:
        paths = decoder_paths(
            "G2_GRAM_LATTE_FULL",
            lexical_ids={"item": "lex"},
            semantic_codes={"item": (4, 5, 6)},
        )
        self.assertEqual(len(paths["item"]), 8)
        self.assertEqual(len(set(paths["item"])), 8)


if __name__ == "__main__":
    unittest.main()
