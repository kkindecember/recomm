from __future__ import annotations

import unittest

import torch

from experiment.phase17.protocol.s2r_r1_sid_runtime import generation_arguments


class S2RR1RuntimeTests(unittest.TestCase):
    def test_latte_only_argument_never_reaches_parallel_decoders(self) -> None:
        moved = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        config = {"num_beams": 50, "top_k": 50}
        for arm in (
            "diffgrm_ar_control",
            "diffgrm_masked",
            "setrec_ar_control",
            "setrec_full",
        ):
            self.assertNotIn(
                "latte_aggregation", generation_arguments(arm, moved, config)
            )
        self.assertEqual(
            generation_arguments("latte_full", moved, config)["latte_aggregation"],
            "logsumexp",
        )


if __name__ == "__main__":
    unittest.main()
