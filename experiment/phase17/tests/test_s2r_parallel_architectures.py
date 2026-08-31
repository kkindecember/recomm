from __future__ import annotations

import unittest

import torch

from experiment.phase17.core.s2r_architectures import parameter_count
from experiment.phase17.core.s2r_parallel_architectures import (
    S2RParallelIDModel,
    parallel_gradient_norm,
    parallel_smoke_config,
)
from experiment.phase17.core.s2r_sid import SemanticIDCodec


class S2RParallelArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2023)
        self.codec = SemanticIDCodec(
            {
                "i1": (0, 0),
                "i2": (0, 1),
                "i3": (1, 0),
                "i4": (1, 1),
            },
            [2, 2],
            n_latent_tokens=2,
            max_history_items=3,
        )
        self.config = parallel_smoke_config(self.codec, capacity="tiny")

    def batch(self) -> dict[str, torch.Tensor]:
        input_ids, attention = self.codec.encode_input("u", ["i1", "i2"])
        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention]),
            "labels": torch.tensor([self.codec.encode_label("i3")]),
            "target_item_index": torch.tensor([self.codec.item_to_index["i3"]]),
        }

    def test_matched_pairs_have_identical_parameter_counts(self) -> None:
        for control, treatment in (
            ("diffgrm_ar_control", "diffgrm_masked"),
            ("setrec_ar_control", "setrec_full"),
        ):
            left = S2RParallelIDModel(self.codec, arm=control, config=self.config)
            right = S2RParallelIDModel(self.codec, arm=treatment, config=self.config)
            self.assertEqual(parameter_count(left), parameter_count(right))

    def test_all_parallel_objectives_backpropagate(self) -> None:
        for arm in (
            "diffgrm_ar_control",
            "diffgrm_masked",
            "setrec_ar_control",
            "setrec_full",
        ):
            model = S2RParallelIDModel(self.codec, arm=arm, config=self.config)
            output = model(**self.batch())
            output.loss.backward()
            self.assertGreater(parallel_gradient_norm(model), 0.0)

    def test_all_parallel_decoders_resolve_catalog_items(self) -> None:
        for arm in (
            "diffgrm_ar_control",
            "diffgrm_masked",
            "setrec_ar_control",
            "setrec_full",
        ):
            model = S2RParallelIDModel(self.codec, arm=arm, config=self.config)
            model.eval()
            batch = self.batch()
            with torch.no_grad():
                rows = model.generate_ranked(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=3,
                    top_k=3,
                )
            self.assertTrue(rows[0])
            self.assertTrue(
                all(row.item_id in self.codec.item_to_code for row in rows[0])
            )

    def test_r2_parallel_capacity_is_larger_than_r1(self) -> None:
        r1 = S2RParallelIDModel(
            self.codec,
            arm="diffgrm_masked",
            config=parallel_smoke_config(self.codec, capacity="r1"),
        )
        r2 = S2RParallelIDModel(
            self.codec,
            arm="diffgrm_masked",
            config=parallel_smoke_config(self.codec, capacity="r2"),
        )
        self.assertGreater(parameter_count(r2), parameter_count(r1))


if __name__ == "__main__":
    unittest.main()
