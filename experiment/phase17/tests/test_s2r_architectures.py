from __future__ import annotations

import unittest

import torch

from experiment.phase17.core.s2r_architectures import (
    S2RSemanticIDModel,
    item_scorer_gradient_norm,
    parameter_count,
    smoke_t5_config,
)
from experiment.phase17.core.s2r_sid import SemanticIDCodec


class S2RArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2023)
        mapping = {
            "i1": (0, 0),
            "i2": (0, 1),
            "i3": (1, 0),
            "i4": (1, 1),
        }
        self.codec = SemanticIDCodec(
            mapping, [2, 2], n_latent_tokens=3, max_history_items=3
        )
        self.config = smoke_t5_config(self.codec, capacity="tiny")

    def batch(self, *, latte: bool) -> dict[str, torch.Tensor]:
        input_ids, attention = self.codec.encode_input("u1", ["i1", "i2"])
        latent = self.codec.base_latent_token if latte else None
        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention]),
            "labels": torch.tensor([self.codec.encode_label("i3", latent_token=latent)]),
            "target_item_index": torch.tensor([self.codec.item_to_index["i3"]]),
        }

    def test_gryphon_control_and_treatment_have_identical_capacity(self) -> None:
        control = S2RSemanticIDModel(
            self.codec, arm="gryphon_beam_control", config=self.config
        )
        treatment = S2RSemanticIDModel(
            self.codec, arm="gryphon_item", config=self.config
        )
        self.assertEqual(parameter_count(control), parameter_count(treatment))

    def test_gryphon_item_objective_reaches_item_scorer(self) -> None:
        model = S2RSemanticIDModel(
            self.codec, arm="gryphon_item", config=self.config
        )
        output = model(**self.batch(latte=False))
        output.loss.backward()
        self.assertGreater(item_scorer_gradient_norm(model), 0.0)

    def test_latte_and_psid_emit_only_catalog_items(self) -> None:
        for arm, latte in (("psid_control", False), ("latte_full", True)):
            model = S2RSemanticIDModel(self.codec, arm=arm, config=self.config)
            model.eval()
            batch = self.batch(latte=latte)
            rankings = model.generate_ranked(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=3,
                top_k=3,
            )
            self.assertTrue(rankings[0])
            self.assertTrue(
                all(row.item_id in self.codec.item_to_code for row in rankings[0])
            )

    def test_r2_capacity_is_larger_than_r1(self) -> None:
        r1 = S2RSemanticIDModel(
            self.codec,
            arm="latte_full",
            config=smoke_t5_config(self.codec, capacity="r1"),
        )
        r2 = S2RSemanticIDModel(
            self.codec,
            arm="latte_full",
            config=smoke_t5_config(self.codec, capacity="r2"),
        )
        self.assertGreater(parameter_count(r2), parameter_count(r1))


if __name__ == "__main__":
    unittest.main()
