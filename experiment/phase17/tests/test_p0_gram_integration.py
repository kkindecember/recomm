from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "GRAM/src"))
from model.gram import GRAM  # noqa: E402
from model.gram_t5_config import T5Config  # noqa: E402


def tiny_config(module_id: str) -> T5Config:
    config = T5Config(
        vocab_size=67,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    config.max_seq_len = 4
    config.max_item_num = 3
    config.use_position_embedding = False
    config.sample_num = 1
    config.cf0_enabled = False
    config.hi_gram_enabled = False
    config.s17_modules = module_id
    config.s17_transition_map = ""
    return config


class P0GRAMIntegrationTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(
            [
                [[2, 3, 0, 0], [4, 5, 6, 0], [7, 8, 0, 0]],
                [[2, 9, 0, 0], [10, 11, 0, 0], [0, 0, 0, 0]],
            ]
        )
        return input_ids, input_ids.ne(0).long(), torch.tensor([[3, 2], [4, 0]])

    def _run(self, module_id: str) -> None:
        torch.manual_seed(17)
        model = GRAM(tiny_config(module_id)).train()
        input_ids, attention, history_ids = self._inputs()
        labels = torch.tensor([[12, 13, 1], [14, 15, 1]])
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            history_item_ids=history_ids,
            history_item_mask=history_ids.ne(0),
            target_item_ids=torch.tensor([5, 6]),
            labels=labels,
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        runtime_parameters = list(model.encoder.migration_runtime.parameters())
        if runtime_parameters:
            self.assertTrue(
                all(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in runtime_parameters
                )
            )

    def _run_generation_without_labels(self, module_id: str) -> None:
        torch.manual_seed(17)
        model = GRAM(tiny_config(module_id)).eval()
        input_ids, attention, history_ids = self._inputs()
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                history_item_ids=history_ids,
                history_item_mask=history_ids.ne(0),
                max_length=4,
                num_beams=2,
            )
        self.assertEqual(output.size(0), input_ids.size(0))

    def test_a0_bear_in_gram(self) -> None:
        self._run("A0_bear")

    def test_a1_prefix_curriculum_in_gram(self) -> None:
        self._run("A1_prefixcurr")

    def test_a0_generation_bypasses_training_loss(self) -> None:
        self._run_generation_without_labels("A0_bear_proxy")

    def test_a1_generation_bypasses_training_loss(self) -> None:
        self._run_generation_without_labels("A1_prefixcurr")

    def test_b0_mvi_runtime_in_gram(self) -> None:
        self._run("B0_mvi")

    def test_b1_latte_runtime_in_gram(self) -> None:
        self._run("B1_latte")

    def test_c0_biflow_in_gram(self) -> None:
        self._run("C0_biflow")

    def test_d0_transition_teacher_in_gram(self) -> None:
        self._run("D0_ted")

    def test_e0_shortcut_fid_in_gram(self) -> None:
        self._run("E0_shortcut_fid")

    def test_e0_controls_in_gram(self) -> None:
        self._run("E0_shortcut_fid_full_control")
        self._run("E0_shortcut_fid_random_control")


if __name__ == "__main__":
    unittest.main()
