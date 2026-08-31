from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "GRAM/src"))
from model.gram import GRAM  # noqa: E402
from model.gram_t5_config import T5Config  # noqa: E402


def tiny_config() -> T5Config:
    config = T5Config(
        vocab_size=41,
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
    config.max_item_num = 2
    config.use_position_embedding = False
    config.sample_num = 1
    config.cf0_enabled = False
    config.hi_gram_enabled = False
    return config


class AllFlagsOffEquivalenceTests(unittest.TestCase):
    def test_absent_and_explicit_empty_module_flags_are_bitwise_equal(self) -> None:
        torch.manual_seed(17)
        parent = GRAM(tiny_config()).eval()
        config = tiny_config()
        config.s17_modules = ""
        migrated = GRAM(config).eval()
        migrated.load_state_dict(copy.deepcopy(parent.state_dict()))
        input_ids = torch.tensor([[[2, 3, 0, 0], [4, 5, 6, 0]]])
        attention = input_ids.ne(0).long()
        labels = torch.tensor([[7, 8, 1]])
        with torch.no_grad():
            left = parent(input_ids=input_ids, attention_mask=attention, labels=labels)
            right = migrated(input_ids=input_ids, attention_mask=attention, labels=labels)
        self.assertTrue(parent.encoder.migration_runtime.is_identity)
        self.assertTrue(migrated.encoder.migration_runtime.is_identity)
        self.assertTrue(torch.equal(left.logits, right.logits))
        self.assertTrue(torch.equal(left.loss, right.loss))

    def test_identity_runtime_does_not_add_parameters(self) -> None:
        model = GRAM(tiny_config())
        runtime_parameters = list(model.encoder.migration_runtime.parameters())
        self.assertEqual(runtime_parameters, [])
