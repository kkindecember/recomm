"""Data boundary, item grounding, and pinned native mechanism checks on CPU."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from experiment.phase18.protocol import s18_diffgrm_native as native


class NativeContracts(unittest.TestCase):
    def test_original_split_boundary_and_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "GRAM/rec_datasets/Beauty"
            data.mkdir(parents=True)
            (data / "item_plain_text.txt").write_text("a first\nb second\nc third\nd fourth\ne fifth\n")
            (data / "user_sequence.txt").write_text("u a b c d e\n")
            with patch.object(native, "ROOT", root):
                dataset = native.LocalOriginalDataset({"dataset": "Beauty", "minimum_history": 1, "max_history": 50}, root)
            self.assertEqual(dataset.train, [
                {"user_id": "u", "history": [1], "target": 2},
                {"user_id": "u", "history": [1, 2], "target": 3}])
            self.assertEqual(dataset.validation, [{"user_id": "u", "history": [1, 2, 3], "target": 4}])
            self.assertEqual(list(dataset.split()["train"]["item_seq"]), [["a", "b", "c"]])
            self.assertEqual(set(dataset.counts), {1, 2, 3})

    def test_sid_padding_and_inference_has_no_target(self):
        codes = torch.tensor([[-1] * 4, [1] * 4, [2] * 4, [3] * 4])
        rows = [{"user_id": "u", "history": [2, 1], "target": 3}]
        batch = next(iter(DataLoader(native.SidDataset(rows, codes, 4), batch_size=1)))
        self.assertEqual(batch["history_sid"].tolist(), [[[2] * 4, [1] * 4, [-1] * 4, [-1] * 4]])
        self.assertEqual(batch["history_mask"].tolist(), [[True, True, False, False]])
        self.assertEqual(set(native.inference_batch(batch)), {"history_sid", "history_mask"})

    def test_collision_grounding_is_unique_and_does_not_fill_invalid(self):
        mapping = {(0, 0, 0, 0): [4, 2], (1, 1, 1, 1): [3, 4]}
        raw = [[9] * 4, [0] * 4, [0] * 4, [1] * 4]
        self.assertEqual(native.resolve_items(raw, mapping), [4, 2, 3])
        self.assertEqual(native.resolve_items(raw, mapping, topk=2), [4, 2])
        self.assertEqual(native.resolve_items([[9] * 4], mapping), [])

    def test_pinned_guided_training_and_history_only_cpd_roundtrip(self):
        config = json.loads((native.ROOT / "experiment/phase18/config/s18_diffgrm_native_beauty.json").read_text())
        with TemporaryDirectory() as directory:
            values = native.official_config(config, Path(directory), "cpu")
            values.update(n_embd=32, n_inner=64, encoder_n_layer=1, decoder_n_layer=2, codebook_size=8,
                          max_history_len=4, max_hist_len=4)
            values["vectorized_beam_search"].update(top_k_final=8, val={"beam_act": 8, "beam_max": 8})
            native.seed_all(2023)
            tokenizer = SimpleNamespace(vocab_size=35, sid_offset=3, mask_token=-1,
                                        codebooks_to_item_id=lambda code: None)
            model = native.model_for(values, tokenizer)
            batch = {"history_sid": torch.randint(0, 8, (2, 4, 4)),
                     "history_mask": torch.tensor([[True, True, True, False], [True, True, True, True]]),
                     "decoder_labels": torch.randint(0, 8, (2, 4))}
            batch["history_sid"][0, -1] = -1
            batch["decoder_input_ids"] = batch["decoder_labels"].clone()
            self.assertEqual(model.masking_strategy, "guided")
            self.assertEqual(model.guided_select, "least")
            self.assertEqual(model.augment_factor, 4)
            result = model(batch)
            self.assertTrue(torch.isfinite(result.loss))
            result.loss.backward()
            for fragment in ("item_mlp", "encoder_blocks", "decoder_blocks", "mask_emb_table"):
                grads = [p.grad for name, p in model.named_parameters() if fragment in name and p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            model.eval()
            generated = model.generate(native.inference_batch(batch), n_return_sequences=8)
            self.assertEqual(tuple(generated.shape), (2, 8, 4))
            self.assertTrue(((generated >= 0) & (generated < 8)).all())
            native.checkpoint(Path(directory) / "model.pt", model)
            restored = native.model_for(values, tokenizer)
            restored.load_state_dict(torch.load(Path(directory) / "model.pt", weights_only=True)["model"])
            batch["decoder_labels"].fill_(7)
            batch["decoder_input_ids"].fill_(7)
            repeated = restored.generate(native.inference_batch(batch), n_return_sequences=8)
            self.assertTrue(torch.equal(generated, repeated))


if __name__ == "__main__":
    unittest.main()
