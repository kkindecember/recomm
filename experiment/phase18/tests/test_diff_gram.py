"""Behavioral checks for the trainable DIFF memory/GRAM integration."""

import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import T5Config

from experiment.phase18.core.diff_sequence import DiffConfig, DiffSequence, FrequencyFilter, alignment_loss
from experiment.phase18.core.diff_gram import DiffGram
from experiment.phase18.core.diff_data import DiffDataset, fields

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "GRAM/src"))
from model import create_model


def tiny_model():
    config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                      num_decoder_layers=1, num_heads=2, dropout_rate=0.0,
                      decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    config.max_seq_len, config.max_item_num = 6, 20
    config.use_position_embedding, config.sample_num = 1, "1"
    config.cf0_enabled, config.hi_gram_enabled, config.s17_modules = False, False, ""
    gram = create_model("gram", config=config)
    table = [[0], [2], [3], [2], [4], [3]]
    diff_config = DiffConfig(hidden_size=16, inner_size=16, layers=2, dropout=0.0, cutoff=3)
    return DiffGram(gram, 5, [table], [5], diff_config)


def batch():
    ids = torch.tensor([[[2, 3, 1, 0], [4, 5, 1, 0], [0, 0, 0, 0]],
                        [[2, 3, 1, 0], [6, 7, 1, 0], [8, 9, 1, 0]]])
    return dict(input_ids=ids, attention_mask=ids.ne(0), history_item_ids=torch.tensor([[1, 0], [2, 3]]),
                labels=torch.tensor([[4, 1], [5, 1]]), target_item_ids=torch.tensor([4, 5]))


class DiffBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(23)

    def test_fixed_padding_and_history_order(self):
        model = tiny_model().diff.eval()
        short = torch.tensor([[3, 2, 1]])
        padded = torch.tensor([[3, 2, 1, 0, 0, 0]])
        self.assertEqual(model.chronological_ids(short)[0, :5].tolist(), [1, 2, 3, 0, 0])
        for left, right in zip(model(short), model(padded)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(torch.tensor([[1, 0, 2]]))
        with self.assertRaises(ValueError):
            model(torch.tensor([[0, 0]]))

    def test_frequency_beta_one_is_residual_identity_before_norm(self):
        module = FrequencyFilter(DiffConfig(hidden_size=4, dropout=0.0)).eval()
        module.beta.data.fill_(1)
        value = torch.randn(2, 20, 4)
        mask = torch.ones(2, 20, dtype=torch.bool)
        torch.testing.assert_close(module(value, mask), module.norm(2 * value), atol=1e-5, rtol=1e-5)

    def test_alignment_matches_explicit_cross_distribution(self):
        item, attrs = torch.randn(2, 3, 4), [torch.randn(2, 3, 4)]
        valid = torch.ones(2, 3, dtype=torch.bool)
        x = torch.nn.functional.normalize(item, dim=-1)
        a = torch.nn.functional.normalize(attrs[0], dim=-1)
        left = torch.log_softmax(x @ a.transpose(-1, -2) / .1, dim=-1)
        right = torch.log_softmax(a @ x.transpose(-1, -2) / .1, dim=-1)
        expected = -(left.diagonal(dim1=-2, dim2=-1) + right.diagonal(dim1=-2, dim2=-1)).mean() / 3
        torch.testing.assert_close(alignment_loss(item, attrs, valid), expected)

    def test_generation_loss_reaches_both_paths_and_attributes_while_base_frozen(self):
        model = tiny_model()
        model.set_backbone_frozen(True)
        inputs = batch()
        output = model(**inputs)
        self.assertTrue(torch.isfinite(output.loss))
        # Only the generation objective: auxiliary loss must not conceal a broken memory interface.
        generation = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1), inputs["labels"].flatten())
        generation.backward()
        for fragment in ("intermediate", ".early.", "attribute_embeddings", "memory_projection", ".beta"):
            norm = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                       if fragment in name and p.grad is not None)
            self.assertGreater(norm, 0, fragment)
        self.assertTrue(all(p.grad is None for p in model.gram.parameters()))
        self.assertFalse(model.gram.training)

    def test_memory_mask_and_checkpoint_generation(self):
        model = tiny_model().eval()
        data = batch()
        inference = {k: v for k, v in data.items() if k not in ("labels", "target_item_ids")}
        encoded, mask, _, _ = model.encode(**inference)
        self.assertEqual(encoded.last_hidden_state.shape[1], 12 + 20)
        self.assertEqual(mask[:, 12:].sum(1).tolist(), [1, 2])
        # Masking all appended memory recovers native decoder logits without running a baseline experiment.
        memory_off = mask.clone()
        memory_off[:, 12:] = False
        with torch.no_grad():
            original = model.gram(input_ids=data["input_ids"], attention_mask=data["attention_mask"], labels=data["labels"])
            masked = model.gram(encoder_outputs=encoded, attention_mask=memory_off, labels=data["labels"])
            torch.testing.assert_close(original.logits, masked.logits, rtol=1e-5, atol=1e-5)
            generated = model.generate(**inference, num_beams=2, max_length=5)
            restored = tiny_model().eval()
            restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            regenerated = restored.generate(**inference, num_beams=2, max_length=5)
        self.assertTrue(torch.equal(generated, regenerated))
        self.assertEqual(generated.size(0), 2)

    def test_target_change_cannot_change_encoder_input(self):
        catalog = {"lexical_ids": ["", "a", "b", "c"], "passages": ["", "A", "B", "C"]}
        rows = [{"user_id": "u", "history": [1], "target": target} for target in (2, 3)]
        dataset = DiffDataset(rows, catalog)
        self.assertEqual(dataset[0]["input"], dataset[1]["input"])
        self.assertEqual(dataset[0]["history_item_ids"], dataset[1]["history_item_ids"])
        self.assertNotEqual(dataset[0]["output"], dataset[1]["output"])
        self.assertEqual(fields("categories: games, puzzles; brand: na; description: okay"),
                         [["games", "puzzles"], []])

    def test_training_phases_tail_batch_and_checkpoint_selection(self):
        from experiment.phase18.protocol.s18_diff_gram import train_candidate
        config = json.loads((ROOT / "experiment/phase18/config/s18_diff_gram_toys.json").read_text())
        config["training"].update(joint_epochs=2, min_joint_epochs=2, effective_batch_size=4, num_workers=0)
        model = tiny_model()
        initial_base = model.gram.shared.weight.detach().clone()
        initial_new = model.memory_projection.weight.detach().clone()
        source = batch()
        rows = [{key: value[index % 2] for key, value in source.items()} for index in range(5)]

        def collate(examples):
            names = {"input_ids": "item_text_ids", "attention_mask": "item_text_masks", "labels": "target_ids",
                     "history_item_ids": "history_item_ids", "target_item_ids": "target_item_ids"}
            return {destination: torch.stack([row[key] for row in examples]) for key, destination in names.items()}

        frozen_at_evaluation = []

        def evaluation(model, *args, **kwargs):
            frozen_at_evaluation.append(model.backbone_frozen)
            if model.backbone_frozen:
                torch.testing.assert_close(initial_base, model.gram.shared.weight, rtol=0, atol=0)
            model.eval()
            return {"metrics": {"ndcg@10": .01 * len(frozen_at_evaluation)}}

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary)
            with patch("experiment.phase18.protocol.s18_diff_gram.evaluate", side_effect=evaluation):
                train_candidate(model, rows, rows, collate, None, config,
                                {"baseline_validation": {"ndcg@10": .015}}, output, torch.device("cpu"), 2)
            result = json.loads((output / "result.json").read_text())
            last = torch.load(output / "last.pt", map_location="cpu")
            self.assertEqual(result["state"], "COMPLETED")
            self.assertEqual(result["best_epoch"], 3)
            self.assertEqual(last["optimizer_steps"], 6)  # partial group is retained in every epoch
            self.assertEqual(frozen_at_evaluation, [True, False])
            self.assertFalse(torch.equal(initial_new, model.memory_projection.weight))
            self.assertFalse(torch.equal(initial_base, model.gram.shared.weight))

    def test_cross_cache_reuse_matches_full_reorder_and_rejects_cross_user_moves(self):
        from experiment.phase18.core.diff_generation_cache import reorder_with_shared_cross_cache
        states = (torch.randn(6, 2, 3, 4), torch.randn(6, 2, 3, 4),
                  torch.randn(2, 2, 9, 4).repeat_interleave(3, dim=0),
                  torch.randn(2, 2, 9, 4).repeat_interleave(3, dim=0))
        indices = torch.tensor([2, 0, 0, 4, 5, 3])
        result = reorder_with_shared_cross_cache((states,), indices, beams=3)[0]
        for original, reused in zip(states, result):
            torch.testing.assert_close(original.index_select(0, indices), reused, rtol=0, atol=0)
        self.assertIs(result[2], states[2])
        with self.assertRaises(ValueError):
            reorder_with_shared_cross_cache((states,), torch.tensor([3, 0, 0, 4, 5, 3]), beams=3)

    def test_cross_cache_reuse_preserves_multibatch_generation_scores(self):
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        model = tiny_model().eval()
        inference = {k: v for k, v in batch().items() if k not in ("labels", "target_item_ids")}
        kwargs = dict(num_beams=3, num_return_sequences=3, max_length=5,
                      return_dict_in_generate=True, output_scores=True)
        with torch.no_grad():
            before = model.generate(**inference, **kwargs)
            install_cross_cache_reuse(model.gram, beams=3)
            after = model.generate(**inference, **kwargs)
        self.assertTrue(torch.equal(before.sequences, after.sequences))
        torch.testing.assert_close(before.sequences_scores, after.sequences_scores, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
