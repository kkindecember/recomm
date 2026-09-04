from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from experiment.phase17.core.full_setrec_cf_tokenizer import (
    RollingNextItemDataset,
    SetRecCFSpec,
    SetRecSASRec,
    evaluate_full_catalog,
    sampled_bce_loss,
    train_setrec_cf_tokenizer,
)
from experiment.phase17.core.fullport_data import FullportExample


class FullSetRecCFTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = ("a", "b", "c", "d", "e")
        self.item_to_index = {
            item: index + 1 for index, item in enumerate(self.items)
        }
        self.examples = [
            FullportExample("u1", ("a",), "b"),
            FullportExample("u1", ("a", "b"), "c"),
            FullportExample("u2", ("d",), "e"),
            FullportExample("u2", ("d", "e"), "c"),
        ]

    def test_rolling_dataset_reserves_zero_and_tracks_last_history(self) -> None:
        dataset = RollingNextItemDataset(self.examples, self.item_to_index, 4)
        sequence, last, target = dataset[1]
        self.assertEqual(sequence.tolist(), [1, 2, 0, 0])
        self.assertEqual(int(last), 1)
        self.assertEqual(int(target), 3)

    def test_sampled_bce_is_finite_and_backpropagates(self) -> None:
        spec = SetRecCFSpec(
            hidden_size=8, max_history_items=4, num_blocks=1, num_heads=2
        )
        model = SetRecSASRec(len(self.items), spec)
        dataset = RollingNextItemDataset(self.examples, self.item_to_index, 4)
        sequence, last, target = dataset[0]
        encoded = model(sequence[None])
        final = encoded[torch.arange(1), last[None]]
        loss = sampled_bce_loss(
            final,
            target[None],
            model,
            torch.arange(1, len(self.items) + 1),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.item_embedding.weight.grad)

    def test_tiny_training_exports_catalog_aligned_best_checkpoint(self) -> None:
        spec = SetRecCFSpec(
            hidden_size=8,
            max_history_items=4,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
            batch_size=2,
            epochs=2,
            eval_batch_size=2,
            top_k=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sasrec_item_embeddings.pt"
            payload = train_setrec_cf_tokenizer(
                ordered_items=self.items,
                train_examples=self.examples,
                dev_examples=self.examples[:2],
                output_path=output,
                device=torch.device("cpu"),
                spec=spec,
            )
            load_kwargs = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False
            saved = torch.load(output, **load_kwargs)
        self.assertEqual(payload["ordered_items"], list(self.items))
        self.assertEqual(saved["item_embeddings"].shape, (5, 8))
        self.assertIn(payload["best_epoch"], (1, 2))
        self.assertFalse(payload["external_target_materialized"])

    def test_full_catalog_evaluator_reports_bounded_metrics(self) -> None:
        spec = SetRecCFSpec(
            hidden_size=8,
            max_history_items=4,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
        )
        model = SetRecSASRec(len(self.items), spec)
        dataset = RollingNextItemDataset(self.examples, self.item_to_index, 4)
        metrics = evaluate_full_catalog(
            model,
            dataset,
            batch_size=2,
            top_k=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(metrics["n"], 4)
        self.assertGreaterEqual(metrics["hit@10"], 0.0)
        self.assertLessEqual(metrics["hit@10"], 1.0)
        self.assertGreaterEqual(metrics["ndcg@10"], 0.0)
        self.assertLessEqual(metrics["ndcg@10"], 1.0)


if __name__ == "__main__":
    unittest.main()
