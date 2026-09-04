from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from transformers import T5Config, T5ForConditionalGeneration

from experiment.phase17.core.full_setrec_backend import (
    SETREC_ARMS,
    FullSetRecModel,
    SetRecBatch,
    collate_setrec_examples,
    history_visibility_mask,
    query_visibility_mask,
    repo_grouped_position_ids,
)


def tiny_t5() -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=64,
        d_model=16,
        d_ff=32,
        d_kv=8,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        use_cache=False,
    )
    return T5ForConditionalGeneration(config)


class FullSetRecBackendTests(unittest.TestCase):
    def test_collator_reserves_zero_for_padding(self) -> None:
        examples = [
            SimpleNamespace(history=("a", "b"), target="c"),
            SimpleNamespace(history=("b",), target="a"),
        ]
        batch = collate_setrec_examples(
            examples, item_to_index={"a": 0, "b": 1, "c": 2}, max_history_items=3
        )
        self.assertEqual(batch.history_item_ids.tolist(), [[0, 1, 2], [0, 0, 2]])
        self.assertEqual(
            batch.history_item_mask.tolist(),
            [[False, True, True], [False, False, True]],
        )
        self.assertEqual(batch.target_item_indices.tolist(), [2, 0])

    def test_repo_grouped_positions_share_all_five_item_tokens(self) -> None:
        self.assertEqual(
            repo_grouped_position_ids(12).tolist(),
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
        )

    def test_paper_visibility_has_no_cross_dimension_or_future_leak(self) -> None:
        item_mask = torch.tensor([[True, True, True]])
        visibility, valid = history_visibility_mask(
            "S1P_SETREC_PAPER_FAITHFUL", item_mask
        )
        self.assertEqual(tuple(visibility.shape), (1, 15, 15))
        self.assertTrue(bool(visibility[0, 7, 7]))
        self.assertFalse(bool(visibility[0, 7, 8]))
        self.assertTrue(bool(visibility[0, 7, 1]))
        self.assertFalse(bool(visibility[0, 7, 11]))
        self.assertTrue(bool(valid.all()))

    def test_query_visibility_separates_control_and_treatments(self) -> None:
        ordered = query_visibility_mask(
            "S0_SETREC_ORDERED_CONTROL", device=torch.device("cpu")
        )
        independent = query_visibility_mask(
            "S1R_SETREC_REPO_PARITY", device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(ordered, torch.ones(5, 5, dtype=torch.bool).tril()))
        self.assertTrue(torch.equal(independent, torch.eye(5, dtype=torch.bool)))

    def test_all_four_arms_run_full_catalog_grounding_and_backprop(self) -> None:
        torch.manual_seed(2023)
        cf = torch.randn(7, 64)
        semantic = torch.randn(7, 12)
        history = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long)
        history_mask = history.ne(0)
        targets = torch.tensor([2, 5], dtype=torch.long)
        parameter_counts = []
        for arm in SETREC_ARMS:
            model = FullSetRecModel(
                arm_id=arm,
                backbone=tiny_t5(),
                cf_embeddings=cf,
                semantic_features=semantic,
                prompt_input_ids=torch.tensor([[2, 3, 1]], dtype=torch.long),
            )
            gram_ids = gram_mask = None
            if arm == "S2_GRAM_SETREC_PAPER_FULL":
                gram_ids = torch.tensor(
                    [[[2, 3, 1, 0], [4, 5, 1, 0]], [[2, 6, 1, 0], [7, 8, 1, 0]]]
                )
                gram_mask = gram_ids.ne(0)
            batch = SetRecBatch(
                history_item_ids=history,
                history_item_mask=history_mask,
                target_item_indices=targets,
                gram_input_ids=gram_ids,
                gram_attention_mask=gram_mask,
            )
            output = model(batch, beta=0.7)
            self.assertEqual(
                tuple(output.grounding.per_dimension_scores.shape), (5, 2, 7)
            )
            self.assertEqual(tuple(output.grounding.item_scores.shape), (2, 7))
            self.assertTrue(bool(torch.isfinite(output.loss)))
            output.loss.backward()
            self.assertIsNotNone(model.query_vectors.grad)
            parameter_counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(parameter_counts)), 1)


if __name__ == "__main__":
    unittest.main()
