from __future__ import annotations

import unittest

import torch

from experiment.phase17.core.full_setrec_contracts import (
    SemanticSetAutoencoder,
    full_set_recovery,
    ground_continuous_queries,
    independent_query_mask,
    item_group_position_ids,
    paper_sparse_history_mask,
    setrec_joint_loss,
)


class FullSetRecContractTests(unittest.TestCase):
    def test_repo_position_parity_groups_tokens_without_hiding_them(self) -> None:
        positions = item_group_position_ids(
            n_items=3, n_tokens_per_item=5, prefix_tokens=2, suffix_tokens=1
        )
        self.assertEqual(positions[:2].tolist(), [0, 1])
        self.assertEqual(positions[2:7].tolist(), [2] * 5)
        self.assertEqual(positions[7:12].tolist(), [3] * 5)
        self.assertEqual(positions[-1].item(), 5)

    def test_paper_sparse_history_removes_intra_item_and_future_visibility(self) -> None:
        mask = paper_sparse_history_mask(n_items=3, n_tokens_per_item=5)
        self.assertEqual(mask.shape, (15, 15))
        self.assertTrue(mask[7, 7])
        self.assertFalse(mask[7, 8])
        self.assertTrue(mask[7, 1])
        self.assertFalse(mask[7, 11])
        self.assertEqual(int(mask[0].sum().item()), 1)

    def test_query_mask_is_identity(self) -> None:
        mask = independent_query_mask(5)
        self.assertTrue(torch.equal(mask, torch.eye(5, dtype=torch.bool)))

    def test_semantic_ae_emits_four_continuous_tokens_and_backpropagates(self) -> None:
        torch.manual_seed(2023)
        model = SemanticSetAutoencoder(
            semantic_dim=12,
            model_dim=8,
            n_semantic_tokens=4,
            hidden_dims=(16, 12),
        )
        features = torch.randn(6, 12)
        tokens, reconstruction = model(features)
        self.assertEqual(tokens.shape, (6, 4, 8))
        self.assertEqual(reconstruction.shape, features.shape)
        loss = torch.nn.functional.mse_loss(reconstruction, features)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_full_catalog_grounding_and_joint_loss_recover_target(self) -> None:
        torch.manual_seed(2023)
        batch_size, n_query, catalog_size, dim = 3, 5, 7, 8
        targets = torch.tensor([1, 3, 5])
        corpus = torch.randn(n_query, catalog_size, dim)
        queries = torch.stack(
            [corpus[:, target, :] for target in targets], dim=0
        ) * 5.0
        grounded = ground_continuous_queries(queries, corpus, beta=0.7)
        self.assertEqual(grounded.per_dimension_scores.shape, (5, 3, 7))
        self.assertEqual(grounded.item_scores.shape, (3, 7))
        self.assertTrue(full_set_recovery(grounded.per_dimension_scores, targets).all())
        semantic = torch.randn(batch_size, 12)
        reconstruction = semantic + 0.01
        losses = setrec_joint_loss(
            grounded.per_dimension_scores,
            targets,
            semantic_features=semantic,
            semantic_reconstruction=reconstruction,
            alpha=0.7,
        )
        self.assertTrue(torch.isfinite(losses.loss))
        self.assertAlmostEqual(
            float(losses.loss.item()),
            float((losses.generation_loss + 0.7 * losses.reconstruction_loss).item()),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
