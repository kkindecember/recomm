from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from official_specgr_runtime import RECBOLE_ROOT, SPECGR_ROOT, load_official_unisrec_class
from specgr_faithful import (
    GRAMSelfDrafter,
    OfficialUniSRecDrafterGRAM,
    TrainingBudget,
    adaptive_exit,
    assert_splus_control_budget_match,
    constrained_draft,
    finalize_recommendations,
    guided_prefix_mask,
    sequence_item_contrastive_loss,
    splus_finetune_loss,
    splus_pretrain_loss,
    strict_accept,
    target_aware_score,
    validate_cold_content_only,
)


class SpecGRFaithfulTests(unittest.TestCase):
    def test_official_unisrec_and_recbole_source_files(self) -> None:
        klass = load_official_unisrec_class()
        self.assertEqual(
            Path(inspect.getsourcefile(klass)).resolve(),
            (SPECGR_ROOT / "models/draft/UniSRec/model.py").resolve(),
        )
        embeddings = torch.randn(17, 1024)
        embeddings[0].zero_()
        wrapper = OfficialUniSRecDrafterGRAM(embeddings)
        transformer = type(wrapper.model.trm_encoder)
        self.assertEqual(
            Path(inspect.getsourcefile(transformer)).resolve(),
            (RECBOLE_ROOT / "recbole/model/layers.py").resolve(),
        )
        sequence = torch.randint(1, 17, (3, 20))
        loss = wrapper.calculate_loss(sequence, torch.tensor([20, 20, 20]), torch.tensor([1, 2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_constrained_draft_without_replacement(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        first = constrained_draft(logits, 2)
        second = constrained_draft(logits, 2)
        self.assertEqual(first.tolist(), [[1, 2]])
        self.assertEqual(second.tolist(), [[3, 0]])

    def test_target_aware_variable_score_matches_manual_fixed_width(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(2, 4, 9)
        candidates = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        lengths = torch.tensor([4, 2])
        actual = target_aware_score(logits, candidates, lengths)
        losses = F.cross_entropy(logits.reshape(-1, 9), candidates.reshape(-1), reduction="none").reshape(2, 4)
        expected = torch.stack([-losses[0].mean(), -losses[1, :2].mean()])
        self.assertTrue(torch.allclose(actual, expected))

    def test_acceptance_is_strict(self) -> None:
        self.assertEqual(strict_accept(torch.tensor([-1.7, -1.8, -1.9]), -1.8).tolist(), [True, False, False])

    def test_guided_prefix_supports_variable_complete_paths(self) -> None:
        paths = [(1, 2), (1, 2, 3), (1, 4, 5), (2, 1)]
        self.assertEqual(guided_prefix_mask(paths, [(1, 2)]).tolist(), [True, True, False, False])

    def test_adaptive_exit(self) -> None:
        self.assertFalse(adaptive_exit(2, 3, 2, 4))
        self.assertTrue(adaptive_exit(3, 3, 2, 4))
        self.assertTrue(adaptive_exit(0, 3, 4, 4))

    def test_finalize_deduplicates_and_falls_back(self) -> None:
        result = finalize_recommendations(
            [("a", 0.9), ("a", 0.8)],
            [("b", 0.4), ("c", 0.2)],
            [("b", 0.5), ("d", 0.3)],
            3,
        )
        self.assertEqual([item for item, _ in result], ["a", "b", "d"])

    def test_contrastive_loss_matches_official_equation(self) -> None:
        sequence = F.normalize(torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]), dim=-1)
        positive = F.normalize(torch.tensor([[1.5, 1.0], [1.0, 1.5], [2.0, 2.0]]), dim=-1)
        ids = torch.tensor([7, 8, 8])
        actual = sequence_item_contrastive_loss(sequence, positive, ids, temperature=0.7)
        logits = sequence @ positive.T / 0.7
        duplicate = ids[:, None].eq(ids[None, :]) ^ torch.eye(3, dtype=torch.bool)
        logits = torch.where(duplicate, torch.tensor(0.0), logits)
        expected = -(torch.diag(logits) - torch.logsumexp(logits, dim=1)).mean()
        self.assertTrue(torch.allclose(actual, expected))

    def test_weighted_objectives(self) -> None:
        emb, gen = torch.tensor(2.0), torch.tensor(3.0)
        self.assertEqual(float(splus_pretrain_loss(emb, gen)), 15.0)
        self.assertEqual(float(splus_finetune_loss(emb, gen)), 15.0)

    def test_normalized_self_draft(self) -> None:
        seq = torch.tensor([[3.0, 4.0]])
        index = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        scores = GRAMSelfDrafter.draft_logits(seq, index)
        self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))

    def test_splus_control_budget_must_match_every_field(self) -> None:
        budget = TrainingBudget("sha", 10, 2, "AdamW", 1e-3, 0.05, 10, 1, 4, 5, 1, 100)
        self.assertTrue(assert_splus_control_budget_match(budget, budget)["matched"])
        mismatch = TrainingBudget("sha", 10, 3, "AdamW", 1e-3, 0.05, 10, 1, 4, 5, 1, 100)
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            assert_splus_control_budget_match(budget, mismatch)

    def test_cold_items_are_candidates_not_labels(self) -> None:
        audit = validate_cold_content_only(["w1", "w2"], ["c1", "c2"], ["w1", "c1", "c2"])
        self.assertEqual(audit["cold_interaction_label_count"], 0)
        with self.assertRaises(ValueError):
            validate_cold_content_only(["w1", "c1"], ["c1"], ["c1"])


if __name__ == "__main__":
    unittest.main()
