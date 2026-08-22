from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from specgr_gram_adapter import (  # noqa: E402
    AuxiliaryContentDrafter,
    PathCatalog,
    VerifiedCandidate,
    finalize_recommendations,
    guided_redraft,
    candidate_token_log_probabilities,
    padded_candidate_labels,
    target_aware_scores_tensor,
    target_aware_score,
    drafter_cross_entropy,
    rank_drafter_items,
    validate_specgr_budget_trace,
)


def fixture_catalog() -> PathCatalog:
    return PathCatalog.build(
        {
            "w1": "|a|b|c",
            "w2": "|a|d|e",
            "c1": "|a|b|x|0",
            "c2": "|a|d|y",
            "c3": "|z|q|r",
        },
        warm_items={"w1", "w2"},
        cold_items={"c1", "c2", "c3"},
    )


class TestSpecGRGramAdapter(unittest.TestCase):
    def test_auxiliary_content_drafter_is_inductive_and_train_only(self):
        torch.manual_seed(7)
        model = AuxiliaryContentDrafter(
            item_content_embeddings=torch.randn(4, 6),
            hidden_size=4,
            max_history=3,
            transformer_layers=1,
            attention_heads=2,
            feedforward_size=8,
            dropout=0.0,
            temperature=0.07,
        )
        histories = torch.tensor([[0, 1, -1], [1, 2, 0]])
        lengths = torch.tensor([2, 3])
        logits = model(histories, lengths)
        self.assertEqual(tuple(logits.shape), (2, 4))
        loss = drafter_cross_entropy(
            logits, torch.tensor([1, 2]), warm_catalog_indices={0, 1, 2}
        )
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, "warm train-only"):
            drafter_cross_entropy(
                logits, torch.tensor([1, 3]), warm_catalog_indices={0, 1, 2}
            )

    def test_drafter_ranking_has_item_id_tie_break_and_exclusion(self):
        ranked = rank_drafter_items(
            torch.tensor([0.5, 0.5, -1.0]),
            ["b", "a", "c"],
            exclude_items={"c"},
        )
        self.assertEqual(ranked, ["a", "b"])

    def test_variable_length_target_aware_score(self):
        catalog = fixture_catalog()
        self.assertEqual(catalog.score_length("w1"), 3)
        self.assertEqual(catalog.score_length("c1"), 2)
        self.assertEqual(catalog.score_length("c3"), 2)
        self.assertAlmostEqual(target_aware_score([-1.0, -2.0, -100.0], 2), -1.5)

    def test_guided_redraft_is_unique_and_prefix_constrained(self):
        catalog = fixture_catalog()
        selected = guided_redraft(
            ["c1", "w1", "c2", "w2", "c3"],
            catalog=catalog,
            verifier_prefixes=[("a", "d")],
            prefix_depth=2,
            already_drafted={"c2"},
            draft_size=2,
        )
        self.assertEqual(selected, ["w2"])

    def test_finalize_uses_verifier_rank_then_unique_beam_fallback(self):
        catalog = fixture_catalog()
        output = finalize_recommendations(
            verified=[
                VerifiedCandidate("c1", -1.0, True),
                VerifiedCandidate("c2", -0.5, True),
                VerifiedCandidate("c3", -0.1, False),
            ],
            beam_fallback=[("c2", -0.2), ("w1", -0.3), ("w2", -0.4)],
            catalog=catalog,
            k=4,
        )
        self.assertEqual([row[0] for row in output], ["c2", "c1", "w1", "w2"])
        self.assertEqual([row[2] for row in output], [
            "accepted_draft", "accepted_draft", "gram_beam_fallback", "gram_beam_fallback"
        ])

    def test_collision_unknown_and_budget_violations_hard_fail(self):
        with self.assertRaisesRegex(ValueError, "collisions"):
            PathCatalog.build(
                {"w": "|a|b", "c": "|a|b"}, {"w"}, {"c"}
            )

    def test_gpu_hook_primitives_mask_padding_and_use_target_aware_length(self):
        labels = padded_candidate_labels([(1, 2, 3), (2,)], device=torch.device("cpu"))
        logits = torch.zeros(2, 3, 4)
        logits[0, 0, 1] = 3.0
        logits[0, 1, 2] = 2.0
        logits[0, 2, 3] = -10.0
        logits[1, 0, 2] = 4.0
        token_logp, mask = candidate_token_log_probabilities(logits, labels)
        scores = target_aware_scores_tensor(token_logp, mask, [2, 1])
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertAlmostEqual(
            float(scores[0]),
            float((token_logp[0, 0] + token_logp[0, 1]) / 2),
        )
        self.assertAlmostEqual(float(scores[1]), float(token_logp[1, 0]))
        self.assertEqual(float(token_logp[1, 1]), 0.0)

    def test_gpu_hook_primitives_reject_invalid_score_length(self):
        labels = padded_candidate_labels([(1,)], device=torch.device("cpu"))
        token_logp, mask = candidate_token_log_probabilities(
            torch.zeros(1, 1, 3), labels
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            target_aware_scores_tensor(token_logp, mask, [2])
        with self.assertRaisesRegex(ValueError, "unknown"):
            guided_redraft(
                ["missing"],
                catalog=fixture_catalog(),
                verifier_prefixes=[],
                prefix_depth=0,
                already_drafted=[],
                draft_size=1,
            )
        with self.assertRaisesRegex(ValueError, "re-verified"):
            validate_specgr_budget_trace(
                drafted_by_round=[["c1"], ["c1"]],
                draft_size=1,
                max_path_depth=4,
                verifier_forward_candidates=2,
            )


if __name__ == "__main__":
    unittest.main()
