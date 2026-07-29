import unittest

import torch

from experiment.phase5.cet_c1 import (
    compose_loss,
    legal_child_kl,
    structured_passage_mask,
)


class FakeTrie:
    def get(self, prefix):
        return {
            (0,): [2, 3],
            (0, 2): [1],
        }[tuple(prefix)]


class CETC1Tests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {
                "user_id": "u1",
                "positive_item": "gold-a",
                "history_items": ["a", "b", "c"],
            },
            {
                "user_id": "u2",
                "positive_item": "gold-b",
                "history_items": ["d", "e", "f"],
            },
        ]
        self.attention = torch.ones(2, 4, 3, dtype=torch.bool)

    def test_structured_mask_keeps_coarse_and_newest(self):
        masked, decisions = structured_passage_mask(
            self.attention, self.samples, "Toys", 2023, 1.0
        )
        self.assertTrue(torch.equal(masked[:, 0], self.attention[:, 0]))
        self.assertTrue(torch.equal(masked[:, 1], self.attention[:, 1]))
        self.assertFalse(bool(masked[:, 2:].any()))
        self.assertTrue(bool(decisions[:, 2:].all()))

    def test_mask_is_target_independent_and_deterministic(self):
        first, first_decisions = structured_passage_mask(
            self.attention, self.samples, "Beauty", 2023, 0.25
        )
        altered = [dict(row, positive_item="changed") for row in self.samples]
        second, second_decisions = structured_passage_mask(
            self.attention, altered, "Beauty", 2023, 0.25
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_decisions, second_decisions))

    def test_legal_child_kl_zero_for_identical_logits(self):
        logits = torch.zeros(1, 2, 5, requires_grad=True)
        loss, count = legal_child_kl(
            logits, logits, [[0, 2, 1]], FakeTrie(), 1, 1.0
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_legal_child_kl_has_perturbed_gradient_only(self):
        clean = torch.zeros(1, 2, 5, requires_grad=True)
        perturbed = torch.zeros(1, 2, 5, requires_grad=True)
        perturbed.data[0, 0, 3] = 2.0
        loss, _ = legal_child_kl(
            clean, perturbed, [[0, 2, 1]], FakeTrie(), 1, 1.0
        )
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertGreater(float(perturbed.grad.abs().sum()), 0.0)

    def test_zero_alpha_beta_is_clean_ce(self):
        clean = torch.tensor(2.0)
        total = compose_loss(
            clean, torch.tensor(3.0), torch.tensor(4.0), 0.0, 0.0
        )
        self.assertEqual(float(total), float(clean))


if __name__ == "__main__":
    unittest.main()
