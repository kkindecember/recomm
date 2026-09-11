"""Mathematical/gradient regressions; no GPU or recommendation data required."""
import math
import unittest
import torch
from experiment.phase20.preference_core import path_labels, preference_loss, sequence_logps, sampling_prefix_function


class PreferenceTests(unittest.TestCase):
    def test_reference_identity_and_preference_gradient(self):
        chosen = torch.tensor([-2., -3.], requires_grad=True)
        rejected = torch.tensor([-4., -1.], requires_grad=True)
        reference_chosen = chosen.detach().clone().requires_grad_()
        reference_rejected = rejected.detach().clone().requires_grad_()
        loss, margin = preference_loss(chosen, rejected, reference_chosen, reference_rejected, .1)
        torch.testing.assert_close(loss, torch.full_like(loss, math.log(2)))
        torch.testing.assert_close(margin, torch.zeros_like(margin))
        loss.mean().backward()
        self.assertTrue((chosen.grad < 0).all())
        self.assertTrue((rejected.grad > 0).all())
        self.assertIsNone(reference_chosen.grad)
        self.assertIsNone(reference_rejected.grad)

    def test_eos_included_padding_ignored_and_sequence_sum(self):
        labels = torch.tensor([[2, 1, -100], [2, 3, 1]])
        logits = torch.zeros(2, 3, 5, requires_grad=True)
        result = sequence_logps(logits, labels)
        torch.testing.assert_close(result, torch.tensor([-2*math.log(5), -3*math.log(5)]))
        result.sum().backward()
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0.)

    def test_reference_offsets_cancel_and_extremes_finite(self):
        cp, rp = torch.tensor([1000., -1000.]), torch.tensor([-1000., 1000.])
        loss, _ = preference_loss(cp, rp, torch.zeros(2), torch.zeros(2), .1)
        self.assertTrue(torch.isfinite(loss).all())
        a, _ = preference_loss(cp, rp, torch.tensor([3., 2.]), torch.tensor([1., 4.]), .1)
        b, _ = preference_loss(cp, rp, torch.tensor([13., 12.]), torch.tensor([11., 14.]), .1)
        torch.testing.assert_close(a, b)

    def test_start_token_is_not_scored(self):
        labels = path_labels([[0, 4, 1], [0, 3, 2, 1]], [2, 1], 'cpu')
        self.assertEqual(labels.tolist(), [[3, 2, 1], [4, 1, -100]])

    def test_variable_length_sampling_terminal_row_has_finite_distribution(self):
        class Trie:
            def prefix_allowed_tokens_fn(self):
                return lambda batch_id, ids: [2, 3] if ids.tolist() == [0] else []
        allowed = sampling_prefix_function(Trie())
        self.assertEqual(allowed(0, torch.tensor([0])), [2, 3])
        self.assertEqual(allowed(0, torch.tensor([0, 2, 1, 0])), [0])
        logits = torch.full((5,), -float('inf'))
        logits[allowed(0, torch.tensor([0, 2, 1]))] = 0.
        self.assertTrue(torch.isfinite(logits.softmax(0)).all())
        with self.assertRaises(ValueError):
            allowed(0, torch.tensor([0, 4]))


if __name__ == '__main__':
    unittest.main()
