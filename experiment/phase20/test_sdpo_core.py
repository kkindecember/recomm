import math
import unittest
import torch
from experiment.phase20.preference_core import preference_loss
from experiment.phase20.sdpo_core import softmax_preference_loss, choose_hard_negatives


class SoftmaxPreferenceTests(unittest.TestCase):
    def test_single_negative_reduces_to_dpo(self):
        p, n = torch.tensor([-2., -6.]), torch.tensor([-3., -2.])
        rp, rn = torch.tensor([-1., -5.]), torch.tensor([-4., -1.])
        a, _ = preference_loss(p, n, rp, rn, .1)
        b, _ = softmax_preference_loss(p, n[:, None], rp, rn[:, None], .1)
        torch.testing.assert_close(a, b)

    def test_identical_reference_log_five_and_detached_reference(self):
        p = torch.tensor([-3., -1.], requires_grad=True)
        n = torch.tensor([[-4., -2., -5., -7.], [-1., -2., -3., -4.]], requires_grad=True)
        rp, rn = p.detach().clone().requires_grad_(), n.detach().clone().requires_grad_()
        loss, margin = softmax_preference_loss(p, n, rp, rn, .1)
        torch.testing.assert_close(loss, torch.full((2,), math.log(5)))
        torch.testing.assert_close(margin, torch.zeros_like(n))
        loss.mean().backward()
        self.assertTrue((p.grad < 0).all())
        self.assertTrue((n.grad > 0).all())
        self.assertIsNone(rp.grad)
        self.assertIsNone(rn.grad)

    def test_permutation_and_hard_negative_gradient(self):
        p = torch.tensor([0.], requires_grad=True)
        n = torch.tensor([[3., -3.]], requires_grad=True)
        a, _ = softmax_preference_loss(p, n, torch.zeros_like(p), torch.zeros_like(n), 1.)
        b, _ = softmax_preference_loss(p, n.flip(1), torch.zeros_like(p), torch.zeros_like(n), 1.)
        torch.testing.assert_close(a, b)
        a.sum().backward()
        self.assertGreater(float(n.grad[0, 0]), float(n.grad[0, 1]))

    def test_stability_and_target_exclusion(self):
        loss, _ = softmax_preference_loss(torch.tensor([-10000.]), torch.tensor([[10000., 0.]]),
                                          torch.zeros(1), torch.zeros(1, 2), .1)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(choose_hard_negatives([9, 3, 7, 8, 1], 3, 4), [9, 7, 8, 1])
        with self.assertRaises(ValueError):
            choose_hard_negatives([3, 9], 3, 2)


if __name__ == '__main__':
    unittest.main()
