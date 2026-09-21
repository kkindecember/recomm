"""Scientific-contract checks for the matched event-exposure experiment."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from experiment.phase20.cf_exposure_screen import (
    CF0B2ItemHead, cf_features, masked_ce, padded_histories, ranks,
    split_users, terminal_indices,
)


class ExposureContractTests(unittest.TestCase):
    def test_masked_targets_cannot_change_cf_loss_or_parameter_gradients(self):
        torch.manual_seed(4)
        model = CF0B2ItemHead(11, 20, 8, 1, 2, 0., .07)
        histories = torch.tensor([[1, 2], [4, 5], [8, 9]])
        keep = torch.tensor([True, False, True])
        gradients = []
        for targets in (torch.tensor([3, 6, 10]), torch.tensor([3, 11, 10])):
            model.zero_grad()
            loss = masked_ce(model.score(histories, histories.ne(0)), targets, keep)
            loss.backward()
            gradients.append([p.grad.clone() for p in model.parameters()])
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # With dropout disabled and no batch-coupled layers, masked full-batch
        # gradients equal training only the retained histories.
        model.zero_grad()
        loss = F.cross_entropy(model.score(histories[keep], histories[keep].ne(0)), torch.tensor([2, 9]))
        loss.backward()
        for a, p in zip(gradients[0], model.parameters()):
            torch.testing.assert_close(a, p.grad, rtol=2e-5, atol=2e-5)

    def test_seen_loss_is_original_full_catalog_cross_entropy(self):
        logits = torch.randn(8, 17, dtype=torch.float64, requires_grad=True)
        targets = torch.arange(1, 9)
        expected = F.cross_entropy(logits, targets - 1)
        actual = masked_ce(logits, targets, torch.ones(8, dtype=torch.bool))
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            masked_ce(logits, targets, torch.zeros(8, dtype=torch.bool))

    def test_terminal_selection_excludes_only_last_user_event(self):
        records = [dict(user_id='a', history=[1], target=2),
                   dict(user_id='b', history=[4], target=5),
                   dict(user_id='a', history=[1, 2], target=3)]
        terminal = terminal_indices(records)
        self.assertEqual(terminal, {'a': 2, 'b': 1})
        histories = padded_histories(records)
        self.assertEqual(histories.shape, (3, 20))
        self.assertEqual(histories[terminal['a'], :2].tolist(), [1, 2])
        self.assertNotIn(3, histories[terminal['a']])

    def test_fusion_partition_depends_only_on_user_identity(self):
        users = [f'user-{i}' for i in range(100)]
        a = split_users(users, 20260916)
        b = split_users(users[::-1], 20260916)
        np.testing.assert_array_equal(a, b[::-1])
        self.assertTrue(a.any() and (~a).any())
        duplicate = split_users(['same', 'same'], 20260916)
        self.assertEqual(duplicate[0], duplicate[1])

    def test_source_cf_cannot_change_candidate_order_or_target_alignment(self):
        rng = np.random.default_rng(1)
        g = np.tile(np.linspace(1., 0., 50), (3, 1))
        g[0, :2] = 1.  # Stable tie order is part of frozen baseline contract.
        data = dict(g=g, frequencies=np.full((3, 50), 8),
                    raw_positions=np.array([0, 10, -1]),
                    base_rank=np.tile(1. / np.arange(1, 51), (3, 1)))
        a = cf_features(data, rng.normal(size=(3, 50)))
        b = cf_features(data, rng.normal(size=(3, 50)))
        np.testing.assert_array_equal(a[2], b[2])
        np.testing.assert_array_equal(a[3], b[3])
        np.testing.assert_array_equal(a[1][:, :, [0, 2, 3, 4]], b[1][:, :, [0, 2, 3, 4]])
        self.assertEqual(int(ranks(a[0], a[2])[-1]), 51)
        self.assertEqual(int(a[3][0, 0]), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
