"""Scientific invariants: target isolation, reference matching and score parity."""
import unittest

import numpy as np
import torch

from experiment.phase20.conditional_reference_core import (
    Correction, arm_features, background_scores, category_vectors, frozen_fusion,
    history_features, matched_neighbors, ranks, stable_nearest, visible_key)
from experiment.phase20.gacr_fusion_core import numpy_ranks, pairwise_losses, standardize_tensor


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_background_is_mean_logits_not_renormalized_vector(self):
        bank = np.asarray([[1., 0.], [0., 1.], [-1., 0.]])
        items = np.asarray([[0., 0.], [1., 0.], [0., 1.], [.6, .8]])
        neighbors = np.asarray([[0, 1], [1, 2]])
        candidates = np.asarray([[1, 3], [2, 1]])
        result = background_scores(neighbors, bank, items, candidates, 3., batch_size=1)
        expected = np.asarray([[np.mean([3 * bank[j] @ items[i] for j in ns]) for i in ids]
                               for ns, ids in zip(neighbors, candidates)])
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)

    def test_stable_boundary_and_self_exclusion(self):
        values = np.asarray([np.inf, 1., 1., 0., 1., 1.])
        np.testing.assert_array_equal(stable_nearest(values, 3), [3, 1, 2])

    def matching_fixture(self):
        users = ['u%02d' % i for i in range(24)]
        s = np.asarray([[np.cos(i), np.sin(i)] for i in range(24)], dtype=np.float32)
        n = np.asarray([[i % 3, 0., i % 2, i / 100] for i in range(24)])
        bins = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
        q = (s[[0, 15, 4]], n[[0, 15, 4]], bins[[0, 15, 4]])
        return users, (s, n, bins), [users[i] for i in [0, 15, 4]], q

    def test_same_blocks_semantic_dominance_and_no_self(self):
        users, bank, query_users, query = self.matching_fixture()
        c, d, diag = matched_neighbors(query_users, ['a', 'b', 'c'], query, users, bank,
                                       neighbors=2, block_width=3, batch_size=1)
        self.assertTrue(np.all(diag[:, 0] >= diag[:, 1] - 1e-7))
        for i, u in enumerate(query_users):
            self.assertTrue(all(users[j] != u for j in np.r_[c[i], d[i]]))
            self.assertTrue(np.all(bank[2][c[i]] == query[2][i]))
            self.assertTrue(np.all(bank[2][d[i]] == query[2][i]))
        cp, dp, _ = matched_neighbors(query_users[::-1], ['c', 'b', 'a'], tuple(x[::-1] for x in query),
                                       users, bank, neighbors=2, block_width=3, batch_size=5)
        np.testing.assert_array_equal(c, cp[::-1])
        np.testing.assert_array_equal(d, dp[::-1])

    def test_simple_reference_does_not_use_semantics(self):
        users, bank, query_users, query = self.matching_fixture()
        args = dict(neighbors=2, block_width=3)
        c, d, _ = matched_neighbors(query_users, ['a', 'b', 'c'], query, users, bank, **args)
        q2, b2 = (query[0] * -1, *query[1:]), (bank[0], *bank[1:])
        c2, d2, _ = matched_neighbors(query_users, ['a', 'b', 'c'], q2, users, b2, **args)
        np.testing.assert_array_equal(d, d2)
        self.assertTrue(np.any(c != c2))

    def test_key_and_features_use_visible_history_only(self):
        first = {'user_id': 'u', 'history': [1, 2], 'target': 3, 'sample_key': 'u:2:3'}
        changed = dict(first, target=9, sample_key='u:2:9')
        self.assertEqual(visible_key(first['user_id'], first['history']),
                         visible_key(changed['user_id'], changed['history']))
        items, _ = category_vectors([[0, 0], [1, 2], [1, 3], [1, 2]], 4)
        a = history_features([first['history']], items, np.array([0, 3, 5, 2]), 3)
        b = history_features([changed['history']], items, np.array([0, 3, 5, 2]), 3)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)

    def test_ubiquitous_category_removed_and_zero_profiles_finite(self):
        vectors, weights = category_vectors([[0, 0], [1, 2], [1, 3], [1, 0]], 4)
        self.assertEqual(weights[1], 0)
        self.assertTrue(np.all(vectors[3] == 0))
        self.assertTrue(np.isfinite(vectors).all())

    def test_frozen_pcrf_exact_rank_parity_with_ties_and_missing_targets(self):
        generator = np.random.default_rng(14)
        g, cf = generator.normal(size=(8, 50)), generator.normal(size=(8, 50))
        g[0] = 0
        g[1, :5] = 2.
        cf[0] = 1
        freq = generator.integers(1, 25, size=(8, 50))
        positions = np.asarray([-1, 0, 49, 10, 3, 7, 20, -1])
        base = np.broadcast_to(1. / np.arange(1, 51), g.shape)
        _, expected, reliability = numpy_ranks(g, cf, freq, positions, 5)
        scores, common, p, order, got_rel = frozen_fusion(g, cf, freq, positions, base, 5)
        np.testing.assert_array_equal(ranks(scores, p), expected)
        np.testing.assert_array_equal(got_rel, reliability)
        x = arm_features(common, cf, order)
        self.assertEqual(x.shape, (8, 50, 6))
        self.assertTrue(np.isfinite(x).all())

    def test_zero_initialization_and_chunk_gradient_agreement(self):
        torch.manual_seed(14)
        model = Correction(.5)
        x = torch.randn(9, 50, 6)
        base = torch.randn(9, 50, dtype=torch.float64)
        positions = torch.arange(9)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 129)
        self.assertEqual(torch.count_nonzero(model(x)).item(), 0)
        loss = pairwise_losses(standardize_tensor(base + model(x).double()), positions).mean()
        loss.backward()
        full = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        for start in range(0, 9, 4):
            score = base[start:start+4] + model(x[start:start+4]).double()
            pairwise_losses(standardize_tensor(score), positions[start:start+4]).sum().div(9).backward()
        for expected, parameter in zip(full, model.parameters()):
            self.assertTrue(torch.allclose(expected, parameter.grad, atol=1e-7, rtol=1e-5))
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        opt.step()
        self.assertGreater(torch.abs(model(x)).sum().item(), 0)
        self.assertLessEqual(torch.abs(model(x)).max().item(), .5)


if __name__ == '__main__':
    unittest.main()
