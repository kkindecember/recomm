"""Scientific checks for masking, controls, target isolation and gradients."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.catalog_scope_core import ranks
from experiment.phase20.source_competition_core import (
    candidate_union, target_positions, pair_masks, ResidualRanker, row_losses, independent_ranks)


class SourceCompetitionTests(unittest.TestCase):
    def test_union_masks_history_and_duplicates_without_targets(self):
        g = np.array([[4, 2, 8, 1], [3, 2, 8, 0]])
        l = np.array([[2, 5, 7, 4], [2, 6, 5, 1]])
        ids, ing, inl = candidate_union(g, l, [[1, 5], [2]])
        self.assertEqual(set(ids[0][ids[0] > 0]), {2, 4, 7, 8})
        self.assertEqual(set(ids[1][ids[1] > 0]), {1, 3, 5, 6, 8})
        for row in ids:
            self.assertEqual(len(set(row[row > 0])), int((row > 0).sum()))
        before = ids.copy()
        np.testing.assert_array_equal(target_positions(ids, np.array([7, 99])) >= 0, [True, False])
        target_positions(ids, np.array([99, 6]))
        np.testing.assert_array_equal(before, candidate_union(g, l, [[1, 5], [2]])[0])
        self.assertFalse(((ing | inl) & (ids == 0)).any())

    def test_negative_matching_and_source_rule(self):
        p = np.array([[4., 3., 2., 1., 0.]] * 3)
        valid = np.array([[True]*4+[False]]*3)
        ing = np.array([[True, True, False, False, False], [True, True, False, False, False], [True]*5])
        pos = np.array([0, 2, 0])
        m = pair_masks(p, valid, ing, pos, 2)
        np.testing.assert_array_equal(m['cross_source'], [[0, 0, 1, 1, 0], [1, 1, 0, 0, 0], [0]*5])
        np.testing.assert_array_equal(m['hard_negative'].sum(1), [2, 2, 0])
        for v in m.values():
            self.assertFalse(v[np.arange(3), pos].any())
            self.assertFalse((v & ~valid).any())

    def test_gradient_promotes_target_and_suppresses_cross_source(self):
        s = torch.tensor([[0., 1., 2., 0.]], requires_grad=True)
        valid = torch.tensor([[True, True, True, False]])
        pos = torch.tensor([0]); pairs = torch.tensor([[False, False, True, False]])
        row_losses(s, valid, pos, pairs).sum().backward()
        self.assertLess(s.grad[0, 0], 0)
        self.assertGreater(s.grad[0, 2], 0)
        self.assertEqual(s.grad[0, 3], 0)
        torch.testing.assert_close(row_losses(s, valid, pos, torch.zeros_like(pairs)),
                                   F.cross_entropy(s.masked_fill(~valid, -1e9), pos, reduction='none'))

    def test_chunk_gradient_equivalence_and_zero_initialization(self):
        rng = np.random.RandomState(811)
        x = torch.tensor(rng.randn(7, 12, 10), dtype=torch.float32)
        p = torch.tensor(rng.randn(7, 12), dtype=torch.float32)
        valid = torch.ones_like(p, dtype=torch.bool); valid[:, -2:] = False
        pos = torch.tensor([0, 2, 3, 1, 4, 5, 7])
        pairs = torch.zeros_like(valid); pairs[:, 8:10] = True
        weights = torch.arange(1, 8, dtype=torch.float32); weights /= weights.sum()
        for extra in (None, pairs):
            gradients = []
            for chunk in (7, 3):
                torch.manual_seed(21); model = ResidualRanker()
                torch.testing.assert_close(model(x, p), p, atol=0, rtol=0)
                for i in range(0, len(x), chunk):
                    sl = slice(i, i+chunk)
                    loss = row_losses(model(x[sl], p[sl]), valid[sl], pos[sl], None if extra is None else extra[sl])
                    (loss * weights[sl]).sum().backward()
                gradients.append(torch.cat([t.grad.flatten() for t in model.parameters()]))
            torch.testing.assert_close(*gradients, atol=1e-7, rtol=1e-5)

    def test_ties_padding_and_ineligible_target(self):
        ids = np.array([[4, 2, 9, 0], [2, 3, 1, 0], [7, 4, 0, 0]])
        s = np.array([[1., 1., 0., -np.inf], [1., 0., 2., -np.inf], [0., -np.inf, -np.inf, -np.inf]])
        targets = np.array([4, 9, 4])
        expected = np.array([2, 2**30, 2**30])
        np.testing.assert_array_equal(ranks(s, ids, targets), expected)
        np.testing.assert_array_equal(independent_ranks(s, ids, targets), expected)


if __name__ == '__main__': unittest.main()
