import unittest

import torch

from experiment.phase4.gacr_s0 import (
    BoundedResidualRanker,
    base_scores,
    finite_catalog_zscore,
    stable_ranking,
    target_free_union,
)


class GACRS0Tests(unittest.TestCase):
    def test_target_free_union_preserves_source_order(self):
        self.assertEqual(
            target_free_union(["a", "b"], ["b", "c"]),
            ["a", "b", "c"],
        )

    def test_generator_reciprocal_rank_base(self):
        result = base_scores(["a", "c", "b"], ["a", "b"])
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 0.0, 0.5])))

    def test_zero_initialized_residual_identity(self):
        model = BoundedResidualRanker(6, 16, 0.2)
        features = torch.randn(4, 6)
        base = torch.tensor([1.0, 0.5, 0.0, 0.0])
        residual = model(features)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertEqual(stable_ranking(base), stable_ranking(base + residual))

    def test_residual_is_bounded(self):
        model = BoundedResidualRanker(6, 16, 0.2)
        with torch.no_grad():
            model.network[-1].bias.fill_(100)
        self.assertLessEqual(float(model(torch.randn(8, 6)).abs().max()), 0.2 + 1e-7)

    def test_masked_catalog_logit_uses_finite_sentinel(self):
        result = finite_catalog_zscore(
            torch.tensor(float("-inf")), torch.tensor(2.0), torch.tensor(0.5)
        )
        self.assertEqual(float(result), -10.0)
        self.assertTrue(torch.isfinite(result))


if __name__ == "__main__":
    unittest.main()
