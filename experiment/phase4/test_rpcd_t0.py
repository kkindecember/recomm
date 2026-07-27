import math
import unittest

import torch

from experiment.phase4.rpcd_t0 import (
    SASRec,
    deduplicate,
    fuse,
    make_eval_tensor,
    metric,
    stable_fraction,
)


class RPCDT0Tests(unittest.TestCase):
    def test_hash_split_is_deterministic(self):
        self.assertEqual(stable_fraction("u1", "salt"), stable_fraction("u1", "salt"))
        self.assertNotEqual(stable_fraction("u1", "salt"), stable_fraction("u1", "other"))

    def test_metric(self):
        self.assertEqual(metric(["a", "b"], "x", 10), (0.0, 0.0))
        self.assertEqual(metric(["a", "b"], "a", 10), (1.0, 1.0))
        self.assertAlmostEqual(metric(["a", "b"], "b", 10)[1], 1 / math.log2(3))

    def test_deduplicate_stable(self):
        self.assertEqual(deduplicate(["a", "b", "a"]), ["a", "b"])

    def test_fusion_endpoints(self):
        gram = ["g1", "g2"]
        sasrec = ["s1", "s2"]
        self.assertEqual(fuse(gram, sasrec, 0.0)[:2], gram)
        self.assertEqual(fuse(gram, sasrec, 1.0)[:2], sasrec)

    def test_right_padded_causal_forward_is_finite(self):
        model = SASRec(
            item_count=20,
            hidden_size=8,
            max_length=5,
            num_blocks=2,
            num_heads=2,
            dropout=0.0,
        )
        sequence = make_eval_tensor(["a", "b"], {"a": 1, "b": 2}, 5)[None]
        encoded = model.encode(sequence)
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertEqual(sequence.tolist(), [[1, 2, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
