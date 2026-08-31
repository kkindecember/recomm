from __future__ import annotations

import random
import unittest

import numpy as np

from experiment.phase17.core.full_latte_contracts import (
    LatteBeamPath,
    LattePathCodec,
    resolve_rqkmeans_psid_conflicts,
)


class FullLatteContractTests(unittest.TestCase):
    def build_codec(self) -> LattePathCodec:
        return LattePathCodec(
            {"a": (0, 0), "b": (0, 1), "c": (1, 0)},
            codebook_sizes=(3, 3),
            n_latent_tokens=8,
        )

    def test_psid_reassigns_collisions_without_suffix(self) -> None:
        centroids = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]],
            ],
            dtype=np.float32,
        )
        resolved, summary = resolve_rqkmeans_psid_conflicts(
            {"a": (0, 0), "b": (0, 0), "c": (1, 1)},
            centroids,
            top_k_per_digit=3,
        )
        self.assertEqual(len(set(resolved.values())), 3)
        self.assertEqual(resolved["a"], (0, 0))
        self.assertNotEqual(resolved["b"], (0, 0))
        self.assertEqual(summary.collisions_before, 1)
        self.assertEqual(summary.collisions_after, 0)
        self.assertEqual(summary.collision_suffix_size, 0)

    def test_every_exposure_samples_a_latent_but_keeps_the_same_sid(self) -> None:
        codec = self.build_codec()
        rng = random.Random(2023)
        targets = [codec.sample_training_target("a", rng=rng) for _ in range(1000)]
        self.assertEqual({target[0] for target in targets}, set(codec.latent_tokens))
        self.assertEqual({target[1:-1] for target in targets}, {codec.semantic_by_item["a"]})

    def test_forest_exposes_all_roots_and_only_legal_item_paths(self) -> None:
        codec = self.build_codec()
        self.assertEqual(codec.legal_next[()], codec.latent_tokens)
        for latent in codec.latent_tokens:
            path = (latent,) + codec.semantic_by_item["b"] + (codec.eos_token,)
            codec.assert_legal_path(path)
            self.assertEqual(codec.decode_path(path), "b")

    def test_item_aggregation_collapses_latent_paths(self) -> None:
        codec = self.build_codec()
        a = codec.semantic_by_item["a"]
        b = codec.semantic_by_item["b"]
        paths = [
            LatteBeamPath((codec.latent_tokens[0],) + a, -0.8),
            LatteBeamPath((codec.latent_tokens[1],) + a, -0.8),
            LatteBeamPath((codec.latent_tokens[0],) + b, -0.4),
        ]
        max_rank = codec.aggregate_paths(paths, method="agg_max", top_k=2)
        sum_rank = codec.aggregate_paths(paths, method="agg_sum", top_k=2)
        self.assertEqual(max_rank[0].item_id, "b")
        self.assertEqual(sum_rank[0].item_id, "a")
        self.assertEqual(sum_rank[0].path_count, 2)


if __name__ == "__main__":
    unittest.main()
