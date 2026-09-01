from __future__ import annotations

import math
import random
import unittest
from pathlib import Path

from experiment.phase17.core.full_latte_gram_backend import (
    PrefixTree,
    aggregate_generated_paths,
    cpu_preflight_gram_arm,
    load_fullport_examples,
    load_gram_catalog,
    render_gram_example,
)


ROOT = Path(__file__).resolve().parents[3]


class FullLatteGramBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train, cls.internal_dev = load_fullport_examples(ROOT)

    def test_real_data_counts_and_semantic_passage_replacement(self) -> None:
        catalog = load_gram_catalog(ROOT, "G1_GRAM_PSID_FULL")
        self.assertEqual(len(catalog.ordered_items), 11924)
        self.assertEqual(len(self.train), 56421)
        self.assertEqual(len(self.internal_dev), 1283)
        row = render_gram_example(
            self.train[0],
            arm_id="G1_GRAM_PSID_FULL",
            catalog=catalog,
            rng=random.Random(2023),
        )
        self.assertTrue(row.output.startswith("<s17_sid0_"))
        self.assertIn("item: <s17_sid0_", row.input[1])
        self.assertIn("similar items: <s17_sid0_", row.input[1])
        self.assertNotIn(self.train[0].target, " ".join(row.input))

    def test_g2_samples_latent_per_exposure_not_per_item(self) -> None:
        catalog = load_gram_catalog(ROOT, "G2_GRAM_LATTE_FULL")
        rng = random.Random(2023)
        outputs = {
            render_gram_example(
                self.train[0],
                arm_id="G2_GRAM_LATTE_FULL",
                catalog=catalog,
                rng=rng,
            ).output
            for _ in range(1000)
        }
        self.assertEqual({value.split()[0] for value in outputs}, {
            f"<s17_latent_{index}>" for index in range(8)
        })
        self.assertEqual(len({" ".join(value.split()[1:]) for value in outputs}), 1)

    def test_prefix_tree_and_item_aggregation(self) -> None:
        item_paths = {
            "a": ((0, 1, 9), (0, 2, 9)),
            "b": ((0, 1, 8),),
        }
        trie = PrefixTree(path for paths in item_paths.values() for path in paths)
        self.assertEqual(trie.allowed(()), (0,))
        self.assertEqual(trie.allowed((0,)), (1, 2))
        ranked = aggregate_generated_paths(
            [(0, 1, 9), (0, 2, 9), (0, 1, 8)],
            [-0.7, -0.2, -0.3],
            item_paths=item_paths,
            method="agg_max",
            top_k=2,
        )
        self.assertEqual([row[0] for row in ranked], ["a", "b"])
        summed = aggregate_generated_paths(
            [(0, 1, 9), (0, 2, 9)],
            [-0.7, -0.2],
            item_paths=item_paths,
            method="agg_sum",
            top_k=1,
        )
        self.assertAlmostEqual(summed[0][1], math.log(math.exp(-0.7) + math.exp(-0.2)))

    def test_all_three_real_cpu_preflights_pass(self) -> None:
        results = {
            arm: cpu_preflight_gram_arm(ROOT, arm)
            for arm in (
                "G0_GRAM_B0_FRESH",
                "G1_GRAM_PSID_FULL",
                "G2_GRAM_LATTE_FULL",
            )
        }
        self.assertEqual(results["G0_GRAM_B0_FRESH"]["decoder_paths"], 11924)
        self.assertEqual(results["G1_GRAM_PSID_FULL"]["target_token_length"], 4)
        self.assertEqual(results["G2_GRAM_LATTE_FULL"]["decoder_paths"], 8 * 11924)
        for result in results.values():
            self.assertEqual(result["state"], "PASS_CPU_PREFLIGHT")
            self.assertEqual(result["passages_in_checked_batch"], 21)
            self.assertFalse(result["external_target_materialized"])
            self.assertFalse(result["effect_metrics_computed"])


if __name__ == "__main__":
    unittest.main()
