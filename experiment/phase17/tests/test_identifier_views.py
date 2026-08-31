from __future__ import annotations

import unittest

import torch

from experiment.phase17.core.identifier_views import (
    build_identifier_views,
    flatten_views,
    select_training_view,
)
from experiment.phase17.core.item_aggregation import aggregate_item_scores, ranked_items


class IdentifierViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identifiers = {
            "item-a": "|▁red|fox|▁toy",
            "item-b": "|▁blue|car|▁set",
        }

    def test_b0_provides_two_paths_and_alternates_training_targets(self) -> None:
        views = build_identifier_views(self.identifiers, "B0_mvi")
        self.assertTrue(all(len(paths) == 2 for paths in views.values()))
        self.assertNotEqual(
            select_training_view(views["item-a"], 0),
            select_training_view(views["item-a"], 1),
        )
        self.assertEqual(len(flatten_views(views)), 4)

    def test_b1_preserves_native_suffix_below_latent_root(self) -> None:
        views = build_identifier_views(self.identifiers, "B1_latte")
        for item_id, paths in views.items():
            self.assertEqual(paths[0], self.identifiers[item_id])
            self.assertTrue(paths[1].endswith(self.identifiers[item_id]))

    def test_item_logsumexp_collapses_duplicate_paths_before_ranking(self) -> None:
        scores = torch.tensor([-1.0, -1.2, -0.8])
        aggregated = aggregate_item_scores(
            ["item-a", "item-a", "item-b"], scores, method="logsumexp"
        )
        self.assertEqual(ranked_items(aggregated)[0], "item-a")
        self.assertEqual(len(aggregated), 2)


if __name__ == "__main__":
    unittest.main()
