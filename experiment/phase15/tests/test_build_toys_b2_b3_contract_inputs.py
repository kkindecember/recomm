from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from build_toys_b2_b3_contract_inputs import (  # noqa: E402
    choose_occurrence,
    collect_train_occurrences,
    deterministic_topk,
)


class TestBuildToysB2B3ContractInputs(unittest.TestCase):
    def test_occurrences_use_only_preceding_train_history(self):
        rows = {"u": ("a", "warm", "b", "warm")}
        occurrences = collect_train_occurrences(
            rows, eligible_items={"warm"}, max_history=2
        )
        self.assertEqual(
            occurrences["warm"],
            [("u", 1, ("a",)), ("u", 3, ("warm", "b"))],
        )

    def test_occurrence_choice_is_deterministic(self):
        rows = [("u1", 1, ("a",)), ("u2", 2, ("b",))]
        first = choose_occurrence(rows, cold_item="c", warm_item="w", seed=7)
        second = choose_occurrence(list(reversed(rows)), cold_item="c", warm_item="w", seed=7)
        self.assertEqual(first, second)

    def test_topk_has_item_id_tie_break(self):
        selected = deterministic_topk(
            torch.tensor([0.5, 0.7, 0.5]), ["z", "m", "a"], 2
        )
        self.assertEqual([row[0] for row in selected], [1, 2])


if __name__ == "__main__":
    unittest.main()
