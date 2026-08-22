from __future__ import annotations

import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from common_adapter import TrainTransition  # noqa: E402
from toys_b3_edit_state_smoke import _make_sample, select_covariance_transitions  # noqa: E402


class TestToysB3EditStateSmoke(unittest.TestCase):
    def test_covariance_sample_is_deterministic_and_covers_longest_path(self):
        transitions = [
            TrainTransition(str(index), ("h",), f"i{index}") for index in range(8)
        ]
        lengths = {f"i{index}": 6 if index < 3 else 5 for index in range(8)}
        first = select_covariance_transitions(
            transitions,
            path_lengths=lengths,
            sample_size=5,
            long_path_minimum=2,
            seed=1502,
        )
        second = select_covariance_transitions(
            list(reversed(transitions)),
            path_lengths=lengths,
            sample_size=5,
            long_path_minimum=2,
            seed=1502,
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(sum(lengths[row.target] == 6 for row in first), 2)

    def test_covariance_sample_rejects_missing_longest_path_coverage(self):
        transitions = [TrainTransition("u", ("h",), "short")]
        with self.assertRaisesRegex(ValueError, "longest-path"):
            select_covariance_transitions(
                transitions,
                path_lengths={"short": 5, "long": 6},
                sample_size=1,
                long_path_minimum=1,
                seed=1502,
            )

    def test_collator_sample_matches_frozen_reverse_history_semantics(self):
        sample = _make_sample(
            context_items=("a", "b", "c"),
            target_item="d",
            user_id="u",
            item_to_lexical={"a": "A", "b": "B", "c": "C", "d": "D"},
            item_text={"a": "ta", "b": "tb", "c": "tc", "d": "td"},
            item_to_cfid={"a": 1, "b": 2, "c": 3, "d": 4},
            max_history=2,
            reverse_history=True,
            history_separator=" ; ",
        )
        self.assertEqual(sample["input"], ["What would user purchase after C ; B ?", "tc", "tb"])
        self.assertEqual(sample["history_item_ids"], [3, 2])
        self.assertEqual(sample["output"], "D")


if __name__ == "__main__":
    unittest.main()
