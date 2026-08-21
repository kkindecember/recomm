from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from oracle_prefix_probe import (  # noqa: E402
    BeamPrefixTracker,
    LiveBeamObserver,
    build_trie_children,
    distribution_profile,
    tie_aware_midrank,
)


class TestBeamPrefixTracker(unittest.TestCase):
    def test_callback_matches_legal_children_without_recording_unscored_rows(self):
        children = build_trie_children([(10, 20), (10, 30)], eos_id=1)
        tracker = BeamPrefixTracker(children, (10, 20))
        self.assertEqual(tracker(0, torch.tensor([0])), [10])
        self.assertEqual(tracker(0, torch.tensor([0, 10])), [20, 30])
        self.assertEqual(tracker(0, torch.tensor([0, 10, 20])), [1])
        self.assertEqual(tracker.survived, [True, False, False])

    def test_score_observer_records_only_live_target_prefix(self):
        children = build_trie_children([(10, 20), (10, 30)], eos_id=1)
        tracker = BeamPrefixTracker(children, (10, 20))
        observer = LiveBeamObserver(tracker)
        scores = torch.tensor([[0.0, float("-inf")], [float("-inf"), -0.2]])
        returned = observer(torch.tensor([[0, 10, 20], [0, 10, 30]]), scores)
        self.assertIs(returned, scores)
        self.assertEqual(tracker.survived, [True, False, True])

    def test_score_observer_ignores_hf_inactive_sentinel_row(self):
        children = build_trie_children([(10, 20), (10, 30)], eos_id=1)
        tracker = BeamPrefixTracker(children, (10, 20))
        observer = LiveBeamObserver(tracker)
        observer(
            torch.tensor([[0, 10, 20], [0, 10, 30]]),
            torch.tensor([[-1e9, float("-inf")], [float("-inf"), -0.2]]),
        )
        self.assertEqual(tracker.survived, [True, False, False])

    def test_all_unknown_fillers_match_frozen_trie_empty_behavior(self):
        children = build_trie_children([(10, 20), (10, 30, 40)], eos_id=1)
        tracker = BeamPrefixTracker(children, (10, 20))
        self.assertEqual(tracker(0, torch.tensor([0, 10, 20, 0])), [])
        self.assertEqual(tracker(0, torch.tensor([0, 10, 20, 2])), [])
        self.assertEqual(tracker(0, torch.tensor([0, 10, 0])), [])
        self.assertEqual(tracker.empty_callback_count, 3)


class TestDistributionProjection(unittest.TestCase):
    def test_tie_aware_midrank(self):
        values = torch.tensor([[3.0, 2.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([2.0, 0.0])
        self.assertEqual(tie_aware_midrank(values, target).tolist(), [2.5, 2.5])

    def test_target_subtree_mass_and_rank(self):
        item_ids = ["a", "b", "c"]
        paths = {"a": ("x", "a"), "b": ("x", "b"), "c": ("y", "c")}
        groups = {
            1: ({("x",): 0, ("y",): 1}, torch.tensor([0, 0, 1])),
            2: ({("x", "a"): 0, ("x", "b"): 1, ("y", "c"): 2}, torch.tensor([0, 1, 2])),
        }
        result = distribution_profile(
            torch.tensor([[3.0, 2.0, 1.0]]),
            torch.tensor([0]),
            ["a"],
            paths,
            groups,
            temperature=1.0,
        )[0]
        self.assertEqual(result["item_rank"], 1.0)
        self.assertEqual(result["depth"][0]["target_prefix_rank"], 1.0)
        self.assertGreater(result["depth"][0]["target_prefix_mass"], 0.8)


if __name__ == "__main__":
    unittest.main()
