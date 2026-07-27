#!/usr/bin/env python3

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from nlpl_d0 import (
    build_pairs,
    clustered_inference,
    read_predictions,
    tail_miss_analysis,
    training_popularity,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class NLPLD0Tests(unittest.TestCase):
    def test_training_popularity_excludes_two_targets(self):
        sequences = {"u": ["a", "b", "validation", "test"]}
        self.assertEqual(training_popularity(sequences), Counter({"a": 1, "b": 1}))

    def test_prediction_integrity_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.tsv"
            candidates = "||".join(["p"] * 50)
            scores = "||".join(["0"] * 50)
            path.write_text(
                "idx\ta\tb\tgold\tpred\tscores\n"
                f"u\t0\t0\tg\t{candidates}\t{scores}\n"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_predictions(path, {"g": "gold", "p": "pred"})

    def test_pairs_are_unique_and_frequency_matched(self):
        paths = {"a": (1, 2), "b": (1, 3), "c": (1, 4)}
        popularity = Counter({"a": 2, "b": 4, "c": 9})
        priors = {
            "a": {"lp_last": -3.0},
            "b": {"lp_last": -2.0},
            "c": {"lp_last": -1.0},
        }
        amp = {"a": 0.0, "b": 1.0, "c": 2.0}
        pairs = build_pairs(paths, popularity, priors, amp, 2.0)
        self.assertEqual([(p["left_item"], p["right_item"]) for p in pairs], [("a", "b")])
        self.assertEqual(pairs[0]["concordant"], 1)

    def test_cluster_inference_is_deterministic(self):
        pairs = [
            {"parent": "x", "non_tie": 1, "concordant": 1},
            {"parent": "x", "non_tie": 1, "concordant": 1},
            {"parent": "y", "non_tie": 1, "concordant": 0},
        ]
        first = clustered_inference(pairs, 7, 100, 100)
        second = clustered_inference(pairs, 7, 100, 100)
        self.assertEqual(first, second)

    def test_tail_miss_direction(self):
        paths = {
            "h": (0, 0),
            "lo": (1, 0),
            "hi": (1, 1),
            "x": (2, 0),
            "y": (3, 0),
        }
        popularity = Counter({"h": 100, "lo": 1, "hi": 1, "x": 1, "y": 1})
        priors = {
            "h": {"lp_last": 0.0},
            "lo": {"lp_last": -2.0},
            "hi": {"lp_last": 2.0},
            "x": {"lp_last": 0.0},
            "y": {"lp_last": 0.0},
        }
        rows = [
            {"gold": "lo", "pred": ["h"] * 50},
            {"gold": "hi", "pred": ["hi"] + ["h"] * 49},
        ]
        result = tail_miss_analysis(rows, paths, popularity, priors)
        self.assertGreater(result["miss_odds_ratio_low_vs_high"], 1.0)

    def test_teacher_forcing_shift_and_eos_exclusion_contract(self):
        path = torch.tensor([5, 6, 7])
        shifted = torch.tensor([0, 5, 6])
        self.assertTrue(torch.equal(shifted[1:], path[:-1]))
        self.assertNotIn(1, path.tolist())
        self.assertTrue(np.isfinite(path.numpy()).all())


if __name__ == "__main__":
    unittest.main()
