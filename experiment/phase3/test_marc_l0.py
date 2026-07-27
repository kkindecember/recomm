#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from marc_l0 import (
    build_passage,
    js_divergence,
    local_distribution,
)


class DummyTrie:
    def __init__(self):
        self.children = {
            (0,): [2, 3],
            (0, 2): [1],
        }

    def get(self, prefix):
        return self.children.get(tuple(prefix), [])


class MarcL0Test(unittest.TestCase):
    def test_passage_matches_locked_grammar(self):
        item2lexid = {"a": "A", "b": "B", "c": "C"}
        item_text = {"a": "title: alpha", "b": "title: beta", "c": "title: gamma"}
        neighbors = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
        self.assertEqual(
            build_passage(
                "a", item2lexid, item_text, neighbors, 2, True
            ),
            "item: A; similar items: B, C; title: alpha",
        )
        self.assertEqual(
            build_passage(
                "a", item2lexid, item_text, neighbors, 0, True
            ),
            "item: A; title: alpha",
        )

    def test_trie_local_distribution_excludes_eos_row(self):
        logits = torch.zeros(2, 5)
        logits[0, 2] = 2.0
        logits[0, 3] = 1.0
        labels = torch.tensor([2, 1])
        rows, checked, valid = local_distribution(
            logits, labels, DummyTrie(), eos_token_id=1
        )
        self.assertEqual(checked, 2)
        self.assertEqual(valid, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["top1_token"], 2)
        self.assertAlmostEqual(
            rows[0]["gold_log_probability"],
            float(torch.log_softmax(torch.tensor([2.0, 1.0]), dim=0)[0]),
            places=6,
        )

    def test_js_identity_and_symmetry(self):
        first = np.asarray([0.2, 0.8])
        second = np.asarray([0.7, 0.3])
        self.assertAlmostEqual(js_divergence(first, first), 0.0, places=10)
        self.assertAlmostEqual(
            js_divergence(first, second),
            js_divergence(second, first),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
