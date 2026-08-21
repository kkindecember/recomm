from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from r2pd_pseudo_cold_screen import (  # noqa: E402
    build_filtered_item_inputs,
    clean_transitions,
    path_weight_normalizer,
    weighted_path_nll,
)


class TestScreenIsolation(unittest.TestCase):
    def test_clean_transition_rejects_forbidden_item(self):
        with self.assertRaisesRegex(RuntimeError, "Forbidden"):
            clean_transitions(
                [("u", ["warm", "pseudo", "warm2"])],
                {"pseudo"},
                20,
                1401,
                10,
            )

    def test_similar_item_edges_are_filtered(self):
        paths = {"a": ("x",), "b": ("y",), "p": ("z",)}
        texts = {"a": "A", "b": "B", "p": "P"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "similar.txt"
            path.write_text("a p b\nb a p\np a b\n")
            outputs, report = build_filtered_item_inputs(paths, texts, path, {"p"}, 2)
        self.assertNotIn("z", outputs["a"])
        self.assertGreater(report["forbidden_similar_edges_removed"], 0)


class TestWeightedPathObjective(unittest.TestCase):
    def test_normalizer_never_depends_on_vocabulary_size(self):
        value = path_weight_normalizer([0.8, 0.2], [(100, 200), (300,)], 0.5)
        self.assertGreater(value, 0)

    def test_higher_mass_path_has_larger_gradient_weight(self):
        labels = torch.tensor([[1, 2], [3, 4]])
        logits = torch.zeros((2, 2, 6), requires_grad=True)
        numerator, normalizer = weighted_path_nll(
            logits, labels, [0.8, 0.2], [(1,), (3,)], 1.0
        )
        (numerator / normalizer).backward()
        self.assertGreater(abs(float(logits.grad[0, 0, 1])), abs(float(logits.grad[1, 0, 3])))


if __name__ == "__main__":
    unittest.main()
