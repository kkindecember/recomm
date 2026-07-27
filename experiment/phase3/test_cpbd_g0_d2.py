#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cpbd_g0_d2 import metadata_first_passage  # noqa: E402


class CPBDD2Test(unittest.TestCase):
    def test_metadata_first_round_trip(self):
        current = (
            "item: |a|b; similar items: |c|d, |e|f; "
            "title: red toy; description: useful"
        )
        alternate, spans, components = metadata_first_passage(current)
        self.assertEqual(components["link"], "|a|b")
        self.assertEqual(components["cf_values"], ["|c|d", "|e|f"])
        self.assertEqual(
            alternate,
            "item: |a|b; title: red toy; description: useful; "
            "similar items: |c|d, |e|f",
        )
        self.assertIn("metadata_total", spans)

    def test_invalid_grammar_fails(self):
        with self.assertRaises(ValueError):
            metadata_first_passage("title only")

    def test_components_are_unchanged(self):
        current = (
            "item: |x|y; similar items: |a|b; "
            "title: item; brand: maker; description: text"
        )
        alternate, _, components = metadata_first_passage(current)
        for value in (
            components["link"],
            components["metadata"],
            *components["cf_values"],
        ):
            self.assertIn(value, current)
            self.assertIn(value, alternate)


if __name__ == "__main__":
    unittest.main()
