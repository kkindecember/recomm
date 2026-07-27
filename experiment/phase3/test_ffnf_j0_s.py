#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ffnf_j0_s import build_cf_stream, build_metadata_stream


class FFNFJ0STest(unittest.TestCase):
    def test_cf_stream_has_exact_components(self):
        text, spans = build_cf_stream("L0", ["C1", "C2"])
        self.assertEqual(text, "item: L0; similar items: C1, C2")
        self.assertEqual(text[slice(*spans["link"][0])], "L0")
        self.assertEqual(text[slice(*spans["collaborative"][0])], "C1, C2")

    def test_metadata_stream_has_no_added_link(self):
        metadata = "title: A; brand: B; categories: C"
        text, spans = build_metadata_stream(metadata)
        self.assertEqual(text, metadata)
        self.assertEqual(spans["metadata_total"], [(0, len(metadata))])
        self.assertEqual(len(spans["title"]), 1)
        self.assertEqual(len(spans["brand"]), 1)
        self.assertEqual(len(spans["categories"]), 1)


if __name__ == "__main__":
    unittest.main()
