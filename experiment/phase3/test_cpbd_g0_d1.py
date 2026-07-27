#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cpbd_g0_d1 import (  # noqa: E402
    build_serialization,
    count_fields,
    encode_exact,
    metadata_spans,
    popularity_counts,
)


class CPBDD1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            "t5-small", local_files_only=True
        )

    def test_serialization_preserves_components(self):
        link = "|a|b"
        cf = ["|c|d", "|e|f"]
        metadata = "title: red toy; description: a useful red toy"
        current, _ = build_serialization(link, cf, metadata, False)
        alternate, _ = build_serialization(link, cf, metadata, True)
        for value in (link, *cf, metadata):
            self.assertIn(value, current)
            self.assertIn(value, alternate)
        self.assertLess(current.index("similar items:"), current.index(metadata))
        self.assertGreater(alternate.index("similar items:"), alternate.index(metadata))

    def test_metadata_field_spans(self):
        text = "title: x; brand: y; unknown: z"
        spans = metadata_spans(text, 10)
        self.assertEqual(len(spans["title"]), 1)
        self.assertEqual(len(spans["brand"]), 1)
        self.assertEqual(len(spans["other_metadata"]), 1)

    def test_short_text_has_no_loss(self):
        text, spans = build_serialization(
            "|a|b", ["|c|d"], "title: short; description: tiny", False
        )
        encoded = encode_exact(self.tokenizer, text, 999, 128)
        counts = count_fields(encoded, spans, self.tokenizer)
        self.assertEqual(counts["metadata_total"][2], 0)

    def test_long_metadata_is_recoverable(self):
        metadata = "title: x; description: " + "informative token " * 300
        current_text, current_spans = build_serialization(
            "|a|b", ["|c|d"] * 10, metadata, False
        )
        alternate_text, alternate_spans = build_serialization(
            "|a|b", ["|c|d"] * 10, metadata, True
        )
        current = count_fields(
            encode_exact(self.tokenizer, current_text, 999, 128),
            current_spans,
            self.tokenizer,
        )
        alternate = count_fields(
            encode_exact(self.tokenizer, alternate_text, 999, 128),
            alternate_spans,
            self.tokenizer,
        )
        self.assertGreater(
            alternate["metadata_total"][1], current["metadata_total"][1]
        )
        self.assertLess(
            alternate["collaborative"][1], current["collaborative"][1]
        )

    def test_popularity_excludes_last_two(self):
        counts = popularity_counts({"u1": ["a", "b", "valid", "test"]})
        self.assertEqual(counts["a"], 1)
        self.assertEqual(counts["b"], 1)
        self.assertEqual(counts["valid"], 0)
        self.assertEqual(counts["test"], 0)


if __name__ == "__main__":
    unittest.main()
