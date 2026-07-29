import unittest

from experiment.phase4.cpia_n1 import (
    filtered_lexical_tokens,
    find_subsequence,
    valid_span_count,
)


class DummyTokenizer:
    def encode(self, _text):
        return [1820, 4, 5, 9175, 1]


class CPIAN1Test(unittest.TestCase):
    def test_filtered_lexical_tokens(self):
        self.assertEqual(filtered_lexical_tokens(DummyTokenizer(), "x"), (4, 5))

    def test_find_subsequence(self):
        self.assertEqual(find_subsequence([1, 2, 3, 2, 3], (2, 3)), [1, 3])
        self.assertEqual(find_subsequence([1, 2], (2, 3)), [])

    def test_valid_span_count_counts_coarse_and_fine_spans(self):
        self.assertEqual(valid_span_count(True), 2)
        self.assertEqual(valid_span_count(False), 0)


if __name__ == "__main__":
    unittest.main()
