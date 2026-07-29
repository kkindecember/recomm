import unittest

import torch

from experiment.phase4.chpr_a0 import (
    earliest_divergence,
    pad_labels,
    select_unique_user_samples,
)


class CHPRA0Tests(unittest.TestCase):
    def test_earliest_divergence(self):
        self.assertEqual(earliest_divergence([0, 4, 5, 1], [0, 4, 7, 1]), 2)

    def test_pad_labels_removes_decoder_start(self):
        labels = pad_labels([[0, 4, 1], [0, 7, 8, 1]], torch.device("cpu"))
        self.assertTrue(
            torch.equal(
                labels,
                torch.tensor([[4, 1, -100], [7, 8, 1]]),
            )
        )

    def test_unique_user_stratified_selection(self):
        rows = [
            {"sample_key": "a1", "user_id": "a", "positive_item": "h"},
            {"sample_key": "a2", "user_id": "a", "positive_item": "h"},
            {"sample_key": "b1", "user_id": "b", "positive_item": "h"},
            {"sample_key": "c1", "user_id": "c", "positive_item": "t"},
            {"sample_key": "d1", "user_id": "d", "positive_item": "t"},
        ]
        selected = select_unique_user_samples(rows, {"h"}, 1, "Toys", 2, 2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["user_id"] for row in selected}), 4)


if __name__ == "__main__":
    unittest.main()
