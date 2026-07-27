#!/usr/bin/env python3

import unittest

from hbtr_pilot_split import history_bin, largest_remainder_allocation


class HbtrPilotSplitTests(unittest.TestCase):
    def test_history_bins_are_locked(self):
        self.assertEqual(history_bin(1), "1-5")
        self.assertEqual(history_bin(5), "1-5")
        self.assertEqual(history_bin(6), "6-10")
        self.assertEqual(history_bin(20), "11-20")
        self.assertEqual(history_bin(21), "21+")

    def test_allocation_is_exact_and_capacity_bounded(self):
        sizes = {"a": 3, "b": 7, "c": 10}
        allocation = largest_remainder_allocation(sizes, 9)
        self.assertEqual(sum(allocation.values()), 9)
        self.assertTrue(all(allocation[key] <= sizes[key] for key in sizes))

    def test_allocation_is_deterministic(self):
        sizes = {"b": 5, "a": 5}
        self.assertEqual(
            largest_remainder_allocation(sizes, 3),
            largest_remainder_allocation(sizes, 3),
        )


if __name__ == "__main__":
    unittest.main()
