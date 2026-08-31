from __future__ import annotations

import unittest

from experiment.phase17.core.leakage_guard import assert_no_future_read


class NoFutureContextTests(unittest.TestCase):
    def test_prefix_positions_pass(self) -> None:
        assert_no_future_read([0, 1, 2], cutoff=3)

    def test_cutoff_and_later_positions_fail(self) -> None:
        with self.assertRaises(PermissionError):
            assert_no_future_read([0, 3, 4], cutoff=3)
