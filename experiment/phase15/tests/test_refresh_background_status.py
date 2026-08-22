from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from refresh_background_status import _pid_alive  # noqa: E402


class TestRefreshBackgroundStatus(unittest.TestCase):
    def test_permission_error_means_pid_exists(self):
        with mock.patch("os.kill", side_effect=PermissionError):
            self.assertTrue(_pid_alive(123))

    def test_missing_pid_is_not_alive(self):
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            self.assertFalse(_pid_alive(123))


if __name__ == "__main__":
    unittest.main()
