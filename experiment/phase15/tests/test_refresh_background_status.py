from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from refresh_background_status import _last_progress, _pid_alive  # noqa: E402


class TestRefreshBackgroundStatus(unittest.TestCase):
    def test_permission_error_means_pid_exists(self):
        with mock.patch("os.kill", side_effect=PermissionError):
            self.assertTrue(_pid_alive(123))

    def test_missing_pid_is_not_alive(self):
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            self.assertFalse(_pid_alive(123))

    def test_b3_full_validation_progress_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            log.write_text(
                "[s3b-eval] events=32/8789\n"
                "[s3b-b3-eval] events=48/8789\n",
                encoding="utf-8",
            )
            self.assertEqual(_last_progress(log, 8789), (48, 8789))


if __name__ == "__main__":
    unittest.main()
