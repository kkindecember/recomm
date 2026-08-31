from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiment.phase17.core.metrics import require_result_selection_eligible
from experiment.phase17.core.run_manager import (
    assert_runtime_isolation,
    background_required,
    freeze_run_snapshot,
    isolated_runtime_dir,
    launch_background_tmux,
    verify_run_snapshot,
)
from experiment.phase17.core.status_writer import assert_neutral_name


class RuntimeIsolationTests(unittest.TestCase):
    def test_snapshot_does_not_drift_when_live_source_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-s1-snapshot-") as temporary:
            root = Path(temporary)
            source = root / "worker.py"
            source.write_text("print('v1')\n", encoding="utf-8")
            manifest = freeze_run_snapshot(
                root=root,
                experiment_id="s17_s3_a0_toys_d0_seed2023",
                attempt_id="attempt_001",
                command=["python", "worker.py"],
                source_paths=[source],
                config={"seed": 2023},
            )
            source.write_text("print('v2')\n", encoding="utf-8")
            verify_run_snapshot(root, manifest)
            payload = json.loads(manifest.read_text())
            frozen = root / payload["files"][0]["snapshot_path"]
            self.assertIn("v1", frozen.read_text())

    def test_runtime_tree_is_disjoint_and_neutrally_named(self) -> None:
        root = Path("/repo")
        canonical = root / "artifacts/phase17/s3/a0/attempt_001"
        runtime = isolated_runtime_dir(root, "s17_s3_a0_toys_d0_seed2023", 2)
        assert_runtime_isolation(canonical, runtime)
        self.assertTrue(str(runtime).endswith("run-0002"))
        with self.assertRaises(ValueError):
            assert_neutral_name("s17_s3_a0_repeat")

    def test_evaluator_rejects_noncanonical_runtime_results(self) -> None:
        with self.assertRaises(PermissionError):
            require_result_selection_eligible(
                {
                    "result_selection_eligible": False,
                    "affects_scientific_result": False,
                    "test_read": False,
                    "sports_read": False,
                }
            )
        require_result_selection_eligible(
            {
                "result_selection_eligible": True,
                "affects_scientific_result": True,
                "test_read": False,
                "sports_read": False,
            }
        )

    def test_over_ten_minutes_or_unknown_requires_background(self) -> None:
        self.assertFalse(background_required(600))
        self.assertTrue(background_required(601))
        self.assertTrue(background_required(None))

    @mock.patch("experiment.phase17.core.run_manager.shutil.which", return_value="/usr/bin/tmux")
    @mock.patch("experiment.phase17.core.run_manager.subprocess.run")
    def test_background_launcher_uses_neutral_tmux_name_without_retry(self, run, _which) -> None:
        run.side_effect = [mock.Mock(returncode=1), mock.Mock(returncode=0)]
        session = launch_background_tmux(
            experiment_id="s17_s3_a0_toys_d0_seed2023",
            argv=["python", "frozen_worker.py"],
            cwd=Path("/repo"),
        )
        self.assertEqual(session, "s17_s3_a0_toys_d0_seed2023")
        self.assertEqual(run.call_count, 2)
        launched = run.call_args_list[1].args[0]
        self.assertEqual(launched[:6], ["tmux", "new-session", "-d", "-s", session, "-c"])
