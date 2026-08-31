from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.status_writer import AttemptLedger, StatusWriter, rebuild_phase_index


class StatusStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="s17-s1-status-")
        self.root = Path(self.temporary.name)
        self.writer = StatusWriter(self.root / "status", "s17_s3_a0_toys_d0_seed2023")
        self.writer.initialize(
            step_id="S17-3",
            attempt_id="attempt_001",
            canonical_result_dir="artifacts/phase17/s3/a0/attempt_001",
            log_path="artifacts/phase17/s3/a0/attempt_001/run.log",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_science_and_execution_are_independent(self) -> None:
        self.writer.transition("PREFLIGHT", "PREFLIGHT", "PREFLIGHT")
        self.writer.transition("RUNNING", "RUNNING_SCIENTIFIC", "TRAINING", process_alive=True)
        self.writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "SCIENTIFIC_COMPLETED",
            process_alive=False,
            result_selection_eligible=True,
            scientific_completed_at="2026-08-29T00:00:00+00:00",
        )
        payload = self.writer.start_runtime_cycle(
            iteration=2,
            runtime_result_dir="artifacts/phase17/runtime/s17_s3_a0_toys_d0_seed2023/run-0002",
        )
        self.assertEqual(payload["scientific_state"], "COMPLETED")
        self.assertEqual(payload["execution_state"], "RUNNING_OCCUPANCY_REPEAT")
        self.assertFalse(payload["result_selection_eligible"])
        self.assertTrue(payload["repeat_metrics_ignored"])
        self.assertFalse(payload["affects_scientific_result"])

    def test_runtime_cycle_before_success_fails(self) -> None:
        with self.assertRaises(ValueError):
            self.writer.start_runtime_cycle(
                iteration=2,
                runtime_result_dir="artifacts/phase17/runtime/s17_s3_a0_toys_d0_seed2023/run-0002",
            )

    def test_illegal_terminal_reopen_fails(self) -> None:
        self.writer.transition("PREFLIGHT", "PREFLIGHT", "PREFLIGHT")
        self.writer.transition("RUNNING", "RUNNING_SCIENTIFIC", "TRAINING")
        self.writer.transition("FAILED", "SCIENTIFIC_FAILED", "FAILED")
        with self.assertRaises(ValueError):
            self.writer.transition("RUNNING", "RUNNING_SCIENTIFIC", "REOPEN")

    def test_index_and_attempt_ledger_are_machine_readable(self) -> None:
        index = json.loads((self.root / "status/phase17.index.json").read_text())
        self.assertIn("s17_s3_a0_toys_d0_seed2023", index["experiments"])
        ledger = AttemptLedger(self.root / "attempts/S17-3.attempts.jsonl")
        record = {
            "attempt_id": "attempt_001",
            "step_id": "S17-3",
            "kind": "formal",
            "started_at": "2026-08-29T00:00:00+00:00",
            "state": "RUNNING",
            "scientific_result_eligible": True,
        }
        ledger.append(record)
        with self.assertRaises(ValueError):
            ledger.append(record)

    def test_heartbeat_preserves_scientific_and_execution_states(self) -> None:
        self.writer.transition("PREFLIGHT", "PREFLIGHT", "PREFLIGHT")
        payload = self.writer.heartbeat(stage="checking", progress={"current": 1, "total": 3})
        self.assertEqual(payload["scientific_state"], "PREFLIGHT")
        self.assertEqual(payload["execution_state"], "PREFLIGHT")
        self.assertIn("heartbeat_at", payload)

    def test_index_rebuild_includes_legacy_execution_state(self) -> None:
        legacy = self.root / "status/s17_s0_legacy.status.json"
        legacy.write_text(
            json.dumps(
                {
                    "experiment_id": "s17_s0_legacy",
                    "step_id": "S17-0",
                    "scientific_state": "COMPLETED",
                    "execution_state": "COMPLETED_WITH_POSTRUN_CONTROL_RECOVERY",
                    "status_code": "COMPLETE",
                    "updated_at": "2026-08-29T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        index = rebuild_phase_index(self.root / "status")
        self.assertIn("s17_s0_legacy", index["experiments"])
