from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.phase17.protocol.s17_fp0_reconcile_audit import Binding, inspect_binding


class FP0ReconciliationTests(unittest.TestCase):
    def test_terminal_status_requires_exact_base_attempt_and_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "status.json"
            ledger_path = root / "ledger.jsonl"
            summary_path = root / "summary.json"
            summary_path.write_text('{"ok": true}\n', encoding="utf-8")
            import hashlib

            summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            status_path.write_text(
                json.dumps(
                    {
                        "attempt_id": "attempt_001",
                        "step_id": "S17-FP0-X",
                        "scientific_state": "COMPLETED",
                        "execution_state": "SCIENTIFIC_COMPLETED",
                        "status_code": "PASS_X",
                        "updated_at": "2026-08-31T00:00:00+00:00",
                        "summary_path": "summary.json",
                        "summary_sha256": summary_hash,
                    }
                ),
                encoding="utf-8",
            )
            ledger_path.write_text(
                json.dumps(
                    {
                        "attempt_id": "attempt_001",
                        "step_id": "S17-FP0-X",
                        "kind": "test",
                        "started_at": "2026-08-31T00:00:00+00:00",
                        "state": "RUNNING",
                        "scientific_result_eligible": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = inspect_binding(root, Binding("status.json", "ledger.jsonl"))
            self.assertFalse(row["closeout_present"])
            self.assertTrue(row["summary"]["matches_status"])

    def test_nonterminal_status_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "attempt_id": "attempt_001",
                        "scientific_state": "RUNNING",
                    }
                ),
                encoding="utf-8",
            )
            (root / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                inspect_binding(root, Binding("status.json", "ledger.jsonl"))


if __name__ == "__main__":
    unittest.main()
