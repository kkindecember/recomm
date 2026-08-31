from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment.phase17.core.report_contract import enforce_one_report


class ReportContractTests(unittest.TestCase):
    def test_terminal_step_requires_exactly_one_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-s1-report-") as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                enforce_one_report(root, "S17-1", "COMPLETED")
            report = root / "Stage17_S1_公共迁移框架与运行合约报告.md"
            report.write_text("ok\n", encoding="utf-8")
            self.assertEqual(enforce_one_report(root, "S17-1", "COMPLETED"), report)
            (root / "Stage17_S1_trial2.md").write_text("bad\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                enforce_one_report(root, "S17-1", "COMPLETED")

    def test_nonterminal_step_must_not_publish_final_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-s1-report-") as temporary:
            self.assertIsNone(enforce_one_report(Path(temporary), "S17-2", "RUNNING"))
