"""Exactly-one consolidated report gate for terminal Stage17 steps."""

from __future__ import annotations

import re
from pathlib import Path


TERMINAL = {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}


def reports_for_step(report_dir: Path, step_id: str) -> list[Path]:
    number = int(step_id.split("-")[1])
    pattern = re.compile(rf"^Stage17_S{number}(?:_|\.).*\.md$")
    return sorted(path for path in report_dir.glob("*.md") if pattern.match(path.name))


def enforce_one_report(report_dir: Path, step_id: str, state: str) -> Path | None:
    reports = reports_for_step(report_dir, step_id)
    if state in TERMINAL:
        if len(reports) != 1:
            raise ValueError(f"terminal {step_id} requires exactly one report, observed {len(reports)}")
        return reports[0]
    if reports:
        raise ValueError(f"non-terminal {step_id} must not publish a final step report")
    return None
