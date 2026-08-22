#!/usr/bin/env python3
"""Atomically refresh the user-facing status contract for a background run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


PROGRESS_RE = re.compile(r"\[s3(?:a|b(?:-b3)?)-eval\]\s+events=(\d+)/(\d+)")


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        # EPERM means the PID exists but belongs to a process namespace/user
        # that the status reader is not allowed to signal.
        return True
    except OSError:
        return False
    return True


def _last_progress(log_path: Path, default_total: int) -> tuple[int, int]:
    current, total = 0, default_total
    if not log_path.is_file():
        return current, total
    for match in PROGRESS_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        current, total = int(match.group(1)), int(match.group(2))
    return current, total


def refresh(
    status_path: Path,
    log_path: Path,
    default_total: int,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if summary_path is not None and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed" and isinstance(summary.get("verdict"), str):
            status["reason"] = summary["verdict"]
    current, total = _last_progress(log_path, default_total)
    workload_rc = status.get("workload_rc", -1)
    state = str(status.get("status", "unknown"))

    status["status_code"] = state.upper()
    status["exit_code"] = workload_rc if isinstance(workload_rc, int) else -1
    status["exit_code_pending"] = state in {"starting", "running"}
    status["test_read"] = bool(status.get("test_read", status.get("test_opened", False)))
    status["process_alive"] = _pid_alive(status.get("workload_pid"))
    status["progress_current"] = current
    status["progress_total"] = total
    status["progress_unit"] = "events"
    status["updated_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    tmp_path = status_path.with_name(f"{status_path.name}.refresh.{os.getpid()}")
    tmp_path.write_text(json.dumps(status, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp_path, status_path)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--total", required=True, type=int)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    refreshed = refresh(args.status, args.log, args.total, args.summary)
    print(json.dumps(refreshed, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
