#!/usr/bin/env python3
"""Resume the v2-to-v3 migration using host-visible process identities.

The managed command sandbox can query the host GPU and tmux server but does not
expose host GPU PIDs under its own /proc.  The original v3 migration therefore
failed closed before stopping anything.  This one-shot helper validates the
same PID independently through the exact tmux pane and GPU4's nvidia-smi row,
then stops only that tmux session and launches the already-frozen v3 worker.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import tmux_session_exists
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_g1_runtime_guard_v3 as v3


ROOT = Path(__file__).resolve().parents[3]
STATUS_DIR_SUFFIX = Path("artifacts/phase17/status")
STOP_TIMEOUT_SECONDS = 30


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout


def tmux_pane_identity(session: str) -> dict[str, Any]:
    output = _run_text(
        [
            "tmux",
            "list-panes",
            "-t",
            session,
            "-F",
            "#{session_name}|#{pane_pid}|#{pane_current_command}|#{pane_dead}",
        ]
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError("expected exactly one v2 tmux pane")
    session_name, pane_pid, command, dead = rows[0]
    return {
        "session": session_name,
        "pid": int(pane_pid),
        "command": command,
        "dead": dead == "1",
    }


def gpu_uuid_by_index() -> dict[int, str]:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    mapping: dict[int, str] = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) >= 2:
            mapping[int(row[0].strip())] = row[1].strip()
    return mapping


def gpu_processes() -> list[dict[str, Any]]:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 4:
            continue
        rows.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "used_memory_mib": int(row[3].strip()),
            }
        )
    return rows


def validate_host_visible_v2(root: Path) -> dict[str, Any]:
    status = _read(
        root / STATUS_DIR_SUFFIX / f"{v3.V2_EXPERIMENT_ID}.status.json"
    )
    pid, session = v3._validate_v2_status(status)
    # Query the pane directly.  A concurrent tmux `has-session` probe can
    # occasionally report a transient false negative while the pane remains
    # addressable; list-panes also gives us the PID identity we actually need.
    pane = tmux_pane_identity(session)
    if pane != {
        "session": session,
        "pid": pid,
        "command": "python",
        "dead": False,
    }:
        raise RuntimeError(f"v2 tmux pane identity drifted: {pane!r}")
    gpu_uuids = gpu_uuid_by_index()
    target_uuid = gpu_uuids.get(v3.PHYSICAL_GPU)
    matches = [row for row in gpu_processes() if row["pid"] == pid]
    if len(matches) != 1:
        raise RuntimeError("v2 PID is not uniquely visible in nvidia-smi")
    gpu_process = matches[0]
    if gpu_process["gpu_uuid"] != target_uuid:
        raise RuntimeError("v2 PID is no longer on the frozen physical GPU")
    if "gram-repro/bin/python" not in gpu_process["process_name"]:
        raise RuntimeError("v2 GPU process executable identity drifted")
    return {
        "status": status,
        "tmux_pane": pane,
        "gpu_process": gpu_process,
        "target_gpu_index": v3.PHYSICAL_GPU,
        "target_gpu_uuid": target_uuid,
    }


def resume_migration(root: Path) -> int:
    root = root.resolve()
    v3_status = _read(root / STATUS_DIR_SUFFIX / f"{v3.EXPERIMENT_ID}.status.json")
    if v3_status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError("prepared v3 is not in PREFLIGHT")
    evidence = validate_host_visible_v2(root)
    status = evidence["status"]
    pid = int(evidence["tmux_pane"]["pid"])
    session = str(evidence["tmux_pane"]["session"])
    v2_result = root / Path(status["canonical_result_dir"])
    before_path = v2_result / "status_before_v3_migration.json"
    supersession_path = v2_result / "supersession_by_v3.json"
    if before_path.exists() or supersession_path.exists():
        raise FileExistsError("v2 migration evidence already exists")
    atomic_json(
        before_path,
        {
            **status,
            "host_visible_identity_evidence": {
                "tmux_pane": evidence["tmux_pane"],
                "gpu_process": evidence["gpu_process"],
                "target_gpu_uuid": evidence["target_gpu_uuid"],
            },
        },
    )
    subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        session_gone = not tmux_session_exists(session)
        pid_gone = all(row["pid"] != pid for row in gpu_processes())
        if session_gone and pid_gone:
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("v2 did not disappear from both tmux and nvidia-smi")
    writer = StatusWriter(root / STATUS_DIR_SUFFIX, v3.V2_EXPERIMENT_ID)
    stopped = writer.transition(
        "COMPLETED",
        "STOPPED",
        "S17_FP12_G1_GUARD_V2_SUPERSEDED_BY_FRESH_PROCESS_V3",
        process_alive=False,
        workload_pid=0,
        gpu_ids=[],
        tmux_session=None,
        occupancy_mode="none",
        superseded_by_experiment_id=v3.EXPERIMENT_ID,
        result_selection_eligible=False,
        repeat_metrics_ignored=True,
        affects_scientific_result=False,
    )
    atomic_json(
        supersession_path,
        {
            "schema_version": "phase17.g1_guard_supersession.v2",
            "stopped_at": utc_now(),
            "stopped_pid": pid,
            "stopped_tmux_session": session,
            "host_visible_identity_evidence": {
                "tmux_pane": evidence["tmux_pane"],
                "gpu_process": evidence["gpu_process"],
                "target_gpu_uuid": evidence["target_gpu_uuid"],
            },
            "superseded_by_experiment_id": v3.EXPERIMENT_ID,
            "v2_status_after_stop": stopped,
            "scientific_result_affected": False,
        },
    )
    return v3.launch(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "resume-migration"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "validate":
        print(
            json.dumps(
                validate_host_visible_v2(args.root.resolve()),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return resume_migration(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
