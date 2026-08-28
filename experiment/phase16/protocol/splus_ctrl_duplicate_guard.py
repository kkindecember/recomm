#!/usr/bin/env python3
"""Fail-closed guard for stopping only the redundant GPU5 CTRL arm.

The guard never touches the running S-PLUS arm.  It may signal the frozen a3
runner only after the S-PLUS arm has a complete PASS summary, a3 has spawned
its serial S-PLUS-CTRL child, and the isolated GPU7 CTRL attempt is healthy or
already complete.  SIGTERM is sent to the runner (not directly to the child)
so the runner's existing terminal trap remains responsible for child cleanup
and holder restoration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
STAGES = ("pretrain", "finetune")
SCIENTIFIC_FIELDS = (
    "seed",
    "domain",
    "inputs",
    "model",
    "formal_budget",
    "admission",
    "resource_evidence",
    "batching_adaptation",
    "compatibility_patch",
)


class GuardError(RuntimeError):
    """A fail-closed guard validation failure."""


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    start_ticks: int
    cmdline: str


@dataclass(frozen=True)
class Decision:
    state: str
    reason: str
    ready_to_signal: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def age_seconds(value: str, now: datetime) -> float:
    return max(0.0, (now.astimezone(timezone.utc) - parse_time(value)).total_seconds())


def scientific_core(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in SCIENTIFIC_FIELDS}


def validate_static_files(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for relative, expected in config["code_freeze"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise GuardError(f"SHA drift: {relative}")
    a3_path = ROOT / config["a3"]["resolved_config_path"]
    a4_path = ROOT / config["a4"]["resolved_config_path"]
    if sha256(a3_path) != config["a3"]["resolved_config_sha256"]:
        raise GuardError("a3 resolved-config SHA drift")
    if sha256(a4_path) != config["a4"]["resolved_config_sha256"]:
        raise GuardError("a4 resolved-config SHA drift")
    a3_config = read_json(a3_path)
    a4_config = read_json(a4_path)
    if a3_config.get("attempt_id") != config["a3"]["attempt_id"]:
        raise GuardError("a3 attempt identity drift")
    if a4_config.get("attempt_id") != config["a4"]["attempt_id"]:
        raise GuardError("a4 attempt identity drift")
    if scientific_core(a3_config) != scientific_core(a4_config):
        raise GuardError("a3/a4 frozen scientific core mismatch")
    return a3_config, a4_config


def validate_plus_summary(summary: dict[str, Any], config: dict[str, Any]) -> None:
    if summary.get("status") != "completed":
        raise GuardError("S-PLUS summary is not completed")
    if summary.get("arm") != "S-PLUS" or summary.get("verdict") != "PASS_S16_2_S_PLUS_FORMAL_EXECUTION":
        raise GuardError("S-PLUS formal arm verdict is not PASS")
    expected_steps = sum(config["formal_budget"][stage]["optimizer_steps"] for stage in STAGES)
    if summary.get("arm_optimizer_steps") != expected_steps or summary.get("expected_arm_optimizer_steps") != expected_steps:
        raise GuardError("S-PLUS optimizer-step contract failed")
    internal = summary.get("internal_dev_generation_admission") or {}
    if not internal.get("all_finite") or internal.get("events") != config["formal_budget"]["internal_dev_transitions"]:
        raise GuardError("S-PLUS internal-dev admission is incomplete or non-finite")
    pseudo = summary.get("pseudo_cold_full_catalog_admission") or {}
    if (
        not pseudo.get("all_finite")
        or pseudo.get("events") != config["formal_budget"]["pseudo_cold_events"]
        or pseudo.get("candidate_items") != config["formal_budget"]["full_catalog_items"]
    ):
        raise GuardError("S-PLUS pseudo-cold full-catalog admission failed")
    if (
        not summary.get("base_checkpoint_unchanged")
        or summary.get("base_checkpoint_sha256_before") != summary.get("base_checkpoint_sha256_after")
        or summary.get("test_read") is not False
        or summary.get("validation_used") is not False
    ):
        raise GuardError("S-PLUS checkpoint or sealed-data contract failed")
    if summary.get("peak_cuda_reserved_mib", float("inf")) > config["admission"]["maximum_eligible_peak_reserved_mib"]:
        raise GuardError("S-PLUS peak reserved memory exceeded the admission ceiling")


def validate_plus_checkpoints(output: Path) -> None:
    required = [
        output / "arms" / "S-PLUS" / "checkpoints" / stage / name
        for stage in STAGES
        for name in ("last_state.pt", "final_model.pt")
    ]
    missing = [str(path.relative_to(output)) for path in required if not path.is_file()]
    if missing:
        raise GuardError(f"S-PLUS recovery checkpoints missing: {missing}")


def validate_a4_completed(summary: dict[str, Any], arm_summary: dict[str, Any], config: dict[str, Any]) -> None:
    if summary.get("status") != "completed" or summary.get("verdict") != "PASS_S16_2_S_PLUS_CTRL_SPLIT_FORMAL_EXECUTION":
        raise GuardError("a4 split attempt is not completed PASS")
    if not summary.get("same_frozen_scientific_config_as_parent") or not summary.get("formal_training_completed"):
        raise GuardError("a4 split completion contract failed")
    if arm_summary.get("status") != "completed":
        raise GuardError("a4 CTRL arm summary is not completed")
    if arm_summary.get("arm") != "S-PLUS-CTRL" or arm_summary.get("verdict") != "PASS_S16_2_S_PLUS_CTRL_FORMAL_EXECUTION":
        raise GuardError("a4 CTRL arm verdict is not PASS")
    expected_steps = sum(config["formal_budget"][stage]["optimizer_steps"] for stage in STAGES)
    if arm_summary.get("arm_optimizer_steps") != expected_steps:
        raise GuardError("a4 CTRL optimizer-step contract failed")
    internal = arm_summary.get("internal_dev_generation_admission") or {}
    if not internal.get("all_finite") or internal.get("events") != config["formal_budget"]["internal_dev_transitions"]:
        raise GuardError("a4 internal-dev admission is incomplete or non-finite")
    if (
        arm_summary.get("pseudo_cold_full_catalog_admission") is not None
        or not arm_summary.get("base_checkpoint_unchanged")
        or arm_summary.get("test_read") is not False
        or arm_summary.get("validation_used") is not False
    ):
        raise GuardError("a4 CTRL sealed-data or checkpoint contract failed")


def inspect_process(pid: int, proc_root: Path = Path("/proc")) -> ProcessInfo | None:
    base = proc_root / str(pid)
    try:
        stat_text = (base / "stat").read_text(encoding="utf-8")
        close = stat_text.rfind(")")
        fields = stat_text[close + 2 :].split()
        ppid = int(fields[1])
        start_ticks = int(fields[19])
        cmdline = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None
    return ProcessInfo(pid=pid, ppid=ppid, start_ticks=start_ticks, cmdline=cmdline)


def require_frozen_process(info: ProcessInfo | None, spec: dict[str, Any], label: str) -> None:
    if info is None:
        raise GuardError(f"{label} process is not alive")
    if info.pid != spec["pid"] or info.start_ticks != spec["start_ticks"] or info.cmdline != spec["cmdline"]:
        raise GuardError(f"{label} PID/start-time/cmdline identity mismatch")


def require_child_process(info: ProcessInfo | None, parent_pid: int, fragments: list[str], label: str) -> None:
    if info is None:
        raise GuardError(f"{label} process is not alive")
    if info.ppid != parent_pid or not all(fragment in info.cmdline for fragment in fragments):
        raise GuardError(f"{label} parent/cmdline identity mismatch")


def a4_document_state(
    config: dict[str, Any],
    a4_config: dict[str, Any],
    status: dict[str, Any],
    overall_summary: dict[str, Any] | None,
    arm_summary: dict[str, Any] | None,
    now: datetime,
) -> Decision:
    if status.get("attempt_id") != config["a4"]["attempt_id"] or status.get("physical_gpu") != 7:
        return Decision("BLOCKED", "a4 status identity drift")
    if status.get("test_read") is not False or status.get("validation_used") is not False:
        return Decision("BLOCKED", "a4 sealed-data status failed")
    if overall_summary is not None:
        if arm_summary is None:
            return Decision("WAIT", "a4 overall summary exists but arm summary is not yet visible")
        try:
            validate_a4_completed(overall_summary, arm_summary, a4_config)
        except GuardError as error:
            return Decision("BLOCKED", str(error))
        return Decision("SAFE", "a4 CTRL is completed PASS")
    if status.get("status") in {"failed", "timeout", "blocked"}:
        return Decision("BLOCKED", f"a4 reached terminal non-PASS state {status.get('status_code')}")
    if (
        status.get("status") != "running"
        or status.get("status_code") != "RUNNING"
        or status.get("current_arm") != "S-PLUS-CTRL"
        or status.get("process_alive") is not True
        or int(status.get("progress_current", 0)) < config["requirements"]["minimum_a4_optimizer_steps"]
    ):
        return Decision("WAIT", "a4 is not presently a healthy progressing CTRL workload")
    if age_seconds(status["updated_at"], now) > config["requirements"]["maximum_status_age_seconds"]:
        return Decision("WAIT", "a4 heartbeat is stale")
    if age_seconds(status["last_progress_at"], now) > config["requirements"]["maximum_a4_progress_age_seconds"]:
        return Decision("WAIT", "a4 optimizer progress is stale")
    return Decision("SAFE", "a4 CTRL is running and fresh")


def document_decision(
    config: dict[str, Any],
    a3_config: dict[str, Any],
    a4_config: dict[str, Any],
    a3_status: dict[str, Any],
    plus_summary: dict[str, Any] | None,
    a3_ctrl_summary: dict[str, Any] | None,
    a4_status: dict[str, Any],
    a4_summary: dict[str, Any] | None,
    a4_arm_summary: dict[str, Any] | None,
    now: datetime,
) -> Decision:
    if a3_status.get("attempt_id") != config["a3"]["attempt_id"] or a3_status.get("physical_gpu") != 5:
        return Decision("BLOCKED", "a3 status identity drift")
    if a3_status.get("test_read") is not False or a3_status.get("validation_used") is not False:
        return Decision("BLOCKED", "a3 sealed-data status failed")
    if a3_status.get("status") in {"failed", "timeout", "blocked", "completed"}:
        return Decision("NO_ACTION", f"a3 is already terminal: {a3_status.get('status_code')}")
    if a3_status.get("status") != "running":
        return Decision("WAIT", "a3 has not reached a stable running state")
    if age_seconds(a3_status["updated_at"], now) > config["requirements"]["maximum_status_age_seconds"]:
        return Decision("WAIT", "a3 heartbeat is stale")
    if a3_status.get("current_arm") == "S-PLUS":
        return Decision("WAIT", "a3 S-PLUS is still running")
    if a3_status.get("current_arm") != "S-PLUS-CTRL":
        return Decision("WAIT", f"a3 is in transition state {a3_status.get('current_arm')}")
    if a3_ctrl_summary is not None:
        return Decision("NO_ACTION", "a3 duplicate CTRL already completed; no signal is useful")
    if plus_summary is None:
        return Decision("WAIT", "a3 entered CTRL before S-PLUS summary became visible")
    try:
        validate_plus_summary(plus_summary, a3_config)
    except GuardError as error:
        return Decision("BLOCKED", str(error))
    a4_state = a4_document_state(config, a4_config, a4_status, a4_summary, a4_arm_summary, now)
    if a4_state.state != "SAFE":
        return a4_state
    return Decision("READY", f"S-PLUS is PASS and duplicate CTRL is replaceable; {a4_state.reason}", True)


def paths(config: dict[str, Any]) -> dict[str, Path]:
    a3_output = ROOT / config["a3"]["output_dir"]
    a4_output = ROOT / config["a4"]["output_dir"]
    guard_output = ROOT / config["guard"]["output_dir"]
    return {
        "a3_output": a3_output,
        "a4_output": a4_output,
        "a3_status": a3_output / "status.json",
        "a3_plus_summary": a3_output / "arms" / "S-PLUS" / "summary.json",
        "a3_ctrl_summary": a3_output / "arms" / "S-PLUS-CTRL" / "summary.json",
        "a4_status": a4_output / "status.json",
        "a4_summary": a4_output / "summary.json",
        "a4_arm_summary": a4_output / "arms" / "S-PLUS-CTRL" / "summary.json",
        "guard_output": guard_output,
        "guard_status": guard_output / "status.json",
        "guard_events": guard_output / "events.jsonl",
        "guard_summary": guard_output / "summary.json",
    }


def live_decision(
    config: dict[str, Any],
    process_reader: Callable[[int], ProcessInfo | None] = inspect_process,
) -> tuple[Decision, dict[str, Any]]:
    a3_config, a4_config = validate_static_files(config)
    target_paths = paths(config)
    now = datetime.now(timezone.utc)
    a3_status = read_json(target_paths["a3_status"])
    a4_status = read_json(target_paths["a4_status"])
    decision = document_decision(
        config,
        a3_config,
        a4_config,
        a3_status,
        read_optional_json(target_paths["a3_plus_summary"]),
        read_optional_json(target_paths["a3_ctrl_summary"]),
        a4_status,
        read_optional_json(target_paths["a4_summary"]),
        read_optional_json(target_paths["a4_arm_summary"]),
        now,
    )
    if decision.state in {"BLOCKED", "NO_ACTION"}:
        return decision, {"a3_status": a3_status, "a4_status": a4_status}

    a3_runner = process_reader(int(a3_status["runner_pid"]))
    require_frozen_process(a3_runner, config["a3"]["runner"], "a3 runner")
    if a3_status.get("current_arm") == "S-PLUS":
        if int(a3_status.get("workload_pid", 0)) <= 0:
            return Decision("WAIT", "a3 is between S-PLUS completion and the next runner state"), {
                "a3_status": a3_status,
                "a4_status": a4_status,
            }
        splus_process = process_reader(int(a3_status["workload_pid"]))
        if splus_process is None:
            return Decision("WAIT", "a3 S-PLUS child exited; waiting for runner transition"), {
                "a3_status": a3_status,
                "a4_status": a4_status,
            }
        require_child_process(
            splus_process,
            a3_runner.pid,
            config["a3"]["splus_workload_cmdline_fragments"],
            "a3 S-PLUS workload",
        )
    elif decision.ready_to_signal:
        validate_plus_checkpoints(target_paths["a3_output"])
        ctrl_process = process_reader(int(a3_status.get("workload_pid", 0)))
        if ctrl_process is None:
            return Decision("WAIT", "a3 duplicate CTRL child is not yet stably visible"), {
                "a3_status": a3_status,
                "a4_status": a4_status,
            }
        require_child_process(
            ctrl_process,
            a3_runner.pid,
            config["a3"]["ctrl_workload_cmdline_fragments"],
            "a3 duplicate CTRL workload",
        )

    if (
        a4_status.get("status") == "running"
        and a4_status.get("process_alive") is True
        and int(a4_status.get("workload_pid", 0)) > 0
        and decision.state in {"WAIT", "READY"}
    ):
        a4_runner = process_reader(int(a4_status["runner_pid"]))
        a4_workload = process_reader(int(a4_status["workload_pid"]))
        if a4_runner is None or a4_workload is None:
            return Decision("WAIT", "a4 process transition is in progress; no signal is allowed"), {
                "a3_status": a3_status,
                "a4_status": a4_status,
            }
        require_frozen_process(a4_runner, config["a4"]["runner"], "a4 runner")
        require_frozen_process(a4_workload, config["a4"]["workload"], "a4 workload")
    return decision, {"a3_status": a3_status, "a4_status": a4_status}


def status_payload(config: dict[str, Any], state: str, reason: str, started_at: str, signal_sent: bool) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "guard_id": config["guard_id"],
        "status": state,
        "reason": reason,
        "started_at": started_at,
        "updated_at": utc_now(),
        "guard_pid": os.getpid(),
        "target_runner_pid": config["a3"]["runner"]["pid"],
        "signal": "SIGTERM_TO_A3_RUNNER_ONLY",
        "signal_sent": signal_sent,
        "automatic_retry": False,
        "test_read": False,
        "validation_used": False,
    }


def validate_holder_restored(config: dict[str, Any], process_reader: Callable[[int], ProcessInfo | None]) -> tuple[bool, str]:
    holder = config["holder"]
    status_path = ROOT / holder["state_root"] / "status.json"
    gpu_path = ROOT / holder["state_root"] / "gpu.txt"
    if not status_path.is_file() or not gpu_path.is_file():
        return False, "holder state files are missing"
    status = read_json(status_path)
    if status.get("state") != "running" or int(status.get("reserve_mib", -1)) != holder["reserve_mib"]:
        return False, "holder state/reserve contract is not restored"
    if gpu_path.read_text(encoding="utf-8").strip() != str(holder["physical_gpu"]):
        return False, "holder GPU identity drift"
    pid = int(status.get("pid", 0))
    info = process_reader(pid)
    if info is None or not all(fragment in info.cmdline for fragment in holder["cmdline_fragments"]):
        return False, "restored holder process identity mismatch"
    tmux_check = subprocess.run(
        ["tmux", "has-session", "-t", holder["session"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tmux_check.returncode != 0:
        return False, "restored holder tmux session is missing"
    return True, f"holder restored as PID {pid} with reserve_mib={holder['reserve_mib']}"


def run(config_path: Path, mode: str, armed: bool) -> int:
    config = read_json(config_path)
    target_paths = paths(config)
    started_at = utc_now()
    signal_sent = False
    if mode == "watch" and not armed:
        raise SystemExit("--armed is required for watch mode")
    if mode == "watch":
        if target_paths["guard_summary"].exists():
            raise SystemExit("Refusing to replay a guard with an existing terminal summary")
        target_paths["guard_output"].mkdir(parents=True, exist_ok=True)
        lock_path = target_paths["guard_output"] / "armed.lock"
        try:
            lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise SystemExit("Refusing to start a second armed duplicate-CTRL guard") from error
        os.write(lock_fd, f"pid={os.getpid()} started_at={started_at}\n".encode())
        os.close(lock_fd)
    deadline = time.monotonic() + config["guard"]["maximum_wait_seconds"]
    last_state = ""
    while True:
        try:
            decision, snapshot = live_decision(config)
        except (GuardError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            decision = Decision("BLOCKED", f"fail-closed validation error: {error}")
            snapshot = {}
        if decision.state != last_state or mode == "check":
            append_event(
                target_paths["guard_events"],
                {"at": utc_now(), "state": decision.state, "reason": decision.reason, "signal_sent": signal_sent},
            )
            last_state = decision.state
        atomic_json(
            target_paths["guard_status"],
            status_payload(config, decision.state, decision.reason, started_at, signal_sent),
        )
        if mode == "check":
            return 0 if decision.state not in {"BLOCKED"} else 3
        if decision.state == "READY":
            runner_pid = int(snapshot["a3_status"]["runner_pid"])
            # The immediately preceding live_decision validated PID, start time,
            # exact runner cmdline, child PPID, CTRL cmdline, summaries, and a4.
            os.kill(runner_pid, signal.SIGTERM)
            signal_sent = True
            append_event(
                target_paths["guard_events"],
                {"at": utc_now(), "state": "TERM_SENT", "runner_pid": runner_pid, "signal": "SIGTERM"},
            )
            atomic_json(
                target_paths["guard_status"],
                status_payload(config, "TERM_SENT", "Validated duplicate CTRL; SIGTERM sent to a3 runner", started_at, True),
            )
            restore_deadline = time.monotonic() + config["guard"]["holder_restore_timeout_seconds"]
            while time.monotonic() < restore_deadline:
                runner_alive = inspect_process(runner_pid) is not None
                a3_status = read_optional_json(target_paths["a3_status"])
                holder_ok, holder_reason = validate_holder_restored(config, inspect_process)
                if (
                    not runner_alive
                    and a3_status is not None
                    and a3_status.get("status_code") == "INTERRUPTED"
                    and a3_status.get("holder_restored") is True
                    and holder_ok
                ):
                    summary = {
                        **status_payload(config, "completed", holder_reason, started_at, True),
                        "verdict": "PASS_S16_2_DUPLICATE_CTRL_GUARD",
                        "a3_terminal_status": a3_status,
                        "parent_splus_summary_preserved": True,
                        "gpu7_ctrl_untouched": True,
                    }
                    atomic_json(target_paths["guard_summary"], summary)
                    atomic_json(target_paths["guard_status"], summary)
                    append_event(target_paths["guard_events"], {"at": utc_now(), "state": "COMPLETED", "reason": holder_reason})
                    return 0
                time.sleep(config["guard"]["post_signal_poll_seconds"])
            reason = "SIGTERM was sent, but a3 INTERRUPTED + holder-restored postconditions were not proven"
            atomic_json(target_paths["guard_status"], status_payload(config, "blocked", reason, started_at, True))
            append_event(target_paths["guard_events"], {"at": utc_now(), "state": "BLOCKED_MANUAL_ATTENTION", "reason": reason})
            return 17
        if decision.state in {"BLOCKED", "NO_ACTION"}:
            summary = {
                **status_payload(config, decision.state.lower(), decision.reason, started_at, signal_sent),
                "verdict": "NO_SIGNAL_SENT",
            }
            atomic_json(target_paths["guard_summary"], summary)
            return 3 if decision.state == "BLOCKED" else 0
        if time.monotonic() >= deadline:
            reason = "Guard wait timeout reached; no signal was sent"
            atomic_json(target_paths["guard_status"], status_payload(config, "timeout", reason, started_at, False))
            append_event(target_paths["guard_events"], {"at": utc_now(), "state": "TIMEOUT_NO_ACTION", "reason": reason})
            return 4
        time.sleep(config["guard"]["poll_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "watch"), default="check")
    parser.add_argument("--armed", action="store_true")
    args = parser.parse_args()
    return run(args.config.resolve(), args.mode, args.armed)


if __name__ == "__main__":
    raise SystemExit(main())
