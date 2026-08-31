"""Atomic scientific/execution status machine and append-only attempt ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCIENTIFIC_STATES = {"PENDING", "PREFLIGHT", "RUNNING", "COMPLETED", "FAILED", "STOPPED", "BLOCKED"}
SCIENTIFIC_TRANSITIONS = {
    "PENDING": {"PREFLIGHT", "BLOCKED", "STOPPED"},
    "PREFLIGHT": {"RUNNING", "FAILED", "BLOCKED", "STOPPED"},
    "RUNNING": {"COMPLETED", "FAILED", "BLOCKED", "STOPPED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "STOPPED": set(),
    "BLOCKED": set(),
}
EXECUTION_STATES = {
    "PENDING",
    "PREFLIGHT",
    "WAITING_FOR_GPU",
    "BACKGROUND_STARTED",
    "RUNNING_SCIENTIFIC",
    "SCIENTIFIC_COMPLETED",
    "SCIENTIFIC_FAILED",
    "STOPPED",
    "BLOCKED",
    "RUNNING_OCCUPANCY_REPEAT",
}
TERMINAL_SCIENCE = {"COMPLETED", "FAILED", "STOPPED", "BLOCKED"}
FORBIDDEN_RUNTIME_NAME_TOKENS = {"occupancy", "repeat", "holder", "占卡", "重复"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def assert_neutral_name(value: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in FORBIDDEN_RUNTIME_NAME_TOKENS):
        raise ValueError(f"runtime-maintenance state leaked into a public name: {value}")


@dataclass(frozen=True)
class StatusPaths:
    status_dir: Path

    def status(self, experiment_id: str) -> Path:
        return self.status_dir / f"{experiment_id}.status.json"

    @property
    def index(self) -> Path:
        return self.status_dir / "phase17.index.json"


def rebuild_phase_index(status_dir: Path) -> dict[str, Any]:
    """Rebuild the aggregate index, including legacy Stage17 status variants."""
    experiments: dict[str, dict[str, Any]] = {}
    for path in sorted(status_dir.glob("s17_*.status.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment_id = payload["experiment_id"]
        experiments[experiment_id] = {
            "step_id": payload["step_id"],
            "scientific_state": payload["scientific_state"],
            "execution_state": payload["execution_state"],
            "status_code": payload["status_code"],
            "updated_at": payload["updated_at"],
            "status_path": str(path),
        }
    index = {
        "schema_version": "phase17.index.v1",
        "updated_at": utc_now(),
        "experiments": experiments,
    }
    atomic_json(status_dir / "phase17.index.json", index)
    return index


class StatusWriter:
    def __init__(self, status_dir: Path, experiment_id: str) -> None:
        assert_neutral_name(experiment_id)
        self.experiment_id = experiment_id
        self.paths = StatusPaths(status_dir)

    def initialize(
        self,
        *,
        step_id: str,
        attempt_id: str,
        canonical_result_dir: str,
        log_path: str | None,
        track_id: str | None = None,
        started_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for name in (canonical_result_dir, log_path or ""):
            assert_neutral_name(name)
        payload: dict[str, Any] = {
            "schema_version": "phase17.status.v2",
            "experiment_id": self.experiment_id,
            "attempt_id": attempt_id,
            "step_id": step_id,
            "track_id": track_id,
            "scientific_state": "PENDING",
            "execution_state": "PENDING",
            "status_code": "PENDING",
            "started_at": started_at or utc_now(),
            "updated_at": utc_now(),
            "launcher_pid": os.getpid(),
            "workload_pid": 0,
            "process_alive": False,
            "tmux_session": None,
            "gpu_ids": [],
            "gpu_snapshot": {},
            "stage": "pending",
            "progress": {"current": 0, "total": 1},
            "canonical_result_dir": canonical_result_dir,
            "log_path": log_path,
            "test_read": False,
            "sports_read": False,
            "result_selection_eligible": False,
            "occupancy_mode": "none",
            "repeat_iteration": 0,
            "repeat_metrics_ignored": False,
            "affects_scientific_result": True,
        }
        payload.update(extra or {})
        self._validate(payload)
        self._write(payload)
        return payload

    def read(self) -> dict[str, Any]:
        return json.loads(self.paths.status(self.experiment_id).read_text(encoding="utf-8"))

    def transition(
        self,
        scientific_state: str,
        execution_state: str,
        status_code: str,
        **updates: Any,
    ) -> dict[str, Any]:
        payload = self.read()
        previous = payload["scientific_state"]
        if scientific_state != previous and scientific_state not in SCIENTIFIC_TRANSITIONS[previous]:
            raise ValueError(f"illegal scientific transition {previous} -> {scientific_state}")
        payload.update(updates)
        payload.update(
            scientific_state=scientific_state,
            execution_state=execution_state,
            status_code=status_code,
            updated_at=utc_now(),
        )
        self._validate(payload)
        self._write(payload)
        return payload

    def start_runtime_cycle(
        self, *, iteration: int, runtime_result_dir: str, workload_pid: int = 0
    ) -> dict[str, Any]:
        assert_neutral_name(runtime_result_dir)
        if not runtime_result_dir.endswith(f"run-{iteration:04d}"):
            raise ValueError("runtime output must use a neutral run-NNNN directory")
        payload = self.read()
        if payload["scientific_state"] != "COMPLETED":
            raise ValueError("runtime cycle is allowed only after canonical scientific completion")
        return self.transition(
            "COMPLETED",
            "RUNNING_OCCUPANCY_REPEAT",
            "SCIENTIFIC_COMPLETED_REPEATING_FOR_GPU_OCCUPANCY",
            occupancy_mode="repeat_after_success",
            repeat_iteration=iteration,
            repeat_result_dir=runtime_result_dir,
            workload_pid=workload_pid,
            process_alive=workload_pid > 0,
            result_selection_eligible=False,
            repeat_metrics_ignored=True,
            affects_scientific_result=False,
        )

    def heartbeat(
        self, *, stage: str, progress: dict[str, Any] | None = None, process_alive: bool = True
    ) -> dict[str, Any]:
        payload = self.read()
        updates: dict[str, Any] = {
            "stage": stage,
            "heartbeat_at": utc_now(),
            "process_alive": process_alive,
        }
        if progress is not None:
            updates["progress"] = progress
        return self.transition(
            payload["scientific_state"],
            payload["execution_state"],
            payload["status_code"],
            **updates,
        )

    def _validate(self, payload: dict[str, Any]) -> None:
        if payload.get("scientific_state") not in SCIENTIFIC_STATES:
            raise ValueError("unknown scientific state")
        if payload.get("execution_state") not in EXECUTION_STATES:
            raise ValueError("unknown execution state")
        if payload.get("test_read") or payload.get("sports_read"):
            raise PermissionError("Stage17 status cannot authorize official test or Sports reads")
        if payload["execution_state"] == "RUNNING_OCCUPANCY_REPEAT":
            required = {
                "scientific_state": "COMPLETED",
                "result_selection_eligible": False,
                "repeat_metrics_ignored": True,
                "affects_scientific_result": False,
            }
            for key, expected in required.items():
                if payload.get(key) != expected:
                    raise ValueError(f"runtime isolation field {key} must be {expected!r}")
        for key in ("experiment_id", "tmux_session", "log_path", "canonical_result_dir"):
            if payload.get(key):
                assert_neutral_name(str(payload[key]))

    def _write(self, payload: dict[str, Any]) -> None:
        atomic_json(self.paths.status(self.experiment_id), payload)
        self._update_index(payload)

    def _update_index(self, payload: dict[str, Any]) -> None:
        self.paths.status_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.paths.status_dir / ".phase17.index.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            index = {"schema_version": "phase17.index.v1", "experiments": {}}
            if self.paths.index.exists():
                index = json.loads(self.paths.index.read_text(encoding="utf-8"))
            index.setdefault("experiments", {})[self.experiment_id] = {
                "step_id": payload["step_id"],
                "scientific_state": payload["scientific_state"],
                "execution_state": payload["execution_state"],
                "status_code": payload["status_code"],
                "updated_at": payload["updated_at"],
                "status_path": str(self.paths.status(self.experiment_id)),
            }
            index["updated_at"] = utc_now()
            atomic_json(self.paths.index, index)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class AttemptLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        required = {"attempt_id", "step_id", "kind", "started_at", "state", "scientific_result_eligible"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"attempt record is missing fields: {missing}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            existing = [json.loads(line) for line in handle if line.strip()]
            if any(row["attempt_id"] == record["attempt_id"] for row in existing):
                raise ValueError(f"attempt id already exists: {record['attempt_id']}")
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
