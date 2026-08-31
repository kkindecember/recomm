"""Immutable command snapshots and neutral, isolated runtime directories."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .status_writer import assert_neutral_name, atomic_json, utc_now


def background_required(estimated_seconds: float | None) -> bool:
    """Unknown duration and anything above ten minutes must use background mode."""
    return estimated_seconds is None or estimated_seconds > 600


def launch_background_tmux(
    *,
    experiment_id: str,
    argv: list[str],
    cwd: Path,
    tmux_session: str | None = None,
    startup_log_path: Path | None = None,
) -> str:
    """Launch one immutable worker in a neutral tmux session; never retry implicitly."""
    session = tmux_session or experiment_id
    assert_neutral_name(experiment_id)
    assert_neutral_name(session)
    if not argv:
        raise ValueError("background command is empty")
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is unavailable; background experiment was not started")
    existing = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if existing.returncode == 0:
        raise FileExistsError(f"tmux session already exists: {session}")
    command = shlex.join(argv)
    if startup_log_path is not None:
        startup_log_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            f"exec {command} >> {shlex.quote(str(startup_log_path))} 2>&1"
        )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(cwd), command],
        check=True,
    )
    return session


def tmux_session_exists(session: str) -> bool:
    """Read-only liveness check used for bounded startup handshakes."""

    assert_neutral_name(session)
    completed = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def wait_for_tmux_startup(session: str, *, grace_seconds: float = 2.0) -> bool:
    """Wait briefly for import-time failures; this is not experiment monitoring."""

    if grace_seconds < 0 or grace_seconds > 10:
        raise ValueError("startup grace must be between zero and ten seconds")
    time.sleep(grace_seconds)
    return tmux_session_exists(session)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_run_snapshot(
    *,
    root: Path,
    experiment_id: str,
    attempt_id: str,
    command: list[str],
    source_paths: Iterable[Path],
    config: dict,
) -> Path:
    assert_neutral_name(experiment_id)
    snapshot_dir = root / "artifacts/phase17/snapshots" / experiment_id / attempt_id
    if snapshot_dir.exists():
        raise FileExistsError(f"run snapshot already exists: {snapshot_dir}")
    source_dir = snapshot_dir / "src"
    source_dir.mkdir(parents=True)
    files = []
    for index, source in enumerate(source_paths):
        source = source.resolve()
        if root.resolve() not in source.parents:
            raise PermissionError(f"snapshot source is outside repository: {source}")
        target = source_dir / f"{index:03d}_{source.name}"
        shutil.copy2(source, target)
        files.append(
            {
                "source_path": str(source.relative_to(root.resolve())),
                "snapshot_path": str(target.relative_to(root)),
                "sha256": sha256(target),
            }
        )
        target.chmod(0o444)
    config_path = snapshot_dir / "config.json"
    atomic_json(config_path, config)
    config_path.chmod(0o444)
    manifest = {
        "schema_version": "phase17.run_snapshot.v1",
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "created_at": utc_now(),
        "command": command,
        "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
        "config_sha256": sha256(config_path),
        "files": files,
    }
    manifest_path = snapshot_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    return manifest_path


def verify_run_snapshot(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = manifest_path.parent / "config.json"
    if sha256(config_path) != manifest["config_sha256"]:
        raise RuntimeError("immutable config snapshot hash changed")
    for record in manifest["files"]:
        if sha256(root / record["snapshot_path"]) != record["sha256"]:
            raise RuntimeError(f"immutable source snapshot hash changed: {record['snapshot_path']}")


def isolated_runtime_dir(root: Path, experiment_id: str, iteration: int) -> Path:
    assert_neutral_name(experiment_id)
    if iteration < 2:
        raise ValueError("run-0001 is reserved for the canonical scientific execution")
    path = root / "artifacts/phase17/runtime" / experiment_id / f"run-{iteration:04d}"
    assert_neutral_name(str(path))
    return path


def assert_runtime_isolation(canonical_dir: Path, runtime_dir: Path) -> None:
    canonical = canonical_dir.resolve()
    runtime = runtime_dir.resolve()
    if canonical == runtime or canonical in runtime.parents or runtime in canonical.parents:
        raise PermissionError("runtime output and canonical result trees must be disjoint")
    assert_neutral_name(str(runtime))
