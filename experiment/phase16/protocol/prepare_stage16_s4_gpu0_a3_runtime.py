#!/usr/bin/env python3
"""Build and verify the immutable S16-4 GPU0 a3 runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUNTIME = ROOT / ".runtime/phase16_s3r_gridge_f3_runtime"
DEFAULT_SNAPSHOT = ROOT / ".runtime/phase16_s4_toys_gpu0_a3_runtime"
CONFIG = "experiment/phase16/configs/stage16_s4_toys_standalone_gpu0_a3.json"
OVERLAYS = (
    CONFIG,
    "experiment/phase16/protocol/stage16_s4_toys_validation.py",
    "experiment/phase16/protocol/finalize_stage16_s4_toys.py",
    "experiment/phase16/protocol/prepare_stage16_s4_gpu0_a3_runtime.py",
    "experiment/phase16/tests/test_stage16_s4_toys_frozen_preflight.py",
    "experiment/phase16/tests/test_stage16_s4_toys_validation.py",
    "experiment/phase16/configs/stage16_s4_toys_frozen_preflight.json",
    "experiment/phase16/protocol/stage16_s4_toys_frozen_preflight.py",
    "experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh",
    "experiment/phase16/run_stage16_s4_toys_standalone_gpu0_a3.sh",
    "experiment/phase16/run_stage16_s4_toys_standalone_gpu0_a3_inner.sh",
    "experiment/phase16/run_stage16_s4_toys_repeat_gpu0_a3_inner.sh",
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def identity_paths(snapshot: Path) -> list[Path]:
    roots = (
        snapshot / "experiment/phase15/protocol",
        snapshot / "experiment/phase16/protocol",
        snapshot / "experiment/phase16/tests",
        snapshot / "GRAM/src",
        snapshot / ".runtime/phase15_sources/SpecGR",
        snapshot / ".runtime/phase16_sources/RecBole",
    )
    paths: list[Path] = []
    for root in roots:
        paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    paths.extend(snapshot / relative for relative in OVERLAYS if not relative.endswith(".py"))
    return sorted(set(paths))


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "runtime_snapshot_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing S16-4 isolated-runtime manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if snapshot.resolve() != Path(manifest["snapshot_root"]).resolve():
        raise ValueError("S16-4 isolated-runtime root drift")
    for relative, target in (
        ("artifacts", ROOT / "artifacts"),
        ("GRAM/rec_datasets", ROOT / "GRAM/rec_datasets"),
    ):
        link = snapshot / relative
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise ValueError(f"S16-4 shared parent link drift: {relative}")
    for relative, expected in manifest["code_sha256"].items():
        path = snapshot / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"S16-4 isolated code drift: {relative}")
    config_path = snapshot / CONFIG
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("S16-4 isolated config drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("physical_gpu") != 0
        or config.get("runtime_isolation", {}).get("snapshot_root")
        != ".runtime/phase16_s4_toys_gpu0_a3_runtime"
        or config.get("post_terminal_repeat", {}).get("discard_output") is not True
        or config.get("post_terminal_repeat", {}).get("formal_output_read_only") is not True
    ):
        raise ValueError("S16-4 GPU0/repeat isolation contract drift")
    return manifest


def prepare_snapshot(snapshot: Path) -> dict[str, Any]:
    if snapshot.exists() or snapshot.is_symlink():
        return verify_snapshot(snapshot)
    if not SOURCE_RUNTIME.is_dir():
        raise FileNotFoundError("Frozen f3 source runtime is missing")
    temporary = snapshot.with_name(f"{snapshot.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Stale S16-4 runtime temporary exists: {temporary}")
    try:
        shutil.copytree(
            SOURCE_RUNTIME,
            temporary,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for relative in OVERLAYS:
            source = ROOT / relative
            destination = temporary / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Missing/non-regular S16-4 overlay: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        code_sha: dict[str, str] = {}
        for path in identity_paths(temporary):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing/non-regular S16-4 identity file: {path}")
            code_sha[str(path.relative_to(temporary))] = sha256(path)
        config_path = temporary / CONFIG
        source_manifest = SOURCE_RUNTIME / "runtime_snapshot_manifest.json"
        manifest = {
            "schema_version": "stage16_s4_toys_gpu0_a3_isolated_runtime_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_root": str(snapshot.resolve()),
            "source_repository_root": str(ROOT.resolve()),
            "source_runtime": str(SOURCE_RUNTIME.resolve()),
            "source_runtime_manifest_sha256": sha256(source_manifest),
            "config_path": CONFIG,
            "config_sha256": sha256(config_path),
            "code_sha256": code_sha,
            "shared_parent_links": {
                "artifacts": str((ROOT / "artifacts").resolve()),
                "GRAM/rec_datasets": str((ROOT / "GRAM/rec_datasets").resolve()),
            },
            "main_worktree_mutations_visible": False,
            "formal_output_write_scope": "artifacts/phase16/s4_toys_standalone/formal/toys_seed1502_gpu0_a3",
            "repeat_output_mode": "discard_only_no_formal_writes",
        }
        write_json(temporary / "runtime_snapshot_manifest.json", manifest)
        os.replace(temporary, snapshot)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_snapshot(snapshot)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = args.snapshot_root.resolve()
    result = prepare_snapshot(snapshot) if args.command == "prepare" else verify_snapshot(snapshot)
    print(
        json.dumps(
            {
                "status": "PASS",
                "snapshot_root": str(snapshot),
                "config_path": result["config_path"],
                "config_sha256": result["config_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
