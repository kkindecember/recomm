#!/usr/bin/env python3
"""Build and verify the isolated runtime used by Stage16 S16-3R formal f2."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = ROOT / ".runtime/phase16_s3r_gridge_f2_runtime"
F2_CONFIG = "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f2.json"
REQUIRED_GRAM_PY_SHA256 = "275f10a94fdcfac9dd7323b43ba1932563bc21b4647906a2d7a0f70a75516466"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def export_head_gram_src(destination: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", commit, "GRAM/src"], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(destination)
    return commit


def load_formal_code_paths(snapshot: Path) -> tuple[str, ...]:
    namespace: dict[str, Any] = {}
    source = snapshot / "experiment/phase16/protocol/gridge_formal_admission.py"
    # Avoid importing torch while preparing.  Extract the literal tuple from
    # the copied source with the stdlib AST instead.
    import ast

    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FORMAL_CODE_PATHS":
                    value = ast.literal_eval(node.value)
                    return tuple(map(str, value))
    raise ValueError("Copied formal worker does not declare FORMAL_CODE_PATHS")


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "runtime_snapshot_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Missing isolated-runtime manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if snapshot.resolve() != Path(manifest["snapshot_root"]).resolve():
        raise ValueError("Isolated-runtime root drift")
    if not (snapshot / "artifacts").is_symlink():
        raise ValueError("Isolated runtime must share only the parent artifact tree")
    if (snapshot / "artifacts").resolve() != (ROOT / "artifacts").resolve():
        raise ValueError("Isolated artifact link target drift")
    if not (snapshot / "GRAM/rec_datasets").is_symlink():
        raise ValueError("Isolated runtime must use the frozen dataset parent link")
    if (snapshot / "GRAM/rec_datasets").resolve() != (ROOT / "GRAM/rec_datasets").resolve():
        raise ValueError("Isolated dataset link target drift")
    gram_py = snapshot / "GRAM/src/model/gram.py"
    if sha256(gram_py) != REQUIRED_GRAM_PY_SHA256:
        raise ValueError("Isolated GRAM source is not the f1-frozen HEAD version")
    for relative, expected in manifest["code_sha256"].items():
        path = snapshot / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Isolated runtime code drift: {relative}")
    config_path = snapshot / F2_CONFIG
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Isolated f2 config drift")
    source = snapshot / ".runtime/phase15_sources/GenRecEdit"
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if head != manifest["official_genrecedit_commit"] or dirty:
        raise ValueError("Isolated official GenRecEdit source drift")
    recbole = snapshot / ".runtime/phase16_sources/RecBole"
    recbole_head = subprocess.check_output(
        ["git", "-C", str(recbole), "rev-parse", "HEAD"], text=True
    ).strip()
    recbole_dirty = subprocess.check_output(
        ["git", "-C", str(recbole), "status", "--porcelain"], text=True
    ).strip()
    if recbole_head != manifest["official_recbole_commit"] or recbole_dirty:
        raise ValueError("Isolated official RecBole source drift")
    return manifest


def prepare_snapshot(snapshot: Path) -> dict[str, Any]:
    if snapshot.exists() or snapshot.is_symlink():
        return verify_snapshot(snapshot)
    temporary = snapshot.with_name(f"{snapshot.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale temporary snapshot: {temporary}")
    temporary.mkdir(parents=True)
    try:
        (temporary / "experiment").mkdir()
        shutil.copytree(ROOT / "experiment/phase15", temporary / "experiment/phase15")
        shutil.copytree(ROOT / "experiment/phase16", temporary / "experiment/phase16")
        commit = export_head_gram_src(temporary)
        os.symlink((ROOT / "GRAM/rec_datasets").resolve(), temporary / "GRAM/rec_datasets")
        os.symlink((ROOT / "artifacts").resolve(), temporary / "artifacts")
        (temporary / ".runtime").mkdir()
        shutil.copytree(
            ROOT / ".runtime/phase15_sources",
            temporary / ".runtime/phase15_sources",
        )
        shutil.copytree(
            ROOT / ".runtime/phase16_sources",
            temporary / ".runtime/phase16_sources",
        )
        config_path = temporary / F2_CONFIG
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            config["attempt_id"] != "s16_s3r_gridge_formal_gpu5_f2"
            or config["runtime_isolation"]["required_gram_py_sha256"]
            != REQUIRED_GRAM_PY_SHA256
            or config["post_terminal_repeat_policy"]["affects_scientific_results"]
            is not False
            or config["post_terminal_repeat_policy"]["promotion_eligible"] is not False
        ):
            raise ValueError("f2 config does not preserve the isolated recovery contract")
        gram_py = temporary / "GRAM/src/model/gram.py"
        if sha256(gram_py) != REQUIRED_GRAM_PY_SHA256:
            raise ValueError("git HEAD no longer matches the f1-frozen GRAM source")
        code_paths = load_formal_code_paths(temporary)
        code_sha = {}
        for relative in code_paths:
            path = temporary / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing isolated formal code: {relative}")
            code_sha[relative] = sha256(path)
        official = temporary / ".runtime/phase15_sources/GenRecEdit"
        official_commit = subprocess.check_output(
            ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
        ).strip()
        recbole = temporary / ".runtime/phase16_sources/RecBole"
        recbole_commit = subprocess.check_output(
            ["git", "-C", str(recbole), "rev-parse", "HEAD"], text=True
        ).strip()
        manifest = {
            "schema_version": "stage16_s3r_f2_isolated_runtime_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_root": str(snapshot.resolve()),
            "source_repository_root": str(ROOT.resolve()),
            "root_git_commit": commit,
            "gram_source_mode": "git HEAD immutable export",
            "gram_py_sha256": sha256(gram_py),
            "config_path": F2_CONFIG,
            "config_sha256": sha256(config_path),
            "official_genrecedit_commit": official_commit,
            "official_recbole_commit": recbole_commit,
            "code_sha256": code_sha,
            "shared_parent_links": {
                "artifacts": str((ROOT / "artifacts").resolve()),
                "GRAM/rec_datasets": str((ROOT / "GRAM/rec_datasets").resolve()),
            },
            "main_worktree_mutations_visible": False,
        }
        write_json(temporary / "runtime_snapshot_manifest.json", manifest)
        temporary.replace(snapshot)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_snapshot(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    snapshot = args.snapshot_root.resolve()
    result = prepare_snapshot(snapshot) if args.command == "prepare" else verify_snapshot(snapshot)
    print(json.dumps({"status": "PASS", "snapshot_root": str(snapshot), "config_sha256": result["config_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
