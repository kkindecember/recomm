#!/usr/bin/env python3
"""Build and verify the immutable Stage16 S16-3R formal f3 runtime."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiment.phase16.protocol.prepare_s3r_gridge_f2_runtime import (
    REQUIRED_GRAM_PY_SHA256,
    export_head_gram_src,
    load_formal_code_paths,
    sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = ROOT / ".runtime/phase16_s3r_gridge_f3_runtime"
F2_CONFIG = "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f2.json"
F3_CONFIG = "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f3.json"
F2_STATUS = "artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2/status.json"
F2_IDENTITY = "artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2/execution_identity.json"
F2_STATUS_SHA256 = "c200f702b87f23152c477cb4b8ce10c21df55dc5f0eed81a03931c5aee86441c"
F2_IDENTITY_SHA256 = "e99239caa6b7de4978c303415e06096ba9cb6a1961e74decc4f4bea695314eee"


def derive_f3_config(f2: Mapping[str, Any], *, f2_config_sha256: str) -> dict[str, Any]:
    """Mechanically derive f3 while leaving every scientific field unchanged."""
    config = copy.deepcopy(dict(f2))
    config.update(
        {
            "schema_version": "stage16_s3r_gridge_formal_admission_v3_isolated_runtime",
            "attempt_id": "s16_s3r_gridge_formal_gpu5_f3",
            "output_dir": "artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f3",
            "exact_start_command": "bash experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3.sh",
        }
    )
    config["inputs"].update(
        {
            "failed_f2_config": {"path": F2_CONFIG, "sha256": f2_config_sha256},
            "failed_f2_status": {"path": F2_STATUS, "sha256": F2_STATUS_SHA256},
            "failed_f2_identity": {
                "path": F2_IDENTITY,
                "sha256": F2_IDENTITY_SHA256,
            },
        }
    )
    config["runtime_isolation"] = {
        **config["runtime_isolation"],
        "snapshot_root": ".runtime/phase16_s3r_gridge_f3_runtime",
    }
    config["post_terminal_repeat_policy"] = {
        **config["post_terminal_repeat_policy"],
        "repeat_root": "artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5_f3",
    }
    return config


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "runtime_snapshot_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Missing isolated-runtime manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if snapshot.resolve() != Path(manifest["snapshot_root"]).resolve():
        raise ValueError("Isolated-runtime root drift")
    for relative, target in (
        ("artifacts", ROOT / "artifacts"),
        ("GRAM/rec_datasets", ROOT / "GRAM/rec_datasets"),
    ):
        link = snapshot / relative
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise ValueError(f"Isolated parent link drift: {relative}")
    gram_py = snapshot / "GRAM/src/model/gram.py"
    if sha256(gram_py) != REQUIRED_GRAM_PY_SHA256:
        raise ValueError("Isolated GRAM source is not the f1-frozen HEAD version")
    for relative, expected in manifest["code_sha256"].items():
        path = snapshot / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Isolated runtime code drift: {relative}")
    config_path = snapshot / F3_CONFIG
    if not config_path.is_file() or sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Isolated f3 config drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("attempt_id") != "s16_s3r_gridge_formal_gpu5_f3"
        or config.get("runtime_isolation", {}).get("snapshot_root")
        != ".runtime/phase16_s3r_gridge_f3_runtime"
        or config.get("post_terminal_repeat_policy", {}).get("promotion_eligible")
        is not False
    ):
        raise ValueError("f3 recovery contract drift")
    for relative, commit_key in (
        (".runtime/phase15_sources/GenRecEdit", "official_genrecedit_commit"),
        (".runtime/phase16_sources/RecBole", "official_recbole_commit"),
    ):
        source = snapshot / relative
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
        if head != manifest[commit_key] or dirty:
            raise ValueError(f"Isolated official source drift: {relative}")
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
        shutil.copytree(ROOT / ".runtime/phase15_sources", temporary / ".runtime/phase15_sources")
        shutil.copytree(ROOT / ".runtime/phase16_sources", temporary / ".runtime/phase16_sources")

        f2_config_path = temporary / F2_CONFIG
        f2 = json.loads(f2_config_path.read_text(encoding="utf-8"))
        if sha256(ROOT / F2_STATUS) != F2_STATUS_SHA256:
            raise ValueError("Sealed f2 status drift")
        if sha256(ROOT / F2_IDENTITY) != F2_IDENTITY_SHA256:
            raise ValueError("Sealed f2 identity drift")
        config_path = temporary / F3_CONFIG
        write_json(
            config_path,
            derive_f3_config(f2, f2_config_sha256=sha256(f2_config_path)),
        )
        gram_py = temporary / "GRAM/src/model/gram.py"
        if sha256(gram_py) != REQUIRED_GRAM_PY_SHA256:
            raise ValueError("git HEAD no longer matches the f1-frozen GRAM source")
        code_sha: dict[str, str] = {}
        for relative in load_formal_code_paths(temporary):
            path = temporary / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing isolated formal code: {relative}")
            code_sha[relative] = sha256(path)
        official = temporary / ".runtime/phase15_sources/GenRecEdit"
        recbole = temporary / ".runtime/phase16_sources/RecBole"
        manifest = {
            "schema_version": "stage16_s3r_f3_isolated_runtime_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_root": str(snapshot.resolve()),
            "source_repository_root": str(ROOT.resolve()),
            "root_git_commit": commit,
            "gram_source_mode": "git HEAD immutable export",
            "gram_py_sha256": sha256(gram_py),
            "config_path": F3_CONFIG,
            "config_sha256": sha256(config_path),
            "official_genrecedit_commit": subprocess.check_output(
                ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
            ).strip(),
            "official_recbole_commit": subprocess.check_output(
                ["git", "-C", str(recbole), "rev-parse", "HEAD"], text=True
            ).strip(),
            "code_sha256": code_sha,
            "shared_parent_links": {
                "artifacts": str((ROOT / "artifacts").resolve()),
                "GRAM/rec_datasets": str((ROOT / "GRAM/rec_datasets").resolve()),
            },
            "main_worktree_mutations_visible": False,
            "f2_status_sha256": F2_STATUS_SHA256,
            "f2_identity_sha256": F2_IDENTITY_SHA256,
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
