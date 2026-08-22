#!/usr/bin/env python3
"""Stage15 S15-0: deterministic, network-free source/artifact compatibility audit.

The script inspects local third-party clones only. It never fetches, installs,
checks out, pulls Git LFS objects, or imports either third-party project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_local_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode:
        return f"ERROR[{proc.returncode}]: {proc.stderr.strip()}"
    return proc.stdout.strip()


def tracked_files(repo: Path) -> list[Path]:
    output = run_local_git(repo, "ls-files", "-z")
    if output.startswith("ERROR["):
        return []
    return [repo / name for name in output.split("\0") if name]


def parse_lfs_pointer(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size > 1024:
        return None
    data = path.read_bytes()
    if not data.startswith(LFS_HEADER):
        return None
    oid_match = re.search(rb"^oid sha256:([0-9a-f]{64})$", data, re.MULTILINE)
    size_match = re.search(rb"^size ([0-9]+)$", data, re.MULTILINE)
    if not oid_match or not size_match:
        raise ValueError(f"Malformed LFS pointer: {path}")
    return {
        "path": str(path),
        "oid_sha256": oid_match.group(1).decode(),
        "object_bytes": int(size_match.group(1)),
        "working_tree_bytes": path.stat().st_size,
        "materialized": False,
    }


def artifact_state(path: Path) -> str:
    pointer = parse_lfs_pointer(path)
    if pointer:
        return "lfs_pointer_not_materialized"
    return "materialized_file"


def read_requirement_lines(repo: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    candidates = list(repo.glob("requirements*.txt"))
    candidates += list(repo.glob("environment*.y*ml"))
    candidates += list(repo.glob("pyproject.toml"))
    for path in sorted(set(candidates)):
        lines = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                lines.append(clean)
        result[str(path.relative_to(repo))] = lines
    return result


def find_licenses(repo: Path) -> list[str]:
    names = ("LICENSE*", "COPYING*", "NOTICE*")
    found: set[Path] = set()
    for name in names:
        found.update(path for path in repo.glob(name) if path.is_file())
        found.update(path for path in repo.glob(f"**/{name}") if ".git" not in path.parts and path.is_file())
    return [str(path.relative_to(repo)) for path in sorted(found)]


def verify_assertions(repo: Path, assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified = []
    for assertion in assertions:
        rel = Path(assertion["path"])
        path = repo / rel
        needle = assertion["line_contains"]
        matches = []
        if path.is_file():
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if needle in line:
                    matches.append({"line": number, "text": line.strip()})
        record = dict(assertion)
        record["evidence"] = matches
        record["assertion_verified"] = bool(matches)
        verified.append(record)
    return verified


def audit_source(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = ROOT / source["path"]
    base = {
        "name": source["name"],
        "role": source["role"],
        "path": source["path"],
        "expected_remote": source["expected_remote"],
        "paper_url": source["paper_url"],
        "exists": repo.is_dir(),
        "s15_0_gate": source["s15_0_gate"],
        "gate_reason": source["gate_reason"],
    }
    if not repo.is_dir():
        base["retrieval_note"] = source.get("retrieval_note")
        return base, {"name": source["name"], "checks": []}, {"name": source["name"], "requirements": {}}, {
            "name": source["name"], "assertions": [], "native_environment": source["native_environment"]
        }

    files = tracked_files(repo)
    pointers = [pointer for path in files if (pointer := parse_lfs_pointer(path))]
    licenses = find_licenses(repo)
    origin = run_local_git(repo, "remote", "get-url", "origin")
    base.update(
        {
            "origin": origin,
            "origin_matches_expected": origin.rstrip("/") == source["expected_remote"].rstrip("/"),
            "branch": run_local_git(repo, "branch", "--show-current"),
            "commit": run_local_git(repo, "rev-parse", "HEAD"),
            "commit_time": run_local_git(repo, "log", "-1", "--format=%cI"),
            "commit_subject": run_local_git(repo, "log", "-1", "--format=%s"),
            "worktree_status": run_local_git(repo, "status", "--short"),
            "submodules": run_local_git(repo, "submodule", "status"),
            "tracked_file_count": len(files),
            "tracked_worktree_bytes": sum(path.stat().st_size for path in files if path.is_file()),
            "license_files": licenses,
            "license_status": "present" if licenses else "NO_LICENSE_FILE_AT_HEAD",
            "lfs_pointer_count": len(pointers),
            "lfs_unmaterialized_bytes": sum(pointer["object_bytes"] for pointer in pointers),
        }
    )

    checks = []
    for check in source.get("artifact_checks", []):
        matches = sorted(repo.glob(check["glob"]))
        checks.append(
            {
                **check,
                "match_count": len(matches),
                "matches": [
                    {
                        "path": str(path.relative_to(repo)),
                        "state": artifact_state(path),
                        "working_tree_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "lfs": parse_lfs_pointer(path),
                    }
                    for path in matches
                    if path.is_file()
                ],
            }
        )
    artifacts = {
        "name": source["name"],
        "checks": checks,
        "lfs_pointers": [
            {**pointer, "path": str(Path(pointer["path"]).relative_to(repo))} for pointer in pointers
        ],
        "lfs_unmaterialized_bytes": sum(pointer["object_bytes"] for pointer in pointers),
    }
    dependencies = {
        "name": source["name"],
        "requirements": read_requirement_lines(repo),
        "native_environment": source["native_environment"],
    }
    compatibility = {
        "name": source["name"],
        "assertions": verify_assertions(repo, source.get("compatibility_assertions", [])),
        "native_environment": source["native_environment"],
    }
    return base, artifacts, dependencies, compatibility


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiment/phase15/configs/stage15_s0_sources.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["output_dir"]
    generated_at = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": config["schema_version"],
        "audit_id": config["audit_id"],
        "generated_at_utc": generated_at,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "network_policy": config["network_policy"],
    }

    manifests, artifacts, dependencies, compatibility = [], [], [], []
    for source in config["sources"]:
        manifest, artifact, dependency, compat = audit_source(source)
        manifests.append(manifest)
        artifacts.append(artifact)
        dependencies.append(dependency)
        compatibility.append(compat)

    if any(
        not assertion["assertion_verified"]
        for record in compatibility
        for assertion in record["assertions"]
    ):
        raise SystemExit("At least one configured compatibility assertion was not found; audit source drifted.")

    write_json(output_dir / "source_manifest.json", {**common, "sources": manifests})
    write_json(output_dir / "artifact_inventory.json", {**common, "sources": artifacts})
    write_json(
        output_dir / "dependency_matrix.json",
        {
            **common,
            "local_environment": config["local_environment"],
            "decision": "USE_TWO_ISOLATED_NATIVE_ENVS_AND_DO_NOT_MUTATE_GRAM_REPRO",
            "sources": dependencies,
        },
    )
    write_json(
        output_dir / "compatibility_matrix.json",
        {
            **common,
            "protocol_boundaries": config["protocol_boundaries"],
            "sources": compatibility,
        },
    )
    print(json.dumps({"audit_id": config["audit_id"], "output_dir": str(output_dir), "sources": [s["name"] for s in manifests]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
