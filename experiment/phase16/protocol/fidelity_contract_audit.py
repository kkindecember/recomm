#!/usr/bin/env python3
"""Run the Stage16 S16-0 network-free fidelity contract audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fidelity_bridge import run_bridge_checks


ROOT = Path(__file__).resolve().parents[3]
ALLOWED_CLASSES = {"F0", "F1", "F2", "F3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed for {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def evidence(repo: Path, relative: str, needle: str) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        return {"path": relative, "contains": needle, "verified": False, "matches": []}
    matches = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if needle in line
    ]
    return {"path": relative, "contains": needle, "verified": bool(matches), "matches": matches}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiment/phase16/configs/stage16_s0_fidelity_contract.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if (output / "summary.json").exists():
        raise SystemExit("Refusing to overwrite an existing S16-0 summary; use a new attempt directory.")
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    sources: dict[str, dict[str, Any]] = {}
    source_records = []
    for source in config["sources"]:
        repo = ROOT / source["path"]
        if not repo.is_dir():
            raise SystemExit(f"Pinned source is missing: {source['path']}")
        record = {
            **source,
            "actual_remote": git(repo, "remote", "get-url", "origin"),
            "actual_commit": git(repo, "rev-parse", "HEAD"),
            "worktree_status": git(repo, "status", "--short"),
            "license_files": sorted(
                str(path.relative_to(repo))
                for pattern in ("LICENSE*", "COPYING*", "NOTICE*")
                for path in repo.glob(pattern)
                if path.is_file()
            ),
        }
        record["remote_verified"] = record["actual_remote"].rstrip("/") == source["expected_remote"].rstrip("/")
        record["commit_verified"] = record["actual_commit"] == source["expected_commit"]
        record["worktree_clean"] = record["worktree_status"] == ""
        record["license_status"] = "present" if record["license_files"] else "NO_LICENSE_FILE_AT_HEAD"
        if not (record["remote_verified"] and record["commit_verified"] and record["worktree_clean"]):
            raise SystemExit(f"Pinned source verification failed: {source['name']}")
        sources[source["name"]] = {**source, "repo": repo}
        source_records.append(record)

    enriched_parameters = []
    opened_paths: set[Path] = {config_path}
    for parameter in config["official_parameters"]:
        source = sources[parameter["source"]]
        proof = evidence(source["repo"], parameter["path"], parameter["contains"])
        opened_paths.add(source["repo"] / parameter["path"])
        enriched_parameters.append({**parameter, "evidence": proof})
    if any(not row["evidence"]["verified"] for row in enriched_parameters):
        raise SystemExit("An official parameter assertion drifted from the pinned source.")

    bridge_results = run_bridge_checks()
    bridge_by_id = {row["id"]: row for row in bridge_results}
    enriched_mappings = []
    for mapping in config["mappings"]:
        if mapping["classification"] not in ALLOWED_CLASSES:
            raise SystemExit(f"Unknown fidelity class: {mapping['classification']}")
        if mapping["classification"] in {"F2", "F3"} and mapping["main_table_eligible"]:
            raise SystemExit(f"Non-faithful mapping marked main-table eligible: {mapping['id']}")
        source = sources[mapping["official"]["source"]]
        proof = evidence(source["repo"], mapping["official"]["path"], mapping["official"]["contains"])
        opened_paths.add(source["repo"] / mapping["official"]["path"])
        check_id = mapping["gram_mapping"]["bridge_check"]
        check = bridge_by_id.get(check_id)
        enriched_mappings.append(
            {
                **mapping,
                "official_evidence": proof,
                "bridge_result": check or {"id": check_id, "status": "MISSING", "detail": "not registered"},
                "implementation_status": "SEMANTICS_FROZEN_IMPLEMENTATION_DEFERRED_TO_S16_2_OR_S16_3",
            }
        )
    if any(not row["official_evidence"]["verified"] for row in enriched_mappings):
        raise SystemExit("An official function-level evidence assertion drifted.")
    if any(row["bridge_result"]["status"] != "PASS" for row in enriched_mappings):
        raise SystemExit("At least one mapped component lacks a passing bridge check.")

    script_paths = [
        ROOT / "experiment/phase16/protocol/fidelity_bridge.py",
        ROOT / "experiment/phase16/protocol/fidelity_contract_audit.py",
        ROOT / "experiment/phase16/tests/test_fidelity_bridge.py",
        ROOT / "experiment/phase16/run_stage16_s0_fidelity_contract.sh",
    ]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(opened_paths)}
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in script_paths}
    gate = "PASS_S16_0_FIDELITY_CONTRACT"
    common = {
        "schema_version": config["schema_version"],
        "audit_id": config["audit_id"],
        "generated_at_utc": generated_at,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "test_read": False,
        "network_used": False,
    }

    write_json(output / "config.json", config)
    write_json(output / "source_manifest.json", {**common, "sources": source_records})
    write_json(
        output / "official_parameters.json",
        {**common, "parameters": enriched_parameters, "parameter_count": len(enriched_parameters)},
    )
    write_json(
        output / "fidelity_matrix.json",
        {
            **common,
            "classification_policy": config["classification_policy"],
            "mappings": enriched_mappings,
            "mapping_count": len(enriched_mappings),
            "main_table_eligible_count": sum(row["main_table_eligible"] for row in enriched_mappings),
            "f2_count": sum(row["classification"] == "F2" for row in enriched_mappings),
            "f3_count": sum(row["classification"] == "F3" for row in enriched_mappings),
        },
    )
    write_json(
        output / "bridge_test_summary.json",
        {
            **common,
            "contract": config["bridge_contract"],
            "checks": bridge_results,
            "passed": sum(row["status"] == "PASS" for row in bridge_results),
            "total": len(bridge_results),
        },
    )
    write_json(output / "input_file_sha256.json", {**common, "files": input_hashes})
    write_json(output / "code_sha256.json", {**common, "files": code_hashes})
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_files": sorted(input_hashes),
            "forbidden_patterns": ["user_sequence.txt", "predictions_test", "test_metrics"],
            "test_read": False,
        },
    )
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "inputs": "commit-pinned local source code and Stage16 contract config only",
            "recommendation_data_opened": False,
            "stage15_process_modified": False,
            "third_party_code_copied": False,
        },
    )
    write_json(
        output / "resource_summary.json",
        {
            **common,
            "resource_type": "CPU_ONLY_STATIC_AUDIT",
            "gpu_count": 0,
            "gpu_memory_mib": 0,
            "network_bytes": 0,
            "third_party_artifact_download_bytes": 0,
        },
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": "bash experiment/phase16/run_stage16_s0_fidelity_contract.sh",
            "working_directory": str(ROOT),
            "hard_timeout_seconds": 300,
            "automatic_retry": False,
        },
    )
    write_json(
        output / "summary.json",
        {
            **common,
            "verdict": gate,
            "source_count": len(source_records),
            "function_level_mappings": len(enriched_mappings),
            "bridge_checks_passed": sum(row["status"] == "PASS" for row in bridge_results),
            "bridge_checks_total": len(bridge_results),
            "blocked_components": [],
            "test_read": False,
            "next_stage": "S16-1_DATA_LEAKAGE_RESOURCE_PREFLIGHT",
        },
    )
    print(json.dumps({"verdict": gate, "output_dir": str(output), "mappings": len(enriched_mappings), "bridge_checks": len(bridge_results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

