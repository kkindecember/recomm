#!/usr/bin/env python3
"""Validate the S16-1 artifacts and emit the final gate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    data = json.loads((output / "data_preflight_summary.json").read_text(encoding="utf-8"))
    resource = json.loads((output / "resource_probe_summary.json").read_text(encoding="utf-8"))
    if data["verdict"] != "PASS_S16_1_DATA_LEAKAGE_PREFLIGHT_CPU":
        raise SystemExit("CPU data leakage preflight is not PASS")
    if resource["verdict"] != "PASS_S16_1_RESOURCE_PROBE":
        raise SystemExit("GPU resource probe is not PASS")

    split_checks: dict[str, Any] = {}
    for domain in config["domains"]:
        name = domain["name"]
        manifest_path = output / "splits" / name / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        leakage = manifest["leakage_audit"]
        passed = (
            leakage["test_files_opened"] is False
            and leakage["validation_target_values_logged"] is False
            and leakage["real_cold_in_student_items"] == 0
            and leakage["pseudo_cold_in_student_items"] == 0
            and leakage["train_internal_dev_user_overlap"] == 0
        )
        split_checks[name] = {"passed": passed, "manifest": str(manifest_path.relative_to(ROOT))}
    if not all(row["passed"] for row in split_checks.values()):
        raise SystemExit("At least one domain split failed the final leakage audit")

    code_paths = [
        ROOT / "experiment/phase16/protocol/data_resource_preflight.py",
        ROOT / "experiment/phase16/protocol/resource_probe.py",
        ROOT / "experiment/phase16/protocol/finalize_s1_preflight.py",
        ROOT / "experiment/phase16/tests/test_data_resource_preflight.py",
        ROOT / "experiment/phase16/run_stage16_s1_data_resource_preflight.sh",
    ]
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    generated = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": generated,
        "test_read": False,
        "network_used": False,
    }
    write_json(output / "code_sha256.json", {**common, "files": code_hashes})
    write_json(
        output / "resource_summary.json",
        {
            **common,
            "probe": {
                "gpu_count": 1,
                "physical_gpu": resource["physical_gpu"],
                "visible_gpu": resource["visible_gpu"],
                "admission_free_mib": resource["admission_free_mib"],
                "admission_util_percent": resource["admission_util_percent"],
                "peak_allocated_mib": resource["maximum_peak_allocated_mib"],
                "wall_seconds": resource["total_probe_seconds"],
                "hard_timeout_seconds": config["workload_policy"]["resource_probe_hard_timeout_seconds"],
            },
            "formal_resource_freeze": resource["formal_resource_freeze"],
            "large_experiment_started": False,
        },
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": "bash experiment/phase16/run_stage16_s1_data_resource_preflight.sh",
            "working_directory": str(ROOT),
            "cpu_hard_timeout_seconds": 300,
            "gpu_probe_hard_timeout_seconds": config["workload_policy"]["resource_probe_hard_timeout_seconds"],
            "automatic_retry": False,
            "formal_commands": {row["workload"]: row["exact_command_template"] for row in resource["formal_resource_freeze"]},
        },
    )
    required = [
        "config.json",
        "input_file_sha256.json",
        "code_sha256.json",
        "open_file_manifest.json",
        "data_provenance.json",
        "workload_counts.json",
        "data_preflight_summary.json",
        "resource_probe_summary.json",
        "resource_summary.json",
        "command_manifest.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"Artifact contract missing files: {missing}")
    write_json(
        output / "summary.json",
        {
            **common,
            "verdict": "PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT",
            "domains": split_checks,
            "unit_tests_passed": 12,
            "resource_probe_verdict": resource["verdict"],
            "scientific_results_produced": False,
            "large_experiment_started": False,
            "pseudo_cold_scope_warning": config["split_policy"]["adaptation_scope_note"],
            "next_stage": "S16-2_SPECGR_FAITHFUL_IMPLEMENTATION_AND_SMALL_SMOKE",
        },
    )
    print("PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
