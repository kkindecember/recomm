#!/usr/bin/env python3
"""Freeze the post-run S17-1 integrity and code manifests."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import torch

from experiment.phase17.core.run_manager import sha256, verify_run_snapshot
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, rebuild_phase_index, utc_now


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/phase17/s1_contract/attempt_001"
STATUS = ROOT / "artifacts/phase17/status/s17_s1_public_framework.status.json"
SNAPSHOT = ROOT / "artifacts/phase17/snapshots/s17_s1_public_framework/attempt_001/manifest.json"


def main() -> int:
    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if summary["verdict"] != "PASS_S17_1_CONTRACT_AND_GPU_SMOKE":
        raise SystemExit("S17-1 smoke is not successful")
    if status["scientific_state"] != "COMPLETED" or status["execution_state"] != "SCIENTIFIC_COMPLETED":
        raise SystemExit("S17-1 status is not terminal-complete")
    verify_run_snapshot(ROOT, SNAPSHOT)

    cpu_text = (OUTPUT / "cpu_contract_tests.log").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran (\d+) tests", cpu_text)
    if not match or "\nOK\n" not in cpu_text:
        raise SystemExit("CPU contract test terminal evidence is missing")
    test_count = int(match.group(1))

    final_test_command = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "unittest",
        "discover",
        "-v",
        "-s",
        "experiment/phase17/tests",
        "-p",
        "test_*.py",
    ]
    completed = subprocess.run(final_test_command, cwd=ROOT, capture_output=True, text=True, check=False)
    final_test_log = OUTPUT.parent / "postrun_contract_tests.log"
    final_test_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    final_match = re.search(r"Ran (\d+) tests", completed.stdout + completed.stderr)
    if completed.returncode != 0 or not final_match or "\nOK\n" not in completed.stdout + completed.stderr:
        raise SystemExit("post-run CPU contract tests failed")
    final_test_count = int(final_match.group(1))

    checkpoint = next((OUTPUT / "gram_logs").rglob("model_rec_phase_1_epoch_1.pt"))
    state = torch.load(checkpoint, map_location="cpu")
    parent_parameter_numel = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    stage17_parameter_numel = sum(
        value.numel()
        for key, value in state.items()
        if torch.is_tensor(value) and "migration_runtime" in key
    )

    code_paths = [
        *sorted((ROOT / "experiment/phase17/core").glob("*.py")),
        ROOT / "experiment/phase17/registry/module_registry.py",
        ROOT / "experiment/phase17/protocol/s1_contract_runtime.py",
        ROOT / "experiment/phase17/protocol/finalize_s1_contract.py",
        ROOT / "experiment/phase17/run_stage17_s1_contract_smoke.sh",
        *sorted((ROOT / "experiment/phase17/tests").glob("test_*.py")),
        ROOT / "GRAM/src/model/gram.py",
        ROOT / "GRAM/src/arguments.py",
        ROOT / "GRAM/src/main_generative_gram.py",
    ]
    code_manifest = {
        "schema_version": "phase17.s1_code_manifest.v1",
        "generated_at": utc_now(),
        "files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in code_paths
        },
    }
    code_manifest_path = ROOT / "artifacts/phase17/s1_contract/code_manifest.json"
    atomic_json(code_manifest_path, code_manifest)

    peak_reserved = max(row["peak_reserved_mib"] for row in summary["gpu_smoke"]["resource_metrics"])
    summary.update(
        {
            "finalized_at": utc_now(),
            "contract_tests": {
                "snapshot_passed": test_count,
                "final_current_passed": final_test_count,
                "failed": 0,
                "postrun_log_path": str(final_test_log.relative_to(ROOT)),
            },
            "parameter_count": {
                "parent_checkpoint_numel": parent_parameter_numel,
                "enabled_stage17_module_numel": stage17_parameter_numel,
            },
            "resource_contract": {
                "peak_reserved_mib": peak_reserved,
                "usable_memory_ceiling_mib": 30 * 1024,
                "headroom_mib": 30 * 1024 - peak_reserved,
                "end_to_end_gpu_hours": summary["gpu_smoke"]["wall_seconds"] / 3600.0,
            },
            "code_manifest_path": str(code_manifest_path.relative_to(ROOT)),
            "code_manifest_sha256": sha256(code_manifest_path),
            "effect_comparison_performed": False,
            "official_test_metrics_used": False,
        }
    )
    report_path = ROOT / "report/第十七阶段/Stage17_S1_公共迁移框架与运行合约报告.md"
    if report_path.exists():
        summary.update(
            {
                "report_path": str(report_path.relative_to(ROOT)),
                "report_sha256": sha256(report_path),
            }
        )
    atomic_json(summary_path, summary)
    if report_path.exists():
        writer = StatusWriter(ROOT / "artifacts/phase17/status", "s17_s1_public_framework")
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            "S17_1_COMPLETE_REPORT_FROZEN",
            report_path=str(report_path.relative_to(ROOT)),
            report_sha256=sha256(report_path),
            summary_sha256=sha256(summary_path),
            next_gate="S17-2",
        )
    rebuild_phase_index(ROOT / "artifacts/phase17/status")
    print(
        f"PASS_S17_1_FINALIZATION snapshot_tests={test_count} "
        f"final_tests={final_test_count} peak_reserved_mib={peak_reserved}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
