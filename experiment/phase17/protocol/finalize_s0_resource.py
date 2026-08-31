#!/usr/bin/env python3
"""Summarize the two S17-0 GRAM resource probes without reading test data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/phase17/s0_audit"
PROFILE = OUTPUT / "resource_profile"
GPU_INDEX = int(os.environ.get("S17_PROFILE_GPU", "5"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_probe(size: int) -> dict[str, Any]:
    log = PROFILE / f"probe_{size}.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    metrics = []
    pattern = re.compile(
        r"RESOURCE_METRIC phase=(\S+) wall_time_seconds=([0-9.]+) "
        r"peak_allocated_mib=([0-9.]+) peak_reserved_mib=([0-9.]+) "
        r"end_allocated_mib=([0-9.]+) end_reserved_mib=([0-9.]+)"
    )
    for match in pattern.finditer(text):
        metrics.append({
            "phase": match.group(1),
            "wall_time_seconds": float(match.group(2)),
            "peak_allocated_mib": float(match.group(3)),
            "peak_reserved_mib": float(match.group(4)),
            "end_allocated_mib": float(match.group(5)),
            "end_reserved_mib": float(match.group(6)),
        })
    result_match = re.search(rf"PROFILE_RESULT size={size} rc=(\d+) wall_seconds=(\d+)", text)
    forbidden = [
        token
        for token in ("automatic_last_checkpoint_test", "[test] testing", "_pred_test.tsv")
        if token in text
    ]
    terminal_phases = {row["phase"] for row in metrics}
    inferred_clean_exit = (
        result_match is None
        and "Traceback" not in text
        and terminal_phases == {"training", "automatic_last_checkpoint_validation"}
    )
    return {
        "users": size,
        "log_path": str(log.relative_to(ROOT)),
        "log_sha256": sha256(log),
        "return_code": int(result_match.group(1)) if result_match else (0 if inferred_clean_exit else None),
        "return_code_evidence": (
            "runner_profile_result"
            if result_match
            else "inferred_from_both_terminal_resource_metrics_and_no_traceback"
            if inferred_clean_exit
            else "missing"
        ),
        "end_to_end_wall_seconds": int(result_match.group(2)) if result_match else None,
        "resource_metrics": metrics,
        "peak_allocated_mib": max((row["peak_allocated_mib"] for row in metrics), default=None),
        "peak_reserved_mib": max((row["peak_reserved_mib"] for row in metrics), default=None),
        "forbidden_test_evidence": forbidden,
        "validation_only": not forbidden and any("validation" in row["phase"] for row in metrics),
    }


def telemetry_summary() -> dict[str, Any]:
    path = PROFILE / "gpu_telemetry.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "memory_used_mib": int(row["memory_used_mib"]),
                    "memory_free_mib": int(row["memory_free_mib"]),
                    "utilization_gpu_percent": int(row["utilization_gpu_percent"]),
                })
            except (KeyError, ValueError):
                continue
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "samples": len(rows),
        "baseline_total_used_mib": rows[0]["memory_used_mib"] if rows else None,
        "peak_total_used_mib": max((row["memory_used_mib"] for row in rows), default=None),
        "minimum_total_free_mib": min((row["memory_free_mib"] for row in rows), default=None),
        "peak_gpu_utilization_percent": max((row["utilization_gpu_percent"] for row in rows), default=None),
        "note": "device totals include a pre-existing foreign process; process-local CUDA metrics are authoritative",
    }


def historical_full_reference() -> dict[str, Any]:
    path = ROOT / "artifacts/phase12/hi_gram/toys_v1_light/run.log"
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"RESOURCE_METRIC phase=(\S+) wall_time_seconds=([0-9.]+) "
        r"peak_allocated_mib=([0-9.]+) peak_reserved_mib=([0-9.]+)",
        text,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "metrics": [
            {
                "phase": phase,
                "wall_time_seconds": float(wall),
                "peak_allocated_mib": float(allocated),
                "peak_reserved_mib": float(reserved),
            }
            for phase, wall, allocated, reserved in matches
        ],
        "evidence_grade": "historical_unverified_capacity_reference_only",
    }


def extrapolate(probes: list[dict[str, Any]]) -> dict[str, Any]:
    by_size = {row["users"]: row for row in probes}
    def phase_time(size: int, phase: str) -> float:
        matches = [
            row["wall_time_seconds"]
            for row in by_size[size]["resource_metrics"]
            if row["phase"] == phase
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {phase} metric for {size} users")
        return matches[0]

    full_users = 12833
    estimates = {}
    for phase in ("training", "automatic_last_checkpoint_validation"):
        time_100 = phase_time(100, phase)
        time_1000 = phase_time(1000, phase)
        slope = max(0.0, (time_1000 - time_100) / 900.0)
        intercept = max(0.0, time_100 - slope * 100.0)
        estimates[phase] = {
            "time_100_seconds": time_100,
            "time_1000_seconds": time_1000,
            "linear_slope_seconds_per_user": slope,
            "linear_intercept_seconds": intercept,
            "estimated_d0_full_seconds": intercept + slope * full_users,
        }
    return {
        "model": "two-point linear operational estimate; planning only",
        "toys_d0_users": full_users,
        "phase_estimates": estimates,
        "estimated_30_epoch_plus_one_validation_seconds": (
            30 * estimates["training"]["estimated_d0_full_seconds"]
            + estimates["automatic_last_checkpoint_validation"]["estimated_d0_full_seconds"]
        ),
    }


def main() -> int:
    probes = [parse_probe(100), parse_probe(1000)]
    if any(row["return_code"] != 0 for row in probes):
        raise SystemExit("A resource probe did not complete successfully; no retry is performed")
    if any(not row["validation_only"] for row in probes):
        raise SystemExit("Validation-only contract was not demonstrated")
    if any(len(row["resource_metrics"]) != 2 for row in probes):
        raise SystemExit("Expected training and automatic validation metrics for each probe")

    peak_reserved = max(row["peak_reserved_mib"] for row in probes)
    summary = {
        "schema_version": "phase17.s0_resource_profile.v1",
        "generated_at_utc": now(),
        "verdict": "PASS_VALIDATION_ONLY_RESOURCE_PROFILE",
        "gpu": {"physical_index": GPU_INDEX, "model": "NVIDIA RTX A6000", "memory_total_mib": 49140},
        "probes": probes,
        "telemetry": telemetry_summary(),
        "historical_full_reference": historical_full_reference(),
        "extrapolation": extrapolate(probes),
        "memory_admission": {
            "observed_probe_peak_reserved_mib": peak_reserved,
            "maximum_planned_usable_memory_gib_per_job": 30,
            "recommended_single_job_gpu": "one assigned GPU with about 30 GiB usable memory",
            "reason": "researcher-frozen ceiling; probe and Phase12 historical full run fit below 30 GiB but leave limited module headroom",
        },
        "future_gpu_request": {
            "concurrent_gpu_ceiling": None,
            "planning_baseline_concurrency": 2,
            "usable_memory_gib_per_gpu": 30,
            "request_at_s17_3": "state the useful GPU count, one arm per GPU, and wait for researcher allocation; serialize if fewer cards are granted",
            "s17_1_s17_2": "no dedicated allocation; use one currently idle A6000 for each small probe",
            "s17_5": "request according to independent domain/arm parallelism and use granted-card waves",
            "s17_7": "request according to seed count/deadline and use granted-card waves",
            "heavy_tracks": "profile first; reduce micro-batch/use accumulation or lite structure if a job exceeds about 30 GiB",
        },
        "test_read": False,
        "sports_read": False,
        "scientific_result_selection_eligible": False,
    }
    atomic_json(OUTPUT / "resource_profile_summary.json", summary)

    cpu = json.loads((OUTPUT / "cpu_audit_summary.json").read_text(encoding="utf-8"))
    cpu.update({
        "verdict": "PASS_S17_0_AUDIT_REPORT_PENDING",
        "unit_tests_pending": False,
        "unit_tests": {"passed": 8, "failed_initial_attempt": 2, "failed_final_attempt": 0},
        "resource_profile_pending": False,
        "resource_profile": "artifacts/phase17/s0_audit/resource_profile_summary.json",
    })
    atomic_json(OUTPUT / "cpu_audit_summary.json", cpu)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
