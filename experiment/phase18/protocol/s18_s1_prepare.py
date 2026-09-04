#!/usr/bin/env python3
"""CPU-only preparation for the Stage18 S18-1 actionability diagnostic.

This command freezes fold-local materialized views and resource estimates.  It
does not initialize a model, read any S18 confirmation fold, or start the
scientific GPU attempt.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment.phase18.core.contracts import (
    ROOT,
    authorize_path,
    load_json,
    load_shadow_train_prefix_line,
    sha256,
)
from experiment.phase18.core.s1_contracts import (
    S18_1_FOLDS,
    cohort_sha256,
    fold_views,
    lower_empirical_quartile,
    stable_cohort,
)


DEFAULT_CONFIG = ROOT / "experiment/phase18/config/s18_s1_actionability.json"
DEFAULT_OUTPUT = ROOT / "artifacts/phase18/s1_actionability/preflight"
DEFAULT_STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability.status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def verify_prerequisites(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for name, record in config["prerequisites"].items():
        path = ROOT / record["path"]
        actual = sha256(path) if path.is_file() else None
        checks.append(
            {
                "id": f"prerequisite_hash:{name}",
                "passed": actual == record["sha256"],
                "path": record["path"],
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
            }
        )
    s0 = load_json(ROOT / config["prerequisites"]["s18_s0_summary"]["path"])
    checks.append(
        {
            "id": "s18_s0_gate",
            "passed": s0.get("gate", {}).get("status")
            == config["prerequisites"]["s18_s0_summary"]["required_status"],
            "observed": s0.get("gate", {}).get("status"),
        }
    )
    return checks


def verify_backbone(config: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = Path(config["backbone"]["snapshot"])
    checks = []
    for name, expected in config["backbone"]["files"].items():
        path = snapshot / name
        actual = sha256(path) if path.is_file() else None
        checks.append(
            {
                "id": f"backbone_hash:{name}",
                "passed": actual == expected,
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    return checks


def load_histories(relative: str, expected_hash: str, data_contract: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    authorized = authorize_path(relative, "s18_internal_runner", data_contract)
    path = ROOT / authorized
    if sha256(path) != expected_hash:
        raise RuntimeError(f"D0 shadow input hash mismatch: {relative}")
    histories: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            user, history = load_shadow_train_prefix_line(line)
            if user in histories:
                raise ValueError(f"duplicate user in {relative}: {user}")
            histories[user] = history
    return histories


def verify_assets(domain: str, domain_config: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    source_dir = ROOT / "GRAM/rec_datasets" / domain
    checks = []
    for name, expected in domain_config["assets"].items():
        path = source_dir / name
        actual = sha256(path) if path.is_file() else None
        checks.append(
            {
                "id": f"catalog_hash:{domain}:{name}",
                "passed": actual == expected,
                "path": str(path.relative_to(ROOT)),
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    return checks, source_dir


def write_dataset_view(
    target_dir: Path,
    source_dir: Path,
    views: dict[str, tuple[tuple[str, ...], str]],
    asset_names: list[str],
) -> dict[str, Any]:
    target_dir.mkdir(parents=True)
    rows = []
    for user in sorted(views):
        visible, target = views[user]
        guard = visible[0]
        rows.append(" ".join((user, *visible, target, guard)))
    atomic_text(target_dir / "user_sequence.txt", "\n".join(rows) + "\n")
    files = {
        "user_sequence.txt": {
            "sha256": sha256(target_dir / "user_sequence.txt"),
            "size_bytes": (target_dir / "user_sequence.txt").stat().st_size,
        }
    }
    for name in asset_names:
        shutil.copy2(source_dir / name, target_dir / name)
        files[name] = {
            "sha256": sha256(target_dir / name),
            "size_bytes": (target_dir / name).stat().st_size,
        }
    return {"path": str(target_dir.relative_to(ROOT)), "files": files}


def source_manifest(config_path: Path) -> dict[str, str]:
    paths = [
        config_path,
        ROOT / "plan/第十八阶段/GRAM_第十八阶段_S18-1可作用性诊断执行补遗v0.1.md",
        ROOT / "experiment/phase18/core/contracts.py",
        ROOT / "experiment/phase18/core/s1_contracts.py",
        ROOT / "experiment/phase18/protocol/s18_s1_prepare.py",
        ROOT / "experiment/phase18/tests/test_s18_s1_contract.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def runtime_estimate(domain: str, train_samples: int) -> dict[str, float]:
    # Frozen from already-completed historical logs; used only for scheduling.
    reference = {
        "Toys": {"samples": 6836 * 16, "epoch_seconds": 48 * 60, "generation_seconds": 1288.4},
        "Beauty": {"samples": 8214 * 16, "epoch_seconds": 63 * 60, "generation_seconds": 750.3},
    }[domain]
    epoch_seconds = reference["epoch_seconds"] * train_samples / reference["samples"]
    parent_seconds = epoch_seconds * 10
    # Include item-head, materialization and reporting margin without pretending
    # this is a measured S18 runtime.
    total_seconds = parent_seconds + reference["generation_seconds"] + 15 * 60
    return {
        "parent_epoch_seconds_scaled": epoch_seconds,
        "parent_10_epoch_hours_scaled": parent_seconds / 3600,
        "beam50_200_seconds_historical_1024": reference["generation_seconds"],
        "unit_total_hours_with_15min_overhead": total_seconds / 3600,
    }


def prepare(config_path: Path, output: Path, status_path: Path) -> dict[str, Any]:
    if output.exists() or status_path.exists():
        raise FileExistsError("S18-1 preflight already exists; overwrite/retry is forbidden")
    config = load_json(config_path)
    data_contract = load_json(ROOT / config["prerequisites"]["s18_data_contract"]["path"])
    checks = verify_prerequisites(config) + verify_backbone(config)
    if not all(check["passed"] for check in checks):
        raise RuntimeError("S18-1 prerequisite/backbone verification failed")

    output.mkdir(parents=True)
    domains: dict[str, Any] = {}
    for domain, domain_config in config["domains"].items():
        asset_checks, source_dir = verify_assets(domain, domain_config)
        checks.extend(asset_checks)
        if not all(check["passed"] for check in asset_checks):
            raise RuntimeError(f"{domain}: catalog asset verification failed")
        histories = load_histories(
            domain_config["shadow_path"], domain_config["shadow_sha256"], data_contract
        )
        views_by_fold = {fold: fold_views(histories, fold) for fold in S18_1_FOLDS}
        common = set.intersection(*(set(views) for views in views_by_fold.values()))
        selected = stable_cohort(
            domain, common, config["cohort"]["users_per_domain"], config["seed"]
        )
        observed_cohort_sha = cohort_sha256(selected)
        expected_cohort = config["cohort"]["domains"][domain]
        checks.extend(
            [
                {
                    "id": f"eligible_intersection:{domain}",
                    "passed": len(common) == expected_cohort["eligible_intersection"],
                    "expected": expected_cohort["eligible_intersection"],
                    "actual": len(common),
                },
                {
                    "id": f"cohort_sha256:{domain}",
                    "passed": observed_cohort_sha == expected_cohort["sample_sha256"],
                    "expected": expected_cohort["sample_sha256"],
                    "actual": observed_cohort_sha,
                },
            ]
        )
        atomic_text(output / f"cohort_{domain}.txt", "\n".join(selected) + "\n")
        fold_rows: dict[str, Any] = {}
        for fold, views in views_by_fold.items():
            frequencies = Counter(item for visible, _ in views.values() for item in visible)
            train_samples = sum(max(0, len(visible) - 1) for visible, _ in views.values())
            dataset_name = f"{domain}_s18_s1_{fold.replace('-', 'm')}"
            dataset = write_dataset_view(
                output / "data" / dataset_name,
                source_dir,
                views,
                list(domain_config["assets"]),
            )
            fold_rows[fold] = {
                "dataset_name": dataset_name,
                "eligible_users": len(views),
                "diagnostic_users": len(selected),
                "visible_item_occurrences": sum(len(visible) for visible, _ in views.values()),
                "parent_augmented_train_samples": train_samples,
                "positive_frequency_catalog_items": len(frequencies),
                "q1": lower_empirical_quartile(frequencies.values()),
                "dataset": dataset,
                "runtime_estimate": runtime_estimate(domain, train_samples),
            }
        domains[domain] = {
            "source_users": len(histories),
            "eligible_intersection": len(common),
            "cohort_path": str((output / f"cohort_{domain}.txt").relative_to(ROOT)),
            "cohort_sha256": observed_cohort_sha,
            "folds": fold_rows,
        }

    if not all(check["passed"] for check in checks):
        raise RuntimeError("S18-1 preflight contract verification failed")
    manifest = {
        "schema_version": "phase18.s18_1_preflight.v1",
        "experiment_id": "s18_s1_actionability",
        "scientific_attempt_started": False,
        "created_at": utc_now(),
        "git_head": git_head(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "source_manifest": source_manifest(config_path),
        "checks": checks,
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "domains": domains,
        "protected_data": {
            "d1_read": False,
            "d2_read": False,
            "official_validation_test_read": False,
            "sports_read": False,
            "stage17_external_d0_target_returned_by_loader": False,
            "i1_i2_views_constructed": False,
        },
        "resource_decision": {
            "state": "AWAITING_RESOURCE_AUTHORIZATION",
            **config["resources"],
        },
    }
    atomic_json(output / "manifest.json", manifest)
    status = {
        "schema_version": "phase18.status.v1",
        "experiment_id": "s18_s1_actionability",
        "step_id": "S18-1",
        "stage": "s18_1_actionability_preflight",
        "execution_state": "PREFLIGHT_COMPLETED_AWAITING_RESOURCES",
        "scientific_state": "NOT_STARTED",
        "status_code": "S18_1_AWAITING_RESOURCE_AUTHORIZATION",
        "scientific_attempt_started": False,
        "affects_scientific_result": False,
        "result_selection_eligible": False,
        "gpu_ids": [],
        "tmux_session": None,
        "workload_pid": 0,
        "process_alive": False,
        "progress": {"current": 1, "total": 1, "unit": "cpu_preflight"},
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "manifest_path": str((output / "manifest.json").relative_to(ROOT)),
        "manifest_sha256": sha256(output / "manifest.json"),
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
        "heartbeat_at": utc_now(),
        "updated_at": utc_now(),
        "next_action": "Researcher authorizes 2 or 4 GPUs with >=30 GiB usable memory; then start one named background tmux attempt.",
    }
    atomic_json(status_path, status)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    result = prepare(args.config.resolve(), args.output.resolve(), args.status.resolve())
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_COMPLETED_AWAITING_RESOURCES",
                "checks": f"{result['checks_passed']}/{result['checks_total']}",
                "status_path": str(args.status.resolve().relative_to(ROOT)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
