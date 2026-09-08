#!/usr/bin/env python3
"""CPU-only preflight for the S18-1R-B disjoint confirmation cohort."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment.phase18.core.contracts import (
    ROOT,
    authorize_path,
    load_json,
    load_shadow_train_prefix_line,
    normalize_repo_relative,
    sha256,
)
from experiment.phase18.core.s1_contracts import (
    S18_1_FOLDS,
    cohort_sha256,
    fold_views,
    stable_cohort_slice,
)


CONFIG = ROOT / "experiment/phase18/config/s18_s1r_disjoint_confirmation.json"
OUTPUT = ROOT / "artifacts/phase18/s1r_disjoint_confirmation/preflight"
STATUS = ROOT / "artifacts/phase18/status/s18_s1r_disjoint_confirmation.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1R.attempts.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def append_ledger(payload: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verified_path(record: dict[str, str]) -> Path:
    relative = normalize_repo_relative(record["path"])
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    actual = sha256(path)
    if actual != record["sha256"]:
        raise RuntimeError(
            f"frozen input SHA mismatch for {relative}: expected {record['sha256']}, got {actual}"
        )
    return path


def load_histories(
    record: dict[str, Any], data_contract: dict[str, Any]
) -> dict[str, tuple[str, ...]]:
    relative = authorize_path(record["shadow_path"], "s18_internal_runner", data_contract)
    path = ROOT / relative
    if sha256(path) != record["shadow_sha256"]:
        raise RuntimeError(f"D0 shadow input SHA mismatch: {relative}")
    histories: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            user, history = load_shadow_train_prefix_line(line)
            if user in histories:
                raise ValueError(f"duplicate user in {relative}: {user}")
            histories[user] = history
    return histories


def verify_dataset_files(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for domain in ("Toys", "Beauty"):
        for fold in S18_1_FOLDS:
            dataset = preflight["domains"][domain]["folds"][fold]["dataset"]
            directory = ROOT / normalize_repo_relative(dataset["path"])
            for name, expected in dataset["files"].items():
                path = directory / name
                actual = sha256(path) if path.is_file() else None
                checks.append(
                    {
                        "id": f"dataset:{domain}:{fold}:{name}",
                        "passed": actual == expected["sha256"],
                        "path": str(path.relative_to(ROOT)),
                        "expected_sha256": expected["sha256"],
                        "actual_sha256": actual,
                    }
                )
    return checks


def prepare() -> dict[str, Any]:
    if OUTPUT.exists() or STATUS.exists():
        raise FileExistsError("S18-1R-B preflight already exists; automatic rerun is forbidden")
    config = load_json(CONFIG)
    frozen = config["frozen_inputs"]
    original_config_path = verified_path(frozen["s18_s1_config"])
    original_preflight_path = verified_path(frozen["s18_s1_preflight"])
    retrospective_path = verified_path(frozen["s18_s1r_retrospective"])
    checkpoint_auth_path = verified_path(frozen["checkpoint_authorization"])
    original_config = load_json(original_config_path)
    original_preflight = load_json(original_preflight_path)
    retrospective = load_json(retrospective_path)
    checkpoint_auth = load_json(checkpoint_auth_path)
    if retrospective.get("decision") != frozen["s18_s1r_retrospective"]["required_decision"]:
        raise RuntimeError("S18-1R retrospective prerequisite did not pass")
    boundary = config["authorization_boundary"]
    required_false = (
        "gpu_confirmation_automatic",
        "parent_retraining",
        "item_head_retraining",
        "treatment_training",
        "automatic_retry",
        "automatic_s18_2",
        "i1_i2_allowed",
        "d1_d2_allowed",
        "official_validation_test_allowed",
        "sports_allowed",
    )
    if not boundary["cpu_preflight_allowed"] or any(boundary[name] for name in required_false):
        raise RuntimeError("S18-1R-B preflight authorization boundary is not fail-closed")
    if checkpoint_auth.get("checkpoints") != config["checkpoints"]:
        raise RuntimeError("checkpoint records differ from the frozen recovery authorization")

    data_contract = load_json(
        ROOT / original_config["prerequisites"]["s18_data_contract"]["path"]
    )
    checks = verify_dataset_files(original_preflight)
    domains: dict[str, Any] = {}
    original_start, original_stop = config["cohort"]["original_slice"]
    confirm_start, confirm_stop = config["cohort"]["confirmation_slice"]
    count = int(config["cohort"]["users_per_domain"])
    if original_stop - original_start != count or confirm_stop - confirm_start != count:
        raise ValueError("cohort slice width does not match users_per_domain")
    if original_stop > confirm_start:
        raise ValueError("original and confirmation slice declarations overlap")

    for domain in ("Toys", "Beauty"):
        record = config["cohort"]["domains"][domain]
        histories = load_histories(record, data_contract)
        views = {fold: fold_views(histories, fold) for fold in S18_1_FOLDS}
        common = set.intersection(*(set(rows) for rows in views.values()))
        original = stable_cohort_slice(
            domain, common, original_start, count, config["cohort"]["seed"]
        )
        confirmation = stable_cohort_slice(
            domain, common, confirm_start, count, config["cohort"]["seed"]
        )
        original_hash = cohort_sha256(original)
        confirmation_hash = cohort_sha256(confirmation)
        old_path = ROOT / original_preflight["domains"][domain]["cohort_path"]
        old_file_hash = sha256(old_path)
        domain_checks = [
            {
                "id": f"eligible_intersection:{domain}",
                "passed": len(common) == record["eligible_intersection"],
                "expected": record["eligible_intersection"],
                "actual": len(common),
            },
            {
                "id": f"original_recomputed_sha256:{domain}",
                "passed": original_hash == record["original_sha256"],
                "expected": record["original_sha256"],
                "actual": original_hash,
            },
            {
                "id": f"original_file_sha256:{domain}",
                "passed": old_file_hash == record["original_sha256"],
                "expected": record["original_sha256"],
                "actual": old_file_hash,
            },
            {
                "id": f"confirmation_sha256:{domain}",
                "passed": confirmation_hash == record["confirmation_sha256"],
                "expected": record["confirmation_sha256"],
                "actual": confirmation_hash,
            },
            {
                "id": f"zero_user_overlap:{domain}",
                "passed": not (set(original) & set(confirmation)),
                "actual_overlap": len(set(original) & set(confirmation)),
            },
        ]
        checks.extend(domain_checks)
        domains[domain] = {
            "source_users": len(histories),
            "eligible_intersection": len(common),
            "original_cohort_sha256": original_hash,
            "confirmation_cohort_sha256": confirmation_hash,
            "original_confirmation_overlap": len(set(original) & set(confirmation)),
            "confirmation_users": len(confirmation),
            "fold_eligible_users": {fold: len(rows) for fold, rows in views.items()},
            "folds": {
                fold: {
                    "dataset_name": original_preflight["domains"][domain]["folds"][fold]["dataset_name"],
                    "q1": original_preflight["domains"][domain]["folds"][fold]["q1"],
                    "dataset": original_preflight["domains"][domain]["folds"][fold]["dataset"],
                }
                for fold in S18_1_FOLDS
            },
        }

    for unit, roles in config["checkpoints"].items():
        for role, record in roles.items():
            path = ROOT / normalize_repo_relative(record["path"])
            actual = sha256(path) if path.is_file() else None
            checks.append(
                {
                    "id": f"checkpoint:{unit}:{role}",
                    "passed": actual == record["sha256"],
                    "path": record["path"],
                    "expected_sha256": record["sha256"],
                    "actual_sha256": actual,
                }
            )
    if not all(check["passed"] for check in checks):
        failed = [check["id"] for check in checks if not check["passed"]]
        raise RuntimeError(f"S18-1R-B preflight checks failed: {failed}")

    OUTPUT.mkdir(parents=True)
    for domain in ("Toys", "Beauty"):
        record = config["cohort"]["domains"][domain]
        histories = load_histories(record, data_contract)
        views = {fold: fold_views(histories, fold) for fold in S18_1_FOLDS}
        common = set.intersection(*(set(rows) for rows in views.values()))
        confirmation = stable_cohort_slice(
            domain, common, confirm_start, count, config["cohort"]["seed"]
        )
        cohort_path = OUTPUT / f"cohort_{domain}.txt"
        atomic_text(cohort_path, "\n".join(confirmation) + "\n")
        domains[domain]["confirmation_cohort_path"] = str(cohort_path.relative_to(ROOT))

    created = utc_now()
    manifest = {
        "schema_version": "phase18.s18_1r_disjoint_preflight.v1",
        "experiment_id": config["experiment_id"],
        "step_id": config["step_id"],
        "execution_state": "PREFLIGHT_COMPLETED_AWAITING_RESOURCE_AUTHORIZATION",
        "scientific_attempt_started": False,
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "frozen_inputs": frozen,
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "domains": domains,
        "checkpoints": config["checkpoints"],
        "protected_data": {
            "i1_i2_read": False,
            "d1_read": False,
            "d2_read": False,
            "official_validation_test_read": False,
            "sports_read": False,
        },
        "parent_retraining": False,
        "item_head_retraining": False,
        "treatment_training": False,
        "automatic_s18_2": False,
        "created_at": created,
    }
    atomic_json(OUTPUT / "manifest.json", manifest)
    status = {
        "schema_version": "phase18.status.v1",
        "experiment_id": config["experiment_id"],
        "attempt_id": "preflight-0001",
        "step_id": config["step_id"],
        "stage": "s18_1r_disjoint_preflight",
        "execution_state": "PREFLIGHT_COMPLETED_AWAITING_RESOURCE_AUTHORIZATION",
        "scientific_state": "NOT_STARTED",
        "status_code": "S18_1R_DISJOINT_AWAITING_RESOURCE_AUTHORIZATION",
        "scientific_attempt_started": False,
        "affects_scientific_result": False,
        "result_selection_eligible": False,
        "gpu_ids": [],
        "tmux_session": None,
        "workload_pid": 0,
        "process_alive": False,
        "progress": {"current": 1, "total": 1, "unit": "cpu_preflight"},
        "manifest_path": str((OUTPUT / "manifest.json").relative_to(ROOT)),
        "manifest_sha256": sha256(OUTPUT / "manifest.json"),
        "automatic_retry": False,
        "automatic_s18_2": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "heartbeat_at": created,
        "updated_at": created,
        "next_action": "Authorize a named checkpoint-only GPU confirmation; do not start S18-2.",
    }
    atomic_json(STATUS, status)
    append_ledger(
        {
            "at": created,
            "attempt_id": "preflight-0001",
            "event": "disjoint_confirmation_preflight_completed",
            "manifest_sha256": status["manifest_sha256"],
            "automatic_s18_2": False,
        }
    )
    return manifest


def main() -> int:
    manifest = prepare()
    print(
        json.dumps(
            {
                "status": manifest["execution_state"],
                "checks": f"{manifest['checks_passed']}/{manifest['checks_total']}",
                "manifest": str((OUTPUT / "manifest.json").relative_to(ROOT)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
