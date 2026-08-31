#!/usr/bin/env python3
"""S17-0 source, historical evidence, data-projection, and lexical-contract audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "artifacts/phase17/s0_audit"
DEFAULT_STATUS = ROOT / "artifacts/phase17/status/s17_s0_audit.status.json"
FOLDS = {
    "D0": {"target_from_end": 5, "purpose": "discovery"},
    "D1": {"target_from_end": 4, "purpose": "independent_admission"},
    "D2": {"target_from_end": 3, "purpose": "fresh_confirmation"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest_path(path: Path) -> str:
    """Prefer stable repository-relative paths while supporting isolated tests."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row)
            handle.write("\n")
    os.replace(temporary, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def git_head(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def detect_license(path: Path) -> dict[str, Any]:
    candidates = sorted([*path.glob("LICENSE*"), *path.glob("COPYING*")]) if path.is_dir() else []
    if not candidates:
        return {"state": "NO_ROOT_LICENSE_FILE", "file": None, "sha256": None, "detected": None}
    license_path = candidates[0]
    text = license_path.read_text(encoding="utf-8", errors="replace")[:5000].lower()
    detected = "UNKNOWN"
    if "mit license" in text:
        detected = "MIT"
    elif "gnu general public license" in text and "version 3" in text:
        detected = "GPL-3.0"
    elif "apache license" in text and "version 2.0" in text:
        detected = "Apache-2.0"
    return {
        "state": "LOCAL_FILE_VERIFIED",
        "file": license_path.name,
        "sha256": sha256(license_path),
        "detected": detected,
    }


def audit_sources(registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for source in registry["sources"]:
        row = dict(source)
        local_dir_value = source.get("local_audit_dir")
        local_dir = Path(local_dir_value) if local_dir_value else None
        commit = git_head(local_dir) if local_dir else None
        key_files: dict[str, Any] = {}
        if local_dir and local_dir.is_dir():
            for relative in source.get("key_files", []):
                path = local_dir / relative
                key_files[relative] = {
                    "present": path.is_file(),
                    "sha256": sha256(path) if path.is_file() else None,
                }
        license_info = detect_license(local_dir) if local_dir else {
            "state": "NOT_LOCALLY_AUDITED",
            "file": None,
            "sha256": None,
            "detected": source.get("license_expected"),
        }
        expected = source.get("expected_commit")
        if commit and expected and commit != expected:
            verification = "COMMIT_MISMATCH"
        elif commit:
            verification = "LOCAL_COMMIT_AND_FILES_AUDITED"
        elif source.get("code") and expected:
            verification = "PRIMARY_URL_AND_REMOTE_HEAD_FROZEN_LOCAL_FILES_PENDING"
        elif source.get("code"):
            verification = "PRIMARY_URL_VERIFIED_LOCAL_COMMIT_PENDING"
        else:
            verification = "PAPER_IDEA_ONLY_NO_OFFICIAL_CODE"
        row["audit"] = {
            "verification": verification,
            "local_commit": commit,
            "expected_commit": expected,
            "commit_matches_expected": bool(commit and expected and commit == expected),
            "license": license_info,
            "key_files": key_files,
            "code_copy_allowed_by_s17": license_info.get("detected") in {"MIT", "Apache-2.0"},
            "license_reuse_action": (
                "COPY_WITH_ATTRIBUTION"
                if license_info.get("detected") in {"MIT", "Apache-2.0"}
                else "EXPLICIT_GPL_COMPATIBILITY_DECISION_REQUIRED"
                if license_info.get("detected") == "GPL-3.0"
                else "INDEPENDENT_REIMPLEMENTATION_ONLY"
            ),
            "independent_implementation_required": license_info["state"] != "LOCAL_FILE_VERIFIED",
        }
        rows.append(row)
    return {
        "schema_version": "phase17.source_manifest.v1",
        "generated_at_utc": now(),
        "registry_path": str(registry_path.relative_to(ROOT)),
        "registry_sha256": sha256(registry_path),
        "sources": rows,
        "counts": dict(Counter(row["audit"]["verification"] for row in rows)),
        "network_used_for_source_discovery": True,
        "third_party_code_executed": False,
    }


def parse_metric(text: str, split: str, metric: str) -> float | None:
    matches = re.findall(rf"^{re.escape(split)} {re.escape(metric)}:\s*([0-9.eE+-]+)$", text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def audit_phase12() -> dict[str, Any]:
    base = ROOT / "artifacts/phase12/hi_gram"
    runs: list[dict[str, Any]] = []
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        status_path = directory / "status.json"
        log_path = directory / "run.log"
        if not status_path.is_file() or not log_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        started_epochs = [int(value) for value in re.findall(r"Start training recommender for phase 1, epoch (\d+)", log_text)]
        completed = [(int(epoch), loss) for epoch, loss in re.findall(
            r"average training loss for rec phase 1 epoch (\d+) is ([^\s]+)", log_text
        )]
        planned_match = re.search(r"'rec_epochs':\s*(\d+)", log_text)
        model_epochs = sorted({
            int(match.group(1))
            for path in directory.rglob("model_rec_phase_1_epoch_*.pt")
            if (match := re.search(r"epoch_(\d+)\.pt$", path.name))
        })
        validation_predictions = sorted(directory.rglob("*_pred_validation.tsv"))
        test_predictions = sorted(directory.rglob("*_pred_test.tsv"))
        test_evidence = bool(test_predictions or re.search(r"^\[test\] testing", log_text, flags=re.MULTILINE))
        contradictions: list[str] = []
        if status.get("status") == "running" and not process_alive(int(status.get("runner_pid", 0))):
            contradictions.append("STATUS_RUNNING_BUT_RECORDED_RUNNER_DEAD")
        if status.get("test_read") is False and test_evidence:
            contradictions.append("STATUS_TEST_READ_FALSE_BUT_TEST_EVIDENCE_EXISTS")
        planned = int(planned_match.group(1)) if planned_match else None
        last_completed = max((epoch for epoch, _ in completed), default=0)
        if status.get("status") == "succeeded" and planned and last_completed < planned:
            contradictions.append("STATUS_SUCCEEDED_BUT_PLANNED_EPOCHS_INCOMPLETE")
        if status.get("status") == "succeeded" and started_epochs and last_completed < max(started_epochs):
            contradictions.append("STATUS_SUCCEEDED_WITH_INTERRUPTED_FINAL_EPOCH")
        runs.append({
            "run": directory.name,
            "status_path": str(status_path.relative_to(ROOT)),
            "status_sha256": sha256(status_path),
            "log_path": str(log_path.relative_to(ROOT)),
            "log_sha256": sha256(log_path),
            "declared_status": status.get("status"),
            "declared_test_read": status.get("test_read"),
            "recorded_runner_pid_alive_now": process_alive(int(status.get("runner_pid", 0))),
            "planned_epochs": planned,
            "last_started_epoch": max(started_epochs, default=0),
            "last_completed_epoch": last_completed,
            "last_completed_loss": completed[-1][1] if completed else None,
            "nan_loss_observed": any(loss.lower() == "nan" for _, loss in completed),
            "checkpoint_epochs": model_epochs,
            "validation_prediction_count": len(validation_predictions),
            "test_prediction_count": len(test_predictions),
            "test_prediction_paths": [str(path.relative_to(ROOT)) for path in test_predictions],
            "latest_validation": {
                "hit@10": parse_metric(log_text, "validation", "hit@10"),
                "ndcg@10": parse_metric(log_text, "validation", "ndcg@10"),
            },
            "latest_test": {
                "hit@10": parse_metric(log_text, "test", "hit@10"),
                "ndcg@10": parse_metric(log_text, "test", "ndcg@10"),
            },
            "contradictions": contradictions,
            "evidence_grade": "HISTORICAL_UNVERIFIED_SIGNAL" if contradictions else "HISTORICAL_AUDITED_ONLY",
        })
    return {
        "schema_version": "phase17.phase12_forensic.v1",
        "generated_at_utc": now(),
        "scope": "read-only audit of Phase12 statuses, logs, checkpoints, and prediction filenames",
        "runs": runs,
        "overall_verdict": "PHASE12_HI_GRAM_NOT_ADMISSIBLE_AS_CONFIRMED_BASELINE",
        "retrain_required_for_phase17_baseline": True,
        "sports_read": False,
    }


def parse_lexical_paths(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    item_to_path: dict[str, tuple[str, ...]] = {}
    path_to_item: dict[tuple[str, ...], str] = {}
    special_collisions: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed lexical row {line_number}: {path}")
        item, serialized = fields
        tokens = tuple(token for token in serialized.split("|") if token)
        if not tokens or item in item_to_path:
            raise ValueError(f"Empty path or duplicate item at row {line_number}: {path}")
        if tokens in path_to_item:
            raise ValueError(f"Lexical collision: {item} and {path_to_item[tokens]}")
        if any(token.startswith("<extra_id_") or token in {"<pad>", "</s>", "<unk>"} for token in tokens):
            special_collisions.append(item)
        item_to_path[item] = tokens
        path_to_item[tokens] = item
    strict_prefix_pairs = 0
    path_set = set(path_to_item)
    for tokens in path_set:
        if any(tokens[:length] in path_set for length in range(1, len(tokens))):
            strict_prefix_pairs += 1
    lengths = Counter(len(tokens) for tokens in item_to_path.values())
    return item_to_path, {
        "items": len(item_to_path),
        "unique_paths": len(path_to_item),
        "path_length_counts": {str(key): value for key, value in sorted(lengths.items())},
        "variable_length": len(lengths) > 1,
        "duplicate_item_count": 0,
        "duplicate_path_count": 0,
        "strict_prefix_path_count": strict_prefix_pairs,
        "t5_special_token_collision_count": len(special_collisions),
        "eos_serialized_in_path": False,
        "eos_appended_by_collator": True,
    }


def metadata_items(path: Path) -> set[str]:
    result: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"Malformed or duplicate metadata row {line_number}: {path}")
        result.add(fields[0])
    return result


def build_shadow_fold(
    source: Path,
    output: Path,
    target_from_end: int,
    catalog: set[str],
) -> dict[str, Any]:
    kept_rows: list[tuple[str, str]] = []
    seen_users: set[str] = set()
    excluded_short = 0
    source_rows = 0
    train_events = 0
    max_source_position_serialized = -1
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        source_rows += 1
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed sequence row {line_number}: {source}")
        user, items = fields[0], fields[1:]
        if user in seen_users:
            raise ValueError(f"Duplicate user {user}: {source}")
        seen_users.add(user)
        target_position = len(items) - target_from_end
        if target_position < 1:
            excluded_short += 1
            continue
        train_prefix = items[:target_position]
        target = items[target_position]
        guard_item = train_prefix[0]
        serialized_items = [*train_prefix, target, guard_item]
        unknown = set(serialized_items) - catalog
        if unknown:
            raise ValueError(f"Unknown item(s) at row {line_number}: {sorted(unknown)[:3]}")
        if serialized_items[:-2] != train_prefix or serialized_items[-2] != target:
            raise AssertionError("Shadow fold positional contract failed")
        kept_rows.append((user, " ".join([user, *serialized_items])))
        train_events += len(train_prefix)
        max_source_position_serialized = max(max_source_position_serialized, target_position)
    atomic_lines(output, (row for _, row in kept_rows))
    return {
        "source_path": manifest_path(source),
        "source_sha256": sha256(source),
        "output_path": manifest_path(output),
        "output_sha256": sha256(output),
        "source_users": source_rows,
        "output_users": len(kept_rows),
        "excluded_users_without_train_history": excluded_short,
        "train_prefix_item_occurrences": train_events,
        "target_from_original_end": target_from_end,
        "loader_train_slice": "shadow_items[:-2]",
        "loader_validation_position": "shadow_items[-2]",
        "loader_test_position": "train-prefix guard item; forbidden for evaluation",
        "guard_source_position": 0,
        "official_validation_position_serialized": False,
        "official_test_position_serialized": False,
        "official_heldout_values_logged": False,
        "target_in_train_by_position": False,
        "minimum_train_history": 1,
        "maximum_original_source_position_serialized": max_source_position_serialized,
    }


def build_profile_view(
    domain: str,
    fold_output: Path,
    dataset_dir: Path,
    size: int,
    output_root: Path,
) -> dict[str, Any]:
    rows = []
    for line in fold_output.read_text(encoding="utf-8").splitlines():
        user = line.split(maxsplit=1)[0]
        rows.append((stable_hash(f"phase17-profile|{domain}|{user}"), line))
    selected = [line for _, line in sorted(rows)[:size]]
    view_name = f"{domain}_s17_d0_{size}"
    target = output_root / "profile_data" / view_name
    atomic_lines(target / "user_sequence.txt", selected)
    copied: dict[str, str] = {}
    for source in sorted(dataset_dir.iterdir()):
        if not source.is_file() or source.name == "user_sequence.txt":
            continue
        destination = target / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied[source.name] = sha256(destination)
    return {
        "dataset_name": view_name,
        "users": len(selected),
        "path": str(target.relative_to(ROOT)),
        "user_sequence_sha256": sha256(target / "user_sequence.txt"),
        "copied_catalog_files": copied,
        "selection": "target-independent sha256(user_id) ordering",
        "test_position_semantics": "train-prefix guard only; no official heldout position",
    }


def build_data_and_lexical(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    domains: dict[str, Any] = {}
    lexical: dict[str, Any] = {}
    profile_views: dict[str, Any] = {}
    configs = {
        "Beauty": "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
        "Toys": "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
    }
    for domain, lexical_filename in configs.items():
        dataset_dir = ROOT / "GRAM/rec_datasets" / domain
        source = dataset_dir / "user_sequence.txt"
        lexical_path = dataset_dir / lexical_filename
        metadata_path = dataset_dir / "item_plain_text.txt"
        paths, lexical_audit = parse_lexical_paths(lexical_path)
        metadata = metadata_items(metadata_path)
        if set(paths) != metadata:
            raise ValueError(f"Catalog/metadata mismatch for {domain}")
        lexical[domain] = {
            **lexical_audit,
            "path_file": str(lexical_path.relative_to(ROOT)),
            "path_file_sha256": sha256(lexical_path),
            "metadata_file": str(metadata_path.relative_to(ROOT)),
            "metadata_file_sha256": sha256(metadata_path),
            "catalog_matches_metadata": True,
            "trie_legality_precondition": lexical_audit["duplicate_path_count"] == 0,
        }
        fold_rows: dict[str, Any] = {}
        for fold, spec in FOLDS.items():
            fold_output = output / "shadow_data" / domain / fold / "user_sequence.txt"
            fold_rows[fold] = build_shadow_fold(source, fold_output, spec["target_from_end"], set(paths))
        domains[domain] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "folds": fold_rows,
            "projection_only_job_opened_original_monolithic_sequence": True,
            "downstream_jobs_must_deny_original_monolithic_sequence": True,
            "sports_read": False,
        }
        if domain == "Toys":
            d0_path = output / "shadow_data" / domain / "D0" / "user_sequence.txt"
            profile_views["100"] = build_profile_view(domain, d0_path, dataset_dir, 100, output)
            profile_views["1000"] = build_profile_view(domain, d0_path, dataset_dir, 1000, output)
    return (
        {
            "schema_version": "phase17.shadow_data_manifest.v1",
            "generated_at_utc": now(),
            "projection_rule": "train-prefix + shadow validation target + train-prefix guard item",
            "folds": FOLDS,
            "domains": domains,
            "official_test_values_logged": False,
            "sports_opened": False,
        },
        {
            "schema_version": "phase17.lexical_contract.v1",
            "generated_at_utc": now(),
            "domains": lexical,
            "collator_path": "GRAM/src/processor/Collator.py",
            "collator_sha256": sha256(ROOT / "GRAM/src/processor/Collator.py"),
            "generation_trie_path": "GRAM/src/utils/generation_trie.py",
            "generation_trie_sha256": sha256(ROOT / "GRAM/src/utils/generation_trie.py"),
        },
        {
            "schema_version": "phase17.profile_views.v1",
            "generated_at_utc": now(),
            "views": profile_views,
        },
    )


def code_manifest() -> dict[str, Any]:
    paths = [
        ROOT / "experiment/phase17/protocol/s0_audit.py",
        ROOT / "experiment/phase17/protocol/finalize_s0_resource.py",
        ROOT / "experiment/phase17/run_stage17_s0_resource_profile.sh",
        ROOT / "experiment/phase17/tests/test_s0_audit.py",
        ROOT / "experiment/phase17/registry/source_registry.json",
        ROOT / "experiment/phase17/registry/idea_registry.yaml",
        ROOT / "experiment/phase17/schemas/status.schema.json",
        ROOT / "experiment/phase17/schemas/attempt.schema.json",
        ROOT / "experiment/phase17/schemas/migration_card.schema.json",
    ]
    paths.extend(sorted((ROOT / "experiment/phase17/registry/migration_cards").glob("*.yaml")))
    return {
        "schema_version": "phase17.code_manifest.v1",
        "generated_at_utc": now(),
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    }


def write_status(status_path: Path, state: str, execution: str, code: str, stage: str, output: Path) -> None:
    previous: dict[str, Any] = {}
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
    generated = now()
    payload = {
        "experiment_id": "s17_s0_audit",
        "step_id": "S17-0",
        "track_id": None,
        "scientific_state": state,
        "execution_state": execution,
        "status_code": code,
        "started_at": previous.get("started_at", generated),
        "updated_at": generated,
        "launcher_pid": os.getpid(),
        "workload_pid": os.getpid() if state == "RUNNING" else 0,
        "process_alive": state == "RUNNING",
        "tmux_session": None,
        "gpu_ids": [],
        "stage": stage,
        "progress": {"current": 5 if state == "RUNNING" else 6, "total": 6},
        "canonical_result_dir": str(output.relative_to(ROOT)),
        "log_path": None,
        "test_read": False,
        "sports_read": False,
        "result_selection_eligible": False,
        "occupancy_mode": "none",
    }
    atomic_json(status_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    status = args.status if args.status.is_absolute() else ROOT / args.status
    if (output / "cpu_audit_summary.json").exists():
        raise SystemExit("Refusing to overwrite completed S17-0 CPU audit")
    output.mkdir(parents=True, exist_ok=True)
    write_status(status, "RUNNING", "RUNNING_SCIENTIFIC", "S17_0_CPU_AUDIT", "source_and_data_audit", output)
    try:
        source_manifest = audit_sources(ROOT / "experiment/phase17/registry/source_registry.json")
        phase12 = audit_phase12()
        data_manifest, lexical_contract, profile_views = build_data_and_lexical(output)
        code = code_manifest()
        atomic_json(output / "source_manifest.json", source_manifest)
        atomic_json(output / "phase12_forensic_audit.json", phase12)
        atomic_json(output / "shadow_data_manifest.json", data_manifest)
        atomic_json(output / "lexical_contract.json", lexical_contract)
        atomic_json(output / "profile_views.json", profile_views)
        atomic_json(output / "code_manifest.json", code)
        opened_files = [
            "GRAM/rec_datasets/Beauty/user_sequence.txt",
            "GRAM/rec_datasets/Toys/user_sequence.txt",
            "GRAM/rec_datasets/Beauty/item_plain_text.txt",
            "GRAM/rec_datasets/Toys/item_plain_text.txt",
            "GRAM/rec_datasets/Beauty/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
            "GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
        ]
        opened_files.extend(
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "artifacts/phase12/hi_gram").glob("*/status.json"))
        )
        opened_files.extend(
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "artifacts/phase12/hi_gram").glob("*/run.log"))
        )
        atomic_json(output / "open_file_manifest.json", {
            "schema_version": "phase17.open_file_manifest.v1",
            "generated_at_utc": now(),
            "opened_files": opened_files,
            "phase12_prediction_contents_opened": False,
            "phase12_prediction_filenames_audited": True,
            "sports_opened": False,
            "official_heldout_values_logged": False,
        })
        summary = {
            "schema_version": "phase17.s0_cpu_audit_summary.v1",
            "generated_at_utc": now(),
            "verdict": "PASS_S17_0_CPU_AUDIT_RESOURCE_PROFILE_PENDING",
            "source_manifest": "artifacts/phase17/s0_audit/source_manifest.json",
            "phase12_verdict": phase12["overall_verdict"],
            "shadow_data_manifest": "artifacts/phase17/s0_audit/shadow_data_manifest.json",
            "lexical_contract": "artifacts/phase17/s0_audit/lexical_contract.json",
            "profile_views": "artifacts/phase17/s0_audit/profile_views.json",
            "source_local_audit_counts": source_manifest["counts"],
            "domains": sorted(data_manifest["domains"]),
            "folds": sorted(FOLDS),
            "unit_tests_pending": True,
            "resource_profile_pending": True,
            "scientific_results_produced": False,
            "test_read": False,
            "sports_read": False,
        }
        atomic_json(output / "cpu_audit_summary.json", summary)
        write_status(status, "RUNNING", "RUNNING_SCIENTIFIC", "S17_0_RESOURCE_PROFILE_PENDING", "resource_profile_pending", output)
        print(summary["verdict"])
        return 0
    except Exception as exc:
        atomic_json(output / "cpu_audit_failure.json", {"generated_at_utc": now(), "error": repr(exc)})
        write_status(status, "FAILED", "SCIENTIFIC_FAILED", "S17_0_CPU_AUDIT_FAILED", "failed", output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
