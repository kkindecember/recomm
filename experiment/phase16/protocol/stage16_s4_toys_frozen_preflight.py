#!/usr/bin/env python3
"""Freeze Stage16 S16-4 Toys inputs, source Gates, arms, and statistics.

This is deliberately CPU-only.  It opens the already authorized validation
projection only to verify identity, membership, and event counts; it computes
no ranking metric and cannot unlock the GPU evaluation by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_ARMS = ("F0", "R2", "S-AUX", "S-PLUS-CTRL", "S-PLUS", "G-RIDGE")
EXPECTED_CONTROLS = {
    "F0": None,
    "R2": "F0",
    "S-AUX": "F0",
    "S-PLUS-CTRL": "F0",
    "S-PLUS": "S-PLUS-CTRL",
    "G-RIDGE": "F0",
}
CODE_FILES = (
    "experiment/phase16/protocol/stage16_s4_toys_frozen_preflight.py",
    "experiment/phase16/tests/test_stage16_s4_toys_frozen_preflight.py",
    "experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_regular_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Frozen input must be a regular non-symlink file: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Frozen input escapes repository root: {relative}") from error
    return resolved


def validate_config_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "stage16_s4_toys_frozen_preflight_v1":
        raise ValueError("Unexpected S16-4 preflight schema")
    if config.get("seed") != 1502 or config.get("domain") != "Toys_cold50":
        raise ValueError("S16-4 seed/domain drift")
    if config.get("split") != "validation" or config.get("test_read") is True:
        raise ValueError("S16-4 preflight may use validation only; test remains sealed")

    arms = config.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != REQUIRED_ARMS:
        raise ValueError(f"S16-4 arm order/set drift: {tuple(arms or {})}")
    observed_controls = {name: arms[name].get("control") for name in REQUIRED_ARMS}
    if observed_controls != EXPECTED_CONTROLS:
        raise ValueError(f"S16-4 matched-control drift: {observed_controls}")

    for name, draft_size in (("S-AUX", 50), ("S-PLUS", 20)):
        arm = arms[name]
        if arm.get("faithful_reproduction") is not True:
            raise ValueError(f"{name} must remain a faithful SpecGR arm")
        if arm.get("draft_size") != draft_size or arm.get("threshold") != -1.8:
            raise ValueError(f"{name} faithful draft-size/threshold drift")
        if arm.get("acceptance") != "strict_score_greater_than_threshold":
            raise ValueError(f"{name} must use strict > acceptance")
        if arm.get("guided_redraft") != "current_live_verifier_beam_prefixes":
            raise ValueError(f"{name} must use live verifier beam prefixes")

    excluded = config.get("excluded_arms", {})
    if not all(name in excluded for name in ("G-FULL", "Stage15-B2", "Stage15-B3")):
        raise ValueError("Historical blocked/pilot arms must be explicitly excluded")

    evaluation = config.get("evaluation_contract", {})
    expected = {
        "beam_size": 50,
        "paired_bootstrap_resamples": 10_000,
        "paired_bootstrap_confidence": 0.95,
        "paired_bootstrap_seed": 20260822,
        "gate_cold_signal": "paired_bootstrap_95pct_ci_low_of_cold_hit50_gain_gt_0",
    }
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError("S16-4 evaluator/statistical contract drift")
    if evaluation.get("validation_may_not_change_method_or_threshold") is not True:
        raise ValueError("Validation must not tune method or threshold")
    if evaluation.get("test_read") is not False:
        raise ValueError("Test must remain sealed")

    launch = config.get("launch_contract", {})
    if launch.get("gpu_launch_ready_after_this_preflight") is not False:
        raise ValueError("CPU input freeze must not authorize GPU launch")
    if launch.get("automatic_retry") is not False or launch.get("test_read") is not False:
        raise ValueError("Automatic retry/test read must remain disabled")


def read_item_set(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"Empty frozen item set: {path}")
    return values


def read_key_universe(path: Path, *, separator: str | None = None) -> set[str]:
    values: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            key = line.split(separator, 1)[0].strip() if separator else line.split()[0]
            if not key or key in values:
                raise ValueError(f"Duplicate/empty item key at {path}:{line_number}")
            values.add(key)
    return values


def validate_source_gates(paths: Mapping[str, Path]) -> dict[str, Any]:
    saux_status = load_json(paths["saux_status"])
    saux = load_json(paths["saux_summary"])
    saux_contract = load_json(paths["saux_artifact_contract"])
    if saux_status.get("status_code") != "COMPLETED" or saux_status.get("exit_code") != 0:
        raise ValueError("S-AUX terminal status is not completed/exit 0")
    if saux.get("verdict") != "PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION":
        raise ValueError("S-AUX faithful Gate is not PASS")
    if saux_contract.get("verdict") != "PASS_SAUX_FORMAL_ARTIFACT_CONTRACT":
        raise ValueError("S-AUX artifact contract is not PASS")
    if saux.get("test_read") is not False or "validation" in str(saux.get("scientific_scope", "")) and "no source validation" not in str(saux.get("scientific_scope", "")):
        raise ValueError("S-AUX source-set boundary is not sealed")

    pair_status = load_json(paths["splus_ctrl_pair_status"])
    pair = load_json(paths["splus_ctrl_pair_summary"])
    pair_contract = load_json(paths["splus_ctrl_pair_artifact_contract"])
    splus = load_json(paths["splus_summary"])
    control = load_json(paths["splus_ctrl_summary"])
    if pair_status.get("status_code") != "COMPLETED" or pair_status.get("exit_code") != 0:
        raise ValueError("S-PLUS/CTRL pair terminal status is not completed/exit 0")
    if pair.get("verdict") != "PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION":
        raise ValueError("S-PLUS faithful Gate is not PASS")
    if pair.get("control_execution_verdict") != "PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION":
        raise ValueError("S-PLUS matched control Gate is not PASS")
    if pair_contract.get("verdict") != "PASS_SPLUS_CTRL_SPLIT_PAIR_ARTIFACT_CONTRACT":
        raise ValueError("S-PLUS/CTRL pair artifact contract is not PASS")
    if splus.get("verdict") != "PASS_S16_2_S_PLUS_FORMAL_EXECUTION":
        raise ValueError("S-PLUS source arm is not PASS")
    if control.get("verdict") != "PASS_S16_2_S_PLUS_CTRL_FORMAL_EXECUTION":
        raise ValueError("S-PLUS-CTRL source arm is not PASS")
    if any(payload.get("test_read") is not False for payload in (pair, splus, control)):
        raise ValueError("S-PLUS/CTRL source test boundary is not sealed")
    if any(payload.get("validation_used") is not False for payload in (pair, splus, control)):
        raise ValueError("S-PLUS/CTRL source validation boundary is not sealed")
    if not splus.get("base_checkpoint_unchanged") or not control.get("base_checkpoint_unchanged"):
        raise ValueError("S-PLUS/CTRL source base-checkpoint parity failed")

    gridge_status = load_json(paths["gridge_status"])
    gridge = load_json(paths["gridge_formal_summary"])
    manifest = load_json(paths["gridge_checkpoint_manifest"])
    if gridge_status.get("status") != "COMPLETED" or gridge_status.get("exit_code") != 0:
        raise ValueError("G-RIDGE terminal status is not completed/exit 0")
    if gridge_status.get("status_code") != "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION":
        raise ValueError("G-RIDGE terminal Gate is not PASS")
    if gridge.get("formal_gate") != "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION":
        raise ValueError("G-RIDGE formal summary Gate is not PASS")
    if gridge.get("method", {}).get("faithful_reproduction") is not False:
        raise ValueError("G-RIDGE must remain explicitly non-faithful")
    if not manifest.get("state_frozen") or manifest.get("completed_positions") != 6:
        raise ValueError("G-RIDGE checkpoint state is not frozen/complete")
    if manifest.get("held_ground_truth_opened") is not False:
        raise ValueError("G-RIDGE admission opened held validation/test ground truth")

    return {
        "S-AUX": saux.get("verdict"),
        "S-PLUS": pair.get("verdict"),
        "S-PLUS-CTRL": pair.get("control_execution_verdict"),
        "G-RIDGE": gridge.get("formal_gate"),
        "all_source_tests_sealed": True,
        "all_source_validation_unused": True,
    }


def verify_gridge_manifest(root: Path, manifest_path: Path, aggregate_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    required = {"full_covariance", "aggregate_deltas"} | {
        f"position_{position}_delta" for position in range(6)
    }
    if set(artifacts) != required:
        raise ValueError(f"G-RIDGE checkpoint-manifest artifact drift: {set(artifacts)}")
    observed: dict[str, Any] = {}
    for name in sorted(required):
        declaration = artifacts[name]
        path = resolve_regular_file(root, str(declaration["path"]))
        digest = sha256_file(path)
        if digest != declaration.get("sha256"):
            raise ValueError(f"G-RIDGE checkpoint SHA drift: {name}")
        observed[name] = {
            "path": str(path.relative_to(root)),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    if observed["aggregate_deltas"]["path"] != str(aggregate_path.relative_to(root)):
        raise ValueError("Configured G-RIDGE aggregate delta differs from frozen manifest")
    return observed


def validate_validation_identity(
    paths: Mapping[str, Path], expected: Mapping[str, int]
) -> dict[str, Any]:
    projected: dict[str, list[str]] = {}
    with paths["projected_train_validation_sequences"].open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            fields = raw.strip().split()
            if len(fields) < 3:
                raise ValueError(f"Projected row too short at line {line_number}")
            user, items = fields[0], fields[1:]
            if user in projected:
                raise ValueError(f"Duplicate projected user: {user}")
            projected[user] = items

    cold = read_item_set(paths["cold_items"])
    warm = read_item_set(paths["warm_items"])
    lexical = read_key_universe(paths["lexical_paths"], separator="|")
    metadata = read_key_universe(paths["item_metadata"])
    if cold & warm:
        raise ValueError("Frozen cold/warm catalog partitions overlap")
    if cold | warm != lexical or metadata != lexical:
        raise ValueError("Cold/warm/path/metadata catalog universes differ")
    if len(cold) != expected["cold_catalog_items"] or len(warm) != expected["warm_catalog_items"]:
        raise ValueError("Frozen cold/warm catalog counts drifted")
    if len(lexical) != expected["catalog_items"]:
        raise ValueError("Frozen catalog size drifted")

    prediction_users: set[str] = set()
    cold_events = 0
    warm_events = 0
    ranking_size = expected["ranking_size"]
    with paths["frozen_f0_r2_validation_predictions"].open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row.get("user_id"))
            if user in prediction_users or user not in projected:
                raise ValueError(f"Prediction user mismatch at line {line_number}: {user}")
            prediction_users.add(user)
            target = str(row.get("target"))
            if target != projected[user][-1]:
                raise ValueError(f"Projected/prediction validation target mismatch: {user}")
            is_cold = target in cold
            if target not in lexical or bool(row.get("is_cold")) != is_cold:
                raise ValueError(f"Prediction target partition mismatch: {user}")
            cold_events += int(is_cold)
            warm_events += int(not is_cold)
            for key in ("v0_top50", "r2_top50"):
                ranking = row.get(key)
                if not isinstance(ranking, list) or len(ranking) != ranking_size:
                    raise ValueError(f"Invalid {key} width for {user}")
                if len(set(ranking)) != ranking_size or not set(ranking).issubset(lexical):
                    raise ValueError(f"Unknown/duplicate item in {key} for {user}")

    if prediction_users != set(projected):
        raise ValueError("Frozen F0/R2 users differ from projected validation users")
    observed = {
        "validation_events": len(projected),
        "cold_validation_events": cold_events,
        "warm_validation_events": warm_events,
        "catalog_items": len(lexical),
        "cold_catalog_items": len(cold),
        "warm_catalog_items": len(warm),
        "ranking_size": ranking_size,
    }
    if observed != dict(expected):
        raise ValueError(f"Frozen S16-4 universe drift: {observed}")
    historical = load_json(paths["frozen_f0_r2_summary"])
    if historical.get("split") != "validation" or historical.get("test_predictions_opened") is not False:
        raise ValueError("Frozen F0/R2 source split/test boundary drift")
    return observed


def run(config_path: Path) -> dict[str, Any]:
    root = REPO_ROOT.resolve()
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config_contract(config)
    output_dir = root / str(config["output_dir"])
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing existing S16-4 preflight artifact root: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = utc_now()
    config_sha = sha256_file(config_path)
    shutil.copyfile(config_path, output_dir / "config.json")
    atomic_json(
        output_dir / "status.json",
        {
            "experiment_id": config["experiment_id"],
            "attempt_id": config["attempt_id"],
            "status": "RUNNING",
            "status_code": "RUNNING",
            "stage": "cpu_input_state_gate_freeze",
            "started_at": started_at,
            "updated_at": started_at,
            "process_alive": True,
            "runner_pid": os.getpid(),
            "gpu_count": 0,
            "validation_identity_opened": True,
            "validation_efficacy_metric_produced": False,
            "test_read": False,
            "automatic_retry": False,
            "exact_start_command": "bash experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh",
            "output_dir": config["output_dir"],
        },
    )

    input_paths: dict[str, Path] = {}
    input_manifest: dict[str, Any] = {}
    for name, declaration in config["inputs"].items():
        relative = str(declaration["path"])
        lowered = relative.lower()
        if "predictions_test" in lowered or relative.endswith("/user_sequence.txt"):
            raise ValueError(f"Sealed/raw data path is forbidden in S16-4 preflight: {relative}")
        path = resolve_regular_file(root, relative)
        digest = sha256_file(path)
        if digest != declaration.get("sha256"):
            raise ValueError(f"Frozen input SHA drift: {name}")
        input_paths[name] = path
        input_manifest[name] = {
            "path": relative,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }

    source_gates = validate_source_gates(input_paths)
    gridge_checkpoints = verify_gridge_manifest(
        root,
        input_paths["gridge_checkpoint_manifest"],
        input_paths["gridge_aggregate_deltas"],
    )
    universe = validate_validation_identity(input_paths, config["expected_universe"])

    code_manifest: dict[str, str] = {}
    for relative in CODE_FILES:
        path = resolve_regular_file(root, relative)
        code_manifest[relative] = sha256_file(path)
    code_manifest[str(config_path.relative_to(root))] = config_sha

    atomic_json(output_dir / "input_file_sha256.json", input_manifest)
    atomic_json(output_dir / "source_gate_manifest.json", source_gates)
    atomic_json(output_dir / "gridge_checkpoint_sha256.json", gridge_checkpoints)
    atomic_json(output_dir / "code_sha256.json", code_manifest)
    atomic_json(output_dir / "arm_manifest.json", config["arms"])
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "validation_identity_files": [
                config["inputs"]["projected_train_validation_sequences"]["path"],
                config["inputs"]["frozen_f0_r2_validation_predictions"]["path"],
                config["inputs"]["frozen_f0_r2_summary"]["path"],
            ],
            "validation_target_used_for_metric_or_selection": False,
            "validation_efficacy_metric_produced": False,
            "raw_user_sequence_opened": False,
            "test_opened": False,
            "test_read": False,
        },
    )
    atomic_json(
        output_dir / "command_manifest.json",
        {
            "exact_start_command": "bash experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh",
            "gpu_used": False,
            "automatic_retry": False,
        },
    )
    summary = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "completed",
        "verdict": "PASS_S16_4_TOYS_INPUT_STATE_GATE_FREEZE",
        "config_sha256": config_sha,
        "generated_at_utc": utc_now(),
        "source_gates": source_gates,
        "universe": universe,
        "arms": list(REQUIRED_ARMS),
        "excluded_arms": config["excluded_arms"],
        "strict_specgr_contract_frozen": True,
        "validation_identity_opened": True,
        "validation_efficacy_metric_produced": False,
        "scientific_efficacy_metric_produced": False,
        "test_read": False,
        "gpu_used": False,
        "gpu_launch_ready": False,
        "launch_gate": "LOCKED_PENDING_ISOLATED_EVALUATOR_CODE_AND_USER_GPU_CONFIRMATION",
        "remaining_requirements": config["launch_contract"]["remaining_requirements"],
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "artifact_contract.json",
        {
            "verdict": "PASS_S16_4_TOYS_PREFLIGHT_ARTIFACT_CONTRACT",
            "required": [
                "status.json",
                "summary.json",
                "config.json",
                "input_file_sha256.json",
                "source_gate_manifest.json",
                "gridge_checkpoint_sha256.json",
                "code_sha256.json",
                "arm_manifest.json",
                "open_file_manifest.json",
                "command_manifest.json",
            ],
        },
    )
    atomic_json(
        output_dir / "status.json",
        {
            "experiment_id": config["experiment_id"],
            "attempt_id": config["attempt_id"],
            "status": "COMPLETED",
            "status_code": "PASS_S16_4_TOYS_INPUT_STATE_GATE_FREEZE",
            "stage": "finished",
            "reason": "S16-4 inputs, source Gates, arms, metric contract, and state SHAs are frozen; GPU launch remains locked pending isolated faithful evaluator freeze and user confirmation.",
            "started_at": started_at,
            "updated_at": utc_now(),
            "process_alive": False,
            "runner_pid": os.getpid(),
            "exit_code": 0,
            "exit_code_pending": False,
            "gpu_count": 0,
            "validation_identity_opened": True,
            "validation_efficacy_metric_produced": False,
            "scientific_efficacy_metric_produced": False,
            "test_read": False,
            "automatic_retry": False,
            "gpu_launch_ready": False,
            "exact_start_command": "bash experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh",
            "output_dir": config["output_dir"],
            "summary_path": f"{config['output_dir']}/summary.json",
        },
    )
    return summary


def write_failure(config_path: Path, error: BaseException) -> None:
    try:
        config = load_json(config_path)
        output_dir = REPO_ROOT / str(config["output_dir"])
        if output_dir.is_dir():
            atomic_json(
                output_dir / "status.json",
                {
                    "experiment_id": config.get("experiment_id"),
                    "attempt_id": config.get("attempt_id"),
                    "status": "FAILED",
                    "status_code": "S16_4_TOYS_PREFLIGHT_FAILED",
                    "stage": "finished",
                    "reason": f"{type(error).__name__}: {error}",
                    "updated_at": utc_now(),
                    "process_alive": False,
                    "runner_pid": os.getpid(),
                    "exit_code": 3,
                    "exit_code_pending": False,
                    "gpu_count": 0,
                    "validation_efficacy_metric_produced": False,
                    "scientific_efficacy_metric_produced": False,
                    "test_read": False,
                    "automatic_retry": False,
                    "gpu_launch_ready": False,
                },
            )
    except BaseException:
        pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args.config)
    except BaseException as error:
        write_failure(args.config, error)
        print(f"S16-4 frozen preflight failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
