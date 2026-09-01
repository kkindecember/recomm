#!/usr/bin/env python3
"""Shared-server, memory-only FP1/FP2 resource profiles (attempt_002).

This revision records GPU utilization and every pre-existing compute process,
but neither is an admission gate.  Launch admission is based only on two fresh
free-memory snapshots and a frozen per-arm safety margin.  Existing processes
are always preserved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_resource_profile_runtime as v1


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "attempt_002"
REVISION_ID = "r2_shared_server_memory_only"
SAFETY_MARGIN_MIB = 4096
RESEARCHER_DIRECTION = (
    "我觉得你要求太高了 而且利用率不可能达到你说的这么低的 "
    "因为我们这是共享的服务器 所以你只要保证不oom就行"
)

PROFILE_SPECS = {
    "G0_GRAM_B0_FRESH": v1.ProfileSpec(
        "G0_GRAM_B0_FRESH", "S17-FP2-PROFILE", 1, 12288, 16384, 2, 1, "gram"
    ),
    "G1_GRAM_PSID_FULL": v1.ProfileSpec(
        "G1_GRAM_PSID_FULL", "S17-FP2-PROFILE", 0, 12288, 16384, 2, 1, "gram"
    ),
    "G2_GRAM_LATTE_FULL": v1.ProfileSpec(
        "G2_GRAM_LATTE_FULL", "S17-FP2-PROFILE", 7, 12288, 16384, 2, 1, "gram"
    ),
    "N0_NATIVE_PSID": v1.ProfileSpec(
        "N0_NATIVE_PSID", "S17-FP1-PROFILE", 4, 4096, 8192, 256, 1, "native"
    ),
    "N1_NATIVE_LATTE": v1.ProfileSpec(
        "N1_NATIVE_LATTE", "S17-FP1-PROFILE", 4, 4096, 8192, 256, 1, "native"
    ),
}


_V1_PREPARE = v1.prepare
_V1_LAUNCH = v1.launch
_V1_WORKER = v1.worker
_V1_INSPECT = v1.inspect
_V1_FROZEN_CONFIG = v1.frozen_config
_V1_SNAPSHOT_SOURCES = v1.snapshot_sources
_V1_VERIFY_AUTHORIZATION = v1.verify_launch_authorization
_V1_TWO_SNAPSHOT_ADMISSION = v1.two_snapshot_admission


def arm_slug(arm_id: str) -> str:
    if arm_id not in PROFILE_SPECS:
        raise ValueError(f"unknown profile arm: {arm_id}")
    return arm_id.lower()


def experiment_id(arm_id: str) -> str:
    return f"s17_fp12_profile_r2_{arm_slug(arm_id)}"


def paths(root: Path, arm_id: str) -> dict[str, Path]:
    slug = arm_slug(arm_id)
    exp_id = experiment_id(arm_id)
    result = root / f"artifacts/phase17/fullport/profiles/{slug}/{ATTEMPT_ID}"
    snapshot_manifest = (
        root / f"artifacts/phase17/snapshots/{exp_id}/{ATTEMPT_ID}/manifest.json"
    )
    return {
        "result": result,
        "config": result / "config.json",
        "cpu_preflight": result / "cpu_preflight.json",
        "summary": result / "summary.json",
        "failure": result / "failure.json",
        "log": result / "run.log",
        "matrix": root / v1.MATRIX_SUFFIX,
        "allocation": root / v1.ALLOCATION_SUFFIX,
        "tokenizer_status": root / v1.TOKENIZER_STATUS_SUFFIX,
        "vocab_manifest": root / v1.VOCAB_MANIFEST_SUFFIX,
        "authorization": root
        / f"artifacts/phase17/authorizations/{exp_id}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / v1.LEDGER_SUFFIX,
        "snapshot": snapshot_manifest,
        "snapshot_worker": snapshot_manifest.parent
        / "src/000_s17_fp12_resource_profile_r2_runtime.py",
        "native_python": root / v1.NATIVE_PYTHON_SUFFIX,
    }


def frozen_config(
    root: Path, arm_id: str, preflight: dict[str, Any]
) -> dict[str, Any]:
    with _configured_v1():
        config = _V1_FROZEN_CONFIG(root, arm_id, preflight)
    config.update(
        schema_version="phase17.s17_fp12_resource_profile_config.v2",
        revision_id=REVISION_ID,
        admission_policy="two_snapshot_free_memory_only",
        utilization_hard_gate=False,
        utilization_recorded_only=True,
        compute_pid_allowlist_hard_gate=False,
        preserve_all_preexisting_processes=True,
        shared_server_race_risk_acknowledged=True,
        safety_margin_mib=SAFETY_MARGIN_MIB,
        supersedes_attempt_id="attempt_001",
    )
    return config


def snapshot_sources(root: Path, spec: v1.ProfileSpec) -> list[Path]:
    original = _V1_SNAPSHOT_SOURCES(root, spec)
    return [Path(__file__).resolve(), Path(v1.__file__).resolve(), *original[1:]]


def gpu_snapshot_once(spec: v1.ProfileSpec) -> dict[str, Any]:
    return v1.gpu_snapshot_once(spec)


def two_snapshot_admission(
    spec: v1.ProfileSpec, authorization: dict[str, Any]
) -> dict[str, Any]:
    first = gpu_snapshot_once(spec)
    v1.time.sleep(5)
    second = gpu_snapshot_once(spec)
    for row in (first, second):
        selected = row["selected"]
        if selected["free_mib"] < spec.minimum_free_mib:
            raise RuntimeError(
                f"GPU{spec.physical_gpu} free={selected['free_mib']} MiB below "
                f"memory-only gate {spec.minimum_free_mib} MiB"
            )
    observed = sorted(
        {
            int(process["pid"])
            for row in (first, second)
            for process in row["selected_compute_processes"]
        }
    )
    return {
        "admission_policy": "two_snapshot_free_memory_only",
        "required_interval_seconds": 5,
        "minimum_free_mib": spec.minimum_free_mib,
        "peak_reserved_cap_mib": spec.peak_cap_mib,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "first": first,
        "second": second,
        "observed_preexisting_compute_pids": observed,
        "compute_pids_recorded_only": True,
        "utilization_recorded_only": True,
        "automatic_process_termination": False,
        "shared_server_post_snapshot_allocation_race_cannot_be_eliminated": True,
    }


def verify_launch_authorization(root: Path, arm_id: str) -> dict[str, Any]:
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    allocation = json.loads(resolved["allocation"].read_text(encoding="utf-8"))
    profile_allocation = allocation["arm_specific_resource_profiles_r2"]
    if profile_allocation["physical_gpu_by_arm"].get(arm_id) != spec.physical_gpu:
        raise PermissionError("R2 arm-to-GPU allocation drifted")
    if profile_allocation["profile_launch_authorized_by_arm"].get(arm_id) is not True:
        raise PermissionError(f"R2 launch is not authorized for {arm_id}")
    if not resolved["authorization"].is_file():
        raise PermissionError(
            f"missing attempt-specific authorization: {resolved['authorization']}"
        )
    authorization = json.loads(
        resolved["authorization"].read_text(encoding="utf-8")
    )
    expected = {
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "revision_id": REVISION_ID,
        "arm_id": arm_id,
        "authorized": True,
        "physical_gpu": spec.physical_gpu,
        "resource_profile_only": True,
        "effect_experiment_authorized": False,
        "memory_only_admission": True,
        "shared_server_coexistence_authorized": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"invalid R2 profile authorization field: {key}")
    if authorization.get("researcher_direction") != RESEARCHER_DIRECTION:
        raise PermissionError("R2 authorization does not preserve researcher direction")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256(resolved["authorization"]),
        "allocation_sha256": sha256(resolved["allocation"]),
    }


@contextlib.contextmanager
def _configured_v1() -> Iterator[None]:
    replacements = {
        "ATTEMPT_ID": ATTEMPT_ID,
        "PROFILE_SPECS": PROFILE_SPECS,
        "SAFETY_MARGIN_MIB": SAFETY_MARGIN_MIB,
        "experiment_id": experiment_id,
        "paths": paths,
        "frozen_config": frozen_config,
        "snapshot_sources": snapshot_sources,
        "two_snapshot_admission": two_snapshot_admission,
        "verify_launch_authorization": verify_launch_authorization,
    }
    previous = {name: getattr(v1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(v1, name, value)


def _supersede_attempt_001(root: Path, arm_id: str) -> None:
    old_experiment_id = f"s17_fp12_profile_{arm_slug(arm_id)}"
    old_writer = StatusWriter(root / "artifacts/phase17/status", old_experiment_id)
    old_status_path = old_writer.paths.status(old_experiment_id)
    if not old_status_path.is_file():
        return
    old_status = old_writer.read()
    if old_status["scientific_state"] == "PREFLIGHT":
        old_writer.transition(
            "STOPPED",
            "STOPPED",
            "S17_FP12_PROFILE_SUPERSEDED_BY_SHARED_SERVER_MEMORY_ONLY_R2",
            process_alive=False,
            launch_authorized=False,
            stage="superseded_before_launch",
            superseded_by_experiment_id=experiment_id(arm_id),
            superseded_by_attempt_id=ATTEMPT_ID,
            gpu_ids=[],
            automatic_process_termination=False,
            automatic_retry=False,
        )


def prepare(root: Path, arm_id: str) -> int:
    with _configured_v1():
        result = _V1_PREPARE(root, arm_id)
    if result == 0:
        _supersede_attempt_001(root.resolve(), arm_id)
    return result


def authorize(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    if resolved["authorization"].exists():
        raise FileExistsError(
            f"R2 authorization already exists: {resolved['authorization']}"
        )
    status_writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    status = status_writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"R2 profile is not authorizable: {status['scientific_state']}")
    allocation = json.loads(resolved["allocation"].read_text(encoding="utf-8"))
    contract = allocation["arm_specific_resource_profiles_r2"]
    if contract["profile_launch_authorized_by_arm"].get(arm_id) is not True:
        raise PermissionError(f"allocation does not authorize R2 profile {arm_id}")
    current = gpu_snapshot_once(spec)
    if current["selected"]["free_mib"] < spec.minimum_free_mib:
        raise RuntimeError(
            f"GPU{spec.physical_gpu} is below the R2 memory gate at authorization"
        )
    observed_pids = sorted(
        int(row["pid"]) for row in current["selected_compute_processes"]
    )
    payload = {
        "schema_version": "phase17.s17_fp12_profile_authorization.v2",
        "authorized_at": utc_now(),
        "experiment_id": experiment_id(arm_id),
        "attempt_id": ATTEMPT_ID,
        "revision_id": REVISION_ID,
        "arm_id": arm_id,
        "authorized": True,
        "physical_gpu": spec.physical_gpu,
        "resource_profile_only": True,
        "effect_experiment_authorized": False,
        "memory_only_admission": True,
        "minimum_free_mib": spec.minimum_free_mib,
        "peak_reserved_cap_mib": spec.peak_cap_mib,
        "safety_margin_mib": SAFETY_MARGIN_MIB,
        "shared_server_coexistence_authorized": True,
        "observed_preexisting_compute_pids": observed_pids,
        "preserve_all_preexisting_compute_processes": True,
        "utilization_recorded_only": True,
        "authorization_snapshot": current,
        "researcher_direction": RESEARCHER_DIRECTION,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
    atomic_json(resolved["authorization"], payload)
    status_writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_R2_MEMORY_ONLY_PROFILE_AUTHORIZED",
        launch_authorized=True,
        stage="memory_only_authorized_waiting_launch",
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        gpu_snapshot={"authorization": current},
        gpu_ids=[],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def launch(root: Path, arm_id: str) -> int:
    with _configured_v1():
        return _V1_LAUNCH(root, arm_id)


def worker(root: Path, arm_id: str, manifest_path: Path) -> int:
    with _configured_v1():
        return _V1_WORKER(root, arm_id, manifest_path)


def inspect(root: Path, arm_id: str) -> dict[str, Any]:
    with _configured_v1():
        result = _V1_INSPECT(root, arm_id)
    result["revision_id"] = REVISION_ID
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "authorize", "launch", "worker", "inspect")
    )
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        return prepare(root, args.arm)
    if args.action == "authorize":
        return authorize(root, args.arm)
    if args.action == "launch":
        return launch(root, args.arm)
    if args.action == "inspect":
        print(json.dumps(inspect(root, args.arm), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.manifest is None:
        raise ValueError("worker requires --manifest")
    return worker(root, args.arm, args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
