#!/usr/bin/env python3
"""Exact low-memory beam profiles and PSID probe correction (attempt_004)."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

from experiment.phase17.core.full_latte_arm_contracts import ARM_IDS
from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now
from experiment.phase17.protocol import s17_fp12_resource_profile_r3_runtime as r3


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "attempt_004"
REVISION_ID = "r4_exact_low_memory_beam_and_psid_probe_fix"
SAFETY_MARGIN_MIB = 4096
RESEARCHER_DIRECTION = (
    "允许串行重试授权；有几张卡有20多g的可以启动G0/G1/G2，"
    "N0/N1见缝插针；只需保证不OOM并将状态写入artifacts"
)

# use_cache=False keeps FP32, beam=500, top-k=50 and the exact catalog tree,
# while recomputing short decoder K/V states instead of retaining one copy per
# layer and beam.  Native arms retain their measured 4 GiB cap.
PROFILE_SPECS = {
    "G0_GRAM_B0_FRESH": r3.r2.v1.ProfileSpec(
        "G0_GRAM_B0_FRESH", "S17-FP2-PROFILE", 5, 16384, 20480, 2, 1, "gram"
    ),
    "G1_GRAM_PSID_FULL": r3.r2.v1.ProfileSpec(
        "G1_GRAM_PSID_FULL", "S17-FP2-PROFILE", 3, 16384, 20480, 2, 1, "gram"
    ),
    "G2_GRAM_LATTE_FULL": r3.r2.v1.ProfileSpec(
        "G2_GRAM_LATTE_FULL", "S17-FP2-PROFILE", 1, 16384, 20480, 2, 1, "gram"
    ),
    "N0_NATIVE_PSID": r3.r2.v1.ProfileSpec(
        "N0_NATIVE_PSID", "S17-FP1-PROFILE", 2, 4096, 8192, 256, 1, "native"
    ),
    "N1_NATIVE_LATTE": r3.r2.v1.ProfileSpec(
        "N1_NATIVE_LATTE", "S17-FP1-PROFILE", 6, 4096, 8192, 256, 1, "native"
    ),
}


_R3_PREPARE = r3.prepare
_R3_LAUNCH = r3.launch
_R3_WORKER = r3.worker
_R3_INSPECT = r3.inspect
_R3_FROZEN_CONFIG = r3.frozen_config
_R3_TWO_SNAPSHOT_ADMISSION = r3.two_snapshot_admission


def arm_slug(arm_id: str) -> str:
    if arm_id not in PROFILE_SPECS:
        raise ValueError(f"unknown profile arm: {arm_id}")
    return arm_id.lower()


def experiment_id(arm_id: str) -> str:
    return f"s17_fp12_profile_r4_{arm_slug(arm_id)}"


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
        "matrix": root / r3.r2.v1.MATRIX_SUFFIX,
        "allocation": root / r3.r2.v1.ALLOCATION_SUFFIX,
        "tokenizer_status": root / r3.r2.v1.TOKENIZER_STATUS_SUFFIX,
        "vocab_manifest": root / r3.r2.v1.VOCAB_MANIFEST_SUFFIX,
        "authorization": root
        / f"artifacts/phase17/authorizations/{exp_id}_{ATTEMPT_ID}.json",
        "status_dir": root / "artifacts/phase17/status",
        "ledger": root / r3.r2.v1.LEDGER_SUFFIX,
        "snapshot": snapshot_manifest,
        "snapshot_worker": snapshot_manifest.parent
        / "src/000_s17_fp12_resource_profile_r4_runtime.py",
        "native_python": root / r3.r2.v1.NATIVE_PYTHON_SUFFIX,
    }


def frozen_config(
    root: Path, arm_id: str, preflight: dict[str, Any]
) -> dict[str, Any]:
    with _configured_r3():
        config = _R3_FROZEN_CONFIG(root, arm_id, preflight)
    config.update(
        schema_version="phase17.s17_fp12_resource_profile_config.v4",
        revision_id=REVISION_ID,
        allocation_policy="live_free_memory_assignment_exact_low_memory_beam",
        generation_kv_cache=False if PROFILE_SPECS[arm_id].family == "gram" else True,
        scientific_protocol_changes=[],
        execution_only_changes=(
            ["disable_generation_kv_cache_recompute_exact_logits"]
            if PROFILE_SPECS[arm_id].family == "gram"
            else ["accept_official_fresh_model_early_eos_width_in_capacity_probe"]
        ),
        supersedes_attempt_id="attempt_003",
        researcher_directed_retry=True,
    )
    return config


def snapshot_sources(root: Path, spec: r3.r2.v1.ProfileSpec) -> list[Path]:
    original = r3.r2._V1_SNAPSHOT_SOURCES(root, spec)
    return [
        Path(__file__).resolve(),
        Path(r3.__file__).resolve(),
        Path(r3.r2.__file__).resolve(),
        Path(r3.r2.v1.__file__).resolve(),
        *original[1:],
    ]


def two_snapshot_admission(
    spec: r3.r2.v1.ProfileSpec, authorization: dict[str, Any]
) -> dict[str, Any]:
    with _configured_r3():
        result = _R3_TWO_SNAPSHOT_ADMISSION(spec, authorization)
    result["revision_id"] = REVISION_ID
    return result


def verify_launch_authorization(root: Path, arm_id: str) -> dict[str, Any]:
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    allocation = json.loads(resolved["allocation"].read_text(encoding="utf-8"))
    profile_allocation = allocation["arm_specific_resource_profiles_r4"]
    if profile_allocation["physical_gpu_by_arm"].get(arm_id) != spec.physical_gpu:
        raise PermissionError("R4 arm-to-GPU allocation drifted")
    if profile_allocation["profile_launch_authorized_by_arm"].get(arm_id) is not True:
        raise PermissionError(f"R4 launch is not authorized for {arm_id}")
    if not resolved["authorization"].is_file():
        raise PermissionError(
            f"missing attempt-specific authorization: {resolved['authorization']}"
        )
    authorization = json.loads(resolved["authorization"].read_text(encoding="utf-8"))
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
            raise PermissionError(f"invalid R4 profile authorization field: {key}")
    if authorization.get("researcher_direction") != RESEARCHER_DIRECTION:
        raise PermissionError("R4 authorization does not preserve researcher direction")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256(resolved["authorization"]),
        "allocation_sha256": sha256(resolved["allocation"]),
    }


@contextlib.contextmanager
def _configured_r3() -> Iterator[None]:
    replacements = {
        "ATTEMPT_ID": ATTEMPT_ID,
        "REVISION_ID": REVISION_ID,
        "SAFETY_MARGIN_MIB": SAFETY_MARGIN_MIB,
        "RESEARCHER_DIRECTION": RESEARCHER_DIRECTION,
        "PROFILE_SPECS": PROFILE_SPECS,
        "experiment_id": experiment_id,
        "paths": paths,
        "frozen_config": frozen_config,
        "snapshot_sources": snapshot_sources,
        "two_snapshot_admission": two_snapshot_admission,
        "verify_launch_authorization": verify_launch_authorization,
    }
    previous = {name: getattr(r3, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(r3, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(r3, name, value)


def _supersede_attempt_003(root: Path, arm_id: str) -> None:
    old_experiment_id = f"s17_fp12_profile_r3_{arm_slug(arm_id)}"
    writer = StatusWriter(root / "artifacts/phase17/status", old_experiment_id)
    status_path = writer.paths.status(old_experiment_id)
    if not status_path.is_file():
        return
    status = writer.read()
    if status["scientific_state"] == "PREFLIGHT":
        writer.transition(
            "STOPPED",
            "STOPPED",
            "S17_FP12_PROFILE_SUPERSEDED_BY_EXACT_LOW_MEMORY_R4",
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
    with _configured_r3():
        result = _R3_PREPARE(root, arm_id)
    if result == 0:
        _supersede_attempt_003(root.resolve(), arm_id)
    return result


def authorize(root: Path, arm_id: str) -> int:
    root = root.resolve()
    spec = PROFILE_SPECS[arm_id]
    resolved = paths(root, arm_id)
    if resolved["authorization"].exists():
        raise FileExistsError(f"R4 authorization exists: {resolved['authorization']}")
    writer = StatusWriter(resolved["status_dir"], experiment_id(arm_id))
    status = writer.read()
    if status["scientific_state"] != "PREFLIGHT":
        raise RuntimeError(f"R4 profile is not authorizable: {status['scientific_state']}")
    allocation = json.loads(resolved["allocation"].read_text(encoding="utf-8"))
    contract = allocation["arm_specific_resource_profiles_r4"]
    if contract["profile_launch_authorized_by_arm"].get(arm_id) is not True:
        raise PermissionError(f"allocation does not authorize R4 profile {arm_id}")
    current = r3.r2.gpu_snapshot_once(spec)
    if current["selected"]["free_mib"] < spec.minimum_free_mib:
        raise RuntimeError(
            f"GPU{spec.physical_gpu} is below the R4 memory gate at authorization"
        )
    payload = {
        "schema_version": "phase17.s17_fp12_profile_authorization.v4",
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
        "observed_preexisting_compute_pids": sorted(
            int(row["pid"]) for row in current["selected_compute_processes"]
        ),
        "preserve_all_preexisting_compute_processes": True,
        "utilization_recorded_only": True,
        "authorization_snapshot": current,
        "researcher_direction": RESEARCHER_DIRECTION,
        "researcher_directed_retry": True,
        "automatic_process_termination": False,
        "automatic_retry": False,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
    atomic_json(resolved["authorization"], payload)
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "S17_FP12_R4_EXACT_LOW_MEMORY_PROFILE_AUTHORIZED",
        launch_authorized=True,
        stage="exact_low_memory_authorized_waiting_launch",
        authorization_path=str(resolved["authorization"].relative_to(root)),
        authorization_sha256=sha256(resolved["authorization"]),
        gpu_snapshot={"authorization": current},
        gpu_ids=[],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def launch(root: Path, arm_id: str) -> int:
    with _configured_r3():
        return _R3_LAUNCH(root, arm_id)


def worker(root: Path, arm_id: str, manifest_path: Path) -> int:
    with _configured_r3():
        return _R3_WORKER(root, arm_id, manifest_path)


def inspect(root: Path, arm_id: str) -> dict[str, Any]:
    with _configured_r3():
        result = _R3_INSPECT(root, arm_id)
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
