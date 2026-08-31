#!/usr/bin/env python3
"""One preregistered coverage revision for the borderline S17-2R Latte family."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.resource_profiler import query_gpus, snapshot  # noqa: E402
from experiment.phase17.core.run_manager import (  # noqa: E402
    freeze_run_snapshot,
    launch_background_tmux,
    verify_run_snapshot,
)
from experiment.phase17.core.s2r_r2_evaluator import compare_family_predictions  # noqa: E402
from experiment.phase17.core.s2r_sid import (  # noqa: E402
    build_r2_external_examples,
    build_r2_training_examples,
    parse_shadow_sequences,
    read_cohort_user_ids,
    sha256_file,
)
from experiment.phase17.core.status_writer import (  # noqa: E402
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)
from experiment.phase17.protocol.s2r_r2_runtime import (  # noqa: E402
    EARLY_STOP_PATH,
    EXPERIMENT_ID,
    PREFLIGHT_DIR,
    SEQUENCE_PATH,
    SID_PATH,
    codec_from_sid,
)
from experiment.phase17.protocol.s2r_r2_screen_runtime import (  # noqa: E402
    _load_best,
    _save_external_evaluation,
    evaluate_model,
)


CONFIG_PATH = ROOT / "experiment/phase17/config/s17_s2r_r2_latte_revision.json"
BASE_CONFIG_PATH = PREFLIGHT_DIR / "frozen_config.json"
PROFILE_ATTEMPT = "r2-latte-revision-profile-0001"
PROFILE_ROOT = ROOT / "artifacts/phase17/s2r_r2/latte_revision/profile/run-0001"
PROFILE_SNAPSHOT = (
    ROOT
    / "artifacts/phase17/snapshots"
    / EXPERIMENT_ID
    / PROFILE_ATTEMPT
    / "manifest.json"
)
RECOVERY_CONFIG_PATH = (
    ROOT
    / "experiment/phase17/config/s17_s2r_r2_latte_revision_profile_recovery.json"
)
PROFILE_RECOVERY_ATTEMPT = "r2-latte-revision-profile-recovery-0002"
PROFILE_RECOVERY_ROOT = (
    ROOT / "artifacts/phase17/s2r_r2/latte_revision/profile/run-0002"
)
PROFILE_RECOVERY_SNAPSHOT = (
    ROOT
    / "artifacts/phase17/snapshots"
    / EXPERIMENT_ID
    / PROFILE_RECOVERY_ATTEMPT
    / "manifest.json"
)
REVISION_ATTEMPT = "r2-latte-revision-0001"
REVISION_ROOT = ROOT / "artifacts/phase17/s2r_r2/latte_revision/run-0001"
REVISION_SNAPSHOT = (
    ROOT
    / "artifacts/phase17/snapshots"
    / EXPERIMENT_ID
    / REVISION_ATTEMPT
    / "manifest.json"
)
ARMS = ("latte_full", "psid_control")
REVISION_ADMISSION_FREE_MIB = 8192


def _configs() -> tuple[dict, dict]:
    revision = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    base["optimization"]["num_beams"] = revision["change_scope"]["num_beams_after"]
    base["optimization"]["top_k"] = revision["change_scope"]["top_k"]
    base["optimization"]["evaluation_batch_size"] = revision["change_scope"][
        "evaluation_batch_size"
    ]
    return revision, base


def prepare() -> dict:
    revision, base = _configs()
    source = ROOT / revision["source_screen_summary"]
    if sha256_file(source) != revision["source_screen_summary_sha256"]:
        raise RuntimeError("Latte source screen summary changed after revision freeze")
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    if source_payload["comparison"]["decision"] != "BORDERLINE_ONE_REVISION":
        raise RuntimeError("Latte is not eligible for its single R2 revision")
    for arm in ARMS:
        path = ROOT / revision["arms"][arm]["checkpoint"]
        if sha256_file(path) != revision["arms"][arm]["checkpoint_sha256"]:
            raise RuntimeError(f"Latte revision checkpoint changed: {arm}")
    payload = {
        "schema_version": "phase17.s17_2r_r2_latte_revision_preflight.v1",
        "state": "READY",
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "base_config_sha256": sha256_file(BASE_CONFIG_PATH),
        "sid_sha256": sha256_file(SID_PATH),
        "arms": revision["arms"],
        "num_beams": base["optimization"]["num_beams"],
        "top_k": base["optimization"]["top_k"],
        "evaluation_batch_size": base["optimization"]["evaluation_batch_size"],
        "external_target_read": False,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "prepared_at": utc_now(),
    }
    atomic_json(PREFLIGHT_DIR / "latte_revision_preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _load_arm(arm: str, device: torch.device, revision: dict, base: dict):
    sid = json.loads(SID_PATH.read_text(encoding="utf-8"))
    codec = codec_from_sid(sid)
    checkpoint = ROOT / revision["arms"][arm]["checkpoint"]
    model = _load_best(arm, codec, base, checkpoint, device)
    return codec, model


def profile_worker(
    physical_gpu: int, snapshot_path: Path, output_root: Path = PROFILE_ROOT
) -> dict:
    if physical_gpu == 1:
        raise PermissionError("GPU1 is reserved for the non-scientific repeat")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for Latte revision profile")
    verify_run_snapshot(ROOT, snapshot_path)
    frozen = json.loads(
        (snapshot_path.parent / "config.json").read_text(encoding="utf-8")
    )
    revision, base = frozen["revision"], frozen
    try:
        users = parse_shadow_sequences(SEQUENCE_PATH)
        early_ids = tuple(
            line.strip()
            for line in EARLY_STOP_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        _, early = build_r2_training_examples(users, early_ids)
        device = torch.device("cuda:0")
        results = {}
        for arm in ARMS:
            codec, model = _load_arm(arm, device, revision, base)
            torch.cuda.reset_peak_memory_stats(device)
            evaluation = evaluate_model(
                arm=arm,
                model=model,
                codec=codec,
                examples=early[: revision["capacity_profile"]["users"]],
                device=device,
                config=base,
            )
            results[arm] = {
                "prediction_rows": len(evaluation["metrics_by_user"]),
                "valid_item_rate": evaluation["valid_item_rate"],
                "mean_unique_candidates": evaluation["mean_unique_candidates"],
                "multi_path_item_rate": evaluation["multi_path_item_rate"],
                "generation_seconds": evaluation["generation_seconds"],
                "peak_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                ),
            }
            del model
            torch.cuda.empty_cache()
        payload = {
            "schema_version": "phase17.s17_2r_r2_latte_revision_profile.v1",
            "state": "PASS"
            if all(
                row["prediction_rows"] == 8 and row["valid_item_rate"] == 1.0
                for row in results.values()
            )
            else "FAIL",
            "physical_gpu": physical_gpu,
            "device_name": torch.cuda.get_device_name(device),
            "evaluation_batch_size": base["optimization"]["evaluation_batch_size"],
            "arms": results,
            "maximum_peak_allocated_mib": max(
                row["peak_allocated_mib"] for row in results.values()
            ),
            "external_target_read": False,
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
            "completed_at": utc_now(),
        }
        atomic_json(output_root / "summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload
    except Exception as error:
        atomic_json(
            output_root / "failure.json",
            {
                "schema_version": "phase17.s17_2r_r2_latte_revision_profile_failure.v1",
                "physical_gpu": physical_gpu,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": __import__("traceback").format_exc(),
                "external_target_read": False,
                "official_test_read": False,
                "sports_read": False,
                "d1_read": False,
                "failed_at": utc_now(),
            },
        )
        raise


def source_paths(*, recovery: bool = False) -> list[Path]:
    paths = [
        ROOT / "experiment/phase17/protocol/s2r_r2_latte_revision_runtime.py",
        ROOT / "experiment/phase17/protocol/s2r_r2_screen_runtime.py",
        ROOT / "experiment/phase17/core/s2r_architectures.py",
        ROOT / "experiment/phase17/core/s2r_sid.py",
        ROOT / "experiment/phase17/core/s2r_r2_evaluator.py",
        CONFIG_PATH,
        BASE_CONFIG_PATH,
        PREFLIGHT_DIR / "latte_revision_preflight.json",
    ]
    if recovery:
        paths.append(RECOVERY_CONFIG_PATH)
    return paths


def launch_profile(gpu_id: int) -> dict:
    revision, base = _configs()
    if gpu_id == 1:
        raise PermissionError("GPU1 is excluded from Latte revision profile")
    records = query_gpus()
    by_id = {row.index: row for row in records}
    required = revision["capacity_profile"]["minimum_free_mib"]
    free = by_id[gpu_id].free_mib if gpu_id in by_id else 0
    if free < required:
        raise RuntimeError(f"GPU {gpu_id} has {free} MiB free; profile needs {required}")
    if PROFILE_ROOT.exists() or PROFILE_SNAPSHOT.parent.exists():
        raise FileExistsError("Latte revision profile run-0001 exists; retry forbidden")
    command = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_latte_revision_runtime",
        "profile-worker",
        "--physical-gpu",
        str(gpu_id),
        "--snapshot",
        str(PROFILE_SNAPSHOT),
    ]
    frozen = {
        **base,
        "revision": revision,
        "profile_command": command,
        "gpu_snapshot": snapshot(records),
        "gpu1_repeat_preserved": True,
    }
    outer = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_latte_revision_runtime",
        "launch-profile",
        "--gpu",
        str(gpu_id),
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=PROFILE_ATTEMPT,
        command=outer,
        source_paths=source_paths(),
        config=frozen,
    )
    verify_run_snapshot(ROOT, manifest)
    session = launch_background_tmux(
        experiment_id="s17_s2r_r2_latte_revision_profile",
        argv=command,
        cwd=ROOT,
        tmux_session="s17_s2r_r2_latte_revision_profile",
    )
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": PROFILE_ATTEMPT,
            "step_id": "S17-2R",
            "kind": "R2_LATTE_ONE_REVISION_CAPACITY_PROFILE",
            "started_at": utc_now(),
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "gpu_ids": [gpu_id],
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_2R_LATTE_REVISION_PROFILE_STARTED",
        stage="r2_latte_revision_profile",
        progress={"current": 0, "total": 2, "unit": "revision_profile_arm"},
        gpu_ids=[gpu_id],
        tmux_session=session,
        process_alive=True,
        run_snapshot_manifest=str(manifest.relative_to(ROOT)),
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    payload = {
        "gpu_id": gpu_id,
        "command": command,
        "tmux_session": session,
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def record_missing_profile_failure() -> dict:
    summary = PROFILE_ROOT / "summary.json"
    failure = PROFILE_ROOT / "failure.json"
    if summary.exists() or failure.exists():
        raise FileExistsError("initial Latte revision profile already has a terminal artifact")
    payload = {
        "schema_version": "phase17.s17_2r_r2_latte_revision_profile_failure.v1",
        "attempt_id": PROFILE_ATTEMPT,
        "state": "ENGINEERING_FAILURE",
        "error_type": "MissingTerminalArtifactAfterWorkerExit",
        "error": "tmux worker exited after allocating GPU memory without summary or traceback",
        "probable_root_cause": "shared GPU free-memory contraction during beam-200 generation",
        "root_cause_certainty": "PROBABLE_NOT_PROVEN",
        "evidence": {
            "worker_pid": 2554273,
            "observed_worker_memory_mib": 6776,
            "observed_gpu_free_mib_after_launch": 10541,
            "tmux_alive_after_exit": False,
            "cpu_equivalent_interface_survived_seconds_before_manual_stop": 660,
            "automatic_retry": False,
        },
        "scientific_result_eligible": False,
        "external_target_read": False,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "failed_at": utc_now(),
    }
    atomic_json(failure, payload)
    ledger = ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl"
    AttemptLedger(ledger).append(
        {
            "attempt_id": f"{PROFILE_ATTEMPT}-closeout",
            "step_id": "S17-2R",
            "kind": "R2_LATTE_ONE_REVISION_CAPACITY_PROFILE_CLOSEOUT",
            "started_at": payload["failed_at"],
            "ended_at": payload["failed_at"],
            "state": "FAILED",
            "scientific_result_eligible": False,
            "closes_attempt_id": PROFILE_ATTEMPT,
            "failure": str(failure.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_2R_LATTE_REVISION_PROFILE_ENGINEERING_FAILURE_RECOVERY_PREFLIGHT",
        stage="r2_latte_revision_profile_recovery_preflight",
        progress={"current": 0, "total": 2, "unit": "revision_profile_arm"},
        gpu_ids=[],
        tmux_session=None,
        process_alive=False,
        latte_revision_profile_failure=str(failure.relative_to(ROOT)),
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def launch_profile_recovery(gpu_id: int) -> dict:
    if gpu_id == 1:
        raise PermissionError("GPU1 is excluded from Latte revision profile recovery")
    if not (PROFILE_ROOT / "failure.json").exists():
        raise FileNotFoundError("initial profile failure must be recorded before recovery")
    recovery = json.loads(RECOVERY_CONFIG_PATH.read_text(encoding="utf-8"))
    revision, base = _configs()
    revision["change_scope"]["evaluation_batch_size"] = recovery[
        "engineering_change"
    ]["evaluation_batch_size_after"]
    base["optimization"]["evaluation_batch_size"] = recovery["engineering_change"][
        "evaluation_batch_size_after"
    ]
    records = query_gpus()
    by_id = {row.index: row for row in records}
    free = by_id[gpu_id].free_mib if gpu_id in by_id else 0
    if free < recovery["minimum_free_mib"]:
        raise RuntimeError(
            f"GPU {gpu_id} has {free} MiB free; recovery needs {recovery['minimum_free_mib']}"
        )
    if PROFILE_RECOVERY_ROOT.exists() or PROFILE_RECOVERY_SNAPSHOT.parent.exists():
        raise FileExistsError("Latte revision recovery run-0002 exists; retry forbidden")
    command = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_latte_revision_runtime",
        "profile-worker",
        "--physical-gpu",
        str(gpu_id),
        "--snapshot",
        str(PROFILE_RECOVERY_SNAPSHOT),
        "--output-root",
        str(PROFILE_RECOVERY_ROOT),
    ]
    frozen = {
        **base,
        "revision": revision,
        "profile_recovery": recovery,
        "profile_command": command,
        "gpu_snapshot": snapshot(records),
        "gpu1_repeat_preserved": True,
    }
    outer = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_latte_revision_runtime",
        "launch-profile-recovery",
        "--gpu",
        str(gpu_id),
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=PROFILE_RECOVERY_ATTEMPT,
        command=outer,
        source_paths=source_paths(recovery=True),
        config=frozen,
    )
    verify_run_snapshot(ROOT, manifest)
    session = launch_background_tmux(
        experiment_id="s17_s2r_r2_latte_revision_profile_r2",
        argv=command,
        cwd=ROOT,
        tmux_session="s17_s2r_r2_latte_revision_profile_r2",
    )
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": PROFILE_RECOVERY_ATTEMPT,
            "step_id": "S17-2R",
            "kind": "R2_LATTE_ONE_REVISION_CAPACITY_PROFILE_RECOVERY",
            "started_at": utc_now(),
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "recovery_of": PROFILE_ATTEMPT,
            "gpu_ids": [gpu_id],
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
        }
    )
    payload = {
        "gpu_id": gpu_id,
        "command": command,
        "tmux_session": session,
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
        "recovery_of": PROFILE_ATTEMPT,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def finalize_profile_recovery() -> dict:
    summary_path = PROFILE_RECOVERY_ROOT / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Latte revision recovery profile summary is missing")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload["state"] != "PASS":
        raise RuntimeError("Latte revision recovery profile did not pass")
    closeout = f"{PROFILE_RECOVERY_ATTEMPT}-closeout"
    ledger_path = ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl"
    ids = {
        json.loads(line)["attempt_id"]
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if closeout not in ids:
        AttemptLedger(ledger_path).append(
            {
                "attempt_id": closeout,
                "step_id": "S17-2R",
                "kind": "R2_LATTE_ONE_REVISION_CAPACITY_PROFILE_RECOVERY_CLOSEOUT",
                "started_at": payload["completed_at"],
                "ended_at": payload["completed_at"],
                "state": "COMPLETED",
                "scientific_result_eligible": False,
                "closes_attempt_id": PROFILE_RECOVERY_ATTEMPT,
                "summary": str(summary_path.relative_to(ROOT)),
            }
        )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_2R_LATTE_REVISION_PROFILE_COMPLETE_FORMAL_PREFLIGHT",
        stage="r2_latte_revision_formal_preflight",
        progress={"current": 1, "total": 2, "unit": "revision_gate"},
        gpu_ids=[],
        tmux_session=None,
        process_alive=False,
        latte_revision_profile_summary=str(summary_path.relative_to(ROOT)),
        latte_revision_profile_pass=True,
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def revision_worker(
    arm: str, physical_gpu: int, snapshot_path: Path, output_root: Path = REVISION_ROOT
) -> dict:
    if arm not in ARMS or physical_gpu == 1:
        raise PermissionError("invalid Latte revision arm or reserved GPU1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for formal Latte revision")
    verify_run_snapshot(ROOT, snapshot_path)
    frozen = json.loads(
        (snapshot_path.parent / "config.json").read_text(encoding="utf-8")
    )
    revision, base = frozen["revision"], frozen
    arm_dir = output_root / arm
    if arm_dir.exists():
        raise FileExistsError(f"Latte revision arm output exists: {arm_dir}")
    arm_dir.mkdir(parents=True)
    device = torch.device("cuda:0")
    try:
        users = parse_shadow_sequences(SEQUENCE_PATH)
        external = build_r2_external_examples(users)
        codec, model = _load_arm(arm, device, revision, base)
        torch.cuda.reset_peak_memory_stats(device)
        evaluation = evaluate_model(
            arm=arm,
            model=model,
            codec=codec,
            examples=external,
            device=device,
            config=base,
        )
        saved = _save_external_evaluation(arm_dir, evaluation)
        payload = {
            "schema_version": "phase17.s17_2r_r2_latte_revision_arm.v1",
            "arm": arm,
            "physical_gpu": physical_gpu,
            "device_name": torch.cuda.get_device_name(device),
            "formal_result_eligible": True,
            "checkpoint": revision["arms"][arm],
            "evaluation": saved,
            "peak_allocated_mib": float(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            ),
            "external_target_evaluation_count": 1,
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
            "completed_at": utc_now(),
        }
        atomic_json(arm_dir / "summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload
    except Exception as error:
        atomic_json(
            arm_dir / "failure.json",
            {
                "schema_version": "phase17.s17_2r_r2_latte_revision_failure.v1",
                "arm": arm,
                "physical_gpu": physical_gpu,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "official_test_read": False,
                "sports_read": False,
                "d1_read": False,
                "failed_at": utc_now(),
            },
        )
        raise


def launch_revision(gpu_latte: int, gpu_psid: int) -> dict:
    if gpu_latte == gpu_psid or 1 in {gpu_latte, gpu_psid}:
        raise PermissionError("Latte revision needs two distinct non-GPU1 cards")
    profile = json.loads(
        (PROFILE_RECOVERY_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    if profile["state"] != "PASS":
        raise RuntimeError("Latte revision profile recovery did not pass")
    recovery = json.loads(RECOVERY_CONFIG_PATH.read_text(encoding="utf-8"))
    revision, base = _configs()
    revision["change_scope"]["evaluation_batch_size"] = recovery[
        "engineering_change"
    ]["evaluation_batch_size_after"]
    base["optimization"]["evaluation_batch_size"] = recovery["engineering_change"][
        "evaluation_batch_size_after"
    ]
    records = query_gpus()
    by_id = {row.index: row for row in records}
    for arm, gpu_id in (("latte_full", gpu_latte), ("psid_control", gpu_psid)):
        free = by_id[gpu_id].free_mib if gpu_id in by_id else 0
        if free < REVISION_ADMISSION_FREE_MIB:
            raise RuntimeError(
                f"GPU {gpu_id} for {arm} has {free} MiB free; revision needs {REVISION_ADMISSION_FREE_MIB}"
            )
    if REVISION_ROOT.exists() or REVISION_SNAPSHOT.parent.exists():
        raise FileExistsError("formal Latte revision run-0001 exists; retry forbidden")
    assignments = {"latte_full": gpu_latte, "psid_control": gpu_psid}
    commands = {
        arm: [
            "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
            "-m",
            "experiment.phase17.protocol.s2r_r2_latte_revision_runtime",
            "revision-worker",
            "--arm",
            arm,
            "--physical-gpu",
            str(gpu_id),
            "--snapshot",
            str(REVISION_SNAPSHOT),
        ]
        for arm, gpu_id in assignments.items()
    }
    frozen = {
        **base,
        "revision": revision,
        "profile_summary_path": str(
            (PROFILE_RECOVERY_ROOT / "summary.json").relative_to(ROOT)
        ),
        "profile_summary_sha256": sha256_file(
            PROFILE_RECOVERY_ROOT / "summary.json"
        ),
        "revision_commands": commands,
        "gpu_assignments": assignments,
        "gpu_snapshot": snapshot(records),
        "gpu1_repeat_preserved": True,
    }
    outer = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_latte_revision_runtime",
        "launch-revision",
        "--gpu-latte",
        str(gpu_latte),
        "--gpu-psid",
        str(gpu_psid),
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=REVISION_ATTEMPT,
        command=outer,
        source_paths=[
            *source_paths(recovery=True),
            PROFILE_RECOVERY_ROOT / "summary.json",
        ],
        config=frozen,
    )
    verify_run_snapshot(ROOT, manifest)
    sessions = {}
    for arm, command in commands.items():
        session = f"s17_s2r_r2_latte_revision_{'full' if arm == 'latte_full' else 'psid'}"
        sessions[arm] = launch_background_tmux(
            experiment_id=session, argv=command, cwd=ROOT, tmux_session=session
        )
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": REVISION_ATTEMPT,
            "step_id": "S17-2R",
            "kind": "R2_LATTE_BORDERLINE_ONE_FORMAL_REVISION",
            "started_at": utc_now(),
            "state": "RUNNING",
            "scientific_result_eligible": True,
            "gpu_ids": [gpu_latte, gpu_psid],
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_2R_LATTE_ONE_FORMAL_REVISION_STARTED",
        stage="r2_latte_one_formal_revision",
        progress={"current": 0, "total": 2, "unit": "revision_arm"},
        gpu_ids=[gpu_latte, gpu_psid],
        tmux_session=",".join(sessions.values()),
        process_alive=True,
        run_snapshot_manifest=str(manifest.relative_to(ROOT)),
        gpu1_repeat_preserved=True,
        affects_scientific_result=True,
        result_selection_eligible=True,
    )
    payload = {
        "gpu_assignments": assignments,
        "commands": commands,
        "tmux_sessions": sessions,
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _load_user_metrics(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_revision() -> dict:
    revision, base = _configs()
    summaries = {}
    for arm in ARMS:
        path = REVISION_ROOT / arm / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Latte revision arm summary missing: {path}")
        summaries[arm] = json.loads(path.read_text(encoding="utf-8"))
    treatment_eval = summaries["latte_full"]["evaluation"]
    control_eval = summaries["psid_control"]["evaluation"]
    treatment = _load_user_metrics(ROOT / treatment_eval["user_metrics_path"])
    control = _load_user_metrics(ROOT / control_eval["user_metrics_path"])
    cohorts = read_cohort_user_ids(
        [ROOT / path for path in base["data"]["evaluation_cohort_paths"]]
    )
    mechanism = {
        "valid_item_rate": min(
            treatment_eval["valid_item_rate"], control_eval["valid_item_rate"]
        ),
        "multi_path_item_rate": treatment_eval["multi_path_item_rate"],
        "mean_unique_candidates": treatment_eval["mean_unique_candidates"],
    }
    comparison = compare_family_predictions(
        treatment=treatment,
        control=control,
        cohorts=cohorts,
        mechanism_metrics=mechanism,
        family="latte",
        bootstrap_replicates=base["uncertainty"]["replicates"],
        seed=base["uncertainty"]["seed"],
    )
    coverage_pass = (
        mechanism["mean_unique_candidates"]
        >= revision["revision_mechanism_gate"]["mean_unique_candidates_min"]
    )
    strong = comparison["decision"] == "STRONG_PROMOTE" and coverage_pass
    final_decision = "STRONG_PROMOTE" if strong else "REJECT_AFTER_ONE_REVISION"
    payload = {
        "schema_version": "phase17.s17_2r_r2_latte_revision_closeout.v1",
        "state": "COMPLETED",
        "family": "latte",
        "revision_index": 1,
        "revision_budget_consumed": True,
        "no_further_revision": True,
        "arm_summaries": {
            arm: str((REVISION_ROOT / arm / "summary.json").relative_to(ROOT))
            for arm in ARMS
        },
        "comparison": comparison,
        "coverage_gate_pass": coverage_pass,
        "final_decision": final_decision,
        "r3_eligible": strong,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "gpu1_repeat_preserved": True,
        "completed_at": utc_now(),
    }
    output = REVISION_ROOT / "revision_summary.json"
    atomic_json(output, payload)
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": f"{REVISION_ATTEMPT}-closeout",
            "step_id": "S17-2R",
            "kind": "R2_LATTE_BORDERLINE_ONE_FORMAL_REVISION_CLOSEOUT",
            "started_at": payload["completed_at"],
            "ended_at": payload["completed_at"],
            "state": "COMPLETED",
            "scientific_result_eligible": True,
            "closes_attempt_id": REVISION_ATTEMPT,
            "summary": str(output.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_2R_LATTE_ONE_REVISION_COMPLETE_R3_GATE",
        stage="r2_latte_revision_complete_r3_gate",
        progress={"current": 2, "total": 2, "unit": "revision_gate"},
        gpu_ids=[],
        tmux_session=None,
        process_alive=False,
        latte_revision_summary=str(output.relative_to(ROOT)),
        latte_revision_final_decision=final_decision,
        gpu1_repeat_preserved=True,
        affects_scientific_result=True,
        result_selection_eligible=True,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    launch_profile_parser = subparsers.add_parser("launch-profile")
    launch_profile_parser.add_argument("--gpu", type=int, required=True)
    worker_parser = subparsers.add_parser("profile-worker")
    worker_parser.add_argument("--physical-gpu", type=int, required=True)
    worker_parser.add_argument("--snapshot", type=Path, required=True)
    worker_parser.add_argument("--output-root", type=Path, default=PROFILE_ROOT)
    recovery_parser = subparsers.add_parser("launch-profile-recovery")
    recovery_parser.add_argument("--gpu", type=int, required=True)
    subparsers.add_parser("record-missing-profile-failure")
    subparsers.add_parser("finalize-profile-recovery")
    launch_revision_parser = subparsers.add_parser("launch-revision")
    launch_revision_parser.add_argument("--gpu-latte", type=int, required=True)
    launch_revision_parser.add_argument("--gpu-psid", type=int, required=True)
    revision_worker_parser = subparsers.add_parser("revision-worker")
    revision_worker_parser.add_argument("--arm", choices=ARMS, required=True)
    revision_worker_parser.add_argument("--physical-gpu", type=int, required=True)
    revision_worker_parser.add_argument("--snapshot", type=Path, required=True)
    revision_worker_parser.add_argument("--output-root", type=Path, default=REVISION_ROOT)
    subparsers.add_parser("finalize-revision")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "launch-profile":
        launch_profile(args.gpu)
    elif args.command == "launch-profile-recovery":
        launch_profile_recovery(args.gpu)
    elif args.command == "record-missing-profile-failure":
        record_missing_profile_failure()
    elif args.command == "finalize-profile-recovery":
        finalize_profile_recovery()
    elif args.command == "launch-revision":
        launch_revision(args.gpu_latte, args.gpu_psid)
    elif args.command == "revision-worker":
        revision_worker(args.arm, args.physical_gpu, args.snapshot, args.output_root)
    elif args.command == "finalize-revision":
        finalize_revision()
    else:
        profile_worker(args.physical_gpu, args.snapshot, args.output_root)


if __name__ == "__main__":
    main()
