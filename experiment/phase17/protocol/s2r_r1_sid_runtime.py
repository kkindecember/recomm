#!/usr/bin/env python3
"""Prepare and run the non-formal S17-2R R1 Semantic-ID contract profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.s2r_architectures import (  # noqa: E402
    S2RSemanticIDModel,
    item_scorer_gradient_norm,
    parameter_count,
    smoke_t5_config,
)
from experiment.phase17.core.s2r_parallel_architectures import (  # noqa: E402
    PARALLEL_ARMS,
    S2RParallelIDModel,
    parallel_gradient_norm,
    parallel_smoke_config,
)
from experiment.phase17.core.resource_profiler import query_gpus, snapshot  # noqa: E402
from experiment.phase17.core.run_manager import (  # noqa: E402
    freeze_run_snapshot,
    launch_background_tmux,
    verify_run_snapshot,
)
from experiment.phase17.core.s2r_sid import (  # noqa: E402
    SIDSequenceDataset,
    SemanticIDCodec,
    build_examples,
    build_residual_kmeans_ids,
    build_train_only_cf_codes,
    collate_sid_batch,
    parse_item_text,
    parse_shadow_sequences,
    sha256_file,
    tfidf_embeddings,
    train_catalog_items,
    write_sid_artifact,
)
from experiment.phase17.core.status_writer import (  # noqa: E402
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)


SEED = 2023
DATA_DIR = ROOT / "artifacts/phase17/s2r_preflight/data/Toys_s17_d0_3000"
SEQUENCE_PATH = DATA_DIR / "r1_smoke_user_sequence.txt"
ITEM_TEXT_PATH = DATA_DIR / "item_plain_text.txt"
OUTPUT_ROOT = ROOT / "artifacts/phase17/s2r_r1/run-0001"
SID_PATH = ROOT / "artifacts/phase17/s2r_preflight/sid/r1_tfidf_rq32x3.json"
SET_SID_PATH = ROOT / "artifacts/phase17/s2r_preflight/sid/r1_cf32_tfidf_rq32x3.json"
FAMILIES = {
    "diffgrm": ("diffgrm_ar_control", "diffgrm_masked"),
    "latte": ("psid_control", "latte_full"),
    "gryphon": ("gryphon_item",),
    "setrec": ("setrec_ar_control", "setrec_full"),
}
EXPERIMENT_ID = "s17_s2r_architecture_reselection"
ATTEMPT_ID = "run-0001"
EXPECTED_R1_PEAK_MIB = 6000
SAFETY_MARGIN_MIB = 4096


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared)


def candidate_gradient_norm(model: torch.nn.Module, arm: str) -> float:
    if arm == "gryphon_item":
        return item_scorer_gradient_norm(model)
    if arm == "latte_full":
        gradient = model.t5.shared.weight.grad
        if gradient is None:
            return 0.0
        start = model.codec.base_latent_token
        end = start + model.codec.n_latent_tokens
        return float(gradient[start:end].detach().float().norm().item())
    if arm in PARALLEL_ARMS:
        return parallel_gradient_norm(model)
    return 0.0


def load_or_build_sid() -> dict:
    if SID_PATH.exists():
        payload = json.loads(SID_PATH.read_text(encoding="utf-8"))
        if payload["sequence_sha256"] != sha256_file(SEQUENCE_PATH):
            raise RuntimeError("frozen R1 sequence hash changed after SID construction")
        if payload["item_text_sha256"] != sha256_file(ITEM_TEXT_PATH):
            raise RuntimeError("frozen item-text hash changed after SID construction")
        return payload

    users = parse_shadow_sequences(SEQUENCE_PATH)
    item_text = parse_item_text(ITEM_TEXT_PATH)
    item_ids = sorted(item_text)
    embeddings = tfidf_embeddings(
        item_ids,
        item_text,
        output_dim=128,
        max_features=4096,
        seed=SEED,
    )
    item_to_code, summary = build_residual_kmeans_ids(
        item_ids,
        embeddings,
        train_catalog_items(users),
        n_codebooks=3,
        codebook_size=32,
        seed=SEED,
    )
    return write_sid_artifact(
        SID_PATH,
        item_to_code=item_to_code,
        summary=summary,
        sequence_path=SEQUENCE_PATH,
        item_text_path=ITEM_TEXT_PATH,
    )


def codec_from_payload(payload: dict) -> SemanticIDCodec:
    return SemanticIDCodec(
        payload["item_to_code"],
        payload["summary"]["codebook_sizes"],
        n_latent_tokens=8,
        max_history_items=20,
    )


def load_or_build_set_sid(base_sid: dict) -> dict:
    if SET_SID_PATH.exists():
        payload = json.loads(SET_SID_PATH.read_text(encoding="utf-8"))
        if payload["base_sid_sha256"] != sha256_file(SID_PATH):
            raise RuntimeError("base SID changed after CF+semantic set construction")
        return payload
    users = parse_shadow_sequences(SEQUENCE_PATH)
    item_ids = sorted(base_sid["item_to_code"])
    cf_codes, summary = build_train_only_cf_codes(
        item_ids, users, codebook_size=32, hash_buckets=64, seed=SEED
    )
    item_to_code = {
        item: [cf_codes[item], *base_sid["item_to_code"][item]] for item in item_ids
    }
    payload = {
        "schema_version": "phase17.s17_2r_set_sid.v1",
        "formal_result_eligible": False,
        "fidelity": "R1_CONTRACT_TRAIN_PREFIX_CF_PLUS_TFIDF_RQ_SEMANTIC",
        "base_sid_path": str(SID_PATH.relative_to(ROOT)),
        "base_sid_sha256": sha256_file(SID_PATH),
        "sequence_sha256": sha256_file(SEQUENCE_PATH),
        "cf_summary": asdict(summary),
        "codebook_sizes": [summary.codebook_size, *base_sid["summary"]["codebook_sizes"]],
        "item_to_code": item_to_code,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
    }
    SET_SID_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(SET_SID_PATH, payload)
    return payload


def set_codec_from_payload(payload: dict) -> SemanticIDCodec:
    return SemanticIDCodec(
        payload["item_to_code"],
        payload["codebook_sizes"],
        n_latent_tokens=8,
        max_history_items=20,
    )


def prepare() -> dict:
    data_manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest["official_test_read"] or data_manifest["sports_read"]:
        raise PermissionError("sealed evaluation boundary was crossed")
    sid = load_or_build_sid()
    set_sid = load_or_build_set_sid(sid)
    users = parse_shadow_sequences(SEQUENCE_PATH)
    train, validation = build_examples(users)
    codec = codec_from_payload(sid)
    payload = {
        "schema_version": "phase17.s17_2r_r1_profile.v1",
        "step_id": "S17-2R",
        "attempt_id": "run-0001",
        "formal_result_eligible": False,
        "purpose": "architecture contract and resource profile only",
        "families": sorted(FAMILIES),
        "arms": [arm for family in sorted(FAMILIES) for arm in FAMILIES[family]],
        "seed": SEED,
        "dataset": "Toys_s17_d0_r1_100",
        "sequence_path": str(SEQUENCE_PATH.relative_to(ROOT)),
        "sequence_sha256": sha256_file(SEQUENCE_PATH),
        "item_text_path": str(ITEM_TEXT_PATH.relative_to(ROOT)),
        "item_text_sha256": sha256_file(ITEM_TEXT_PATH),
        "sid_path": str(SID_PATH.relative_to(ROOT)),
        "sid_sha256": sha256_file(SID_PATH),
        "set_sid_path": str(SET_SID_PATH.relative_to(ROOT)),
        "set_sid_sha256": sha256_file(SET_SID_PATH),
        "users": len(users),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "catalog_items": len(codec.item_ids),
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_beams": 50,
        "top_k": 50,
        "capacity": "r1",
        "model_fidelity": {
            "common": "scaled R1 contract model; not paper-level reproduction",
            "latte_full": "native latent-token paths and item aggregation active",
            "psid_control": "same SID codec, backbone capacity, steps, beam and evaluator",
            "gryphon_item": "joint item-level CE and same-candidate item reranking active",
            "gryphon_beam_control": "derived from identical Gryphon checkpoint and candidates",
            "diffgrm_masked": "scaled masked parallel denoising; not official PSE/OCN/CPD fidelity",
            "diffgrm_ar_control": "same encoder/decoder parameters and IDs with causal objective",
            "setrec_full": "train-prefix CF+semantic token set with simultaneous set loss",
            "setrec_ar_control": "same CF+semantic tokens and parameters with ordered causal loss",
        },
        "protected_contracts": {
            "train_only_quantizer_fit": sid["summary"]["train_only_quantizer_fit"],
            "validation_target_used_for_training": False,
            "guard_item_used_for_training": False,
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
        },
        "worker_template": [
            "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
            "-m",
            "experiment.phase17.protocol.s2r_r1_sid_runtime",
            "worker",
            "--family",
            "<diffgrm|gryphon|latte|setrec>",
            "--physical-gpu",
            "<gpu-id-not-1>",
        ],
        "created_at": utc_now(),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / "frozen_config.json", payload)
    return payload


def source_paths() -> list[Path]:
    return [
        ROOT / "experiment/phase17/protocol/s2r_r1_sid_runtime.py",
        ROOT / "experiment/phase17/protocol/s2r_data_contract.py",
        ROOT / "experiment/phase17/core/s2r_sid.py",
        ROOT / "experiment/phase17/core/s2r_architectures.py",
        ROOT / "experiment/phase17/core/s2r_parallel_architectures.py",
        ROOT / "experiment/phase17/config/s17_s2r_architecture_reselection_budget.json",
        ROOT / "artifacts/phase17/s2r_preflight/source_audit.json",
        DATA_DIR / "manifest.json",
        OUTPUT_ROOT / "frozen_config.json",
    ]


def _session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def launch(gpu_a: int, gpu_b: int) -> dict:
    """Freeze and launch two R1 waves after the researcher approves this command."""

    if gpu_a == gpu_b or 1 in {gpu_a, gpu_b}:
        raise PermissionError("R1 needs two distinct GPUs and must preserve GPU1")
    records = query_gpus()
    by_id = {row.index: row for row in records}
    for gpu_id in (gpu_a, gpu_b):
        if gpu_id not in by_id:
            raise ValueError(f"GPU {gpu_id} does not exist")
        required = EXPECTED_R1_PEAK_MIB + SAFETY_MARGIN_MIB
        if by_id[gpu_id].free_mib < required:
            raise RuntimeError(
                f"GPU {gpu_id} has only {by_id[gpu_id].free_mib} MiB free; "
                f"R1 admission requires {required} MiB"
            )
    sessions = {
        "wave_a": "s17_s2r_r1_wave_a",
        "wave_b": "s17_s2r_r1_wave_b",
    }
    occupied_sessions = [name for name in sessions.values() if _session_exists(name)]
    if occupied_sessions:
        raise FileExistsError(f"R1 tmux sessions already exist: {occupied_sessions}")
    family_dirs = [OUTPUT_ROOT / family for family in FAMILIES]
    if any(path.exists() for path in family_dirs):
        raise FileExistsError("an R1 family output already exists; implicit retry is forbidden")

    snapshot_path = (
        ROOT
        / "artifacts/phase17/snapshots"
        / EXPERIMENT_ID
        / ATTEMPT_ID
        / "manifest.json"
    )
    wave_a = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r1_sid_runtime",
        "wave",
        "--families",
        "latte,diffgrm",
        "--physical-gpu",
        str(gpu_a),
        "--snapshot",
        str(snapshot_path),
    ]
    wave_b = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r1_sid_runtime",
        "wave",
        "--families",
        "gryphon,setrec",
        "--physical-gpu",
        str(gpu_b),
        "--snapshot",
        str(snapshot_path),
    ]
    base_config = json.loads((OUTPUT_ROOT / "frozen_config.json").read_text(encoding="utf-8"))
    launch_config = {
        **base_config,
        "gpu_request": {
            "gpu_ids": [gpu_a, gpu_b],
            "expected_peak_mib_per_wave": EXPECTED_R1_PEAK_MIB,
            "safety_margin_mib": SAFETY_MARGIN_MIB,
            "gpu1_preserved": True,
            "snapshot": snapshot(records),
        },
        "wave_commands": [wave_a, wave_b],
        "snapshot_manifest": str(snapshot_path.relative_to(ROOT)),
        "launched_at": utc_now(),
    }
    outer_command = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r1_sid_runtime",
        "launch",
        "--gpu-a",
        str(gpu_a),
        "--gpu-b",
        str(gpu_b),
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=outer_command,
        source_paths=source_paths(),
        config=launch_config,
    )
    verify_run_snapshot(ROOT, manifest)
    first = launch_background_tmux(
        experiment_id="s17_s2r_r1_wave_a",
        argv=wave_a,
        cwd=ROOT,
        tmux_session=sessions["wave_a"],
    )
    second = launch_background_tmux(
        experiment_id="s17_s2r_r1_wave_b",
        argv=wave_b,
        cwd=ROOT,
        tmux_session=sessions["wave_b"],
    )
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": "S17-2R",
            "kind": "R1_CONTRACT_AND_RESOURCE_PROFILE",
            "started_at": launch_config["launched_at"],
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "gpu_ids": [gpu_a, gpu_b],
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_2R_R1_BACKGROUND_STARTED",
        stage="r1_contract_and_resource_profile",
        progress={"current": 0, "total": 4, "unit": "family"},
        gpu_ids=[gpu_a, gpu_b],
        gpu_snapshot={"records": snapshot(records)},
        tmux_session=f"{first},{second}",
        process_alive=True,
        run_snapshot_manifest=str(manifest.relative_to(ROOT)),
        expected_peak_mib=EXPECTED_R1_PEAK_MIB,
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    result = {
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
        "tmux_sessions": [first, second],
        "gpu_ids": [gpu_a, gpu_b],
        "wave_commands": [wave_a, wave_b],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def recover(
    gpu_a: int,
    gpu_b: int,
    *,
    recovery_id: str = "run-0002",
    recovery_of: str = "run-0001",
) -> dict:
    """Run the two missing R1 families in an isolated recovery attempt."""

    if gpu_a == gpu_b or 1 in {gpu_a, gpu_b}:
        raise PermissionError("R1 recovery needs two distinct GPUs and must preserve GPU1")
    records = query_gpus()
    by_id = {row.index: row for row in records}
    required = EXPECTED_R1_PEAK_MIB + SAFETY_MARGIN_MIB
    for gpu_id in (gpu_a, gpu_b):
        if gpu_id not in by_id or by_id[gpu_id].free_mib < required:
            available = by_id[gpu_id].free_mib if gpu_id in by_id else 0
            raise RuntimeError(
                f"GPU {gpu_id} has {available} MiB free; recovery requires {required} MiB"
            )

    if recovery_id not in {"run-0002", "run-0003"}:
        raise ValueError("R1 recovery attempt must be run-0002 or run-0003")
    if recovery_of not in {"run-0001", "run-0002"}:
        raise ValueError("R1 recovery parent must be run-0001 or run-0002")
    recovery_root = ROOT / "artifacts/phase17/s2r_r1" / recovery_id
    if recovery_root.exists():
        raise FileExistsError(
            f"R1 {recovery_id} already exists; implicit retry is forbidden"
        )
    attempt_suffix = recovery_id.replace("run-", "r")
    sessions = [
        f"s17_s2r_r1_{attempt_suffix}_diff",
        f"s17_s2r_r1_{attempt_suffix}_set",
    ]
    occupied = [name for name in sessions if _session_exists(name)]
    if occupied:
        raise FileExistsError(f"R1 recovery tmux sessions already exist: {occupied}")

    snapshot_path = (
        ROOT
        / "artifacts/phase17/snapshots"
        / EXPERIMENT_ID
        / recovery_id
        / "manifest.json"
    )
    commands = []
    for family, gpu_id in (("diffgrm", gpu_a), ("setrec", gpu_b)):
        commands.append(
            [
                "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
                "-m",
                "experiment.phase17.protocol.s2r_r1_sid_runtime",
                "worker",
                "--family",
                family,
                "--physical-gpu",
                str(gpu_id),
                "--snapshot",
                str(snapshot_path),
                "--output-root",
                str(recovery_root),
            ]
        )
    base_config = json.loads((OUTPUT_ROOT / "frozen_config.json").read_text(encoding="utf-8"))
    recovery_config = {
        **base_config,
        "attempt_id": recovery_id,
        "recovery_of": recovery_of,
        "recovery_scope": (
            "missing DiffGRM and SETRec families only; objectives, data, seed, "
            "epochs, batch, beam and evaluator unchanged; persistent failure logging; "
            "parallel decoder call excludes Latte-only aggregation keyword"
        ),
        "completed_run_0001_families": ["latte", "gryphon"],
        "commands": commands,
        "output_root": str(recovery_root.relative_to(ROOT)),
        "gpu_request": {
            "gpu_ids": [gpu_a, gpu_b],
            "expected_peak_mib_per_job": EXPECTED_R1_PEAK_MIB,
            "safety_margin_mib": SAFETY_MARGIN_MIB,
            "gpu1_preserved": True,
            "snapshot": snapshot(records),
        },
        "snapshot_manifest": str(snapshot_path.relative_to(ROOT)),
        "created_at": utc_now(),
    }
    outer_command = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r1_sid_runtime",
        "recover",
        "--gpu-a",
        str(gpu_a),
        "--gpu-b",
        str(gpu_b),
        "--attempt-id",
        recovery_id,
        "--recovery-of",
        recovery_of,
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=recovery_id,
        command=outer_command,
        source_paths=source_paths(),
        config=recovery_config,
    )
    verify_run_snapshot(ROOT, manifest)
    launched = [
        launch_background_tmux(
            experiment_id=session,
            argv=command,
            cwd=ROOT,
            tmux_session=session,
        )
        for session, command in zip(sessions, commands)
    ]
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": recovery_id,
            "step_id": "S17-2R",
            "kind": "R1_ENGINEERING_RECOVERY",
            "started_at": recovery_config["created_at"],
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "gpu_ids": [gpu_a, gpu_b],
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
            "recovery_of": recovery_of,
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_2R_R1_RECOVERY_STARTED",
        stage="r1_missing_family_recovery",
        progress={"current": 2, "total": 4, "unit": "family"},
        gpu_ids=[gpu_a, gpu_b],
        gpu_snapshot={"records": snapshot(records)},
        tmux_session=",".join(launched),
        process_alive=True,
        run_snapshot_manifest=str(manifest.relative_to(ROOT)),
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    result = {
        "attempt_id": recovery_id,
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
        "tmux_sessions": launched,
        "gpu_ids": [gpu_a, gpu_b],
        "commands": commands,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def cpu_smoke() -> dict:
    """Exercise every R1 objective and decoder on real frozen catalog IDs."""

    config = json.loads((OUTPUT_ROOT / "frozen_config.json").read_text(encoding="utf-8"))
    base_sid = json.loads(SID_PATH.read_text(encoding="utf-8"))
    set_sid = json.loads(SET_SID_PATH.read_text(encoding="utf-8"))
    users = parse_shadow_sequences(SEQUENCE_PATH)
    train_examples, _ = build_examples(users)
    results = {}
    for family in sorted(FAMILIES):
        codec = (
            set_codec_from_payload(set_sid)
            if family == "setrec"
            else codec_from_payload(base_sid)
        )
        for arm in FAMILIES[family]:
            set_seed(SEED)
            dataset = SIDSequenceDataset(
                train_examples[:2],
                codec,
                latte_training=arm == "latte_full",
                seed=SEED,
            )
            batch = collate_sid_batch([dataset[0], dataset[1]])
            if arm in PARALLEL_ARMS:
                model = S2RParallelIDModel(
                    codec,
                    arm=arm,
                    config=parallel_smoke_config(codec, capacity="tiny"),
                )
            else:
                model = S2RSemanticIDModel(
                    codec,
                    arm=arm,
                    config=smoke_t5_config(codec, capacity="tiny"),
                )
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                target_item_index=batch["target_item_index"],
            )
            output.loss.backward()
            model.eval()
            with torch.no_grad():
                predictions = model.generate_ranked(
                    input_ids=batch["input_ids"][:1],
                    attention_mask=batch["attention_mask"][:1],
                    num_beams=3,
                    top_k=3,
                )[0]
            results[arm] = {
                "finite_loss": math.isfinite(float(output.loss.item())),
                "loss": float(output.loss.item()),
                "gradient_nonzero": model_gradient_norm(model) > 0.0,
                "predictions": [row.item_id for row in predictions],
                "catalog_item_resolution": bool(predictions)
                and all(row.item_id in codec.item_to_code for row in predictions),
                "parameter_count": parameter_count(model),
            }
            del model
    payload = {
        "schema_version": "phase17.s17_2r_r1_cpu_smoke.v1",
        "formal_result_eligible": False,
        "catalog_items": config["catalog_items"],
        "arms": results,
        "all_finite": all(row["finite_loss"] for row in results.values()),
        "all_resolve_catalog_items": all(
            row["catalog_item_resolution"] for row in results.values()
        ),
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "completed_at": utc_now(),
    }
    atomic_json(OUTPUT_ROOT / "cpu_smoke.json", payload)
    return payload


def finalize_r1() -> dict:
    family_paths = {
        "latte": OUTPUT_ROOT / "latte/summary.json",
        "gryphon": OUTPUT_ROOT / "gryphon/summary.json",
        "diffgrm": ROOT / "artifacts/phase17/s2r_r1/run-0003/diffgrm/summary.json",
        "setrec": ROOT / "artifacts/phase17/s2r_r1/run-0003/setrec/summary.json",
    }
    missing = [str(path) for path in family_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R1 family summaries are missing: {missing}")
    families = {
        family: json.loads(path.read_text(encoding="utf-8"))
        for family, path in family_paths.items()
    }
    arms = {
        arm: result
        for family in families.values()
        for arm, result in family["arms"].items()
    }
    latte_gradient_path = (
        OUTPUT_ROOT / "latte/latte_full/latte_gradient_reachability.json"
    )
    latte_gradient = json.loads(latte_gradient_path.read_text(encoding="utf-8"))
    all_losses_decreased = all(
        result["training"]["loss_decreased"] for result in arms.values()
    )
    all_predictions_nonempty = all(
        result["evaluation"]["prediction_file_nonempty"]
        and result["evaluation"]["prediction_rows"] == 100
        for result in arms.values()
    )
    all_items_resolved = all(
        result["evaluation"]["catalog_item_resolution"]
        and result["evaluation"]["metrics"]["valid_item_rate"] == 1.0
        for result in arms.values()
    )
    treatment_gradients = {
        "latte_full": latte_gradient["candidate_specific_gradient_nonzero"],
        "gryphon_item": arms["gryphon_item"]["training"][
            "candidate_specific_gradient_nonzero"
        ],
        "diffgrm_masked": arms["diffgrm_masked"]["training"][
            "candidate_specific_gradient_nonzero"
        ],
        "setrec_full": arms["setrec_full"]["training"][
            "candidate_specific_gradient_nonzero"
        ],
    }
    peak_mib = max(
        result["training"]["peak_allocated_mib"] for result in arms.values()
    )
    r1_pass = (
        all_losses_decreased
        and all_predictions_nonempty
        and all_items_resolved
        and all(treatment_gradients.values())
    )
    payload = {
        "schema_version": "phase17.s17_2r_r1_closeout.v1",
        "step_id": "S17-2R",
        "gate": "R1",
        "state": "PASS" if r1_pass else "FAIL",
        "formal_result_eligible": False,
        "effect_metrics_used_for_selection": False,
        "attempt_chain": [
            {
                "attempt_id": "run-0001",
                "completed_families": ["latte", "gryphon"],
                "partial_failure": "missing persistent stderr for second wave",
            },
            {
                "attempt_id": "run-0002",
                "completed_families": [],
                "failure": "Latte-only aggregation keyword crossed parallel decoder interface",
            },
            {
                "attempt_id": "run-0003",
                "completed_families": ["diffgrm", "setrec"],
                "recovery_scope": "interface dispatch only; science configuration unchanged",
            },
        ],
        "family_summaries": {
            family: str(path.relative_to(ROOT)) for family, path in family_paths.items()
        },
        "checks": {
            "all_training_losses_decreased": all_losses_decreased,
            "all_prediction_files_have_100_users": all_predictions_nonempty,
            "all_predictions_resolve_to_catalog_items": all_items_resolved,
            "treatment_specific_gradients": treatment_gradients,
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
        },
        "resource_profile": {
            "maximum_peak_allocated_mib": peak_mib,
            "parameter_counts": {
                arm: result["training"]["parameter_count"]
                for arm, result in arms.items()
            },
            "mean_generation_batch_seconds": {
                arm: result["evaluation"]["metrics"][
                    "mean_generation_batch_seconds"
                ]
                for arm, result in arms.items()
            },
            "profile_scope": "scaled R1 models only; not faithful-scale memory evidence",
        },
        "mechanism_warnings": [
            "R1 effect metrics are intentionally ignored for architecture selection",
            "base SID needed collision reassignment for 11653 of 11924 catalog items",
            "Latte produced 56.46% duplicate paths and 21.77 mean unique candidates from beam 50",
            "Gryphon same-candidate target rank gain was negative on the two covered users",
            "DiffGRM masked validation Hit@50 was zero in the non-selection smoke",
            "SETRec simultaneous set-token recovery was zero in the non-selection smoke",
        ],
        "r2_preflight_eligible": r1_pass,
        "completed_at": utc_now(),
    }
    output = ROOT / "artifacts/phase17/s2r_r1/r1_summary.json"
    atomic_json(output, payload)
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_2R_R1_PROFILE_COMPLETE_R2_PREFLIGHT",
        stage="r1_complete_r2_preflight",
        progress={"current": 2, "total": 4, "unit": "r0_to_r3"},
        gpu_ids=[],
        tmux_session=None,
        process_alive=False,
        r1_summary=str(output.relative_to(ROOT)),
        r1_pass=r1_pass,
        r2_preflight_eligible=r1_pass,
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _rank(target: str, item_ids: list[str]) -> int | None:
    try:
        return item_ids.index(target) + 1
    except ValueError:
        return None


def _ranking_metrics(targets: list[str], rankings: list[list[str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in (5, 10, 20, 50):
        hits = []
        ndcgs = []
        for target, items in zip(targets, rankings):
            rank = _rank(target, items[:cutoff])
            hits.append(float(rank is not None))
            ndcgs.append(0.0 if rank is None else 1.0 / math.log2(rank + 1.0))
        result[f"hit@{cutoff}"] = mean(hits)
        result[f"ndcg@{cutoff}"] = mean(ndcgs)
    return result


def generation_arguments(arm: str, moved: dict, config: dict) -> dict:
    arguments = {
        "input_ids": moved["input_ids"],
        "attention_mask": moved["attention_mask"],
        "num_beams": config["num_beams"],
        "top_k": config["top_k"],
    }
    if arm not in PARALLEL_ARMS:
        arguments["latte_aggregation"] = "logsumexp"
    return arguments


def train_arm(
    *,
    arm: str,
    codec: SemanticIDCodec,
    train_examples: list,
    device: torch.device,
    config: dict,
) -> tuple[torch.nn.Module, dict]:
    set_seed(SEED)
    if arm in PARALLEL_ARMS:
        model = S2RParallelIDModel(
            codec,
            arm=arm,
            config=parallel_smoke_config(codec, capacity=config["capacity"]),
        ).to(device)
    else:
        model = S2RSemanticIDModel(
            codec,
            arm=arm,
            config=smoke_t5_config(codec, capacity=config["capacity"]),
            item_loss_weight=0.2,
        ).to(device)
    dataset = SIDSequenceDataset(
        train_examples,
        codec,
        latte_training=arm == "latte_full",
        seed=SEED,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    epoch_losses: list[float] = []
    step_seconds: list[float] = []
    scorer_gradients: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for epoch in range(config["epochs"]):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED + epoch),
            collate_fn=collate_sid_batch,
        )
        losses = []
        for batch in loader:
            started = time.perf_counter()
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                target_item_index=batch["target_item_index"],
            )
            output.loss.backward()
            scorer_gradients.append(candidate_gradient_norm(model, arm))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - started)
            losses.append(float(output.loss.detach().cpu().item()))
        epoch_losses.append(mean(losses))
    peak_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )
    training = {
        "epochs": config["epochs"],
        "epoch_losses": epoch_losses,
        "loss_decreased": epoch_losses[-1] < epoch_losses[0],
        "optimizer_steps": len(step_seconds),
        "mean_step_seconds": mean(step_seconds),
        "max_step_seconds": max(step_seconds),
        "peak_allocated_mib": peak_mib,
        "item_scorer_gradient_norm_max": max(scorer_gradients)
        if arm == "gryphon_item"
        else 0.0,
        "item_scorer_gradient_nonzero": max(scorer_gradients) > 0.0
        if arm == "gryphon_item"
        else None,
        "candidate_specific_gradient_norm_max": max(scorer_gradients),
        "candidate_specific_gradient_nonzero": max(scorer_gradients) > 0.0
        if arm in {"latte_full", "gryphon_item", "diffgrm_masked", "setrec_full"}
        else None,
        "parameter_count": parameter_count(model),
    }
    return model, training


def evaluate_arm(
    *,
    arm: str,
    model: torch.nn.Module,
    codec: SemanticIDCodec,
    validation_examples: list,
    device: torch.device,
    config: dict,
    output_dir: Path,
) -> dict:
    dataset = SIDSequenceDataset(
        validation_examples,
        codec,
        latte_training=arm == "latte_full",
        seed=SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_sid_batch,
    )
    targets: list[str] = []
    users: list[str] = []
    native_rows: list[list] = []
    diagnostics: list[dict[str, float]] = []
    generation_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            users.extend(batch["user_id"])
            targets.extend(batch["target_item"])
            moved = _move_batch(batch, device)
            if arm in PARALLEL_ARMS:
                diagnostics.append(
                    model.mechanism_diagnostics(
                        input_ids=moved["input_ids"],
                        attention_mask=moved["attention_mask"],
                        labels=moved["labels"],
                    )
                )
            started = time.perf_counter()
            native_rows.extend(
                model.generate_ranked(**generation_arguments(arm, moved, config))
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds.append(time.perf_counter() - started)

    native_rankings = [[row.item_id for row in ranking] for ranking in native_rows]
    metrics = _ranking_metrics(targets, native_rankings)
    paths = [getattr(row, "path_count", 1) for ranking in native_rows for row in ranking]
    metrics.update(
        valid_item_rate=mean([float(bool(ranking)) for ranking in native_rankings]),
        mean_unique_candidates=mean(len(ranking) for ranking in native_rankings),
        duplicate_path_rate=(sum(paths) - len(paths)) / max(sum(paths), 1),
        multi_path_item_rate=mean([float(value > 1) for value in paths]) if paths else 0.0,
        mean_generation_batch_seconds=mean(generation_seconds),
    )
    if diagnostics:
        for key in sorted(diagnostics[0]):
            metrics[key] = mean(row[key] for row in diagnostics)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.tsv"
    with prediction_path.open("w", encoding="utf-8") as handle:
        handle.write("user_id\ttarget_item\tranked_items\n")
        for user, target, ranking in zip(users, targets, native_rankings):
            handle.write(f"{user}\t{target}\t{' '.join(ranking)}\n")

    result = {
        "arm": arm,
        "metrics": metrics,
        "predictions": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_rows": len(native_rankings),
        "prediction_file_nonempty": prediction_path.stat().st_size > 0,
        "catalog_item_resolution": all(
            item in codec.item_to_code for ranking in native_rankings for item in ranking
        ),
    }

    if arm == "gryphon_item":
        beam_rows = [
            sorted(ranking, key=lambda row: (-row.beam_score, row.item_id))
            for ranking in native_rows
        ]
        beam_rankings = [[row.item_id for row in ranking] for ranking in beam_rows]
        control_metrics = _ranking_metrics(targets, beam_rankings)
        rank_gains = []
        for target, native, beam in zip(targets, native_rankings, beam_rankings):
            native_rank = _rank(target, native)
            beam_rank = _rank(target, beam)
            if native_rank is not None and beam_rank is not None:
                rank_gains.append(float(beam_rank - native_rank))
        control_path = output_dir / "same_candidates_beam_control.tsv"
        with control_path.open("w", encoding="utf-8") as handle:
            handle.write("user_id\ttarget_item\tranked_items\n")
            for user, target, ranking in zip(users, targets, beam_rankings):
                handle.write(f"{user}\t{target}\t{' '.join(ranking)}\n")
        candidate_hashes_native = [
            hashlib.sha256("\0".join(sorted(row)).encode()).hexdigest()
            for row in native_rankings
        ]
        candidate_hashes_control = [
            hashlib.sha256("\0".join(sorted(row)).encode()).hexdigest()
            for row in beam_rankings
        ]
        result["same_candidate_control"] = {
            "metrics": control_metrics,
            "predictions": str(control_path.relative_to(ROOT)),
            "prediction_sha256": sha256_file(control_path),
            "candidate_sets_identical": candidate_hashes_native
            == candidate_hashes_control,
            "mean_target_rank_gain": mean(rank_gains) if rank_gains else 0.0,
            "comparable_target_users": len(rank_gains),
        }
    return result


def worker(
    family: str,
    physical_gpu: int,
    snapshot_path: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if physical_gpu == 1:
        raise PermissionError("GPU1 is reserved for the existing non-scientific repeat")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the S17-2R R1 worker")
    if snapshot_path is not None:
        verify_run_snapshot(ROOT, snapshot_path)
        config_path = snapshot_path.parent / "config.json"
    else:
        config_path = OUTPUT_ROOT / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sid = json.loads(SID_PATH.read_text(encoding="utf-8"))
    if sha256_file(SID_PATH) != config["sid_sha256"]:
        raise RuntimeError("Semantic-ID artifact changed after R1 freeze")
    if family == "setrec":
        set_sid = json.loads(SET_SID_PATH.read_text(encoding="utf-8"))
        if sha256_file(SET_SID_PATH) != config["set_sid_sha256"]:
            raise RuntimeError("CF+semantic SID artifact changed after R1 freeze")
        codec = set_codec_from_payload(set_sid)
    else:
        codec = codec_from_payload(sid)
    users = parse_shadow_sequences(SEQUENCE_PATH)
    train_examples, validation_examples = build_examples(users)
    device = torch.device("cuda:0")
    allowed_output_parent = (ROOT / "artifacts/phase17/s2r_r1").resolve()
    output_root = output_root.resolve()
    if output_root.parent != allowed_output_parent:
        raise PermissionError("R1 output root must be one isolated run-NNNN directory")
    family_dir = output_root / family
    if family_dir.exists():
        raise FileExistsError(f"R1 family output already exists: {family_dir}")
    family_dir.mkdir(parents=True)

    arm_results = {}
    try:
        for arm in FAMILIES[family]:
            arm_dir = family_dir / arm
            model, training = train_arm(
                arm=arm,
                codec=codec,
                train_examples=train_examples,
                device=device,
                config=config,
            )
            evaluation = evaluate_arm(
                arm=arm,
                model=model,
                codec=codec,
                validation_examples=validation_examples,
                device=device,
                config=config,
                output_dir=arm_dir,
            )
            torch.save(model.state_dict(), arm_dir / "model_state.pt")
            arm_results[arm] = {"training": training, "evaluation": evaluation}
            del model
            torch.cuda.empty_cache()
    except Exception as error:
        failure = {
            "schema_version": "phase17.s17_2r_r1_failure.v1",
            "family": family,
            "physical_gpu": physical_gpu,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
            "formal_result_eligible": False,
            "test_read": False,
            "sports_read": False,
        }
        atomic_json(family_dir / "failure.json", failure)
        raise

    family_result = {
        "schema_version": "phase17.s17_2r_r1_family.v1",
        "family": family,
        "physical_gpu": physical_gpu,
        "device_name": torch.cuda.get_device_name(device),
        "formal_result_eligible": False,
        "result_selection_eligible": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "arms": arm_results,
        "completed_at": utc_now(),
    }
    atomic_json(family_dir / "summary.json", family_result)
    print(json.dumps(family_result, ensure_ascii=False, indent=2, sort_keys=True))
    return family_result


def wave(
    families: str,
    physical_gpu: int,
    snapshot_path: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    requested = [value.strip() for value in families.split(",") if value.strip()]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("wave needs a non-empty, duplicate-free family list")
    for family in requested:
        worker(family, physical_gpu, snapshot_path, output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("cpu-smoke")
    subparsers.add_parser("finalize-r1")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--gpu-a", type=int, required=True)
    launch_parser.add_argument("--gpu-b", type=int, required=True)
    recovery_parser = subparsers.add_parser("recover")
    recovery_parser.add_argument("--gpu-a", type=int, required=True)
    recovery_parser.add_argument("--gpu-b", type=int, required=True)
    recovery_parser.add_argument("--attempt-id", default="run-0002")
    recovery_parser.add_argument("--recovery-of", default="run-0001")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    worker_parser.add_argument("--physical-gpu", type=int, required=True)
    worker_parser.add_argument("--snapshot", type=Path)
    worker_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    wave_parser = subparsers.add_parser("wave")
    wave_parser.add_argument("--families", required=True)
    wave_parser.add_argument("--physical-gpu", type=int, required=True)
    wave_parser.add_argument("--snapshot", type=Path)
    wave_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "cpu-smoke":
        print(json.dumps(cpu_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "finalize-r1":
        finalize_r1()
    elif args.command == "launch":
        launch(args.gpu_a, args.gpu_b)
    elif args.command == "recover":
        recover(
            args.gpu_a,
            args.gpu_b,
            recovery_id=args.attempt_id,
            recovery_of=args.recovery_of,
        )
    elif args.command == "worker":
        worker(args.family, args.physical_gpu, args.snapshot, args.output_root)
    else:
        wave(args.families, args.physical_gpu, args.snapshot, args.output_root)


if __name__ == "__main__":
    main()
