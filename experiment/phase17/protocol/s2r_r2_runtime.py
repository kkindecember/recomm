#!/usr/bin/env python3
"""S17-2R R2 contract preparation and profile-first launcher."""

from __future__ import annotations

import argparse
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

import numpy as np
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
from experiment.phase17.core.s2r_architectures import (  # noqa: E402
    S2RSemanticIDModel,
    parameter_count,
    smoke_t5_config,
)
from experiment.phase17.core.s2r_parallel_architectures import (  # noqa: E402
    PARALLEL_ARMS,
    S2RParallelIDModel,
    parallel_smoke_config,
)
from experiment.phase17.core.s2r_sid import (  # noqa: E402
    SIDSequenceDataset,
    SemanticIDCodec,
    build_r2_examples,
    build_residual_kmeans_ids,
    build_train_only_cf_codes,
    collate_sid_batch,
    parse_item_text,
    parse_shadow_sequences,
    read_cohort_user_ids,
    sha256_file,
    train_catalog_items,
)
from experiment.phase17.core.status_writer import (  # noqa: E402
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)
from experiment.phase17.protocol.s2r_r1_sid_runtime import (  # noqa: E402
    candidate_gradient_norm,
    model_gradient_norm,
)


SEED = 2023
EXPERIMENT_ID = "s17_s2r_architecture_reselection"
PROFILE_ATTEMPT_ID = "r2-profile-0001"
DATA_DIR = ROOT / "artifacts/phase17/s2r_preflight/data/Toys_s17_d0_3000"
SEQUENCE_PATH = DATA_DIR / "user_sequence.txt"
ITEM_TEXT_PATH = DATA_DIR / "item_plain_text.txt"
CONTRACT_DIR = ROOT / "artifacts/phase17/s2r_preflight/r2_contract"
CONTRACT_PATH = CONTRACT_DIR / "manifest.json"
EARLY_STOP_PATH = CONTRACT_DIR / "early_stop_user_ids.txt"
CONFIG_PATH = ROOT / "experiment/phase17/config/s17_s2r_r2_screen.json"
EMBEDDING_PATH = ROOT / "artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt"
SID_PATH = ROOT / "artifacts/phase17/s2r_preflight/sid/r2_bge_rq256x3_ci.json"
SET_SID_PATH = ROOT / "artifacts/phase17/s2r_preflight/sid/r2_cf32_bge_rq256x3_ci.json"
PREFLIGHT_DIR = ROOT / "artifacts/phase17/s2r_preflight/r2"
PROFILE_ROOT = ROOT / "artifacts/phase17/s2r_r2/profile/run-0001"
FAMILIES = {
    "diffgrm": ("diffgrm_ar_control", "diffgrm_masked"),
    "gryphon": ("gryphon_item",),
    "latte": ("psid_control", "latte_full"),
    "setrec": ("setrec_ar_control", "setrec_full"),
}
EXPECTED_PROFILE_PEAK_MIB = 8192
SAFETY_MARGIN_MIB = 4096


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frozen_semantic_embeddings(
    item_ids: list[str], item_text_sha256: str, tokenizer: dict
) -> np.ndarray:
    configured_path = ROOT / tokenizer["semantic_embedding_path"]
    if configured_path != EMBEDDING_PATH:
        raise RuntimeError("R2 semantic embedding path differs from the frozen path")
    actual_sha256 = sha256_file(configured_path)
    if actual_sha256 != tokenizer["semantic_embedding_sha256"]:
        raise RuntimeError("R2 semantic embedding SHA256 mismatch")
    artifact = torch.load(configured_path, map_location="cpu")
    required = {"item_ids", "embeddings", "text_source_sha256", "l2_normalized"}
    if not required.issubset(artifact):
        raise RuntimeError("R2 semantic embedding artifact lacks required metadata")
    artifact_ids = [str(item_id) for item_id in artifact["item_ids"]]
    if len(artifact_ids) != len(set(artifact_ids)) or set(artifact_ids) != set(item_ids):
        raise RuntimeError("R2 semantic embedding item IDs differ from item text catalog")
    if artifact["text_source_sha256"] != item_text_sha256:
        raise RuntimeError("R2 semantic embedding was built from different item text")
    if artifact["l2_normalized"] is not True:
        raise RuntimeError("R2 semantic embeddings are not frozen L2-normalized vectors")
    tensor = artifact["embeddings"]
    if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[0] != len(artifact_ids):
        raise RuntimeError("R2 semantic embedding tensor has an invalid shape")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("R2 semantic embeddings contain non-finite values")
    row_by_item = {item_id: row for row, item_id in enumerate(artifact_ids)}
    rows = [row_by_item[item_id] for item_id in item_ids]
    return tensor[rows].detach().cpu().numpy().astype(np.float32, copy=False)


def build_or_load_sid(config: dict) -> dict:
    tokenizer = config["tokenizer"]
    if SID_PATH.exists():
        payload = json.loads(SID_PATH.read_text(encoding="utf-8"))
        if payload["sequence_sha256"] != sha256_file(SEQUENCE_PATH):
            raise RuntimeError("R2 sequence changed after SID construction")
        if payload["item_text_sha256"] != sha256_file(ITEM_TEXT_PATH):
            raise RuntimeError("R2 item text changed after SID construction")
        if payload["embedding_source"]["sha256"] != sha256_file(EMBEDDING_PATH):
            raise RuntimeError("R2 semantic embedding changed after SID construction")
        expected_prefix = [tokenizer["semantic_codebook_size"]] * tokenizer["semantic_codebooks"]
        if payload["summary"]["codebook_sizes"][: tokenizer["semantic_codebooks"]] != expected_prefix:
            raise RuntimeError("R2 SID codebook sizes differ from frozen tokenizer config")
        if payload["summary"]["collision_resolution"] != "append_group_ordinal":
            raise RuntimeError("R2 SID does not preserve semantic digits with collision suffix")
        return payload
    users = parse_shadow_sequences(SEQUENCE_PATH)
    item_text = parse_item_text(ITEM_TEXT_PATH)
    item_ids = sorted(item_text)
    item_text_sha256 = sha256_file(ITEM_TEXT_PATH)
    embeddings = load_frozen_semantic_embeddings(
        item_ids, item_text_sha256, tokenizer
    )
    item_to_code, summary = build_residual_kmeans_ids(
        item_ids,
        embeddings,
        train_catalog_items(users),
        n_codebooks=tokenizer["semantic_codebooks"],
        codebook_size=tokenizer["semantic_codebook_size"],
        seed=SEED,
        embedding_method=tokenizer["semantic_embedding_method"],
        collision_resolution="append_group_ordinal",
    )
    payload = {
        "schema_version": "phase17.s17_2r_sid.v2",
        "gate": "R2",
        "formal_result_eligible": True,
        "fidelity": "R2_FROZEN_BGE_RQ256X3_COLLISION_SUFFIX_PROFILE_FIRST",
        "sequence_path": str(SEQUENCE_PATH.relative_to(ROOT)),
        "sequence_sha256": sha256_file(SEQUENCE_PATH),
        "item_text_path": str(ITEM_TEXT_PATH.relative_to(ROOT)),
        "item_text_sha256": item_text_sha256,
        "embedding_source": {
            "path": str(EMBEDDING_PATH.relative_to(ROOT)),
            "sha256": sha256_file(EMBEDDING_PATH),
            "method": tokenizer["semantic_embedding_method"],
            "fit_scope": "frozen item text only; no sequence target labels",
        },
        "summary": asdict(summary),
        "item_to_code": {item: list(code) for item, code in item_to_code.items()},
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
    }
    atomic_json(SID_PATH, payload)
    return payload


def build_or_load_set_sid(base_sid: dict, config: dict) -> dict:
    if SET_SID_PATH.exists():
        payload = json.loads(SET_SID_PATH.read_text(encoding="utf-8"))
        if payload["base_sid_sha256"] != sha256_file(SID_PATH):
            raise RuntimeError("R2 base SID changed after set tokenizer construction")
        return payload
    users = parse_shadow_sequences(SEQUENCE_PATH)
    item_ids = sorted(base_sid["item_to_code"])
    cf_codes, summary = build_train_only_cf_codes(
        item_ids,
        users,
        codebook_size=config["tokenizer"]["set_cf_codebook_size"],
        hash_buckets=64,
        seed=SEED,
    )
    payload = {
        "schema_version": "phase17.s17_2r_set_sid.v2",
        "gate": "R2",
        "formal_result_eligible": True,
        "fidelity": "R2_TRAIN_PREFIX_CF_PLUS_FROZEN_BGE_RQ_SEMANTIC",
        "base_sid_path": str(SID_PATH.relative_to(ROOT)),
        "base_sid_sha256": sha256_file(SID_PATH),
        "sequence_sha256": sha256_file(SEQUENCE_PATH),
        "cf_summary": asdict(summary),
        "codebook_sizes": [summary.codebook_size, *base_sid["summary"]["codebook_sizes"]],
        "item_to_code": {
            item: [cf_codes[item], *base_sid["item_to_code"][item]]
            for item in item_ids
        },
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
    }
    atomic_json(SET_SID_PATH, payload)
    return payload


def codec_from_sid(payload: dict, *, set_codec: bool = False) -> SemanticIDCodec:
    sizes = payload["codebook_sizes"] if set_codec else payload["summary"]["codebook_sizes"]
    return SemanticIDCodec(
        payload["item_to_code"], sizes, n_latent_tokens=8, max_history_items=20
    )


def create_model(arm: str, codec: SemanticIDCodec, capacity: str) -> torch.nn.Module:
    if arm in PARALLEL_ARMS:
        return S2RParallelIDModel(
            codec, arm=arm, config=parallel_smoke_config(codec, capacity=capacity)
        )
    return S2RSemanticIDModel(
        codec,
        arm=arm,
        config=smoke_t5_config(codec, capacity=capacity),
        item_loss_weight=0.2,
    )


def prepare() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not contract["cohorts_partition_selected_users"]:
        raise RuntimeError("R2 cohort contract is not complete")
    rejected_sid = ROOT / "artifacts/phase17/s2r_preflight/sid/r2_tfidf_rq32x3.json"
    if rejected_sid.exists():
        rejected_payload = json.loads(rejected_sid.read_text(encoding="utf-8"))
        atomic_json(
            PREFLIGHT_DIR / "rejected_sid32.json",
            {
                "schema_version": "phase17.s17_2r_rejected_tokenizer.v1",
                "path": str(rejected_sid.relative_to(ROOT)),
                "sha256": sha256_file(rejected_sid),
                "reason": "R2 effect screen rejected: excessive collision repair changes code geometry",
                "collisions_before_resolution": rejected_payload["summary"]["collisions_before_resolution"],
                "reassigned_items": rejected_payload["summary"]["reassigned_items"],
                "catalog_items": rejected_payload["summary"]["catalog_items"],
                "selected_for_r2": False,
            },
        )
    rejected_reassignment_sid = ROOT / "artifacts/phase17/s2r_preflight/sid/r2_bge_rq256x3.json"
    if rejected_reassignment_sid.exists():
        rejected_payload = json.loads(rejected_reassignment_sid.read_text(encoding="utf-8"))
        atomic_json(
            PREFLIGHT_DIR / "rejected_sid_reassignment.json",
            {
                "schema_version": "phase17.s17_2r_rejected_tokenizer.v1",
                "path": str(rejected_reassignment_sid.relative_to(ROOT)),
                "sha256": sha256_file(rejected_reassignment_sid),
                "reason": "R2 rejected: unique nearest-code reassignment perturbs semantic geometry for most catalog items",
                "collisions_before_resolution": rejected_payload["summary"]["collisions_before_resolution"],
                "reassigned_items": rejected_payload["summary"]["reassigned_items"],
                "catalog_items": rejected_payload["summary"]["catalog_items"],
                "selected_for_r2": False,
            },
        )
    base_sid = build_or_load_sid(config)
    set_sid = build_or_load_set_sid(base_sid, config)
    base_codec = codec_from_sid(base_sid)
    set_codec = codec_from_sid(set_sid, set_codec=True)
    capacities = {
        "semantic": config["optimization"]["semantic_capacity"],
        "parallel": config["optimization"]["parallel_capacity"],
    }
    parameter_counts = {}
    for family, arms in FAMILIES.items():
        codec = set_codec if family == "setrec" else base_codec
        for arm in arms:
            capacity = capacities["parallel"] if arm in PARALLEL_ARMS else capacities["semantic"]
            model = create_model(arm, codec, capacity)
            parameter_counts[arm] = parameter_count(model)
            del model
    payload = {
        **config,
        "contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "sid_path": str(SID_PATH.relative_to(ROOT)),
        "sid_sha256": sha256_file(SID_PATH),
        "set_sid_path": str(SET_SID_PATH.relative_to(ROOT)),
        "set_sid_sha256": sha256_file(SET_SID_PATH),
        "parameter_counts": parameter_counts,
        "profile_required": True,
        "screen_gpu_started": False,
        "prepared_at": utc_now(),
    }
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(PREFLIGHT_DIR / "frozen_config.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _batch(examples: list, codec: SemanticIDCodec, arm: str, count: int) -> dict:
    dataset = SIDSequenceDataset(
        examples[:count], codec, latte_training=arm == "latte_full", seed=SEED
    )
    return collate_sid_batch([dataset[index] for index in range(len(dataset))])


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def profile_family(
    family: str,
    physical_gpu: int,
    snapshot_path: Path,
    output_root: Path = PROFILE_ROOT,
) -> dict:
    if family not in FAMILIES or physical_gpu == 1:
        raise PermissionError("invalid R2 profile family or reserved GPU1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for R2 profile")
    verify_run_snapshot(ROOT, snapshot_path)
    config = json.loads((snapshot_path.parent / "config.json").read_text(encoding="utf-8"))
    base_sid = json.loads(SID_PATH.read_text(encoding="utf-8"))
    set_sid = json.loads(SET_SID_PATH.read_text(encoding="utf-8"))
    if sha256_file(SID_PATH) != config["sid_sha256"]:
        raise RuntimeError("R2 SID changed after profile freeze")
    codec = codec_from_sid(set_sid, set_codec=True) if family == "setrec" else codec_from_sid(base_sid)
    users = parse_shadow_sequences(SEQUENCE_PATH)
    early_stop_ids = tuple(
        line.strip() for line in EARLY_STOP_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    train, early_stop, _ = build_r2_examples(users, early_stop_ids)
    family_dir = output_root / family
    if family_dir.exists():
        raise FileExistsError(f"R2 profile output exists: {family_dir}")
    family_dir.mkdir(parents=True)
    results = {}
    device = torch.device("cuda:0")
    try:
        for arm in FAMILIES[family]:
            set_seed(SEED)
            capacity = (
                config["optimization"]["parallel_capacity"]
                if arm in PARALLEL_ARMS
                else config["optimization"]["semantic_capacity"]
            )
            model = create_model(arm, codec, capacity).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["optimization"]["learning_rate"]
            )
            train_batch = _to_device(
                _batch(train, codec, arm, config["optimization"]["train_batch_size"]),
                device,
            )
            eval_batch = _to_device(
                _batch(
                    early_stop,
                    codec,
                    arm,
                    config["optimization"]["evaluation_batch_size"],
                ),
                device,
            )
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            output = model(
                input_ids=train_batch["input_ids"],
                attention_mask=train_batch["attention_mask"],
                labels=train_batch["labels"],
                target_item_index=train_batch["target_item_index"],
            )
            output.loss.backward()
            gradient_norm = candidate_gradient_norm(model, arm)
            total_gradient_norm = model_gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - started
            model.eval()
            started = time.perf_counter()
            arguments = {
                "input_ids": eval_batch["input_ids"],
                "attention_mask": eval_batch["attention_mask"],
                "num_beams": config["optimization"]["num_beams"],
                "top_k": config["optimization"]["top_k"],
            }
            if arm not in PARALLEL_ARMS:
                arguments["latte_aggregation"] = "logsumexp"
            with torch.no_grad():
                predictions = model.generate_ranked(**arguments)
            torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - started
            results[arm] = {
                "parameter_count": parameter_count(model),
                "train_batch_size": len(train_batch["user_id"]),
                "eval_batch_size": len(eval_batch["user_id"]),
                "loss": float(output.loss.detach().cpu().item()),
                "total_gradient_norm": total_gradient_norm,
                "candidate_gradient_norm": gradient_norm,
                "train_step_seconds": train_seconds,
                "generation_batch_seconds": generation_seconds,
                "peak_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                ),
                "prediction_rows": len(predictions),
                "all_prediction_rows_nonempty": all(bool(row) for row in predictions),
            }
            del model, optimizer
            torch.cuda.empty_cache()
        payload = {
            "schema_version": "phase17.s17_2r_r2_profile.v1",
            "family": family,
            "physical_gpu": physical_gpu,
            "device_name": torch.cuda.get_device_name(device),
            "arms": results,
            "formal_result_eligible": False,
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
            "completed_at": utc_now(),
        }
        atomic_json(family_dir / "summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload
    except Exception as error:
        atomic_json(
            family_dir / "failure.json",
            {
                "schema_version": "phase17.s17_2r_r2_profile_failure.v1",
                "family": family,
                "physical_gpu": physical_gpu,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
            },
        )
        raise


def _session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def source_paths() -> list[Path]:
    return [
        ROOT / "experiment/phase17/protocol/s2r_r2_runtime.py",
        ROOT / "experiment/phase17/protocol/s2r_r2_contract.py",
        ROOT / "experiment/phase17/core/s2r_sid.py",
        ROOT / "experiment/phase17/core/s2r_architectures.py",
        ROOT / "experiment/phase17/core/s2r_parallel_architectures.py",
        ROOT / "experiment/phase17/core/s2r_r2_evaluator.py",
        CONFIG_PATH,
        CONTRACT_PATH,
        PREFLIGHT_DIR / "frozen_config.json",
    ]


def launch_profile(gpu_a: int, gpu_b: int) -> dict:
    if gpu_a == gpu_b or 1 in {gpu_a, gpu_b}:
        raise PermissionError("R2 profile must use distinct non-GPU1 devices")
    records = query_gpus()
    by_id = {row.index: row for row in records}
    required = EXPECTED_PROFILE_PEAK_MIB + SAFETY_MARGIN_MIB
    for gpu_id in (gpu_a, gpu_b):
        if gpu_id not in by_id or by_id[gpu_id].free_mib < required:
            free = by_id[gpu_id].free_mib if gpu_id in by_id else 0
            raise RuntimeError(f"GPU {gpu_id} has {free} MiB free; profile needs {required}")
    if PROFILE_ROOT.exists():
        raise FileExistsError("R2 profile run-0001 exists; implicit retry is forbidden")
    snapshot_path = (
        ROOT / "artifacts/phase17/snapshots" / EXPERIMENT_ID / PROFILE_ATTEMPT_ID / "manifest.json"
    )
    commands = []
    for families, gpu_id in (("latte,diffgrm", gpu_a), ("gryphon,setrec", gpu_b)):
        commands.append(
            [
                "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
                "-m",
                "experiment.phase17.protocol.s2r_r2_runtime",
                "profile-wave",
                "--families",
                families,
                "--physical-gpu",
                str(gpu_id),
                "--snapshot",
                str(snapshot_path),
            ]
        )
    config = json.loads((PREFLIGHT_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    frozen = {
        **config,
        "profile_attempt_id": PROFILE_ATTEMPT_ID,
        "profile_commands": commands,
        "profile_gpu_request": {
            "gpu_ids": [gpu_a, gpu_b],
            "expected_peak_mib": EXPECTED_PROFILE_PEAK_MIB,
            "safety_margin_mib": SAFETY_MARGIN_MIB,
            "gpu1_preserved": True,
            "snapshot": snapshot(records),
        },
    }
    outer = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_runtime",
        "launch-profile",
        "--gpu-a",
        str(gpu_a),
        "--gpu-b",
        str(gpu_b),
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=PROFILE_ATTEMPT_ID,
        command=outer,
        source_paths=source_paths(),
        config=frozen,
    )
    verify_run_snapshot(ROOT, manifest)
    sessions = ["s17_s2r_r2_profile_a", "s17_s2r_r2_profile_b"]
    if any(_session_exists(name) for name in sessions):
        raise FileExistsError("R2 profile tmux session already exists")
    launched = [
        launch_background_tmux(
            experiment_id=session, argv=command, cwd=ROOT, tmux_session=session
        )
        for session, command in zip(sessions, commands)
    ]
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": PROFILE_ATTEMPT_ID,
            "step_id": "S17-2R",
            "kind": "R2_CAPACITY_PROFILE",
            "started_at": utc_now(),
            "state": "RUNNING",
            "scientific_result_eligible": False,
            "gpu_ids": [gpu_a, gpu_b],
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_2R_R2_CAPACITY_PROFILE_STARTED",
        stage="r2_capacity_profile",
        progress={"current": 0, "total": 4, "unit": "r2_profile_family"},
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
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
        "tmux_sessions": launched,
        "commands": commands,
        "gpu_ids": [gpu_a, gpu_b],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def profile_wave(families: str, physical_gpu: int, snapshot_path: Path) -> None:
    requested = [value.strip() for value in families.split(",") if value.strip()]
    for family in requested:
        profile_family(family, physical_gpu, snapshot_path)


def finalize_profile() -> dict:
    summaries = {}
    for family in FAMILIES:
        path = PROFILE_ROOT / family / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"R2 profile summary is missing: {path}")
        summaries[family] = json.loads(path.read_text(encoding="utf-8"))
    arms = {
        arm: metrics
        for family in summaries.values()
        for arm, metrics in family["arms"].items()
    }
    treatment_arms = {"latte_full", "gryphon_item", "diffgrm_masked", "setrec_full"}
    finite = all(
        math.isfinite(row["loss"])
        and math.isfinite(row["total_gradient_norm"])
        and row["total_gradient_norm"] > 0.0
        for row in arms.values()
    )
    treatment_gradients = {
        arm: arms[arm]["candidate_gradient_norm"] > 0.0 for arm in treatment_arms
    }
    nonempty = all(
        row["prediction_rows"] == 16 and row["all_prediction_rows_nonempty"]
        for row in arms.values()
    )
    forbidden_read_safe = all(
        not family["official_test_read"]
        and not family["sports_read"]
        and not family["d1_read"]
        for family in summaries.values()
    )
    maximum_peak = max(row["peak_allocated_mib"] for row in arms.values())
    profile_pass = finite and all(treatment_gradients.values()) and nonempty and forbidden_read_safe
    payload = {
        "schema_version": "phase17.s17_2r_r2_profile_closeout.v1",
        "gate": "R2_CAPACITY_PROFILE",
        "state": "PASS" if profile_pass else "FAIL",
        "formal_result_eligible": False,
        "effect_metrics_used_for_selection": False,
        "family_summaries": {
            family: str((PROFILE_ROOT / family / "summary.json").relative_to(ROOT))
            for family in FAMILIES
        },
        "checks": {
            "all_losses_and_gradients_finite": finite,
            "treatment_specific_gradients": treatment_gradients,
            "all_prediction_batches_nonempty": nonempty,
            "forbidden_read_safe": forbidden_read_safe,
        },
        "resources": {
            "maximum_peak_allocated_mib": maximum_peak,
            "recommended_admission_free_mib": int(math.ceil(maximum_peak + 4096)),
            "arms": {
                arm: {
                    key: row[key]
                    for key in (
                        "parameter_count",
                        "train_step_seconds",
                        "generation_batch_seconds",
                        "peak_allocated_mib",
                    )
                }
                for arm, row in arms.items()
            },
        },
        "mechanism_profile": {
            "diffgrm_generation_speedup": arms["diffgrm_ar_control"][
                "generation_batch_seconds"
            ]
            / arms["diffgrm_masked"]["generation_batch_seconds"],
            "setrec_generation_speedup": arms["setrec_ar_control"][
                "generation_batch_seconds"
            ]
            / arms["setrec_full"]["generation_batch_seconds"],
        },
        "gpu1_repeat_preserved": True,
        "completed_at": utc_now(),
    }
    output = PROFILE_ROOT / "profile_summary.json"
    atomic_json(output, payload)
    ledger_path = ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl"
    closeout_id = f"{PROFILE_ATTEMPT_ID}-closeout"
    existing_ids = {
        json.loads(line)["attempt_id"]
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if closeout_id not in existing_ids:
        AttemptLedger(ledger_path).append(
            {
                "attempt_id": closeout_id,
                "step_id": "S17-2R",
                "kind": "R2_CAPACITY_PROFILE_CLOSEOUT",
                "started_at": payload["completed_at"],
                "ended_at": payload["completed_at"],
                "state": "COMPLETED" if profile_pass else "FAILED",
                "scientific_result_eligible": False,
                "closes_attempt_id": PROFILE_ATTEMPT_ID,
                "summary": str(output.relative_to(ROOT)),
            }
        )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_2R_R2_CAPACITY_PROFILE_COMPLETE_SCREEN_PREFLIGHT",
        stage="r2_profile_complete_screen_preflight",
        progress={"current": 2, "total": 4, "unit": "r0_to_r3"},
        gpu_ids=[],
        tmux_session=None,
        process_alive=False,
        r2_profile_summary=str(output.relative_to(ROOT)),
        r2_profile_pass=profile_pass,
        r2_screen_gpu_started=False,
        gpu1_repeat_preserved=True,
        affects_scientific_result=False,
        result_selection_eligible=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("finalize-profile")
    launch_parser = subparsers.add_parser("launch-profile")
    launch_parser.add_argument("--gpu-a", type=int, required=True)
    launch_parser.add_argument("--gpu-b", type=int, required=True)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    profile_parser.add_argument("--physical-gpu", type=int, required=True)
    profile_parser.add_argument("--snapshot", type=Path, required=True)
    wave_parser = subparsers.add_parser("profile-wave")
    wave_parser.add_argument("--families", required=True)
    wave_parser.add_argument("--physical-gpu", type=int, required=True)
    wave_parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "finalize-profile":
        finalize_profile()
    elif args.command == "launch-profile":
        launch_profile(args.gpu_a, args.gpu_b)
    elif args.command == "profile":
        profile_family(args.family, args.physical_gpu, args.snapshot)
    else:
        profile_wave(args.families, args.physical_gpu, args.snapshot)


if __name__ == "__main__":
    main()
