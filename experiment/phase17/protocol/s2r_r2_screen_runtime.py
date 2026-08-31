#!/usr/bin/env python3
"""Formal S17-2R 3k-user architecture screen with matched native controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase17.core.resource_profiler import query_gpus, snapshot  # noqa: E402
from experiment.phase17.core.run_manager import (  # noqa: E402
    freeze_run_snapshot,
    launch_background_tmux,
    verify_run_snapshot,
)
from experiment.phase17.core.s2r_parallel_architectures import PARALLEL_ARMS  # noqa: E402
from experiment.phase17.core.s2r_r2_evaluator import (  # noqa: E402
    compare_family_predictions,
    user_ranking_contribution,
)
from experiment.phase17.core.s2r_sid import (  # noqa: E402
    SIDSequenceDataset,
    SemanticIDCodec,
    build_r2_external_examples,
    build_r2_training_examples,
    collate_sid_batch,
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
from experiment.phase17.protocol.s2r_r1_sid_runtime import (  # noqa: E402
    candidate_gradient_norm,
    model_gradient_norm,
)
from experiment.phase17.protocol.s2r_r2_runtime import (  # noqa: E402
    CONFIG_PATH,
    CONTRACT_PATH,
    EARLY_STOP_PATH,
    EXPERIMENT_ID,
    ITEM_TEXT_PATH,
    PREFLIGHT_DIR,
    PROFILE_ROOT,
    SEED,
    SEQUENCE_PATH,
    SET_SID_PATH,
    SID_PATH,
    codec_from_sid,
    create_model,
)


ATTEMPT_ID = "r2-screen-0001"
SCREEN_ROOT = ROOT / "artifacts/phase17/s2r_r2/screen/run-0001"
PROFILE_SUMMARY_PATH = PROFILE_ROOT / "profile_summary.json"
SCREEN_PLAN_PATH = PREFLIGHT_DIR / "screen_launch_plan.json"
SNAPSHOT_PATH = (
    ROOT
    / "artifacts/phase17/snapshots"
    / EXPERIMENT_ID
    / ATTEMPT_ID
    / "manifest.json"
)
FAMILY_ARMS = {
    "latte": ("psid_control", "latte_full"),
    "gryphon": ("gryphon_item",),
    "diffgrm": ("diffgrm_ar_control", "diffgrm_masked"),
    "setrec": ("setrec_ar_control", "setrec_full"),
}
FAMILY_TREATMENT_CONTROL = {
    "latte": ("latte_full", "psid_control"),
    "gryphon": ("gryphon_item", "same_checkpoint_beam_control"),
    "diffgrm": ("diffgrm_masked", "diffgrm_ar_control"),
    "setrec": ("setrec_full", "setrec_ar_control"),
}
SCREEN_ADMISSION_FREE_MIB = 7168


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _rank(target: str, ranking: list[str]) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def should_early_stop(
    *, completed_epochs: int, stale_epochs: int, minimum_epochs: int, patience: int
) -> bool:
    return completed_epochs >= minimum_epochs and stale_epochs >= patience


def gryphon_beam_control(native_rows: list[list]) -> list[list]:
    return [
        sorted(rows, key=lambda row: (-row.beam_score, row.item_id))
        for rows in native_rows
    ]


def _prediction_payload(
    users: list[str],
    targets: list[str],
    native_rows: list[list],
    *,
    use_beam_score: bool = False,
) -> list[dict]:
    return [
        {
            "user_id": user,
            "target_item": target,
            "ranked_items": [row.item_id for row in rows],
            "scores": [
                float(
                    row.beam_score
                    if use_beam_score
                    else row.final_score
                    if hasattr(row, "final_score")
                    else row.score
                )
                for row in rows
            ],
        }
        for user, target, rows in zip(users, targets, native_rows)
    ]


def _write_prediction_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _average_diagnostics(rows: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    if not rows:
        return {}
    total = sum(count for count, _ in rows)
    return {
        key: sum(count * values[key] for count, values in rows) / total
        for key in sorted(rows[0][1])
    }


def evaluate_model(
    *,
    arm: str,
    model: torch.nn.Module,
    codec: SemanticIDCodec,
    examples: list,
    device: torch.device,
    config: dict,
) -> dict:
    dataset = SIDSequenceDataset(
        examples, codec, latte_training=arm == "latte_full", seed=SEED
    )
    loader = DataLoader(
        dataset,
        batch_size=config["optimization"]["evaluation_batch_size"],
        shuffle=False,
        collate_fn=collate_sid_batch,
    )
    users: list[str] = []
    targets: list[str] = []
    native_rows: list[list] = []
    diagnostics: list[tuple[int, dict[str, float]]] = []
    generation_seconds = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            users.extend(batch["user_id"])
            targets.extend(batch["target_item"])
            moved = _move(batch, device)
            if arm in PARALLEL_ARMS:
                diagnostics.append(
                    (
                        len(batch["user_id"]),
                        model.mechanism_diagnostics(
                            input_ids=moved["input_ids"],
                            attention_mask=moved["attention_mask"],
                            labels=moved["labels"],
                        ),
                    )
                )
            arguments = {
                "input_ids": moved["input_ids"],
                "attention_mask": moved["attention_mask"],
                "num_beams": config["optimization"]["num_beams"],
                "top_k": config["optimization"]["top_k"],
            }
            if arm not in PARALLEL_ARMS:
                arguments["latte_aggregation"] = "logsumexp"
            started = time.perf_counter()
            native_rows.extend(model.generate_ranked(**arguments))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds += time.perf_counter() - started
    rankings = [[row.item_id for row in rows] for rows in native_rows]
    valid_rows = [
        bool(ranking) and all(item in codec.item_to_code for item in ranking)
        for ranking in rankings
    ]
    metrics_by_user = {
        user: user_ranking_contribution(target, ranking)
        for user, target, ranking in zip(users, targets, rankings)
    }
    paths = [getattr(row, "path_count", 1) for rows in native_rows for row in rows]
    return {
        "users": users,
        "targets": targets,
        "native_rows": native_rows,
        "rankings": rankings,
        "metrics_by_user": metrics_by_user,
        "valid_item_rate": mean(float(value) for value in valid_rows),
        "mean_unique_candidates": mean(len(row) for row in rankings),
        "multi_path_item_rate": mean(float(value > 1) for value in paths)
        if paths
        else 0.0,
        "generation_seconds": generation_seconds,
        "diagnostics": _average_diagnostics(diagnostics),
    }


def _checkpoint_save(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, temporary)
    os.replace(temporary, path)


def train_arm_to_best(
    *,
    family: str,
    arm: str,
    codec: SemanticIDCodec,
    train_examples: list,
    early_stop_examples: list,
    device: torch.device,
    config: dict,
    arm_dir: Path,
) -> dict:
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
    dataset = SIDSequenceDataset(
        train_examples, codec, latte_training=arm == "latte_full", seed=SEED
    )
    curve = []
    best_metric = -math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint = arm_dir / "best_checkpoint.pt"
    gradient_max = 0.0
    started_all = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch_index in range(config["optimization"]["maximum_epochs"]):
        epoch = epoch_index + 1
        dataset.set_epoch(epoch_index)
        loader = DataLoader(
            dataset,
            batch_size=config["optimization"]["train_batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED + epoch_index),
            collate_fn=collate_sid_batch,
        )
        model.train()
        losses = []
        train_started = time.perf_counter()
        for batch in loader:
            moved = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                labels=moved["labels"],
                target_item_index=moved["target_item_index"],
            )
            output.loss.backward()
            gradient_max = max(gradient_max, candidate_gradient_norm(model, arm))
            if not math.isfinite(model_gradient_norm(model)):
                raise FloatingPointError(f"non-finite gradient in {family}/{arm}")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["optimization"]["gradient_clip_norm"]
            )
            optimizer.step()
            losses.append(float(output.loss.detach().cpu().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        early = evaluate_model(
            arm=arm,
            model=model,
            codec=codec,
            examples=early_stop_examples,
            device=device,
            config=config,
        )
        early_ndcg = mean(
            row["ndcg@10"] for row in early["metrics_by_user"].values()
        )
        improved = (
            best_epoch == 0
            or early_ndcg
            > best_metric + config["optimization"]["early_stop_min_delta_ndcg_at_10"]
        )
        if improved:
            best_metric = early_ndcg
            best_epoch = epoch
            stale_epochs = 0
            _checkpoint_save(
                checkpoint,
                model,
                {
                    "family": family,
                    "arm": arm,
                    "epoch": epoch,
                    "internal_ndcg@10": early_ndcg,
                    "external_target_read": False,
                    "saved_at": utc_now(),
                },
            )
        else:
            stale_epochs += 1
        curve.append(
            {
                "epoch": epoch,
                "train_loss": mean(losses),
                "train_seconds": train_seconds,
                "internal_ndcg@10": early_ndcg,
                "internal_generation_seconds": early["generation_seconds"],
                "improved": improved,
                "stale_epochs": stale_epochs,
                "external_target_read": False,
            }
        )
        atomic_json(arm_dir / "learning_curve.json", {"epochs": curve})
        if should_early_stop(
            completed_epochs=epoch,
            stale_epochs=stale_epochs,
            minimum_epochs=config["optimization"]["minimum_epochs"],
            patience=config["optimization"]["early_stop_patience"],
        ):
            break
    if not checkpoint.exists():
        raise FileNotFoundError(f"best checkpoint missing for {family}/{arm}")
    result = {
        "arm": arm,
        "epochs_completed": len(curve),
        "best_epoch": best_epoch,
        "best_internal_ndcg@10": best_metric,
        "early_stop_triggered": len(curve)
        < config["optimization"]["maximum_epochs"],
        "candidate_gradient_norm_max": gradient_max,
        "peak_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if device.type == "cuda"
            else 0.0
        ),
        "training_wall_seconds": time.perf_counter() - started_all,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "learning_curve": str((arm_dir / "learning_curve.json").relative_to(ROOT)),
        "external_target_read_during_checkpoint_selection": False,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def _load_best(
    arm: str, codec: SemanticIDCodec, config: dict, checkpoint: Path, device: torch.device
) -> torch.nn.Module:
    capacity = (
        config["optimization"]["parallel_capacity"]
        if arm in PARALLEL_ARMS
        else config["optimization"]["semantic_capacity"]
    )
    model = create_model(arm, codec, capacity).to(device)
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state"], strict=True)
    return model


def _save_external_evaluation(
    arm_dir: Path, evaluation: dict, *, control_rows: list[list] | None = None
) -> dict:
    rows = _prediction_payload(
        evaluation["users"], evaluation["targets"], evaluation["native_rows"]
    )
    prediction_path = arm_dir / "external_predictions.jsonl"
    _write_prediction_jsonl(prediction_path, rows)
    metrics_path = arm_dir / "external_user_metrics.json"
    atomic_json(metrics_path, evaluation["metrics_by_user"])
    result = {
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "user_metrics_path": str(metrics_path.relative_to(ROOT)),
        "user_metrics_sha256": sha256_file(metrics_path),
        "prediction_rows": len(rows),
        "valid_item_rate": evaluation["valid_item_rate"],
        "mean_unique_candidates": evaluation["mean_unique_candidates"],
        "multi_path_item_rate": evaluation["multi_path_item_rate"],
        "generation_seconds": evaluation["generation_seconds"],
        "diagnostics": evaluation["diagnostics"],
        "external_evaluation_count": 1,
    }
    if control_rows is not None:
        control_payload = _prediction_payload(
            evaluation["users"],
            evaluation["targets"],
            control_rows,
            use_beam_score=True,
        )
        control_path = arm_dir / "same_checkpoint_beam_control_predictions.jsonl"
        _write_prediction_jsonl(control_path, control_payload)
        control_rankings = [[row.item_id for row in rows] for rows in control_rows]
        control_metrics = {
            user: user_ranking_contribution(target, ranking)
            for user, target, ranking in zip(
                evaluation["users"], evaluation["targets"], control_rankings
            )
        }
        control_metrics_path = arm_dir / "same_checkpoint_beam_control_user_metrics.json"
        atomic_json(control_metrics_path, control_metrics)
        result["same_checkpoint_beam_control"] = {
            "prediction_path": str(control_path.relative_to(ROOT)),
            "prediction_sha256": sha256_file(control_path),
            "user_metrics_path": str(control_metrics_path.relative_to(ROOT)),
            "user_metrics_sha256": sha256_file(control_metrics_path),
            "metrics_by_user": control_metrics,
            "rankings": control_rankings,
        }
    return result


def _cohorts(config: dict) -> tuple[tuple[str, ...], ...]:
    return read_cohort_user_ids(
        [ROOT / path for path in config["data"]["evaluation_cohort_paths"]]
    )


def cpu_smoke() -> dict:
    """Exercise train/early-stop/checkpoint/restore without external targets or GPU."""

    config = json.loads(
        (PREFLIGHT_DIR / "frozen_config.json").read_text(encoding="utf-8")
    )
    config["optimization"].update(
        {
            "semantic_capacity": "tiny",
            "maximum_epochs": 2,
            "minimum_epochs": 1,
            "early_stop_patience": 1,
            "train_batch_size": 2,
            "evaluation_batch_size": 2,
            "num_beams": 2,
            "top_k": 2,
        }
    )
    sid = json.loads(SID_PATH.read_text(encoding="utf-8"))
    codec = codec_from_sid(sid)
    users = parse_shadow_sequences(SEQUENCE_PATH)
    early_stop_ids = tuple(
        line.strip()
        for line in EARLY_STOP_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    train_examples, early_stop_examples = build_r2_training_examples(
        users, early_stop_ids
    )
    output_dir = PREFLIGHT_DIR / "screen_cpu_smoke/latte_full"
    output_dir.mkdir(parents=True, exist_ok=True)
    training = train_arm_to_best(
        family="latte",
        arm="latte_full",
        codec=codec,
        train_examples=train_examples[:2],
        early_stop_examples=early_stop_examples[:2],
        device=torch.device("cpu"),
        config=config,
        arm_dir=output_dir,
    )
    model = _load_best(
        "latte_full",
        codec,
        config,
        ROOT / training["checkpoint"],
        torch.device("cpu"),
    )
    evaluation = evaluate_model(
        arm="latte_full",
        model=model,
        codec=codec,
        examples=early_stop_examples[:2],
        device=torch.device("cpu"),
        config=config,
    )
    payload = {
        "schema_version": "phase17.s17_2r_r2_screen_cpu_smoke.v1",
        "state": "PASS"
        if training["best_epoch"] >= 1
        and evaluation["valid_item_rate"] == 1.0
        and len(evaluation["metrics_by_user"]) == 2
        else "FAIL",
        "training": training,
        "prediction_rows": len(evaluation["metrics_by_user"]),
        "valid_item_rate": evaluation["valid_item_rate"],
        "external_target_materialized": False,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "completed_at": utc_now(),
    }
    atomic_json(PREFLIGHT_DIR / "screen_cpu_smoke.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def prepare() -> dict:
    config = json.loads(
        (PREFLIGHT_DIR / "frozen_config.json").read_text(encoding="utf-8")
    )
    profile = json.loads(PROFILE_SUMMARY_PATH.read_text(encoding="utf-8"))
    smoke = json.loads(
        (PREFLIGHT_DIR / "screen_cpu_smoke.json").read_text(encoding="utf-8")
    )
    if profile["state"] != "PASS" or smoke["state"] != "PASS":
        raise RuntimeError("R2 formal screen requires passing profile and CPU smoke")
    users = parse_shadow_sequences(SEQUENCE_PATH)
    early_stop_ids = tuple(
        line.strip()
        for line in EARLY_STOP_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    train_examples, early_stop_examples = build_r2_training_examples(
        users, early_stop_ids
    )
    train_batches = math.ceil(
        len(train_examples) / config["optimization"]["train_batch_size"]
    )
    early_batches = math.ceil(
        len(early_stop_examples) / config["optimization"]["evaluation_batch_size"]
    )
    external_batches = math.ceil(
        len(users) / config["optimization"]["evaluation_batch_size"]
    )
    profiled_arms = profile["resources"]["arms"]
    estimates = {}
    for family, arms in FAMILY_ARMS.items():
        train_per_epoch = train_batches * sum(
            profiled_arms[arm]["train_step_seconds"] for arm in arms
        )
        internal_per_epoch = early_batches * sum(
            profiled_arms[arm]["generation_batch_seconds"] for arm in arms
        )
        external_once = external_batches * sum(
            profiled_arms[arm]["generation_batch_seconds"] for arm in arms
        )
        raw_upper = (
            config["optimization"]["maximum_epochs"]
            * (train_per_epoch + internal_per_epoch)
            + external_once
        )
        estimates[family] = {
            "train_batches_per_epoch": train_batches,
            "internal_batches_per_epoch": early_batches,
            "external_batches_once": external_batches,
            "profile_scaled_raw_upper_minutes": raw_upper / 60.0,
            "contention_adjusted_upper_minutes": raw_upper * 1.75 / 60.0 + 5.0,
        }
    payload = {
        "schema_version": "phase17.s17_2r_r2_screen_launch_plan.v1",
        "state": "READY",
        "families": list(FAMILY_ARMS),
        "recommended_parallel_gpus": 4,
        "one_family_per_gpu": True,
        "gpu1_excluded": True,
        "screen_admission_free_mib": SCREEN_ADMISSION_FREE_MIB,
        "profile_maximum_peak_allocated_mib": profile["resources"][
            "maximum_peak_allocated_mib"
        ],
        "runtime_estimates": estimates,
        "four_gpu_wall_upper_minutes": max(
            row["contention_adjusted_upper_minutes"] for row in estimates.values()
        ),
        "maximum_epochs": config["optimization"]["maximum_epochs"],
        "early_stop_may_reduce_runtime": True,
        "external_evaluation_after_checkpoint_freeze_only": True,
        "profile_summary_sha256": sha256_file(PROFILE_SUMMARY_PATH),
        "cpu_smoke_sha256": sha256_file(PREFLIGHT_DIR / "screen_cpu_smoke.json"),
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "prepared_at": utc_now(),
    }
    atomic_json(SCREEN_PLAN_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def worker(family: str, physical_gpu: int, snapshot_path: Path) -> dict:
    if family not in FAMILY_ARMS or physical_gpu == 1:
        raise PermissionError("invalid R2 screen family or reserved GPU1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for R2 formal screen")
    verify_run_snapshot(ROOT, snapshot_path)
    config = json.loads((snapshot_path.parent / "config.json").read_text(encoding="utf-8"))
    if sha256_file(SID_PATH) != config["sid_sha256"]:
        raise RuntimeError("R2 SID changed after formal screen freeze")
    if sha256_file(SET_SID_PATH) != config["set_sid_sha256"]:
        raise RuntimeError("R2 SET SID changed after formal screen freeze")
    base_sid = json.loads(SID_PATH.read_text(encoding="utf-8"))
    set_sid = json.loads(SET_SID_PATH.read_text(encoding="utf-8"))
    codec = codec_from_sid(set_sid, set_codec=True) if family == "setrec" else codec_from_sid(base_sid)
    users = parse_shadow_sequences(SEQUENCE_PATH)
    early_stop_ids = tuple(
        line.strip()
        for line in EARLY_STOP_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    train_examples, early_stop_examples = build_r2_training_examples(
        users, early_stop_ids
    )
    family_dir = SCREEN_ROOT / family
    if family_dir.exists():
        raise FileExistsError(f"R2 screen family output exists: {family_dir}")
    family_dir.mkdir(parents=True)
    device = torch.device("cuda:0")
    training = {}
    try:
        for arm in FAMILY_ARMS[family]:
            arm_dir = family_dir / arm
            arm_dir.mkdir(parents=True)
            training[arm] = train_arm_to_best(
                family=family,
                arm=arm,
                codec=codec,
                train_examples=train_examples,
                early_stop_examples=early_stop_examples,
                device=device,
                config=config,
                arm_dir=arm_dir,
            )

        # The external target split is materialized only after every checkpoint
        # in this matched family has been selected and frozen.
        external_examples = build_r2_external_examples(users)
        evaluations = {}
        internal_evaluations = {}
        for arm in FAMILY_ARMS[family]:
            arm_dir = family_dir / arm
            model = _load_best(
                arm,
                codec,
                config,
                ROOT / training[arm]["checkpoint"],
                device,
            )
            evaluation = evaluate_model(
                arm=arm,
                model=model,
                codec=codec,
                examples=external_examples,
                device=device,
                config=config,
            )
            control_rows = (
                gryphon_beam_control(evaluation["native_rows"])
                if arm == "gryphon_item"
                else None
            )
            evaluations[arm] = _save_external_evaluation(
                arm_dir, evaluation, control_rows=control_rows
            )
            internal_evaluations[arm] = evaluation
            del model
            torch.cuda.empty_cache()

        treatment_arm, control_arm = FAMILY_TREATMENT_CONTROL[family]
        treatment = internal_evaluations[treatment_arm]
        if family == "gryphon":
            control = evaluations[treatment_arm]["same_checkpoint_beam_control"]
            control_metrics = control["metrics_by_user"]
            control_rankings = control["rankings"]
            treatment_rankings = treatment["rankings"]
            candidate_sets_identical = all(
                set(left) == set(right)
                for left, right in zip(treatment_rankings, control_rankings)
            )
            gains = []
            for target, left, right in zip(
                treatment["targets"], treatment_rankings, control_rankings
            ):
                treatment_rank = _rank(target, left)
                control_rank = _rank(target, right)
                if treatment_rank is not None and control_rank is not None:
                    gains.append(float(control_rank - treatment_rank))
            mechanism = {
                "valid_item_rate": treatment["valid_item_rate"],
                "candidate_sets_identical": candidate_sets_identical,
                "mean_target_rank_gain": mean(gains) if gains else 0.0,
                "comparable_target_users": len(gains),
            }
        else:
            control = internal_evaluations[control_arm]
            control_metrics = control["metrics_by_user"]
            if family == "latte":
                mechanism = {
                    "valid_item_rate": min(
                        treatment["valid_item_rate"], control["valid_item_rate"]
                    ),
                    "multi_path_item_rate": treatment["multi_path_item_rate"],
                }
            elif family == "diffgrm":
                mechanism = {
                    "valid_item_rate": min(
                        treatment["valid_item_rate"], control["valid_item_rate"]
                    ),
                    "treatment_generation_seconds": treatment["generation_seconds"],
                    "control_generation_seconds": control["generation_seconds"],
                }
            else:
                mechanism = {
                    "valid_item_rate": min(
                        treatment["valid_item_rate"], control["valid_item_rate"]
                    ),
                    "treatment_generation_seconds": treatment["generation_seconds"],
                    "control_generation_seconds": control["generation_seconds"],
                    "set_token_recovery": treatment["diagnostics"].get(
                        "set_token_recovery", 0.0
                    ),
                }
        comparison = compare_family_predictions(
            treatment=treatment["metrics_by_user"],
            control=control_metrics,
            cohorts=_cohorts(config),
            mechanism_metrics=mechanism,
            family=family,
            bootstrap_replicates=config["uncertainty"]["replicates"],
            seed=config["uncertainty"]["seed"],
        )
        for value in evaluations.values():
            if "same_checkpoint_beam_control" in value:
                value["same_checkpoint_beam_control"].pop("metrics_by_user", None)
                value["same_checkpoint_beam_control"].pop("rankings", None)
        payload = {
            "schema_version": "phase17.s17_2r_r2_family_screen.v1",
            "family": family,
            "physical_gpu": physical_gpu,
            "device_name": torch.cuda.get_device_name(device),
            "formal_result_eligible": True,
            "result_selection_eligible": True,
            "training": training,
            "external_evaluations": evaluations,
            "comparison": comparison,
            "external_target_materialized_after_all_best_checkpoints": True,
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
                "schema_version": "phase17.s17_2r_r2_family_failure.v1",
                "family": family,
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


def source_paths() -> list[Path]:
    return [
        ROOT / "experiment/phase17/protocol/s2r_r2_screen_runtime.py",
        ROOT / "experiment/phase17/protocol/s2r_r2_runtime.py",
        ROOT / "experiment/phase17/core/s2r_sid.py",
        ROOT / "experiment/phase17/core/s2r_architectures.py",
        ROOT / "experiment/phase17/core/s2r_parallel_architectures.py",
        ROOT / "experiment/phase17/core/s2r_r2_evaluator.py",
        CONFIG_PATH,
        CONTRACT_PATH,
        PREFLIGHT_DIR / "frozen_config.json",
        PROFILE_SUMMARY_PATH,
        SCREEN_PLAN_PATH,
    ]


def launch(gpu_by_family: dict[str, int]) -> dict:
    if set(gpu_by_family) != set(FAMILY_ARMS):
        raise ValueError("R2 screen needs one GPU assignment for every family")
    gpu_ids = list(gpu_by_family.values())
    if len(set(gpu_ids)) != 4 or 1 in gpu_ids:
        raise PermissionError("R2 formal screen needs four distinct non-GPU1 cards")
    records = query_gpus()
    by_id = {row.index: row for row in records}
    for family, gpu_id in gpu_by_family.items():
        free = by_id[gpu_id].free_mib if gpu_id in by_id else 0
        if free < SCREEN_ADMISSION_FREE_MIB:
            raise RuntimeError(
                f"GPU {gpu_id} for {family} has {free} MiB free; screen needs {SCREEN_ADMISSION_FREE_MIB}"
            )
    if SCREEN_ROOT.exists() or SNAPSHOT_PATH.parent.exists():
        raise FileExistsError("R2 screen run-0001 already exists; implicit retry forbidden")
    base_config = json.loads(
        (PREFLIGHT_DIR / "frozen_config.json").read_text(encoding="utf-8")
    )
    profile_summary = json.loads(PROFILE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if profile_summary["state"] != "PASS":
        raise RuntimeError("R2 capacity profile did not pass")
    screen_plan = json.loads(SCREEN_PLAN_PATH.read_text(encoding="utf-8"))
    if screen_plan["state"] != "READY":
        raise RuntimeError("R2 screen launch plan is not ready")
    commands = {}
    for family, gpu_id in gpu_by_family.items():
        commands[family] = [
            "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
            "-m",
            "experiment.phase17.protocol.s2r_r2_screen_runtime",
            "worker",
            "--family",
            family,
            "--physical-gpu",
            str(gpu_id),
            "--snapshot",
            str(SNAPSHOT_PATH),
        ]
    frozen = {
        **base_config,
        "screen_attempt_id": ATTEMPT_ID,
        "screen_commands": commands,
        "screen_gpu_assignments": gpu_by_family,
        "screen_admission_free_mib": SCREEN_ADMISSION_FREE_MIB,
        "profile_summary_path": str(PROFILE_SUMMARY_PATH.relative_to(ROOT)),
        "profile_summary_sha256": sha256_file(PROFILE_SUMMARY_PATH),
        "screen_plan_path": str(SCREEN_PLAN_PATH.relative_to(ROOT)),
        "screen_plan_sha256": sha256_file(SCREEN_PLAN_PATH),
        "gpu_snapshot": snapshot(records),
        "gpu1_repeat_preserved": True,
    }
    outer = [
        "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python",
        "-m",
        "experiment.phase17.protocol.s2r_r2_screen_runtime",
        "launch",
        "--gpu-latte",
        str(gpu_by_family["latte"]),
        "--gpu-gryphon",
        str(gpu_by_family["gryphon"]),
        "--gpu-diffgrm",
        str(gpu_by_family["diffgrm"]),
        "--gpu-setrec",
        str(gpu_by_family["setrec"]),
    ]
    manifest = freeze_run_snapshot(
        root=ROOT,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        command=outer,
        source_paths=source_paths(),
        config=frozen,
    )
    verify_run_snapshot(ROOT, manifest)
    sessions = {}
    for family, command in commands.items():
        session = f"s17_s2r_r2_{family}"
        sessions[family] = launch_background_tmux(
            experiment_id=session, argv=command, cwd=ROOT, tmux_session=session
        )
    AttemptLedger(ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl").append(
        {
            "attempt_id": ATTEMPT_ID,
            "step_id": "S17-2R",
            "kind": "R2_FORMAL_3K_ARCHITECTURE_SCREEN",
            "started_at": utc_now(),
            "state": "RUNNING",
            "scientific_result_eligible": True,
            "gpu_ids": gpu_ids,
            "snapshot_manifest": str(manifest.relative_to(ROOT)),
        }
    )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "BACKGROUND_STARTED",
        "S17_2R_R2_FORMAL_SCREEN_STARTED",
        stage="r2_formal_3k_screen",
        progress={"current": 0, "total": 4, "unit": "r2_family"},
        gpu_ids=gpu_ids,
        gpu_snapshot={"records": snapshot(records)},
        tmux_session=",".join(sessions.values()),
        process_alive=True,
        run_snapshot_manifest=str(manifest.relative_to(ROOT)),
        r2_screen_gpu_started=True,
        gpu1_repeat_preserved=True,
        affects_scientific_result=True,
        result_selection_eligible=True,
    )
    payload = {
        "snapshot_manifest": str(manifest.relative_to(ROOT)),
        "gpu_assignments": gpu_by_family,
        "tmux_sessions": sessions,
        "commands": commands,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def finalize() -> dict:
    summaries = {}
    for family in FAMILY_ARMS:
        path = SCREEN_ROOT / family / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"R2 screen summary missing: {path}")
        summaries[family] = json.loads(path.read_text(encoding="utf-8"))
    decisions = {
        family: summary["comparison"]["decision"]
        for family, summary in summaries.items()
    }
    strong = sorted(
        family for family, decision in decisions.items() if decision == "STRONG_PROMOTE"
    )
    borderline = sorted(
        family
        for family, decision in decisions.items()
        if decision == "BORDERLINE_ONE_REVISION"
    )
    payload = {
        "schema_version": "phase17.s17_2r_r2_screen_closeout.v1",
        "gate": "R2",
        "state": "COMPLETED",
        "formal_result_eligible": True,
        "family_summaries": {
            family: str((SCREEN_ROOT / family / "summary.json").relative_to(ROOT))
            for family in FAMILY_ARMS
        },
        "decisions": decisions,
        "strong_promotions": strong,
        "borderline_one_revision": borderline,
        "rejected": sorted(
            family for family, decision in decisions.items() if decision == "REJECT"
        ),
        "r3_eligible": strong[:2],
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "gpu1_repeat_preserved": True,
        "completed_at": utc_now(),
    }
    output = SCREEN_ROOT / "screen_summary.json"
    atomic_json(output, payload)
    closeout_id = f"{ATTEMPT_ID}-closeout"
    ledger_path = ROOT / "artifacts/phase17/attempts/S17-2R.attempts.jsonl"
    ids = {
        json.loads(line)["attempt_id"]
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if closeout_id not in ids:
        AttemptLedger(ledger_path).append(
            {
                "attempt_id": closeout_id,
                "step_id": "S17-2R",
                "kind": "R2_FORMAL_3K_ARCHITECTURE_SCREEN_CLOSEOUT",
                "started_at": payload["completed_at"],
                "ended_at": payload["completed_at"],
                "state": "COMPLETED",
                "scientific_result_eligible": True,
                "closes_attempt_id": ATTEMPT_ID,
                "summary": str(output.relative_to(ROOT)),
            }
        )
    StatusWriter(ROOT / "artifacts/phase17/status", EXPERIMENT_ID).transition(
        "RUNNING",
        "RUNNING_SCIENTIFIC",
        "S17_2R_R2_FORMAL_SCREEN_COMPLETE_R3_GATE",
        stage="r2_complete_r3_gate",
        progress={"current": 3, "total": 4, "unit": "r0_to_r3"},
        gpu_ids=[],
        tmux_session=None,
        process_alive=False,
        r2_screen_summary=str(output.relative_to(ROOT)),
        r2_screen_gpu_started=False,
        gpu1_repeat_preserved=True,
        affects_scientific_result=True,
        result_selection_eligible=True,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--gpu-latte", type=int, required=True)
    launch_parser.add_argument("--gpu-gryphon", type=int, required=True)
    launch_parser.add_argument("--gpu-diffgrm", type=int, required=True)
    launch_parser.add_argument("--gpu-setrec", type=int, required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--family", choices=sorted(FAMILY_ARMS), required=True)
    worker_parser.add_argument("--physical-gpu", type=int, required=True)
    worker_parser.add_argument("--snapshot", type=Path, required=True)
    subparsers.add_parser("finalize")
    subparsers.add_parser("cpu-smoke")
    subparsers.add_parser("prepare")
    args = parser.parse_args()
    if args.command == "launch":
        launch(
            {
                "latte": args.gpu_latte,
                "gryphon": args.gpu_gryphon,
                "diffgrm": args.gpu_diffgrm,
                "setrec": args.gpu_setrec,
            }
        )
    elif args.command == "worker":
        worker(args.family, args.physical_gpu, args.snapshot)
    elif args.command == "cpu-smoke":
        cpu_smoke()
    elif args.command == "prepare":
        prepare()
    else:
        finalize()


if __name__ == "__main__":
    main()
