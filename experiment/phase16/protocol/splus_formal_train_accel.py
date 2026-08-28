#!/usr/bin/env python3
"""Accelerated FP32 S-PLUS formal entrypoint with sample-correct tail batching."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

import splus_formal_train as base


ROOT = Path(__file__).resolve().parents[3]


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_overlay(overlay_path: Path) -> Path:
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    base_spec = overlay["base_config"]
    base_path = ROOT / base_spec["path"]
    if base.sha256(base_path) != base_spec["sha256"]:
        raise ValueError("Accelerated formal base-config SHA drift")
    resolved = deep_merge(json.loads(base_path.read_text(encoding="utf-8")), overlay["overrides"])
    resolved["overlay_config"] = {
        "path": str(overlay_path.relative_to(ROOT)),
        "sha256": base.sha256(overlay_path),
        "base_path": base_spec["path"],
        "base_sha256": base_spec["sha256"],
    }
    resolved_path = ROOT / resolved["output_dir"] / "resolved_config.json"
    base.atomic_json(resolved_path, resolved)
    return resolved_path


def generation_microbatches(permutation: list[int], batch_size: int) -> list[list[int]]:
    if not permutation or batch_size < 1:
        raise ValueError("Invalid generation-microbatch arguments")
    return [permutation[start : start + batch_size] for start in range(0, len(permutation), batch_size)]


def accumulation_batch_window(
    batches: list[list[int]], accumulation: int, start: int
) -> tuple[int, int]:
    if not batches or accumulation < 1 or start < 0 or start >= len(batches):
        raise ValueError("Invalid accelerated accumulation-window arguments")
    window = batches[start : start + accumulation]
    return len(window), sum(len(batch) for batch in window)


def train_stage(
    config: dict[str, Any],
    config_sha: str,
    arm: str,
    stage_name: str,
    model,
    drafter,
    cache: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    item_index: dict[str, int],
    frozen_train_index: torch.Tensor | None,
    output: Path,
    arm_output: Path,
    device: torch.device,
    arm_step_offset: int,
    starting_arm_optimizer_step: int,
    resume: bool,
) -> tuple[int, dict[str, Any]]:
    stage = config["formal_budget"][stage_name]
    generation_batch_size = stage["generation_microbatch"] if stage_name == "pretrain" else stage["microbatch"]
    parameters = list(model.parameters()) + (list(drafter.parameters()) if drafter is not None else [])
    optimizer = base.AdamW(parameters, lr=stage["learning_rate"], weight_decay=stage["weight_decay"])
    scheduler = base.get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=stage["scheduler_warmup_optimizer_steps"],
        num_training_steps=stage["scheduler_total_optimizer_steps"],
    )
    last_path = arm_output / "checkpoints" / stage_name / "last_state.pt"
    final_path = arm_output / "checkpoints" / stage_name / "final_model.pt"
    metrics_path = arm_output / "metrics.jsonl"
    start_epoch = 1
    stage_optimizer_step = 0
    arm_optimizer_step = starting_arm_optimizer_step
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device)
        if state["config_sha256"] != config_sha or state["arm"] != arm or state["stage"] != stage_name:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        if drafter is not None:
            drafter.load_state_dict(state["drafter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_epoch = int(state["completed_epoch"]) + 1
        stage_optimizer_step = int(state["stage_optimizer_step"])
        arm_optimizer_step = int(state["arm_optimizer_step"])
    elif not resume and (last_path.exists() or final_path.exists()):
        raise RuntimeError("Refusing to overwrite or implicitly resume formal checkpoints")

    stage_started = time.time()
    optimizer.zero_grad(set_to_none=True)
    physical_microsteps_per_epoch = math.ceil(len(rows) / generation_batch_size)
    for epoch in range(start_epoch, stage["epochs"] + 1):
        model.train()
        if drafter is not None:
            drafter.train()
        permutation = base.epoch_permutation(len(rows), config["seed"], arm, stage_name, epoch)
        generation_batches = generation_microbatches(permutation, generation_batch_size)
        if len(generation_batches) != physical_microsteps_per_epoch:
            raise RuntimeError("Accelerated physical-microstep drift")
        epoch_generation_sum = torch.zeros((), device=device)
        epoch_auxiliary_sum = torch.zeros((), device=device)
        finite_window = torch.ones((), dtype=torch.bool, device=device)
        epoch_optimizer_steps = 0
        window_microsteps = 0
        window_examples = 0
        for microstep, generation_indices in enumerate(generation_batches):
            if microstep % stage["gradient_accumulation"] == 0:
                window_microsteps, window_examples = accumulation_batch_window(
                    generation_batches, stage["gradient_accumulation"], microstep
                )
                finite_window.fill_(True)
            generation_context, _ = base.select_cache(cache, generation_indices, device)
            generation_weight = len(generation_indices) / window_examples
            if arm == "S-PLUS" and stage_name == "pretrain":
                embedding_indices = base.embedding_microbatch_indices(
                    permutation, microstep, stage["embedding_microbatch"]
                )
                embedding_context, embedding_target = base.select_cache(cache, embedding_indices, device)
                sequence_embeddings = base.encode_gram(model, drafter, embedding_context)
                item_embeddings = base.encode_gram(model, drafter, embedding_target)
                target_ids = torch.tensor(
                    [item_index[rows[index]["target"]] for index in embedding_indices], device=device
                )
                auxiliary = base.sequence_item_contrastive_loss(
                    sequence_embeddings,
                    item_embeddings,
                    target_ids,
                    temperature=config["model"]["temperature"],
                )
                generation = model(**generation_context, use_cache=False).loss
                objective = (
                    stage["lambda_embedding"] * auxiliary / window_microsteps
                    + stage["lambda_generation"] * generation * generation_weight
                )
                epoch_auxiliary_sum += auxiliary.detach()
            elif arm == "S-PLUS" and stage_name == "finetune":
                if frozen_train_index is None:
                    raise RuntimeError("S-PLUS finetune requires frozen train item index")
                sequence_embeddings = base.encode_gram(model, drafter, generation_context)
                targets = torch.tensor(
                    [item_index[rows[index]["target"]] for index in generation_indices], device=device
                )
                auxiliary = drafter.ranking_loss(
                    sequence_embeddings,
                    frozen_train_index,
                    targets,
                    temperature=config["model"]["temperature"],
                )
                generation = model(**generation_context, use_cache=False).loss
                objective = base.splus_finetune_loss(
                    auxiliary, generation, stage["lambda_embedding"], stage["lambda_generation"]
                ) * generation_weight
                epoch_auxiliary_sum += auxiliary.detach() * len(generation_indices)
            else:
                auxiliary = torch.zeros((), device=device)
                generation = model(**generation_context, use_cache=False).loss
                objective = generation * generation_weight
            finite_window.logical_and_(torch.isfinite(objective.detach()))
            objective.backward()
            epoch_generation_sum += generation.detach() * len(generation_indices)
            end_of_window = (
                (microstep + 1) % stage["gradient_accumulation"] == 0
                or microstep + 1 == len(generation_batches)
            )
            if end_of_window:
                if not bool(finite_window.item()):
                    raise FloatingPointError(f"Non-finite {arm} {stage_name} objective")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, config["formal_budget"]["gradient_clip_norm"]
                )
                if not bool(torch.isfinite(gradient_norm).item()):
                    raise FloatingPointError(f"Non-finite {arm} {stage_name} gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                stage_optimizer_step += 1
                arm_optimizer_step += 1
                epoch_optimizer_steps += 1
                base.write_progress(
                    output,
                    arm,
                    stage_name,
                    epoch - 1,
                    stage_optimizer_step,
                    arm_optimizer_step,
                    arm_step_offset + arm_optimizer_step,
                    2 * (
                        config["formal_budget"]["pretrain"]["optimizer_steps"]
                        + config["formal_budget"]["finetune"]["optimizer_steps"]
                    ),
                )
        if epoch_optimizer_steps != stage["optimizer_steps_per_epoch"]:
            raise RuntimeError(f"{arm} {stage_name} epoch optimizer-step drift")
        auxiliary_denominator = len(generation_batches) if stage_name == "pretrain" else len(rows)
        record = {
            "arm": arm,
            "stage": stage_name,
            "epoch": epoch,
            "optimizer_steps_epoch": epoch_optimizer_steps,
            "stage_optimizer_step": stage_optimizer_step,
            "arm_optimizer_step": arm_optimizer_step,
            "mean_generation_loss": float((epoch_generation_sum / len(rows)).item()),
            "mean_auxiliary_loss": float((epoch_auxiliary_sum / auxiliary_denominator).item()),
            "learning_rate_after_epoch": optimizer.param_groups[0]["lr"],
            "all_finite": True,
            "updated_at": base.utc_now(),
            "test_read": False,
        }
        base.append_jsonl(metrics_path, record)
        if epoch % stage["checkpoint_interval_epochs"] == 0 or epoch == stage["epochs"]:
            base.atomic_torch(
                last_path,
                base.checkpoint_payload(
                    config_sha,
                    arm,
                    stage_name,
                    epoch,
                    arm_optimizer_step,
                    stage_optimizer_step,
                    model,
                    drafter,
                    optimizer,
                    scheduler,
                ),
            )
        base.write_progress(
            output,
            arm,
            stage_name,
            epoch,
            stage_optimizer_step,
            arm_optimizer_step,
            arm_step_offset + arm_optimizer_step,
            2 * (
                config["formal_budget"]["pretrain"]["optimizer_steps"]
                + config["formal_budget"]["finetune"]["optimizer_steps"]
            ),
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if stage_optimizer_step != stage["optimizer_steps"]:
        raise RuntimeError(f"{arm} {stage_name} total optimizer-step drift")
    base.atomic_torch(
        final_path,
        {
            "config_sha256": config_sha,
            "arm": arm,
            "stage": stage_name,
            "completed_epochs": stage["epochs"],
            "stage_optimizer_steps": stage_optimizer_step,
            "arm_optimizer_step": arm_optimizer_step,
            "model": model.state_dict(),
            "drafter": drafter.state_dict() if drafter is not None else None,
        },
    )
    return arm_optimizer_step, {
        "stage": stage_name,
        "epochs_completed": stage["epochs"],
        "physical_microsteps": physical_microsteps_per_epoch * stage["epochs"],
        "optimizer_steps": stage_optimizer_step,
        "scheduler_total_optimizer_steps": stage["scheduler_total_optimizer_steps"],
        "scheduler_warmup_optimizer_steps": stage["scheduler_warmup_optimizer_steps"],
        "runtime_seconds": time.time() - stage_started,
        "final_checkpoint": str(final_path.relative_to(ROOT)),
        "all_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolve-only", action="store_true")
    known, _ = parser.parse_known_args()
    resolved_path = resolve_overlay(known.config.resolve())
    if known.resolve_only:
        print(resolved_path, flush=True)
        return 0
    base.train_stage = train_stage
    rewritten = list(sys.argv)
    config_position = rewritten.index("--config") + 1
    rewritten[config_position] = str(resolved_path)
    sys.argv = rewritten
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
