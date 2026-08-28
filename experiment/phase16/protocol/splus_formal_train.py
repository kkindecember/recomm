#!/usr/bin/env python3
"""FP32 formal S-PLUS or matched S-PLUS-CTRL training on sealed Stage16 data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler

from official_specgr_runtime import sha256, verify_sources
from resource_probe import load_gram
from specgr_contract_smoke import encode_gram, read_jsonl, read_metadata, read_paths, transitions
from specgr_faithful import (
    GRAMSelfDrafter,
    TrainingBudget,
    assert_splus_control_budget_match,
    sequence_item_contrastive_loss,
    splus_finetune_loss,
    splus_pretrain_loss,
)


ROOT = Path(__file__).resolve().parents[3]
MIB = 1024**2


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_set(path: Path) -> set[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate items in {path.relative_to(ROOT)}")
    return set(rows)


def accumulation_window_size(total: int, accumulation: int, start: int) -> int:
    if min(total, accumulation) < 1 or start < 0 or start >= total:
        raise ValueError("Invalid accumulation-window arguments")
    return min(accumulation, total - start)


def _epoch_seed(seed: int, arm: str, stage: str, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}|{arm}|{stage}|{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def epoch_permutation(size: int, seed: int, arm: str, stage: str, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_epoch_seed(seed, arm, stage, epoch))
    return torch.randperm(size, generator=generator).tolist()


def embedding_microbatch_indices(permutation: list[int], microstep: int, batch_size: int) -> list[int]:
    if not permutation or microstep < 0 or batch_size < 1:
        raise ValueError("Invalid embedding-microbatch arguments")
    start = microstep * batch_size
    return [permutation[(start + offset) % len(permutation)] for offset in range(batch_size)]


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def cuda_device_index(device: torch.device) -> int:
    """Return an integer index accepted by the frozen PyTorch 1.11 CUDA API."""
    if device.type != "cuda":
        raise ValueError("CUDA device required")
    return 0 if device.index is None else int(device.index)


def reset_peak_memory_stats_compat(device: torch.device) -> None:
    """Initialize CUDA before the allocator reset required by PyTorch 1.11."""
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))


def build_tensor_cache(
    rows: list[dict[str, Any]],
    metadata: dict[str, str],
    paths: dict[str, tuple[str, ...]],
    tokenizer,
    chunk_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Tokenize once on CPU; compact dtypes avoid millions of tokenizer calls."""
    max_passages, max_length = 21, 128
    label_width = max(len(paths[row["target"]]) + 1 for row in rows)
    pieces: dict[str, list[torch.Tensor]] = {
        "input_ids": [], "attention_mask": [], "labels": [], "target_ids": [], "target_attention": []
    }
    for start in range(0, len(rows), chunk_size):
        selected = rows[start : start + chunk_size]
        all_passages: list[str] = []
        active_counts: list[int] = []
        for row in selected:
            history = row["history"][-20:]
            lexical = " > ".join("|".join(paths[item]) for item in history)
            passages = [f"What would user purchase after {lexical} ?"] + [metadata[item] for item in reversed(history)]
            active_counts.append(len(passages))
            all_passages.extend(passages + [""] * (max_passages - len(passages)))
        encoded = tokenizer(
            all_passages, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        input_ids = encoded.input_ids.reshape(len(selected), max_passages, max_length)
        attention = encoded.attention_mask.reshape(len(selected), max_passages, max_length)
        for row_number, count in enumerate(active_counts):
            input_ids[row_number, count:] = tokenizer.pad_token_id
            attention[row_number, count:] = 0
        labels = torch.full((len(selected), label_width), -100, dtype=torch.int32)
        for row_number, row in enumerate(selected):
            token_ids = tokenizer.convert_tokens_to_ids(list(paths[row["target"]]))
            if any(token == tokenizer.unk_token_id for token in token_ids):
                raise ValueError(f"Lexical token maps to UNK for {row['target']}")
            values = token_ids + [tokenizer.eos_token_id]
            labels[row_number, : len(values)] = torch.tensor(values, dtype=torch.int32)
        targets = tokenizer(
            [metadata[row["target"]] for row in selected],
            padding="max_length", truncation=True, max_length=max_length, return_tensors="pt",
        )
        pieces["input_ids"].append(input_ids.to(torch.int32))
        pieces["attention_mask"].append(attention.to(torch.uint8))
        pieces["labels"].append(labels)
        pieces["target_ids"].append(targets.input_ids[:, None, :].to(torch.int32))
        pieces["target_attention"].append(targets.attention_mask[:, None, :].to(torch.uint8))
    return {name: torch.cat(values) for name, values in pieces.items()}


def select_cache(
    cache: dict[str, torch.Tensor], indices: list[int], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    index = torch.tensor(indices, dtype=torch.long)
    context = {
        "input_ids": cache["input_ids"].index_select(0, index).to(device=device, dtype=torch.long),
        "attention_mask": cache["attention_mask"].index_select(0, index).to(device=device, dtype=torch.long),
        "labels": cache["labels"].index_select(0, index).to(device=device, dtype=torch.long),
    }
    target = {
        "input_ids": cache["target_ids"].index_select(0, index).to(device=device, dtype=torch.long),
        "attention_mask": cache["target_attention"].index_select(0, index).to(device=device, dtype=torch.long),
    }
    return context, target


@torch.no_grad()
def encode_item_index(
    model,
    drafter: GRAMSelfDrafter,
    ordered_items: list[str],
    metadata: dict[str, str],
    tokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    drafter.eval()
    vectors = []
    for start in range(0, len(ordered_items), batch_size):
        items = ordered_items[start : start + batch_size]
        encoded = tokenizer(
            [metadata[item] for item in items],
            padding="max_length", truncation=True, max_length=128, return_tensors="pt",
        )
        batch = {
            "input_ids": encoded.input_ids[:, None, :].to(device),
            "attention_mask": encoded.attention_mask[:, None, :].to(device),
        }
        vectors.append(encode_gram(model, drafter, batch).to(torch.float32))
    return torch.cat(vectors)


def training_budget(config: dict[str, Any], stage_name: str) -> TrainingBudget:
    stage = config["formal_budget"][stage_name]
    microbatch = stage["generation_microbatch"] if stage_name == "pretrain" else stage["microbatch"]
    return TrainingBudget(
        dataset_manifest_sha256=config["inputs"]["split_manifest"]["sha256"],
        transitions=config["formal_budget"]["train_transitions"],
        epochs=stage["epochs"], optimizer="AdamW", learning_rate=stage["learning_rate"],
        weight_decay=stage["weight_decay"], warmup_steps=stage["official_warmup_steps_argument"],
        physical_microbatch=microbatch, gradient_accumulation=stage["gradient_accumulation"],
        optimizer_steps=stage["optimizer_steps"], gpu_count=1,
        timeout_seconds=config["formal_budget"]["per_arm_hard_timeout_seconds"],
    )


def checkpoint_payload(
    config_sha: str,
    arm: str,
    stage_name: str,
    completed_epoch: int,
    arm_optimizer_step: int,
    stage_optimizer_step: int,
    model,
    drafter,
    optimizer,
    scheduler,
) -> dict[str, Any]:
    return {
        "config_sha256": config_sha,
        "arm": arm,
        "stage": stage_name,
        "completed_epoch": completed_epoch,
        "arm_optimizer_step": arm_optimizer_step,
        "stage_optimizer_step": stage_optimizer_step,
        "model": model.state_dict(),
        "drafter": drafter.state_dict() if drafter is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }


def write_progress(
    output: Path,
    arm: str,
    stage: str,
    epoch: int,
    stage_optimizer_step: int,
    arm_optimizer_step: int,
    paired_optimizer_step: int,
    total: int,
) -> None:
    atomic_json(
        output / "progress.json",
        {
            "arm": arm, "stage": stage, "completed_epoch": epoch,
            "stage_optimizer_step": stage_optimizer_step,
            "arm_optimizer_step": arm_optimizer_step,
            "paired_optimizer_step": paired_optimizer_step,
            "paired_optimizer_steps_total": total,
            "updated_at": utc_now(), "test_read": False,
        },
    )


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
    parameters = list(model.parameters()) + (list(drafter.parameters()) if drafter is not None else [])
    optimizer = AdamW(parameters, lr=stage["learning_rate"], weight_decay=stage["weight_decay"])
    scheduler = get_scheduler(
        "cosine", optimizer,
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
    for epoch in range(start_epoch, stage["epochs"] + 1):
        model.train()
        if drafter is not None:
            drafter.train()
        permutation = epoch_permutation(len(rows), config["seed"], arm, stage_name, epoch)
        epoch_generation_sum = torch.zeros((), device=device)
        epoch_auxiliary_sum = torch.zeros((), device=device)
        finite_window = torch.ones((), dtype=torch.bool, device=device)
        epoch_optimizer_steps = 0
        window_size = 0
        for microstep, generation_index in enumerate(permutation):
            if microstep % stage["gradient_accumulation"] == 0:
                window_size = accumulation_window_size(len(rows), stage["gradient_accumulation"], microstep)
                finite_window.fill_(True)
            generation_context, _ = select_cache(cache, [generation_index], device)
            if arm == "S-PLUS" and stage_name == "pretrain":
                embedding_indices = embedding_microbatch_indices(
                    permutation, microstep, stage["embedding_microbatch"]
                )
                embedding_context, embedding_target = select_cache(cache, embedding_indices, device)
                sequence_embeddings = encode_gram(model, drafter, embedding_context)
                item_embeddings = encode_gram(model, drafter, embedding_target)
                target_ids = torch.tensor(
                    [item_index[rows[index]["target"]] for index in embedding_indices], device=device
                )
                auxiliary = sequence_item_contrastive_loss(
                    sequence_embeddings, item_embeddings, target_ids,
                    temperature=config["model"]["temperature"],
                )
                generation = model(**generation_context, use_cache=False).loss
                objective = splus_pretrain_loss(
                    auxiliary, generation, stage["lambda_embedding"], stage["lambda_generation"]
                )
            elif arm == "S-PLUS" and stage_name == "finetune":
                if frozen_train_index is None:
                    raise RuntimeError("S-PLUS finetune requires frozen train item index")
                sequence_embeddings = encode_gram(model, drafter, generation_context)
                targets = torch.tensor([item_index[rows[generation_index]["target"]]], device=device)
                auxiliary = drafter.ranking_loss(
                    sequence_embeddings, frozen_train_index, targets,
                    temperature=config["model"]["temperature"],
                )
                generation = model(**generation_context, use_cache=False).loss
                objective = splus_finetune_loss(
                    auxiliary, generation, stage["lambda_embedding"], stage["lambda_generation"]
                )
            else:
                auxiliary = torch.zeros((), device=device)
                generation = model(**generation_context, use_cache=False).loss
                objective = generation
            finite_window.logical_and_(torch.isfinite(objective.detach()))
            (objective / window_size).backward()
            epoch_generation_sum += generation.detach()
            epoch_auxiliary_sum += auxiliary.detach()
            end_of_window = (microstep + 1) % stage["gradient_accumulation"] == 0 or microstep + 1 == len(rows)
            if end_of_window:
                if not bool(finite_window.item()):
                    raise FloatingPointError(f"Non-finite {arm} {stage_name} objective")
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config["formal_budget"]["gradient_clip_norm"])
                if not bool(torch.isfinite(gradient_norm).item()):
                    raise FloatingPointError(f"Non-finite {arm} {stage_name} gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                stage_optimizer_step += 1
                arm_optimizer_step += 1
                epoch_optimizer_steps += 1
                paired_step = arm_step_offset + arm_optimizer_step
                write_progress(
                    output, arm, stage_name, epoch - 1, stage_optimizer_step,
                    arm_optimizer_step, paired_step,
                    2 * (config["formal_budget"]["pretrain"]["optimizer_steps"] + config["formal_budget"]["finetune"]["optimizer_steps"]),
                )
        if epoch_optimizer_steps != stage["optimizer_steps_per_epoch"]:
            raise RuntimeError(f"{arm} {stage_name} epoch optimizer-step drift")
        record = {
            "arm": arm, "stage": stage_name, "epoch": epoch,
            "optimizer_steps_epoch": epoch_optimizer_steps,
            "stage_optimizer_step": stage_optimizer_step,
            "arm_optimizer_step": arm_optimizer_step,
            "mean_generation_loss": float((epoch_generation_sum / len(rows)).item()),
            "mean_auxiliary_loss": float((epoch_auxiliary_sum / len(rows)).item()),
            "learning_rate_after_epoch": optimizer.param_groups[0]["lr"],
            "all_finite": True, "updated_at": utc_now(), "test_read": False,
        }
        append_jsonl(metrics_path, record)
        if epoch % stage["checkpoint_interval_epochs"] == 0 or epoch == stage["epochs"]:
            atomic_torch(
                last_path,
                checkpoint_payload(
                    config_sha, arm, stage_name, epoch, arm_optimizer_step,
                    stage_optimizer_step, model, drafter, optimizer, scheduler,
                ),
            )
        write_progress(
            output, arm, stage_name, epoch, stage_optimizer_step, arm_optimizer_step,
            arm_step_offset + arm_optimizer_step,
            2 * (config["formal_budget"]["pretrain"]["optimizer_steps"] + config["formal_budget"]["finetune"]["optimizer_steps"]),
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if stage_optimizer_step != stage["optimizer_steps"]:
        raise RuntimeError(f"{arm} {stage_name} total optimizer-step drift")
    atomic_torch(
        final_path,
        {
            "config_sha256": config_sha, "arm": arm, "stage": stage_name,
            "completed_epochs": stage["epochs"], "stage_optimizer_steps": stage_optimizer_step,
            "arm_optimizer_step": arm_optimizer_step, "model": model.state_dict(),
            "drafter": drafter.state_dict() if drafter is not None else None,
        },
    )
    return arm_optimizer_step, {
        "stage": stage_name, "epochs_completed": stage["epochs"],
        "physical_microsteps": len(rows) * stage["epochs"],
        "optimizer_steps": stage_optimizer_step,
        "scheduler_total_optimizer_steps": stage["scheduler_total_optimizer_steps"],
        "scheduler_warmup_optimizer_steps": stage["scheduler_warmup_optimizer_steps"],
        "runtime_seconds": time.time() - stage_started,
        "final_checkpoint": str(final_path.relative_to(ROOT)),
        "all_finite": True,
    }


@torch.no_grad()
def evaluate_generation(
    model, cache: dict[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    model.eval()
    total = torch.zeros((), device=device)
    finite = True
    for index in range(len(cache["input_ids"])):
        context, _ = select_cache(cache, [index], device)
        loss = model(**context, use_cache=False).loss
        finite = finite and bool(torch.isfinite(loss).item())
        total += loss
    return {
        "events": len(cache["input_ids"]),
        "mean_generation_loss": float((total / len(cache["input_ids"])).item()),
        "all_finite": finite,
    }


@torch.no_grad()
def evaluate_pseudo_cold(
    model,
    drafter: GRAMSelfDrafter,
    cache: dict[str, torch.Tensor],
    targets: list[str],
    ordered_catalog: list[str],
    full_index: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    drafter.eval()
    catalog_index = {item: index for index, item in enumerate(ordered_catalog)}
    candidate_order = torch.arange(len(ordered_catalog), device=device)
    hit50 = ndcg10 = reciprocal_rank = 0.0
    all_finite = True
    for start in range(0, len(targets), batch_size):
        indices = list(range(start, min(start + batch_size, len(targets))))
        context, _ = select_cache(cache, indices, device)
        sequence = encode_gram(model, drafter, context)
        scores = F.normalize(sequence, dim=-1) @ F.normalize(full_index, dim=-1).T
        batch_targets = torch.tensor([catalog_index[targets[index]] for index in indices], device=device)
        target_scores = scores.gather(1, batch_targets[:, None])
        greater = (scores > target_scores).sum(dim=1)
        ties_before = ((scores == target_scores) & (candidate_order[None, :] < batch_targets[:, None])).sum(dim=1)
        ranks = greater + ties_before + 1
        hit50 += float((ranks <= 50).sum())
        eligible = ranks <= 10
        ndcg10 += float((eligible / torch.log2(ranks.to(torch.float32) + 1)).sum())
        reciprocal_rank += float((1.0 / ranks.to(torch.float32)).sum())
        all_finite = all_finite and bool(torch.isfinite(scores).all().item())
    count = len(targets)
    return {
        "events": count, "candidate_items": len(ordered_catalog),
        "pseudo_cold_hit_at_50": hit50 / count,
        "pseudo_cold_ndcg_at_10": ndcg10 / count,
        "pseudo_cold_mrr": reciprocal_rank / count,
        "all_finite": all_finite,
        "efficacy_gate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=["S-PLUS", "S-PLUS-CTRL"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["model"]["execution_precision"] != "fp32":
        raise ValueError("This formal protocol is frozen to FP32")
    for relative, expected in config["code_freeze"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Formal code SHA drift: {relative}")
    output = ROOT / config["output_dir"]
    arm_output = output / "arms" / args.arm
    if (arm_output / "summary.json").exists():
        raise SystemExit("Refusing to overwrite completed formal arm")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Formal S-PLUS/CTRL requires exactly one visible GPU")
    device = torch.device("cuda:0")
    started = time.time()
    config_sha = sha256(config_path)

    input_paths: dict[str, Path] = {}
    for name, spec in config["inputs"].items():
        path = (ROOT / spec["path"]).resolve()
        if path.name == "user_sequence.txt" or "test" in path.name.lower():
            raise ValueError(f"Forbidden formal input: {name}")
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"Missing or SHA-drifted formal input: {name}")
        input_paths[name] = path
    resource = json.loads(input_paths["resource_sweep_summary"].read_text(encoding="utf-8"))
    if resource["verdict"] != config["resource_evidence"]["verdict"] or resource["execution_precision"] != "fp32":
        raise ValueError("Formal resource evidence contract failed")
    sources = verify_sources()
    base_checkpoint_before = sha256(input_paths["gram_checkpoint"])
    train_rows = transitions(read_jsonl(input_paths["train_sequences"]), config["seed"])
    internal_dev_rows = transitions(read_jsonl(input_paths["internal_dev_sequences"]), config["seed"])
    pseudo_events = read_jsonl(input_paths["pseudo_cold_events"])
    pseudo_rows = [
        {"user_id": row["user_id"], "position": row["source_position"], "history": row["history"], "target": row["target_item"]}
        for row in pseudo_events
    ]
    metadata = read_metadata(input_paths["metadata"])
    paths = read_paths(input_paths["lexical_paths"])
    retained_warm = read_set(input_paths["retained_warm_items"])
    pseudo_cold = read_set(input_paths["pseudo_cold_items"])
    real_cold = read_set(input_paths["real_cold_items"])
    ordered_train = sorted(retained_warm)
    ordered_catalog = sorted(metadata)
    item_index = {item: index for index, item in enumerate(ordered_train)}
    if len(train_rows) != config["formal_budget"]["train_transitions"]:
        raise ValueError("Formal train transition count drift")
    if len(internal_dev_rows) != config["formal_budget"]["internal_dev_transitions"]:
        raise ValueError("Formal internal-dev transition count drift")
    if len(ordered_train) != config["formal_budget"]["train_catalog_items"]:
        raise ValueError("Formal train catalog count drift")
    if len(ordered_catalog) != config["formal_budget"]["full_catalog_items"]:
        raise ValueError("Formal full catalog count drift")
    if len(pseudo_rows) != config["formal_budget"]["pseudo_cold_events"]:
        raise ValueError("Formal pseudo-cold event count drift")
    if set(ordered_catalog) != retained_warm | pseudo_cold | real_cold:
        raise ValueError("Formal catalog partition mismatch")
    train_labels = {row["target"] for row in train_rows}
    if not train_labels <= retained_warm or train_labels & (pseudo_cold | real_cold):
        raise ValueError("Cold item leaked into formal training labels")
    if any(row["target"] not in retained_warm for row in internal_dev_rows):
        raise ValueError("Internal-dev target outside retained-warm catalog")

    common = {
        "schema_version": config["schema_version"], "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"], "config_sha256": config_sha,
        "test_read": False, "validation_used": False, "network_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "config.json", config)
    atomic_json(output / "source_manifest.json", {**common, **sources})
    atomic_json(
        output / "input_file_sha256.json",
        {**common, "files": {spec["path"]: spec["sha256"] for spec in config["inputs"].values()}},
    )
    atomic_json(
        output / "data_provenance.json",
        {
            **common,
            "training": "S16-1 Toys student-readable interaction train transitions only",
            "admission": "S16-1 user-disjoint internal-dev and held pseudo-cold events only",
            "pseudo_cold_used_for_training": False,
            "cold_interaction_label_leaks": 0,
            "full_catalog_content_only_index": True,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    write_progress(
        output, args.arm, "preprocessing", 0, 0, 0,
        0 if args.arm == "S-PLUS" else config["formal_budget"]["pretrain"]["optimizer_steps"] + config["formal_budget"]["finetune"]["optimizer_steps"],
        2 * (config["formal_budget"]["pretrain"]["optimizer_steps"] + config["formal_budget"]["finetune"]["optimizer_steps"]),
    )
    train_cache = build_tensor_cache(train_rows, metadata, paths, tokenizer)
    internal_dev_cache = build_tensor_cache(internal_dev_rows, metadata, paths, tokenizer)
    pseudo_cache = build_tensor_cache(pseudo_rows, metadata, paths, tokenizer)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    reset_peak_memory_stats_compat(device)
    model = load_gram(input_paths["gram_config"], input_paths["gram_checkpoint"], device).train()
    drafter = GRAMSelfDrafter(model.config.d_model, config["model"]["projection_dimension"]).to(device) if args.arm == "S-PLUS" else None
    # Reset training RNG after optional drafter construction so the shared GRAM
    # dropout stream begins from the same seed in both arms.
    torch.manual_seed(config["seed"] + 1)
    torch.cuda.manual_seed_all(config["seed"] + 1)

    arm_offset = 0 if args.arm == "S-PLUS" else config["formal_budget"]["pretrain"]["optimizer_steps"] + config["formal_budget"]["finetune"]["optimizer_steps"]
    arm_optimizer_step = 0
    arm_optimizer_step, pretrain_result = train_stage(
        config, config_sha, args.arm, "pretrain", model, drafter, train_cache, train_rows,
        item_index, None, output, arm_output, device, arm_offset, arm_optimizer_step, args.resume,
    )
    frozen_train_index = None
    if args.arm == "S-PLUS":
        frozen_train_index = encode_item_index(
            model, drafter, ordered_train, metadata, tokenizer,
            config["formal_budget"]["finetune"]["frozen_train_index_encode_batch"], device,
        ).detach()
        if frozen_train_index.shape != (len(ordered_train), config["model"]["projection_dimension"]):
            raise ValueError("Frozen train item index shape drift")
    arm_optimizer_step, finetune_result = train_stage(
        config, config_sha, args.arm, "finetune", model, drafter, train_cache, train_rows,
        item_index, frozen_train_index, output, arm_output, device, arm_offset,
        arm_optimizer_step, args.resume,
    )
    generation_admission = evaluate_generation(model, internal_dev_cache, device)
    pseudo_admission = None
    if args.arm == "S-PLUS":
        full_index = encode_item_index(
            model, drafter, ordered_catalog, metadata, tokenizer,
            config["admission"]["full_catalog_index_encode_batch"], device,
        ).detach()
        pseudo_admission = evaluate_pseudo_cold(
            model, drafter, pseudo_cache, [row["target"] for row in pseudo_rows],
            ordered_catalog, full_index, config["admission"]["evaluation_microbatch"], device,
        )
    base_checkpoint_after = sha256(input_paths["gram_checkpoint"])
    expected_arm_steps = config["formal_budget"]["pretrain"]["optimizer_steps"] + config["formal_budget"]["finetune"]["optimizer_steps"]
    all_finite = generation_admission["all_finite"] and (pseudo_admission is None or pseudo_admission["all_finite"])
    completed = arm_optimizer_step == expected_arm_steps and base_checkpoint_before == base_checkpoint_after and all_finite
    verdict = f"PASS_S16_2_{args.arm.replace('-', '_')}_FORMAL_EXECUTION" if completed else f"FAIL_S16_2_{args.arm.replace('-', '_')}_FORMAL_EXECUTION"
    budget = {
        "pretrain": assert_splus_control_budget_match(training_budget(config, "pretrain"), training_budget(config, "pretrain")),
        "finetune": assert_splus_control_budget_match(training_budget(config, "finetune"), training_budget(config, "finetune")),
    }
    summary = {
        **common, "status": "completed", "verdict": verdict, "arm": args.arm,
        "execution_precision": "fp32", "formal_training_started": True,
        "stage_results": [pretrain_result, finetune_result],
        "arm_optimizer_steps": arm_optimizer_step,
        "expected_arm_optimizer_steps": expected_arm_steps,
        "budget_audit": budget,
        "internal_dev_generation_admission": generation_admission,
        "pseudo_cold_full_catalog_admission": pseudo_admission,
        "train_transitions": len(train_rows), "internal_dev_transitions": len(internal_dev_rows),
        "train_catalog_items": len(ordered_train), "full_catalog_items": len(ordered_catalog),
        "pseudo_cold_events": len(pseudo_rows), "cold_interaction_label_leaks": 0,
        "base_checkpoint_sha256_before": base_checkpoint_before,
        "base_checkpoint_sha256_after": base_checkpoint_after,
        "base_checkpoint_unchanged": base_checkpoint_before == base_checkpoint_after,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "runtime_seconds": time.time() - started,
        "scientific_scope": "train-only contract/admission; no source validation or test; admission metrics are not efficacy promotion",
        "scientific_efficacy_metric_produced": False,
    }
    atomic_json(arm_output / "summary.json", summary)
    print(verdict, flush=True)
    del train_cache, internal_dev_cache, pseudo_cache, model, drafter
    clear_cuda(device)
    return 0 if completed else 3


if __name__ == "__main__":
    raise SystemExit(main())
