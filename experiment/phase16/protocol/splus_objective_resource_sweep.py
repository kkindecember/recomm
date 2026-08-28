#!/usr/bin/env python3
"""Objective-complete S-PLUS/CTRL resource and throughput sweep; no efficacy result."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from official_specgr_runtime import sha256, verify_sources
from resource_probe import load_gram
from specgr_contract_smoke import (
    encode_gram,
    read_jsonl,
    read_metadata,
    read_paths,
    tokenize_passage_batch,
    transitions,
)
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def precision_context(config: dict[str, Any], device: torch.device):
    precision = config["formal_budget"]["precision"]
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16-mixed":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported resource-sweep precision: {precision}")


def measurement(device: torch.device, started: float, microstep_seconds: list[float], optimizer_seconds: float) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    return {
        "runtime_seconds": time.perf_counter() - started,
        "microstep_seconds": microstep_seconds,
        "median_microstep_seconds": statistics.median(microstep_seconds),
        "optimizer_step_seconds": optimizer_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "all_finite": True,
    }


def prepare_batches(
    rows: list[dict[str, Any]],
    count: int,
    batch_size: int,
    offset: int,
    metadata: dict[str, str],
    paths: dict[str, tuple[str, ...]],
    tokenizer,
    device: torch.device,
) -> list[tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    batches = []
    for step in range(count):
        start = offset + step * batch_size
        selected = rows[start : start + batch_size]
        if len(selected) != batch_size:
            raise ValueError("Insufficient deterministic resource-sweep rows")
        context, target = tokenize_passage_batch(selected, metadata, paths, tokenizer, device)
        batches.append((selected, context, target))
    return batches


def run_pretrain_splus(
    config: dict[str, Any], rows: list[dict[str, Any]], metadata: dict[str, str], paths: dict[str, tuple[str, ...]],
    tokenizer, historical: Path, checkpoint: Path, item_index: dict[str, int], device: torch.device,
) -> dict[str, Any]:
    formal, sweep = config["formal_budget"], config["sweep"]
    stage = formal["pretrain"]
    count = sweep["pretrain_microsteps_per_arm"]
    embedding_batches = prepare_batches(rows, count, stage["embedding_microbatch"], 0, metadata, paths, tokenizer, device)
    generation_batches = prepare_batches(rows, count, stage["generation_microbatch"], 100, metadata, paths, tokenizer, device)
    clear_cuda(device)
    started = time.perf_counter()
    model = load_gram(historical, checkpoint, device).train()
    drafter = GRAMSelfDrafter(model.config.d_model, formal["projection_dimension"]).to(device).train()
    optimizer = AdamW(list(model.parameters()) + list(drafter.parameters()), lr=stage["learning_rate"], weight_decay=stage["weight_decay"])
    optimizer.zero_grad()
    timings, losses = [], []
    for (emb_rows, emb_context, emb_target), (_, gen_context, _) in zip(embedding_batches, generation_batches):
        step_started = time.perf_counter()
        with precision_context(config, device):
            sequence_embeddings = encode_gram(model, drafter, emb_context)
            item_embeddings = encode_gram(model, drafter, emb_target)
            item_ids = torch.tensor([item_index[row["target"]] for row in emb_rows], device=device)
            emb_loss = sequence_item_contrastive_loss(sequence_embeddings, item_embeddings, item_ids)
            gen_loss = model(**gen_context, use_cache=False).loss
            loss = splus_pretrain_loss(emb_loss, gen_loss) / stage["gradient_accumulation"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite S-PLUS pretrain sweep loss")
        loss.backward()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - step_started)
        losses.append({"embedding": float(emb_loss.detach()), "generation": float(gen_loss.detach())})
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(drafter.parameters()), formal["gradient_clip_norm"])
    optimizer_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    result = {"arm": "S-PLUS", "stage": "pretrain", "microsteps": count, "losses": losses, **measurement(device, started, timings, optimizer_seconds)}
    del optimizer, drafter, model, embedding_batches, generation_batches
    clear_cuda(device)
    return result


def run_generative_control(
    arm_stage: str, config: dict[str, Any], rows: list[dict[str, Any]], metadata: dict[str, str],
    paths: dict[str, tuple[str, ...]], tokenizer, historical: Path, checkpoint: Path, device: torch.device,
) -> dict[str, Any]:
    formal, sweep = config["formal_budget"], config["sweep"]
    stage = formal[arm_stage]
    count = sweep[f"{arm_stage}_microsteps_per_arm"]
    batch_size = stage["generation_microbatch"] if arm_stage == "pretrain" else stage["microbatch"]
    batches = prepare_batches(rows, count, batch_size, 500 if arm_stage == "pretrain" else 900, metadata, paths, tokenizer, device)
    clear_cuda(device)
    started = time.perf_counter()
    model = load_gram(historical, checkpoint, device).train()
    optimizer = AdamW(model.parameters(), lr=stage["learning_rate"], weight_decay=stage["weight_decay"])
    optimizer.zero_grad()
    timings, losses = [], []
    for _, context, _ in batches:
        step_started = time.perf_counter()
        with precision_context(config, device):
            generation = model(**context, use_cache=False).loss
            loss = generation / stage["gradient_accumulation"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite CTRL {arm_stage} sweep loss")
        loss.backward()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - step_started)
        losses.append(float(generation.detach()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), formal["gradient_clip_norm"])
    optimizer_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    result = {"arm": "S-PLUS-CTRL", "stage": arm_stage, "microsteps": count, "losses": losses, **measurement(device, started, timings, optimizer_seconds)}
    del optimizer, model, batches
    clear_cuda(device)
    return result


@torch.no_grad()
def build_item_index(
    config: dict[str, Any], model, drafter: GRAMSelfDrafter, ordered_items: list[str], metadata: dict[str, str], tokenizer,
    batch_size: int, device: torch.device,
) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    vectors = []
    for start in range(0, len(ordered_items), batch_size):
        items = ordered_items[start : start + batch_size]
        encoded = tokenizer([metadata[item] for item in items], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        batch = {"input_ids": encoded.input_ids[:, None, :].to(device), "attention_mask": encoded.attention_mask[:, None, :].to(device)}
        with precision_context(config, device):
            vectors.append(encode_gram(model, drafter, batch).to(torch.float32))
    torch.cuda.synchronize(device)
    return torch.cat(vectors), time.perf_counter() - started


def run_finetune_splus(
    config: dict[str, Any], rows: list[dict[str, Any]], metadata: dict[str, str], paths: dict[str, tuple[str, ...]],
    tokenizer, historical: Path, checkpoint: Path, ordered_items: list[str], item_index: dict[str, int], device: torch.device,
) -> dict[str, Any]:
    formal, sweep = config["formal_budget"], config["sweep"]
    stage = formal["finetune"]
    count = sweep["finetune_microsteps_per_arm"]
    batches = prepare_batches(rows, count, stage["microbatch"], 700, metadata, paths, tokenizer, device)
    clear_cuda(device)
    started = time.perf_counter()
    model = load_gram(historical, checkpoint, device).train()
    drafter = GRAMSelfDrafter(model.config.d_model, formal["projection_dimension"]).to(device).train()
    model.eval()
    drafter.eval()
    frozen_index, index_seconds = build_item_index(config, model, drafter, ordered_items, metadata, tokenizer, sweep["item_index_encode_batch"], device)
    model.train()
    drafter.train()
    optimizer = AdamW(list(model.parameters()) + list(drafter.parameters()), lr=stage["learning_rate"], weight_decay=stage["weight_decay"])
    optimizer.zero_grad()
    timings, losses = [], []
    for selected, context, _ in batches:
        step_started = time.perf_counter()
        with precision_context(config, device):
            sequence = encode_gram(model, drafter, context)
            logits = F.normalize(sequence, dim=-1) @ F.normalize(frozen_index, dim=-1).T / formal["temperature"]
            targets = torch.tensor([item_index[row["target"]] for row in selected], device=device)
            ranking = F.cross_entropy(logits, targets)
            generation = model(**context, use_cache=False).loss
            loss = splus_finetune_loss(ranking, generation) / stage["gradient_accumulation"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite S-PLUS finetune sweep loss")
        loss.backward()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - step_started)
        losses.append({"ranking": float(ranking.detach()), "generation": float(generation.detach())})
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(drafter.parameters()), formal["gradient_clip_norm"])
    optimizer_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    result = {"arm": "S-PLUS", "stage": "finetune", "microsteps": count, "item_index_items": len(ordered_items), "item_index_build_seconds": index_seconds, "losses": losses, **measurement(device, started, timings, optimizer_seconds)}
    del optimizer, frozen_index, drafter, model, batches
    clear_cuda(device)
    return result


def project_runtime(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    formal, sweep = config["formal_budget"], config["sweep"]
    lookup = {(row["arm"], row["stage"]): row for row in results}
    components = {}
    total = 0.0
    for arm in formal["arms"]:
        for stage_name in ("pretrain", "finetune"):
            stage = formal[stage_name]
            row = lookup[(arm, stage_name)]
            seconds = stage["physical_microsteps"] * row["median_microstep_seconds"] + stage["optimizer_steps"] * row["optimizer_step_seconds"]
            if arm == "S-PLUS" and stage_name == "finetune":
                seconds += row["item_index_build_seconds"]
            components[f"{arm}:{stage_name}"] = seconds / 3600
            total += seconds
    return {
        "core_pair_gpu_hours": total / 3600,
        "conservative_pair_gpu_hours_lower": total / 3600 * sweep["runtime_projection_lower_multiplier"],
        "conservative_pair_gpu_hours_upper": total / 3600 * sweep["runtime_projection_upper_multiplier"],
        "component_gpu_hours": components,
        "basis": "measured median objective-complete physical microstep plus measured AdamW step; conservative multipliers cover dataloading, internal-dev evaluation, checkpointing, and scheduler overhead",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise SystemExit("Refusing to overwrite completed S-PLUS resource sweep")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("S-PLUS resource sweep requires exactly one visible GPU")
    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    input_paths = {}
    for name, spec in config["inputs"].items():
        path = (ROOT / spec["path"]).resolve()
        if "test" in path.name.lower() or path.name == "user_sequence.txt" or not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"Forbidden, missing, or SHA-drifted sweep input: {name}")
        input_paths[name] = path
    sources = verify_sources()
    checkpoint_before = sha256(input_paths["gram_checkpoint"])
    rows = transitions(read_jsonl(input_paths["train_sequences"]), config["seed"])
    metadata = read_metadata(input_paths["metadata"])
    paths = read_paths(input_paths["lexical_paths"])
    ordered_items = sorted(read_set(input_paths["retained_warm_items"]))
    item_index = {item: index for index, item in enumerate(ordered_items)}
    if len(rows) != config["formal_budget"]["train_transitions"] or len(ordered_items) != config["formal_budget"]["train_catalog_items"]:
        raise ValueError("Formal transition or train-catalog count drift")
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)

    results = [
        run_pretrain_splus(config, rows, metadata, paths, tokenizer, input_paths["gram_config"], input_paths["gram_checkpoint"], item_index, device),
        run_generative_control("pretrain", config, rows, metadata, paths, tokenizer, input_paths["gram_config"], input_paths["gram_checkpoint"], device),
        run_finetune_splus(config, rows, metadata, paths, tokenizer, input_paths["gram_config"], input_paths["gram_checkpoint"], ordered_items, item_index, device),
        run_generative_control("finetune", config, rows, metadata, paths, tokenizer, input_paths["gram_config"], input_paths["gram_checkpoint"], device),
    ]
    pre = config["formal_budget"]["pretrain"]
    fine = config["formal_budget"]["finetune"]
    pre_budget = TrainingBudget(config["inputs"]["split_manifest"]["sha256"], len(rows), pre["epochs"], "AdamW", pre["learning_rate"], pre["weight_decay"], pre["warmup_steps"], pre["generation_microbatch"], pre["gradient_accumulation"], pre["optimizer_steps"], 1, 259200)
    fine_budget = TrainingBudget(config["inputs"]["split_manifest"]["sha256"], len(rows), fine["epochs"], "AdamW", fine["learning_rate"], fine["weight_decay"], fine["warmup_steps"], fine["microbatch"], fine["gradient_accumulation"], fine["optimizer_steps"], 1, 259200)
    budget_audit = {"pretrain": assert_splus_control_budget_match(pre_budget, pre_budget), "finetune": assert_splus_control_budget_match(fine_budget, fine_budget)}
    maximum_peak = max(row["peak_reserved_mib"] for row in results)
    recommended_minimum_free_mib = math.ceil((maximum_peak + 4096) / 1024) * 1024
    eligible = all(row["all_finite"] for row in results) and maximum_peak <= config["sweep"]["maximum_eligible_peak_reserved_mib"] and checkpoint_before == sha256(input_paths["gram_checkpoint"])
    verdict = "PASS_S16_2_SPLUS_OBJECTIVE_RESOURCE_SWEEP" if eligible else "FAIL_S16_2_SPLUS_OBJECTIVE_RESOURCE_SWEEP"
    common = {"schema_version": config["schema_version"], "experiment_id": config["experiment_id"], "attempt_id": config["attempt_id"], "config_sha256": sha256(config_path), "test_read": False, "network_used": False}
    write_json(output / "config.json", config)
    write_json(output / "source_manifest.json", {**common, **sources})
    write_json(output / "input_file_sha256.json", {**common, "files": {spec["path"]: spec["sha256"] for spec in config["inputs"].values()}})
    summary = {**common, "status": "completed", "verdict": verdict, "scientific_efficacy_metric_produced": False, "physical_gpu": config["resources"]["physical_gpu"], "visible_gpu": 0, "holder_released": False, "execution_precision": config["formal_budget"]["precision"], "train_transitions": len(rows), "train_catalog_items": len(ordered_items), "measurements": results, "maximum_peak_reserved_mib": maximum_peak, "recommended_minimum_free_mib": recommended_minimum_free_mib, "recommended_headroom_basis": "ceil_to_1024(maximum_peak_reserved_mib + 4096 MiB)", "budget_audit": budget_audit, "runtime_projection": project_runtime(config, results), "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": sha256(input_paths["gram_checkpoint"]), "checkpoint_unchanged": checkpoint_before == sha256(input_paths["gram_checkpoint"]), "formal_training_started": False, "validation_used": False, "test_read": False}
    write_json(output / "summary.json", summary)
    print(verdict, flush=True)
    print(json.dumps(summary["runtime_projection"]), flush=True)
    return 0 if eligible else 3


if __name__ == "__main__":
    raise SystemExit(main())
