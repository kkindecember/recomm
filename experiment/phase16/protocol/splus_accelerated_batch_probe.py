#!/usr/bin/env python3
"""Train-only FP32 S-PLUS batch probe; produces no efficacy metric."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from official_specgr_runtime import sha256, verify_sources
from resource_probe import load_gram
from specgr_contract_smoke import encode_gram, read_jsonl, read_metadata, read_paths, tokenize_passage_batch, transitions
from specgr_faithful import GRAMSelfDrafter, sequence_item_contrastive_loss, splus_pretrain_loss


ROOT = Path(__file__).resolve().parents[3]
MIB = 1024**2


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def prepare_batches(rows, count, batch_size, offset, metadata, paths, tokenizer, device):
    batches = []
    for step in range(count):
        start = offset + step * batch_size
        selected = rows[start : start + batch_size]
        if len(selected) != batch_size:
            raise ValueError("Insufficient deterministic batch-probe rows")
        context, target = tokenize_passage_batch(selected, metadata, paths, tokenizer, device)
        batches.append((selected, context, target))
    return batches


def candidate_spec(config: dict[str, Any], candidate_id: str) -> dict[str, int]:
    for candidate in config["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise ValueError(f"Unknown candidate: {candidate_id}")


def run_candidate(config_path: Path, candidate_id: str) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = candidate_spec(config, candidate_id)
    output = ROOT / config["output_dir"] / "candidates" / candidate_id
    result_path = output / "result.json"
    if result_path.exists():
        raise SystemExit(f"Refusing to overwrite candidate result: {result_path}")
    source_path = ROOT / config["source_formal_config"]["path"]
    if sha256(source_path) != config["source_formal_config"]["sha256"]:
        raise ValueError("Source formal config SHA drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "candidate_id": candidate_id,
        "config_sha256": sha256(config_path),
        "candidate": candidate,
        "physical_gpu": config["resources"]["physical_gpu"],
        "visible_gpu": 0,
        "execution_precision": "fp32",
        "scientific_efficacy_metric_produced": False,
        "validation_used": False,
        "test_read": False,
        "automatic_retry": False,
    }
    started = time.perf_counter()
    checkpoint_before = "unread"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Batch probe requires exactly one visible CUDA device")
        device = torch.device("cuda:0")
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        inputs = {}
        for name in ("train_sequences", "retained_warm_items", "lexical_paths", "metadata", "gram_config", "gram_checkpoint"):
            spec = source["inputs"][name]
            path = ROOT / spec["path"]
            if not path.is_file() or sha256(path) != spec["sha256"]:
                raise ValueError(f"Missing or SHA-drifted probe input: {name}")
            inputs[name] = path
        verify_sources()
        checkpoint_before = sha256(inputs["gram_checkpoint"])
        rows = transitions(read_jsonl(inputs["train_sequences"]), source["seed"])
        metadata = read_metadata(inputs["metadata"])
        paths = read_paths(inputs["lexical_paths"])
        ordered_items = sorted(read_set(inputs["retained_warm_items"]))
        item_index = {item: index for index, item in enumerate(ordered_items)}
        tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
        count = config["probe_microsteps"]
        emb_batches = prepare_batches(
            rows, count, candidate["embedding_microbatch"], 0, metadata, paths, tokenizer, device
        )
        gen_batches = prepare_batches(
            rows, count, candidate["generation_microbatch"], 4096, metadata, paths, tokenizer, device
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        model = load_gram(inputs["gram_config"], inputs["gram_checkpoint"], device).train()
        drafter = GRAMSelfDrafter(model.config.d_model, source["model"]["projection_dimension"]).to(device).train()
        parameters = list(model.parameters()) + list(drafter.parameters())
        optimizer = AdamW(
            parameters,
            lr=source["formal_budget"]["pretrain"]["learning_rate"],
            weight_decay=source["formal_budget"]["pretrain"]["weight_decay"],
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        timings = []
        losses = []
        for (emb_rows, emb_context, emb_target), (_, gen_context, _) in zip(emb_batches, gen_batches):
            step_started = time.perf_counter()
            sequence_embeddings = encode_gram(model, drafter, emb_context)
            item_embeddings = encode_gram(model, drafter, emb_target)
            item_ids = torch.tensor([item_index[row["target"]] for row in emb_rows], device=device)
            embedding_loss = sequence_item_contrastive_loss(
                sequence_embeddings,
                item_embeddings,
                item_ids,
                temperature=source["model"]["temperature"],
            )
            generation_loss = model(**gen_context, use_cache=False).loss
            objective = splus_pretrain_loss(
                embedding_loss,
                generation_loss,
                source["formal_budget"]["pretrain"]["lambda_embedding"],
                source["formal_budget"]["pretrain"]["lambda_generation"],
            ) / count
            if not bool(torch.isfinite(objective).item()):
                raise FloatingPointError("Non-finite accelerated-batch objective")
            objective.backward()
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - step_started)
            losses.append({"embedding": float(embedding_loss.detach()), "generation": float(generation_loss.detach())})
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, source["formal_budget"]["gradient_clip_norm"])
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("Non-finite accelerated-batch gradient norm")
        optimizer_started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        peak_allocated = torch.cuda.max_memory_allocated(0) / MIB
        peak_reserved = torch.cuda.max_memory_reserved(0) / MIB
        warm_timings = timings[1:] if len(timings) > 1 else timings
        eligible = peak_reserved <= config["maximum_eligible_peak_reserved_mib"]
        result = {
            **common,
            "status": "completed",
            "verdict": "PASS" if eligible else "FAIL_PEAK_CEILING",
            "probe_microsteps": count,
            "losses": losses,
            "all_finite": True,
            "gradient_norm": float(gradient_norm.detach()),
            "microstep_seconds": timings,
            "median_warm_microstep_seconds": statistics.median(warm_timings),
            "optimizer_step_seconds": optimizer_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "maximum_eligible_peak_reserved_mib": config["maximum_eligible_peak_reserved_mib"],
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": sha256(inputs["gram_checkpoint"]),
            "checkpoint_unchanged": checkpoint_before == sha256(inputs["gram_checkpoint"]),
        }
        atomic_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if eligible else 3
    except Exception as error:
        is_cuda_oom = isinstance(error, RuntimeError) and "CUDA out of memory" in str(error)
        result = {
            **common,
            "status": "failed",
            "verdict": "FAIL_CUDA_OOM" if is_cuda_oom else "FAIL_PROBE_ERROR",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
            "checkpoint_sha256_before": checkpoint_before,
            "scientific_failure": False if is_cuda_oom else None,
        }
        atomic_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 4 if is_cuda_oom else 5


def aggregate(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise SystemExit("Refusing to overwrite accelerated-batch summary")
    results = []
    selected = None
    for candidate in config["candidates"]:
        result_path = output / "candidates" / candidate["candidate_id"] / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["result_sha256"] = sha256(result_path)
        results.append(result)
        if selected is None and result["verdict"] == "PASS":
            selected = result
    summary = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "config_sha256": sha256(config_path),
        "status": "completed" if selected else "failed",
        "verdict": "PASS_S16_2_SPLUS_ACCELERATED_BATCH_SWEEP" if selected else "FAIL_S16_2_SPLUS_ACCELERATED_BATCH_SWEEP",
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_measurement": selected,
        "candidate_results": results,
        "selection_rule": "first PASS in preregistered descending-throughput order",
        "maximum_eligible_peak_reserved_mib": config["maximum_eligible_peak_reserved_mib"],
        "holder_released_during_probe": True,
        "holder_restore_required_after_probe": True,
        "scientific_efficacy_metric_produced": False,
        "validation_used": False,
        "test_read": False,
        "automatic_retry": False,
    }
    atomic_json(summary_path, summary)
    print(summary["verdict"], flush=True)
    return 0 if selected else 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.aggregate:
        return aggregate(config_path)
    if not args.candidate_id:
        parser.error("--candidate-id is required unless --aggregate is used")
    return run_candidate(config_path, args.candidate_id)


if __name__ == "__main__":
    raise SystemExit(main())
