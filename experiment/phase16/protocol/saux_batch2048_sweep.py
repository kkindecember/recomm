#!/usr/bin/env python3
"""One official-UniSRec batch-2048 optimizer step for S-AUX memory calibration."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam

from official_specgr_runtime import sha256, verify_sources
from saux_formal_train import atomic_json, build_transitions, read_jsonl, read_set
from specgr_faithful import OfficialUniSRecDrafterGRAM, validate_cold_content_only


ROOT = Path(__file__).resolve().parents[3]
MIB = 1024**2


def ceil_to(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise SystemExit("Refusing to overwrite a completed memory sweep")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Memory sweep requires exactly one visible GPU")

    started = time.time()
    device = torch.device("cuda:0")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_paths: dict[str, Path] = {}
    for name, spec in config["inputs"].items():
        path = (ROOT / spec["path"]).resolve()
        if path.name == "user_sequence.txt" or "test" in path.name.lower():
            raise ValueError(f"Forbidden memory-sweep input path: {path.relative_to(ROOT)}")
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"Missing or SHA-drifted memory-sweep input: {name}")
        input_paths[name] = path
    sources = verify_sources()

    sequences = read_jsonl(input_paths["train_sequences"])
    retained_warm = read_set(input_paths["retained_warm_items"])
    pseudo_cold = read_set(input_paths["pseudo_cold_items"])
    real_cold = read_set(input_paths["real_cold_items"])
    embedding_payload = torch.load(input_paths["content_embeddings"], map_location="cpu")
    item_ids = [str(item) for item in embedding_payload["item_ids"]]
    full_embeddings = embedding_payload["embeddings"].to(torch.float32)
    full_index = {item: index for index, item in enumerate(item_ids)}
    if set(item_ids) != retained_warm | pseudo_cold | real_cold:
        raise ValueError("Embedding/cold/warm catalog partition mismatch")

    measurement = config["measurement"]
    ordered_train = sorted(retained_warm)
    if len(ordered_train) != measurement["expected_train_catalog_items"]:
        raise ValueError("Train catalog size drift")
    item_index = {item: index + 1 for index, item in enumerate(ordered_train)}
    train_embeddings = torch.cat(
        [torch.zeros(1, full_embeddings.shape[1]), full_embeddings[[full_index[item] for item in ordered_train]]],
        dim=0,
    )
    histories, lengths, labels, label_items = build_transitions(
        sequences, item_index, measurement["maximum_history"]
    )
    if len(histories) != measurement["expected_train_transitions"]:
        raise ValueError("Train transition count drift")
    cold_audit = validate_cold_content_only(label_items, real_cold | pseudo_cold, item_ids)

    generator = torch.Generator().manual_seed(measurement["selection_order_seed"])
    selected = torch.randperm(len(histories), generator=generator)[: measurement["batch_size"]]
    if len(selected) != 2048:
        raise ValueError("The calibrated batch must contain exactly 2048 transitions")
    batch_histories = histories[selected].to(device)
    batch_lengths = lengths[selected].to(device)
    batch_labels = labels[selected].to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    step_started = time.time()
    wrapper = OfficialUniSRecDrafterGRAM(train_embeddings).to(device).train()
    optimizer = Adam(
        wrapper.parameters(),
        lr=measurement["learning_rate"],
        weight_decay=measurement["weight_decay"],
    )
    optimizer.zero_grad()
    loss = wrapper.calculate_loss(batch_histories, batch_lengths, batch_labels)
    finite_loss = bool(torch.isfinite(loss).item())
    if not finite_loss:
        raise FloatingPointError("Non-finite loss in memory sweep")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.time() - step_started
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / MIB
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / MIB
    free_after, _ = torch.cuda.mem_get_info(device)

    safety_margin_mib = max(4096, int(math.ceil(0.5 * peak_reserved_mib)))
    recommended_free_mib = max(
        int(config["recalibration_rule"]["minimum_recommended_free_mib"]),
        ceil_to(peak_reserved_mib + safety_margin_mib, int(config["recalibration_rule"]["rounding_mib"])),
    )
    eligible = (
        finite_loss
        and len(selected) == 2048
        and len(ordered_train) == 4799
        and peak_reserved_mib <= measurement["maximum_eligible_peak_reserved_mib"]
    )
    verdict = "PASS_S16_2_SAUX_BATCH2048_MEMORY_SWEEP" if eligible else "FAIL_S16_2_SAUX_BATCH2048_MEMORY_SWEEP"
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "config_sha256": sha256(config_path),
        "test_read": False,
        "network_used": False,
    }
    atomic_json(output / "config.json", config)
    atomic_json(output / "source_manifest.json", {**common, **sources})
    atomic_json(
        output / "input_file_sha256.json",
        {**common, "files": {spec["path"]: spec["sha256"] for spec in config["inputs"].values()}},
    )
    summary: dict[str, Any] = {
        **common,
        "status": "completed",
        "verdict": verdict,
        "official_runtime": "PINNED_SPECGR_UNISREC_PLUS_RECBOLE_V1_2_0",
        "physical_gpu": config["resources"]["physical_gpu"],
        "visible_gpu": 0,
        "batch_size": len(selected),
        "optimizer_steps": 1,
        "train_transitions": len(histories),
        "train_catalog_items": len(ordered_train),
        "finite_loss": finite_loss,
        "loss": float(loss.detach()),
        "peak_cuda_allocated_mib": peak_allocated_mib,
        "peak_cuda_reserved_mib": peak_reserved_mib,
        "cuda_total_mib": total_memory / MIB,
        "cuda_free_before_mib": free_before / MIB,
        "cuda_free_after_mib": free_after / MIB,
        "optimizer_step_seconds": step_seconds,
        "runtime_seconds": time.time() - started,
        "recalibration_eligible": eligible,
        "safety_margin_mib": safety_margin_mib,
        "recommended_minimum_free_mib": recommended_free_mib,
        "recalibration_rule": config["recalibration_rule"],
        "cold_content_only": cold_audit,
        "validation_used": False,
        "scientific_scope": config["scientific_scope"],
        "test_read": False,
    }
    atomic_json(output / "summary.json", summary)
    print(verdict, flush=True)
    print(json.dumps({"peak_reserved_mib": peak_reserved_mib, "recommended_minimum_free_mib": recommended_free_mib}), flush=True)
    return 0 if eligible else 3


if __name__ == "__main__":
    raise SystemExit(main())
