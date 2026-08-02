#!/usr/bin/env python3
"""P0-R full CUDA lifecycle pilot and frozen-GRAM hard-negative harvest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gacr_s0 import build_candidate_record, select_stratified_samples
from experiment.phase4.gcdh_p0 import build_train_samples, prepare, read_users
from experiment.phase7.st_gcgd_v2 import load_inputs, sha256, train_arm, validate_config, write_json


def run(dataset: str, config: dict, output_root: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("memory pilot requires CUDA_VISIBLE_DEVICES=0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    parent_config_path = ROOT / config["inputs"]["phase4_parent_config"]
    parent = json.loads(parent_config_path.read_text())
    prepared = prepare(dataset, parent, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"
    expected = config["inputs"]["expected_parent_checkpoint_sha256"][dataset]
    if sha256(checkpoint) != expected:
        raise ValueError(f"{dataset} checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()
    users = read_users(ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt")
    graph, sequence_path, catalog_path = load_inputs(dataset, config)
    graph_records = {record.user_id: record for record in graph.records if record.user_id in users}
    sample_pool = build_train_samples(prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"])
    samples = []
    for sample in sample_pool:
        record = graph_records.get(sample["user_id"])
        if record is None or sample["positive_item"] != graph.items[record.target]:
            continue
        expected_index = len(prepared["sequences"][sample["user_id"]][:-2]) - 1
        if sample["sample_key"] != f"{sample['user_id']}:{expected_index}:{sample['positive_item']}":
            continue
        sample = dict(sample)
        sample["sample_key"] = record.sample_key
        samples.append(sample)
    count = int(config["memory_pilot"]["hard_negative_samples_per_group"])
    selected = select_stratified_samples(samples, prepared["heads"], int(config["seed"]), f"{dataset}|st-gcgd-v2-p0-r", count, count)
    comparator = {
        "generator_top_k": int(config["memory_pilot"]["generator_top_k"]),
        "catalog_top_k": int(config["memory_pilot"]["generator_top_k"]),
    }
    hard_cache: dict[str, list[str]] = {}
    for index, sample in enumerate(selected, 1):
        record = build_candidate_record(sample, prepared, comparator, device)
        forbidden = set(sample["history_items"]) | {sample["positive_item"]}
        hard_cache[sample["sample_key"]] = [item for item in record["union"][: comparator["generator_top_k"]] if item not in forbidden]
        if index % 16 == 0:
            print(f"ST_GCGD_V2_P0_R_BEAM dataset={dataset} samples={index}/{len(selected)}", flush=True)
    # Keep the frozen GRAM model resident while exercising the complete ST graph
    # forward/backward lifecycle. This is the conservative P1 peak path.
    pilot_training = dict(config["training"])
    pilot_training["epochs"] = int(config["memory_pilot"]["graph_epochs"])
    model, history = train_arm(graph, "full", pilot_training, hard_cache, device)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    del model
    torch.cuda.empty_cache()
    hard_path = output_root / dataset / "gram_hard_negatives.json"
    write_json(hard_path, hard_cache)
    summary = {
        "experiment_id": config["experiment_id"], "dataset": dataset,
        "status": "P0_R_MEMORY_PILOT_COMPLETE",
        "workload_peak_allocated_mib": peak_allocated,
        "workload_peak_reserved_mib": peak_reserved,
        "recommended_workload_budget_mib": int((peak_reserved + 255) // 256 * 256),
        "recommended_sidecar_mib": 30720 - int((peak_reserved + 255) // 256 * 256),
        "hard_negative_records": len(hard_cache), "hard_negative_cache_sha256": sha256(hard_path),
        "graph_training": history,
        "input_lineage": {"checkpoint_sha256_before": expected, "checkpoint_sha256_after": sha256(checkpoint), "user_sequence_sha256": sha256(sequence_path), "item_index_sha256": sha256(catalog_path)},
        "integrity": {"train_only": True, "validation_read": False, "test_read": False, "sports_read": False, "sidecar_active_during_measurement": False},
    }
    if summary["recommended_sidecar_mib"] < 0:
        summary["status"] = "RESOURCE_LEASE_OVERSHOOT"
    write_json(output_root / dataset / "memory_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, "p0-r")
    result = run(args.dataset, config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
