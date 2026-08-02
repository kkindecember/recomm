#!/usr/bin/env python3
"""ST-GCGD-v2.1 P0-R2: expanded GRAM bank plus graph-only CUDA peak."""

from __future__ import annotations

import argparse
import gc
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
from experiment.phase7.st_gcgd_v2 import sha256, write_json
from experiment.phase7.st_gcgd_v21 import hard_negative_coverage, load_inputs_v21, train_arm_v21


def harvest_bank(dataset: str, config: dict, graph, device: torch.device) -> tuple[dict[str, list[str]], str]:
    parent_path = ROOT / config["inputs"]["phase4_parent_config"]
    parent = json.loads(parent_path.read_text())
    prepared = prepare(dataset, parent, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"
    expected = config["inputs"]["expected_parent_checkpoint_sha256"][dataset]
    if sha256(checkpoint) != expected:
        raise ValueError(f"{dataset} checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()
    users = read_users(ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt")
    graph_records = {record.user_id: record for record in graph.records if record.user_id in users}
    pool = build_train_samples(prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"])
    candidates = []
    for sample in pool:
        record = graph_records.get(sample["user_id"])
        if record is None or sample["positive_item"] != graph.items[record.target]:
            continue
        expected_index = len(prepared["sequences"][sample["user_id"]][:-2]) - 1
        if sample["sample_key"] != f"{sample['user_id']}:{expected_index}:{sample['positive_item']}":
            continue
        row = dict(sample)
        row["sample_key"] = record.sample_key
        candidates.append(row)
    per_group = int(config["hard_negative_harvest"]["samples_per_group"])
    selected = select_stratified_samples(
        candidates, prepared["heads"], int(config["seed"]),
        f"{dataset}|{config['hard_negative_harvest']['salt']}", per_group, per_group,
    )
    top_k = int(config["hard_negative_harvest"]["generator_top_k"])
    comparator = {"generator_top_k": top_k, "catalog_top_k": top_k}
    bank: dict[str, list[str]] = {}
    for index, sample in enumerate(selected, 1):
        record = build_candidate_record(sample, prepared, comparator, device)
        forbidden = set(sample["history_items"]) | {sample["positive_item"]}
        bank[sample["sample_key"]] = [item for item in record["union"][:top_k] if item not in forbidden]
        if index % 128 == 0:
            print(f"ST_GCGD_V21_BANK dataset={dataset} samples={index}/{len(selected)}", flush=True)
    after = sha256(checkpoint)
    if after != expected:
        raise ValueError(f"{dataset} checkpoint mutated")
    del prepared
    gc.collect()
    torch.cuda.empty_cache()
    return bank, after


def run(dataset: str, config: dict, output_root: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P0-R2 requires CUDA_VISIBLE_DEVICES=0")
    device = torch.device("cuda:0")
    graph, sequence_path, catalog_path = load_inputs_v21(dataset, config)
    bank, checkpoint_after = harvest_bank(dataset, config, graph, device)
    bank_path = output_root / dataset / "gram_hard_negatives.json"
    write_json(bank_path, bank)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    pilot = dict(config["training"])
    pilot["backbone_epochs"] = int(config["memory_pilot"]["backbone_epochs"])
    pilot["mixer_epochs"] = int(config["memory_pilot"]["mixer_epochs"])
    model, history = train_arm_v21(graph, "full", pilot, bank, device)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    budget = int(peak_reserved)
    del model
    torch.cuda.empty_cache()
    summary = {
        "experiment_id": config["experiment_id"], "dataset": dataset,
        "status": "P0_R2_COMPLETE" if budget <= 30720 else "RESOURCE_LEASE_OVERSHOOT",
        "model": {"embedding_dim": config["training"]["embedding_dim"], "layers": config["training"]["layers"], "session_encoder": "GRU", "mixer": "transition-first bounded UI residual"},
        "graph_only_peak_allocated_mib": peak_allocated,
        "graph_only_peak_reserved_mib": peak_reserved,
        "frozen_workload_budget_mib": budget,
        "frozen_sidecar_mib": 30720 - budget,
        "hard_negative_bank": {"records": len(bank), "sha256": sha256(bank_path), **hard_negative_coverage(graph, bank)},
        "training": history,
        "input_lineage": {"checkpoint_sha256_after": checkpoint_after, "user_sequence_sha256": sha256(sequence_path), "item_index_sha256": sha256(catalog_path)},
        "integrity": {"graph_only_peak_reset_after_gram_release": True, "train_only": True, "validation_read": False, "test_read": False, "sports_read": False, "sidecar_active_during_measurement": False},
    }
    write_json(output_root / dataset / "memory_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("execution_enabled") is not True:
        raise ValueError("P0-R2 config is not enabled")
    result = run(args.dataset, config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
