#!/usr/bin/env python3
"""Train-only GPU interface smoke for graph-conditioned GRAM decoding."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase3.hbtr_b1_smoke import normalized_sequence  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)

GRAM_SRC = ROOT / "GRAM/src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))
from utils import generation_trie as gt  # noqa: E402


def validate_config(config: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    execution = config.get("execution")
    integrity = config.get("integrity")
    if config.get("decision_status") != "PREREGISTERED_FROZEN_READY_TO_RUN":
        errors.append("GPU smoke config must be preregistered and frozen")
    if config.get("execution_enabled") is not True:
        errors.append("GPU smoke execution must be enabled")
    if not isinstance(execution, Mapping) or not isinstance(integrity, Mapping):
        return errors + ["execution and integrity must be objects"]
    if execution.get("physical_gpu") != 0 or execution.get("cuda_visible_devices") != "0":
        errors.append("GPU smoke must use physical GPU0")
    total = execution.get("total_gpu_lease_mib")
    peak = execution.get("expected_workload_peak_mib")
    sidecar = execution.get("sidecar_reservation_mib")
    if total != 30720 or not all(isinstance(x, int) for x in (peak, sidecar)):
        errors.append("GPU lease declaration is invalid")
    elif peak + sidecar != total:
        errors.append("workload peak plus sidecar must equal 30720 MiB")
    for key in ("fresh_validation_read", "test_predictions_read", "sports_read"):
        if integrity.get(key) is not False:
            errors.append(f"integrity.{key} must be false")
    return errors


def graph_item_logits(
    sequences: Mapping[str, Sequence[str]], user: str, catalog: Sequence[str]
) -> dict[str, float]:
    """Frozen train-only pseudo graph evidence for interface testing, not P1 scoring."""
    degree = Counter(item for items in sequences.values() for item in items[:-2])
    history = set(sequences[user][:-2])
    return {
        item: math.log1p(degree.get(item, 0)) + (1.0 if item in history else 0.0)
        for item in catalog
    }


def prefix_log_probabilities(
    item_paths: Mapping[str, Sequence[int]], item_logits: Mapping[str, float]
) -> dict[tuple[int, ...], dict[int, float]]:
    maximum = max(item_logits.values())
    weights = {item: math.exp(value - maximum) for item, value in item_logits.items()}
    masses: dict[tuple[int, ...], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for item, raw_path in item_paths.items():
        path = tuple(raw_path)
        weight = weights[item]
        for depth in range(1, len(path)):
            masses[path[:depth]][path[depth]] += weight
    result: dict[tuple[int, ...], dict[int, float]] = {}
    for prefix, children in masses.items():
        # A masked/seen item can underflow to exactly zero after the global
        # softmax shift.  Zero-mass children carry no graph evidence and a
        # wholly zero-mass prefix means that the graph branch should abstain;
        # omitting it lets the unchanged GRAM logits decide that step.
        positive_children = {
            token: value for token, value in children.items() if value > 0.0
        }
        total = sum(positive_children.values())
        if total <= 0.0:
            continue
        result[prefix] = {
            token: math.log(value / total)
            for token, value in positive_children.items()
        }
    return result


class GraphPrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_scores: Mapping[tuple[int, ...], Mapping[int, float]], alpha: float):
        self.prefix_scores = prefix_scores
        self.alpha = float(alpha)
        self.calls = 0
        self.applied_rows = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.calls += 1
        if self.alpha == 0.0:
            return scores
        output = scores.clone()
        for row, token_ids in enumerate(input_ids.tolist()):
            graph = self.prefix_scores.get(tuple(token_ids))
            if not graph:
                continue
            for token, value in graph.items():
                output[row, token] += self.alpha * float(value)
            self.applied_rows += 1
        return output


@torch.no_grad()
def generate(backbone, batch: dict, trie, max_length: int, beam_size: int, processor=None):
    kwargs = {}
    if processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    return backbone.generate(
        input_ids=batch["item_text_ids"],
        attention_mask=batch["item_text_masks"],
        max_length=max_length,
        prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(trie),
        num_beams=beam_size,
        num_return_sequences=beam_size,
        return_dict_in_generate=True,
        length_penalty=1.0,
        **kwargs,
    )["sequences"]


def run_dataset(dataset: str, config: dict, parent: dict, output_root: Path) -> dict:
    device = torch.device("cuda:0")
    prepared = prepare(dataset, parent, device)
    checkpoint = ROOT / config["datasets"][dataset]["parent_checkpoint"]
    before = sha256(checkpoint)
    expected = config["datasets"][dataset]["parent_checkpoint_sha256"]
    if before != expected:
        raise ValueError(f"{dataset} parent checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()
    users = read_users(ROOT / config["datasets"][dataset]["train_users"])
    samples = build_train_samples(
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    sample = sorted(samples, key=lambda row: row["sample_key"])[0]
    batch = collate(prepared["collator"], [sample])
    batch["item_text_ids"] = batch["item_text_ids"].to(device)
    batch["item_text_masks"] = batch["item_text_masks"].to(device)
    trie = gt.Trie(prepared["encoded_candidates"])
    item_paths = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    prefix_scores = prefix_log_probabilities(
        item_paths,
        graph_item_logits(prepared["sequences"], sample["user_id"], prepared["catalog"]),
    )
    backbone = prepared["model"].backbone
    beam_size = int(config["smoke"]["beam_size"])
    max_length = max(len(row) for row in prepared["encoded_candidates"])
    torch.cuda.reset_peak_memory_stats(device)
    baseline = generate(backbone, batch, trie, max_length, beam_size)
    identity_processor = GraphPrefixLogitsProcessor(prefix_scores, 0.0)
    identity = generate(backbone, batch, trie, max_length, beam_size, identity_processor)
    active_processor = GraphPrefixLogitsProcessor(prefix_scores, float(config["smoke"]["alpha"]))
    active = generate(backbone, batch, trie, max_length, beam_size, active_processor)
    baseline_items = [prepared["sequence_to_item"].get(normalized_sequence(row.tolist())) for row in baseline]
    identity_items = [prepared["sequence_to_item"].get(normalized_sequence(row.tolist())) for row in identity]
    active_items = [prepared["sequence_to_item"].get(normalized_sequence(row.tolist())) for row in active]
    if any(item is None for item in baseline_items + identity_items + active_items):
        raise ValueError("generated sequence is outside the catalog Trie")
    if baseline_items != identity_items:
        raise ValueError("alpha=0 failed exact GRAM identity")
    result = {
        "dataset": dataset,
        "sample_key": sample["sample_key"],
        "parent_checkpoint_sha256_before": before,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
        "beam_size": beam_size,
        "identity_exact": True,
        "active_processor_calls": active_processor.calls,
        "active_processor_applied_rows": active_processor.applied_rows,
        "active_changed_positions": sum(a != b for a, b in zip(active_items, baseline_items)),
        "all_outputs_map_to_catalog": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    write_json(output_root / dataset / "smoke.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    lineage_path = ROOT / config["inputs"]["p0_lineage_summary"]
    if sha256(lineage_path) != config["inputs"]["p0_lineage_summary_sha256"]:
        raise ValueError("P0 lineage summary SHA mismatch")
    parent = json.loads((ROOT / config["inputs"]["parent_config"]).read_text())
    results = {
        dataset: run_dataset(dataset, config, parent, args.output_root)
        for dataset in config["dataset_order"]
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "PASS",
        "datasets": results,
        "integrity": {
            "train_only": True,
            "fresh_validation_read": False,
            "test_predictions_read": False,
            "sports_read": False,
            "effect_claim_allowed": False,
        },
        "next_gate": "FREEZE_P1_FROM_MEASURED_GPU_PEAK_THEN_RESEARCHER_CONFIRM_START",
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
