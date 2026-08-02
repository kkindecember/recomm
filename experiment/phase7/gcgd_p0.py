#!/usr/bin/env python3
"""CPU-only P0 lineage audit for Graph-Conditioned GRAM Decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase7.gcgd_v1 import normalize_item_logits  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parse_user_sequence(line: str) -> tuple[str, tuple[str, ...]]:
    user, *items = line.split()
    if not user or len(items) < 3:
        raise ValueError("each user sequence must contain a user and at least three items")
    return user, tuple(items)


def read_train_sequences(path: Path, holdout_positions: int) -> dict[str, tuple[str, ...]]:
    if holdout_positions != 2:
        raise ValueError("P0 is frozen to exactly two holdout positions")
    result: dict[str, tuple[str, ...]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            user, items = parse_user_sequence(line)
            if user in result:
                raise ValueError(f"duplicate user: {user}")
            train_items = items[:-holdout_positions]
            if not train_items:
                raise ValueError(f"empty train sequence: {user}")
            result[user] = train_items
    if not result:
        raise ValueError("no user sequences found")
    return result


def read_lexical_ids(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            item, separator, lexical_id = line.rstrip("\n").partition(" ")
            if not separator or not lexical_id:
                raise ValueError(f"malformed lexical ID row: {line[:80]!r}")
            if item in result:
                raise ValueError(f"duplicate item ID: {item}")
            result[item] = lexical_id
    if not result:
        raise ValueError("no lexical IDs found")
    return result


def encode_item_paths(
    tokenizer,
    item_to_lexical_id: Mapping[str, str],
    decoder_start_token_id: int,
    discard_token_ids: Iterable[int],
) -> dict[str, tuple[int, ...]]:
    discard = set(discard_token_ids)
    result: dict[str, tuple[int, ...]] = {}
    reverse: dict[tuple[int, ...], str] = {}
    for item, lexical_id in item_to_lexical_id.items():
        encoded = tuple(
            token for token in tokenizer.encode(lexical_id) if token not in discard
        )
        path = (decoder_start_token_id, *encoded)
        if len(path) < 3:
            raise ValueError(f"encoded lexical path is too short: {item}")
        if path in reverse:
            raise ValueError(f"duplicate encoded lexical path: {reverse[path]} and {item}")
        reverse[path] = item
        result[item] = path
    return result


def graph_summary(
    train_sequences: Mapping[str, Sequence[str]], catalog: set[str]
) -> tuple[dict, Counter[str]]:
    item_degree: Counter[str] = Counter()
    unique_edges: set[tuple[str, str]] = set()
    repeated_interactions = 0
    for user, items in train_sequences.items():
        seen: set[str] = set()
        for item in items:
            item_degree[item] += 1
            edge = (user, item)
            if edge in unique_edges:
                repeated_interactions += 1
            unique_edges.add(edge)
            seen.add(item)
    missing_from_catalog = sorted(set(item_degree) - catalog)
    if missing_from_catalog:
        raise ValueError(f"train items missing lexical IDs: {missing_from_catalog[:5]}")
    degrees = [item_degree.get(item, 0) for item in catalog]
    result = {
        "users": len(train_sequences),
        "catalog_items": len(catalog),
        "train_interactions": sum(len(items) for items in train_sequences.values()),
        "unique_user_item_edges": len(unique_edges),
        "repeated_user_item_interactions": repeated_interactions,
        "train_covered_catalog_items": sum(value > 0 for value in degrees),
        "cold_catalog_items": sum(value == 0 for value in degrees),
        "catalog_item_coverage": sum(value > 0 for value in degrees) / len(degrees),
        "item_degree_min": min(degrees),
        "item_degree_median": statistics.median(degrees),
        "item_degree_mean": statistics.fmean(degrees),
        "item_degree_max": max(degrees),
    }
    return result, item_degree


def prefix_summary(
    item_paths: Mapping[str, Sequence[int]], item_degree: Mapping[str, int]
) -> dict:
    child_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)
    child_mass: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    path_lengths: list[int] = []
    pseudo_item_logits = {
        item: math.log1p(float(item_degree.get(item, 0)))
        for item in item_paths
    }
    pseudo_log_prob = normalize_item_logits(pseudo_item_logits)
    for item, path_value in item_paths.items():
        path = tuple(path_value)
        path_lengths.append(len(path))
        probability = math.exp(pseudo_log_prob[item])
        for depth in range(len(path)):
            prefix = path[:depth]
            token = path[depth]
            child_sets[prefix].add(token)
            child_mass[prefix][token] += probability
    nodes_by_depth: Counter[int] = Counter(len(prefix) for prefix in child_sets)
    branching_by_depth: dict[str, dict[str, float | int]] = {}
    for depth in sorted(nodes_by_depth):
        values = [len(children) for prefix, children in child_sets.items() if len(prefix) == depth]
        branching_by_depth[str(depth)] = {
            "prefix_nodes": len(values),
            "branching_min": min(values),
            "branching_mean": statistics.fmean(values),
            "branching_max": max(values),
        }
    maximum_mass_error = 0.0
    for masses in child_mass.values():
        total = sum(masses.values())
        if total <= 0:
            raise ValueError("non-positive prefix probability mass")
        maximum_mass_error = max(
            maximum_mass_error,
            abs(sum(value / total for value in masses.values()) - 1.0),
        )
    return {
        "encoded_item_paths": len(item_paths),
        "unique_encoded_item_paths": len(set(tuple(path) for path in item_paths.values())),
        "path_length_min": min(path_lengths),
        "path_length_median": statistics.median(path_lengths),
        "path_length_mean": statistics.fmean(path_lengths),
        "path_length_max": max(path_lengths),
        "prefix_nodes": len(child_sets),
        "branching_by_depth": branching_by_depth,
        "pseudo_degree_probability_mass_max_abs_error": maximum_mass_error,
    }


def validate_config(config: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    execution = config.get("execution")
    integrity = config.get("integrity")
    graph = config.get("graph")
    if not isinstance(execution, Mapping) or not isinstance(integrity, Mapping):
        return ["execution and integrity must be objects"]
    if not isinstance(graph, Mapping):
        return ["graph must be an object"]
    if config.get("decision_status") != "PREREGISTERED_FROZEN_READY_TO_RUN":
        errors.append("P0 config must be preregistered and frozen")
    if config.get("execution_enabled") is not True:
        errors.append("P0 execution must be enabled")
    if execution.get("mode") != "cpu_only" or execution.get("cuda_visible_devices") != "":
        errors.append("P0 must be CPU-only with CUDA_VISIBLE_DEVICES empty")
    if execution.get("physical_gpu_reserved_by_codellama") != 0:
        errors.append("CodeLlama reservation must remain on physical GPU0")
    if execution.get("codellama_reservation_mib") != 30720:
        errors.append("CodeLlama must reserve 30720 MiB during CPU-only P0")
    if execution.get("background_tmux_required") is not True:
        errors.append("P0 must run in background tmux")
    if graph.get("train_slice") != "items[:-2]":
        errors.append("P0 graph must use items[:-2]")
    for key in (
        "checkpoint_loaded",
        "predictions_generated",
        "validation_or_test_target_values_consumed",
        "fresh_validation_read",
        "test_predictions_read",
        "sports_read",
    ):
        if integrity.get(key) is not False:
            errors.append(f"integrity.{key} must be false")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("P0 must not expose a CUDA device")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["name"], local_files_only=True
    )
    datasets = {}
    for dataset, spec in config["datasets"].items():
        sequence_path = ROOT / spec["user_sequence"]
        index_path = ROOT / spec["item_index"]
        train_sequences = read_train_sequences(
            sequence_path, int(config["graph"]["holdout_positions_per_user"])
        )
        lexical_ids = read_lexical_ids(index_path)
        item_paths = encode_item_paths(
            tokenizer,
            lexical_ids,
            int(config["tokenizer"]["decoder_start_token_id"]),
            config["tokenizer"]["discard_token_ids"],
        )
        graph, degrees = graph_summary(train_sequences, set(lexical_ids))
        datasets[dataset] = {
            "input_lineage": {
                "user_sequence_path": spec["user_sequence"],
                "user_sequence_sha256": sha256(sequence_path),
                "item_index_path": spec["item_index"],
                "item_index_sha256": sha256(index_path),
                "hierarchical_id_type": spec["hierarchical_id_type"],
            },
            "graph": graph,
            "prefix": prefix_summary(item_paths, degrees),
        }
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "mode": "run",
            "verification_status": "UNVERIFIED",
        },
        "experiment_id": config["experiment_id"],
        "status": "P0_LINEAGE_MECHANISM_AUDIT_COMPLETE",
        "tokenizer": {
            "name": config["tokenizer"]["name"],
            "vocab_size": len(tokenizer),
            "decoder_start_token_id": config["tokenizer"]["decoder_start_token_id"],
            "discard_token_ids": config["tokenizer"]["discard_token_ids"],
        },
        "datasets": datasets,
        "integrity": {
            **config["integrity"],
            "train_slice": config["graph"]["train_slice"],
            "holdout_positions_removed_before_graph_edges": 2,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_workload": False,
        },
        "next_gate": "RESEARCHER_REVIEW_REQUIRED_BEFORE_P1_IMPLEMENTATION_OR_EXECUTION",
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
