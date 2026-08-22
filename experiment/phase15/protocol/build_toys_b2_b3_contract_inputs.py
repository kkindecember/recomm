"""Build leak-safe Toys inputs for Stage15 B2/B3 contract work.

The command is CPU-only.  It derives all sequential evidence from the audited
train+validation projection and immediately removes the validation target.
Neither original monolithic sequences nor test artifacts are accepted inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in os.sys.path:
    os.sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json, load_item_paths, semantic_tokens  # noqa: E402

from common_adapter import (  # noqa: E402
    iter_train_transitions,
    read_projected_sequences,
    sha256_file,
    train_only_sequences,
)
from genrecedit_gram_adapter import position_population  # noqa: E402
from specgr_gram_adapter import PathCatalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    if repo_root != path and repo_root not in path.parents:
        raise ValueError(f"Path escapes repository root: {value}")
    return path


def _read_set(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        values = {line.strip() for line in handle if line.strip()}
    if not values:
        raise ValueError(f"Empty item set: {path}")
    return values


def collect_train_occurrences(
    train_rows: Mapping[str, Sequence[str]],
    *,
    eligible_items: set[str],
    max_history: int,
) -> dict[str, list[tuple[str, int, tuple[str, ...]]]]:
    if max_history < 1:
        raise ValueError("max_history must be positive")
    occurrences: dict[str, list[tuple[str, int, tuple[str, ...]]]] = defaultdict(list)
    for user in sorted(train_rows):
        items = tuple(train_rows[user])
        for position in range(1, len(items)):
            item = items[position]
            if item not in eligible_items:
                continue
            history = items[max(0, position - max_history) : position]
            occurrences[item].append((user, position, history))
    return dict(occurrences)


def choose_occurrence(
    occurrences: Sequence[tuple[str, int, tuple[str, ...]]],
    *,
    cold_item: str,
    warm_item: str,
    seed: int,
) -> tuple[str, int, tuple[str, ...]]:
    if not occurrences:
        raise ValueError(f"No train occurrence for warm item {warm_item}")
    return min(
        occurrences,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}:{cold_item}:{warm_item}:{row[0]}:{row[1]}".encode("utf-8")
            ).digest(),
            row[0],
            row[1],
        ),
    )


def deterministic_topk(
    scores: torch.Tensor,
    item_ids: Sequence[str],
    k: int,
) -> list[tuple[int, float]]:
    """Top-k by descending score and ascending item id, including cutoff ties."""

    if scores.ndim != 1 or scores.numel() != len(item_ids):
        raise ValueError("Score vector and item IDs do not align")
    if k < 1 or k > scores.numel():
        raise ValueError("Invalid top-k")
    values = scores.detach().cpu()
    cutoff = float(torch.topk(values, k=k, sorted=True).values[-1])
    strict = [index for index, value in enumerate(values.tolist()) if value > cutoff]
    tied = [index for index, value in enumerate(values.tolist()) if value == cutoff]
    strict.sort(key=lambda index: (-float(values[index]), item_ids[index]))
    tied.sort(key=lambda index: item_ids[index])
    chosen = (strict + tied)[:k]
    return [(index, float(values[index])) for index in chosen]


def _load_embeddings(path: Path, metadata_path: Path, catalog_items: set[str]) -> tuple[list[str], torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu")
    required = {"item_ids", "embeddings", "model_name", "pooling", "l2_normalized", "text_source_sha256"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unexpected item embedding payload")
    item_ids = [str(item) for item in payload["item_ids"]]
    embeddings = payload["embeddings"].float().contiguous()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(item_ids):
        raise ValueError("Embedding matrix does not align with item IDs")
    if len(item_ids) != len(set(item_ids)) or set(item_ids) != catalog_items:
        raise ValueError("Embedding item IDs do not exactly match the catalog")
    if not payload["l2_normalized"]:
        raise ValueError("B3 similarity requires frozen L2-normalized embeddings")
    norms = embeddings.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0):
        raise ValueError("Embedding norms violate the frozen L2 contract")
    if payload["text_source_sha256"] != sha256_file(metadata_path):
        raise ValueError("Embedding text provenance does not match item metadata")
    metadata = {
        key: payload[key]
        for key in ("model_name", "pooling", "l2_normalized", "text_source_sha256")
    }
    metadata["shape"] = list(embeddings.shape)
    return item_ids, embeddings, metadata


def run(config_path: Path, repo_root: Path, output_dir: Path) -> dict:
    started = time.time()
    repo_root = repo_root.resolve()
    if repo_root != REPO_ROOT:
        raise ValueError(f"repo-root must be {REPO_ROOT}")
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("domain") != "Toys_cold50" or config.get("split") != "train_only":
        raise ValueError("This frozen builder is Toys train-only")

    input_paths = {name: _resolve(repo_root, value) for name, value in config["inputs"].items()}
    for name, path in input_paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input {name} must be a regular non-symlink file: {path}")
        lowered = str(path).lower()
        if "pred_test" in lowered or "predictions_test" in lowered or path.name == "user_sequence.txt":
            raise ValueError(f"Forbidden Stage15 input: {path}")
    projected = input_paths["projected_sequences"]
    if projected.name != "user_sequence_train_validation.txt":
        raise ValueError("Sequence input must be the audited projection")

    projected_rows = read_projected_sequences(projected)
    train_rows = train_only_sequences(projected_rows)
    item_to_lexical, decoded_to_items = load_item_paths(input_paths["item_paths"])
    if any(len(items) != 1 for items in decoded_to_items.values()):
        raise ValueError("Formal adapter contract requires collision-free catalog paths")
    warm = _read_set(input_paths["warm_items"])
    cold = _read_set(input_paths["cold_items"])
    catalog = PathCatalog.build(item_to_lexical, warm, cold)
    catalog_items = set(item_to_lexical)
    train_events = [item for items in train_rows.values() for item in items]
    unknown_train = set(train_events) - catalog_items
    if unknown_train:
        raise ValueError(f"Train histories contain {len(unknown_train)} unknown items")
    if set(train_events) & cold:
        raise ValueError("Frozen cold items unexpectedly occur in train histories")

    metadata_ids: list[str] = []
    with input_paths["item_metadata"].open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            item, separator, _text = raw.rstrip("\n").partition(" ")
            if not separator or not item:
                raise ValueError(f"Malformed metadata row {line_number}")
            metadata_ids.append(item)
    if len(metadata_ids) != len(set(metadata_ids)) or set(metadata_ids) != catalog_items:
        raise ValueError("Item metadata does not exactly cover the catalog")

    embedding_ids, embeddings, embedding_meta = _load_embeddings(
        input_paths["item_embeddings"], input_paths["item_metadata"], catalog_items
    )
    embedding_index = {item: index for index, item in enumerate(embedding_ids)}

    max_history = int(config["common"]["max_history"])
    occurrences = collect_train_occurrences(
        train_rows, eligible_items=warm, max_history=max_history
    )
    eligible_warm = sorted(occurrences)
    if not eligible_warm:
        raise ValueError("No warm item has a train-only pseudo context")
    eligible_matrix = embeddings[[embedding_index[item] for item in eligible_warm]]

    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Refusing stale staging directory {staging}")
    try:
        specgr_root = staging / "specgr_gram"
        genrecedit_root = staging / "genrecedit_gram"
        for path in (
            specgr_root / "drafter",
            specgr_root / "projection",
            specgr_root / "index",
            genrecedit_root / "covariance",
            genrecedit_root / "edit_requests",
            genrecedit_root / "deltaW",
        ):
            path.mkdir(parents=True, exist_ok=False)

        transition_count = sum(1 for _ in iter_train_transitions(projected_rows))
        atomic_json(
            specgr_root / "drafter" / "training_manifest.json",
            {
                "status": "inputs_ready_training_not_started",
                "source": "audited projection with validation target removed",
                "users": len(train_rows),
                "train_events": len(train_events),
                "next_item_transitions": transition_count,
                "max_history": max_history,
                "validation_target_used": False,
                "test_used": False,
                "frozen_gram_trainable": False,
            },
        )
        atomic_json(
            specgr_root / "projection" / "contract.json",
            {
                "status": "pending_training",
                "input_embedding_dim": int(embeddings.shape[1]),
                "gram_hidden_dim": int(config["specgr"]["projection_dim"]),
                "trainable_state_allowlist": ["drafter", "projection"],
                "frozen_gram_parameter_hash_required_before_after": True,
            },
        )
        score_length_counts: dict[int, int] = defaultdict(int)
        with (specgr_root / "index" / "catalog_index.jsonl").open("x", encoding="utf-8") as handle:
            for item in sorted(catalog.paths):
                score_length = catalog.score_length(
                    item, int(config["specgr"]["minimum_cold_prefix"])
                )
                score_length_counts[score_length] += 1
                handle.write(
                    json.dumps(
                        {
                            "item_id": item,
                            "path": list(catalog.paths[item]),
                            "split": "cold" if item in cold else "warm",
                            "verifier_score_length": score_length,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        atomic_json(
            specgr_root / "index" / "manifest.json",
            {
                "items": len(catalog.paths),
                "warm_items": len(warm),
                "cold_items": len(cold),
                "max_path_depth": catalog.max_depth,
                "score_length_counts": dict(sorted(score_length_counts.items())),
                "collision_count": 0,
                "guided_redrafting": True,
                "adaptive_exit": True,
                "draft_size": int(config["specgr"]["draft_size"]),
                "beam_size": int(config["specgr"]["beam_size"]),
            },
        )

        cold_items = sorted(cold)
        neighbors_per_cold = int(config["genrecedit"]["number_knowledge"])
        batch_size = int(config["genrecedit"]["similarity_batch_size"])
        pseudo_rows = 0
        position_request_count = 0
        neighbor_path = genrecedit_root / "edit_requests" / "pseudo_contexts.jsonl"
        with neighbor_path.open("x", encoding="utf-8") as handle:
            for start in range(0, len(cold_items), batch_size):
                batch_items = cold_items[start : start + batch_size]
                batch_matrix = embeddings[[embedding_index[item] for item in batch_items]]
                similarities = batch_matrix @ eligible_matrix.T
                for row_index, cold_item in enumerate(batch_items):
                    chosen = deterministic_topk(
                        similarities[row_index], eligible_warm, neighbors_per_cold
                    )
                    cold_path = catalog.paths[cold_item]
                    for warm_index, score in chosen:
                        warm_item = eligible_warm[warm_index]
                        user, position, history = choose_occurrence(
                            occurrences[warm_item],
                            cold_item=cold_item,
                            warm_item=warm_item,
                            seed=int(config["common"]["seed"]),
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "cold_item": cold_item,
                                    "cold_path": list(cold_path),
                                    "source_warm_item": warm_item,
                                    "similarity": score,
                                    "train_context_items": list(history),
                                    "source_occurrence_hash": hashlib.sha256(
                                        f"{user}:{position}".encode("utf-8")
                                    ).hexdigest(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        pseudo_rows += 1
                        position_request_count += len(cold_path)

        cold_paths = {item: catalog.paths[item] for item in cold}
        population = position_population(cold_paths)
        atomic_json(
            genrecedit_root / "edit_requests" / "manifest.json",
            {
                "status": "pseudo_contexts_ready_positionwise_expansion_pending_gpu",
                "cold_universe": len(cold),
                "cold_universe_covered": len(cold),
                "pseudo_contexts": pseudo_rows,
                "positionwise_requests_after_expansion": position_request_count,
                "number_knowledge_per_cold": neighbors_per_cold,
                "similarity_source": "frozen BGE content embeddings",
                "context_source": "train-only warm occurrences",
                "validation_occurrence_used": False,
                "test_occurrence_used": False,
                "eos_edited": False,
                "padding_edited": False,
            },
        )
        atomic_json(
            genrecedit_root / "edit_requests" / "position_map.json",
            {
                "positions": [
                    {
                        "position": position,
                        "cold_items_with_position": population[position],
                        "candidate_decoder_layers": list(range(int(config["genrecedit"]["decoder_layers"]))),
                        "selected_layer": None,
                    }
                    for position in sorted(population)
                ],
                "selection_rule": "highest train-only probe accuracy; shallowest layer tie-break",
                "one_one_trigger_required": True,
                "eos_and_padding_inactive": True,
            },
        )
        atomic_json(
            genrecedit_root / "covariance" / "manifest.json",
            {
                "status": "pending_gpu_extraction",
                "source": "train-only contexts",
                "train_events": len(train_events),
                "validation_used": False,
                "test_used": False,
                "candidate_decoder_layers": list(range(int(config["genrecedit"]["decoder_layers"]))),
            },
        )
        atomic_json(
            genrecedit_root / "deltaW" / "manifest.json",
            {
                "status": "pending_layer_probe_covariance_and_edit_solve",
                "base_checkpoint_sha256": sha256_file(input_paths["checkpoint"]),
                "one_bundle_per_lexical_position": True,
                "one_one_trigger_required": True,
                "base_checkpoint_mutation_allowed": False,
            },
        )

        input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
        summary = {
            "experiment_id": "GRAM_STAGE15_S2_TOYS_B2_B3_CONTRACT_INPUTS",
            "status": "completed",
            "verdict": "PASS_B2_B3_INPUT_CONTRACT",
            "domain": "Toys_cold50",
            "split": "train_only",
            "users": len(train_rows),
            "train_events": len(train_events),
            "train_transitions": transition_count,
            "catalog_items": len(catalog.paths),
            "warm_items": len(warm),
            "warm_items_with_train_context": len(eligible_warm),
            "warm_items_without_train_context": len(warm - set(eligible_warm)),
            "cold_items": len(cold),
            "cold_seen_in_train": 0,
            "specgr_catalog_index_rows": len(catalog.paths),
            "genrecedit_cold_coverage": len(cold),
            "genrecedit_pseudo_context_rows": pseudo_rows,
            "genrecedit_positionwise_request_count": position_request_count,
            "embedding": embedding_meta,
            "original_user_sequence_opened": False,
            "similar_item_sasrec_opened": False,
            "test_predictions_opened": False,
            "test_metrics_opened": False,
            "model_training": False,
            "gpu_used": False,
            "runtime_seconds": time.time() - started,
        }
        atomic_json(staging / "config.json", config)
        atomic_json(staging / "input_file_sha256.json", input_hashes)
        atomic_json(staging / "summary.json", summary)
        atomic_json(
            staging / "open_file_manifest.json",
            {
                "opened": [str(path.relative_to(repo_root)) for path in input_paths.values()],
                "original_user_sequence_opened": False,
                "similar_item_sasrec_opened": False,
                "test_predictions_opened": False,
                "test_metrics_opened": False,
            },
        )
        atomic_json(
            staging / "resource_summary.json",
            {
                "mode": "cpu_only",
                "gpu_used": False,
                "runtime_seconds": summary["runtime_seconds"],
                "embedding_similarity_rows": len(cold),
                "embedding_similarity_columns": len(eligible_warm),
                "similarity_batch_size": batch_size,
            },
        )
        staging.rename(output_dir)
    except BaseException as error:
        if staging.is_dir() and not staging.is_symlink():
            atomic_json(
                staging / "status.json",
                {
                    "status": "failed",
                    "reason": type(error).__name__,
                    "automatic_retry": False,
                    "partial_artifacts_preserved": True,
                },
            )
        raise
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    run(args.config, args.repo_root, args.output_dir)


if __name__ == "__main__":
    main()
