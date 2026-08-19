#!/usr/bin/env python3
"""Assign globally unique fixed-length cold IDs while preserving BGE prefixes.

For each cold item, the first three BGE argmax tokens are frozen. Candidate
level-4/5 pairs are the cross product of each head's top-k logits. Within every
shared prefix-3 group, scipy's rectangular linear assignment chooses the
minimum total logit penalty subject to globally unique complete paths and exact
warm-path reservations.

This is a pre-GRAM diagnostic. It never changes a warm ID, never appends a
collision suffix, and refuses to write partial or infeasible output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hierarchical_id_utils import HierIdVocab, format_id_line, read_item_set
from make_collision_safe_ids import collision_stats, read_rows
from semantic_bridge import build_model


@dataclass(frozen=True)
class TailCandidate:
    token4: str
    token5: str
    score: float
    rank4: int
    rank5: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--raw-assigned-id", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--output-id", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-levels", type=int, default=3)
    parser.add_argument("--top-k4", type=int, default=16)
    parser.add_argument("--top-k5", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def assign_unique_tails(
    rows: list[tuple[str, tuple[str, ...]]],
    cold_items: set[str],
    candidates: dict[str, list[TailCandidate]],
    prefix_levels: int = 3,
) -> tuple[list[tuple[str, tuple[str, ...]]], dict]:
    """Return a deterministic, exact minimum-cost assignment per prefix group."""
    if prefix_levels != 3:
        raise ValueError("This frozen P0 implementation requires prefix_levels=3")
    item_order = [item for item, _tokens in rows]
    input_map = dict(rows)
    missing = cold_items - input_map.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} cold items from raw assigned IDs")
    if candidates.keys() != cold_items:
        missing_candidates = cold_items - candidates.keys()
        extra_candidates = candidates.keys() - cold_items
        raise ValueError(
            f"Candidate coverage mismatch: missing={len(missing_candidates)} "
            f"extra={len(extra_candidates)}"
        )
    for item in cold_items:
        if len(input_map[item]) != 5:
            raise ValueError(f"Cold raw ID must have exactly 5 tokens: {item}")
        if not candidates[item]:
            raise ValueError(f"No tail candidates for cold item: {item}")

    warm_items = set(input_map) - cold_items
    warm_reserved = {input_map[item] for item in warm_items}
    output_map = {item: input_map[item] for item in warm_items}
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for item in item_order:
        if item in cold_items:
            groups[input_map[item][:prefix_levels]].append(item)

    assignment_meta: dict[str, dict] = {}
    infeasible_groups: list[dict] = []
    for prefix in sorted(groups):
        items = sorted(groups[prefix])
        candidate_by_item: dict[str, dict[tuple[str, str], TailCandidate]] = {}
        union: set[tuple[str, str]] = set()
        for item in items:
            dedup: dict[tuple[str, str], TailCandidate] = {}
            for candidate in candidates[item]:
                pair = (candidate.token4, candidate.token5)
                path = prefix + pair
                if path in warm_reserved:
                    continue
                previous = dedup.get(pair)
                if previous is None or candidate.score > previous.score:
                    dedup[pair] = candidate
            candidate_by_item[item] = dedup
            union.update(dedup)

        pairs = sorted(union)
        if len(pairs) < len(items) or any(not candidate_by_item[item] for item in items):
            infeasible_groups.append(
                {"prefix": list(prefix), "n_items": len(items), "n_candidates": len(pairs)}
            )
            continue
        pair_to_col = {pair: col for col, pair in enumerate(pairs)}
        forbidden = 1.0e12
        costs = np.full((len(items), len(pairs)), forbidden, dtype=np.float64)
        for row_index, item in enumerate(items):
            best_score = max(c.score for c in candidate_by_item[item].values())
            for pair, candidate in candidate_by_item[item].items():
                # Candidate-column epsilon makes exact-score ties deterministic
                # without materially changing the logit objective.
                col = pair_to_col[pair]
                costs[row_index, col] = (
                    best_score - candidate.score + (col + 1) * 1.0e-12
                )
        row_ind, col_ind = linear_sum_assignment(costs)
        if len(row_ind) != len(items) or np.any(costs[row_ind, col_ind] >= forbidden / 2):
            infeasible_groups.append(
                {"prefix": list(prefix), "n_items": len(items), "n_candidates": len(pairs)}
            )
            continue
        for row_index, col in zip(row_ind.tolist(), col_ind.tolist()):
            item = items[row_index]
            pair = pairs[col]
            selected = candidate_by_item[item][pair]
            raw_tail = input_map[item][3:5]
            raw_candidate = next(
                (
                    c
                    for c in candidates[item]
                    if (c.token4, c.token5) == raw_tail
                ),
                None,
            )
            if raw_candidate is None:
                raise ValueError(f"Raw BGE tail missing from top-k candidates: {item}")
            output_map[item] = prefix + pair
            assignment_meta[item] = {
                "changed": output_map[item] != input_map[item],
                "rank4": selected.rank4,
                "rank5": selected.rank5,
                "max_level_rank": max(selected.rank4, selected.rank5),
                "raw_score": raw_candidate.score,
                "assigned_score": selected.score,
                "logit_penalty": raw_candidate.score - selected.score,
            }

    if infeasible_groups:
        raise RuntimeError(
            "Capacity assignment infeasible under frozen top-k; examples="
            + json.dumps(infeasible_groups[:5], ensure_ascii=False)
        )

    output_rows = [(item, output_map[item]) for item in item_order]
    if len(output_rows) != len(rows) or [x[0] for x in output_rows] != item_order:
        raise AssertionError("Item coverage or row order changed")
    for item in warm_items:
        if output_map[item] != input_map[item]:
            raise AssertionError(f"Warm ID changed: {item}")
    for item in cold_items:
        if len(output_map[item]) != 5:
            raise AssertionError(f"Cold output is not fixed-length 5: {item}")
        if output_map[item][:prefix_levels] != input_map[item][:prefix_levels]:
            raise AssertionError(f"Frozen BGE prefix changed: {item}")
    output_paths = [tokens for _item, tokens in output_rows]
    if len(set(output_paths)) != len(output_paths):
        raise AssertionError("Capacity-aware output IDs are not globally unique")

    penalties = [float(assignment_meta[item]["logit_penalty"]) for item in cold_items]
    max_ranks = [int(assignment_meta[item]["max_level_rank"]) for item in cold_items]
    group_sizes = [len(items) for items in groups.values()]
    changed = sum(bool(assignment_meta[item]["changed"]) for item in cold_items)
    report = {
        "n_items": len(rows),
        "n_warm": len(warm_items),
        "n_cold": len(cold_items),
        "n_cold_changed_from_bge_top1": changed,
        "cold_changed_rate": changed / len(cold_items),
        "warm_ids_unchanged": True,
        "row_order_unchanged": True,
        "cold_prefix_levels_preserved": prefix_levels,
        "cold_prefixes_unchanged": True,
        "cold_fixed_length": 5,
        "cold_appended_suffix_count": 0,
        "input_collision": collision_stats([tokens for _item, tokens in rows]),
        "output_collision": collision_stats(output_paths),
        "groups": {
            "n_prefix3_groups": len(groups),
            "max_group_size": max(group_sizes),
            "infeasible_groups": 0,
        },
        "assignment_rank": {
            "fraction_max_rank_le_1": sum(x <= 1 for x in max_ranks) / len(max_ranks),
            "fraction_max_rank_le_4": sum(x <= 4 for x in max_ranks) / len(max_ranks),
            "fraction_max_rank_le_8": sum(x <= 8 for x in max_ranks) / len(max_ranks),
            "fraction_max_rank_le_16": sum(x <= 16 for x in max_ranks) / len(max_ranks),
            "max": max(max_ranks),
        },
        "logit_penalty": {
            "mean": float(sum(penalties) / len(penalties)),
            "p50": _percentile(penalties, 50),
            "p95": _percentile(penalties, 95),
            "max": max(penalties),
        },
        "sample_changed_items": [
            {
                "item_id": item,
                "before": list(input_map[item]),
                "after": list(output_map[item]),
                **assignment_meta[item],
            }
            for item in item_order
            if item in cold_items and assignment_meta[item]["changed"]
        ][:20],
    }
    return output_rows, report


def build_tail_candidates(args: argparse.Namespace, rows, cold_items):
    import torch

    vocab = HierIdVocab.load(args.vocab_json)
    if vocab.n_levels != 5:
        raise ValueError(f"Expected 5-level vocab, got {vocab.n_levels}")
    checkpoint = torch.load(args.mlp, map_location="cpu")
    if checkpoint.get("encoder_model") != "BAAI/bge-large-en-v1.5":
        raise ValueError("Checkpoint is not the frozen BGE-large-en-v1.5 bridge")
    if checkpoint["level_sizes"] != vocab.level_sizes:
        raise ValueError("Checkpoint/vocab level-size mismatch")
    model = build_model(checkpoint["text_dim"], checkpoint["level_sizes"])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    payload = torch.load(args.embeddings, map_location="cpu")
    if payload.get("model_name") != "BAAI/bge-large-en-v1.5":
        raise ValueError("Embedding payload is not the frozen BGE encoder")
    embeddings = payload["embeddings"]
    item_ids = payload["item_ids"]
    if embeddings.ndim != 2 or embeddings.shape[1] != checkpoint["text_dim"]:
        raise ValueError("Embedding/checkpoint dimension mismatch")
    item_to_row = {item: index for index, item in enumerate(item_ids)}
    raw_map = dict(rows)
    ordered_cold = [item for item, _tokens in rows if item in cold_items]
    missing = [item for item in ordered_cold if item not in item_to_row]
    if missing:
        raise ValueError(f"Missing {len(missing)} cold BGE embeddings")

    result: dict[str, list[TailCandidate]] = {}
    raw_top1_mismatches: list[str] = []
    with torch.no_grad():
        for start in range(0, len(ordered_cold), args.batch_size):
            batch_items = ordered_cold[start : start + args.batch_size]
            indices = torch.tensor([item_to_row[item] for item in batch_items])
            x = embeddings[indices].to(device)
            logits4 = model.heads[3](x)
            logits5 = model.heads[4](x)
            values4, indices4 = torch.topk(logits4, k=args.top_k4, dim=1)
            values5, indices5 = torch.topk(logits5, k=args.top_k5, dim=1)
            values4 = values4.cpu().tolist()
            indices4 = indices4.cpu().tolist()
            values5 = values5.cpu().tolist()
            indices5 = indices5.cpu().tolist()
            for row_index, item in enumerate(batch_items):
                candidates: list[TailCandidate] = []
                for rank4, (index4, value4) in enumerate(
                    zip(indices4[row_index], values4[row_index]), 1
                ):
                    token4 = vocab.per_level_idx_to_token[3][index4]
                    for rank5, (index5, value5) in enumerate(
                        zip(indices5[row_index], values5[row_index]), 1
                    ):
                        token5 = vocab.per_level_idx_to_token[4][index5]
                        candidates.append(
                            TailCandidate(
                                token4=token4,
                                token5=token5,
                                score=float(value4 + value5),
                                rank4=rank4,
                                rank5=rank5,
                            )
                        )
                candidates.sort(
                    key=lambda c: (-c.score, c.rank4, c.rank5, c.token4, c.token5)
                )
                result[item] = candidates
                if (candidates[0].token4, candidates[0].token5) != tuple(raw_map[item][3:5]):
                    raw_top1_mismatches.append(item)
    if raw_top1_mismatches:
        raise ValueError(
            f"Frozen BGE raw IDs do not match checkpoint top-1 for "
            f"{len(raw_top1_mismatches)} cold items"
        )
    return result


def write_rows(path: Path, rows: Iterable[tuple[str, tuple[str, ...]]]) -> None:
    with path.open("w") as handle:
        for item, tokens in rows:
            handle.write(format_id_line(item, list(tokens)) + "\n")


def main() -> None:
    args = parse_args()
    if args.prefix_levels != 3 or args.top_k4 != 16 or args.top_k5 != 16:
        raise ValueError("Frozen P0 requires prefix_levels=3 and top-k4=top-k5=16")
    for path in [args.output_id, args.report]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    rows = read_rows(args.raw_assigned_id)
    cold_items = read_item_set(args.cold_items)
    candidates = build_tail_candidates(args, rows, cold_items)
    output_rows, report = assign_unique_tails(
        rows, cold_items, candidates, prefix_levels=args.prefix_levels
    )
    report.update(
        {
            "protocol": {
                "encoder": "BAAI/bge-large-en-v1.5",
                "bridge": "frozen one-layer independent heads",
                "prefix_levels_frozen": args.prefix_levels,
                "top_k4": args.top_k4,
                "top_k5": args.top_k5,
                "optimizer": "per-prefix rectangular linear_sum_assignment",
                "gram_training_run": False,
            },
            "inputs": {
                "embeddings": str(args.embeddings.resolve()),
                "mlp": str(args.mlp.resolve()),
                "vocab_json": str(args.vocab_json.resolve()),
                "raw_assigned_id": str(args.raw_assigned_id.resolve()),
                "cold_items": str(args.cold_items.resolve()),
            },
            "output_id": str(args.output_id.resolve()),
        }
    )
    args.output_id.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output_id.with_name(args.output_id.name + ".tmp")
    report_tmp = args.report.with_name(args.report.name + ".tmp")
    write_rows(output_tmp, output_rows)
    report_tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output_tmp.replace(args.output_id)
    report_tmp.replace(args.report)
    print(
        f"[capacity-aware] cold={report['n_cold']} "
        f"changed={report['n_cold_changed_from_bge_top1']} "
        f"output_duplicate_excess={report['output_collision']['duplicate_excess']}"
    )


if __name__ == "__main__":
    main()
