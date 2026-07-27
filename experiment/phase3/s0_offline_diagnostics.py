#!/usr/bin/env python3
"""Stage-3 S0: deterministic, CPU-only diagnostics and beam reranking.

This script never reads test predictions unless ``--mode test`` is explicitly
provided.  Hyperparameters must be selected on validation predictions first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from transformers import AutoTokenizer


METRIC_KS = (5, 10, 50)
COVERAGE_KS = (1, 3, 5, 10, 15, 20)
RERANK_KS = (5, 10, 20)
CONSENSUS_WEIGHTS = (0.0, 0.25)
FUSION_WEIGHTS = (0.05, 0.10, 0.20)
RECENCY_DECAY = 0.90
MAX_HISTORY = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("Beauty", "Toys"))
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("validation", "test"))
    parser.add_argument("--data-root", type=Path, default=Path("GRAM/rec_datasets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="t5-small")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sequences(path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                result[fields[0]] = fields[1:]
    return result


def read_neighbors(path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0] == "anchor":
                continue
            result[fields[0]] = fields[1:]
    return result


def decode_item_ids(
    path: Path, tokenizer_name: str, local_files_only: bool
) -> Tuple[Dict[str, str], Dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=local_files_only
    )
    item_to_text: Dict[str, str] = {}
    text_to_item: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            item, raw_id = line.rstrip("\n").split(" ", 1)
            token_ids = [
                token_id
                for token_id in tokenizer.encode(raw_id)
                if token_id not in (1820, 9175)
            ]
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            item_to_text[item] = decoded
            if decoded in text_to_item:
                duplicates[decoded].extend((text_to_item[decoded], item))
            text_to_item[decoded] = item
    if duplicates:
        examples = list(duplicates.items())[:3]
        raise ValueError(f"Decoded semantic IDs are not unique: {examples}")
    return item_to_text, text_to_item


def read_predictions(
    path: Path, text_to_item: Mapping[str, str]
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    unknown_gold = Counter()
    unknown_pred = Counter()
    footer = {}
    with path.open() as handle:
        header = next(handle, "")
        if not header.startswith("idx\t"):
            raise ValueError(f"Unexpected prediction header in {path}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1 and ": " in fields[0]:
                key, value = fields[0].split(": ", 1)
                footer[key] = float(value)
                continue
            if len(fields) < 6:
                raise ValueError(f"Malformed prediction row {line_number}")
            user = fields[0]
            gold_text, pred_text, score_text = fields[-3:]
            gold_item = text_to_item.get(gold_text)
            if gold_item is None:
                unknown_gold[gold_text] += 1
            pred_strings = pred_text.split("||")
            scores = [float(value) for value in score_text.split("||")]
            if len(pred_strings) != len(scores):
                raise ValueError(f"Prediction/score mismatch on row {line_number}")
            pred_items = []
            for value in pred_strings:
                item = text_to_item.get(value)
                if item is None:
                    unknown_pred[value] += 1
                pred_items.append(item)
            rows.append(
                {
                    "user": user,
                    "gold": gold_item,
                    "gold_text": gold_text,
                    "pred_items": pred_items,
                    "pred_text": pred_strings,
                    "scores": scores,
                }
            )
    audit = {
        "rows": len(rows),
        "footer": footer,
        "unknown_gold_count": sum(unknown_gold.values()),
        "unknown_gold_unique": len(unknown_gold),
        "unknown_prediction_count": sum(unknown_pred.values()),
        "unknown_prediction_unique": len(unknown_pred),
        "unknown_prediction_examples": list(unknown_pred.keys())[:10],
    }
    if unknown_gold:
        raise ValueError(f"Gold semantic IDs failed to map: {list(unknown_gold)[:3]}")
    return rows, audit


def split_sample(sequence: Sequence[str], mode: str) -> Tuple[List[str], str]:
    if mode == "validation":
        return list(sequence[:-2])[-MAX_HISTORY:], sequence[-2]
    return list(sequence[:-1])[-MAX_HISTORY:], sequence[-1]


def training_popularity(sequences: Mapping[str, Sequence[str]]) -> Counter:
    # Both validation and test targets are excluded from the popularity control.
    return Counter(item for sequence in sequences.values() for item in sequence[:-2])


def head_items(popularity: Counter) -> set:
    ordered = sorted(popularity, key=lambda item: (-popularity[item], item))
    cutoff = max(1, math.ceil(len(ordered) * 0.20))
    return set(ordered[:cutoff])


def history_bin(length: int) -> str:
    if length <= 5:
        return "history_01_05"
    if length <= 10:
        return "history_06_10"
    return "history_11_20"


def relation_features(
    history: Sequence[str], candidate: str, neighbors: Mapping[str, Sequence[str]], k: int
) -> Tuple[float, int]:
    best = 0.0
    supporters = 0
    for age, anchor in enumerate(reversed(history)):
        nearest = neighbors.get(anchor, ())[:k]
        try:
            rank = nearest.index(candidate) + 1
        except ValueError:
            continue
        supporters += 1
        recency = RECENCY_DECAY**age
        rank_weight = 1.0 / math.log2(rank + 1.0)
        best = max(best, recency * rank_weight)
    return best, supporters


def metric_at_k(ranking: Sequence[str], gold: str, k: int) -> Tuple[float, float]:
    try:
        rank = ranking[:k].index(gold) + 1
    except ValueError:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1.0)


def summarize(rankings: Iterable[Tuple[Sequence[str], str]]) -> dict:
    totals = {f"recall@{k}": 0.0 for k in METRIC_KS}
    totals.update({f"ndcg@{k}": 0.0 for k in METRIC_KS})
    count = 0
    for ranking, gold in rankings:
        count += 1
        for k in METRIC_KS:
            recall, ndcg = metric_at_k(ranking, gold, k)
            totals[f"recall@{k}"] += recall
            totals[f"ndcg@{k}"] += ndcg
    return {"n": count, **{key: value / count if count else None for key, value in totals.items()}}


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    started = time.time()
    dataset_dir = args.data_root / args.dataset
    index_matches = sorted(dataset_dir.glob("item_generative_indexing_hierarchy_*.txt"))
    if len(index_matches) != 1:
        raise ValueError(f"Expected exactly one hierarchy index, found {index_matches}")
    input_paths = {
        "predictions": args.predictions,
        "user_sequence": dataset_dir / "user_sequence.txt",
        "neighbors": dataset_dir / "similar_item_sasrec.txt",
        "item_index": index_matches[0],
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, text_to_item = decode_item_ids(
        input_paths["item_index"], args.tokenizer, args.local_files_only
    )
    rows, prediction_audit = read_predictions(args.predictions, text_to_item)
    sequences = read_sequences(input_paths["user_sequence"])
    neighbors = read_neighbors(input_paths["neighbors"])
    popularity = training_popularity(sequences)
    heads = head_items(popularity)

    samples = []
    missing_users = []
    target_mismatches = []
    for row in rows:
        sequence = sequences.get(row["user"])
        if sequence is None:
            missing_users.append(row["user"])
            continue
        history, expected_target = split_sample(sequence, args.mode)
        if row["gold"] != expected_target:
            target_mismatches.append((row["user"], row["gold"], expected_target))
        samples.append(
            {
                **row,
                "history": history,
                "pop_group": "head" if expected_target in heads else "tail",
                "history_bin": history_bin(len(history)),
            }
        )
    if missing_users or target_mismatches:
        raise ValueError(
            f"Lineage failure: missing_users={missing_users[:3]}, "
            f"target_mismatches={target_mismatches[:3]}"
        )

    coverage_rows = []
    coverage_sets: Dict[int, Dict[str, set]] = {}
    for k in COVERAGE_KS:
        groups = defaultdict(list)
        per_user = {}
        for sample in samples:
            union = set()
            for anchor in sample["history"]:
                union.update(neighbors.get(anchor, ())[:k])
            latest = set(neighbors.get(sample["history"][-1], ())[:k]) if sample["history"] else set()
            covered = sample["gold"] in union
            latest_covered = sample["gold"] in latest
            per_user[sample["user"]] = union
            values = (covered, latest_covered, len(union))
            groups["overall"].append(values)
            groups[sample["pop_group"]].append(values)
            groups[sample["history_bin"]].append(values)
        coverage_sets[k] = per_user
        for group, values in sorted(groups.items()):
            count = len(values)
            coverage_rows.append(
                {
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "k": k,
                    "group": group,
                    "n": count,
                    "relation_coverage": sum(v[0] for v in values) / count,
                    "latest_item_coverage": sum(v[1] for v in values) / count,
                    "mean_union_size": sum(v[2] for v in values) / count,
                }
            )
    write_csv(
        args.output_dir / "coverage.csv",
        coverage_rows,
        ("dataset", "mode", "k", "group", "n", "relation_coverage", "latest_item_coverage", "mean_union_size"),
    )

    baseline = summarize((sample["pred_items"], sample["gold"]) for sample in samples)
    oracle = {
        "n": len(samples),
        "beam50_target_recall": baseline["recall@50"],
        "oracle_recall@5": baseline["recall@50"],
        "oracle_recall@10": baseline["recall@50"],
    }

    grid_rows = []
    rankings_by_config = {}
    for k in RERANK_KS:
        for consensus_weight in CONSENSUS_WEIGHTS:
            for fusion_weight in FUSION_WEIGHTS:
                config_id = f"k{k}_c{consensus_weight:g}_w{fusion_weight:g}"
                rankings = {}
                for sample in samples:
                    scored = []
                    denom = max(1, len(sample["history"]))
                    for original_rank, (item, model_score) in enumerate(
                        zip(sample["pred_items"], sample["scores"])
                    ):
                        if item is None:
                            relation_score = 0.0
                        else:
                            best, supporters = relation_features(
                                sample["history"], item, neighbors, k
                            )
                            relation_score = best + consensus_weight * supporters / denom
                        score = model_score + fusion_weight * relation_score
                        scored.append((score, -original_rank, item))
                    scored.sort(reverse=True)
                    rankings[sample["user"]] = [value[2] for value in scored]
                metrics = summarize(
                    (rankings[sample["user"]], sample["gold"]) for sample in samples
                )
                eligible = metrics["recall@10"] >= baseline["recall@10"] - 0.005
                grid_rows.append(
                    {
                        "config_id": config_id,
                        "dataset": args.dataset,
                        "mode": args.mode,
                        "k": k,
                        "recency_decay": RECENCY_DECAY,
                        "consensus_weight": consensus_weight,
                        "fusion_weight": fusion_weight,
                        **metrics,
                        "ndcg@10_relative_delta": (metrics["ndcg@10"] / baseline["ndcg@10"] - 1.0),
                        "recall@10_absolute_delta": metrics["recall@10"] - baseline["recall@10"],
                        "selection_eligible": eligible,
                    }
                )
                rankings_by_config[config_id] = rankings
    grid_rows.sort(key=lambda row: row["config_id"])
    write_csv(
        args.output_dir / "rerank_grid.csv", grid_rows, tuple(grid_rows[0].keys())
    )

    eligible_rows = [row for row in grid_rows if row["selection_eligible"]]
    selection_pool = eligible_rows or grid_rows
    selected = max(
        selection_pool,
        key=lambda row: (row["ndcg@10"], row["recall@10"], -row["k"], row["config_id"]),
    )
    selected_rankings = rankings_by_config[selected["config_id"]]

    subgroup_rows = []
    for method in ("baseline", "selected_rerank"):
        for group_type in ("overall", "popularity", "history", "coverage"):
            grouped = defaultdict(list)
            for sample in samples:
                if group_type == "overall":
                    group = "overall"
                elif group_type == "popularity":
                    group = sample["pop_group"]
                elif group_type == "history":
                    group = sample["history_bin"]
                else:
                    union = coverage_sets[selected["k"]][sample["user"]]
                    group = "covered" if sample["gold"] in union else "uncovered"
                ranking = (
                    sample["pred_items"]
                    if method == "baseline"
                    else selected_rankings[sample["user"]]
                )
                grouped[group].append((ranking, sample["gold"]))
            for group, values in sorted(grouped.items()):
                subgroup_rows.append(
                    {
                        "dataset": args.dataset,
                        "mode": args.mode,
                        "method": method,
                        "group_type": group_type,
                        "group": group,
                        **summarize(values),
                    }
                )
    write_csv(
        args.output_dir / "subgroup_metrics.csv",
        subgroup_rows,
        tuple(subgroup_rows[0].keys()),
    )

    baseline_groups = {
        (row["group_type"], row["group"]): row
        for row in subgroup_rows
        if row["method"] == "baseline"
    }
    selected_groups = {
        (row["group_type"], row["group"]): row
        for row in subgroup_rows
        if row["method"] == "selected_rerank"
    }
    hard_group_gains = {}
    for key in (("popularity", "tail"), ("coverage", "uncovered")):
        before, after = baseline_groups[key], selected_groups[key]
        hard_group_gains["/".join(key)] = {
            "recall@10_relative_delta": (
                after["recall@10"] / before["recall@10"] - 1.0
                if before["recall@10"]
                else None
            ),
            "ndcg@10_relative_delta": (
                after["ndcg@10"] / before["ndcg@10"] - 1.0
                if before["ndcg@10"]
                else None
            ),
        }
    primary_gate = (
        selected["ndcg@10_relative_delta"] >= 0.01
        and selected["recall@10_absolute_delta"] >= -0.005
    )
    subgroup_gate = (
        selected["ndcg@10_relative_delta"] >= 0.0
        and any(
            value is not None and value >= 0.03
            for gains in hard_group_gains.values()
            for value in gains.values()
        )
    )
    decision = "GO" if primary_gate or subgroup_gate else "STOP_OR_MODIFY"

    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "s0_offline_v1",
        },
        "dataset": args.dataset,
        "mode": args.mode,
        "baseline": baseline,
        "beam_oracle": oracle,
        "selected_config": selected,
        "hard_group_gains": hard_group_gains,
        "promotion_gate": {
            "primary_gate": primary_gate,
            "subgroup_gate": subgroup_gate,
            "decision": decision,
        },
        "audit": {
            **prediction_audit,
            "missing_users": len(missing_users),
            "target_mismatches": len(target_mismatches),
            "item_count": len(text_to_item),
            "neighbor_anchor_count": len(neighbors),
            "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
            "python": sys.version,
            "platform": platform.platform(),
            "wall_time_seconds": time.time() - started,
        },
        "preregistered_grid": {
            "rerank_k": RERANK_KS,
            "consensus_weights": CONSENSUS_WEIGHTS,
            "fusion_weights": FUSION_WEIGHTS,
            "recency_decay": RECENCY_DECAY,
            "max_history": MAX_HISTORY,
            "selection": "max validation NDCG@10 subject to Recall@10 >= baseline - 0.005",
        },
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
