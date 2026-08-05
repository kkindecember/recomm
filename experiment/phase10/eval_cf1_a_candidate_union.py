#!/usr/bin/env python3
"""Evaluate frozen GRAM/CF candidate-union coverage on Toys validation."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9 = REPO_ROOT / "experiment/phase9"
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from eval_cf0_b3_beamfusion import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    DEFAULT_PREDICTIONS,
    load_cached_beams,
    load_catalog,
    load_users,
)
from train_cf0_b2_item_head import CF0B2ItemHead  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_a_toys_candidate_union"
EXPECTED_CF_RECALL50 = 0.17463424685761383


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-users", type=int, default=0)
    return parser.parse_args()


def coverage(values):
    values = np.asarray(list(values), dtype=np.bool_)
    return float(values.mean()) if values.size else float("nan")


def union_diagnostics(gram_ids, cf_ids):
    gram = set(gram_ids)
    cf = set(cf_ids)
    union = gram | cf
    intersection = gram & cf
    return {
        "union_size": len(union),
        "intersection_size": len(intersection),
        "cf_only_size": len(cf - gram),
        "jaccard": len(intersection) / len(union),
    }


def scientific_gate(metrics, full_run):
    if not full_run:
        return {"status": "not_evaluated_smoke", "checks": {}}
    checks = {
        "union_coverage_gain_at_least_0.030": (
            metrics["coverage"]["U50"] - metrics["coverage"]["G50"] >= 0.030
        ),
        "tail_complementary_coverage_at_least_0.020": (
            metrics["stratified"]["target_tail"]["complementary_C50_not_G50"] >= 0.020
        ),
        "union_size_le90_fraction_at_least_0.80": (
            metrics["union_size"]["fraction_le_90"] >= 0.80
        ),
        "cf_recall50_identity": math.isclose(
            metrics["coverage"]["C50"], EXPECTED_CF_RECALL50, abs_tol=1e-12
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed_candidate_union_gate",
        "checks": checks,
    }


def load_item_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["model_config"]
    model = CF0B2ItemHead(
        num_items=config["num_items"],
        max_history=config["max_history"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        temperature=config["temperature_initial"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def retrieve_cf_top50(model, config, histories, batch_size):
    item_vectors = F.normalize(model.item_embedding.weight[1:], dim=-1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    results = []
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            batch = histories[start : start + batch_size]
            history_ids = torch.zeros(len(batch), config["max_history"], dtype=torch.long)
            history_mask = torch.zeros_like(history_ids, dtype=torch.bool)
            for row, history in enumerate(batch):
                trimmed = history[-config["max_history"] :]
                history_ids[row, : len(trimmed)] = torch.tensor(trimmed)
                history_mask[row, : len(trimmed)] = True
            users = model.encode(history_ids, history_mask)
            logits = scale * F.linear(users, item_vectors)
            top50 = torch.topk(logits, k=50, dim=1, largest=True, sorted=True).indices + 1
            results.extend(top50.tolist())
    return results


def summarize_stratum(rows):
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "G50": coverage(row["hit_g50"] for row in rows),
        "C50": coverage(row["hit_c50"] for row in rows),
        "U50": coverage(row["hit_u50"] for row in rows),
        "complementary_C50_not_G50": coverage(
            row["hit_c50"] and not row["hit_g50"] for row in rows
        ),
    }


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(args.data_dir)
    users = load_users(args.data_dir, raw_to_id)
    cache, footer = load_cached_beams(args.predictions)
    if set(users) != set(cache):
        raise ValueError("validation cache/data user sets differ")
    if len(users) != 19412 or len(raw_to_id) != 11924:
        raise ValueError("unexpected validation user or catalog size")

    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    target_frequency_values = sorted(
        frequencies[sequence[-2]] for sequence in users.values()
    )
    q1 = target_frequency_values[len(target_frequency_values) // 4]
    q3 = target_frequency_values[3 * len(target_frequency_values) // 4]
    user_ids = list(users)
    if args.max_users:
        user_ids = user_ids[: args.max_users]
    full_run = len(user_ids) == 19412

    histories = []
    targets = []
    gram_top50 = []
    for user in user_ids:
        sequence = users[user]
        histories.append(sequence[max(0, len(sequence) - 22) : -2])
        targets.append(sequence[-2])
        try:
            gram_ids = [lexical_to_id[value] for value in cache[user]["candidates"]]
        except KeyError as error:
            raise ValueError(f"{user}: unmapped GRAM candidate {error}") from error
        if len(gram_ids) != 50 or len(set(gram_ids)) != 50:
            raise ValueError(f"{user}: invalid GRAM beam set")
        gram_top50.append(gram_ids)

    model, config = load_item_model(args.checkpoint)
    if config["num_items"] != len(raw_to_id):
        raise ValueError("checkpoint/catalog mismatch")
    cf_top50 = retrieve_cf_top50(model, config, histories, args.batch_size)

    rows = []
    for user, history, target, gram_ids, cf_ids in zip(
        user_ids, histories, targets, gram_top50, cf_top50
    ):
        record = {
            "user": user,
            "history_length": len(history),
            "target_frequency": frequencies[target],
            "hit_g50": target in gram_ids,
            "hit_c10": target in cf_ids[:10],
            "hit_c20": target in cf_ids[:20],
            "hit_c50": target in cf_ids,
        }
        for cutoff in (10, 20, 50):
            record[f"hit_u{cutoff}"] = target in set(gram_ids).union(cf_ids[:cutoff])
        record.update(union_diagnostics(gram_ids, cf_ids))
        filtered_union = (set(gram_ids) | set(cf_ids)) - set(history)
        record["filtered_union_size"] = len(filtered_union)
        record["hit_filtered_u50"] = target in filtered_union
        rows.append(record)

    union_sizes = np.asarray([row["union_size"] for row in rows])
    intersection_sizes = np.asarray([row["intersection_size"] for row in rows])
    cf_only_sizes = np.asarray([row["cf_only_size"] for row in rows])
    jaccards = np.asarray([row["jaccard"] for row in rows])
    filtered_sizes = np.asarray([row["filtered_union_size"] for row in rows])
    popularity_groups = {
        "target_tail": [row for row in rows if row["target_frequency"] <= q1],
        "target_middle": [row for row in rows if q1 < row["target_frequency"] < q3],
        "target_head": [row for row in rows if row["target_frequency"] >= q3],
    }
    history_groups = {
        "history_1-5": [row for row in rows if row["history_length"] <= 5],
        "history_6-10": [row for row in rows if 6 <= row["history_length"] <= 10],
        "history_11-20": [row for row in rows if row["history_length"] >= 11],
    }
    metrics = {
        "coverage": {
            "G50": coverage(row["hit_g50"] for row in rows),
            "C10": coverage(row["hit_c10"] for row in rows),
            "C20": coverage(row["hit_c20"] for row in rows),
            "C50": coverage(row["hit_c50"] for row in rows),
            "U10": coverage(row["hit_u10"] for row in rows),
            "U20": coverage(row["hit_u20"] for row in rows),
            "U50": coverage(row["hit_u50"] for row in rows),
            "complementary_C50_not_G50": coverage(
                row["hit_c50"] and not row["hit_g50"] for row in rows
            ),
            "target_in_both_G50_C50": coverage(
                row["hit_c50"] and row["hit_g50"] for row in rows
            ),
            "filtered_U50": coverage(row["hit_filtered_u50"] for row in rows),
        },
        "union_size": {
            "mean": float(union_sizes.mean()),
            "median": float(np.median(union_sizes)),
            "p80": float(np.quantile(union_sizes, 0.80)),
            "p90": float(np.quantile(union_sizes, 0.90)),
            "p95": float(np.quantile(union_sizes, 0.95)),
            "fraction_le_90": float(np.mean(union_sizes <= 90)),
            "filtered_mean": float(filtered_sizes.mean()),
        },
        "source_overlap": {
            "intersection_mean": float(intersection_sizes.mean()),
            "cf_only_mean": float(cf_only_sizes.mean()),
            "jaccard_mean": float(jaccards.mean()),
            "cf_only_candidates_requiring_gram_scoring_total": int(cf_only_sizes.sum()),
        },
        "stratified": {
            **{key: summarize_stratum(value) for key, value in popularity_groups.items()},
            **{key: summarize_stratum(value) for key, value in history_groups.items()},
        },
        "oracle": {
            "U50_Hit@10_upper": coverage(row["hit_u50"] for row in rows),
            "U50_Hit@20_upper": coverage(row["hit_u50"] for row in rows),
            "U50_Hit@50_upper": coverage(row["hit_u50"] for row in rows),
        },
    }
    if full_run and not math.isclose(
        metrics["coverage"]["G50"], footer["hit@50"], abs_tol=1e-12
    ):
        raise ValueError("GRAM validation coverage identity mismatch")
    gate = scientific_gate(metrics, full_run)

    per_user_path = args.output_dir / "per_user_coverage.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "user_id", "history_length", "target_frequency", "hit_g50", "hit_c10",
            "hit_c20", "hit_c50", "hit_u10", "hit_u20", "hit_u50", "union_size",
            "intersection_size", "cf_only_size", "jaccard", "filtered_union_size",
        ])
        for row in rows:
            writer.writerow([
                row["user"], row["history_length"], row["target_frequency"],
                int(row["hit_g50"]), int(row["hit_c10"]), int(row["hit_c20"]),
                int(row["hit_c50"]), int(row["hit_u10"]), int(row["hit_u20"]),
                int(row["hit_u50"]), row["union_size"], row["intersection_size"],
                row["cf_only_size"], f"{row['jaccard']:.8f}", row["filtered_union_size"],
            ])
    summary = {
        "experiment_id": "GRAM_PHASE10_CF1_A_TOYS_CANDIDATE_UNION_V1",
        "status": "completed",
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "sports_read": False,
        "full_run": full_run,
        "users": len(rows),
        "catalog_size": len(raw_to_id),
        "popularity_frequency_boundaries": {"q1": q1, "q3": q3},
        "metrics": metrics,
        "scientific_gate": gate,
        "artifacts": {
            "per_user_coverage_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest()
        },
        "wall_time_seconds": time.time() - started,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": gate, "wall_time_seconds": summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
