#!/usr/bin/env python3
"""Build and audit the frozen CF1-C candidate feature matrix without fitting."""

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
PHASE10 = REPO_ROOT / "experiment/phase10"
for directory in (PHASE9, PHASE10):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_cf0_b3_beamfusion import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_PREDICTIONS,
    load_cached_beams,
    load_catalog,
    load_users,
    metrics_from_ranks,
    standardize,
)
from eval_cf1_a_candidate_union import load_item_model  # noqa: E402
from eval_cf1_a2_budgeted_union import fill_cf_only  # noqa: E402
from eval_cf1_b0_score_identity import deterministic_users  # noqa: E402


DEFAULT_ITEM_CHECKPOINT = REPO_ROOT / "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
DEFAULT_B2 = REPO_ROOT / "artifacts/phase10/cf1_b2_toys_full_scores/candidate_scores.tsv"
DEFAULT_B2_SUMMARY = REPO_ROOT / "artifacts/phase10/cf1_b2_toys_full_scores/summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_c0_toys_feature_audit"
EXPECTED_USERS = 19412
EXPECTED_CANDIDATES = 1698905
EXPECTED_CF_ONLY = 728305
FROZEN_PCRF = (1.0, 0.5, 1.0)
FROZEN_TAIL_THRESHOLD = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--item-checkpoint", type=Path, default=DEFAULT_ITEM_CHECKPOINT)
    parser.add_argument("--b2-scores", type=Path, default=DEFAULT_B2)
    parser.add_argument("--b2-summary", type=Path, default=DEFAULT_B2_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-users", type=int, default=0)
    return parser.parse_args()


def make_folds(user_ids, num_folds=5, seed="2023"):
    ordered = sorted(
        user_ids,
        key=lambda user: (hashlib.sha256(f"P10-C1:{seed}:{user}".encode()).hexdigest(), user),
    )
    return {user: index % num_folds for index, user in enumerate(ordered)}


def source_name(candidate, gram_set, cf_set):
    if candidate in gram_set and candidate in cf_set:
        return "both"
    if candidate in gram_set:
        return "gram"
    return "cf_only"


def target_rank(scores, target_position, missing_rank=91):
    if target_position < 0:
        return missing_rank
    order = np.argsort(-np.asarray(scores), kind="stable")
    return int(np.flatnonzero(order == target_position)[0]) + 1


def pcrf_scores(seq, item, frequency, tail_mass, params=FROZEN_PCRF):
    weight, beta, gamma = params
    seq_z = standardize(np.asarray(seq, dtype=np.float64))
    item_z = standardize(np.asarray(item, dtype=np.float64))
    pop_z = standardize(np.log1p(np.asarray(frequency, dtype=np.float64)))
    adjusted = standardize(item_z - beta * pop_z)
    reliability = (1.0 - float(tail_mass)) ** gamma
    return seq_z + weight * reliability * adjusted


def scientific_gate(metrics, full_run):
    if not full_run:
        return {"status": "not_evaluated_smoke", "checks": {}}
    checks = {
        "users_exact_19412": metrics["users"] == EXPECTED_USERS,
        "candidates_exact_1698905": metrics["total_candidates"] == EXPECTED_CANDIDATES,
        "cf_only_exact_728305": metrics["source_counts"]["cf_only"] == EXPECTED_CF_ONLY,
        "all_union_sizes_50_to_90": metrics["valid_budget_fraction"] == 1.0,
        "no_duplicate_user_candidate": metrics["duplicate_pairs"] == 0,
        "all_features_finite": metrics["finite_fraction"] == 1.0,
        "b2_rows_exact": metrics["b2_rows_matched_fraction"] == 1.0,
        "b2_sha256_identity": metrics["b2_sha256_identity"],
        "cached_G50_footer_identity": metrics["cached_G50_footer_identity"],
        "folds_cover_all_users_once": metrics["fold_integrity"],
        "target_excluded_from_inference_schema": metrics["target_excluded_from_inference_schema"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed_feature_identity_gate",
        "checks": checks,
    }


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(args.data_dir)
    users, cache_footer = load_users(args.data_dir, raw_to_id), load_cached_beams(args.predictions)
    cache, footer = cache_footer
    if set(users) != set(cache) or len(users) != EXPECTED_USERS or len(raw_to_id) != 11924:
        raise ValueError("unexpected validation data/cache identity")
    selected = deterministic_users(cache, args.max_users or EXPECTED_USERS)
    full_run = len(selected) == EXPECTED_USERS
    id_to_lexical = {raw_to_id[raw]: lexical for raw, lexical in raw_to_lexical.items()}

    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])

    records = []
    for user in selected:
        sequence = users[user]
        gram_lexical = cache[user]["candidates"]
        gram_ids = [lexical_to_id[value] for value in gram_lexical]
        records.append({
            "user": user,
            "history": sequence[max(0, len(sequence) - 22) : -2],
            "history_length": min(len(sequence) - 2, 20),
            "target": sequence[-2],
            "target_frequency": frequencies[sequence[-2]],
            "gram_lexical": gram_lexical,
            "gram_ids": gram_ids,
            "cached_seq": np.asarray(cache[user]["seq"], dtype=np.float64),
        })

    model, config = load_item_model(args.item_checkpoint)
    if config["num_items"] != len(raw_to_id):
        raise ValueError("item checkpoint/catalog mismatch")
    item_vectors = F.normalize(model.item_embedding.weight[1:], dim=-1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            history_ids = torch.zeros(len(batch), config["max_history"], dtype=torch.long)
            history_mask = torch.zeros_like(history_ids, dtype=torch.bool)
            for row, record in enumerate(batch):
                history = record["history"][-config["max_history"] :]
                history_ids[row, : len(history)] = torch.tensor(history)
                history_mask[row, : len(history)] = True
            encoded = model.encode(history_ids, history_mask)
            logits = scale * F.linear(encoded, item_vectors)
            cf_top50 = torch.topk(logits, k=50, dim=1, sorted=True).indices + 1
            for row, record in enumerate(batch):
                cf_ids = cf_top50[row].tolist()
                cf_lexical = [id_to_lexical[item] for item in cf_ids]
                union_lexical = fill_cf_only(record["gram_lexical"], cf_lexical, 40)
                union_ids = [lexical_to_id[value] for value in union_lexical]
                record["cf_ids"] = cf_ids
                record["union_lexical"] = union_lexical
                record["union_ids"] = union_ids
                record["item_scores"] = logits[row, torch.tensor(union_ids) - 1].numpy().astype(np.float64)
    del model

    expected_b2_hash = json.load(args.b2_summary.open())["artifacts"]["candidate_scores_sha256"]
    actual_b2_hash = hashlib.sha256(args.b2_scores.read_bytes()).hexdigest()
    b2_rows = 0
    duplicate_pairs = 0
    with args.b2_scores.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["user_id", "union_rank", "candidate", "source", "gram_score"]:
            raise ValueError("unexpected B2 schema")
        for record in records:
            gram_set, cf_set = set(record["gram_lexical"]), {id_to_lexical[item] for item in record["cf_ids"]}
            scores, seen = [], set()
            for rank, candidate in enumerate(record["union_lexical"], 1):
                row = next(reader, None)
                if row is None:
                    raise ValueError("B2 ended before reconstructed union")
                expected_source = source_name(candidate, gram_set, cf_set)
                if row["user_id"] != record["user"] or int(row["union_rank"]) != rank:
                    raise ValueError("B2 user/rank order mismatch")
                if row["candidate"] != candidate or row["source"] != expected_source:
                    raise ValueError("B2 candidate/source identity mismatch")
                if candidate in seen:
                    duplicate_pairs += 1
                seen.add(candidate)
                scores.append(float(row["gram_score"]))
                b2_rows += 1
            record["gram_scores"] = np.asarray(scores, dtype=np.float64)
        if full_run and next(reader, None) is not None:
            raise ValueError("B2 contains extra rows")

    total = sum(len(record["union_ids"]) for record in records)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(record["union_ids"]) for record in records])
    arrays = {
        "candidate_item_id": np.empty(total, dtype=np.int32),
        "source": np.empty(total, dtype=np.uint8),
        "gram_score": np.empty(total, dtype=np.float32),
        "gram_z": np.empty(total, dtype=np.float32),
        "item_score": np.empty(total, dtype=np.float32),
        "item_z": np.empty(total, dtype=np.float32),
        "corrected_item_z": np.empty(total, dtype=np.float32),
        "gram_rr": np.empty(total, dtype=np.float32),
        "cf_rr": np.empty(total, dtype=np.float32),
        "agreement": np.empty(total, dtype=np.float32),
        "item_log_frequency": np.empty(total, dtype=np.float32),
        "gold_label": np.empty(total, dtype=np.uint8),
    }
    user_history = np.empty(len(records), dtype=np.int16)
    user_target_frequency = np.empty(len(records), dtype=np.int32)
    user_tail_mass = np.empty(len(records), dtype=np.float32)
    user_reliability = np.empty(len(records), dtype=np.float32)
    user_target_position = np.empty(len(records), dtype=np.int16)
    gram_ranks, pcrf_ranks, cf_ranks, sum_ranks, oracle_ranks = [], [], [], [], []
    source_counts = {"gram": 0, "both": 0, "cf_only": 0}
    finite_count = 0

    for index, record in enumerate(records):
        left, right = offsets[index], offsets[index + 1]
        ids = np.asarray(record["union_ids"], dtype=np.int32)
        gram = record["gram_scores"]
        item = record["item_scores"]
        logfreq = np.log1p([frequencies[int(item_id)] for item_id in ids])
        gram_z, item_z, pop_z = standardize(gram), standardize(item), standardize(logfreq)
        corrected = standardize(item_z - 0.5 * pop_z)
        gram_rank = {item_id: rank for rank, item_id in enumerate(record["gram_ids"], 1)}
        cf_rank = {item_id: rank for rank, item_id in enumerate(record["cf_ids"], 1)}
        sources = np.asarray([1 if item_id in gram_rank and item_id in cf_rank else 0 if item_id in gram_rank else 2 for item_id in ids], dtype=np.uint8)
        grr = np.asarray([1.0 / gram_rank[item_id] if item_id in gram_rank else 0.0 for item_id in ids])
        crr = np.asarray([1.0 / cf_rank[item_id] if item_id in cf_rank else 0.0 for item_id in ids])
        agreement = np.asarray([1.0 / (abs(gram_rank[item_id] - cf_rank[item_id]) + 1.0) if item_id in gram_rank and item_id in cf_rank else 0.0 for item_id in ids])
        target_position = int(np.flatnonzero(ids == record["target"])[0]) if record["target"] in ids else -1
        labels = (ids == record["target"]).astype(np.uint8)
        tail_mass = float(np.mean(np.asarray([frequencies[item] for item in record["gram_ids"][:10]]) <= FROZEN_TAIL_THRESHOLD))

        values = [gram, gram_z, item, item_z, corrected, grr, crr, agreement, logfreq]
        finite_count += sum(int(np.isfinite(value).sum()) for value in values)
        arrays["candidate_item_id"][left:right] = ids
        arrays["source"][left:right] = sources
        arrays["gram_score"][left:right] = gram
        arrays["gram_z"][left:right] = gram_z
        arrays["item_score"][left:right] = item
        arrays["item_z"][left:right] = item_z
        arrays["corrected_item_z"][left:right] = corrected
        arrays["gram_rr"][left:right] = grr
        arrays["cf_rr"][left:right] = crr
        arrays["agreement"][left:right] = agreement
        arrays["item_log_frequency"][left:right] = logfreq
        arrays["gold_label"][left:right] = labels
        user_history[index] = record["history_length"]
        user_target_frequency[index] = record["target_frequency"]
        user_tail_mass[index] = tail_mass
        user_reliability[index] = 1.0 - tail_mass
        user_target_position[index] = target_position
        for code, name in ((0, "gram"), (1, "both"), (2, "cf_only")):
            source_counts[name] += int(np.sum(sources == code))

        gpos = record["gram_ids"].index(record["target"]) if record["target"] in record["gram_ids"] else -1
        gram_ranks.append(target_rank(record["cached_seq"], gpos, 51))
        pcrf = pcrf_scores(record["cached_seq"], item[:50], [frequencies[x] for x in record["gram_ids"]], tail_mass)
        pcrf_ranks.append(target_rank(pcrf, gpos, 51))
        cf_ranks.append(cf_rank.get(record["target"], 51))
        sum_ranks.append(target_rank(gram_z + item_z, target_position, 91))
        oracle_ranks.append(1 if target_position >= 0 else 91)

    folds = make_folds(selected)
    fold_array = np.asarray([folds[user] for user in selected], dtype=np.uint8)
    fold_counts = np.bincount(fold_array, minlength=5)
    inference_features = [
        "gram_z", "corrected_item_z", "source", "gram_rr", "cf_rr", "agreement",
        "item_log_frequency", "user_tail_mass", "user_reliability", "user_history_length",
    ]
    feature_path = args.output_dir / "feature_table.npz"
    np.savez_compressed(
        feature_path,
        users=np.asarray(selected), offsets=offsets, fold=fold_array,
        user_history_length=user_history, user_target_frequency=user_target_frequency,
        user_tail_mass=user_tail_mass, user_reliability=user_reliability,
        user_target_position=user_target_position, **arrays,
    )
    fold_path = args.output_dir / "fold_assignments.tsv"
    with fold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "fold"])
        writer.writerows((user, folds[user]) for user in selected)

    gram_metrics = metrics_from_ranks(gram_ranks)
    cached_identity = all(
        math.isclose(gram_metrics[key.replace("hit", "Hit").replace("ndcg", "NDCG")], value, abs_tol=1e-12)
        for key, value in footer.items() if key in {"hit@1", "hit@5", "hit@10", "hit@20", "hit@50", "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50"}
    )
    metrics = {
        "users": len(records),
        "total_candidates": total,
        "source_counts": source_counts,
        "union_size_mean": float(np.mean(np.diff(offsets))),
        "union_size_max": int(np.max(np.diff(offsets))),
        "valid_budget_fraction": float(np.mean((np.diff(offsets) >= 50) & (np.diff(offsets) <= 90))),
        "duplicate_pairs": duplicate_pairs,
        "finite_fraction": finite_count / (total * 9),
        "b2_rows_matched_fraction": b2_rows / total,
        "b2_sha256_identity": actual_b2_hash == expected_b2_hash,
        "cached_G50_footer_identity": cached_identity,
        "fold_counts": fold_counts.tolist(),
        "fold_integrity": bool(int(fold_counts.sum()) == len(records) and np.all(fold_counts > 0)),
        "target_excluded_from_inference_schema": not any("target" in name or "gold" in name or "label" in name for name in inference_features),
        "baselines": {
            "GRAM_G50": gram_metrics,
            "frozen_PCRF_1.0_0.5_1.0": metrics_from_ranks(pcrf_ranks),
            "pure_CF50": metrics_from_ranks(cf_ranks),
            "source_agnostic_sum_union": metrics_from_ranks(sum_ranks),
            "union_oracle": metrics_from_ranks(oracle_ranks),
        },
    }
    gate = scientific_gate(metrics, full_run)
    summary = {
        "experiment_id": "GRAM_PHASE10_CF1_C0_TOYS_FEATURE_AUDIT_V1",
        "status": "completed",
        "dataset": "Toys", "split": "validation", "test_read": False,
        "beauty_read": False, "sports_read": False,
        "primary_policy": "fill_cf_only_40",
        "frozen_PCRF": {"lambda": 1.0, "beta": 0.5, "gamma": 1.0, "tail_frequency_threshold": 5},
        "inference_feature_schema": inference_features,
        "metrics": metrics,
        "scientific_gate": gate,
        "artifacts": {
            "b2_scores_sha256": actual_b2_hash,
            "feature_table_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "fold_assignments_sha256": hashlib.sha256(fold_path.read_bytes()).hexdigest(),
        },
        "wall_time_seconds": time.time() - started,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": gate, "metrics": metrics, "wall_time_seconds": summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
