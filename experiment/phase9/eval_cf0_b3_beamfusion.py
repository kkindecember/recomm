#!/usr/bin/env python3
"""Evaluate frozen GRAM/item-head BeamFusion on cached legal beams."""

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


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from train_cf0_b2_item_head import CF0B2ItemHead  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "GRAM/rec_datasets/Toys"
DEFAULT_PREDICTIONS = (
    REPO_ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/cf0_b3_toys_beamfusion_p2c"
DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-size", type=int, default=4096)
    parser.add_argument("--partition-seed", default="2023")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    return parser.parse_args()


def normalize_lexical_id(raw):
    return raw.replace("|▁", " ").replace("|", "").strip()


def load_catalog(data_dir):
    item_path = data_dir / "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
    raw_to_lexical = {}
    lexical_to_raw = {}
    with item_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw_item, lexical = line.rstrip("\n").split(" ", 1)
            normalized = normalize_lexical_id(lexical)
            if normalized in lexical_to_raw:
                raise ValueError(f"duplicate normalized lexical ID: {normalized}")
            raw_to_lexical[raw_item] = normalized
            lexical_to_raw[normalized] = raw_item
    sorted_items = sorted(raw_to_lexical)
    raw_to_id = {item: index + 1 for index, item in enumerate(sorted_items)}
    lexical_to_id = {
        lexical: raw_to_id[raw_item] for lexical, raw_item in lexical_to_raw.items()
    }
    return raw_to_lexical, raw_to_id, lexical_to_id


def load_users(data_dir, raw_to_id):
    users = {}
    with (data_dir / "user_sequence.txt").open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields[0] in users:
                raise ValueError(f"duplicate user: {fields[0]}")
            users[fields[0]] = [raw_to_id[item] for item in fields[1:]]
    return users


def load_cached_beams(path):
    rows = {}
    footer = {}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n")
        if not header.startswith("idx\t"):
            raise ValueError("unexpected prediction header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1:
                key, value = fields[0].split(":", 1)
                footer[key] = float(value.strip())
                continue
            if len(fields) < 4:
                raise ValueError("malformed cached prediction row")
            user, gold, prediction_text, score_text = fields[0], fields[-3], fields[-2], fields[-1]
            if user in rows:
                raise ValueError(f"duplicate prediction user: {user}")
            candidates = prediction_text.split("||")
            scores = np.asarray([float(value) for value in score_text.split("||")])
            if len(candidates) != 50 or scores.shape != (50,):
                raise ValueError(f"{user}: expected 50 candidates/scores")
            if len(set(candidates)) != 50 or not np.isfinite(scores).all():
                raise ValueError(f"{user}: duplicate candidate or non-finite score")
            rows[user] = {"gold": gold, "candidates": candidates, "seq": scores}
    return rows, footer


def make_partition(user_ids, calibration_size, partition_seed):
    ordered = sorted(
        user_ids,
        key=lambda user: (
            hashlib.sha256(f"{partition_seed}:{user}".encode()).hexdigest(),
            user,
        ),
    )
    return set(ordered[:calibration_size]), ordered


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    result = {
        "count": int(ranks.size),
        "mrr": float(np.mean(np.where(ranks <= 50, 1.0 / ranks, 0.0))),
    }
    for cutoff in (1, 5, 10, 20, 50):
        hit = ranks <= cutoff
        result[f"Hit@{cutoff}"] = float(hit.mean())
        result[f"NDCG@{cutoff}"] = float(
            np.mean(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0))
        )
    return result


def ranks_for_lambda(records, indices, weight):
    ranks = []
    for index in indices:
        record = records[index]
        seq = record["seq_z"]
        cf = record["cf_z"]
        joint = seq + weight * cf
        order = np.argsort(-joint, kind="stable")
        target_position = record["target_position"]
        if target_position < 0:
            ranks.append(51)
        else:
            ranks.append(int(np.flatnonzero(order == target_position)[0]) + 1)
    return np.asarray(ranks, dtype=np.int64)


def bootstrap_hit10_delta(baseline_ranks, fused_ranks, replicates, seed):
    baseline = (np.asarray(baseline_ranks) <= 10).astype(np.float64)
    fused = (np.asarray(fused_ranks) <= 10).astype(np.float64)
    paired = fused - baseline
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, paired.size, paired.size)
        values[index] = paired[sample].mean()
    low, high = np.quantile(values, [0.025, 0.975])
    return {"replicates": replicates, "lower": float(low), "upper": float(high)}


def subgroup_metrics(records, indices, ranks, q1, q3):
    groups = {
        "history_1-5": [],
        "history_6-10": [],
        "history_11-20": [],
        "target_tail": [],
        "target_middle": [],
        "target_head": [],
    }
    for index, rank in zip(indices, ranks):
        record = records[index]
        length = record["history_length"]
        frequency = record["target_frequency"]
        groups["history_1-5" if length <= 5 else "history_6-10" if length <= 10 else "history_11-20"].append(rank)
        groups["target_tail" if frequency <= q1 else "target_middle" if frequency < q3 else "target_head"].append(rank)
    return {key: metrics_from_ranks(value) for key, value in groups.items() if value}


def score_item_head(records, checkpoint_path, batch_size):
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
    items = torch.nn.functional.normalize(model.item_embedding.weight[1:], dim=-1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            histories = torch.zeros(len(batch), config["max_history"], dtype=torch.long)
            masks = torch.zeros_like(histories, dtype=torch.bool)
            for row, record in enumerate(batch):
                history = record["history"][-config["max_history"] :]
                histories[row, : len(history)] = torch.tensor(history)
                masks[row, : len(history)] = True
            users = model.encode(histories, masks)
            candidate_ids = torch.tensor([record["candidate_ids"] for record in batch]) - 1
            candidate_vectors = items[candidate_ids]
            scores = scale * torch.einsum("bd,bkd->bk", users, candidate_vectors)
            for record, values in zip(batch, scores.numpy()):
                record["cf"] = values.astype(np.float64)
    return config


def standardize(values):
    std = float(np.std(values))
    return (values - np.mean(values)) / max(std, 1e-6)


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(args.data_dir)
    id_to_lexical = {
        raw_to_id[raw_item]: lexical for raw_item, lexical in raw_to_lexical.items()
    }
    users = load_users(args.data_dir, raw_to_id)
    cache, footer = load_cached_beams(args.predictions)
    if set(users) != set(cache):
        raise ValueError("prediction/data user sets differ")
    if len(users) != 19412 or len(raw_to_id) != 11924:
        raise ValueError("unexpected user or catalog size")

    train_frequencies = Counter()
    records = []
    for sequence in users.values():
        train_frequencies.update(sequence[:-2])
    frequency_values = sorted(train_frequencies[sequence[-2]] for sequence in users.values())
    q1, q3 = frequency_values[len(frequency_values) // 4], frequency_values[3 * len(frequency_values) // 4]

    for user, sequence in users.items():
        cached = cache[user]
        target_id = sequence[-2]
        target_lexical = id_to_lexical[target_id]
        if cached["gold"] != target_lexical:
            raise ValueError(f"{user}: cached/current gold mismatch")
        try:
            candidate_ids = [lexical_to_id[value] for value in cached["candidates"]]
        except KeyError as error:
            raise ValueError(f"{user}: unmapped legal candidate {error}") from error
        target_position = candidate_ids.index(target_id) if target_id in candidate_ids else -1
        records.append(
            {
                "user": user,
                "history": sequence[max(0, len(sequence) - 2 - 20) : -2],
                "history_length": min(len(sequence) - 2, 20),
                "target_id": target_id,
                "target_frequency": train_frequencies[target_id],
                "candidate_ids": candidate_ids,
                "target_position": target_position,
                "seq": cached["seq"],
            }
        )

    config = score_item_head(records, args.checkpoint, args.batch_size)
    if config["num_items"] != len(raw_to_id):
        raise ValueError("checkpoint/catalog size mismatch")
    for record in records:
        record["seq_z"] = standardize(record["seq"])
        record["cf_z"] = standardize(record["cf"])

    calibration_users, partition_order = make_partition(users, args.calibration_size, args.partition_seed)
    calibration_indices = [i for i, record in enumerate(records) if record["user"] in calibration_users]
    evaluation_indices = [i for i, record in enumerate(records) if record["user"] not in calibration_users]
    position = {user: index for index, user in enumerate(partition_order)}
    partition_path = args.output_dir / "partition.tsv"
    with partition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "split", "hash_order"])
        for record in records:
            user = record["user"]
            writer.writerow([user, "calibration" if user in calibration_users else "evaluation", position[user]])

    full_baseline = ranks_for_lambda(records, range(len(records)), 0.0)
    full_metrics = metrics_from_ranks(full_baseline)
    for key in ("hit@5", "hit@10", "hit@20", "hit@50", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50"):
        metric_key = key.replace("hit", "Hit").replace("ndcg", "NDCG")
        if not math.isclose(full_metrics[metric_key], footer[key], abs_tol=1e-12):
            raise ValueError(f"baseline identity mismatch for {key}")

    calibration_grid = []
    calibration_ranks = {}
    for weight in args.lambdas:
        ranks = ranks_for_lambda(records, calibration_indices, weight)
        calibration_ranks[weight] = ranks
        calibration_grid.append({"lambda": weight, "metrics": metrics_from_ranks(ranks)})
    selected = min(
        calibration_grid,
        key=lambda row: (-row["metrics"]["Hit@10"], -row["metrics"]["NDCG@10"], row["lambda"]),
    )
    baseline_calibration = next(row for row in calibration_grid if row["lambda"] == 0.0)
    calibration_delta = {
        "Hit@10": selected["metrics"]["Hit@10"] - baseline_calibration["metrics"]["Hit@10"],
        "NDCG@10": selected["metrics"]["NDCG@10"] - baseline_calibration["metrics"]["NDCG@10"],
    }
    calibration_checks = {
        "lambda_positive": selected["lambda"] > 0,
        "Hit@10_delta_at_least_0.002": calibration_delta["Hit@10"] >= 0.002,
        "NDCG@10_non_degradation": calibration_delta["NDCG@10"] >= 0,
    }
    calibration_passed = all(calibration_checks.values())

    summary = {
        "experiment_id": "GRAM_PHASE9_CF0_B3_TOYS_BEAMFUSION_P2C_V1",
        "status": "completed",
        "dataset": "Toys",
        "split": "validation_calibration_plus_holdout",
        "test_read": False,
        "sports_read": False,
        "integrity_gate": {
            "status": "passed",
            "users": len(users),
            "catalog_size": len(raw_to_id),
            "beams_per_user": 50,
            "footer_metrics": len(footer),
            "baseline_identity": full_metrics,
            "target_in_beam_oracle_Hit@50": full_metrics["Hit@50"],
        },
        "partition": {"calibration": len(calibration_indices), "evaluation": len(evaluation_indices), "seed": args.partition_seed},
        "calibration": {
            "grid": calibration_grid,
            "selected_lambda": selected["lambda"],
            "baseline": baseline_calibration["metrics"],
            "selected": selected["metrics"],
            "delta": calibration_delta,
            "checks": calibration_checks,
            "status": "passed" if calibration_passed else "failed",
        },
        "evaluation": {"status": "not_run_calibration_gate_failed"},
        "scientific_gate": {"status": "failed_calibration_gate"},
        "popularity_frequency_boundaries": {"q1": q1, "q3": q3},
    }

    selected_ranks_all = np.full(len(records), -1, dtype=np.int64)
    for index, rank in zip(calibration_indices, calibration_ranks[selected["lambda"]]):
        selected_ranks_all[index] = rank
    if calibration_passed:
        baseline_eval = ranks_for_lambda(records, evaluation_indices, 0.0)
        fused_eval = ranks_for_lambda(records, evaluation_indices, selected["lambda"])
        pure_cf_eval = []
        for index in evaluation_indices:
            record = records[index]
            order = np.argsort(-record["cf"], kind="stable")
            pure_cf_eval.append(51 if record["target_position"] < 0 else int(np.flatnonzero(order == record["target_position"])[0]) + 1)
        baseline_metrics = metrics_from_ranks(baseline_eval)
        fused_metrics = metrics_from_ranks(fused_eval)
        baseline_groups = subgroup_metrics(records, evaluation_indices, baseline_eval, q1, q3)
        fused_groups = subgroup_metrics(records, evaluation_indices, fused_eval, q1, q3)
        delta = {key: fused_metrics[key] - baseline_metrics[key] for key in fused_metrics if key != "count"}
        ci = bootstrap_hit10_delta(baseline_eval, fused_eval, args.bootstrap_replicates, args.seed)
        checks = {
            "Hit@10_delta_at_least_0.002": delta["Hit@10"] >= 0.002,
            "Hit@10_bootstrap_lower_positive": ci["lower"] > 0,
            "NDCG@10_non_degradation": delta["NDCG@10"] >= 0,
            "tail_Hit@10_non_degradation": fused_groups["target_tail"]["Hit@10"] >= baseline_groups["target_tail"]["Hit@10"],
            "Hit@50_identity": math.isclose(fused_metrics["Hit@50"], baseline_metrics["Hit@50"], abs_tol=1e-12),
        }
        gate_status = "passed" if all(checks.values()) else "failed_holdout_gate"
        summary["evaluation"] = {
            "status": "completed",
            "baseline": baseline_metrics,
            "fused": fused_metrics,
            "delta": delta,
            "pure_cf_within_beam": metrics_from_ranks(pure_cf_eval),
            "baseline_subgroups": baseline_groups,
            "fused_subgroups": fused_groups,
            "Hit@10_paired_bootstrap_95ci": ci,
            "checks": checks,
        }
        summary["scientific_gate"] = {"status": gate_status}
        for index, rank in zip(evaluation_indices, fused_eval):
            selected_ranks_all[index] = rank

    per_user_path = args.output_dir / "per_user.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "split", "history_length", "target_frequency", "baseline_rank", "fused_rank"])
        baseline_lookup = ranks_for_lambda(records, range(len(records)), 0.0)
        for index, record in enumerate(records):
            writer.writerow([
                record["user"],
                "calibration" if record["user"] in calibration_users else "evaluation",
                record["history_length"],
                record["target_frequency"],
                baseline_lookup[index],
                selected_ranks_all[index] if selected_ranks_all[index] > 0 else "WITHHELD",
            ])

    summary["artifacts"] = {
        "partition_sha256": hashlib.sha256(partition_path.read_bytes()).hexdigest(),
        "per_user_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest(),
    }
    summary["wall_time_seconds"] = time.time() - started
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": summary["scientific_gate"], "selected_lambda": selected["lambda"], "wall_time_seconds": summary["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
