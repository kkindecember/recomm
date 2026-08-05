#!/usr/bin/env python3
"""Re-decode a frozen Toys validation subset and reproduce fixed PCRF."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = REPO_ROOT / "GRAM/src"
PHASE9 = REPO_ROOT / "experiment/phase9"
PHASE10 = REPO_ROOT / "experiment/phase10"
for directory in (GRAM_SRC, PHASE9, PHASE10):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import utils  # noqa: E402,F401
from data import TestDatasetGRAM  # noqa: E402
from eval_cf0_b3_beamfusion import (  # noqa: E402
    load_cached_beams,
    load_catalog,
    load_users,
    metrics_from_ranks,
    score_item_head,
    standardize,
)
from eval_cf1_b0_score_identity import gram_args, load_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt"
DEFAULT_CACHE = REPO_ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
DEFAULT_ITEM_HEAD = REPO_ROOT / "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/p9r_toys_fresh_beam_512"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cached-predictions", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--item-head", type=Path, default=DEFAULT_ITEM_HEAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    return parser.parse_args()


def deterministic_users(user_ids, count, seed=2023):
    user_ids = list(user_ids)
    if count <= 0 or count > len(user_ids):
        raise ValueError("users must be in [1, available users]")
    return sorted(
        user_ids,
        key=lambda user: (hashlib.sha256(f"{seed}:{user}".encode()).hexdigest(), user),
    )[:count]


def rank_values(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def correlation(left, right):
    if len(left) < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(rank_values(left), rank_values(right))[0, 1]),
    }


def encoded_catalog(tokenizer, candidates):
    encoded = []
    for candidate in candidates:
        tokens = [token for token in tokenizer.encode(candidate) if token not in (1820, 9175)]
        encoded.append([0] + tokens)
    return encoded


def decode_one(model, batch, prefix_allowed_tokens, max_length, beam_size, device):
    with torch.no_grad():
        prediction = model.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            history_item_ids=batch["history_item_ids"].to(device),
            history_item_mask=batch["history_item_mask"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
    return prediction["sequences"], prediction["sequences_scores"]


def build_records(selected, beams, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1):
    id_to_lexical = {raw_to_id[item]: lexical for item, lexical in raw_to_lexical.items()}
    records = []
    for user in selected:
        sequence = users[user]
        beam = beams[user]
        target_id = sequence[-2]
        if beam["gold"] != id_to_lexical[target_id]:
            raise ValueError(f"{user}: gold mismatch")
        candidate_ids = [lexical_to_id[value] for value in beam["candidates"]]
        target_position = candidate_ids.index(target_id) if target_id in candidate_ids else -1
        records.append({
            "user": user,
            "history": sequence[max(0, len(sequence) - 22):-2],
            "candidate_ids": candidate_ids,
            "candidate_frequencies": np.asarray([frequencies[item] for item in candidate_ids], dtype=np.float64),
            "target_position": target_position,
            "seq": np.asarray(beam["seq"], dtype=np.float64),
            "q1": q1,
        })
    return records


def pcrf_details(records, item_head, weight, beta, gamma):
    score_item_head(records, item_head, 512)
    ranks, top10 = [], []
    for record in records:
        seq_z = standardize(record["seq"])
        cf_z = standardize(record["cf"])
        pop_z = standardize(np.log1p(record["candidate_frequencies"]))
        adjusted = standardize(cf_z - beta * pop_z)
        tail_mass = float(np.mean(record["candidate_frequencies"][:10] <= record["q1"]))
        reliability = (1.0 - tail_mass) ** gamma
        joint = seq_z + weight * reliability * adjusted
        order = np.argsort(-joint, kind="stable")
        top10.append([record["candidate_ids"][index] for index in order[:10]])
        position = record["target_position"]
        ranks.append(51 if position < 0 else int(np.flatnonzero(order == position)[0]) + 1)
    return np.asarray(ranks), top10


def baseline_ranks(records):
    ranks = []
    for record in records:
        order = np.argsort(-record["seq"], kind="stable")
        position = record["target_position"]
        ranks.append(51 if position < 0 else int(np.flatnonzero(order == position)[0]) + 1)
    return np.asarray(ranks)


def main():
    cli = parse_args()
    started = time.time()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    cache, _ = load_cached_beams(cli.cached_predictions)
    selected = deterministic_users(cache, cli.users, cli.seed)
    args = gram_args()
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = TestDatasetGRAM(args, "Toys", "sequential", None, tokenizer, regenerate=False, phase=0, mode="validation")
    dataset_index = {dataset.data["user_id"][index]: index for index in range(len(dataset))}
    if set(selected) - set(dataset_index):
        raise ValueError("selected users missing from validation dataset")
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="valid")
    encoded = encoded_catalog(tokenizer, dataset.all_items)
    trie = gt.Trie(encoded)
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(map(len, encoded))
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = load_model(cli.checkpoint, device)

    fresh = {}
    output_path = cli.output_dir / "fresh_beams.tsv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["idx", "gold", "pred", "scores"])
        for ordinal, user in enumerate(selected, 1):
            batch = collator([dataset[dataset_index[user]]])
            sequences, scores = decode_one(model, batch, prefix_allowed_tokens, max_length, 50, device)
            candidates = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            values = scores.detach().float().cpu().numpy()
            fresh[user] = {"gold": cache[user]["gold"], "candidates": candidates, "seq": values}
            writer.writerow([user, cache[user]["gold"], "||".join(candidates), "||".join(map(str, values.tolist()))])
            if ordinal % 16 == 0:
                handle.flush()
                print(json.dumps({"progress_users": ordinal, "total_users": len(selected), "elapsed_seconds": time.time() - started}), flush=True)

    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(REPO_ROOT / "GRAM/rec_datasets/Toys")
    users = load_users(REPO_ROOT / "GRAM/rec_datasets/Toys", raw_to_id)
    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    target_frequency_values = sorted(frequencies[sequence[-2]] for sequence in users.values())
    q1 = target_frequency_values[len(target_frequency_values) // 4]
    old_records = build_records(selected, cache, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1)
    fresh_records = build_records(selected, fresh, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1)

    legal = []
    candidate_overlaps, seq_top10_overlaps, matched_old, matched_fresh = [], [], [], []
    for user in selected:
        old, new = cache[user], fresh[user]
        legal.append(len(new["candidates"]) == 50 and len(set(new["candidates"])) == 50 and np.isfinite(new["seq"]).all() and all(value in lexical_to_id for value in new["candidates"]))
        old_map = dict(zip(old["candidates"], old["seq"]))
        new_map = dict(zip(new["candidates"], new["seq"]))
        shared = sorted(set(old_map) & set(new_map))
        candidate_overlaps.append(len(shared) / 50.0)
        matched_old.extend(old_map[value] for value in shared)
        matched_fresh.extend(new_map[value] for value in shared)
        seq_top10_overlaps.append(len(set(old["candidates"][:10]) & set(new["candidates"][:10])) / 10.0)

    old_base, fresh_base = baseline_ranks(old_records), baseline_ranks(fresh_records)
    old_pcrf, old_top10 = pcrf_details(old_records, cli.item_head, cli.lambda_weight, cli.beta, cli.gamma)
    fresh_pcrf, fresh_top10 = pcrf_details(fresh_records, cli.item_head, cli.lambda_weight, cli.beta, cli.gamma)
    pcrf_overlaps = [len(set(left) & set(right)) / 10.0 for left, right in zip(old_top10, fresh_top10)]
    corr = correlation(matched_old, matched_fresh)
    old_base_metrics, fresh_base_metrics = metrics_from_ranks(old_base), metrics_from_ranks(fresh_base)
    old_pcrf_metrics, fresh_pcrf_metrics = metrics_from_ranks(old_pcrf), metrics_from_ranks(fresh_pcrf)
    metrics = {
        "users": len(selected),
        "legal_fraction": float(np.mean(legal)),
        "mean_candidate_set_overlap": float(np.mean(candidate_overlaps)),
        "mean_sequence_top10_set_overlap": float(np.mean(seq_top10_overlaps)),
        "matched_score_pairs": len(matched_old),
        "score_pearson": corr["pearson"],
        "score_spearman": corr["spearman"],
        "mean_pcrf_top10_set_overlap": float(np.mean(pcrf_overlaps)),
        "cached_baseline": old_base_metrics,
        "fresh_baseline": fresh_base_metrics,
        "cached_pcrf": old_pcrf_metrics,
        "fresh_pcrf": fresh_pcrf_metrics,
        "baseline_Hit@10_abs_delta": abs(fresh_base_metrics["Hit@10"] - old_base_metrics["Hit@10"]),
        "pcrf_Hit@10_abs_delta": abs(fresh_pcrf_metrics["Hit@10"] - old_pcrf_metrics["Hit@10"]),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "wall_time_seconds": time.time() - started,
    }
    checks = {
        "all_512_users_legal": len(selected) == 512 and metrics["legal_fraction"] == 1.0,
        "score_pearson_at_least_0.995": math.isfinite(metrics["score_pearson"]) and metrics["score_pearson"] >= 0.995,
        "score_spearman_at_least_0.995": math.isfinite(metrics["score_spearman"]) and metrics["score_spearman"] >= 0.995,
        "pcrf_top10_overlap_at_least_0.98": metrics["mean_pcrf_top10_set_overlap"] >= 0.98,
        "baseline_Hit10_abs_delta_at_most_0.001": metrics["baseline_Hit@10_abs_delta"] <= 0.001,
        "pcrf_Hit10_abs_delta_at_most_0.001": metrics["pcrf_Hit@10_abs_delta"] <= 0.001,
    }
    gate = {"status": "passed" if all(checks.values()) else "failed_reproducibility_gate", "checks": checks}
    per_user_path = cli.output_dir / "per_user.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "candidate_overlap", "sequence_top10_overlap", "pcrf_top10_overlap", "cached_baseline_rank", "fresh_baseline_rank", "cached_pcrf_rank", "fresh_pcrf_rank"])
        for index, user in enumerate(selected):
            writer.writerow([user, candidate_overlaps[index], seq_top10_overlaps[index], pcrf_overlaps[index], old_base[index], fresh_base[index], old_pcrf[index], fresh_pcrf[index]])
    summary = {
        "experiment_id": "GRAM_PHASE9_P9R_TOYS_FRESH_BEAM_512_V1",
        "status": "completed",
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "sample_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "frozen_pcrf": {"lambda": cli.lambda_weight, "beta": cli.beta, "gamma": cli.gamma},
        "metrics": metrics,
        "scientific_gate": gate,
        "artifacts": {
            "fresh_beams_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "per_user_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
            "item_head_sha256": hashlib.sha256(cli.item_head.read_bytes()).hexdigest(),
        },
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": gate, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
