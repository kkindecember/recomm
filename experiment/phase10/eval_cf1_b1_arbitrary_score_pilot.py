#!/usr/bin/env python3
"""Resource and legality pilot for arbitrary budgeted-union GRAM path scoring."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE10 = REPO_ROOT / "experiment/phase10"
PHASE9 = REPO_ROOT / "experiment/phase9"
for directory in (PHASE10, PHASE9):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_cf1_a_candidate_union import (  # noqa: E402
    load_item_model,
    retrieve_cf_top50,
)
from eval_cf1_a2_budgeted_union import fill_cf_only  # noqa: E402
from eval_cf1_b0_score_identity import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_PREDICTIONS,
    correlation,
    deterministic_users,
    gram_args,
    load_model as load_gram_model,
    raw_lexical_map,
    score_user,
)
from eval_cf0_b3_beamfusion import load_cached_beams, load_catalog, load_users  # noqa: E402
from data import TestDatasetGRAM  # noqa: E402
from processor import CollatorGRAM  # noqa: E402


DEFAULT_ITEM_CHECKPOINT = REPO_ROOT / "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_b1_toys_arbitrary_score_pilot"
FULL_VALIDATION_USERS = 19412


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gram-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--item-checkpoint", type=Path, default=DEFAULT_ITEM_CHECKPOINT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=512)
    parser.add_argument("--candidate-batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def scientific_gate(metrics):
    checks = {
        "all_users_valid_budget": metrics["valid_budget_fraction"] == 1.0,
        "all_paths_legal": metrics["legal_path_fraction"] == 1.0,
        "all_scores_finite": metrics["finite_fraction"] == 1.0,
        "G50_pearson_at_least_0.995": metrics["G50_pearson"] >= 0.995,
        "G50_spearman_at_least_0.995": metrics["G50_spearman"] >= 0.995,
        "G50_top10_overlap_at_least_0.98": metrics["G50_mean_top10_set_overlap"] >= 0.98,
        "G50_hit10_identity_within_0.001": abs(metrics["G50_recomputed_Hit@10"] - metrics["G50_cached_Hit@10"]) <= 0.001,
        "peak_allocated_mib_at_most_12000": metrics["peak_allocated_mib"] <= 12000,
        "wall_time_at_most_600": metrics["wall_time_seconds"] <= 600,
        "projected_full_hours_at_most_4": metrics["projected_full_validation_hours"] <= 4.0,
    }
    return {"status": "passed" if all(checks.values()) else "failed_arbitrary_score_resource_gate", "checks": checks}


def main():
    cli = parse_args()
    started = time.time()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    cache, _ = load_cached_beams(cli.predictions)
    selected = deterministic_users(cache, cli.users)
    raw_to_lexical, raw_to_id, _ = load_catalog(REPO_ROOT / "GRAM/rec_datasets/Toys")
    users = load_users(REPO_ROOT / "GRAM/rec_datasets/Toys", raw_to_id)
    id_to_lexical = {raw_to_id[raw]: lexical for raw, lexical in raw_to_lexical.items()}
    histories = [users[user][max(0, len(users[user]) - 22) : -2] for user in selected]

    item_model, item_config = load_item_model(cli.item_checkpoint)
    cf_top50 = retrieve_cf_top50(item_model, item_config, histories, 256)
    del item_model

    args = gram_args()
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = TestDatasetGRAM(args, "Toys", "sequential", None, tokenizer, regenerate=False, phase=0, mode="validation")
    dataset_index = {dataset.data["user_id"][idx]: idx for idx in range(len(dataset))}
    if set(selected) - set(dataset_index):
        raise ValueError("pilot users missing from validation dataset")
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="test")
    raw_lexical = raw_lexical_map()
    gram_model = load_gram_model(cli.gram_checkpoint, device)

    output_rows = []
    cached_g50, recomputed_g50, overlaps = [], [], []
    cached_hits, recomputed_hits = [], []
    union_sizes, cf_only_counts = [], []
    legal_count = finite_count = total_count = 0

    for user, cf_ids in zip(selected, cf_top50):
        cached = cache[user]
        gram_ids = cached["candidates"]
        cf_lexical = [id_to_lexical[item_id] for item_id in cf_ids]
        union = fill_cf_only(gram_ids, cf_lexical, 40)
        union_sizes.append(len(union))
        cf_only_counts.append(len(union) - 50)
        raw_candidates = []
        for candidate in union:
            if candidate not in raw_lexical:
                raise ValueError(f"illegal lexical path: {candidate}")
            raw_candidates.append(raw_lexical[candidate])
            legal_count += 1
        batch = collator([dataset[dataset_index[user]]])
        scores = np.asarray(score_user(gram_model, batch, raw_candidates, collator, device, cli.candidate_batch_size))
        finite_count += int(np.isfinite(scores).sum())
        total_count += len(scores)

        old = np.asarray(cached["seq"])
        new_g50 = scores[:50]
        cached_order = np.argsort(-old, kind="stable")
        new_order = np.argsort(-new_g50, kind="stable")
        overlaps.append(len(set(cached_order[:10]) & set(new_order[:10])) / 10.0)
        target_position = gram_ids.index(cached["gold"]) if cached["gold"] in gram_ids else -1
        cached_rank = int(np.flatnonzero(cached_order == target_position)[0]) + 1 if target_position >= 0 else 51
        new_rank = int(np.flatnonzero(new_order == target_position)[0]) + 1 if target_position >= 0 else 51
        cached_hits.append(cached_rank <= 10)
        recomputed_hits.append(new_rank <= 10)
        cached_g50.extend(old.tolist())
        recomputed_g50.extend(new_g50.tolist())

        gram_set = set(gram_ids)
        cf_set = set(cf_lexical)
        for rank, (candidate, score) in enumerate(zip(union, scores), 1):
            source = "both" if candidate in gram_set and candidate in cf_set else "gram" if candidate in gram_set else "cf_only"
            output_rows.append((user, rank, candidate, source, score))

    elapsed = time.time() - started
    corr = correlation(cached_g50, recomputed_g50)
    rate = total_count / elapsed
    projected_candidates = total_count / len(selected) * FULL_VALIDATION_USERS
    metrics = {
        "users": len(selected),
        "total_candidates": total_count,
        "G50_pairs": len(cached_g50),
        "cf_only_candidates": int(sum(cf_only_counts)),
        "union_size_mean": float(np.mean(union_sizes)),
        "union_size_max": int(max(union_sizes)),
        "valid_budget_fraction": float(np.mean([(50 <= size <= 90) for size in union_sizes])),
        "legal_path_fraction": legal_count / total_count,
        "finite_fraction": finite_count / total_count,
        "G50_pearson": corr["pearson"],
        "G50_spearman": corr["spearman"],
        "G50_mean_top10_set_overlap": float(np.mean(overlaps)),
        "G50_cached_Hit@10": float(np.mean(cached_hits)),
        "G50_recomputed_Hit@10": float(np.mean(recomputed_hits)),
        "candidates_per_second": rate,
        "projected_full_validation_hours": projected_candidates / rate / 3600,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        "wall_time_seconds": elapsed,
    }
    gate = scientific_gate(metrics)
    evidence = cli.output_dir / "candidate_scores.tsv"
    with evidence.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "union_rank", "candidate", "source", "gram_score"])
        writer.writerows(output_rows)
    summary = {
        "experiment_id": "GRAM_PHASE10_CF1_B1_TOYS_ARBITRARY_SCORE_PILOT_V1",
        "status": "completed",
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "primary_policy": "fill_cf_only_40",
        "sample_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "metrics": metrics,
        "scientific_gate": gate,
        "artifacts": {"candidate_scores_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()},
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": gate, "metrics": metrics}))


if __name__ == "__main__":
    main()

