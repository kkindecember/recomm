#!/usr/bin/env python3
"""Teacher-forced identity check for cached GRAM beam sequence scores."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5Config
from transformers.modeling_outputs import BaseModelOutput


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = REPO_ROOT / "GRAM/src"
PHASE9 = REPO_ROOT / "experiment/phase9"
for directory in (GRAM_SRC, PHASE9):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import utils  # noqa: E402,F401  # Initialize GRAM's utils package before data to avoid its legacy cycle.
from arguments import create_parser  # noqa: E402
from data import TestDatasetGRAM  # noqa: E402
from eval_cf0_b3_beamfusion import load_cached_beams, normalize_lexical_id  # noqa: E402
from model import create_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt"
DEFAULT_PREDICTIONS = REPO_ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_b0_toys_score_identity"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=64)
    parser.add_argument("--candidate-batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def deterministic_users(user_ids, count):
    return sorted(
        user_ids,
        key=lambda user: (hashlib.sha256(f"2023:{user}".encode()).hexdigest(), user),
    )[:count]


def rank_values(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rank_values(x), rank_values(y))[0, 1]),
    }


def sequence_scores(logits, target_ids, target_mask, length_penalty=1.0):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    summed = (gathered * target_mask).sum(dim=1)
    lengths = target_mask.sum(dim=1).to(summed.dtype)
    return summed / lengths.pow(length_penalty)


def scientific_gate(metrics):
    checks = {
        "all_scores_finite": metrics["finite_fraction"] == 1.0,
        "pearson_at_least_0.995": metrics["pearson"] >= 0.995,
        "spearman_at_least_0.995": metrics["spearman"] >= 0.995,
        "top10_overlap_at_least_0.98": metrics["mean_top10_set_overlap"] >= 0.98,
        "hit10_identity_within_0.001": abs(metrics["recomputed_Hit@10"] - metrics["cached_Hit@10"]) <= 0.001,
        "peak_allocated_mib_at_most_12000": metrics["peak_allocated_mib"] <= 12000,
        "wall_time_at_most_1800": metrics["wall_time_seconds"] <= 1800,
    }
    return {"status": "passed" if all(checks.values()) else "failed_score_identity_gate", "checks": checks}


def gram_args():
    args = create_parser().parse_args([
        "--hierarchical_id_type", "hierarchy_v1_c32_l5_len32768_split",
        "--datasets", "Toys",
        "--data_path", str(REPO_ROOT / "GRAM/rec_datasets"),
        "--prompt_file", str(REPO_ROOT / "GRAM/prompt.txt"),
        "--item_prompt", "all_text",
        "--top_k_similar_item", "5",
        "--cf_model", "sasrec",
        "--id_linking", "1",
        "--max_his", "20",
        "--item_prompt_max_len", "128",
        "--target_max_len", "32",
        "--item_id_type", "split",
        "--reverse_history", "1",
        "--user_id_without_target_item", "1",
        "--train", "0",
    ])
    args.rank = 0
    args.debug_test_100 = 0
    return args


def raw_lexical_map():
    path = REPO_ROOT / "GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
    mapping = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            _, raw_lexical = line.rstrip("\n").split(" ", 1)
            mapping[normalize_lexical_id(raw_lexical)] = raw_lexical
    return mapping


def load_model(checkpoint, device):
    config = T5Config.from_pretrained("t5-small", local_files_only=True)
    config.max_seq_len = 128
    config.max_item_num = 20
    config.use_position_embedding = 1
    config.sample_num = "1"
    config.cf0_enabled = False
    config.cf0_num_items = 0
    model = create_model("gram", config=config)
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    return model


def score_user(model, batch, raw_candidates, collator, device, micro_batch):
    input_ids = batch["item_text_ids"].to(device)
    attention_mask = batch["item_text_masks"].to(device)
    model.encoder.n_passages = input_ids.size(1)
    flat_ids = input_ids.reshape(1, -1)
    flat_mask = attention_mask.reshape(1, -1)
    with torch.no_grad():
        hidden = model.encoder(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True)[0]
        result = []
        for start in range(0, len(raw_candidates), micro_batch):
            candidate_text = raw_candidates[start : start + micro_batch]
            target = collator.encode_target_split(candidate_text)
            target_ids = target["input_ids"].to(device)
            target_mask = target["attention_mask"].to(device).bool()
            labels = target_ids.masked_fill(~target_mask, -100)
            count = target_ids.size(0)
            outputs = model(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden.expand(count, -1, -1)),
                attention_mask=flat_mask.expand(count, -1),
                labels=labels,
                return_dict=True,
            )
            result.extend(sequence_scores(outputs.logits, target_ids, target_mask).cpu().tolist())
    return result


def main():
    cli = parse_args()
    started = time.time()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    cache, _ = load_cached_beams(cli.predictions)
    selected = deterministic_users(cache, cli.users)
    args = gram_args()
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = TestDatasetGRAM(args, "Toys", "sequential", None, tokenizer, regenerate=False, phase=0, mode="validation")
    dataset_index = {dataset.data["user_id"][idx]: idx for idx in range(len(dataset))}
    if set(selected) - set(dataset_index):
        raise ValueError("pilot users missing from validation dataset")
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="test")
    lexical = raw_lexical_map()
    model = load_model(cli.checkpoint, device)

    rows, cached_all, recomputed_all, overlaps = [], [], [], []
    cached_hits, recomputed_hits = [], []
    for user in selected:
        cached = cache[user]
        raw_candidates = [lexical[candidate] for candidate in cached["candidates"]]
        batch = collator([dataset[dataset_index[user]]])
        recomputed = np.asarray(score_user(model, batch, raw_candidates, collator, device, cli.candidate_batch_size))
        cached_scores = np.asarray(cached["seq"])
        cached_order = np.argsort(-cached_scores, kind="stable")
        recomputed_order = np.argsort(-recomputed, kind="stable")
        overlap = len(set(cached_order[:10]) & set(recomputed_order[:10])) / 10.0
        overlaps.append(overlap)
        gold = cached["gold"]
        target_position = cached["candidates"].index(gold) if gold in cached["candidates"] else -1
        cached_rank = int(np.flatnonzero(cached_order == target_position)[0]) + 1 if target_position >= 0 else 51
        recomputed_rank = int(np.flatnonzero(recomputed_order == target_position)[0]) + 1 if target_position >= 0 else 51
        cached_hits.append(cached_rank <= 10)
        recomputed_hits.append(recomputed_rank <= 10)
        for rank, (candidate, old, new) in enumerate(zip(cached["candidates"], cached_scores, recomputed), 1):
            rows.append((user, rank, candidate, old, new, new - old))
        cached_all.extend(cached_scores.tolist())
        recomputed_all.extend(recomputed.tolist())

    elapsed = time.time() - started
    corr = correlation(cached_all, recomputed_all)
    metrics = {
        "users": len(selected),
        "pairs": len(rows),
        "finite_fraction": float(np.isfinite(recomputed_all).mean()),
        **corr,
        "mean_top10_set_overlap": float(np.mean(overlaps)),
        "cached_Hit@10": float(np.mean(cached_hits)),
        "recomputed_Hit@10": float(np.mean(recomputed_hits)),
        "mean_absolute_score_error": float(np.mean(np.abs(np.asarray(recomputed_all) - np.asarray(cached_all)))),
        "max_absolute_score_error": float(np.max(np.abs(np.asarray(recomputed_all) - np.asarray(cached_all)))),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        "wall_time_seconds": elapsed,
    }
    gate = scientific_gate(metrics)
    evidence = cli.output_dir / "score_pairs.tsv"
    with evidence.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "cached_rank", "candidate", "cached_score", "recomputed_score", "delta"])
        writer.writerows(rows)
    summary = {
        "experiment_id": "GRAM_PHASE10_CF1_B0_TOYS_SCORE_IDENTITY_V1",
        "status": "completed",
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "sample_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "score_definition": "full-vocabulary teacher-forced log-prob sum including EOS divided by predicted-token count",
        "metrics": metrics,
        "scientific_gate": gate,
        "artifacts": {"score_pairs_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()},
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"scientific_gate": gate, "metrics": metrics}))


if __name__ == "__main__":
    main()
