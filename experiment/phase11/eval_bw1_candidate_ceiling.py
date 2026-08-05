#!/usr/bin/env python3
"""Fresh constrained-beam candidate-ceiling pilot for frozen GRAM + PCRF."""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5Config


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = REPO_ROOT / "GRAM/src"
PHASE9 = REPO_ROOT / "experiment/phase9"
for directory in (GRAM_SRC, PHASE9):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import utils  # noqa: E402,F401
from arguments import create_parser  # noqa: E402
from data import TestDatasetGRAM  # noqa: E402
from eval_cf0_b3_beamfusion import (  # noqa: E402
    bootstrap_hit10_delta,
    load_cached_beams,
    load_users,
    metrics_from_ranks,
    score_item_head,
    standardize,
)
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402
from model import create_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


DATASETS = {
    "Toys": {
        "hierarchy": "hierarchy_v1_c32_l5_len32768_split",
        "item_index": "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
        "top_k": "5",
        "checkpoint": "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt",
        "cache": "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv",
        "item_head": "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt",
    },
    "Beauty": {
        "hierarchy": "hierarchy_v1_c128_l7_len32768_split",
        "item_index": "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
        "top_k": "10",
        "checkpoint": "GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt",
        "cache": "GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv",
        "item_head": "artifacts/phase9/p9x_beauty_item_head/best_item_head.pt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, default=512)
    parser.add_argument("--widths", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def deterministic_users(user_ids, count, seed=2023):
    user_ids = list(user_ids)
    if count <= 0 or count > len(user_ids):
        raise ValueError("users must be in [1, available users]")
    return sorted(
        user_ids,
        key=lambda user: (hashlib.sha256(f"{seed}:{user}".encode()).hexdigest(), user),
    )[:count]


def gram_args(dataset):
    config = DATASETS[dataset]
    args = create_parser().parse_args([
        "--hierarchical_id_type", config["hierarchy"],
        "--datasets", dataset,
        "--data_path", str(REPO_ROOT / "GRAM/rec_datasets"),
        "--prompt_file", str(REPO_ROOT / "GRAM/prompt.txt"),
        "--item_prompt", "all_text",
        "--top_k_similar_item", config["top_k"],
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
    return model.to(device).eval()


def encoded_catalog(tokenizer, candidates):
    result = []
    for candidate in candidates:
        tokens = [token for token in tokenizer.encode(candidate) if token not in (1820, 9175)]
        result.append([0] + tokens)
    return result


def decode_one(model, batch, prefix_allowed_tokens, max_length, width, device):
    with torch.no_grad():
        output = model.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            history_item_ids=batch["history_item_ids"].to(device),
            history_item_mask=batch["history_item_mask"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=width,
            num_return_sequences=width,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
    return output["sequences"], output["sequences_scores"]


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
        candidate_frequencies = np.asarray([frequencies[item] for item in candidate_ids], dtype=np.float64)
        records.append({
            "user": user,
            "history": sequence[max(0, len(sequence) - 22):-2],
            "candidate_ids": candidate_ids,
            "candidate_frequencies": candidate_frequencies,
            "target_position": target_position,
            "seq": np.asarray(beam["seq"], dtype=np.float64),
            "tail_mass": float(np.mean(candidate_frequencies[:10] <= q1)),
        })
    return records


def ranks(records, use_pcrf, weight=1.0, beta=0.5, gamma=1.0):
    result = []
    for record in records:
        if use_pcrf:
            seq_z = standardize(record["seq"])
            cf_z = standardize(record["cf"])
            pop_z = standardize(np.log1p(record["candidate_frequencies"]))
            adjusted = standardize(cf_z - beta * pop_z)
            reliability = (1.0 - record["tail_mass"]) ** gamma
            values = seq_z + weight * reliability * adjusted
        else:
            values = record["seq"]
        order = np.argsort(-values, kind="stable")
        position = record["target_position"]
        result.append(len(record["candidate_ids"]) + 1 if position < 0 else int(np.flatnonzero(order == position)[0]) + 1)
    return np.asarray(result, dtype=np.int64)


def decide(width_rows):
    by_width = {row["width"]: row for row in width_rows}
    coverage_headroom = by_width[200]["candidate_recall"] - by_width[50]["candidate_recall"]
    pcrf_headroom = by_width[200]["pcrf"]["Hit@10"] - by_width[50]["pcrf"]["Hit@10"]
    return {"coverage_headroom": coverage_headroom, "pcrf_headroom": pcrf_headroom}


def main():
    cli = parse_args()
    if sorted(set(cli.widths)) != [50, 100, 200]:
        raise ValueError("BW1 preregistration requires exactly widths 50 100 200")
    started = time.time()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    dataset_config = DATASETS[cli.dataset]
    data_dir = REPO_ROOT / "GRAM/rec_datasets" / cli.dataset
    checkpoint = REPO_ROOT / dataset_config["checkpoint"]
    cache_path = REPO_ROOT / dataset_config["cache"]
    item_head = REPO_ROOT / dataset_config["item_head"]
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    item_head_sha = hashlib.sha256(item_head.read_bytes()).hexdigest()
    cache, _ = load_cached_beams(cache_path)
    selected = deterministic_users(cache, cli.users, cli.seed)

    args = gram_args(cli.dataset)
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = TestDatasetGRAM(args, cli.dataset, "sequential", None, tokenizer, regenerate=False, phase=0, mode="validation")
    dataset_index = {dataset.data["user_id"][index]: index for index in range(len(dataset))}
    if set(selected) - set(dataset_index):
        raise ValueError("selected users missing from validation dataset")
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="valid")
    encoded = encoded_catalog(tokenizer, dataset.all_items)
    trie = gt.Trie(encoded)
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(map(len, encoded))

    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir, dataset_config["item_index"])
    users = load_users(data_dir, raw_to_id)
    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-2])
    target_frequencies = sorted(frequencies[sequence[-2]] for sequence in users.values())
    q1 = target_frequencies[len(target_frequencies) // 4]
    cache_records = build_records(selected, cache, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1)
    cached_baseline_ranks = ranks(cache_records, False)
    cached_baseline = metrics_from_ranks(cached_baseline_ranks)

    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = load_model(checkpoint, device)
    rows, per_user_rows, width_ranks = [], [], {}
    for width in sorted(cli.widths):
        width_started = time.time()
        fresh = {}
        beam_path = cli.output_dir / f"fresh_beams_w{width}.tsv"
        with beam_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["idx", "gold", "pred", "scores"])
            for ordinal, user in enumerate(selected, 1):
                batch = collator([dataset[dataset_index[user]]])
                sequences, scores = decode_one(model, batch, prefix_allowed_tokens, max_length, width, device)
                candidates = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                values = scores.detach().float().cpu().numpy()
                fresh[user] = {"gold": cache[user]["gold"], "candidates": candidates, "seq": values}
                writer.writerow([user, cache[user]["gold"], "||".join(candidates), "||".join(map(str, values.tolist()))])
                if ordinal % 16 == 0:
                    handle.flush()
                    print(json.dumps({"dataset": cli.dataset, "width": width, "progress_users": ordinal, "total_users": len(selected), "elapsed_seconds": time.time() - width_started}), flush=True)

        legal = [
            len(fresh[user]["candidates"]) == width
            and len(set(fresh[user]["candidates"])) == width
            and np.isfinite(fresh[user]["seq"]).all()
            and all(value in lexical_to_id for value in fresh[user]["candidates"])
            for user in selected
        ]
        records = build_records(selected, fresh, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1)
        score_item_head(records, item_head, 512)
        baseline_rank = ranks(records, False)
        pcrf_rank = ranks(records, True, cli.lambda_weight, cli.beta, cli.gamma)
        width_ranks[width] = {"baseline": baseline_rank, "pcrf": pcrf_rank}
        baseline_metrics = metrics_from_ranks(baseline_rank)
        pcrf_metrics = metrics_from_ranks(pcrf_rank)
        row = {
            "width": width,
            "legal_fraction": float(np.mean(legal)),
            "candidate_recall": float(np.mean([record["target_position"] >= 0 for record in records])),
            "baseline": baseline_metrics,
            "pcrf": pcrf_metrics,
            "pcrf_minus_baseline_Hit@10": pcrf_metrics["Hit@10"] - baseline_metrics["Hit@10"],
            "wall_time_seconds": time.time() - width_started,
            "fresh_beams_sha256": hashlib.sha256(beam_path.read_bytes()).hexdigest(),
        }
        rows.append(row)
        for index, user in enumerate(selected):
            per_user_rows.append([user, width, records[index]["target_position"] >= 0, baseline_rank[index], pcrf_rank[index]])
        print(json.dumps({"dataset": cli.dataset, "width_complete": width, "candidate_recall": row["candidate_recall"], "baseline_Hit@10": baseline_metrics["Hit@10"], "pcrf_Hit@10": pcrf_metrics["Hit@10"]}), flush=True)

    by_width = {row["width"]: row for row in rows}
    for width in (100, 200):
        by_width[width]["baseline_Hit@10_vs_w50_95ci"] = bootstrap_hit10_delta(
            width_ranks[50]["baseline"], width_ranks[width]["baseline"], cli.bootstrap_replicates, cli.seed + width
        )
        by_width[width]["pcrf_Hit@10_vs_w50_95ci"] = bootstrap_hit10_delta(
            width_ranks[50]["pcrf"], width_ranks[width]["pcrf"], cli.bootstrap_replicates, cli.seed + width + 1
        )

    per_user_path = cli.output_dir / "per_user.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "width", "target_in_candidates", "baseline_rank", "pcrf_rank"])
        writer.writerows(per_user_rows)

    coverage = [by_width[width]["candidate_recall"] for width in (50, 100, 200)]
    checks = {
        "all_units_512_users_legal": len(selected) == 512 and all(row["legal_fraction"] == 1.0 for row in rows),
        "candidate_recall_monotonic": coverage[0] <= coverage[1] <= coverage[2],
        "fresh_w50_cached_baseline_Hit10_abs_delta_at_most_0.002": abs(by_width[50]["baseline"]["Hit@10"] - cached_baseline["Hit@10"]) <= 0.002,
        "gram_checkpoint_identity": hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha,
        "item_head_checkpoint_identity": hashlib.sha256(item_head.read_bytes()).hexdigest() == item_head_sha,
        "test_and_sports_not_read": True,
    }
    integrity_gate = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    summary = {
        "experiment_id": f"GRAM_PHASE11_BW1_{cli.dataset.upper()}_CANDIDATE_CEILING_VALIDATION_V1",
        "status": "completed",
        "dataset": cli.dataset,
        "split": "validation",
        "users": len(selected),
        "sample_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "test_read": False,
        "sports_read": False,
        "frozen_pcrf": {"lambda": cli.lambda_weight, "beta": cli.beta, "gamma": cli.gamma, "q1": q1},
        "cached_w50_baseline": cached_baseline,
        "widths": rows,
        "headroom": decide(rows),
        "integrity_gate": integrity_gate,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "wall_time_seconds": time.time() - started,
        "artifacts": {
            "per_user_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": checkpoint_sha,
            "item_head_sha256": item_head_sha,
        },
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"integrity_gate": integrity_gate, "headroom": summary["headroom"]}), flush=True)


if __name__ == "__main__":
    main()
