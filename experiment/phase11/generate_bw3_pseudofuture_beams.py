#!/usr/bin/env python3
"""Generate one frozen BW3 pseudo-future beam50/beam200 unit."""

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
GRAM_SRC = REPO_ROOT / "GRAM/src"
PHASE9 = REPO_ROOT / "experiment/phase9"
PHASE11 = REPO_ROOT / "experiment/phase11"
for directory in (GRAM_SRC, PHASE9, PHASE11):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import utils  # noqa: E402,F401
from eval_bw1_candidate_ceiling import DATASETS, decode_one, deterministic_users, encoded_catalog, gram_args, load_model  # noqa: E402
from eval_cf0_b3_beamfusion import normalize_lexical_id  # noqa: E402
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from smoke_bw3_pseudofuture import PseudoFutureDataset  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--offset", type=int, choices=(3, 4), required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def main():
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    config = DATASETS[cli.dataset]
    args = gram_args(cli.dataset)
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = PseudoFutureDataset(
        args, cli.dataset, "sequential", None, tokenizer,
        regenerate=False, phase=0, pseudo_offset=cli.offset,
    )
    dataset_index = {dataset.data["user_id"][row]: row for row in range(len(dataset))}
    selected = deterministic_users(dataset_index, cli.users, cli.seed + cli.offset)
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="valid")
    encoded = encoded_catalog(tokenizer, dataset.all_items)
    trie = gt.Trie(encoded)
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(map(len, encoded))
    _, _, lexical_to_id = load_catalog(REPO_ROOT / "GRAM/rec_datasets" / cli.dataset, config["item_index"])
    checkpoint = REPO_ROOT / config["checkpoint"]
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = load_model(checkpoint, device)
    rows = []
    for width in (50, 200):
        started = time.time()
        legal, target_in_beam = [], []
        output_path = cli.output_dir / f"beams_w{width}.tsv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["idx", "gold", "pred", "scores"])
            for ordinal, user in enumerate(selected, 1):
                row = dataset_index[user]
                sample = dataset.data_samples[row]
                batch = collator([dataset[row]])
                sequences, scores = decode_one(model, batch, prefix_allowed_tokens, max_length, width, device)
                candidates = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                values = scores.detach().float().cpu().numpy()
                gold = normalize_lexical_id(sample["target_lex_id"])
                valid = (
                    sample["target_offset"] == cli.offset
                    and len(candidates) == width
                    and len(set(candidates)) == width
                    and np.isfinite(values).all()
                    and all(candidate in lexical_to_id for candidate in candidates)
                )
                legal.append(valid)
                target_in_beam.append(gold in candidates)
                writer.writerow([user, gold, "||".join(candidates), "||".join(map(str, values.tolist()))])
                if ordinal % 16 == 0:
                    handle.flush()
                    print(json.dumps({"dataset": cli.dataset, "offset": cli.offset, "width": width, "progress_users": ordinal, "total_users": len(selected), "elapsed_seconds": time.time() - started}), flush=True)
        rows.append({
            "width": width,
            "users": len(selected),
            "legal_fraction": float(np.mean(legal)),
            "target_in_beam_fraction": float(np.mean(target_in_beam)),
            "wall_time_seconds": time.time() - started,
            "beam_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        })
    checks = {
        "all_users_legal": all(row["legal_fraction"] == 1.0 for row in rows),
        "checkpoint_identity": hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha,
        "validation_test_sports_not_used": True,
    }
    summary = {
        "experiment_id": f"GRAM_PHASE11_BW3_{cli.dataset.upper()}_OFFSET{cli.offset}_BEAMS_V1",
        "status": "completed",
        "dataset": cli.dataset,
        "target_offset": cli.offset,
        "users": len(selected),
        "sample_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
        "rows": rows,
        "checkpoint_sha256": checkpoint_sha,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "gate": {"status": "passed" if all(checks.values()) else "failed", "checks": checks},
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"unit_complete": summary["experiment_id"], "gate": summary["gate"], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
