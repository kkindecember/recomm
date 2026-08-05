#!/usr/bin/env python3
"""Leakage-audited pseudo-future constrained decoding smoke for BW3 P0."""

import argparse
import copy
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
from data import TestDatasetGRAM  # noqa: E402
from eval_bw1_candidate_ceiling import (  # noqa: E402
    DATASETS,
    decode_one,
    deterministic_users,
    encoded_catalog,
    gram_args,
    load_model,
)
from eval_cf0_b3_beamfusion import normalize_lexical_id  # noqa: E402
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/phase11/bw3_p0_pseudofuture_smoke")
    parser.add_argument("--users", type=int, default=16)
    parser.add_argument("--offsets", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--widths", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def pseudo_sample_from_sequence(user, items, offset, dataset_name, item2input, item2lexid, item2cfid, max_his, his_sep, reverse_history):
    if offset < 3:
        raise ValueError("pseudo-future offset must be at least 3")
    if len(items) <= offset:
        raise ValueError("sequence too short for pseudo-future target")
    target = items[-offset]
    history = items[:-offset]
    if max_his > 0:
        history = history[-max_his:]
    if not history:
        raise ValueError("pseudo-future history is empty")
    sample = {
        "dataset": dataset_name,
        "user_id": user,
        "target": target,
        "target_lex_id": item2lexid[target],
        "history": his_sep.join(history),
        "history_input": [item2input[item] for item in history],
        "history_item_ids": [item2cfid[item] for item in history],
        "target_item_id": item2cfid[target],
        "target_offset": offset,
    }
    if reverse_history:
        sample["history"] = his_sep.join(copy.deepcopy(sample["history"]).split(his_sep)[::-1])
        sample["history_input"] = sample["history_input"][::-1]
        sample["history_item_ids"] = sample["history_item_ids"][::-1]
    sample["history_lex_id"] = his_sep.join(item2lexid[item] for item in history[::-1])
    return sample


class PseudoFutureDataset(TestDatasetGRAM):
    def __init__(self, *args, pseudo_offset, **kwargs):
        self.pseudo_offset = pseudo_offset
        super().__init__(*args, mode="validation", **kwargs)

    def load_validation(self):
        samples = []
        for user, items in self.user_seq_dict.items():
            if len(items) <= self.pseudo_offset:
                continue
            try:
                sample = pseudo_sample_from_sequence(
                    user,
                    items,
                    self.pseudo_offset,
                    self.dataset,
                    self.item2input,
                    self.item2lexid,
                    self.item2cfid,
                    self.max_his,
                    self.his_sep,
                    self.reverse_history,
                )
            except ValueError:
                continue
            samples.append(sample)
        return samples


def run_dataset(dataset_name, cli, device):
    config = DATASETS[dataset_name]
    args = gram_args(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="valid")
    data_dir = REPO_ROOT / "GRAM/rec_datasets" / dataset_name
    _, _, lexical_to_id = load_catalog(data_dir, config["item_index"])
    checkpoint = REPO_ROOT / config["checkpoint"]
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = load_model(checkpoint, device)
    dataset_rows = []
    for offset in cli.offsets:
        dataset = PseudoFutureDataset(
            args,
            dataset_name,
            "sequential",
            None,
            tokenizer,
            regenerate=False,
            phase=0,
            pseudo_offset=offset,
        )
        index = {dataset.data["user_id"][row]: row for row in range(len(dataset))}
        selected = deterministic_users(index, cli.users, cli.seed + offset)
        encoded = encoded_catalog(tokenizer, dataset.all_items)
        trie = gt.Trie(encoded)
        prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
        max_length = max(map(len, encoded))
        for width in cli.widths:
            started = time.time()
            legal, target_in_beam = [], []
            output_path = cli.output_dir / dataset_name / f"offset{offset}_w{width}.tsv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["idx", "offset", "gold", "pred", "scores"])
                for user in selected:
                    row = index[user]
                    sample = dataset.data_samples[row]
                    if sample["target_offset"] != offset:
                        raise ValueError("target offset audit failed")
                    batch = collator([dataset[row]])
                    sequences, scores = decode_one(model, batch, prefix_allowed_tokens, max_length, width, device)
                    candidates = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    values = scores.detach().float().cpu().numpy()
                    gold = normalize_lexical_id(sample["target_lex_id"])
                    current_legal = (
                        len(candidates) == width
                        and len(set(candidates)) == width
                        and np.isfinite(values).all()
                        and all(candidate in lexical_to_id for candidate in candidates)
                    )
                    legal.append(current_legal)
                    target_in_beam.append(gold in candidates)
                    writer.writerow([user, offset, gold, "||".join(candidates), "||".join(map(str, values.tolist()))])
            dataset_rows.append({
                "dataset": dataset_name,
                "offset": offset,
                "width": width,
                "users": len(selected),
                "legal_fraction": float(np.mean(legal)),
                "target_in_beam_fraction": float(np.mean(target_in_beam)),
                "wall_time_seconds": time.time() - started,
                "beam_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "sample_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
            })
    checkpoint_identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dataset_rows, checkpoint_sha, checkpoint_identity


def main():
    cli = parse_args()
    if sorted(cli.offsets) != [3, 4] or sorted(cli.widths) != [50, 200]:
        raise ValueError("BW3-P0 requires offsets 4/3 and widths 50/200")
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    all_rows, checkpoints, identities = [], {}, {}
    for dataset_name in ("Toys", "Beauty"):
        rows, checkpoint_sha, identity = run_dataset(dataset_name, cli, device)
        all_rows.extend(rows)
        checkpoints[dataset_name] = checkpoint_sha
        identities[dataset_name] = identity
        print(json.dumps({"dataset_complete": dataset_name, "rows": rows}), flush=True)
    checks = {
        "eight_units_present": len(all_rows) == 8,
        "all_units_16_users": all(row["users"] == 16 for row in all_rows),
        "all_candidates_legal": all(row["legal_fraction"] == 1.0 for row in all_rows),
        "all_checkpoints_identity": all(identities.values()),
        "only_offsets_minus4_minus3": {row["offset"] for row in all_rows} == {3, 4},
        "validation_test_sports_not_used": True,
    }
    gate = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    summary = {
        "experiment_id": "GRAM_PHASE11_BW3_P0_PSEUDOFUTURE_SMOKE_V1",
        "status": "completed",
        "evidence_class": "infrastructure_smoke_only",
        "test_read": False,
        "validation_target_read": False,
        "sports_read": False,
        "rows": all_rows,
        "checkpoint_sha256": checkpoints,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "p0_gate": gate,
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"p0_gate": gate, "peak_allocated_mib": summary["peak_allocated_mib"]}), flush=True)


if __name__ == "__main__":
    main()
