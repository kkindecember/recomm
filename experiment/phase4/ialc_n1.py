#!/usr/bin/env python3
"""IALC N1: frozen training-prefix audit of Trie/full-vocabulary support mismatch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.chpr_a0 import pad_labels  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from utils import generation_trie as gt  # noqa: E402


def stable_key(seed: int, dataset: str, sample_key: str) -> str:
    payload = f"{seed}|{dataset}|ialc-n1|{sample_key}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_unique_user_samples(
    samples: list[dict],
    head_items: set[str],
    seed: int,
    dataset: str,
    head_count: int,
    tail_count: int,
) -> list[dict]:
    ordered = sorted(
        samples, key=lambda row: stable_key(seed, dataset, row["sample_key"])
    )
    limits = {"head": head_count, "tail": tail_count}
    counts = {"head": 0, "tail": 0}
    users, selected = set(), []
    for row in ordered:
        group = "head" if row["positive_item"] in head_items else "tail"
        if row["user_id"] in users or counts[group] >= limits[group]:
            continue
        selected.append(row)
        users.add(row["user_id"])
        counts[group] += 1
        if counts == limits:
            break
    if counts != limits:
        raise ValueError(f"insufficient unique-user samples for {dataset}: {counts}")
    return sorted(selected, key=lambda row: row["sample_key"])


def support_metrics(
    logits: torch.Tensor, allowed: list[int], gold: int
) -> dict[str, float | int]:
    if not allowed or gold not in allowed:
        raise ValueError("gold token is not a legal Trie child")
    values = logits.float()
    legal_indices = torch.as_tensor(allowed, dtype=torch.long, device=values.device)
    legal_values = values.index_select(0, legal_indices)
    full_lse = torch.logsumexp(values, dim=0)
    legal_lse = torch.logsumexp(legal_values, dim=0)
    log_legal_mass = legal_lse - full_lse
    legal_mass = torch.exp(log_legal_mass).clamp(0.0, 1.0)
    loss_gap = -log_legal_mass
    gold_value = values[gold]
    legal_position = allowed.index(gold)
    return {
        "legal_child_count": len(allowed),
        "legal_mass": float(legal_mass),
        "illegal_mass": float(1.0 - legal_mass),
        "loss_gap": float(loss_gap),
        "full_rank": int((values > gold_value).sum().item() + 1),
        "legal_rank": int(
            (legal_values > legal_values[legal_position]).sum().item() + 1
        ),
    }


@torch.no_grad()
def audit_batch(
    samples: list[dict],
    prepared: dict,
    trie: gt.Trie,
    item_to_sequence: dict[str, list[int]],
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    batch = collate(prepared["collator"], samples)
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    sequences = [item_to_sequence[row["positive_item"]] for row in samples]
    labels = pad_labels(sequences, device)
    output = prepared["model"].backbone(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        return_dict=True,
    )
    logits = output.logits
    eos = int(prepared["tokenizer"].eos_token_id)
    step_rows, sample_rows = [], []
    for batch_index, (sample, sequence) in enumerate(zip(samples, sequences)):
        sample_steps = []
        target_group = (
            "head"
            if sample["positive_item"] in prepared["heads"]
            else "tail"
        )
        for position, gold in enumerate(sequence[1:]):
            allowed = trie.get(sequence[: position + 1])
            metrics = support_metrics(logits[batch_index, position], allowed, gold)
            formula_error = abs(
                metrics["loss_gap"] + math.log(max(metrics["legal_mass"], 1e-45))
            )
            row = {
                "sample_key": sample["sample_key"],
                "user_id": sample["user_id"],
                "target_item": sample["positive_item"],
                "target_group": target_group,
                "depth": position,
                "gold_token": gold,
                "is_eos": int(gold == eos),
                "competitive": int(gold != eos and len(allowed) >= 2),
                **metrics,
                "formula_error": formula_error,
            }
            step_rows.append(row)
            if row["competitive"]:
                sample_steps.append(row)
        sample_rows.append(
            {
                "sample_key": sample["sample_key"],
                "user_id": sample["user_id"],
                "target_group": target_group,
                "competitive_steps": len(sample_steps),
                "max_illegal_mass": max(
                    (row["illegal_mass"] for row in sample_steps), default=0.0
                ),
                "mean_loss_gap": float(
                    np.mean([row["loss_gap"] for row in sample_steps])
                )
                if sample_steps
                else 0.0,
            }
        )
    return step_rows, sample_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(
    dataset: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT
        / config["inputs"]["checkpoint_root"]
        / dataset
        / "C0"
        / "model.pt"
    )
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    item_to_sequence = {
        item: sequence
        for item, sequence in zip(
            prepared["catalog"], prepared["encoded_candidates"]
        )
    }
    users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    all_samples = build_train_samples(
        prepared["sequences"],
        users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    samples = select_unique_user_samples(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        int(config["head_samples"]),
        int(config["tail_samples"]),
    )
    step_rows, sample_rows = [], []
    batch_size = int(config["audit"]["batch_size"])
    for start in range(0, len(samples), batch_size):
        steps, summaries = audit_batch(
            samples[start : start + batch_size],
            prepared,
            trie,
            item_to_sequence,
            device,
        )
        step_rows.extend(steps)
        sample_rows.extend(summaries)
        done = min(start + batch_size, len(samples))
        if done % 32 == 0:
            print(
                f"IALC_N1_PROGRESS dataset={dataset} samples={done}/{len(samples)}",
                flush=True,
            )

    output_dir = output_root / dataset
    step_path = output_dir / "support_steps.csv"
    sample_path = output_dir / "sample_summary.csv"
    write_csv(step_path, step_rows)
    write_csv(sample_path, sample_rows)
    competitive = [row for row in step_rows if row["competitive"]]
    tail = [row for row in competitive if row["target_group"] == "tail"]
    depth_counts = Counter(row["depth"] for row in competitive)
    gates = config["scientific_gates"]
    supported_depths = {
        str(depth): count
        for depth, count in sorted(depth_counts.items())
        if count >= int(gates["minimum_competitive_steps_per_supported_depth"])
    }
    sample_coverage = float(
        np.mean([row["competitive_steps"] > 0 for row in sample_rows])
    )
    mean_gap = float(np.mean([row["loss_gap"] for row in competitive]))
    tail_gap = float(np.mean([row["loss_gap"] for row in tail]))
    large_mass_rate = float(
        np.mean(
            [
                row["max_illegal_mass"]
                >= float(gates["large_illegal_mass_threshold"])
                for row in sample_rows
            ]
        )
    )
    tail_ratio = tail_gap / mean_gap if mean_gap > 0 else 0.0
    metrics = {
        "samples": len(sample_rows),
        "unique_users": len({row["user_id"] for row in sample_rows}),
        "steps": len(step_rows),
        "competitive_steps": len(competitive),
        "competitive_steps_by_depth": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "supported_depths": supported_depths,
        "sample_competitive_coverage": sample_coverage,
        "mean_loss_gap": mean_gap,
        "mean_illegal_mass": float(
            np.mean([row["illegal_mass"] for row in competitive])
        ),
        "tail_mean_loss_gap": tail_gap,
        "tail_to_overall_mean_loss_gap_ratio": tail_ratio,
        "sample_large_illegal_mass_rate": large_mass_rate,
    }
    checks = {
        "sample_competitive_coverage": sample_coverage
        >= float(gates["sample_competitive_coverage_min"]),
        "supported_depths": len(supported_depths)
        >= int(gates["minimum_supported_depths"]),
        "mean_loss_gap": mean_gap >= float(gates["mean_loss_gap_min"]),
        "sample_large_illegal_mass_rate": large_mass_rate
        >= float(gates["sample_large_illegal_mass_rate_min"]),
        "tail_loss_gap": tail_ratio
        >= float(gates["tail_to_overall_mean_loss_gap_ratio_min"]),
    }
    max_formula_error = max(row["formula_error"] for row in step_rows)
    integrity = {
        "mapping_rate": 1.0,
        "trie_membership_rate": 1.0,
        "finite_rate": float(
            np.mean(
                [
                    math.isfinite(row["legal_mass"])
                    and math.isfinite(row["loss_gap"])
                    for row in step_rows
                ]
            )
        ),
        "unique_user_rate": metrics["unique_users"] / metrics["samples"],
        "max_formula_error": max_formula_error,
        "optimizer_steps": 0,
        "parameter_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "validation_test_predictions_read": False,
        "sports_read": False,
        "checkpoint_sha256": checkpoint_sha,
        "step_file_sha256": sha256(step_path),
        "sample_file_sha256": sha256(sample_path),
    }
    integrity_valid = (
        integrity["mapping_rate"] == 1.0
        and integrity["trie_membership_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["unique_user_rate"] == 1.0
        and integrity["max_formula_error"]
        <= float(config["integrity"]["formula_tolerance"])
        and integrity["optimizer_steps"] == 0
        and integrity["parameter_sha_unchanged"]
        and not integrity["validation_test_predictions_read"]
        and not integrity["sports_read"]
    )
    del prepared
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("IALC N1 requires CUDA")
    config = json.loads(args.config.read_text())
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(
            dataset, config, p0_config, args.output_root, device
        )
        for dataset in config["datasets"]
    }
    integrity_valid = all(row["integrity_valid"] for row in results.values())
    scientific_pass = all(row["scientific_pass"] for row in results.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "IALC_S0_DESIGN_ALLOWED"
        if scientific_pass
        else "STOP_IALC_NO_SUPPORT_MISMATCH"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "integrity_valid": integrity_valid,
        "validation_test_predictions_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
