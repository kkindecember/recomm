#!/usr/bin/env python3
"""CPIA N1: frozen training-prefix audit of GRAM's coarse/fine ID bridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)


def stable_hash(seed: int, dataset: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|cpia-n1-v1|{value}".encode()).hexdigest()


def filtered_lexical_tokens(tokenizer, text: str) -> tuple[int, ...]:
    excluded = {0, 1, 1820, 9175}
    tokens = tuple(token for token in tokenizer.encode(text) if token not in excluded)
    if not tokens:
        raise ValueError(f"empty lexical token sequence for {text!r}")
    return tokens


def find_subsequence(sequence: list[int], query: tuple[int, ...]) -> list[int]:
    width = len(query)
    return [
        start
        for start in range(len(sequence) - width + 1)
        if tuple(sequence[start : start + width]) == query
    ]


def valid_span_count(pair_is_valid: bool) -> int:
    """Count the two coarse/fine spans represented by one validity check."""
    return 2 * int(pair_is_valid)


def choose_samples(prepared: dict, train_users: set[str], config: dict, dataset: str):
    all_samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    latest = {}
    minimum = int(config["minimum_history_items"])
    for sample in all_samples:
        if len(sample["history_items"]) >= minimum:
            latest[sample["user_id"]] = sample
    ordered = sorted(
        latest,
        key=lambda user: stable_hash(int(config["seed"]), dataset, user),
    )
    count = int(config["users_per_dataset"])
    if len(ordered) < count:
        raise ValueError(f"{dataset}: only {len(ordered)} eligible unique users")
    return [latest[user] for user in ordered[:count]]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_users(
    user_rows: list[list[dict]], replicates: int, seed: int, chance: float
) -> dict:
    rng = np.random.default_rng(seed)
    top1, delta = [], []
    for _ in range(replicates):
        indices = rng.integers(0, len(user_rows), len(user_rows))
        rows = [row for index in indices for row in user_rows[index]]
        top1.append(float(np.mean([row["top1_correct"] for row in rows])))
        delta.append(
            float(np.mean([row["matched_minus_mean_mismatch"] for row in rows]))
        )
    return {
        "top1_accuracy": {
            "point": float(
                np.mean([row["top1_correct"] for rows in user_rows for row in rows])
            ),
            "ci_low": float(np.quantile(top1, 0.025)),
            "ci_high": float(np.quantile(top1, 0.975)),
            "chance": chance,
        },
        "matched_minus_mean_mismatch": {
            "point": float(
                np.mean(
                    [
                        row["matched_minus_mean_mismatch"]
                        for rows in user_rows
                        for row in rows
                    ]
                )
            ),
            "ci_low": float(np.quantile(delta, 0.025)),
            "ci_high": float(np.quantile(delta, 0.975)),
        },
    }


@torch.no_grad()
def audit_dataset(
    dataset: str, prepared: dict, config: dict, device: torch.device
) -> tuple[dict, list[dict]]:
    split = (
        ROOT
        / config["inputs"]["split_root"]
        / dataset
        / "train_users.txt"
    )
    train_users = read_users(split)
    samples = choose_samples(prepared, train_users, config, dataset)
    tokenizer = prepared["tokenizer"]
    model = prepared["model"].backbone
    model.eval()
    item_count = int(config["items_per_user"])
    batch_size = int(config["batch_size"])
    rows: list[dict] = []
    mapping = {
        "coarse_expected": 0,
        "coarse_mapped": 0,
        "fine_expected": 0,
        "fine_mapped": 0,
        "attention_checked": 0,
        "attention_valid": 0,
        "finite_checked": 0,
        "finite": 0,
    }
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batch = collate(prepared["collator"], chunk)
        input_ids = batch["item_text_ids"].to(device)
        attention = batch["item_text_masks"].to(device)
        passages, width = input_ids.shape[1], input_ids.shape[2]
        model.encoder.n_passages = passages
        hidden = model.encoder(
            input_ids=input_ids.view(len(chunk), -1),
            attention_mask=attention.view(len(chunk), -1),
            return_dict=True,
        )[0].view(len(chunk), passages, width, -1)
        for batch_index, sample in enumerate(chunk):
            audited_items = list(reversed(sample["history_items"]))[:item_count]
            if len(audited_items) != item_count:
                raise ValueError("locked item count not available")
            coarse_vectors, fine_vectors, item_meta = [], [], []
            for item_index, item in enumerate(audited_items):
                query = filtered_lexical_tokens(
                    tokenizer, prepared["item2lexid"][item]
                )
                coarse_length = int(attention[batch_index, 0].sum())
                fine_passage = item_index + 1
                fine_length = int(attention[batch_index, fine_passage].sum())
                coarse_ids = input_ids[
                    batch_index, 0, :coarse_length
                ].detach().cpu().tolist()
                fine_ids = input_ids[
                    batch_index, fine_passage, :fine_length
                ].detach().cpu().tolist()
                coarse_hits = find_subsequence(coarse_ids, query)
                fine_hits = find_subsequence(fine_ids, query)
                mapping["coarse_expected"] += 1
                mapping["fine_expected"] += 1
                mapping["coarse_mapped"] += int(len(coarse_hits) == 1)
                mapping["fine_mapped"] += int(len(fine_hits) >= 1)
                if len(coarse_hits) != 1 or not fine_hits:
                    raise ValueError(
                        f"{dataset}/{sample['sample_key']}/{item}: "
                        f"coarse_hits={coarse_hits}, fine_hits={fine_hits}"
                    )
                coarse_start, fine_start = coarse_hits[0], fine_hits[0]
                coarse_span = slice(coarse_start, coarse_start + len(query))
                fine_span = slice(fine_start, fine_start + len(query))
                mapping["attention_checked"] += 2
                valid_attention = bool(
                    attention[batch_index, 0, coarse_span].all()
                    and attention[batch_index, fine_passage, fine_span].all()
                )
                mapping["attention_valid"] += valid_span_count(valid_attention)
                if not valid_attention:
                    raise ValueError("mapped ID span intersects padding")
                coarse_vector = F.normalize(
                    hidden[batch_index, 0, coarse_span].float().mean(0), dim=0
                )
                fine_vector = F.normalize(
                    hidden[batch_index, fine_passage, fine_span].float().mean(0),
                    dim=0,
                )
                coarse_vectors.append(coarse_vector)
                fine_vectors.append(fine_vector)
                item_meta.append(
                    (item, coarse_start, fine_passage, fine_start, len(query))
                )
            similarity = torch.stack(coarse_vectors) @ torch.stack(fine_vectors).T
            mapping["finite_checked"] += int(similarity.numel())
            finite = bool(torch.isfinite(similarity).all())
            mapping["finite"] += int(torch.isfinite(similarity).sum())
            if not finite:
                raise ValueError("non-finite similarity matrix")
            matrix = similarity.detach().cpu().numpy()
            for item_index, meta in enumerate(item_meta):
                item, coarse_start, fine_passage, fine_start, token_count = meta
                mismatches = np.delete(matrix[item_index], item_index)
                matched = float(matrix[item_index, item_index])
                hard_margin = matched - float(mismatches.max())
                rows.append(
                    {
                        "dataset": dataset,
                        "sample_key": sample["sample_key"],
                        "user_id": sample["user_id"],
                        "item": item,
                        "history_rank_from_recent": item_index,
                        "coarse_start": coarse_start,
                        "fine_passage": fine_passage,
                        "fine_start": fine_start,
                        "lexical_token_count": token_count,
                        "matched_cosine": matched,
                        "mean_mismatch_cosine": float(mismatches.mean()),
                        "matched_minus_mean_mismatch": matched
                        - float(mismatches.mean()),
                        "hard_margin": hard_margin,
                        "top1_correct": int(
                            int(np.argmax(matrix[item_index])) == item_index
                        ),
                        "mismatch": int(hard_margin <= 0.0),
                    }
                )
    by_user = []
    for user in [sample["user_id"] for sample in samples]:
        user_items = [row for row in rows if row["user_id"] == user]
        if len(user_items) != item_count:
            raise ValueError(f"{dataset}/{user}: incomplete item rows")
        by_user.append(user_items)
    chance = 1.0 / item_count
    bootstrap = bootstrap_users(
        by_user,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + (0 if dataset == "Toys" else 1),
        chance,
    )
    hard_margins = np.asarray([row["hard_margin"] for row in rows])
    integrity = {
        "unique_users": len({row["user_id"] for row in rows}),
        "item_spans": len(rows),
        "coarse_span_mapping_rate": mapping["coarse_mapped"]
        / mapping["coarse_expected"],
        "fine_span_mapping_rate": mapping["fine_mapped"] / mapping["fine_expected"],
        "attention_mask_valid_rate": mapping["attention_valid"]
        / mapping["attention_checked"],
        "finite_rate": mapping["finite"] / mapping["finite_checked"],
    }
    metrics = {
        **bootstrap,
        "signal_excess": bootstrap["top1_accuracy"]["point"] - chance,
        "median_hard_margin": float(np.median(hard_margins)),
        "mismatch_rate": float(np.mean(hard_margins <= 0.0)),
    }
    gates = config["scientific_gates"]
    scientific = {
        "top1_min": metrics["top1_accuracy"]["point"]
        >= float(gates["top1_accuracy_min"]),
        "top1_ci_above_chance": metrics["top1_accuracy"]["ci_low"] > chance,
        "top1_max": metrics["top1_accuracy"]["point"]
        <= float(gates["top1_accuracy_max"]),
        "median_hard_margin": metrics["median_hard_margin"]
        <= float(gates["median_hard_margin_max"]),
        "mismatch_rate": metrics["mismatch_rate"]
        >= float(gates["mismatch_rate_min"]),
        "mean_delta_ci_positive": metrics["matched_minus_mean_mismatch"]["ci_low"]
        > 0.0,
    }
    locked = config["integrity"]
    integrity_pass = (
        integrity["unique_users"] == int(locked["unique_users"])
        and integrity["item_spans"] == int(locked["item_spans"])
        and all(
            integrity[key] == float(locked[key])
            for key in (
                "coarse_span_mapping_rate",
                "fine_span_mapping_rate",
                "attention_mask_valid_rate",
                "finite_rate",
            )
        )
    )
    return {
        "dataset": dataset,
        "metrics": metrics,
        "scientific_gates": scientific,
        "scientific_pass": all(scientific.values()),
        "integrity": integrity,
        "integrity_pass": integrity_pass,
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    code_path = Path(__file__).resolve()
    code_sha = sha256(code_path)
    if code_sha != config["integrity"]["code_sha256"]:
        raise ValueError(
            f"code SHA mismatch: expected {config['integrity']['code_sha256']}, "
            f"got {code_sha}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, all_rows, checkpoint_shas = {}, [], {}
    integrity_ok = True
    for dataset in config["datasets"]:
        p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
        checkpoint = ROOT / p0_config["datasets"][dataset]["checkpoint"]
        before = sha256(checkpoint)
        prepared = prepare(dataset, p0_config, device)
        result, rows = audit_dataset(dataset, prepared, config, device)
        after = sha256(checkpoint)
        result["integrity"]["checkpoint_sha_before"] = before
        result["integrity"]["checkpoint_sha_after"] = after
        result["integrity"]["checkpoint_sha_unchanged"] = before == after
        result["integrity"]["optimizer_steps"] = 0
        result["integrity"]["validation_read"] = False
        result["integrity"]["test_read"] = False
        result["integrity"]["sports_read"] = False
        result["integrity_pass"] = bool(
            result["integrity_pass"] and before == after
        )
        integrity_ok = bool(integrity_ok and result["integrity_pass"])
        checkpoint_shas[dataset] = before
        results[dataset] = result
        all_rows.extend(rows)
        del prepared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scientific_ok = all(results[name]["scientific_pass"] for name in config["datasets"])
    if not integrity_ok:
        decision = "EXECUTION_INVALID"
    elif scientific_ok:
        decision = "CPIA_S0_DESIGN_ALLOWED"
    else:
        decision = "STOP_CPIA_NO_ACTIONABLE_LINK_DEFICIT"
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "code_sha256": code_sha,
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": checkpoint_shas,
        "datasets": results,
        "integrity_pass": integrity_ok,
        "scientific_pass": scientific_ok,
        "validation_read": False,
        "test_read": False,
        "sports_read": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "item_rows.csv", all_rows)
    write_json(args.output_root / "summary.json", summary)
    write_json(
        args.output_root / "status.json",
        {
            "experiment_id": config["experiment_id"],
            "status": "completed",
            "decision": decision,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
