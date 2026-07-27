#!/usr/bin/env python3
"""Locked HBTR 10% pilot cache, training, and validation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE3 = Path(__file__).resolve().parent
GRAM_SRC = ROOT / "GRAM/src"
for path in (PHASE3, GRAM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from utils import indexing  # noqa: E402

from hbtr_b1_objective import (  # noqa: E402
    NEGATIVE_COUNT,
    RANKING_LAMBDA,
    canonical_cache_sha256,
    component_margin,
    load_cache,
    pairwise_ranking_loss,
    sequence_log_scores,
    total_loss,
    validate_cache_row,
)
from hbtr_b1_smoke import (  # noqa: E402
    DATASETS,
    create_model_and_tokenizer,
    encode_candidates,
    make_runtime_args,
    normalized_sequence,
    read_semantic_tokens,
    read_sequences,
    sha256,
)
from hbtr_pilot_split import history_bin  # noqa: E402


CONTROLS = ("C0", "C1", "C2", "C3", "C4")
EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-5
SEED = 2023


def read_user_set(path: Path) -> set[str]:
    return {line.strip() for line in path.open() if line.strip()}


def build_train_samples(
    sequences: dict[str, list[str]],
    selected_users: set[str],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
) -> list[dict]:
    samples = []
    for user in sorted(selected_users):
        items = sequences[user][:-2]
        for target_index in range(1, len(items)):
            target = items[target_index]
            history = items[:target_index][-20:]
            if target not in item2lexid or any(item not in item2input for item in history):
                continue
            reversed_history = list(reversed(history))
            history_lex = " ; ".join(item2lexid[item] for item in reversed_history)
            samples.append(
                {
                    "sample_key": f"{user}:{target_index}:{target}",
                    "user_id": user,
                    "positive_item": target,
                    "history_items": history,
                    "input": [f"What would user purchase after {history_lex} ?"]
                    + [item2input[item] for item in reversed_history],
                    "output": item2lexid[target],
                }
            )
    if len({sample["sample_key"] for sample in samples}) != len(samples):
        raise ValueError("duplicate pilot training sample keys")
    return samples


def build_validation_samples(
    sequences: dict[str, list[str]],
    selected_users: set[str],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
) -> list[dict]:
    samples = []
    for user in sorted(selected_users):
        items = sequences[user]
        target = items[-2]
        raw_history = items[:-2]
        history = raw_history[-20:]
        reversed_history = list(reversed(history))
        history_lex = " ; ".join(item2lexid[item] for item in reversed_history)
        samples.append(
            {
                "sample_key": f"{user}:validation:{target}",
                "user_id": user,
                "positive_item": target,
                "history_items": history,
                "raw_history_length": len(raw_history),
                "input": [f"What would user purchase after {history_lex} ?"]
                + [item2input[item] for item in reversed_history],
                "output": item2lexid[target],
            }
        )
    return samples


def collate_samples(collator: CollatorGRAM, samples: list[dict]):
    return collator(
        [
            {
                "input": sample["input"],
                "output": sample["output"],
                "user_id": sample["user_id"],
            }
            for sample in samples
        ]
    )


def encode_targets(collator: CollatorGRAM, texts: list[str]) -> torch.Tensor:
    target = collator.encode_target_split(texts)
    labels = target["input_ids"]
    mask = target["attention_mask"].bool()
    return labels.masked_fill(~mask, -100)


def training_popularity(sequences: dict[str, list[str]]) -> Counter:
    counts: Counter = Counter()
    for items in sequences.values():
        counts.update(items[:-2])
    return counts


def head_items(popularity: Counter) -> set[str]:
    ordered = sorted(popularity, key=lambda item: (-popularity[item], item))
    return set(ordered[: max(1, math.ceil(len(ordered) * 0.20))])


def prepare_dataset(dataset: str, device: torch.device):
    runtime = make_runtime_args(dataset)
    model, tokenizer, _ = create_model_and_tokenizer(dataset, device)
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    _, item2input, item2lexid = indexing.gram_indexing(
        data_path=runtime.data_path,
        dataset=dataset,
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        args=runtime,
        user_id_without_target_item=True,
        id_linking=True,
    )
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    encoded_candidates, sequence_to_item = encode_candidates(tokenizer, item2lexid)
    index_path = (
        dataset_dir
        / f"item_generative_indexing_{DATASETS[dataset]['hierarchical_id_type']}.txt"
    )
    return {
        "runtime": runtime,
        "model": model,
        "tokenizer": tokenizer,
        "sequences": sequences,
        "item2input": item2input,
        "item2lexid": item2lexid,
        "collator": collator,
        "encoded_candidates": encoded_candidates,
        "sequence_to_item": sequence_to_item,
        "semantic_tokens": read_semantic_tokens(index_path),
        "popularity": training_popularity(sequences),
    }


@torch.no_grad()
def build_cache(dataset: str, prepared: dict, output_dir: Path, device: torch.device):
    split_dir = ROOT / "artifacts/phase3/hbtr_pilot_splits" / dataset
    train_users = read_user_set(split_dir / "train_users.txt")
    samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    model = prepared["model"]
    model.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(len(candidate) for candidate in prepared["encoded_candidates"])
    rows = []
    rank_counts: Counter = Counter()
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        batch = collate_samples(prepared["collator"], [sample])
        prediction = model.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=50,
            num_return_sequences=50,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
        predicted_items = [
            prepared["sequence_to_item"].get(normalized_sequence(ids.tolist()))
            for ids in prediction["sequences"]
        ]
        if any(item is None for item in predicted_items) or len(set(predicted_items)) != 50:
            raise ValueError("pilot cache beam violated locked Trie uniqueness")
        positive = sample["positive_item"]
        rank = predicted_items.index(positive) + 1 if positive in predicted_items else None
        rank_counts["miss50" if rank is None else ("hit10" if rank <= 10 else "rank11_50")] += 1
        if index % 250 == 0:
            print(
                f"CACHE_PROGRESS dataset={dataset} samples={index}/{len(samples)} "
                f"valid_rows={len(rows)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        if rank is None or not 11 <= rank <= 50:
            continue
        history = set(sample["history_items"])
        negatives = []
        for item in predicted_items[:10]:
            if item == positive or item in history or item in negatives:
                continue
            negatives.append(item)
            if len(negatives) == NEGATIVE_COUNT:
                break
        if len(negatives) != NEGATIVE_COUNT:
            rank_counts["insufficient_negatives"] += 1
            continue
        row = {
            "sample_key": sample["sample_key"],
            "user_id": sample["user_id"],
            "positive_item": positive,
            "positive_rank": rank,
            "history_items": sample["history_items"],
            "negative_items": negatives,
            "prefix_depths": [0] * len(negatives),
            "positive_frequency": int(prepared["popularity"][positive]),
        }
        # Record the exact semantic-token longest common prefix depth.
        for neg_index, negative in enumerate(negatives):
            depth = 0
            for lhs, rhs in zip(
                prepared["semantic_tokens"][positive],
                prepared["semantic_tokens"][negative],
            ):
                if lhs != rhs:
                    break
                depth += 1
            row["prefix_depths"][neg_index] = depth
        validate_cache_row(row, valid_items=set(prepared["item2lexid"]))
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = DATASETS[dataset]["checkpoint"]
    payload = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "hbtr_pilot_cache_v1",
            "design_status": "STATIC_SHARED_C1_C4",
        },
        "dataset": dataset,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(checkpoint),
        "split_manifest_sha256": sha256(split_dir / "manifest.json"),
        "samples": len(samples),
        "rank_counts": dict(rank_counts),
        "test_data_read": False,
        "rows": rows,
        "rows_sha256": canonical_cache_sha256(rows),
        "wall_time_seconds": time.time() - started,
    }
    with (output_dir / "negative_cache.json").open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


def compute_batch_loss(
    control: str,
    model,
    collator,
    batch_samples: list[dict],
    cache_by_key: dict[str, dict],
    item2lexid: dict[str, str],
    median_frequency: float,
    device: torch.device,
):
    batch = collate_samples(collator, batch_samples)
    labels = batch["target_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    positive = model(
        input_ids=batch["item_text_ids"].to(device),
        attention_mask=attention,
        labels=labels,
        return_dict=True,
    )
    if control == "C0":
        zero = positive.loss.detach() * 0.0
        return positive.loss, positive.loss.detach(), zero, 0

    eligible = [
        (index, cache_by_key[sample["sample_key"]])
        for index, sample in enumerate(batch_samples)
        if sample["sample_key"] in cache_by_key
    ]
    if not eligible:
        zero = positive.loss * 0.0
        return positive.loss, positive.loss.detach(), zero.detach(), 0

    negative_texts = []
    margins = []
    encoder_rows = []
    attention_rows = []
    for batch_index, row in eligible:
        for negative, depth in zip(row["negative_items"], row["prefix_depths"]):
            negative_texts.append(item2lexid[negative])
            margins.append(
                component_margin(
                    control,
                    depth,
                    row["positive_frequency"],
                    median_frequency,
                )
            )
            encoder_rows.append(positive.encoder_last_hidden_state[batch_index])
            attention_rows.append(attention[batch_index])
    negative_labels = encode_targets(collator, negative_texts).to(device)
    repeated_hidden = torch.stack(encoder_rows)
    repeated_attention = torch.stack(attention_rows)
    negative = model(
        input_ids=None,
        attention_mask=repeated_attention,
        encoder_outputs=(repeated_hidden,),
        labels=negative_labels,
        return_dict=True,
    )
    positive_scores = sequence_log_scores(positive.logits, labels)
    selected_positive = positive_scores[
        torch.tensor([index for index, _ in eligible], device=device)
    ]
    negative_scores = sequence_log_scores(negative.logits, negative_labels).view(
        len(eligible), NEGATIVE_COUNT
    )
    margin_tensor = torch.tensor(
        margins, device=device, dtype=positive_scores.dtype
    ).view(len(eligible), NEGATIVE_COUNT)
    ranking = pairwise_ranking_loss(
        selected_positive, negative_scores, margin_tensor
    )
    return (
        total_loss(positive.loss, ranking, RANKING_LAMBDA),
        positive.loss.detach(),
        ranking.detach(),
        len(eligible),
    )


def train_control(
    dataset: str,
    control: str,
    prepared: dict,
    cache_path: Path,
    output_dir: Path,
    device: torch.device,
):
    split_dir = ROOT / "artifacts/phase3/hbtr_pilot_splits" / dataset
    train_users = read_user_set(split_dir / "train_users.txt")
    samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    cache_rows = load_cache(cache_path, set(prepared["item2lexid"]))
    cache_by_key = {row["sample_key"]: row for row in cache_rows}
    unknown = set(cache_by_key).difference(sample["sample_key"] for sample in samples)
    if unknown:
        raise ValueError(f"cache contains {len(unknown)} unknown training samples")
    median_frequency = float(statistics.median(prepared["popularity"].values()))
    model = prepared["model"]
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    update_steps_per_epoch = math.ceil(
        math.ceil(len(samples) / BATCH_SIZE) / GRADIENT_ACCUMULATION
    )
    total_updates = update_steps_per_epoch * EPOCHS
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * 0.05),
        num_training_steps=total_updates,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    epoch_records = []
    optimizer.zero_grad(set_to_none=True)
    update_count = 0
    for epoch in range(1, EPOCHS + 1):
        indices = list(range(len(samples)))
        random.Random(SEED + epoch).shuffle(indices)
        totals = defaultdict(float)
        batches = math.ceil(len(indices) / BATCH_SIZE)
        for batch_number, start in enumerate(range(0, len(indices), BATCH_SIZE), start=1):
            batch_samples = [samples[index] for index in indices[start : start + BATCH_SIZE]]
            loss, ce, ranking, eligible = compute_batch_loss(
                control,
                model,
                prepared["collator"],
                batch_samples,
                cache_by_key,
                prepared["item2lexid"],
                median_frequency,
                device,
            )
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite pilot loss dataset={dataset} control={control}")
            (loss / GRADIENT_ACCUMULATION).backward()
            should_step = batch_number % GRADIENT_ACCUMULATION == 0 or batch_number == batches
            if should_step:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise ValueError("non-finite pilot gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["ce"] += float(ce.cpu())
            totals["ranking"] += float(ranking.cpu())
            totals["eligible"] += eligible
            if batch_number % 100 == 0:
                print(
                    f"TRAIN_PROGRESS dataset={dataset} control={control} epoch={epoch}/{EPOCHS} "
                    f"batch={batch_number}/{batches} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
        epoch_records.append(
            {
                "epoch": epoch,
                "mean_total_loss": totals["loss"] / batches,
                "mean_token_ce": totals["ce"] / batches,
                "mean_ranking_loss": totals["ranking"] / batches,
                "eligible_sample_occurrences": int(totals["eligible"]),
                "optimizer_updates_cumulative": update_count,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    training_summary = {
        "dataset": dataset,
        "control": control,
        "status": "TRAINED",
        "samples": len(samples),
        "epochs": EPOCHS,
        "optimizer_updates": update_count,
        "epoch_records": epoch_records,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "test_data_read": False,
    }
    with (output_dir / "training_summary.json").open("w") as handle:
        json.dump(training_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return training_summary


def run_preflight(
    dataset: str,
    prepared: dict,
    output_dir: Path,
    device: torch.device,
):
    """Exercise the optimized shared-encoder C0-C4 path without updating weights."""
    b1_cache_path = (
        ROOT / "artifacts/phase3/hbtr_b1_smoke" / dataset / "negative_cache.json"
    )
    rows = load_cache(b1_cache_path, set(prepared["item2lexid"]))
    prefix_rows = [row for row in rows if max(row["prefix_depths"]) > 0]
    tail_rows = [
        row
        for row in rows
        if row["positive_frequency"]
        < statistics.median(prepared["popularity"].values())
    ]
    selected = []
    for pool in (prefix_rows, tail_rows, rows):
        for row in pool:
            if row not in selected:
                selected.append(row)
                break
        if len(selected) == 2:
            break
    if len(selected) < 2:
        raise ValueError("pilot preflight needs two B1 cache rows")
    samples = []
    for row in selected:
        reversed_history = list(reversed(row["history_items"]))
        history_lex = " ; ".join(
            prepared["item2lexid"][item] for item in reversed_history
        )
        samples.append(
            {
                "sample_key": row["sample_key"],
                "user_id": row["user_id"],
                "positive_item": row["positive_item"],
                "history_items": row["history_items"],
                "input": [f"What would user purchase after {history_lex} ?"]
                + [prepared["item2input"][item] for item in reversed_history],
                "output": prepared["item2lexid"][row["positive_item"]],
            }
        )
    cache_by_key = {row["sample_key"]: row for row in selected}
    median_frequency = float(statistics.median(prepared["popularity"].values()))
    records = []
    prepared["model"].train()
    for control in CONTROLS:
        prepared["model"].zero_grad(set_to_none=True)
        loss, ce, ranking, eligible = compute_batch_loss(
            control,
            prepared["model"],
            prepared["collator"],
            samples,
            cache_by_key,
            prepared["item2lexid"],
            median_frequency,
            device,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite preflight loss for {control}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            prepared["model"].parameters(), 1.0
        )
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError(f"invalid preflight gradient for {control}")
        records.append(
            {
                "control": control,
                "total_loss": float(loss.detach().cpu()),
                "token_ce": float(ce.cpu()),
                "ranking_loss": float(ranking.cpu()),
                "eligible_samples": eligible,
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset,
        "status": "PASS",
        "controls": records,
        "weights_updated": False,
        "test_data_read": False,
    }
    with (output_dir / "preflight_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def rank_metrics(rank: int | None) -> dict[str, float]:
    result = {}
    for k in (5, 10):
        hit = float(rank is not None and rank <= k)
        result[f"Recall@{k}"] = hit
        result[f"NDCG@{k}"] = hit / math.log2(rank + 1) if hit else 0.0
    return result


def summarize_metric_rows(values: list[dict]) -> dict[str, float | int | None]:
    metrics = ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10")
    if not values:
        return {metric: None for metric in metrics} | {"n": 0}
    return {
        metric: sum(row[metric] for row in values) / len(values)
        for metric in metrics
    } | {"n": len(values)}


@torch.no_grad()
def validate_control(
    dataset: str,
    control: str,
    prepared: dict,
    output_dir: Path,
    device: torch.device,
    execution_mode: str = "post_training",
):
    split_dir = ROOT / "artifacts/phase3/hbtr_pilot_splits" / dataset
    validation_users = read_user_set(split_dir / "validation_users.txt")
    samples = build_validation_samples(
        prepared["sequences"],
        validation_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    model = prepared["model"]
    model.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(len(candidate) for candidate in prepared["encoded_candidates"])
    heads = head_items(prepared["popularity"])
    rows = []
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        batch = collate_samples(prepared["collator"], [sample])
        prediction = model.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=50,
            num_return_sequences=50,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
        predicted_items = [
            prepared["sequence_to_item"].get(normalized_sequence(ids.tolist()))
            for ids in prediction["sequences"]
        ]
        if any(item is None for item in predicted_items) or len(set(predicted_items)) != 50:
            raise ValueError("pilot validation beam violated locked Trie uniqueness")
        target = sample["positive_item"]
        rank = predicted_items.index(target) + 1 if target in predicted_items else None
        metrics = rank_metrics(rank)
        rows.append(
            {
                "user_id": sample["user_id"],
                "target_item": target,
                "target_group": "head" if target in heads else "tail",
                "history_bin": history_bin(sample["raw_history_length"]),
                "rank": "" if rank is None else rank,
                **metrics,
            }
        )
        if index % 250 == 0:
            print(
                f"VALID_PROGRESS dataset={dataset} control={control} "
                f"users={index}/{len(samples)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = output_dir / "validation_per_user.csv"
    with per_user_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {"overall": summarize_metric_rows(rows)}
    for group_name in ("head", "tail"):
        values = [row for row in rows if row["target_group"] == group_name]
        groups[group_name] = summarize_metric_rows(values)
    for bin_name in ("1-5", "6-10", "11-20", "21+"):
        values = [row for row in rows if row["history_bin"] == bin_name]
        groups[f"history_{bin_name}"] = summarize_metric_rows(values)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "hbtr_pilot_result_v1",
            "design_status": "EXPLORATORY_NO_EFFECT_CLAIM",
        },
        "dataset": dataset,
        "control": control,
        "status": "VALIDATED",
        "execution_mode": execution_mode,
        "groups": groups,
        "users": len(rows),
        "per_user_sha256": sha256(per_user_path),
        "validation_wall_time_seconds": time.time() - started,
        "per_user_latency_seconds": (time.time() - started) / len(rows),
        "test_data_read": False,
        "effect_claim_allowed": False,
    }
    with (output_dir / "validation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "cache", "train", "validate"),
        required=True,
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--control", choices=CONTROLS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/phase3/hbtr_pilot")
    args = parser.parse_args()
    if args.stage in ("train", "validate") and args.control is None:
        parser.error(f"--control is required for {args.stage} stage")
    if args.stage not in ("train", "validate") and args.control is not None:
        parser.error("--control is only valid for train/validate stages")
    if not torch.cuda.is_available():
        raise RuntimeError("HBTR pilot requires CUDA")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    prepared = prepare_dataset(args.dataset, device)
    dataset_root = args.output_root / args.dataset
    cache_dir = dataset_root / "cache"
    if args.stage == "preflight":
        run_preflight(args.dataset, prepared, dataset_root / "preflight", device)
        return 0
    if args.stage == "cache":
        build_cache(args.dataset, prepared, cache_dir, device)
        return 0
    cache_path = cache_dir / "negative_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    output_dir = dataset_root / args.control
    if args.stage == "validate":
        checkpoint = output_dir / "model.pt"
        training_summary_path = output_dir / "training_summary.json"
        if not checkpoint.is_file() or not training_summary_path.is_file():
            raise FileNotFoundError(
                f"resume validation requires {checkpoint} and {training_summary_path}"
            )
        with training_summary_path.open() as handle:
            training_summary = json.load(handle)
        if (
            training_summary.get("dataset") != args.dataset
            or training_summary.get("control") != args.control
            or training_summary.get("status") != "TRAINED"
        ):
            raise ValueError("training summary does not match requested resume target")
        expected_checkpoint_sha = training_summary.get("checkpoint_sha256")
        actual_checkpoint_sha = sha256(checkpoint)
        if expected_checkpoint_sha != actual_checkpoint_sha:
            raise ValueError(
                "resume checkpoint hash mismatch: "
                f"expected={expected_checkpoint_sha} actual={actual_checkpoint_sha}"
            )
        prepared["model"].load_state_dict(
            torch.load(checkpoint, map_location="cpu"), strict=True
        )
        validate_control(
            args.dataset,
            args.control,
            prepared,
            output_dir,
            device,
            execution_mode="resumed_checkpoint_validation",
        )
        return 0
    train_control(
        args.dataset,
        args.control,
        prepared,
        cache_path,
        output_dir,
        device,
    )
    validate_control(
        args.dataset,
        args.control,
        prepared,
        output_dir,
        device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
