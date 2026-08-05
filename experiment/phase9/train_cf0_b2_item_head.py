#!/usr/bin/env python3
"""Train and validate the isolated CF0-B2 collaborative item head."""

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cf0_diagnostic_metrics import item_metrics_from_ranks, rank_from_logits


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "GRAM/rec_datasets/Toys"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/cf0_b2_toys_item_p2a"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--item-index-name",
        default="item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
    )
    parser.add_argument("--dataset-name", default="Toys")
    parser.add_argument(
        "--experiment-id", default="GRAM_PHASE9_CF0_B2_TOYS_ITEM_P2A_V1"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gate-relative-margin", type=float, default=0.20)
    parser.add_argument("--nonhead-recall50-min", type=float, default=0.005)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_sequences(data_dir, item_index_name="item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"):
    item_index = data_dir / item_index_name
    user_sequence = data_dir / "user_sequence.txt"
    raw_items = []
    with item_index.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_items.append(line.split(" ", 1)[0])
    item_to_id = {item: index + 1 for index, item in enumerate(sorted(raw_items))}
    users = []
    sequences = []
    with user_sequence.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            users.append(fields[0])
            sequences.append([item_to_id[item] for item in fields[1:]])
    return users, sequences, item_to_id


def build_splits(users, sequences, max_history):
    train = []
    validation = []
    frequencies = Counter()
    for user, sequence in zip(users, sequences):
        train_sequence = sequence[:-2]
        for index in range(1, len(train_sequence)):
            history = train_sequence[max(0, index - max_history) : index]
            target = train_sequence[index]
            train.append((user, history, target))
        frequencies.update(train_sequence)
        validation.append((user, sequence[max(0, len(sequence) - 2 - max_history) : -2], sequence[-2]))
    return train, validation, frequencies


class SequenceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_sequences(batch):
    max_length = max(len(history) for _, history, _ in batch)
    histories = torch.zeros(len(batch), max_length, dtype=torch.long)
    for row, (_, history, _) in enumerate(batch):
        histories[row, : len(history)] = torch.tensor(history, dtype=torch.long)
    return {
        "users": [user for user, _, _ in batch],
        "history_item_ids": histories,
        "history_item_mask": histories.ne(0),
        "target_item_ids": torch.tensor([target for _, _, target in batch]),
    }


class CF0B2ItemHead(nn.Module):
    def __init__(
        self,
        num_items,
        max_history=20,
        d_model=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        temperature=0.07,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.sequence_norm = nn.LayerNorm(d_model)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, history_item_ids, history_item_mask):
        length = history_item_ids.size(1)
        positions = torch.arange(length, device=history_item_ids.device)
        hidden = self.item_embedding(history_item_ids)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=history_item_ids.device
        ).triu(1)
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~history_item_mask,
        )
        hidden = self.sequence_norm(hidden)
        lengths = history_item_mask.long().sum(dim=1).clamp_min(1)
        user_state = hidden[
            torch.arange(hidden.size(0), device=hidden.device), lengths - 1
        ]
        return F.normalize(user_state, dim=-1)

    def score(self, history_item_ids, history_item_mask):
        users = self.encode(history_item_ids, history_item_mask)
        items = F.normalize(self.item_embedding.weight[1:], dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * F.linear(users, items)

    def forward(self, history_item_ids, history_item_mask, target_item_ids):
        logits = self.score(history_item_ids, history_item_mask)
        return F.cross_entropy(logits, target_item_ids - 1), logits


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def popularity_ranks(validation, frequencies, num_items):
    ordered = sorted(range(1, num_items + 1), key=lambda item: (-frequencies[item], item))
    positions = {item: index + 1 for index, item in enumerate(ordered)}
    return [positions[target] for _, _, target in validation]


def target_frequency_boundaries(validation, frequencies):
    values = sorted(frequencies[target] for _, _, target in validation)
    return values[len(values) // 4], values[(3 * len(values)) // 4]


def stratified_metrics(records, q1, q3):
    history_groups = {
        "1-5": [rank for rank, length, _ in records if 1 <= length <= 5],
        "6-10": [rank for rank, length, _ in records if 6 <= length <= 10],
        "11-20": [rank for rank, length, _ in records if 11 <= length <= 20],
    }
    popularity_groups = {
        "tail": [rank for rank, _, frequency in records if frequency <= q1],
        "middle": [rank for rank, _, frequency in records if q1 < frequency < q3],
        "head": [rank for rank, _, frequency in records if frequency >= q3],
    }
    return {
        "by_history_length": {
            key: item_metrics_from_ranks(value)
            for key, value in history_groups.items()
            if value
        },
        "by_target_popularity": {
            key: item_metrics_from_ranks(value)
            for key, value in popularity_groups.items()
            if value
        },
    }


def evaluate(model, loader, device, frequencies, q1, q3, rank_path=None):
    model.eval()
    records = []
    rank_rows = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            logits = model.score(batch["history_item_ids"], batch["history_item_mask"])
            padded_logits = torch.cat(
                [torch.full((logits.size(0), 1), -torch.inf, device=device), logits],
                dim=1,
            )
            ranks = rank_from_logits(padded_logits, batch["target_item_ids"])
            lengths = batch["history_item_mask"].sum(dim=1)
            for user, target, length, rank in zip(
                batch["users"],
                batch["target_item_ids"].tolist(),
                lengths.tolist(),
                ranks.tolist(),
            ):
                frequency = frequencies[target]
                records.append((rank, length, frequency))
                rank_rows.append((user, target, length, frequency, rank))
    result = {"overall": item_metrics_from_ranks([row[0] for row in records])}
    result.update(stratified_metrics(records, q1, q3))
    if rank_path is not None:
        with rank_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                ["user_id", "target_item_id", "history_length", "train_frequency", "item_rank"]
            )
            writer.writerows(rank_rows)
    return result


def scientific_gate(metrics, baseline, margin, nonhead_recall50_min):
    overall = metrics["overall"]
    groups = metrics["by_target_popularity"]
    nonhead_count = groups["tail"]["count"] + groups["middle"]["count"]
    nonhead_hits = (
        groups["tail"]["Recall@50"] * groups["tail"]["count"]
        + groups["middle"]["Recall@50"] * groups["middle"]["count"]
    )
    nonhead_recall50 = nonhead_hits / nonhead_count
    thresholds = {
        "Recall@10": baseline["Recall@10"] * (1.0 + margin),
        "Recall@50": baseline["Recall@50"] * (1.0 + margin),
        "nonhead_Recall@50": nonhead_recall50_min,
    }
    checks = {
        "Recall@10": overall["Recall@10"] >= thresholds["Recall@10"],
        "Recall@50": overall["Recall@50"] >= thresholds["Recall@50"],
        "nonhead_Recall@50": nonhead_recall50 >= thresholds["nonhead_Recall@50"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "thresholds": thresholds,
        "observed": {
            "Recall@10": overall["Recall@10"],
            "Recall@50": overall["Recall@50"],
            "nonhead_Recall@50": nonhead_recall50,
        },
        "checks": checks,
    }


def scheduler_lambda(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    users, sequences, item_to_id = load_sequences(args.data_dir, args.item_index_name)
    train_samples, validation_samples, frequencies = build_splits(
        users, sequences, args.max_history
    )
    if args.max_train_samples:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_validation_samples:
        validation_samples = validation_samples[: args.max_validation_samples]
    q1, q3 = target_frequency_boundaries(validation_samples, frequencies)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SequenceDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    model = CF0B2ItemHead(
        num_items=len(item_to_id),
        max_history=args.max_history,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_lambda(step, total_steps, warmup_steps),
    )
    baseline = item_metrics_from_ranks(
        popularity_ranks(validation_samples, frequencies, len(item_to_id))
    )
    history = []
    best = None
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        grad_norm_sum = 0.0
        finite = True
        epoch_started = time.perf_counter()
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(
                batch["history_item_ids"],
                batch["history_item_mask"],
                batch["target_item_ids"],
            )
            if not torch.isfinite(loss):
                finite = False
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            count = batch["target_item_ids"].size(0)
            loss_sum += float(loss.detach()) * count
            sample_count += count
            grad_norm_sum += float(grad_norm)
        metrics = evaluate(model, validation_loader, device, frequencies, q1, q3)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "mean_preclip_gradient_norm": grad_norm_sum / len(train_loader),
            "finite": finite,
            "learning_rate_end": scheduler.get_last_lr()[0],
            "wall_time_seconds": time.perf_counter() - epoch_started,
            "validation": metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        score = (metrics["overall"]["Recall@10"], metrics["overall"]["NDCG@10"])
        if best is None or score > best[0]:
            best = (score, epoch, metrics)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "num_items": len(item_to_id),
                        "max_history": args.max_history,
                        "d_model": args.d_model,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                        "temperature_initial": args.temperature,
                    },
                    "epoch": epoch,
                    "metrics": metrics,
                },
                args.output_dir / "best_item_head.pt",
            )

    checkpoint = torch.load(args.output_dir / "best_item_head.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    best_metrics = evaluate(
        model,
        validation_loader,
        device,
        frequencies,
        q1,
        q3,
        rank_path=args.output_dir / "best_validation_ranks.tsv",
    )
    gate = scientific_gate(
        best_metrics, baseline, args.gate_relative_margin, args.nonhead_recall50_min
    )
    summary = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "scientific_gate": gate,
        "dataset": args.dataset_name,
        "split": "validation",
        "seed": args.seed,
        "test_read": False,
        "sports_read": False,
        "catalog_size": len(item_to_id),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "popularity_frequency_boundaries": {"q1": q1, "q3": q3},
        "popularity_baseline": baseline,
        "best_epoch": best[1],
        "best_validation": best_metrics,
        "history": history,
        "wall_time_seconds": time.perf_counter() - started,
        "resource": {
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
        },
        "next_action": (
            "Eligible for separately authorized safe-fusion experiment."
            if gate["status"] == "passed"
            else "Stop before fusion; revise the isolated collaborative objective."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
