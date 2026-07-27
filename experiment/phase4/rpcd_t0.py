#!/usr/bin/env python3
"""RPCD T0: preregistered SASRec teacher complementarity gate.

This program trains no GRAM parameters and never indexes sequence[-1].  It
uses a training-prefix-only split to choose one shared SASRec epoch and a
separate validation-user split to choose one shared hybrid rank weight.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(key: str, salt: str) -> float:
    raw = hashlib.sha256(f"{salt}|{key}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / float(2**64)


def read_sequences(path: Path) -> Dict[str, List[str]]:
    sequences = {}
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                sequences[fields[0]] = fields[1:]
    return sequences


def read_catalog(path: Path) -> List[str]:
    items = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            item = line.split(" ", 1)[0]
            if item in seen:
                raise ValueError(f"duplicate catalog item: {item}")
            seen.add(item)
            items.append(item)
    return items


def lexical_text_to_item(path: Path) -> Dict[str, str]:
    snapshots = sorted(
        (ROOT / ".cache/huggingface/models--t5-small/snapshots").glob("*")
    )
    if len(snapshots) != 1:
        raise ValueError(f"expected one local t5-small snapshot, found {snapshots}")
    tokenizer = AutoTokenizer.from_pretrained(snapshots[0], local_files_only=True)
    result = {}
    with path.open() as handle:
        for line in handle:
            item, raw_identifier = line.rstrip("\n").split(" ", 1)
            token_ids = [
                value
                for value in tokenizer.encode(raw_identifier)
                if value not in (1820, 9175)
            ]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            if text in result:
                raise ValueError(f"non-unique decoded lexical identifier: {text}")
            result[text] = item
    return result


def read_gram_predictions(path: Path, text_to_item: Mapping[str, str]) -> List[dict]:
    rows = []
    with path.open() as handle:
        if not next(handle, "").startswith("idx\t"):
            raise ValueError(f"bad prediction header: {path}")
        for line_number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1 and ": " in fields[0]:
                continue
            if len(fields) < 6:
                raise ValueError(f"bad prediction row {line_number}: {path}")
            user = fields[0]
            gold_text, pred_text, score_text = fields[-3:]
            gold = text_to_item.get(gold_text)
            predictions = [text_to_item.get(value) for value in pred_text.split("||")]
            scores = [float(value) for value in score_text.split("||")]
            if gold is None or any(value is None for value in predictions):
                raise ValueError(f"unmapped lexical identifier at row {line_number}")
            if len(predictions) != len(scores):
                raise ValueError(f"score length mismatch at row {line_number}")
            rows.append(
                {
                    "user": user,
                    "gold": gold,
                    "pred_items": predictions,
                    "scores": scores,
                }
            )
    return rows


def metric(ranking: Sequence[str], gold: str, k: int = 10) -> Tuple[float, float]:
    try:
        rank = ranking[:k].index(gold) + 1
    except ValueError:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1.0)


def mean_metric(records: Iterable[Tuple[Sequence[str], str]], k: int = 10) -> dict:
    recall = ndcg = 0.0
    n = 0
    for ranking, gold in records:
        hit, gain = metric(ranking, gold, k)
        recall += hit
        ndcg += gain
        n += 1
    return {"n": n, f"recall@{k}": recall / n, f"ndcg@{k}": ndcg / n}


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Mapping[str, Sequence[str]],
        item_to_index: Mapping[str, int],
        max_length: int,
        calibration_salt: str,
        calibration_fraction: float,
    ) -> None:
        self.examples = []
        for user, full_sequence in sequences.items():
            train_prefix = list(full_sequence[:-2])
            if len(train_prefix) < 2:
                continue
            is_calibration = stable_fraction(user, calibration_salt) < calibration_fraction
            source = train_prefix[:-1] if is_calibration else train_prefix
            encoded = [item_to_index[item] for item in source if item in item_to_index]
            if len(encoded) >= 2:
                self.examples.append(encoded[-(max_length + 1) :])
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        values = self.examples[index]
        inputs = values[:-1]
        positives = values[1:]
        seq = np.zeros(self.max_length, dtype=np.int64)
        pos = np.zeros(self.max_length, dtype=np.int64)
        # Right padding is required with PyTorch's combined causal/key-padding
        # masks.  Left-padded query rows can have every key masked and produce
        # NaNs in torch 1.11, which then contaminate later Transformer layers.
        seq[: len(inputs)] = inputs
        pos[: len(positives)] = positives
        return seq, pos


class SASRec(nn.Module):
    def __init__(
        self,
        item_count: int,
        hidden_size: int,
        max_length: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.item_embedding = nn.Embedding(item_count + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.max_length, device=sequences.device)
        hidden = self.item_embedding(sequences) + self.position_embedding(positions)[None]
        hidden = self.dropout(hidden)
        padding_mask = sequences.eq(0)
        causal_mask = torch.triu(
            torch.ones(self.max_length, self.max_length, device=sequences.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.final_norm(hidden)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.encode(sequences)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_eval_tensor(
    history: Sequence[str], item_to_index: Mapping[str, int], max_length: int
) -> torch.Tensor:
    encoded = [item_to_index[item] for item in history if item in item_to_index][-max_length:]
    result = torch.zeros(max_length, dtype=torch.long)
    if encoded:
        result[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    return result


@torch.no_grad()
def rank_users(
    model: SASRec,
    samples: Sequence[Tuple[str, Sequence[str], str]],
    item_to_index: Mapping[str, int],
    index_to_item: Sequence[str],
    max_length: int,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> Dict[str, dict]:
    model.eval()
    output = {}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = torch.stack(
            [make_eval_tensor(history, item_to_index, max_length) for _, history, _ in batch]
        ).to(device)
        encoded = model.encode(tensors)
        last_indices = tensors.ne(0).sum(dim=1) - 1
        representation = encoded[
            torch.arange(encoded.shape[0], device=device), last_indices
        ]
        logits = representation @ model.item_embedding.weight[1:].T
        for row_index, (_, history, _) in enumerate(batch):
            seen = {item_to_index[item] for item in history if item in item_to_index}
            if seen:
                columns = torch.tensor([value - 1 for value in seen], device=device)
                logits[row_index, columns] = -torch.inf
        values, indices = torch.topk(logits, k=top_k, dim=1)
        values = values.cpu().numpy()
        indices = indices.cpu().numpy()
        for row_index, (user, _, target) in enumerate(batch):
            items = [index_to_item[value + 1] for value in indices[row_index]]
            output[user] = {
                "items": items,
                "scores": values[row_index].astype(float).tolist(),
                "target": target,
            }
    return output


def internal_calibration_samples(
    sequences: Mapping[str, Sequence[str]], salt: str, fraction: float
) -> List[Tuple[str, Sequence[str], str]]:
    samples = []
    for user, sequence in sequences.items():
        train_prefix = list(sequence[:-2])
        if (
            len(train_prefix) >= 2
            and stable_fraction(user, salt) < fraction
        ):
            samples.append((user, train_prefix[:-1], train_prefix[-1]))
    return samples


def train_dataset(
    dataset: str,
    sequences: Mapping[str, Sequence[str]],
    catalog: Sequence[str],
    teacher: Mapping[str, object],
    seed: int,
    device: torch.device,
) -> Tuple[List[dict], List[dict]]:
    item_to_index = {item: index + 1 for index, item in enumerate(catalog)}
    dataset_object = SequenceDataset(
        sequences,
        item_to_index,
        int(teacher["max_length"]),
        str(teacher["internal_calibration_salt"]),
        float(teacher["internal_calibration_fraction"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset_object,
        batch_size=int(teacher["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = SASRec(
        len(catalog),
        int(teacher["hidden_size"]),
        int(teacher["max_length"]),
        int(teacher["num_blocks"]),
        int(teacher["num_heads"]),
        float(teacher["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(teacher["learning_rate"]),
        weight_decay=float(teacher["weight_decay"]),
    )
    calibration_samples = internal_calibration_samples(
        sequences,
        str(teacher["internal_calibration_salt"]),
        float(teacher["internal_calibration_fraction"]),
    )
    epoch_records = []
    states = []
    for epoch in range(1, int(teacher["epochs"]) + 1):
        model.train()
        loss_sum = valid_count = 0
        for seq_np, pos_np in loader:
            seq = seq_np.to(device)
            positives = pos_np.to(device)
            hidden = model(seq)
            negatives = torch.randint(
                1, len(catalog) + 1, positives.shape, device=device
            )
            collision = negatives.eq(positives) & positives.ne(0)
            while collision.any():
                negatives[collision] = torch.randint(
                    1, len(catalog) + 1, (int(collision.sum().item()),), device=device
                )
                collision = negatives.eq(positives) & positives.ne(0)
            positive_logits = (hidden * model.item_embedding(positives)).sum(-1)
            negative_logits = (hidden * model.item_embedding(negatives)).sum(-1)
            valid = positives.ne(0)
            loss = (
                nn.functional.softplus(-positive_logits[valid]).mean()
                + nn.functional.softplus(negative_logits[valid]).mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{dataset} epoch {epoch}: non-finite SASRec loss"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(valid.sum().item())
            valid_count += int(valid.sum().item())
        ranked = rank_users(
            model,
            calibration_samples,
            item_to_index,
            ["<padding>"] + list(catalog),
            int(teacher["max_length"]),
            int(teacher["batch_size"]),
            int(teacher["top_k"]),
            device,
        )
        metrics = mean_metric(
            ((ranked[user]["items"], target) for user, _, target in calibration_samples), 10
        )
        record = {
            "dataset": dataset,
            "epoch": epoch,
            "training_loss": loss_sum / valid_count,
            "internal_calibration_n": metrics["n"],
            "internal_calibration_recall@10": metrics["recall@10"],
            "internal_calibration_ndcg@10": metrics["ndcg@10"],
        }
        if not all(
            math.isfinite(float(record[key]))
            for key in (
                "training_loss",
                "internal_calibration_recall@10",
                "internal_calibration_ndcg@10",
            )
        ):
            raise FloatingPointError(
                f"{dataset} epoch {epoch}: non-finite training/evaluation metric"
            )
        print(json.dumps(record), flush=True)
        epoch_records.append(record)
        states.append(copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
    return epoch_records, states


def deduplicate(values: Sequence[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def fuse(gram: Sequence[str], sasrec: Sequence[str], weight: float) -> List[str]:
    gram = deduplicate(gram)
    sasrec = deduplicate(sasrec)
    gram_score = {
        item: 1.0 / math.log2(rank + 1.0) for rank, item in enumerate(gram, 1)
    }
    sas_score = {
        item: 1.0 / math.log2(rank + 1.0) for rank, item in enumerate(sasrec, 1)
    }
    union = deduplicate(list(gram) + list(sasrec))
    original = {item: index for index, item in enumerate(union)}
    return sorted(
        union,
        key=lambda item: (
            -((1.0 - weight) * gram_score.get(item, 0.0) + weight * sas_score.get(item, 0.0)),
            original[item],
        ),
    )


def popularity_tail(sequences: Mapping[str, Sequence[str]]) -> set:
    counts = Counter(item for sequence in sequences.values() for item in sequence[:-2])
    ordered = sorted(counts, key=lambda item: (-counts[item], item))
    head_count = max(1, math.ceil(len(ordered) * 0.2))
    return set(ordered[head_count:])


def evaluate_partition(
    rows: Sequence[dict],
    teacher_rows: Mapping[str, dict],
    users: set,
    weight: float,
    tail_items: set,
) -> dict:
    selected = [row for row in rows if row["user"] in users]
    gram_records = []
    sas_records = []
    hybrid_records = []
    tail_gram = []
    tail_hybrid = []
    union_hits = 0
    miss_count = miss_sas_hit = 0
    for row in selected:
        gram = deduplicate(row["pred_items"])
        sasrec = teacher_rows[row["user"]]["items"]
        hybrid = fuse(gram, sasrec, weight)
        gold = row["gold"]
        gram_records.append((gram, gold))
        sas_records.append((sasrec, gold))
        hybrid_records.append((hybrid, gold))
        union_hits += int(gold in set(gram[:50]) | set(sasrec[:50]))
        gram_hit10, _ = metric(gram, gold, 10)
        if not gram_hit10:
            miss_count += 1
            miss_sas_hit += int(gold in sasrec[:50])
        if gold in tail_items:
            tail_gram.append((gram, gold))
            tail_hybrid.append((hybrid, gold))
    gram_metrics = mean_metric(gram_records, 10)
    gram50 = mean_metric(gram_records, 50)
    sas_metrics = mean_metric(sas_records, 10)
    sas50 = mean_metric(sas_records, 50)
    hybrid_metrics = mean_metric(hybrid_records, 10)
    tail_gram_metrics = mean_metric(tail_gram, 10)
    tail_hybrid_metrics = mean_metric(tail_hybrid, 10)
    return {
        "n": len(selected),
        "gram": {**gram_metrics, "recall@50": gram50["recall@50"]},
        "sasrec": {**sas_metrics, "recall@50": sas50["recall@50"]},
        "hybrid": hybrid_metrics,
        "union_recall@50": union_hits / len(selected),
        "union_recall50_absolute_gain": union_hits / len(selected) - gram50["recall@50"],
        "gram_miss10_n": miss_count,
        "gram_miss10_sasrec_hit50_rate": miss_sas_hit / miss_count,
        "hybrid_ndcg10_relative_gain": (
            hybrid_metrics["ndcg@10"] / gram_metrics["ndcg@10"] - 1.0
        ),
        "hybrid_recall10_absolute_gain": (
            hybrid_metrics["recall@10"] - gram_metrics["recall@10"]
        ),
        "tail_n": tail_gram_metrics["n"],
        "tail_gram_ndcg@10": tail_gram_metrics["ndcg@10"],
        "tail_hybrid_ndcg@10": tail_hybrid_metrics["ndcg@10"],
        "tail_ndcg10_relative_gain": (
            tail_hybrid_metrics["ndcg@10"] / tail_gram_metrics["ndcg@10"] - 1.0
        ),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def resolve_inputs(config: Mapping[str, object]) -> Dict[str, dict]:
    resolved = {}
    for dataset, spec in config["datasets"].items():
        data_dir = ROOT / "GRAM/rec_datasets" / dataset
        index_paths = list(data_dir.glob(spec["index_glob"]))
        if len(index_paths) != 1:
            raise ValueError(f"{dataset}: expected one index, found {index_paths}")
        resolved[dataset] = {
            "sequence": data_dir / "user_sequence.txt",
            "index": index_paths[0],
            "prediction": ROOT / spec["prediction_path"],
        }
        for path in resolved[dataset].values():
            if not path.is_file():
                raise FileNotFoundError(path)
    return resolved


def load_dataset(paths: Mapping[str, Path]) -> dict:
    sequences = read_sequences(paths["sequence"])
    catalog = read_catalog(paths["index"])
    text_to_item = lexical_text_to_item(paths["index"])
    rows = read_gram_predictions(paths["prediction"], text_to_item)
    if len(rows) != len(sequences):
        raise ValueError(f"user count mismatch: predictions={len(rows)} sequences={len(sequences)}")
    row_users = {row["user"] for row in rows}
    if row_users != set(sequences):
        raise ValueError("prediction/sequence user set mismatch")
    mismatches = [
        row["user"] for row in rows if row["gold"] != sequences[row["user"]][-2]
    ]
    if mismatches:
        raise ValueError(f"validation target mismatch: {mismatches[:3]}")
    catalog_set = set(catalog)
    if any(item not in catalog_set for sequence in sequences.values() for item in sequence):
        raise ValueError("sequence item outside catalog")
    return {"sequences": sequences, "catalog": catalog, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seed_everything(int(config["seed"]))
    resolved = resolve_inputs(config)
    loaded = {}
    preflight = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(args.config),
        "test_predictions_read": False,
        "sequence_test_target_indexed": False,
        "datasets": {},
    }
    for dataset, paths in resolved.items():
        data = load_dataset(paths)
        loaded[dataset] = data
        teacher_cal = sum(
            stable_fraction(user, config["teacher"]["internal_calibration_salt"])
            < config["teacher"]["internal_calibration_fraction"]
            for user in data["sequences"]
        )
        hybrid_cal = sum(
            stable_fraction(user, config["hybrid"]["calibration_salt"])
            < config["hybrid"]["calibration_fraction"]
            for user in data["sequences"]
        )
        preflight["datasets"][dataset] = {
            "users": len(data["sequences"]),
            "catalog_items": len(data["catalog"]),
            "prediction_rows": len(data["rows"]),
            "target_match_rate": 1.0,
            "teacher_internal_calibration_users": teacher_cal,
            "hybrid_calibration_users": hybrid_cal,
            "hybrid_audit_users": len(data["sequences"]) - hybrid_cal,
            "input_sha256": {name: sha256(path) for name, path in paths.items()},
        }
    write_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal T0 requires CUDA; run preflight-only on CPU")
    epoch_metrics = {}
    epoch_states = {}
    for dataset in config["datasets"]:
        records, states = train_dataset(
            dataset,
            loaded[dataset]["sequences"],
            loaded[dataset]["catalog"],
            config["teacher"],
            int(config["seed"]),
            device,
        )
        epoch_metrics[dataset] = records
        epoch_states[dataset] = states
    epoch_grid = []
    for epoch in range(1, int(config["teacher"]["epochs"]) + 1):
        values = [
            epoch_metrics[dataset][epoch - 1]["internal_calibration_ndcg@10"]
            for dataset in config["datasets"]
        ]
        epoch_grid.append({"epoch": epoch, "macro_ndcg@10": sum(values) / len(values)})
    selected_epoch = max(epoch_grid, key=lambda row: (row["macro_ndcg@10"], -row["epoch"]))[
        "epoch"
    ]
    teacher_outputs = {}
    for dataset in config["datasets"]:
        data = loaded[dataset]
        catalog = data["catalog"]
        model = SASRec(
            len(catalog),
            int(config["teacher"]["hidden_size"]),
            int(config["teacher"]["max_length"]),
            int(config["teacher"]["num_blocks"]),
            int(config["teacher"]["num_heads"]),
            float(config["teacher"]["dropout"]),
        ).to(device)
        model.load_state_dict(epoch_states[dataset][selected_epoch - 1])
        item_to_index = {item: index + 1 for index, item in enumerate(catalog)}
        samples = [
            (user, sequence[:-2], sequence[-2])
            for user, sequence in data["sequences"].items()
        ]
        teacher_outputs[dataset] = rank_users(
            model,
            samples,
            item_to_index,
            ["<padding>"] + list(catalog),
            int(config["teacher"]["max_length"]),
            int(config["teacher"]["batch_size"]),
            int(config["teacher"]["top_k"]),
            device,
        )
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "catalog": catalog,
                "selected_shared_epoch": selected_epoch,
                "config_sha256": preflight["config_sha256"],
            },
            args.output_dir / f"sasrec_{dataset}_epoch{selected_epoch}.pt",
        )
        with (args.output_dir / f"teacher_top50_{dataset}.jsonl").open("w") as handle:
            for user in sorted(teacher_outputs[dataset]):
                handle.write(
                    json.dumps({"user": user, **teacher_outputs[dataset][user]}) + "\n"
                )
    weight_grid = []
    for weight in config["hybrid"]["weights"]:
        per_dataset = {}
        eligible = True
        relative_ndcgs = []
        for dataset in config["datasets"]:
            users = {
                user
                for user in loaded[dataset]["sequences"]
                if stable_fraction(user, config["hybrid"]["calibration_salt"])
                < config["hybrid"]["calibration_fraction"]
            }
            result = evaluate_partition(
                loaded[dataset]["rows"],
                teacher_outputs[dataset],
                users,
                float(weight),
                popularity_tail(loaded[dataset]["sequences"]),
            )
            per_dataset[dataset] = result
            relative_ndcgs.append(result["hybrid_ndcg10_relative_gain"])
            eligible &= result["hybrid_recall10_absolute_gain"] >= 0.0
        weight_grid.append(
            {
                "weight": weight,
                "eligible": eligible,
                "macro_relative_ndcg@10": sum(relative_ndcgs) / len(relative_ndcgs),
                "datasets": per_dataset,
            }
        )
    eligible_weights = [row for row in weight_grid if row["eligible"]]
    if not eligible_weights:
        selected_weight = 0.0
    else:
        selected_weight = max(
            eligible_weights,
            key=lambda row: (row["macro_relative_ndcg@10"], -row["weight"]),
        )["weight"]
    audit = {}
    gate_rows = []
    gates = config["gates"]
    for dataset in config["datasets"]:
        users = {
            user
            for user in loaded[dataset]["sequences"]
            if stable_fraction(user, config["hybrid"]["calibration_salt"])
            >= config["hybrid"]["calibration_fraction"]
        }
        result = evaluate_partition(
            loaded[dataset]["rows"],
            teacher_outputs[dataset],
            users,
            float(selected_weight),
            popularity_tail(loaded[dataset]["sequences"]),
        )
        audit[dataset] = result
        checks = {
            "union_recall50_absolute_gain": result["union_recall50_absolute_gain"]
            >= gates["union_recall50_absolute_gain_min"],
            "gram_miss10_sasrec_hit50_rate": result["gram_miss10_sasrec_hit50_rate"]
            >= gates["gram_miss10_sasrec_hit50_min"],
            "hybrid_ndcg10_relative_gain": result["hybrid_ndcg10_relative_gain"]
            >= gates["hybrid_ndcg10_relative_gain_min"],
            "hybrid_recall10_absolute_gain": result["hybrid_recall10_absolute_gain"]
            >= gates["hybrid_recall10_absolute_gain_min"],
            "tail_ndcg10_relative_gain": result["tail_ndcg10_relative_gain"]
            >= gates["tail_ndcg10_relative_gain_min"],
        }
        gate_rows.append({"dataset": dataset, "checks": checks, "pass": all(checks.values())})
    passed = all(row["pass"] for row in gate_rows)
    decision = (
        "RPCD_T1_DESIGN_ALLOWED" if passed else "STOP_RPCD_NO_TEACHER_COMPLEMENTARITY"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "selected_shared_epoch": selected_epoch,
        "selected_shared_weight": selected_weight,
        "epoch_grid": epoch_grid,
        "teacher_epoch_metrics": epoch_metrics,
        "weight_grid": weight_grid,
        "audit": audit,
        "gate_rows": gate_rows,
        "integrity": {
            "preflight_passed": True,
            "test_predictions_read": False,
            "sequence_test_target_indexed": False,
            "target_match_rate": 1.0,
            "shared_epoch": True,
            "shared_weight": True,
        },
        "elapsed_seconds": time.time() - args.output_dir.stat().st_mtime,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"decision": decision, "selected_epoch": selected_epoch, "selected_weight": selected_weight, "audit": audit}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
