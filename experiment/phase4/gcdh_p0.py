#!/usr/bin/env python3
"""GCDH P0: matched 25% continuation with a full-catalog balanced item head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = ROOT / "GRAM/src"
PHASE3 = ROOT / "experiment/phase3"
for path in (GRAM_SRC, PHASE3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from utils import indexing  # noqa: E402

from hbtr_b1_smoke import (  # noqa: E402
    create_model_and_tokenizer,
    encode_candidates,
    make_runtime_args,
    normalized_sequence,
    read_sequences,
    sha256,
)


def read_users(path: Path) -> set[str]:
    return {line.strip() for line in path.open() if line.strip()}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stable_sha(values: list[str] | set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def build_train_samples(
    sequences: dict[str, list[str]],
    users: set[str],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
) -> list[dict]:
    samples = []
    for user in sorted(users):
        items = sequences[user][:-2]
        for index in range(1, len(items)):
            history = items[:index][-20:]
            target = items[index]
            if target not in item2lexid or any(item not in item2input for item in history):
                continue
            reverse = list(reversed(history))
            history_lex = " ; ".join(item2lexid[item] for item in reverse)
            samples.append(
                {
                    "sample_key": f"{user}:{index}:{target}",
                    "user_id": user,
                    "positive_item": target,
                    "history_items": history,
                    "input": [f"What would user purchase after {history_lex} ?"]
                    + [item2input[item] for item in reverse],
                    "output": item2lexid[target],
                }
            )
    if len({row["sample_key"] for row in samples}) != len(samples):
        raise ValueError("duplicate training sample")
    return samples


def build_validation_samples(
    sequences: dict[str, list[str]],
    users: set[str],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
) -> list[dict]:
    samples = []
    for user in sorted(users):
        items = sequences[user]
        target = items[-2]
        raw_history = items[:-2]
        history = raw_history[-20:]
        reverse = list(reversed(history))
        history_lex = " ; ".join(item2lexid[item] for item in reverse)
        samples.append(
            {
                "sample_key": f"{user}:validation:{target}",
                "user_id": user,
                "positive_item": target,
                "history_items": history,
                "raw_history_length": len(raw_history),
                "input": [f"What would user purchase after {history_lex} ?"]
                + [item2input[item] for item in reverse],
                "output": item2lexid[target],
            }
        )
    return samples


def collate(collator: CollatorGRAM, samples: list[dict]) -> dict:
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


def training_popularity(sequences: dict[str, list[str]]) -> Counter:
    counts = Counter()
    for items in sequences.values():
        counts.update(items[:-2])
    return counts


def head_items(popularity: Counter) -> set[str]:
    ordered = sorted(popularity, key=lambda item: (-popularity[item], item))
    return set(ordered[: max(1, math.ceil(len(ordered) * 0.2))])


class CatalogDualHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        catalog: list[str],
        item2lexid: dict[str, str],
        log_counts: torch.Tensor,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.catalog = catalog
        self.item_to_index = {item: index for index, item in enumerate(catalog)}
        self.catalog_head = nn.Linear(backbone.config.d_model, len(catalog), bias=True)
        embedding_cpu = backbone.shared.weight.detach().cpu()
        with torch.no_grad():
            for index, item in enumerate(catalog):
                token_ids = [
                    token
                    for token in tokenizer.encode(item2lexid[item])
                    if token not in (0, 1, 1820, 9175)
                ]
                if not token_ids:
                    raise ValueError(f"empty lexical initialization for {item}")
                ids = torch.tensor(token_ids)
                self.catalog_head.weight[index].copy_(
                    embedding_cpu[ids].mean(dim=0)
                )
            self.catalog_head.bias.zero_()
        self.register_buffer("log_counts", log_counts)

    @staticmethod
    def pool_coarse(
        hidden: torch.Tensor, attention_mask: torch.Tensor, passage_width: int
    ) -> torch.Tensor:
        coarse_hidden = hidden[:, :passage_width, :]
        coarse_mask = attention_mask[:, 0, :passage_width].to(hidden.dtype)
        denominator = coarse_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (coarse_hidden * coarse_mask.unsqueeze(-1)).sum(dim=1) / denominator

    def forward_loss(
        self,
        batch: dict,
        target_indices: torch.Tensor,
        lambda_item: float,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        input_ids = batch["item_text_ids"]
        attention = batch["item_text_masks"]
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention,
            labels=batch["target_ids"],
            return_dict=True,
        )
        pooled = self.pool_coarse(
            output.encoder_last_hidden_state, attention, input_ids.shape[-1]
        )
        logits = self.catalog_head(pooled)
        balanced = F.cross_entropy(logits + self.log_counts[None, :], target_indices)
        total = output.loss + float(lambda_item) * balanced
        return total, {
            "token_ce": output.loss,
            "item_balanced_ce": balanced,
        }, logits

    @torch.no_grad()
    def catalog_logits(self, input_ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        self.backbone.encoder.n_passages = input_ids.size(1)
        flat_ids = input_ids.view(input_ids.size(0), -1)
        flat_attention = attention.view(attention.size(0), -1)
        hidden = self.backbone.encoder(
            input_ids=flat_ids,
            attention_mask=flat_attention,
            return_dict=True,
        )[0]
        pooled = self.pool_coarse(hidden, attention, input_ids.shape[-1])
        return self.catalog_head(pooled)


def prepare(dataset: str, config: dict, device: torch.device) -> dict:
    # Reuse the already verified baseline constructor and override only its path constants.
    from hbtr_b1_smoke import DATASETS

    spec = config["datasets"][dataset]
    DATASETS[dataset]["checkpoint"] = ROOT / spec["checkpoint"]
    DATASETS[dataset]["hierarchical_id_type"] = spec["hierarchical_id_type"]
    DATASETS[dataset]["top_k_similar_item"] = int(spec["top_k_similar_item"])
    backbone, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
    runtime = make_runtime_args(dataset)
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
    catalog = list(item2lexid)
    popularity = training_popularity(sequences)
    smoothed = torch.tensor(
        [
            popularity.get(item, 0)
            + float(config["training"]["balanced_softmax_smoothing"])
            for item in catalog
        ],
        dtype=torch.float32,
        device=device,
    )
    model = CatalogDualHead(
        backbone,
        tokenizer,
        catalog,
        item2lexid,
        torch.log(smoothed),
    ).to(device)
    encoded_candidates, sequence_to_item = encode_candidates(tokenizer, item2lexid)
    return {
        "runtime": runtime,
        "model": model,
        "tokenizer": tokenizer,
        "sequences": sequences,
        "item2input": item2input,
        "item2lexid": item2lexid,
        "catalog": catalog,
        "collator": CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train"),
        "encoded_candidates": encoded_candidates,
        "sequence_to_item": sequence_to_item,
        "popularity": popularity,
        "heads": head_items(popularity),
    }


def target_indices(model: CatalogDualHead, samples: list[dict], device: torch.device):
    return torch.tensor(
        [model.item_to_index[row["positive_item"]] for row in samples],
        dtype=torch.long,
        device=device,
    )


def run_smoke(
    dataset: str, prepared: dict, config: dict, output_dir: Path, device: torch.device
) -> dict:
    train_users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "train_users.txt"
    )
    samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )[:2]
    batch = collate(prepared["collator"], samples)
    for key in ("item_text_ids", "item_text_masks", "target_ids"):
        batch[key] = batch[key].to(device)
    model = prepared["model"]
    model.train()
    loss, components, logits = model.forward_loss(
        batch,
        target_indices(model, samples, device),
        float(config["training"]["lambda_item"]),
    )
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise ValueError("non-finite smoke loss/logits")
    loss.backward()
    gradient = model.catalog_head.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or float(gradient.norm()) <= 0:
        raise ValueError("catalog head gradient missing/non-finite/zero")
    model.zero_grad(set_to_none=True)
    model.eval()
    validation_users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "validation_users.txt"
    )
    sample = build_validation_samples(
        prepared["sequences"],
        set(sorted(validation_users)[:1]),
        prepared["item2input"],
        prepared["item2lexid"],
    )[0]
    one = collate(prepared["collator"], [sample])
    input_ids = one["item_text_ids"].to(device)
    attention = one["item_text_masks"].to(device)
    with torch.no_grad():
        before = model.catalog_logits(input_ids, attention)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(model.state_dict(), handle.name)
        with torch.no_grad():
            model.catalog_head.bias.add_(1.0)
        model.load_state_dict(torch.load(handle.name, map_location=device), strict=True)
    with torch.no_grad():
        after = model.catalog_logits(input_ids, attention)
    reload_difference = float((before - after).abs().max())
    if reload_difference != 0.0:
        raise ValueError(f"checkpoint reload difference={reload_difference}")
    trie = gt.Trie(prepared["encoded_candidates"])
    prediction = model.backbone.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_length=max(len(row) for row in prepared["encoded_candidates"]),
        prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(trie),
        num_beams=50,
        num_return_sequences=50,
        return_dict_in_generate=True,
        length_penalty=1.0,
    )
    generated = [
        prepared["sequence_to_item"].get(normalized_sequence(row.tolist()))
        for row in prediction["sequences"]
    ]
    if any(item is None for item in generated) or len(set(generated)) != 50:
        raise ValueError("smoke constrained beam mapping failure")
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "status": "PASS",
        "samples": len(samples),
        "loss": float(loss.detach().cpu()),
        "token_ce": float(components["token_ce"].detach().cpu()),
        "item_balanced_ce": float(components["item_balanced_ce"].detach().cpu()),
        "catalog_gradient_norm": float(gradient.norm().detach().cpu()),
        "checkpoint_reload_max_abs_difference": reload_difference,
        "beam_items": len(generated),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "test_data_read": False,
    }
    write_json(output_dir / "smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def train(
    dataset: str,
    control: str,
    prepared: dict,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "train_users.txt"
    )
    samples = build_train_samples(
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    training = config["training"]
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation"])
    epochs = int(training["epochs"])
    batches = math.ceil(len(samples) / batch_size)
    updates_per_epoch = math.ceil(batches / accumulation)
    total_updates = updates_per_epoch * epochs
    model = prepared["model"]
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(training["warmup_fraction"])),
        num_training_steps=total_updates,
    )
    lambda_item = 0.0 if control == "C0" else float(training["lambda_item"])
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    update_count = 0
    epoch_rows = []
    for epoch in range(1, epochs + 1):
        indices = list(range(len(samples)))
        random.Random(int(config["seed"]) + epoch).shuffle(indices)
        totals = defaultdict(float)
        for batch_number, start in enumerate(range(0, len(indices), batch_size), 1):
            rows = [samples[index] for index in indices[start : start + batch_size]]
            batch = collate(prepared["collator"], rows)
            for key in ("item_text_ids", "item_text_masks", "target_ids"):
                batch[key] = batch[key].to(device)
            loss, components, logits = model.forward_loss(
                batch, target_indices(model, rows, device), lambda_item
            )
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise ValueError(
                    f"non-finite train output dataset={dataset} control={control}"
                )
            (loss / accumulation).backward()
            should_step = batch_number % accumulation == 0 or batch_number == batches
            if should_step:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                if not torch.isfinite(norm):
                    raise ValueError("non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["token_ce"] += float(components["token_ce"].detach().cpu())
            totals["item_ce"] += float(components["item_balanced_ce"].detach().cpu())
            if batch_number % 100 == 0:
                print(
                    f"TRAIN_PROGRESS dataset={dataset} control={control} "
                    f"epoch={epoch}/{epochs} batch={batch_number}/{batches} "
                    f"updates={update_count}/{total_updates} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
        epoch_rows.append(
            {
                "epoch": epoch,
                "mean_total_loss": totals["loss"] / batches,
                "mean_token_ce": totals["token_ce"] / batches,
                "mean_item_balanced_ce": totals["item_ce"] / batches,
                "optimizer_updates_cumulative": update_count,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "dataset": dataset,
        "control": control,
        "status": "TRAINED",
        "train_users": len(users),
        "train_user_sha256": stable_sha(users),
        "samples": len(samples),
        "epochs": epochs,
        "batches_per_epoch": batches,
        "optimizer_updates": update_count,
        "lambda_item": lambda_item,
        "epoch_records": epoch_rows,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "test_data_read": False,
    }
    write_json(output_dir / "training_summary.json", result)
    return result


def rank_metrics(rank: int | None) -> dict[str, float]:
    return {
        "Recall@10": float(rank is not None and rank <= 10),
        "NDCG@10": (
            1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
        ),
        "Recall@50": float(rank is not None and rank <= 50),
    }


@torch.no_grad()
def validate(
    dataset: str,
    control: str,
    prepared: dict,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "validation_users.txt"
    )
    samples = build_validation_samples(
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    model = prepared["model"]
    model.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    prefix_fn = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(len(row) for row in prepared["encoded_candidates"])
    catalog = prepared["catalog"]
    rows = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        batch = collate(prepared["collator"], [sample])
        input_ids = batch["item_text_ids"].to(device)
        attention = batch["item_text_masks"].to(device)
        prediction = model.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_fn,
            num_beams=50,
            num_return_sequences=50,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
        gram = [
            prepared["sequence_to_item"].get(normalized_sequence(value.tolist()))
            for value in prediction["sequences"]
        ]
        if any(item is None for item in gram) or len(set(gram)) != 50:
            raise ValueError("validation constrained beam mapping failure")
        logits = model.catalog_logits(input_ids, attention)[0]
        for item in sample["history_items"]:
            if item in model.item_to_index:
                logits[model.item_to_index[item]] = -torch.inf
        catalog_indices = torch.topk(logits, k=50).indices.tolist()
        catalog_top50 = [catalog[value] for value in catalog_indices]
        union = list(dict.fromkeys(gram + catalog_top50))
        stable = {item: order for order, item in enumerate(union)}
        final = sorted(
            union,
            key=lambda item: (
                -float(logits[model.item_to_index[item]].cpu()),
                stable[item],
            ),
        )
        target = sample["positive_item"]
        gram_rank = gram.index(target) + 1 if target in gram else None
        head_rank = catalog_top50.index(target) + 1 if target in catalog_top50 else None
        final_rank = final.index(target) + 1 if target in final else None
        rows.append(
            {
                "user_id": sample["user_id"],
                "target_item": target,
                "target_group": "head" if target in prepared["heads"] else "tail",
                "gram_rank": "" if gram_rank is None else gram_rank,
                "catalog_rank": "" if head_rank is None else head_rank,
                "final_rank": "" if final_rank is None else final_rank,
                "union_hit50": int(target in set(gram) | set(catalog_top50)),
                **{f"gram_{key}": value for key, value in rank_metrics(gram_rank).items()},
                **{f"catalog_{key}": value for key, value in rank_metrics(head_rank).items()},
                **{f"final_{key}": value for key, value in rank_metrics(final_rank).items()},
            }
        )
        if index % 250 == 0:
            print(
                f"VALID_PROGRESS dataset={dataset} control={control} "
                f"users={index}/{len(samples)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user = output_dir / "validation_per_user.csv"
    with per_user.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(selected: list[dict]) -> dict:
        fields = [
            "gram_Recall@10",
            "gram_NDCG@10",
            "gram_Recall@50",
            "catalog_Recall@10",
            "catalog_NDCG@10",
            "catalog_Recall@50",
            "final_Recall@10",
            "final_NDCG@10",
            "final_Recall@50",
            "union_hit50",
        ]
        return {"n": len(selected)} | {
            field: sum(float(row[field]) for row in selected) / len(selected)
            for field in fields
        }

    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize([row for row in rows if row["target_group"] == group])
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "control": control,
        "status": "VALIDATED",
        "users": len(rows),
        "validation_user_sha256": stable_sha(users),
        "groups": groups,
        "per_user_sha256": sha256(per_user),
        "validation_wall_time_seconds": time.time() - started,
        "per_user_latency_seconds": (time.time() - started) / len(rows),
        "candidate_mapping_rate": 1.0,
        "test_data_read": False,
    }
    write_json(output_dir / "validation_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "train", "validate"), required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    parser.add_argument("--control", choices=("C0", "C1"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ("train", "validate") and args.control is None:
        parser.error("--control required")
    if args.stage == "smoke" and args.control is not None:
        parser.error("--control forbidden for smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("GCDH P0 requires CUDA")
    config = json.loads(args.config.read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    prepared = prepare(args.dataset, config, device)
    dataset_root = args.output_root / args.dataset
    if args.stage == "smoke":
        run_smoke(args.dataset, prepared, config, dataset_root / "smoke", device)
        return 0
    output_dir = dataset_root / args.control
    if args.stage == "validate":
        summary_path = output_dir / "training_summary.json"
        checkpoint = output_dir / "model.pt"
        if not summary_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("validation resume materials missing")
        summary = json.loads(summary_path.read_text())
        if summary["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("checkpoint hash mismatch")
        prepared["model"].load_state_dict(
            torch.load(checkpoint, map_location=device), strict=True
        )
        validate(args.dataset, args.control, prepared, config, output_dir, device)
        return 0
    train(args.dataset, args.control, prepared, config, output_dir, device)
    validate(args.dataset, args.control, prepared, config, output_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
