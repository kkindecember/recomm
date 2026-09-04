"""Train-prefix-only SASRec feature tokenizer for Stage17 FP3 Full SETRec."""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .fullport_data import FullportExample


@dataclass(frozen=True)
class SetRecCFSpec:
    seed: int = 2023
    hidden_size: int = 64
    max_history_items: int = 20
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    eval_batch_size: int = 256
    top_k: int = 10


class RollingNextItemDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """One exact rolling target per example; zero is reserved for padding."""

    def __init__(
        self,
        examples: Sequence[FullportExample],
        item_to_index: Mapping[str, int],
        max_history_items: int,
    ) -> None:
        self.rows: list[tuple[tuple[int, ...], int]] = []
        self.max_history_items = max_history_items
        for example in examples:
            history = tuple(
                item_to_index[item]
                for item in example.history[-max_history_items:]
            )
            if not history:
                raise ValueError("SASRec rolling example has empty history")
            self.rows.append((history, item_to_index[example.target]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target = self.rows[index]
        sequence = torch.zeros(self.max_history_items, dtype=torch.long)
        sequence[: len(history)] = torch.tensor(history, dtype=torch.long)
        return sequence, torch.tensor(len(history) - 1), torch.tensor(target)


class SetRecSASRec(nn.Module):
    def __init__(self, item_count: int, spec: SetRecCFSpec) -> None:
        super().__init__()
        self.max_history_items = spec.max_history_items
        self.item_embedding = nn.Embedding(
            item_count + 1, spec.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            spec.max_history_items, spec.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden_size,
            nhead=spec.num_heads,
            dim_feedforward=spec.hidden_size * 4,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.num_blocks)
        self.dropout = nn.Dropout(spec.dropout)
        self.final_norm = nn.LayerNorm(spec.hidden_size)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            self.max_history_items, device=sequences.device
        )
        hidden = self.item_embedding(sequences)
        hidden = hidden + self.position_embedding(positions)[None]
        hidden = self.dropout(hidden)
        causal = torch.triu(
            torch.ones(
                self.max_history_items,
                self.max_history_items,
                device=sequences.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal,
            src_key_padding_mask=sequences.eq(0),
        )
        return self.final_norm(encoded)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sampled_bce_loss(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    model: SetRecSASRec,
    train_item_ids: torch.Tensor,
) -> torch.Tensor:
    sampled_rows = torch.randint(
        0, train_item_ids.numel(), targets.shape, device=targets.device
    )
    negatives = train_item_ids[sampled_rows]
    collisions = negatives.eq(targets)
    while bool(collisions.any()):
        replacement = torch.randint(
            0,
            train_item_ids.numel(),
            (int(collisions.sum()),),
            device=targets.device,
        )
        negatives[collisions] = train_item_ids[replacement]
        collisions = negatives.eq(targets)
    positive_logits = (hidden * model.item_embedding(targets)).sum(-1)
    negative_logits = (hidden * model.item_embedding(negatives)).sum(-1)
    return nn.functional.softplus(-positive_logits).mean() + nn.functional.softplus(
        negative_logits
    ).mean()


@torch.no_grad()
def evaluate_full_catalog(
    model: SetRecSASRec,
    dataset: RollingNextItemDataset,
    *,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hits = 0.0
    ndcg = 0.0
    for sequences, last_indices, targets in loader:
        sequences = sequences.to(device)
        last_indices = last_indices.to(device)
        targets = targets.to(device)
        encoded = model(sequences)
        user = encoded[torch.arange(encoded.shape[0], device=device), last_indices]
        logits = user @ model.item_embedding.weight[1:].T
        for row in range(sequences.shape[0]):
            seen = sequences[row][sequences[row].ne(0)].unique() - 1
            logits[row, seen] = -torch.inf
        ranking = torch.topk(logits, k=min(top_k, logits.shape[1]), dim=1).indices + 1
        matches = ranking.eq(targets[:, None])
        hit_rows = matches.any(dim=1)
        hits += float(hit_rows.sum())
        if bool(hit_rows.any()):
            ranks = matches[hit_rows].float().argmax(dim=1) + 1
            ndcg += float((1.0 / torch.log2(ranks.float() + 1.0)).sum())
    n = len(dataset)
    return {"n": n, "hit@10": hits / n, "ndcg@10": ndcg / n}


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_setrec_cf_tokenizer(
    *,
    ordered_items: Sequence[str],
    train_examples: Sequence[FullportExample],
    dev_examples: Sequence[FullportExample],
    output_path: Path,
    device: torch.device,
    spec: SetRecCFSpec = SetRecCFSpec(),
    heartbeat: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not train_examples or not dev_examples:
        raise ValueError("SETRec tokenizer requires non-empty train and internal dev")
    if len(set(ordered_items)) != len(ordered_items):
        raise ValueError("SETRec catalog contains duplicate items")
    seed_everything(spec.seed)
    item_to_index = {item: index + 1 for index, item in enumerate(ordered_items)}
    train_dataset = RollingNextItemDataset(
        train_examples, item_to_index, spec.max_history_items
    )
    dev_dataset = RollingNextItemDataset(
        dev_examples, item_to_index, spec.max_history_items
    )
    generator = torch.Generator().manual_seed(spec.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    train_item_ids = torch.tensor(
        sorted(
            {
                item_to_index[item]
                for example in train_examples
                for item in (*example.history, example.target)
            }
        ),
        dtype=torch.long,
        device=device,
    )
    model = SetRecSASRec(len(ordered_items), spec).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    best_epoch = 0
    best_ndcg = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    learning_curve: list[dict[str, Any]] = []
    for epoch in range(1, spec.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for sequences, last_indices, targets in loader:
            sequences = sequences.to(device)
            last_indices = last_indices.to(device)
            targets = targets.to(device)
            encoded = model(sequences)
            final = encoded[
                torch.arange(encoded.shape[0], device=device), last_indices
            ]
            loss = sampled_bce_loss(final, targets, model, train_item_ids)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite SASRec loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = sequences.shape[0]
            total_loss += float(loss) * rows
            total_rows += rows
        metrics = evaluate_full_catalog(
            model,
            dev_dataset,
            batch_size=spec.eval_batch_size,
            top_k=spec.top_k,
            device=device,
        )
        record = {
            "epoch": epoch,
            "training_loss": total_loss / total_rows,
            **metrics,
        }
        if not all(
            math.isfinite(float(record[key]))
            for key in ("training_loss", "hit@10", "ndcg@10")
        ):
            raise FloatingPointError(f"non-finite SASRec metric at epoch {epoch}")
        learning_curve.append(record)
        if float(record["ndcg@10"]) > best_ndcg:
            best_epoch = epoch
            best_ndcg = float(record["ndcg@10"])
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        print(record, flush=True)
        if heartbeat is not None:
            heartbeat(epoch, record)
    if best_state is None:
        raise AssertionError("SASRec best checkpoint was not selected")
    model.load_state_dict(best_state)
    item_embeddings = model.item_embedding.weight[1:].detach().cpu().float()
    if item_embeddings.shape != (len(ordered_items), spec.hidden_size):
        raise RuntimeError("SASRec output feature shape drifted")
    if not bool(torch.isfinite(item_embeddings).all()):
        raise FloatingPointError("SASRec output contains non-finite values")
    payload = {
        "schema_version": "phase17.s17_fp3_setrec_cf_tokenizer.v1",
        "ordered_items": list(ordered_items),
        "item_embeddings": item_embeddings,
        "model_state_dict": best_state,
        "spec": asdict(spec),
        "train_examples": len(train_dataset),
        "internal_dev_examples": len(dev_dataset),
        "train_fit_item_count": int(train_item_ids.numel()),
        "best_epoch": best_epoch,
        "best_internal_dev_ndcg@10": best_ndcg,
        "learning_curve": learning_curve,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
    _atomic_torch_save(output_path, payload)
    return payload
