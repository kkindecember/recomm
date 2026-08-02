#!/usr/bin/env python3
"""ST-GCGD-v2 train-only temporal multi-relational graph qualification.

The executable modes in this module are deliberately limited to P0-R and P0-G.
They never construct validation/test samples and never select a P1 cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_sequences(path: Path) -> dict[str, list[str]]:
    sequences: dict[str, list[str]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0] in sequences:
                raise ValueError(f"duplicate user at {path}:{line_number}")
            sequences[parts[0]] = parts[1:]
    return sequences


def read_catalog(path: Path) -> list[str]:
    items: list[str] = []
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split(maxsplit=1)
            if fields and fields[0]:
                items.append(fields[0])
    if not items or len(items) != len(set(items)):
        raise ValueError(f"invalid catalog: {path}")
    return items


@dataclass(frozen=True)
class PseudoFutureRecord:
    user_index: int
    user_id: str
    prefix: tuple[int, ...]
    target: int
    sample_key: str
    split: str
    target_group: str


@dataclass(frozen=True)
class TemporalGraph:
    users: tuple[str, ...]
    items: tuple[str, ...]
    ui_edges: torch.LongTensor
    ui_weights: torch.FloatTensor
    transition_edges: torch.LongTensor
    transition_weights: torch.FloatTensor
    records: tuple[PseudoFutureRecord, ...]
    user_visible: tuple[frozenset[int], ...]


def _edge_tensors(weights: Mapping[tuple[int, int], float]) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = sorted(weights)
    if not ordered:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    return (
        torch.tensor(ordered, dtype=torch.long).t().contiguous(),
        torch.tensor([weights[key] for key in ordered], dtype=torch.float32),
    )


def build_temporal_graph(
    sequences: Mapping[str, Sequence[str]],
    catalog: Sequence[str],
    *,
    seed: int,
    calibration_fraction: float,
    recency_decay: float,
    skip_self_transitions: bool,
    head_fraction: float = 0.2,
) -> TemporalGraph:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if not 0.0 < recency_decay <= 1.0:
        raise ValueError("recency_decay must be in (0, 1]")
    items = tuple(catalog)
    item_to_index = {item: index for index, item in enumerate(items)}
    eligible = {u: tuple(seq[:-2]) for u, seq in sequences.items() if len(seq[:-2]) >= 2}
    users = tuple(sorted(eligible))
    popularity: Counter[int] = Counter()
    for train in eligible.values():
        popularity.update(item_to_index[item] for item in train[:-1] if item in item_to_index)
    head_count = max(1, math.ceil(len(items) * head_fraction))
    heads = {item for item, _ in sorted(popularity.items(), key=lambda x: (-x[1], x[0]))[:head_count]}
    ui: dict[tuple[int, int], float] = defaultdict(float)
    transitions: dict[tuple[int, int], float] = defaultdict(float)
    records: list[PseudoFutureRecord] = []
    visible_sets: list[frozenset[int]] = []
    for user_index, user in enumerate(users):
        train = eligible[user]
        unknown = [item for item in train if item not in item_to_index]
        if unknown:
            raise ValueError(f"train item outside catalog: {unknown[0]}")
        prefix = tuple(item_to_index[item] for item in train[:-1])
        target = item_to_index[train[-1]]
        visible_sets.append(frozenset(prefix))
        length = len(prefix)
        for position, item in enumerate(prefix):
            age = length - position - 1
            ui[(user_index, item)] += recency_decay ** age
        for source, target_item in zip(prefix, prefix[1:]):
            if skip_self_transitions and source == target_item:
                continue
            transitions[(source, target_item)] += 1.0
        bucket = int(stable_hash(f"{seed}|{user}")[:16], 16) / float(16**16)
        split = "calibration" if bucket < calibration_fraction else "fit"
        records.append(PseudoFutureRecord(
            user_index=user_index,
            user_id=user,
            prefix=prefix,
            target=target,
            sample_key=f"{user}:train-pseudo-future:{train[-1]}",
            split=split,
            target_group="head" if target in heads else "tail",
        ))
    if len({record.sample_key for record in records}) != len(records):
        raise ValueError("duplicate pseudo-future sample key")
    ui_edges, ui_weights = _edge_tensors(ui)
    tr_edges, tr_weights = _edge_tensors(transitions)
    return TemporalGraph(users, items, ui_edges, ui_weights, tr_edges, tr_weights, tuple(records), tuple(visible_sets))


def _weighted_aggregate(
    source_values: torch.Tensor,
    source: torch.LongTensor,
    target: torch.LongTensor,
    weights: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    output = torch.zeros((target_count, source_values.shape[1]), device=source_values.device, dtype=source_values.dtype)
    degree = torch.zeros(target_count, device=source_values.device, dtype=source_values.dtype)
    output.index_add_(0, target, source_values[source] * weights[:, None])
    degree.index_add_(0, target, weights)
    return output / degree.clamp_min(1e-12)[:, None]


class TemporalMultiRelationGraph(nn.Module):
    """Relation-specific UI and directed transition propagation with a history mixer."""

    def __init__(self, users: int, items: int, dimension: int) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        self.ui_user_projection = nn.Linear(dimension, dimension, bias=False)
        self.ui_item_projection = nn.Linear(dimension, dimension, bias=False)
        self.out_projection = nn.Linear(dimension, dimension, bias=False)
        self.in_projection = nn.Linear(dimension, dimension, bias=False)
        self.mixer = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def propagate(self, graph: TemporalGraph, *, static_ui: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.item.weight.device
        ui_edges = graph.ui_edges.to(device)
        ui_weights = graph.ui_weights.to(device)
        if static_ui:
            ui_weights = torch.ones_like(ui_weights)
        if ui_edges.shape[1]:
            user_from_items = _weighted_aggregate(self.item.weight, ui_edges[1], ui_edges[0], ui_weights, len(graph.users))
            item_from_users = _weighted_aggregate(self.user.weight, ui_edges[0], ui_edges[1], ui_weights, len(graph.items))
        else:
            user_from_items = torch.zeros_like(self.user.weight)
            item_from_users = torch.zeros_like(self.item.weight)
        ui_user = self.user.weight + self.ui_user_projection(user_from_items)
        ui_item = self.item.weight + self.ui_item_projection(item_from_users)
        tr_edges = graph.transition_edges.to(device)
        tr_weights = graph.transition_weights.to(device)
        if tr_edges.shape[1]:
            incoming = _weighted_aggregate(self.item.weight, tr_edges[0], tr_edges[1], tr_weights, len(graph.items))
            outgoing = _weighted_aggregate(self.item.weight, tr_edges[1], tr_edges[0], tr_weights, len(graph.items))
        else:
            incoming = torch.zeros_like(self.item.weight)
            outgoing = torch.zeros_like(self.item.weight)
        tr_context = self.item.weight + self.out_projection(outgoing)
        tr_item = self.item.weight + self.in_projection(incoming)
        return ui_user, ui_item, tr_context, tr_item

    def scores(self, graph: TemporalGraph, records: Sequence[PseudoFutureRecord], arm: str) -> torch.Tensor:
        ui_user, ui_item, tr_context, tr_item = self.propagate(graph, static_ui=arm == "static")
        users = torch.tensor([r.user_index for r in records], device=ui_user.device)
        last = torch.tensor([r.prefix[-1] for r in records], device=ui_user.device)
        q_ui = ui_user[users] @ ui_item.t()
        q_tr = tr_context[last] @ tr_item.t()
        if arm in ("static", "ui"):
            return q_ui
        if arm == "transition":
            return q_tr
        if arm != "full":
            raise ValueError(f"unknown arm: {arm}")
        ui_covered = torch.tensor([float(bool(r.prefix)) for r in records], device=ui_user.device)
        tr_sources = set(graph.transition_edges[0].tolist())
        tr_covered = torch.tensor([float(r.prefix[-1] in tr_sources) for r in records], device=ui_user.device)
        length = torch.tensor([min(len(r.prefix), 20) / 20.0 for r in records], device=ui_user.device)
        repeat = torch.tensor([1.0 - len(set(r.prefix)) / len(r.prefix) for r in records], device=ui_user.device)
        gate = torch.sigmoid(self.mixer(torch.stack((ui_covered, tr_covered, length, repeat), dim=1)))
        return gate * q_ui + (1.0 - gate) * q_tr


def deterministic_negatives(
    graph: TemporalGraph,
    records: Sequence[PseudoFutureRecord],
    count: int,
    seed: int,
    hard_negative_cache: Mapping[str, Sequence[str]],
) -> torch.LongTensor:
    item_to_index = {item: index for index, item in enumerate(graph.items)}
    rows: list[list[int]] = []
    for record in records:
        forbidden = set(record.prefix) | {record.target}
        hard = [item_to_index[item] for item in hard_negative_cache.get(record.sample_key, ()) if item in item_to_index and item_to_index[item] not in forbidden]
        candidates = list(dict.fromkeys(hard))
        cursor = 0
        while len(candidates) < count:
            index = int(stable_hash(f"{seed}|{record.sample_key}|{cursor}")[:16], 16) % len(graph.items)
            cursor += 1
            if index not in forbidden and index not in candidates:
                candidates.append(index)
        rows.append(candidates[:count])
    return torch.tensor(rows, dtype=torch.long)


def loss_for_scores(scores: torch.Tensor, targets: torch.LongTensor, negatives: torch.LongTensor, pairwise_weight: float, listwise_weight: float) -> torch.Tensor:
    positive = scores.gather(1, targets[:, None])
    negative = scores.gather(1, negatives)
    pairwise = -F.logsigmoid(positive - negative).mean()
    listwise = F.cross_entropy(torch.cat((positive, negative), dim=1), torch.zeros(len(targets), dtype=torch.long, device=scores.device))
    return pairwise_weight * pairwise + listwise_weight * listwise


@torch.no_grad()
def evaluate_scores(scores: torch.Tensor, records: Sequence[PseudoFutureRecord]) -> dict[str, object]:
    targets = torch.tensor([record.target for record in records], device=scores.device)
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    ranks = 1 + (scores > target_scores[:, None]).sum(dim=1)
    negatives = scores.clone()
    negatives.scatter_(1, targets[:, None], -torch.inf)
    margins = target_scores - negatives.max(dim=1).values
    result: dict[str, object] = {
        "n": len(records),
        "Recall@10": float((ranks <= 10).float().mean()),
        "NDCG@10": float(torch.where(ranks <= 10, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float32)).mean()),
        "Recall@50": float((ranks <= 50).float().mean()),
        "MRR": float((1.0 / ranks.float()).mean()),
        "mean_target_margin": float(margins.mean()),
    }
    for group in ("head", "tail"):
        selected = [index for index, record in enumerate(records) if record.target_group == group]
        result[group] = {"n": len(selected), "available": bool(selected)}
        if selected:
            group_ranks = ranks[selected]
            result[group].update({
                "Recall@10": float((group_ranks <= 10).float().mean()),
                "NDCG@10": float(torch.where(group_ranks <= 10, 1.0 / torch.log2(group_ranks.float() + 1), torch.zeros_like(group_ranks, dtype=torch.float32)).mean()),
            })
    return result


def audit_arm_rows(arm_rows: Mapping[str, Sequence[Mapping[str, object]]], expected_arms: Sequence[str], expected_n: int) -> dict[str, object]:
    errors: list[str] = []
    reference_keys: set[str] | None = None
    counts: dict[str, int] = {}
    for arm in expected_arms:
        rows = list(arm_rows.get(arm, ()))
        counts[arm] = len(rows)
        keys = [str(row.get("sample_key")) for row in rows]
        if len(rows) != expected_n:
            errors.append(f"{arm}: expected {expected_n} rows, got {len(rows)}")
        if len(keys) != len(set(keys)):
            errors.append(f"{arm}: duplicate sample_key")
        key_set = set(keys)
        if reference_keys is None:
            reference_keys = key_set
        elif key_set != reference_keys:
            errors.append(f"{arm}: unmatched sample keys")
        for row in rows:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    errors.append(f"{arm}:{row.get('sample_key')}:{key}: non-finite")
    if errors:
        raise ValueError("; ".join(errors))
    return {"passed": True, "row_counts": counts, "unique_sample_keys": len(reference_keys or ())}


def train_arm(graph: TemporalGraph, arm: str, config: Mapping[str, object], hard_cache: Mapping[str, Sequence[str]], device: torch.device) -> tuple[TemporalMultiRelationGraph, list[dict[str, float]]]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TemporalMultiRelationGraph(len(graph.users), len(graph.items), int(config["embedding_dim"])).to(device)
    fit = [record for record in graph.records if record.split == "fit"]
    if not fit:
        raise ValueError("empty fit split")
    targets = torch.tensor([record.target for record in fit], device=device)
    negatives = deterministic_negatives(graph, fit, int(config["negative_count"]), seed, hard_cache).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["l2"]))
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model.scores(graph, fit, arm)
        loss = loss_for_scores(scores, targets, negatives, float(config["pairwise_weight"]), float(config["listwise_weight"]))
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {arm}/{epoch}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        if not torch.isfinite(norm):
            raise ValueError(f"non-finite gradient: {arm}/{epoch}")
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), "gradient_norm": float(norm.detach())})
    model.eval()
    return model, history


def load_inputs(dataset: str, config: Mapping[str, object]) -> tuple[TemporalGraph, Path, Path]:
    spec = config["datasets"][dataset]
    sequence_path = ROOT / str(spec["user_sequence"])
    catalog_path = ROOT / str(spec["item_index"])
    graph = build_temporal_graph(
        read_sequences(sequence_path), read_catalog(catalog_path), seed=int(config["seed"]),
        calibration_fraction=float(config["split"]["calibration_fraction"]),
        recency_decay=float(config["graph"]["recency_decay"]),
        skip_self_transitions=bool(config["graph"]["skip_self_transitions"]),
    )
    return graph, sequence_path, catalog_path


def run_p0_graph(dataset: str, config: Mapping[str, object], output_root: Path) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P0-G requires CUDA_VISIBLE_DEVICES=0")
    graph, sequence_path, catalog_path = load_inputs(dataset, config)
    cache_path = output_root / dataset / "gram_hard_negatives.json"
    hard_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    device = torch.device("cuda:0")
    calibration = [record for record in graph.records if record.split == "calibration"]
    arms: dict[str, object] = {}
    for arm in config["arms"]:
        model, history = train_arm(graph, arm, config["training"], hard_cache, device)
        metrics = evaluate_scores(model.scores(graph, calibration, arm), calibration)
        arms[arm] = {"training": history, "calibration": metrics}
        del model
        torch.cuda.empty_cache()
        print(f"ST_GCGD_V2_P0_G_ARM dataset={dataset} arm={arm} ndcg10={metrics['NDCG@10']:.8f}", flush=True)
    full = arms["full"]["calibration"]
    static = arms["static"]["calibration"]
    qualified = bool(full["mean_target_margin"] > static["mean_target_margin"] and not (
        full["Recall@10"] < static["Recall@10"] and full["NDCG@10"] < static["NDCG@10"]
    ))
    summary = {
        "experiment_id": config["experiment_id"], "dataset": dataset,
        "status": "P0_G_QUALIFIED" if qualified else "P0_G_NOT_QUALIFIED",
        "input_lineage": {"user_sequence_sha256": sha256(sequence_path), "item_index_sha256": sha256(catalog_path)},
        "graph": {"users": len(graph.users), "items": len(graph.items), "ui_edges": graph.ui_edges.shape[1], "directed_transition_edges": graph.transition_edges.shape[1], "fit_records": sum(r.split == 'fit' for r in graph.records), "calibration_records": len(calibration)},
        "hard_negative_cache_records": len(hard_cache), "arms": arms,
        "qualification": {"passed": qualified, "rule": "full margin > static margin and not both Recall@10/NDCG@10 lower"},
        "integrity": {"train_slice": "items[:-2]", "pseudo_future_removed_from_edges": True, "validation_read": False, "test_read": False, "sports_read": False},
    }
    write_json(output_root / dataset / "summary.json", summary)
    return summary


def validate_config(config: Mapping[str, object], mode: str) -> None:
    errors = []
    if config.get("seed") != 2023:
        errors.append("seed must be 2023")
    if list(config.get("datasets", {})) != ["Toys", "Beauty"]:
        errors.append("datasets must be Toys and Beauty only")
    integrity = config.get("integrity", {})
    for key in ("validation_forbidden", "test_forbidden", "sports_forbidden"):
        if integrity.get(key) is not True:
            errors.append(f"integrity.{key} must be true")
    if mode == "p0-g" and config.get("execution_enabled") is not True:
        errors.append("P0-G config is not execution enabled")
    if errors:
        raise ValueError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("p0-g",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, args.mode)
    summary = run_p0_graph(args.dataset, config, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
