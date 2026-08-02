#!/usr/bin/env python3
"""ST-GCGD-v2.1: transition-first, fail-closed relation mixing.

This is a new train-only P0 lineage.  It imports v2 data/integrity utilities but
does not alter or overwrite the completed v2 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase7.st_gcgd_v2 import (
    PseudoFutureRecord,
    TemporalGraph,
    TemporalMultiRelationGraph,
    deterministic_negatives,
    loss_for_scores,
    read_catalog,
    read_sequences,
    sha256,
    stable_hash,
    build_temporal_graph,
    write_json,
)


class TransitionFirstGraph(TemporalMultiRelationGraph):
    """Deep temporal multi-relation encoder with a fail-closed UI residual."""

    def __init__(
        self,
        users: int,
        items: int,
        dimension: int,
        maximum_ui_fraction: float,
        layers: int = 3,
        dropout: float = 0.1,
        maximum_session_length: int = 20,
    ) -> None:
        super().__init__(users, items, dimension)
        if not 0.0 <= maximum_ui_fraction <= 1.0:
            raise ValueError("maximum_ui_fraction must be in [0, 1]")
        if layers <= 0 or maximum_session_length <= 0:
            raise ValueError("layers and maximum_session_length must be positive")
        self.maximum_ui_fraction = float(maximum_ui_fraction)
        self.maximum_session_length = int(maximum_session_length)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.ui_layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "user": torch.nn.Linear(dimension, dimension, bias=False),
                "item": torch.nn.Linear(dimension, dimension, bias=False),
                "user_norm": torch.nn.LayerNorm(dimension),
                "item_norm": torch.nn.LayerNorm(dimension),
            }) for _ in range(layers)
        ])
        self.transition_layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "out": torch.nn.Linear(dimension, dimension, bias=False),
                "in": torch.nn.Linear(dimension, dimension, bias=False),
                "out_norm": torch.nn.LayerNorm(dimension),
                "in_norm": torch.nn.LayerNorm(dimension),
            }) for _ in range(layers)
        ])
        self.session_gru = torch.nn.GRU(dimension, dimension, batch_first=True)
        self.session_projection = torch.nn.Linear(dimension, dimension, bias=False)
        self.ui_query = torch.nn.Linear(dimension, dimension, bias=False)
        self.ui_key = torch.nn.Linear(dimension, dimension, bias=False)
        self.transition_query = torch.nn.Linear(dimension, dimension, bias=False)
        self.transition_key = torch.nn.Linear(dimension, dimension, bias=False)
        self.mixer = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.GELU(), torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(64, 1),
        )
        torch.nn.init.zeros_(self.mixer[-1].weight)
        torch.nn.init.constant_(self.mixer[-1].bias, -6.0)

    @staticmethod
    def _aggregate(values, source, target, weights, count):
        output = torch.zeros((count, values.shape[1]), device=values.device, dtype=values.dtype)
        degree = torch.zeros(count, device=values.device, dtype=values.dtype)
        output.index_add_(0, target, values[source] * weights[:, None])
        degree.index_add_(0, target, weights)
        return output / degree.clamp_min(1e-12)[:, None]

    def propagate_deep(self, graph: TemporalGraph, *, static_ui: bool = False):
        device = self.item.weight.device
        ui_edges = graph.ui_edges.to(device)
        ui_weights = graph.ui_weights.to(device)
        if static_ui:
            ui_weights = torch.ones_like(ui_weights)
        ui_user, ui_item = self.user.weight, self.item.weight
        for layer in self.ui_layers:
            from_items = self._aggregate(ui_item, ui_edges[1], ui_edges[0], ui_weights, len(graph.users))
            from_users = self._aggregate(ui_user, ui_edges[0], ui_edges[1], ui_weights, len(graph.items))
            ui_user = layer["user_norm"](ui_user + self.dropout(F.gelu(layer["user"](from_items))))
            ui_item = layer["item_norm"](ui_item + self.dropout(F.gelu(layer["item"](from_users))))
        tr_edges = graph.transition_edges.to(device)
        tr_weights = graph.transition_weights.to(device)
        outgoing = self.item.weight
        incoming = self.item.weight
        for layer in self.transition_layers:
            out_message = self._aggregate(incoming, tr_edges[1], tr_edges[0], tr_weights, len(graph.items))
            in_message = self._aggregate(outgoing, tr_edges[0], tr_edges[1], tr_weights, len(graph.items))
            outgoing = layer["out_norm"](outgoing + self.dropout(F.gelu(layer["out"](out_message))))
            incoming = layer["in_norm"](incoming + self.dropout(F.gelu(layer["in"](in_message))))
        return ui_user, ui_item, outgoing, incoming

    def session_states(self, records: Sequence[PseudoFutureRecord]) -> torch.Tensor:
        device = self.item.weight.device
        lengths = torch.tensor([min(len(record.prefix), self.maximum_session_length) for record in records], device=device)
        padded = torch.zeros((len(records), self.maximum_session_length), dtype=torch.long, device=device)
        for row, record in enumerate(records):
            values = record.prefix[-self.maximum_session_length:]
            padded[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        embedded = self.item(padded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.session_gru(packed)
        return hidden[-1]

    def scores(self, graph: TemporalGraph, records: Sequence[PseudoFutureRecord], arm: str) -> torch.Tensor:
        if arm not in ("static", "ui", "transition", "full"):
            raise ValueError(f"unknown arm: {arm}")
        ui_user, ui_item, tr_context, tr_item = self.propagate_deep(graph, static_ui=arm == "static")
        device = ui_user.device
        users = torch.tensor([record.user_index for record in records], device=device)
        last = torch.tensor([record.prefix[-1] for record in records], device=device)
        scale = math.sqrt(ui_user.shape[1])
        q_ui = self.ui_query(ui_user[users]) @ self.ui_key(ui_item).t() / scale
        if arm in ("static", "ui"):
            return q_ui
        session = self.session_projection(self.session_states(records))
        transition_state = 0.5 * tr_context[last] + 0.5 * session
        q_tr = self.transition_query(transition_state) @ self.transition_key(tr_item).t() / scale
        if arm == "transition":
            return q_tr
        transition_sources = set(graph.transition_edges[0].tolist())
        covered = torch.tensor([float(record.prefix[-1] in transition_sources) for record in records], device=device)
        length = torch.tensor([min(len(record.prefix), 20) / 20.0 for record in records], device=device)
        repeat = torch.tensor([1.0 - len(set(record.prefix)) / len(record.prefix) for record in records], device=device)
        ui_coverage = torch.ones_like(covered)
        session_norm = session.norm(dim=1) / math.sqrt(session.shape[1])
        disagreement = torch.tanh((q_tr.max(dim=1).values - q_ui.max(dim=1).values).abs())
        gamma = self.maximum_ui_fraction * torch.sigmoid(
            self.mixer(torch.stack((ui_coverage, covered, length, repeat, session_norm, disagreement), dim=1))
        )
        mixed = q_tr + gamma * (q_ui - q_tr)
        return torch.where(covered[:, None] > 0, mixed, q_ui)


@torch.no_grad()
def evaluate_scores_v21(scores: torch.Tensor, records: Sequence[PseudoFutureRecord]) -> dict[str, object]:
    targets = torch.tensor([record.target for record in records], device=scores.device)
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    ranks = 1 + (scores > target_scores[:, None]).sum(dim=1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, targets[:, None], False)
    negatives = scores.masked_fill(~mask, 0.0)
    count = scores.shape[1] - 1
    mean = negatives.sum(dim=1) / count
    centered = (scores - mean[:, None]).masked_fill(~mask, 0.0)
    std = torch.sqrt(centered.square().sum(dim=1) / count).clamp_min(1e-8)
    z_separation = (target_scores - mean) / std
    maximum_negative = scores.masked_fill(~mask, -torch.inf).max(dim=1).values

    def group(indices: list[int]) -> dict[str, object]:
        if not indices:
            return {"n": 0, "available": False}
        selected_ranks = ranks[indices]
        return {
            "n": len(indices), "available": True,
            "Recall@10": float((selected_ranks <= 10).float().mean()),
            "NDCG@10": float(torch.where(selected_ranks <= 10, 1.0 / torch.log2(selected_ranks.float() + 1), torch.zeros_like(selected_ranks, dtype=torch.float32)).mean()),
        }

    return {
        "n": len(records),
        "Recall@10": float((ranks <= 10).float().mean()),
        "NDCG@10": float(torch.where(ranks <= 10, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float32)).mean()),
        "Recall@50": float((ranks <= 50).float().mean()),
        "MRR": float((1.0 / ranks.float()).mean()),
        "mean_target_margin": float((target_scores - maximum_negative).mean()),
        "mean_target_z_separation": float(z_separation.mean()),
        "head": group([index for index, record in enumerate(records) if record.target_group == "head"]),
        "tail": group([index for index, record in enumerate(records) if record.target_group == "tail"]),
    }


def load_inputs_v21(dataset: str, config: Mapping[str, object]) -> tuple[TemporalGraph, Path, Path]:
    spec = config["datasets"][dataset]
    sequence_path = ROOT / str(spec["user_sequence"])
    catalog_path = ROOT / str(spec["item_index"])
    split_seed = int(stable_hash(str(config["split"]["salt"]))[:8], 16)
    graph = build_temporal_graph(
        read_sequences(sequence_path), read_catalog(catalog_path), seed=split_seed,
        calibration_fraction=float(config["split"]["calibration_fraction"]),
        recency_decay=float(config["graph"]["recency_decay"]),
        skip_self_transitions=bool(config["graph"]["skip_self_transitions"]),
    )
    return graph, sequence_path, catalog_path


def _optimize(
    model: TransitionFirstGraph,
    graph: TemporalGraph,
    records: Sequence[PseudoFutureRecord],
    arm: str,
    config: Mapping[str, object],
    hard_cache: Mapping[str, Sequence[str]],
    parameters: Sequence[torch.nn.Parameter],
    epochs: int,
) -> list[dict[str, float]]:
    device = model.item.weight.device
    targets = torch.tensor([record.target for record in records], device=device)
    negatives = deterministic_negatives(graph, records, int(config["negative_count"]), int(config["seed"]), hard_cache).to(device)
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=float(config["l2"]))
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model.scores(graph, records, arm)
        loss = loss_for_scores(scores, targets, negatives, float(config["pairwise_weight"]), float(config["listwise_weight"]))
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {arm}/{epoch}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip"]))
        if not torch.isfinite(norm):
            raise ValueError(f"non-finite gradient: {arm}/{epoch}")
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), "gradient_norm": float(norm.detach())})
    return history


def train_arm_v21(
    graph: TemporalGraph,
    arm: str,
    config: Mapping[str, object],
    hard_cache: Mapping[str, Sequence[str]],
    device: torch.device,
) -> tuple[TransitionFirstGraph, dict[str, list[dict[str, float]]]]:
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    model = TransitionFirstGraph(
        len(graph.users), len(graph.items), int(config["embedding_dim"]), float(config["maximum_ui_fraction"]),
        layers=int(config["layers"]), dropout=float(config["dropout"]),
        maximum_session_length=int(config["maximum_session_length"]),
    ).to(device)
    fit = [record for record in graph.records if record.split == "fit"]
    if arm != "full":
        history = _optimize(model, graph, fit, arm, config, hard_cache, list(model.parameters()), int(config["backbone_epochs"]))
        model.eval()
        return model, {"backbone": history, "mixer": []}
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("mixer.")]
    backbone = _optimize(model, graph, fit, "transition", config, hard_cache, backbone_parameters, int(config["backbone_epochs"]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("mixer."))
    mixer_parameters = [parameter for parameter in model.mixer.parameters() if parameter.requires_grad]
    mixer = _optimize(model, graph, fit, "full", config, hard_cache, mixer_parameters, int(config["mixer_epochs"]))
    model.eval()
    return model, {"backbone": backbone, "mixer": mixer}


def hard_negative_coverage(graph: TemporalGraph, hard_cache: Mapping[str, Sequence[str]]) -> dict[str, int | float]:
    fit = [record for record in graph.records if record.split == "fit"]
    covered = sum(record.sample_key in hard_cache and bool(hard_cache[record.sample_key]) for record in fit)
    return {"fit_records": len(fit), "covered_fit_records": covered, "coverage": covered / len(fit)}


def run_p0_g2(dataset: str, config: Mapping[str, object], output_root: Path) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P0-G2 requires CUDA_VISIBLE_DEVICES=0")
    graph, sequence_path, catalog_path = load_inputs_v21(dataset, config)
    cache_path = ROOT / config["hard_negative_bank"]["datasets"][dataset]["path"]
    if sha256(cache_path) != config["hard_negative_bank"]["datasets"][dataset]["sha256"]:
        raise ValueError("hard-negative bank SHA mismatch")
    hard_cache = json.loads(cache_path.read_text())
    calibration = [record for record in graph.records if record.split == "calibration"]
    device = torch.device("cuda:0")
    arms: dict[str, object] = {}
    for arm in config["arms"]:
        model, history = train_arm_v21(graph, arm, config["training"], hard_cache, device)
        metrics = evaluate_scores_v21(model.scores(graph, calibration, arm), calibration)
        arms[arm] = {"training": history, "calibration": metrics}
        del model
        torch.cuda.empty_cache()
        print(f"ST_GCGD_V21_P0_G2_ARM dataset={dataset} arm={arm} ndcg10={metrics['NDCG@10']:.8f}", flush=True)
    transition = arms["transition"]["calibration"]
    full = arms["full"]["calibration"]
    mixer_safe = all(full[key] >= transition[key] for key in ("Recall@10", "NDCG@10", "Recall@50"))
    selected_arm = "full" if mixer_safe else "transition"
    selected = arms[selected_arm]["calibration"]
    static = arms["static"]["calibration"]
    qualified = bool(selected["mean_target_z_separation"] > static["mean_target_z_separation"] and not (
        selected["Recall@10"] < static["Recall@10"] and selected["NDCG@10"] < static["NDCG@10"]
    ))
    summary = {
        "experiment_id": config["experiment_id"], "dataset": dataset,
        "status": "P0_G2_QUALIFIED" if qualified else "P0_G2_NOT_QUALIFIED",
        "input_lineage": {"user_sequence_sha256": sha256(sequence_path), "item_index_sha256": sha256(catalog_path), "hard_negative_bank_sha256": sha256(cache_path)},
        "graph": {"users": len(graph.users), "items": len(graph.items), "ui_edges": graph.ui_edges.shape[1], "directed_transition_edges": graph.transition_edges.shape[1]},
        "hard_negative_coverage": hard_negative_coverage(graph, hard_cache),
        "arms": arms,
        "fail_closed_mixer": {"mixer_safe": mixer_safe, "selected_arm": selected_arm, "rule": "full must preserve transition Recall@10, NDCG@10, and Recall@50; otherwise select transition"},
        "qualification": {"passed": qualified, "rule": "selected z-separation > static and not both Recall@10/NDCG@10 lower"},
        "integrity": {"train_slice": "items[:-2]", "pseudo_future_removed_from_edges": True, "validation_read": False, "test_read": False, "sports_read": False},
    }
    write_json(output_root / dataset / "summary.json", summary)
    return summary


def validate_config(config: Mapping[str, object]) -> None:
    errors = []
    if config.get("seed") != 2023:
        errors.append("seed must be 2023")
    if list(config.get("datasets", {})) != ["Toys", "Beauty"]:
        errors.append("datasets must be Toys and Beauty only")
    for key in ("validation_forbidden", "test_forbidden", "sports_forbidden"):
        if config.get("integrity", {}).get(key) is not True:
            errors.append(f"integrity.{key} must be true")
    if config.get("execution_enabled") is not True:
        errors.append("config is not execution enabled")
    if errors:
        raise ValueError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    result = run_p0_g2(args.dataset, config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
