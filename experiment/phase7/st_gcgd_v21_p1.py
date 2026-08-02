#!/usr/bin/env python3
"""Matched P1 for ST-GCGD-v2.1: A/V3/B/D/E on a fresh cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from transformers import LogitsProcessor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gacr_p0 import split_training_users
from experiment.phase4.gacr_s0 import select_stratified_samples
from experiment.phase4.gcdh_p0 import build_train_samples, build_validation_samples, prepare, read_users, sha256, write_json
from experiment.phase7.gcgd_p0 import read_train_sequences
from experiment.phase7.gcgd_p1 import (
    AdaptiveGraphPrefixLogitsProcessor, arm_metric_row, build_indexed_graph,
    generate_arm_items, graph_logits_for_user, graph_prefix_inputs,
    matched_gacr_v3_rank, select_development_users, train_lightgcn,
)
from experiment.phase7.gcgd_p1_run import bootstrap_rows, gacr_row, summarize_rows
from experiment.phase7.st_gcgd_v2 import PseudoFutureRecord, audit_arm_rows
from experiment.phase7.st_gcgd_v21 import load_inputs_v21, train_arm_v21


def stable_order(seed: int, dataset: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|st-gcgd-v21-p1|{key}".encode()).hexdigest()


class AdvantageGate(torch.nn.Module):
    """A deliberately capacious target-free gate; final output remains scalar."""

    def __init__(self, features: int = 8) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(features, 128), torch.nn.LayerNorm(128), torch.nn.GELU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(values).squeeze(-1))


class ScalarGatePrefixProcessor(AdaptiveGraphPrefixLogitsProcessor):
    """Apply one target-free per-request advantage probability at every prefix."""

    def __init__(self, *args, scalar_gate: float, threshold: float, **kwargs) -> None:
        super().__init__(*args, adapter=None, **kwargs)
        self.scalar_gate = float(scalar_gate)
        self.threshold = float(threshold)

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.calls += 1
        if self.alpha == 0.0 or self.scalar_gate < self.threshold:
            self.gates.extend([0.0] * len(input_ids))
            return scores
        output = scores.clone()
        applied = 0
        for row, ids in enumerate(input_ids.tolist()):
            graph = self.prefix_scores.get(tuple(ids))
            if not graph:
                continue
            for token, value in graph.items():
                output[row, token] += self.alpha * self.scalar_gate * float(value)
            applied += 1
            self.gates.append(self.scalar_gate)
        self.applied_rows += applied
        return output


def transition_logits_for_sample(model, graph, sample: Mapping[str, object], propagated=None) -> dict[str, float]:
    item_to_index = {item: index for index, item in enumerate(graph.items)}
    user_to_index = {user: index for index, user in enumerate(graph.users)}
    history = tuple(str(item) for item in sample["history_items"])
    if not history or str(sample["user_id"]) not in user_to_index:
        raise ValueError("sample lacks transition graph history/user")
    prefix = tuple(item_to_index[item] for item in history)
    record = PseudoFutureRecord(
        user_index=user_to_index[str(sample["user_id"])], user_id=str(sample["user_id"]),
        prefix=prefix, target=0, sample_key=str(sample["sample_key"]), split="inference",
        target_group="unknown",
    )
    with torch.no_grad():
        if propagated is None:
            values = model.scores(graph, [record], "transition")[0]
        else:
            _, _, transition_context, transition_items = propagated
            session = model.session_projection(model.session_states([record]))
            state = 0.5 * transition_context[prefix[-1]] + 0.5 * session[0]
            values = model.transition_query(state) @ model.transition_key(transition_items).t()
            values = values / math.sqrt(transition_items.shape[1])
        values = values.detach().cpu()
    values[list(set(prefix))] = -10000.0
    if not torch.isfinite(values).all():
        raise ValueError("non-finite transition scores")
    return dict(zip(graph.items, map(float, values.tolist())))


def advantage_features(
    graph_logits: Mapping[str, float], history: Sequence[str], baseline_items: Sequence[str],
    transition_sources: set[int], item_to_index: Mapping[str, int],
) -> tuple[float, ...]:
    values = torch.tensor(list(graph_logits.values()), dtype=torch.float32)
    probabilities = torch.softmax(values, dim=0)
    top = torch.topk(probabilities, min(2, probabilities.numel())).values
    entropy = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum() / math.log(len(values)))
    margin = float(top[0] - top[1]) if len(top) > 1 else 1.0
    graph_top = max(graph_logits, key=lambda item: (graph_logits[item], item))
    last_index = item_to_index[str(history[-1])]
    repeat = 1.0 - len(set(history)) / len(history)
    return (
        float(last_index in transition_sources), min(len(history), 50) / 50.0, repeat,
        entropy, margin, float(bool(baseline_items) and baseline_items[0] == graph_top),
        float(graph_top in set(history)), min(len(set(history)), 50) / 50.0,
    )


def improvement_label(target: str, baseline: Sequence[str], candidate: Sequence[str]) -> float:
    a = baseline.index(target) + 1 if target in baseline else None
    d = candidate.index(target) + 1 if target in candidate else None
    if d is None:
        return 0.0
    if a is None:
        return 1.0
    return float(d < a)


def train_advantage_gate(records: list[dict], config: Mapping[str, object], device: torch.device):
    if not records:
        raise ValueError("empty advantage-gate records")
    torch.manual_seed(int(config["seed"]))
    gate = AdvantageGate(len(records[0]["feature"])).to(device)
    features = torch.tensor([row["feature"] for row in records], dtype=torch.float32, device=device)
    labels = torch.tensor([row["label"] for row in records], dtype=torch.float32, device=device)
    positives = int(labels.sum().item())
    pos_weight = torch.tensor([(len(labels) - positives) / max(1, positives)], device=device).clamp(1.0, 20.0)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        gate.train(); optimizer.zero_grad(set_to_none=True)
        probability = gate(features).clamp(1e-6, 1 - 1e-6)
        loss = -(pos_weight * labels * probability.log() + (1 - labels) * (1 - probability).log()).mean()
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 10.0); optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config["epochs"]):
            history.append({"epoch": epoch, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    gate.eval()
    with torch.no_grad():
        probability = gate(features)
    return gate, history, {
        "records": len(records), "positive_labels": positives,
        "positive_rate": positives / len(records), "mean_probability": float(probability.mean()),
        "accuracy_at_fixed_threshold": float(((probability >= float(config["threshold"])) == (labels > 0.5)).float().mean()),
    }


def identity_row(sample: Mapping[str, object], baseline: Sequence[str], group: str) -> dict:
    return arm_metric_row(sample_key=str(sample["sample_key"]), target=str(sample["positive_item"]),
                          baseline_items=baseline, candidate_items=baseline,
                          target_group=group, graph_covered=True)


def write_rows(path: Path, arm_rows: Mapping[str, list[dict]]) -> None:
    rows = [{"arm": arm, **row} for arm, values in arm_rows.items() for row in values]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_domain(dataset: str, config: dict, output_root: Path) -> dict:
    device = torch.device("cuda:0")
    torch.manual_seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"]))
    parent = json.loads((ROOT / config["inputs"]["phase4_parent_config"]).read_text())
    prepared = prepare(dataset, parent, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"
    expected_checkpoint = config["inputs"]["expected_parent_checkpoint_sha256"][dataset]
    if sha256(checkpoint) != expected_checkpoint:
        raise ValueError(f"{dataset} parent checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()

    temporal_graph, sequence_path, catalog_path = load_inputs_v21(dataset, config["temporal_lineage"])
    bank = ROOT / config["temporal_lineage"]["hard_negative_bank"]["datasets"][dataset]["path"]
    if sha256(bank) != config["temporal_lineage"]["hard_negative_bank"]["datasets"][dataset]["sha256"]:
        raise ValueError("hard-negative bank SHA mismatch")
    transition_model, transition_training = train_arm_v21(
        temporal_graph, "transition", config["temporal_training"], json.loads(bank.read_text()), device
    )
    transition_model.eval()
    with torch.no_grad():
        transition_propagated = transition_model.propagate_deep(temporal_graph)

    static_sequences = read_train_sequences(ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt", 2)
    static_graph = build_indexed_graph(static_sequences, prepared["catalog"])
    static_model, static_training = train_lightgcn(static_graph, config["static_graph"], device)
    static_model.eval()
    with torch.no_grad():
        static_propagated = static_model.propagate(static_graph.edges.to(device))

    split_root = ROOT / config["inputs"]["split_root"] / dataset
    phase4_train, phase4_validation = read_users(split_root / "train_users.txt"), read_users(split_root / "validation_users.txt")
    development_users, cohort = select_development_users(
        set(prepared["sequences"]), phase4_train, phase4_validation, dataset,
        config["prior_validation_salts"], config["development_salt"], int(config["development_users_per_dataset"]),
    )
    expected = config["expected_development_cohort"][dataset]
    if cohort["users"] != expected["users"] or cohort["user_sha256"] != expected["user_sha256"]:
        raise ValueError("development cohort lineage mismatch")

    item_paths = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    _, leaf_fractions = graph_prefix_inputs(item_paths, {item: 0.0 for item in prepared["catalog"]})
    maximum_depth = max(map(len, item_paths.values())) - 1
    transition_sources = set(temporal_graph.transition_edges[0].tolist())
    item_to_index = {item: index for index, item in enumerate(temporal_graph.items)}

    fit_users, calibration_users = split_training_users(phase4_train, int(config["seed"]), dataset)
    fit_pool = build_train_samples(prepared["sequences"], fit_users, prepared["item2input"], prepared["item2lexid"])
    calibration_pool = build_train_samples(prepared["sequences"], calibration_users, prepared["item2input"], prepared["item2lexid"])
    count = int(config["advantage_gate"]["fit_samples_per_dataset"]) // 2
    fit_samples = select_stratified_samples(fit_pool, prepared["heads"], int(config["seed"]), f"{dataset}|v21-p1-gate-fit", count, count)
    cal_count = int(config["advantage_gate"]["calibration_samples_per_dataset"]) // 2
    calibration_samples = select_stratified_samples(calibration_pool, prepared["heads"], int(config["seed"]), f"{dataset}|v21-p1-gate-cal", cal_count, cal_count)

    def gate_records(samples: list[dict], label: str) -> list[dict]:
        records = []
        for index, sample in enumerate(samples, 1):
            logits = transition_logits_for_sample(transition_model, temporal_graph, sample, transition_propagated)
            prefix_scores, _ = graph_prefix_inputs(item_paths, logits)
            baseline, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=None)
            processor = AdaptiveGraphPrefixLogitsProcessor(prefix_scores, leaf_fractions, alpha=float(config["decoding"]["D_alpha"]), maximum_depth=maximum_depth, adapter=None)
            candidate, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=processor)
            records.append({"feature": advantage_features(logits, sample["history_items"], baseline, transition_sources, item_to_index),
                            "label": improvement_label(sample["positive_item"], baseline, candidate)})
            if index % 32 == 0:
                print(f"ST_GCGD_V21_P1_GATE_RECORDS dataset={dataset} split={label} records={index}/{len(samples)}", flush=True)
        return records

    fit_records = gate_records(fit_samples, "fit")
    gate, gate_history, gate_fit = train_advantage_gate(fit_records, config["advantage_gate"], device)
    cal_records = gate_records(calibration_samples, "calibration")
    with torch.no_grad():
        cal_features = torch.tensor([row["feature"] for row in cal_records], dtype=torch.float32, device=device)
        cal_labels = torch.tensor([row["label"] for row in cal_records], device=device)
        cal_probability = gate(cal_features)
    gate_calibration = {"records": len(cal_records), "positive_labels": int(cal_labels.sum()),
                        "positive_rate": float(cal_labels.float().mean()), "mean_probability": float(cal_probability.mean()),
                        "accuracy_at_fixed_threshold": float(((cal_probability >= float(config["advantage_gate"]["threshold"])) == (cal_labels > .5)).float().mean())}

    residual_path = ROOT / config["inputs"]["gacr_v3_residual_root"] / dataset / "residual_seed2023.pt"
    if sha256(residual_path) != config["inputs"]["expected_gacr_v3_seed2023_residual_sha256"][dataset]:
        raise ValueError("GACR-v3 residual SHA mismatch")
    residual_state = torch.load(residual_path, map_location=device)
    comparator = {"generator_top_k": 50, "catalog_top_k": 50}
    samples = build_validation_samples(prepared["sequences"], set(development_users), prepared["item2input"], prepared["item2lexid"])
    arms: dict[str, list[dict]] = {key: [] for key in ("A", "V3", "B", "D", "E")}
    identity_exact = False
    torch.cuda.reset_peak_memory_stats(device)
    gate_interventions = 0
    for index, sample in enumerate(samples, 1):
        baseline, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=None)
        group = "head" if sample["positive_item"] in prepared["heads"] else "tail"
        arms["A"].append(identity_row(sample, baseline, group))
        static_logits = graph_logits_for_user(static_model, static_graph, sample["user_id"], visible_history_items=sample["history_items"], propagated_embeddings=static_propagated)
        static_prefix, static_fraction = graph_prefix_inputs(item_paths, static_logits)
        b_processor = AdaptiveGraphPrefixLogitsProcessor(static_prefix, static_fraction, alpha=float(config["decoding"]["B_alpha"]), maximum_depth=maximum_depth, adapter=None)
        b_items, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=b_processor)
        arms["B"].append(arm_metric_row(sample_key=sample["sample_key"], target=sample["positive_item"], baseline_items=baseline, candidate_items=b_items, target_group=group, graph_covered=True))
        transition_logits = transition_logits_for_sample(transition_model, temporal_graph, sample, transition_propagated)
        transition_prefix, transition_fraction = graph_prefix_inputs(item_paths, transition_logits)
        d_processor = AdaptiveGraphPrefixLogitsProcessor(transition_prefix, transition_fraction, alpha=float(config["decoding"]["D_alpha"]), maximum_depth=maximum_depth, adapter=None)
        d_items, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=d_processor)
        arms["D"].append(arm_metric_row(sample_key=sample["sample_key"], target=sample["positive_item"], baseline_items=baseline, candidate_items=d_items, target_group=group, graph_covered=True))
        feature = torch.tensor([advantage_features(transition_logits, sample["history_items"], baseline, transition_sources, item_to_index)], dtype=torch.float32, device=device)
        with torch.no_grad(): probability = float(gate(feature)[0])
        gate_interventions += int(probability >= float(config["advantage_gate"]["threshold"]))
        e_processor = ScalarGatePrefixProcessor(transition_prefix, transition_fraction, alpha=float(config["decoding"]["E_alpha"]), maximum_depth=maximum_depth, scalar_gate=probability, threshold=float(config["advantage_gate"]["threshold"]))
        e_items, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=e_processor)
        arms["E"].append(arm_metric_row(sample_key=sample["sample_key"], target=sample["positive_item"], baseline_items=baseline, candidate_items=e_items, target_group=group, graph_covered=True))
        gram_rank, v3_rank = matched_gacr_v3_rank(sample, prepared, comparator, residual_state, budget=float(config["inputs"]["gacr_v3_domain_budget"][dataset]), device=device)
        arms["V3"].append(gacr_row(sample, baseline, gram_rank, v3_rank, group))
        if index == 1:
            zero = AdaptiveGraphPrefixLogitsProcessor(transition_prefix, transition_fraction, alpha=0.0, maximum_depth=maximum_depth, adapter=None)
            identity, _ = generate_arm_items(sample, prepared, beam_size=50, length_penalty=float(config["decoding"]["length_penalty"]), device=device, processor=zero)
            identity_exact = identity == baseline
            if not identity_exact: raise ValueError("alpha=0 identity failed")
        if index % 16 == 0:
            print(f"ST_GCGD_V21_P1_VALIDATION dataset={dataset} users={index}/{len(samples)}", flush=True)

    audit = audit_arm_rows(arms, ("A", "V3", "B", "D", "E"), len(samples))
    output = output_root / dataset; output.mkdir(parents=True, exist_ok=True)
    torch.save(transition_model.state_dict(), output / "transition_graph.pt")
    torch.save(static_model.state_dict(), output / "static_lightgcn.pt")
    torch.save(gate.state_dict(), output / "advantage_gate.pt")
    write_rows(output / "per_user.csv", arms)
    methods = {arm: {"groups": summarize_rows(rows), "bootstrap": bootstrap_rows(rows, int(config["seed"]) + offset)} for offset, (arm, rows) in enumerate(arms.items(), 1)}
    result = {
        "dataset": dataset, "status": "PASS", "cohort": cohort, "methods": methods,
        "training": {"transition": transition_training, "static": static_training,
                     "advantage_gate": {"fit_users_calibration_users_overlap": len(fit_users & calibration_users), "history": gate_history, "fit": gate_fit, "calibration": gate_calibration}},
        "gate_interventions": gate_interventions, "gate_intervention_rate": gate_interventions / len(samples),
        "arm_semantics": {"A": "true constrained beam@50", "B": "true constrained beam@50", "D": "true constrained beam@50", "E": "true constrained beam@50", "V3": "matched catalog candidate rerank@50; not a generated beam"},
        "audit": audit, "alpha_zero_identity_exact": identity_exact,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "lineage": {"user_sequence_sha256": sha256(sequence_path), "item_index_sha256": sha256(catalog_path), "hard_negative_bank_sha256": sha256(bank)},
        "parent_checkpoint_sha256_before": expected_checkpoint, "parent_checkpoint_sha256_after": sha256(checkpoint),
        "test_read": False, "sports_read": False, "effect_decision_automatic": False,
    }
    write_json(output / "summary.json", result)
    print(json.dumps({"dataset": dataset, "status": "PASS", "gate_intervention_rate": result["gate_intervention_rate"]}, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if config.get("execution_enabled") is not True or config.get("decision_status") != "PREREGISTERED_FROZEN_READY_TO_RUN":
        raise ValueError("P1 config is not frozen/enabled")
    if not torch.cuda.is_available(): raise RuntimeError("P1 requires CUDA")
    run_domain(args.dataset, config, args.output_root); return 0


if __name__ == "__main__":
    raise SystemExit(main())
