#!/usr/bin/env python3
"""LNDR N1: frozen training-prefix audit of node polysemy and readout deficit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from experiment.phase4.ialc_n1 import select_unique_user_samples  # noqa: E402
from utils import generation_trie as gt  # noqa: E402

Edge = tuple[tuple[int, ...], int]


def edge_id(edge: Edge) -> str:
    parent, child = edge
    return f"{','.join(map(str, parent))}>{child}"


def branch_bin(count: int) -> str:
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def normalized_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    pooled = (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    return F.normalize(pooled.float(), dim=-1)


@torch.no_grad()
def catalog_semantic_vectors(
    prepared: dict, config: dict, device: torch.device
) -> dict[str, np.ndarray]:
    """Frozen mean native-embedding representation of catalog metadata."""
    tokenizer = prepared["tokenizer"]
    embedding = prepared["model"].backbone.shared
    catalog = prepared["catalog"]
    batch_size = int(config["semantic_audit"]["metadata_batch_size"])
    maximum = int(config["semantic_audit"]["metadata_max_tokens"])
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(catalog), batch_size):
        items = catalog[start : start + batch_size]
        encoded = tokenizer(
            [prepared["item2input"][item] for item in items],
            padding=True,
            truncation=True,
            max_length=maximum,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        vectors = normalized_mean(embedding(ids), mask).cpu().numpy()
        result.update(zip(items, vectors))
    return result


def build_node_table(
    prepared: dict,
    item_vectors: dict[str, np.ndarray],
    config: dict,
) -> tuple[dict[Edge, dict], dict[tuple[int, int], list[Edge]]]:
    eos = int(prepared["tokenizer"].eos_token_id)
    descendants: dict[Edge, list[str]] = defaultdict(list)
    for item, sequence in zip(prepared["catalog"], prepared["encoded_candidates"]):
        for index, child in enumerate(sequence[1:], start=1):
            if child == eos:
                continue
            descendants[(tuple(sequence[:index]), child)].append(item)
    minimum = int(config["semantic_audit"]["minimum_descendant_items"])
    nodes: dict[Edge, dict] = {}
    groups: dict[tuple[int, int], list[Edge]] = defaultdict(list)
    for edge, items in descendants.items():
        parent, token = edge
        matrix = np.stack([item_vectors[item] for item in items])
        centroid = matrix.mean(0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        nodes[edge] = {
            "token": token,
            "depth": len(parent) - 1,
            "descendant_items": len(items),
            "centroid": centroid,
            "semantic_eligible": len(items) >= minimum,
        }
        if len(items) >= minimum:
            groups[(token, len(parent) - 1)].append(edge)
    low = float(config["semantic_audit"]["low_polysemy_distance_max"])
    high = float(config["semantic_audit"]["high_polysemy_distance_min"])
    for group_edges in groups.values():
        for edge in group_edges:
            distances = [
                1.0
                - float(
                    np.dot(nodes[edge]["centroid"], nodes[other]["centroid"])
                )
                for other in group_edges
                if other != edge
            ]
            score = float(np.mean(distances)) if distances else None
            nodes[edge]["same_token_node_count"] = len(group_edges)
            nodes[edge]["semantic_distance"] = score
            nodes[edge]["cohort"] = (
                "high_polysemy"
                if score is not None and score >= high
                else "control"
                if score is None or score <= low
                else "middle"
            )
    for edge, row in nodes.items():
        if "cohort" not in row:
            row["same_token_node_count"] = 0
            row["semantic_distance"] = None
            row["cohort"] = "ineligible"
    return nodes, groups


def gold_margin(logits: torch.Tensor, allowed: list[int], gold: int) -> float:
    if gold not in allowed or len(allowed) < 2:
        raise ValueError("gold margin requires a competitive legal child set")
    alternatives = [token for token in allowed if token != gold]
    return float(logits[gold].float() - logits[alternatives].float().max())


@torch.no_grad()
def audit_batch(
    samples: list[dict],
    prepared: dict,
    trie: gt.Trie,
    item_to_sequence: dict[str, list[int]],
    nodes: dict[Edge, dict],
    device: torch.device,
) -> tuple[list[dict], list[np.ndarray]]:
    batch = collate(prepared["collator"], samples)
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    sequences = [item_to_sequence[row["positive_item"]] for row in samples]
    labels = pad_labels(sequences, device)
    output = prepared["model"].backbone(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = output.decoder_hidden_states[-1].float()
    eos = int(prepared["tokenizer"].eos_token_id)
    rows: list[dict] = []
    states: list[np.ndarray] = []
    for batch_index, (sample, sequence) in enumerate(zip(samples, sequences)):
        target_group = (
            "head" if sample["positive_item"] in prepared["heads"] else "tail"
        )
        for position, gold in enumerate(sequence[1:]):
            allowed = trie.get(sequence[: position + 1])
            if gold == eos or len(allowed) < 2:
                continue
            edge = (tuple(sequence[: position + 1]), gold)
            node = nodes[edge]
            state = F.normalize(hidden[batch_index, position], dim=0)
            rows.append(
                {
                    "sample_key": sample["sample_key"],
                    "user_id": sample["user_id"],
                    "target_group": target_group,
                    "depth": position,
                    "gold_token": gold,
                    "edge_id": edge_id(edge),
                    "legal_child_count": len(allowed),
                    "branch_bin": branch_bin(len(allowed)),
                    "descendant_items": node["descendant_items"],
                    "same_token_node_count": node["same_token_node_count"],
                    "semantic_distance": node["semantic_distance"],
                    "cohort": node["cohort"],
                    "gold_margin": gold_margin(
                        output.logits[batch_index, position], allowed, gold
                    ),
                }
            )
            states.append(state.cpu().numpy())
    return rows, states


def deterministic_take(rows: list[dict], count: int, salt: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}|{row['sample_key']}|{row['edge_id']}".encode()
        ).hexdigest(),
    )[:count]


def matched_readout(rows: list[dict], seed: int) -> dict:
    strata: dict[tuple, dict[str, list[dict]]] = defaultdict(
        lambda: {"high_polysemy": [], "control": []}
    )
    for row in rows:
        if row["cohort"] in ("high_polysemy", "control"):
            key = (row["depth"], row["target_group"], row["branch_bin"])
            strata[key][row["cohort"]].append(row)
    high, control, used = [], [], {}
    for key in sorted(strata):
        count = min(len(strata[key]["high_polysemy"]), len(strata[key]["control"]))
        if not count:
            continue
        salt = f"{seed}|{key}"
        high.extend(deterministic_take(strata[key]["high_polysemy"], count, salt))
        control.extend(deterministic_take(strata[key]["control"], count, salt))
        used[str(key)] = count
    high_rate = float(np.mean([row["gold_margin"] <= 0 for row in high])) if high else 0.0
    control_rate = (
        float(np.mean([row["gold_margin"] <= 0 for row in control]))
        if control
        else 0.0
    )
    return {
        "matched_per_cohort": len(high),
        "matched_strata": used,
        "high_polysemy_deficit_rate": high_rate,
        "control_deficit_rate": control_rate,
        "deficit_rate_difference": high_rate - control_rate,
        "high_polysemy_mean_margin": float(
            np.mean([row["gold_margin"] for row in high])
        )
        if high
        else 0.0,
        "control_mean_margin": float(
            np.mean([row["gold_margin"] for row in control])
        )
        if control
        else 0.0,
    }


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate([positive, negative])
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8)]
    )
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (
        rank_sum - len(positive) * (len(positive) + 1) / 2
    ) / (len(positive) * len(negative))


def state_separability(
    rows: list[dict], states: list[np.ndarray], config: dict
) -> dict:
    grouped: dict[tuple[int, int], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cap = int(config["state_audit"]["samples_per_node_cap"])
    for row, state in zip(rows, states):
        if row["cohort"] != "high_polysemy":
            continue
        grouped[(row["gold_token"], row["depth"])][row["edge_id"]].append(state)
    same, different = [], []
    eligible_groups = 0
    for key in sorted(grouped):
        node_states = {
            node: values[:cap]
            for node, values in grouped[key].items()
            if len(values) >= 2
        }
        if len(node_states) < 2:
            continue
        eligible_groups += 1
        for values in node_states.values():
            for left, right in combinations(values, 2):
                same.append(1.0 - float(np.dot(left, right)))
        node_names = sorted(node_states)
        for left_node, right_node in combinations(node_names, 2):
            for left in node_states[left_node]:
                for right in node_states[right_node]:
                    different.append(1.0 - float(np.dot(left, right)))
    pair_cap = int(config["state_audit"]["pair_cap"])
    same_array = np.asarray(same[:pair_cap], dtype=np.float64)
    different_array = np.asarray(different[:pair_cap], dtype=np.float64)
    auc = (
        rank_auc(different_array, same_array)
        if len(same_array) and len(different_array)
        else 0.0
    )
    return {
        "eligible_token_depth_groups": eligible_groups,
        "same_node_pairs": len(same_array),
        "different_node_pairs": len(different_array),
        "same_node_mean_cosine_distance": float(same_array.mean())
        if len(same_array)
        else 0.0,
        "different_node_mean_cosine_distance": float(different_array.mean())
        if len(different_array)
        else 0.0,
        "node_separation_auc": auc,
    }


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
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    item_vectors = catalog_semantic_vectors(prepared, config, device)
    nodes, semantic_groups = build_node_table(prepared, item_vectors, config)
    trie = gt.Trie(prepared["encoded_candidates"])
    item_to_sequence = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    all_samples = build_train_samples(
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    samples = select_unique_user_samples(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        int(config["head_samples"]),
        int(config["tail_samples"]),
    )
    rows, states = [], []
    batch_size = int(config["audit"]["batch_size"])
    for start in range(0, len(samples), batch_size):
        batch_rows, batch_states = audit_batch(
            samples[start : start + batch_size],
            prepared,
            trie,
            item_to_sequence,
            nodes,
            device,
        )
        rows.extend(batch_rows)
        states.extend(batch_states)
        done = min(start + batch_size, len(samples))
        if done % 64 == 0:
            print(
                f"LNDR_N1_PROGRESS dataset={dataset} samples={done}/{len(samples)}",
                flush=True,
            )
    output_dir = output_root / dataset
    step_path = output_dir / "audit_steps.csv"
    node_rows = [
        {
            "edge_id": edge_id(edge),
            "token": row["token"],
            "depth": row["depth"],
            "descendant_items": row["descendant_items"],
            "same_token_node_count": row["same_token_node_count"],
            "semantic_distance": row["semantic_distance"],
            "cohort": row["cohort"],
        }
        for edge, row in sorted(nodes.items(), key=lambda pair: edge_id(pair[0]))
    ]
    node_path = output_dir / "node_semantics.csv"
    write_csv(step_path, rows)
    write_csv(node_path, node_rows)
    high_rows = [row for row in rows if row["cohort"] == "high_polysemy"]
    supported_depths = Counter(row["depth"] for row in high_rows)
    semantic_scores = [
        row["semantic_distance"]
        for row in nodes.values()
        if row["semantic_distance"] is not None
    ]
    readout = matched_readout(rows, int(config["seed"]))
    state = state_separability(rows, states, config)
    gates = config["scientific_gates"]
    metrics = {
        "samples": len(samples),
        "unique_users": len({row["user_id"] for row in rows}),
        "competitive_steps": len(rows),
        "catalog_nodes": len(nodes),
        "reused_token_depth_groups": sum(
            len(edges) >= 2 for edges in semantic_groups.values()
        ),
        "semantic_node_occurrences": len(semantic_scores),
        "semantic_distance_median": float(np.median(semantic_scores))
        if semantic_scores
        else 0.0,
        "high_polysemy_steps": len(high_rows),
        "high_polysemy_supported_depths": {
            str(depth): count for depth, count in sorted(supported_depths.items())
        },
        "readout": readout,
        "state": state,
    }
    checks = {
        "semantic_node_occurrences": metrics["semantic_node_occurrences"]
        >= int(gates["semantic_node_occurrences_min"]),
        "semantic_distance_median": metrics["semantic_distance_median"]
        >= float(gates["semantic_distance_median_min"]),
        "high_polysemy_steps": len(high_rows)
        >= int(gates["high_polysemy_steps_min"]),
        "supported_depths": sum(
            count >= int(gates["high_polysemy_steps_per_depth_min"])
            for count in supported_depths.values()
        )
        >= int(gates["supported_depths_min"]),
        "state_groups": state["eligible_token_depth_groups"]
        >= int(gates["state_token_depth_groups_min"]),
        "state_pairs": min(state["same_node_pairs"], state["different_node_pairs"])
        >= int(gates["state_pairs_each_min"]),
        "state_separation_auc": state["node_separation_auc"]
        >= float(gates["state_separation_auc_min"]),
        "matched_readout_steps": readout["matched_per_cohort"]
        >= int(gates["matched_steps_per_cohort_min"]),
        "high_polysemy_deficit_rate": readout["high_polysemy_deficit_rate"]
        >= float(gates["high_polysemy_deficit_rate_min"]),
        "deficit_rate_difference": readout["deficit_rate_difference"]
        >= float(gates["deficit_rate_difference_min"]),
    }
    finite = all(
        math.isfinite(float(row["gold_margin"]))
        and (
            row["semantic_distance"] is None
            or math.isfinite(float(row["semantic_distance"]))
        )
        for row in rows
    )
    integrity = {
        "mapping_rate": 1.0,
        "trie_membership_rate": 1.0,
        "finite_rate": float(finite),
        "unique_user_rate": len({row["user_id"] for row in rows}) / len(samples),
        "optimizer_steps": 0,
        "parameter_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "validation_test_predictions_read": False,
        "sports_read": False,
        "checkpoint_sha256": checkpoint_sha,
        "step_file_sha256": sha256(step_path),
        "node_file_sha256": sha256(node_path),
    }
    integrity_valid = (
        integrity["mapping_rate"] == 1.0
        and integrity["trie_membership_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["unique_user_rate"] == 1.0
        and integrity["optimizer_steps"] == 0
        and integrity["parameter_sha_unchanged"]
        and not integrity["validation_test_predictions_read"]
        and not integrity["sports_read"]
    )
    del prepared, item_vectors, states
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }


def decision_markdown(summary: dict) -> str:
    lines = [
        "# LNDR-N1 Decision",
        "",
        f"- Fixed decision: **`{summary['decision']}`**",
        f"- Integrity valid: `{str(summary['integrity_valid']).lower()}`",
        "- Data scope: training prefixes and catalog metadata only",
        "- Validation/test/Sports read: `false`",
        "",
        "## Frozen gate results",
        "",
    ]
    for dataset, result in summary["results"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        for name, passed in result["checks"].items():
            lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
        lines.append("")
    lines.append(
        "Only `LNDR_S0_DESIGN_ALLOWED` permits a later correctness-smoke design; "
        "a failed scientific conjunction closes LNDR without rescue."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LNDR N1 requires CUDA")
    config = json.loads(args.config.read_text())
    expected_sha = config["integrity"]["code_sha256"]
    actual_sha = sha256(Path(__file__))
    if actual_sha != expected_sha:
        raise ValueError(f"code SHA mismatch expected={expected_sha} actual={actual_sha}")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(dataset, config, p0_config, args.output_root, device)
        for dataset in config["datasets"]
    }
    integrity_valid = all(row["integrity_valid"] for row in results.values())
    scientific_pass = all(row["scientific_pass"] for row in results.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "LNDR_S0_DESIGN_ALLOWED"
        if scientific_pass
        else "STOP_LNDR_NO_NODE_POLYSEMY_DEFICIT"
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
    (args.output_root / "decision.md").write_text(decision_markdown(summary))
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
