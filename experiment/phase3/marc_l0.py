#!/usr/bin/env python3
"""MARC L0: training-only counterfactual utility and critic audit.

The frozen GRAM checkpoint is evaluated on sequence[-3] targets with
sequence[:-3] histories.  sequence[-2:] is never read.  GRAM receives no
optimizer update.  A deterministic sklearn MLP is fit only after all frozen
counterfactual scores and target-free state features have been materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GRAM_SRC = ROOT / "GRAM/src"
for candidate in (GRAM_SRC, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from utils import indexing  # noqa: E402

from cgi_e0 import bootstrap_mean, write_csv  # noqa: E402
from hbtr_b1_smoke import (  # noqa: E402
    create_model_and_tokenizer,
    read_sequences,
    sha256,
)

SOURCE_FEATURES = [
    "depth",
    "log1p_child_count",
    "history_length",
    "metadata_tokens_mean",
    "metadata_tokens_min",
    "metadata_missing_fraction",
    "unique_neighbors_mean",
    "semantic_entropy",
    "semantic_margin",
    "semantic_max_probability",
    "collaborative_entropy",
    "collaborative_margin",
    "collaborative_max_probability",
    "full_entropy",
    "full_margin",
    "full_max_probability",
    "semantic_collaborative_js",
    "semantic_collaborative_top1_agree",
]

BUDGET_FEATURES = [
    "depth",
    "log1p_child_count",
    "history_length",
    "metadata_tokens_mean",
    "metadata_tokens_min",
    "metadata_missing_fraction",
    "unique_neighbors_mean",
    "candidate_k_scaled",
    "current_entropy",
    "current_margin",
    "current_max_probability",
    "semantic_entropy",
    "semantic_margin",
    "semantic_max_probability",
]

CONDITION_KEYS = (
    "semantic",
    "collaborative_baseline",
    "full5",
    "full10",
    "full20",
    "semantic_corrupt",
    "collaborative_baseline_corrupt",
    "full_baseline_semantic_corrupt",
    "full_baseline_collaborative_corrupt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase3/configs/marc_l0_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/marc_l0",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_MARC_L0报告.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def digest_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def user_split(seed: int, dataset: str, user: str, config: dict) -> str:
    bucket = digest_int(f"{seed}|{dataset}|{user}") % config["split_hash_modulus"]
    for name, spec in config["splits"].items():
        lower, upper = spec["hash_buckets"]
        if lower <= bucket < upper:
            return name
    raise ValueError(f"unassigned hash bucket: {bucket}")


def selection_hash(seed: int, dataset: str, split: str, user: str) -> str:
    return hashlib.sha256(
        f"{seed}|{dataset}|{split}|{user}".encode()
    ).hexdigest()


def read_keyed_text(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as handle:
        for line in handle:
            key, value = line.rstrip("\n").split(" ", 1)
            result[key] = value
    return result


def read_neighbors(path: Path) -> dict[str, list[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("anchor"):
                continue
            item, raw = line.rstrip("\n").split(" ", 1)
            result[item] = raw.split()
    return result


def training_popularity(sequences: dict[str, list[str]]) -> Counter:
    counts = Counter()
    for items in sequences.values():
        counts.update(items[:-2])
    return counts


def build_passage(
    item: str,
    item2lexid: dict[str, str],
    item_text: dict[str, str],
    neighbors: dict[str, list[str]],
    k: int,
    include_metadata: bool,
    donor_item: str | None = None,
    corrupt_source: str | None = None,
) -> str:
    link = item2lexid[item]
    metadata_item = donor_item if corrupt_source == "semantic" else item
    neighbor_item = donor_item if corrupt_source == "collaborative" else item
    parts = [f"item: {link}"]
    if k > 0:
        selected = neighbors[neighbor_item][:k]
        if len(selected) < k:
            raise ValueError(f"{neighbor_item} has fewer than {k} neighbors")
        if any(value not in item2lexid for value in selected):
            raise ValueError("neighbor outside lexical index")
        parts.append(
            "similar items: "
            + ", ".join(item2lexid[value] for value in selected)
        )
    if include_metadata:
        parts.append(item_text[metadata_item])
    return "; ".join(parts)


def donor_for_item(
    dataset: str,
    item: str,
    catalog: list[str],
    seed: int,
) -> str:
    start = digest_int(f"{seed}|{dataset}|donor|{item}") % len(catalog)
    for offset in range(len(catalog)):
        donor = catalog[(start + offset) % len(catalog)]
        if donor != item:
            return donor
    raise ValueError("catalog has no distinct corruption donor")


def build_item_conditions(
    dataset: str,
    item2input: dict[str, str],
    item2lexid: dict[str, str],
    item_text: dict[str, str],
    neighbors: dict[str, list[str]],
    config: dict,
) -> tuple[dict[str, dict[str, str]], dict]:
    catalog = sorted(
        set(item2lexid) & set(item_text) & set(neighbors)
    )
    conditions = {name: {} for name in CONDITION_KEYS}
    baseline_k = config["datasets"][dataset]["baseline_neighbor_budget"]
    current_identity = 0
    component_identity = 0
    for item in catalog:
        donor = donor_for_item(dataset, item, catalog, config["seed"])
        values = {
            "semantic": build_passage(
                item, item2lexid, item_text, neighbors, 0, True
            ),
            "collaborative_baseline": build_passage(
                item,
                item2lexid,
                item_text,
                neighbors,
                baseline_k,
                False,
            ),
            "full5": build_passage(
                item, item2lexid, item_text, neighbors, 5, True
            ),
            "full10": build_passage(
                item, item2lexid, item_text, neighbors, 10, True
            ),
            "full20": build_passage(
                item, item2lexid, item_text, neighbors, 20, True
            ),
            "semantic_corrupt": build_passage(
                item,
                item2lexid,
                item_text,
                neighbors,
                0,
                True,
                donor_item=donor,
                corrupt_source="semantic",
            ),
            "collaborative_baseline_corrupt": build_passage(
                item,
                item2lexid,
                item_text,
                neighbors,
                baseline_k,
                False,
                donor_item=donor,
                corrupt_source="collaborative",
            ),
            "full_baseline_semantic_corrupt": build_passage(
                item,
                item2lexid,
                item_text,
                neighbors,
                baseline_k,
                True,
                donor_item=donor,
                corrupt_source="semantic",
            ),
            "full_baseline_collaborative_corrupt": build_passage(
                item,
                item2lexid,
                item_text,
                neighbors,
                baseline_k,
                True,
                donor_item=donor,
                corrupt_source="collaborative",
            ),
        }
        for name, text in values.items():
            conditions[name][item] = text
        current_identity += int(
            values[f"full{baseline_k}"] == item2input[item]
        )
        component_identity += int(
            item2lexid[item] in values["full20"]
            and item_text[item] in values["full20"]
            and all(
                item2lexid[value] in values["full20"]
                for value in neighbors[item][:20]
            )
        )
    return conditions, {
        "catalog_size": len(catalog),
        "current_serialization_identity_rate": current_identity / len(catalog),
        "raw_component_identity_rate": component_identity / len(catalog),
    }


def build_cohort(
    dataset: str,
    sequences: dict[str, list[str]],
    valid_items: set[str],
    config: dict,
) -> tuple[list[dict], dict]:
    candidates = {name: [] for name in config["splits"]}
    rejected = Counter()
    offset = config["target_offset_from_end"]
    for user, items in sequences.items():
        if len(items) < offset + config["min_history"]:
            rejected["short_sequence"] += 1
            continue
        target = items[-offset]
        history = items[:-offset][-config["max_history"] :]
        if (
            target not in valid_items
            or len(history) < config["min_history"]
            or any(item not in valid_items for item in history)
        ):
            rejected["index_or_history"] += 1
            continue
        split = user_split(config["seed"], dataset, user, config)
        candidates[split].append(
            {
                "user": user,
                "split": split,
                "target": target,
                "history": history,
                "history_length": len(history),
                "sample_hash": selection_hash(
                    config["seed"], dataset, split, user
                ),
            }
        )
    selected = []
    available = {}
    for split, values in candidates.items():
        values.sort(key=lambda row: row["sample_hash"])
        available[split] = len(values)
        selected.extend(values[: config["splits"][split]["max_users"]])
    selected.sort(key=lambda row: (row["split"], row["sample_hash"]))
    return selected, {"available": available, "rejected": dict(rejected)}


def make_condition_samples(
    cohort: list[dict],
    condition_items: dict[str, str],
    item2lexid: dict[str, str],
) -> list[dict]:
    result = []
    for row in cohort:
        ordered = list(reversed(row["history"]))
        history_lex = " ; ".join(item2lexid[item] for item in ordered)
        result.append(
            {
                **row,
                "input": [f"What would user purchase after {history_lex} ?"]
                + [condition_items[item] for item in ordered],
                "output": item2lexid[row["target"]],
            }
        )
    return result


def encode_catalog_trie(
    collator: CollatorGRAM,
    item2lexid: dict[str, str],
) -> gt.Trie:
    sequences = []
    values = list(item2lexid.values())
    for start in range(0, len(values), 512):
        encoded = collator.encode_target_split(values[start : start + 512])
        for ids, mask in zip(
            encoded["input_ids"], encoded["attention_mask"].bool()
        ):
            target = ids[mask].tolist()
            sequences.append([0] + target)
    return gt.Trie(sequences)


def local_distribution(
    logits: torch.Tensor,
    labels: torch.Tensor,
    trie: gt.Trie,
    eos_token_id: int,
) -> tuple[list[dict], int, int]:
    rows = []
    checked = 0
    valid = 0
    prefix = [0]
    for depth, token_tensor in enumerate(labels):
        token = int(token_tensor.item())
        if token == -100:
            break
        children = trie.get(prefix)
        checked += 1
        if token in children:
            valid += 1
        else:
            raise ValueError(
                f"gold token {token} not in Trie children at prefix {prefix}"
            )
        child_logits = logits[depth, children].float()
        probs = torch.softmax(child_logits, dim=-1)
        log_probs = torch.log_softmax(child_logits, dim=-1)
        order = torch.argsort(probs, descending=True)
        top1 = int(children[int(order[0])])
        if len(children) > 1:
            margin = float(
                probs[int(order[0])].item() - probs[int(order[1])].item()
            )
        else:
            margin = 1.0
        gold_index = children.index(token)
        entropy = float(
            (-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item()
        )
        if token != eos_token_id:
            rows.append(
                {
                    "depth": depth,
                    "gold_log_probability": float(log_probs[gold_index].item()),
                    "entropy": entropy,
                    "margin": margin,
                    "max_probability": float(probs.max().item()),
                    "top1_token": top1,
                    "children": tuple(children),
                    "probabilities": probs.cpu().numpy(),
                }
            )
        prefix.append(token)
    return rows, checked, valid


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)
    left = np.sum(first * (np.log(first + 1e-12) - np.log(midpoint + 1e-12)))
    right = np.sum(
        second * (np.log(second + 1e-12) - np.log(midpoint + 1e-12))
    )
    return float(0.5 * (left + right))


@torch.no_grad()
def score_condition(
    model,
    collator,
    samples: list[dict],
    trie: gt.Trie,
    batch_size: int,
    device: torch.device,
    eos_token_id: int,
) -> tuple[dict[str, list[dict]], dict]:
    model.eval()
    results = {}
    checked = 0
    valid = 0
    finite = 0
    total_values = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batch = collator(
            [
                {
                    "input": row["input"],
                    "output": row["output"],
                    "user_id": row["user"],
                }
                for row in chunk
            ]
        )
        labels = batch["target_ids"].to(device)
        output = model(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            labels=labels,
            return_dict=True,
        )
        for index, sample in enumerate(chunk):
            node_rows, node_checked, node_valid = local_distribution(
                output.logits[index],
                labels[index],
                trie,
                eos_token_id,
            )
            results[sample["user"]] = node_rows
            checked += node_checked
            valid += node_valid
            values = [
                value
                for row in node_rows
                for key, value in row.items()
                if key
                in {
                    "gold_log_probability",
                    "entropy",
                    "margin",
                    "max_probability",
                }
            ]
            finite += sum(math.isfinite(float(value)) for value in values)
            total_values += len(values)
    return results, {
        "trie_checked": checked,
        "trie_valid": valid,
        "finite_values": finite,
        "total_values": total_values,
    }


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.tokenize(text))


def sample_static_features(
    sample: dict,
    tokenizer,
    item_text: dict[str, str],
    neighbors: dict[str, list[str]],
) -> dict[str, float]:
    metadata_counts = np.asarray(
        [token_count(tokenizer, item_text[item]) for item in sample["history"]],
        dtype=np.float64,
    )
    unique_neighbors = np.asarray(
        [len(set(neighbors[item][:20])) for item in sample["history"]],
        dtype=np.float64,
    )
    return {
        "history_length": float(sample["history_length"]),
        "metadata_tokens_mean": float(metadata_counts.mean()),
        "metadata_tokens_min": float(metadata_counts.min()),
        "metadata_missing_fraction": float((metadata_counts == 0).mean()),
        "unique_neighbors_mean": float(unique_neighbors.mean()),
    }


def condition_at_budget(scores: dict, budget: int) -> list[dict]:
    return scores["semantic"] if budget == 0 else scores[f"full{budget}"]


def make_node_rows(
    dataset: str,
    cohort: list[dict],
    all_scores: dict[str, dict[str, list[dict]]],
    tokenizer,
    item_text: dict[str, str],
    neighbors: dict[str, list[str]],
    baseline_k: int,
) -> list[dict]:
    rows = []
    for sample in cohort:
        user = sample["user"]
        per_condition = {
            name: all_scores[name][user] for name in CONDITION_KEYS
        }
        lengths = {name: len(value) for name, value in per_condition.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"condition target length mismatch: {lengths}")
        static = sample_static_features(
            sample, tokenizer, item_text, neighbors
        )
        for index in range(next(iter(lengths.values()))):
            sem = per_condition["semantic"][index]
            cf = per_condition["collaborative_baseline"][index]
            full20 = per_condition["full20"][index]
            full_source = per_condition[f"full{baseline_k}"][index]
            if not (
                sem["children"]
                == cf["children"]
                == full20["children"]
                == full_source["children"]
            ):
                raise ValueError("Trie children changed across conditions")
            budget_lp = {
                0: sem["gold_log_probability"],
                5: per_condition["full5"][index]["gold_log_probability"],
                10: per_condition["full10"][index]["gold_log_probability"],
                20: full20["gold_log_probability"],
            }
            row = {
                "dataset": dataset,
                "user": user,
                "split": sample["split"],
                "sample_hash": sample["sample_hash"],
                "depth": int(sem["depth"]),
                "child_count": len(sem["children"]),
                "log1p_child_count": math.log1p(len(sem["children"])),
                **static,
                "semantic_entropy": sem["entropy"],
                "semantic_margin": sem["margin"],
                "semantic_max_probability": sem["max_probability"],
                "collaborative_entropy": cf["entropy"],
                "collaborative_margin": cf["margin"],
                "collaborative_max_probability": cf["max_probability"],
                "full_entropy": full_source["entropy"],
                "full_margin": full_source["margin"],
                "full_max_probability": full_source["max_probability"],
                "semantic_collaborative_js": js_divergence(
                    sem["probabilities"], cf["probabilities"]
                ),
                "semantic_collaborative_top1_agree": float(
                    sem["top1_token"] == cf["top1_token"]
                ),
                "lp_semantic": budget_lp[0],
                "lp_collaborative": cf["gold_log_probability"],
                "lp_full_source": full_source["gold_log_probability"],
                "lp_full5": budget_lp[5],
                "lp_full10": budget_lp[10],
                "lp_full20": budget_lp[20],
                "u_semantic": (
                    full_source["gold_log_probability"]
                    - cf["gold_log_probability"]
                ),
                "u_collaborative": (
                    full_source["gold_log_probability"]
                    - sem["gold_log_probability"]
                ),
                "u_k5": budget_lp[5] - budget_lp[0],
                "u_k10": budget_lp[10] - budget_lp[5],
                "u_k20": budget_lp[20] - budget_lp[10],
            }
            rows.append(row)
    return rows


def source_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in SOURCE_FEATURES] for row in rows],
        dtype=np.float64,
    )


def budget_feature_row(
    row: dict,
    condition: dict,
    candidate_k: int,
) -> list[float]:
    values = {
        **row,
        "candidate_k_scaled": candidate_k / 20.0,
        "current_entropy": condition["entropy"],
        "current_margin": condition["margin"],
        "current_max_probability": condition["max_probability"],
    }
    return [float(values[name]) for name in BUDGET_FEATURES]


def make_regressor(config: dict) -> Pipeline:
    spec = config["critic"]
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(spec["hidden_layer_sizes"]),
                    activation=spec["activation"],
                    solver=spec["solver"],
                    alpha=spec["alpha"],
                    max_iter=spec["max_iter"],
                    learning_rate_init=spec["learning_rate_init"],
                    early_stopping=spec["early_stopping"],
                    validation_fraction=spec["validation_fraction"],
                    n_iter_no_change=spec["n_iter_no_change"],
                    random_state=spec["random_state"],
                ),
            ),
        ]
    )


def make_classifier(config: dict) -> Pipeline:
    spec = config["critic"]
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(spec["hidden_layer_sizes"]),
                    activation=spec["activation"],
                    solver=spec["solver"],
                    alpha=spec["alpha"],
                    max_iter=spec["max_iter"],
                    learning_rate_init=spec["learning_rate_init"],
                    early_stopping=spec["early_stopping"],
                    validation_fraction=spec["validation_fraction"],
                    n_iter_no_change=spec["n_iter_no_change"],
                    random_state=spec["random_state"],
                ),
            ),
        ]
    )


def sigmoid_calibrate(
    fit_model: Pipeline,
    calibration_rows: list[dict],
    target: str,
) -> LogisticRegression:
    raw = fit_model.predict_proba(source_matrix(calibration_rows))[:, 1]
    labels = np.asarray(
        [float(row[target]) > 0 for row in calibration_rows], dtype=np.int64
    )
    if len(np.unique(labels)) != 2:
        raise ValueError(f"calibration split has one class for {target}")
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=20260724
    )
    model.fit(raw.reshape(-1, 1), labels)
    return model


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            mask = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(probabilities[mask].mean())
                - float(labels[mask].mean())
            )
    return value


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    observed = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < iterations:
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            values.append(
                roc_auc_score(labels[indices], scores[indices])
            )
    return {
        "value": observed,
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def aggregate_sample_utilities(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["user"]].append(row)
    result = {}
    for user, values in grouped.items():
        result[user] = {
            "semantic": float(np.mean([row["u_semantic"] for row in values])),
            "collaborative": float(
                np.mean([row["u_collaborative"] for row in values])
            ),
        }
    return result


def analyze_l0a(rows: list[dict], config: dict) -> dict:
    sample_utilities = aggregate_sample_utilities(rows)
    heterogeneity = {}
    for source in ("semantic", "collaborative"):
        values = np.asarray(
            [value[source] for value in sample_utilities.values()]
        )
        heterogeneity[source] = {
            "positive_rate": float((values > 0).mean()),
            "negative_rate": float((values < 0).mean()),
            "mean": float(values.mean()),
        }
    lps = np.asarray(
        [
            [
                row["lp_semantic"],
                row["lp_full5"],
                row["lp_full10"],
                row["lp_full20"],
            ]
            for row in rows
        ]
    )
    oracle = lps.max(axis=1)
    fixed = lps[:, 3]
    fixed_ce = float((-fixed).mean())
    oracle_ce = float((-oracle).mean())
    relative_reduction = (
        (fixed_ce - oracle_ce) / fixed_ce if fixed_ce > 0 else 0.0
    )
    oracle_actions = lps.argmax(axis=1)
    k20_dominance = float((oracle_actions == 3).mean())
    depths = sorted({int(row["depth"]) for row in rows})
    table = []
    depth_modes = {}
    for depth in depths:
        mask = np.asarray([int(row["depth"]) == depth for row in rows])
        counts = np.bincount(oracle_actions[mask], minlength=4)
        table.append(counts)
        depth_modes[str(depth)] = int(np.argmax(counts))
    nonempty_table = np.asarray(
        [row for row in table if np.asarray(row).sum() > 0]
    )
    if nonempty_table.shape[0] > 1:
        _, p_value, _, _ = chi2_contingency(nonempty_table)
    else:
        p_value = 1.0
    gates = config["l0a_gates_per_dataset"]
    source_heterogeneity = all(
        heterogeneity[source]["positive_rate"]
        >= gates["source_positive_sample_rate_min"]
        and heterogeneity[source]["negative_rate"]
        >= gates["source_negative_sample_rate_min"]
        for source in heterogeneity
    )
    gate_values = {
        "source_heterogeneity": source_heterogeneity,
        "oracle_headroom": relative_reduction
        >= gates["oracle_relative_ce_reduction_min"],
        "depth_action_heterogeneity": p_value
        <= gates["depth_action_chi2_p_max"]
        and len(set(depth_modes.values()))
        >= gates["distinct_modal_actions_min"],
        "k20_not_dominant": k20_dominance
        <= gates["k20_oracle_dominance_max"],
    }
    return {
        "sample_source_utility": heterogeneity,
        "oracle": {
            "fixed_k20_ce": fixed_ce,
            "oracle_ce": oracle_ce,
            "relative_ce_reduction": relative_reduction,
            "k20_dominance_rate": k20_dominance,
            "depth_action_chi2_p": float(p_value),
            "depth_modal_actions": depth_modes,
        },
        "gates": gate_values,
    }


def corrupted_source_features(
    base_row: dict,
    sem: dict,
    cf: dict,
    full: dict,
) -> list[float]:
    values = {
        **base_row,
        "semantic_entropy": sem["entropy"],
        "semantic_margin": sem["margin"],
        "semantic_max_probability": sem["max_probability"],
        "collaborative_entropy": cf["entropy"],
        "collaborative_margin": cf["margin"],
        "collaborative_max_probability": cf["max_probability"],
        "full_entropy": full["entropy"],
        "full_margin": full["margin"],
        "full_max_probability": full["max_probability"],
        "semantic_collaborative_js": js_divergence(
            sem["probabilities"], cf["probabilities"]
        ),
        "semantic_collaborative_top1_agree": float(
            sem["top1_token"] == cf["top1_token"]
        ),
    }
    return [float(values[name]) for name in SOURCE_FEATURES]


def analyze_l0b(
    rows: list[dict],
    cohort: list[dict],
    all_scores: dict[str, dict[str, list[dict]]],
    config: dict,
) -> dict:
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in config["splits"]
    }
    result = {"sources": {}}
    convergence_flags = []
    source_models = {}
    for offset, (source, target) in enumerate(
        (
            ("semantic", "u_semantic"),
            ("collaborative", "u_collaborative"),
        )
    ):
        fit = split_rows["fit"]
        calibration = split_rows["calibration"]
        audit = split_rows["audit"]
        y_fit = np.asarray([row[target] > 0 for row in fit], dtype=np.int64)
        y_audit = np.asarray(
            [row[target] > 0 for row in audit], dtype=np.int64
        )
        if len(np.unique(y_fit)) != 2 or len(np.unique(y_audit)) != 2:
            result["sources"][source] = {
                "status": "ONE_CLASS",
                "gate": False,
            }
            continue
        classifier = make_classifier(config)
        classifier.fit(source_matrix(fit), y_fit)
        convergence_flags.append(
            classifier.named_steps["mlp"].n_iter_
            < config["critic"]["max_iter"]
        )
        calibrator = sigmoid_calibrate(
            classifier, calibration, target
        )
        raw_audit = classifier.predict_proba(source_matrix(audit))[:, 1]
        probabilities = calibrator.predict_proba(
            raw_audit.reshape(-1, 1)
        )[:, 1]
        actual = np.asarray([row[target] for row in audit], dtype=np.float64)
        active = probabilities >= 0.5
        auc = bootstrap_auc(
            y_audit,
            probabilities,
            config["bootstrap_iterations"],
            config["seed"] + offset,
        )
        active_stats = (
            bootstrap_mean(
                actual[active],
                config["bootstrap_iterations"],
                config["seed"] + 10 + offset,
            )
            if active.any()
            else {
                "mean": float("nan"),
                "median": float("nan"),
                "positive_rate": 0.0,
                "ci95": [float("nan"), float("nan")],
            }
        )
        regression = make_regressor(config)
        regression.fit(source_matrix(fit), np.asarray([row[target] for row in fit]))
        convergence_flags.append(
            regression.named_steps["mlp"].n_iter_
            < config["critic"]["max_iter"]
        )
        predicted_utility = regression.predict(source_matrix(audit))
        correlation = spearmanr(actual, predicted_utility)
        source_models[source] = {
            "classifier": classifier,
            "calibrator": calibrator,
            "regressor": regression,
        }
        result["sources"][source] = {
            "status": "ANALYZED",
            "auroc": auc,
            "brier": float(brier_score_loss(y_audit, probabilities)),
            "ece10": ece(y_audit, probabilities),
            "spearman": {
                "rho": float(correlation.statistic),
                "p": float(correlation.pvalue),
            },
            "active_count": int(active.sum()),
            "active_coverage": float(active.mean()),
            "active_actual_utility": active_stats,
        }

    budget_fit_features = []
    budget_fit_targets = []
    budget_audit_by_user_depth = {}
    row_lookup = {(row["user"], row["depth"]): row for row in rows}
    for sample in cohort:
        user = sample["user"]
        for index, semantic in enumerate(all_scores["semantic"][user]):
            row = row_lookup[(user, semantic["depth"])]
            conditions = {
                5: all_scores["semantic"][user][index],
                10: all_scores["full5"][user][index],
                20: all_scores["full10"][user][index],
            }
            targets = {5: row["u_k5"], 10: row["u_k10"], 20: row["u_k20"]}
            feature_rows = {
                k: budget_feature_row(row, condition, k)
                for k, condition in conditions.items()
            }
            if sample["split"] == "fit":
                budget_fit_features.extend(feature_rows.values())
                budget_fit_targets.extend(targets.values())
            elif sample["split"] == "audit":
                budget_audit_by_user_depth[(user, semantic["depth"])] = {
                    "features": feature_rows,
                    "row": row,
                }
    budget_model = make_regressor(config)
    budget_model.fit(
        np.asarray(budget_fit_features), np.asarray(budget_fit_targets)
    )
    convergence_flags.append(
        budget_model.named_steps["mlp"].n_iter_
        < config["critic"]["max_iter"]
    )
    learned_regret = []
    fixed_regret = []
    for value in budget_audit_by_user_depth.values():
        feature_rows = value["features"]
        row = value["row"]
        predicted = {
            k: float(
                budget_model.predict(
                    np.asarray(feature_rows[k], dtype=np.float64).reshape(1, -1)
                )[0]
            )
            for k in (5, 10, 20)
        }
        predicted_scores = np.asarray(
            [
                0.0,
                predicted[5],
                predicted[5] + predicted[10],
                predicted[5] + predicted[10] + predicted[20],
            ]
        )
        actual_scores = np.asarray(
            [
                row["lp_semantic"],
                row["lp_full5"],
                row["lp_full10"],
                row["lp_full20"],
            ]
        )
        chosen = int(np.argmax(predicted_scores))
        oracle = float(actual_scores.max())
        learned_regret.append(oracle - float(actual_scores[chosen]))
        fixed_regret.append(oracle - float(actual_scores[3]))
    learned_regret_mean = float(np.mean(learned_regret))
    fixed_regret_mean = float(np.mean(fixed_regret))
    regret_ratio = (
        learned_regret_mean / fixed_regret_mean
        if fixed_regret_mean > 1e-12
        else float("inf")
    )
    result["budget"] = {
        "learned_regret": learned_regret_mean,
        "fixed_k20_regret": fixed_regret_mean,
        "regret_ratio": regret_ratio,
    }

    audit_samples = [row for row in cohort if row["split"] == "audit"]
    corruption = {}
    for source in ("semantic", "collaborative"):
        if source not in source_models:
            corruption[source] = {
                "status": "NO_SOURCE_MODEL",
                "predicted_utility_drop": float("nan"),
            }
            continue
        clean_features = []
        corrupt_features = []
        for sample in audit_samples:
            user = sample["user"]
            for index, sem in enumerate(all_scores["semantic"][user]):
                base = row_lookup[(user, sem["depth"])]
                cf = all_scores["collaborative_baseline"][user][index]
                if source == "semantic":
                    corrupt_sem = all_scores["semantic_corrupt"][user][index]
                    corrupt_full = all_scores[
                        "full_baseline_semantic_corrupt"
                    ][user][index]
                    corrupt_cf = cf
                else:
                    corrupt_sem = sem
                    corrupt_cf = all_scores[
                        "collaborative_baseline_corrupt"
                    ][user][index]
                    corrupt_full = all_scores[
                        "full_baseline_collaborative_corrupt"
                    ][user][index]
                clean_features.append([base[name] for name in SOURCE_FEATURES])
                corrupt_features.append(
                    corrupted_source_features(
                        base, corrupt_sem, corrupt_cf, corrupt_full
                    )
                )
        regression = source_models[source]["regressor"]
        clean_prediction = regression.predict(np.asarray(clean_features))
        corrupt_prediction = regression.predict(np.asarray(corrupt_features))
        drop = clean_prediction - corrupt_prediction
        corruption[source] = {
            "status": "ANALYZED",
            "predicted_utility_drop": float(drop.mean()),
            "positive_drop_rate": float((drop > 0).mean()),
        }
    result["corruption"] = corruption

    gates = config["l0b_gates_per_dataset"]
    source_gates = {}
    for source in ("semantic", "collaborative"):
        metrics = result["sources"].get(source, {})
        stats = metrics.get("active_actual_utility", {})
        source_gates[source] = (
            metrics.get("status") == "ANALYZED"
            and metrics["auroc"]["value"]
            >= gates["source_sign_auroc_min"]
            and metrics["active_coverage"] >= gates["active_coverage_min"]
            and stats["ci95"][0]
            > gates[
                "predicted_positive_actual_utility_ci95_lower_strictly_greater_than"
            ]
        )
    result["gates"] = {
        "semantic_predictability": source_gates["semantic"],
        "collaborative_predictability": source_gates["collaborative"],
        "budget_regret": regret_ratio <= gates["budget_regret_ratio_max"],
        "semantic_corruption_direction": corruption["semantic"][
            "predicted_utility_drop"
        ]
        > 0,
        "collaborative_corruption_direction": corruption["collaborative"][
            "predicted_utility_drop"
        ]
        > 0,
    }
    result["critic_converged"] = all(convergence_flags)
    return result


def preflight_dataset(
    dataset: str,
    spec: dict,
    config: dict,
) -> tuple[dict, dict]:
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-small", local_files_only=True
    )
    from hbtr_b1_smoke import make_runtime_args

    runtime = make_runtime_args(dataset)
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
    item_text = read_keyed_text(dataset_dir / "item_plain_text.txt")
    neighbors = read_neighbors(dataset_dir / "similar_item_sasrec.txt")
    conditions, item_integrity = build_item_conditions(
        dataset,
        item2input,
        item2lexid,
        item_text,
        neighbors,
        config,
    )
    valid_items = (
        set(item2lexid)
        & set(item2input)
        & set(item_text)
        & set(neighbors)
        & set(conditions["full20"])
    )
    cohort, availability = build_cohort(
        dataset, sequences, valid_items, config
    )
    counts = Counter(row["split"] for row in cohort)
    expected = {
        split: value["max_users"]
        for split, value in config["splits"].items()
    }
    user_sets = {
        split: {row["user"] for row in cohort if row["split"] == split}
        for split in expected
    }
    overlap = sum(
        len(user_sets[left] & user_sets[right])
        for left, right in (
            ("fit", "calibration"),
            ("fit", "audit"),
            ("calibration", "audit"),
        )
    )
    paths = {
        "checkpoint": ROOT / spec["checkpoint"],
        "run_config": ROOT / spec["run_config"],
        "user_sequence": dataset_dir / "user_sequence.txt",
        "item_index": dataset_dir
        / (
            "item_generative_indexing_"
            + runtime.hierarchical_id_type
            + ".txt"
        ),
        "item_text": dataset_dir / "item_plain_text.txt",
        "neighbors": dataset_dir / "similar_item_sasrec.txt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    integrity = {
        **item_integrity,
        "exact_total_users": len(cohort),
        "exact_split_caps": dict(counts) == expected,
        "user_overlap_across_splits": overlap,
        "heldout_sequence_fields_read": False,
        "target_feature_inclusion_rate": 0.0,
        "model_optimizer_steps": 0,
    }
    return {
        "tokenizer": tokenizer,
        "runtime": runtime,
        "sequences": sequences,
        "item2input": item2input,
        "item2lexid": item2lexid,
        "item_text": item_text,
        "neighbors": neighbors,
        "conditions": conditions,
        "cohort": cohort,
    }, {
        "counts": dict(counts),
        "availability": availability,
        "integrity": integrity,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }


def integrity_pass(result: dict, config: dict) -> bool:
    actual = result["integrity"]
    required = config["integrity_gates"]
    return (
        actual["exact_total_users"] == required["exact_total_users"]
        and actual["exact_split_caps"]
        and actual["user_overlap_across_splits"]
        == required["user_overlap_across_splits"]
        and actual["heldout_sequence_fields_read"]
        == required["heldout_sequence_fields_read"]
        and actual["target_feature_inclusion_rate"]
        == required["target_feature_inclusion_rate"]
        and actual["current_serialization_identity_rate"]
        == required["current_serialization_identity_rate"]
        and actual.get("trie_child_membership_rate", 0.0)
        == required["trie_child_membership_rate"]
        and actual.get("finite_rate", 0.0) == required["finite_rate"]
        and actual["model_optimizer_steps"]
        == required["model_optimizer_steps"]
        and actual.get("critic_converged", False)
        == required["critic_converged"]
    )


def decide(results: dict, config: dict) -> str:
    if not all(integrity_pass(value, config) for value in results.values()):
        return "EXECUTION_INVALID"
    if not all(
        all(value["l0a"]["gates"].values()) for value in results.values()
    ):
        return "STOP_MARC_NO_UTILITY_HETEROGENEITY"
    if not all(
        all(value["l0b"]["gates"].values()) for value in results.values()
    ):
        return "STOP_MARC_UTILITY_NOT_LEARNABLE"
    return "MARC_L1_DESIGN_ALLOWED"


def write_report(path: Path, aggregate: dict) -> None:
    lines = [
        "# GRAM 第三阶段：MARC L0 反事实效用与 critic 可学习性报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-24",
        "- Verification Status: ANALYZED",
        "- Version Label: `marc_l0_v1`",
        "",
        f"固定决策：**`{aggregate['decision']}`**。",
        "",
        "仅使用 `sequence[-3]` training target 与 `sequence[:-3]` history；",
        "未读取 validation/test，未更新 GRAM，未运行 beam 或 RL。",
        "",
        "## 结果",
        "",
        "| Dataset | Integrity | L0-A | L0-B | Oracle CE reduction | K20 dominance | Sem AUROC | CF AUROC | Budget regret ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        oracle = result["l0a"]["oracle"]
        sources = result["l0b"]["sources"]
        lines.append(
            f"| {dataset} | {result['integrity_pass']} | "
            f"{all(result['l0a']['gates'].values())} | "
            f"{all(result['l0b']['gates'].values())} | "
            f"{oracle['relative_ce_reduction']:.6f} | "
            f"{oracle['k20_dominance_rate']:.6f} | "
            f"{sources['semantic'].get('auroc', {}).get('value', float('nan')):.6f} | "
            f"{sources['collaborative'].get('auroc', {}).get('value', float('nan')):.6f} | "
            f"{result['l0b']['budget']['regret_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "L0 只判断 utility 是否异质且能否由 target-free state 预测；",
            "它不证明 MARC 会改善 Recall/NDCG。若固定决策为 STOP，L1、RL、",
            "二次 refinement 与 validation 均不解锁。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_training_prefix_scores"):
        raise ValueError("MARC L0 was not preregistered before scoring")
    started = time.time()
    prepared = {}
    preflight = {}
    for dataset, spec in config["datasets"].items():
        prepared[dataset], preflight[dataset] = preflight_dataset(
            dataset, spec, config
        )
        output_dir = args.output_root / dataset
        write_csv(
            output_dir / "cohort.csv",
            prepared[dataset]["cohort"],
            [
                "user",
                "split",
                "target",
                "history_length",
                "sample_hash",
            ],
        )
    if args.preflight_only:
        summary = {
            "material_passport": {
                "origin_skill": "academic-research-suite / experiment-agent",
                "origin_mode": "run",
                "origin_date": "2026-07-24",
                "verification_status": "ANALYZED_PREFLIGHT_ONLY",
                "version_label": "marc_l0_preflight_v1",
            },
            "decision": "PREFLIGHT_COMPLETE_SCORING_NOT_RUN",
            "datasets": preflight,
            "config_sha256": sha256(config_path),
            "code_sha256": sha256(Path(__file__)),
            "wall_time_seconds": time.time() - started,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "preflight_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({"decision": summary["decision"]}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("MARC L0 frozen-checkpoint scoring requires CUDA")

    results = {}
    device = torch.device("cuda:0")
    for dataset, spec in config["datasets"].items():
        data = prepared[dataset]
        model, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
        if any(parameter.requires_grad is False for parameter in model.parameters()):
            pass
        model.eval()
        collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
        trie = encode_catalog_trie(collator, data["item2lexid"])
        all_scores = {}
        score_integrity = Counter()
        baseline_k = spec["baseline_neighbor_budget"]
        for condition in CONDITION_KEYS:
            samples = make_condition_samples(
                data["cohort"],
                data["conditions"][condition],
                data["item2lexid"],
            )
            scores, audit = score_condition(
                model,
                collator,
                samples,
                trie,
                config["batch_size"],
                device,
                tokenizer.eos_token_id,
            )
            all_scores[condition] = scores
            score_integrity.update(audit)
        node_rows = make_node_rows(
            dataset,
            data["cohort"],
            all_scores,
            tokenizer,
            data["item_text"],
            data["neighbors"],
            baseline_k,
        )
        if not all(
            math.isfinite(float(row[name]))
            for row in node_rows
            for name in (
                SOURCE_FEATURES
                + [
                    "lp_semantic",
                    "lp_collaborative",
                    "lp_full_source",
                    "lp_full5",
                    "lp_full10",
                    "lp_full20",
                    "u_semantic",
                    "u_collaborative",
                    "u_k5",
                    "u_k10",
                    "u_k20",
                ]
            )
        ):
            raise ValueError("non-finite MARC node feature or utility")
        l0a = analyze_l0a(node_rows, config)
        l0b = analyze_l0b(
            node_rows, data["cohort"], all_scores, config
        )
        integrity = dict(preflight[dataset]["integrity"])
        integrity["trie_child_membership_rate"] = (
            score_integrity["trie_valid"]
            / score_integrity["trie_checked"]
        )
        integrity["finite_rate"] = (
            score_integrity["finite_values"]
            / score_integrity["total_values"]
        )
        integrity["critic_converged"] = l0b["critic_converged"]
        result = {
            **preflight[dataset],
            "integrity": integrity,
            "l0a": l0a,
            "l0b": l0b,
            "node_count": len(node_rows),
        }
        result["integrity_pass"] = integrity_pass(result, config)
        results[dataset] = result
        output_dir = args.output_root / dataset
        write_csv(
            output_dir / "node_utilities.csv",
            node_rows,
            [
                "dataset",
                "user",
                "split",
                "sample_hash",
                "depth",
                "child_count",
            ]
            + SOURCE_FEATURES[2:]
            + [
                "lp_semantic",
                "lp_collaborative",
                "lp_full_source",
                "lp_full5",
                "lp_full10",
                "lp_full20",
                "u_semantic",
                "u_collaborative",
                "u_k5",
                "u_k10",
                "u_k20",
            ],
        )
        (output_dir / "diagnostic_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        del model
        torch.cuda.empty_cache()

    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run + validate",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "marc_l0_v1",
        },
        "decision": decide(results, config),
        "datasets": results,
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "wall_time_seconds": time.time() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(args.report, aggregate)
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "wall_time_seconds": aggregate["wall_time_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
