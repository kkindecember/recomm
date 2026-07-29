#!/usr/bin/env python3
"""SCDL N1: frozen catalog-only sibling lexicalization audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mapping(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as handle:
        for line in handle:
            key, value = line.rstrip("\n").split(" ", 1)
            result[key] = value
    return result


def encoded_ids(tokenizer, lexical_id: str, excluded: set[int]) -> list[int]:
    return [token for token in tokenizer.encode(lexical_id) if token not in excluded]


def build_tfidf(
    tokenizer,
    texts: list[str],
    maximum_tokens: int,
    excluded: set[int],
) -> csr_matrix:
    rows, cols, values = [], [], []
    for start in range(0, len(texts), 256):
        batch = tokenizer(
            texts[start : start + 256],
            truncation=True,
            max_length=maximum_tokens,
            add_special_tokens=False,
        )["input_ids"]
        for offset, tokens in enumerate(batch):
            counts = Counter(token for token in tokens if token not in excluded)
            for token, count in counts.items():
                rows.append(start + offset)
                cols.append(token)
                values.append(count)
    counts = csr_matrix(
        (values, (rows, cols)),
        shape=(len(texts), int(tokenizer.vocab_size)),
        dtype=np.float64,
    )
    return TfidfTransformer(norm="l2", smooth_idf=True).fit_transform(counts).tocsr()


def build_sibling_sets(
    sequences: list[list[int]], minimum_descendants: int
) -> list[tuple[tuple[int, ...], dict[int, list[int]]]]:
    descendants: dict[tuple[tuple[int, ...], int], list[int]] = defaultdict(list)
    for item_index, sequence in enumerate(sequences):
        parent = (0,)
        for child in sequence:
            descendants[(parent, child)].append(item_index)
            parent = (*parent, child)
    parents: dict[tuple[int, ...], dict[int, list[int]]] = defaultdict(dict)
    for (parent, child), items in descendants.items():
        if len(items) >= minimum_descendants:
            parents[parent][child] = items
    return [
        (parent, children)
        for parent, children in parents.items()
        if len(children) >= 2
    ]


def top_indices(row: np.ndarray, count: int, excluded: set[int]) -> np.ndarray:
    values = row.copy()
    if excluded:
        values[np.fromiter(excluded, dtype=np.int64)] = -np.inf
    candidates = np.flatnonzero(np.isfinite(values) & (values > 0))
    count = min(count, len(candidates))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    local = np.argpartition(values[candidates], -count)[-count:]
    selected = candidates[local]
    return selected[np.argsort(-values[selected], kind="mergesort")]


def current_metrics(weights: np.ndarray, current_tokens: list[int]) -> dict:
    own, margins = [], []
    for row, token in enumerate(current_tokens):
        token_scores = weights[:, token]
        competitors = np.delete(token_scores, row)
        own.append(float(token_scores[row]))
        margins.append(float(token_scores[row] - competitors.max()))
    return {"own": np.asarray(own), "margins": np.asarray(margins)}


def joint_assignment(
    weights: np.ndarray,
    candidate_count: int,
    excluded: set[int],
    represent_weight: float,
    contrast_weight: float,
) -> dict:
    candidates_by_child = [
        top_indices(weights[row], candidate_count, excluded)
        for row in range(weights.shape[0])
    ]
    union = sorted({int(token) for values in candidates_by_child for token in values})
    if len(union) < weights.shape[0]:
        return {"feasible": False}
    allowed = [
        {int(token) for token in values}
        for values in candidates_by_child
    ]
    objective = np.full((weights.shape[0], len(union)), -1e9, dtype=np.float64)
    margins = np.empty_like(objective)
    for column, token in enumerate(union):
        token_scores = weights[:, token]
        for row in range(weights.shape[0]):
            competitor = float(np.delete(token_scores, row).max())
            margins[row, column] = float(token_scores[row] - competitor)
            if token in allowed[row]:
                objective[row, column] = (
                    represent_weight * float(token_scores[row])
                    + contrast_weight * margins[row, column]
                )
    row_indices, columns = linear_sum_assignment(-objective)
    if len(row_indices) != weights.shape[0] or any(
        union[column] not in allowed[row]
        for row, column in zip(row_indices, columns)
    ):
        return {"feasible": False}
    tokens = [0] * weights.shape[0]
    own = np.zeros(weights.shape[0])
    selected_margins = np.zeros(weights.shape[0])
    for row, column in zip(row_indices, columns):
        token = union[column]
        tokens[row] = token
        own[row] = weights[row, token]
        selected_margins[row] = margins[row, column]
    return {
        "feasible": True,
        "tokens": tokens,
        "own": own,
        "margins": selected_margins,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit_dataset(dataset: str, spec: dict, config: dict, output_root: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    text_map = read_mapping(dataset_dir / "item_plain_text.txt")
    lexical_path = dataset_dir / (
        f"item_generative_indexing_{spec['hierarchical_id_type']}.txt"
    )
    lexical_map = read_mapping(lexical_path)
    if set(text_map) != set(lexical_map):
        raise ValueError(f"{dataset} catalog/text mapping mismatch")
    catalog = list(lexical_map)
    excluded = {int(value) for value in config["assignment"]["exclude_special_token_ids"]}
    sequences = [
        encoded_ids(tokenizer, lexical_map[item], excluded) for item in catalog
    ]
    if any(not sequence for sequence in sequences):
        raise ValueError(f"{dataset} empty lexical sequence")
    unique_identifier_rate = len({tuple(row) for row in sequences}) / len(sequences)
    tfidf = build_tfidf(
        tokenizer,
        [text_map[item] for item in catalog],
        int(config["tfidf"]["metadata_max_tokens"]),
        excluded,
    )
    sibling_sets = build_sibling_sets(
        sequences, int(config["assignment"]["minimum_descendant_items"])
    )
    set_rows, child_rows = [], []
    for set_index, (parent, children) in enumerate(sibling_sets):
        child_tokens = sorted(children)
        weights = np.stack(
            [
                np.asarray(tfidf[children[token]].mean(axis=0)).ravel()
                for token in child_tokens
            ]
        )
        current = current_metrics(weights, child_tokens)
        assigned = joint_assignment(
            weights,
            int(config["assignment"]["candidate_tokens_per_child"]),
            excluded,
            float(config["assignment"]["representativeness_weight"]),
            float(config["assignment"]["contrast_weight"]),
        )
        set_id = hashlib.sha256(
            ",".join(map(str, parent)).encode()
        ).hexdigest()[:16]
        if assigned["feasible"]:
            current_mean = float(current["margins"].mean())
            assigned_mean = float(assigned["margins"].mean())
            improved = assigned_mean > current_mean
            retention_values = [
                float(new / old) if old > 1e-12 else float(new >= old)
                for new, old in zip(assigned["own"], current["own"])
            ]
        else:
            assigned_mean = math.nan
            improved = False
            retention_values = []
        set_rows.append(
            {
                "set_id": set_id,
                "depth": len(parent) - 1,
                "children": len(child_tokens),
                "current_mean_margin": float(current["margins"].mean()),
                "assigned_mean_margin": assigned_mean,
                "margin_gain": assigned_mean - float(current["margins"].mean())
                if assigned["feasible"]
                else math.nan,
                "feasible": int(assigned["feasible"]),
                "improved": int(improved),
                "mean_representativeness_retention": float(
                    np.mean(retention_values)
                )
                if retention_values
                else math.nan,
            }
        )
        for row, token in enumerate(child_tokens):
            child_rows.append(
                {
                    "set_id": set_id,
                    "depth": len(parent) - 1,
                    "current_token": token,
                    "assigned_token": assigned["tokens"][row]
                    if assigned["feasible"]
                    else "",
                    "descendant_items": len(children[token]),
                    "current_own_weight": float(current["own"][row]),
                    "assigned_own_weight": float(assigned["own"][row])
                    if assigned["feasible"]
                    else math.nan,
                    "current_margin": float(current["margins"][row]),
                    "assigned_margin": float(assigned["margins"][row])
                    if assigned["feasible"]
                    else math.nan,
                }
            )
        if (set_index + 1) % 500 == 0:
            print(
                f"SCDL_N1_PROGRESS dataset={dataset} sets={set_index + 1}/{len(sibling_sets)}",
                flush=True,
            )
    output_dir = output_root / dataset
    set_path = output_dir / "sibling_sets.csv"
    child_path = output_dir / "child_assignments.csv"
    write_csv(set_path, set_rows)
    write_csv(child_path, child_rows)
    feasible_sets = [row for row in set_rows if row["feasible"]]
    feasible_children = [
        row for row in child_rows if math.isfinite(float(row["assigned_margin"]))
    ]
    depth_counts = Counter(row["depth"] for row in set_rows)
    current_nonpositive = float(
        np.mean([row["current_margin"] <= 0 for row in child_rows])
    )
    assigned_positive = float(
        np.mean([row["assigned_margin"] > 0 for row in feasible_children])
    )
    current_positive_feasible = float(
        np.mean([row["current_margin"] > 0 for row in feasible_children])
    )
    metrics = {
        "catalog_items": len(catalog),
        "eligible_sibling_sets": len(set_rows),
        "eligible_children": len(child_rows),
        "sets_by_depth": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "current_nonpositive_margin_child_rate": current_nonpositive,
        "assignment_feasible_set_rate": len(feasible_sets) / len(set_rows),
        "improved_set_rate": float(
            np.mean([row["improved"] for row in feasible_sets])
        ),
        "mean_child_margin_gain": float(
            np.mean(
                [
                    row["assigned_margin"] - row["current_margin"]
                    for row in feasible_children
                ]
            )
        ),
        "mean_representativeness_retention": float(
            np.mean(
                [
                    row["assigned_own_weight"] / row["current_own_weight"]
                    if row["current_own_weight"] > 1e-12
                    else float(row["assigned_own_weight"] >= row["current_own_weight"])
                    for row in feasible_children
                ]
            )
        ),
        "current_positive_margin_child_rate_feasible": current_positive_feasible,
        "assigned_positive_margin_child_rate": assigned_positive,
        "positive_margin_child_rate_gain": assigned_positive
        - current_positive_feasible,
    }
    gates = config["scientific_gates"]
    checks = {
        "eligible_sibling_sets": len(set_rows)
        >= int(gates["eligible_sibling_sets_min"]),
        "supported_depths": sum(
            count >= int(gates["eligible_sets_per_supported_depth_min"])
            for count in depth_counts.values()
        )
        >= int(gates["supported_depths_min"]),
        "current_nonpositive_margin_child_rate": current_nonpositive
        >= float(gates["current_nonpositive_margin_child_rate_min"]),
        "assignment_feasible_set_rate": metrics["assignment_feasible_set_rate"]
        >= float(gates["assignment_feasible_set_rate_min"]),
        "improved_set_rate": metrics["improved_set_rate"]
        >= float(gates["improved_set_rate_min"]),
        "mean_child_margin_gain": metrics["mean_child_margin_gain"]
        >= float(gates["mean_child_margin_gain_min"]),
        "mean_representativeness_retention": metrics[
            "mean_representativeness_retention"
        ]
        >= float(gates["mean_representativeness_retention_min"]),
        "positive_margin_child_rate_gain": metrics[
            "positive_margin_child_rate_gain"
        ]
        >= float(gates["positive_margin_child_rate_gain_min"]),
    }
    finite = all(
        math.isfinite(float(row["current_mean_margin"]))
        and (
            not row["feasible"]
            or math.isfinite(float(row["assigned_mean_margin"]))
        )
        for row in set_rows
    )
    integrity = {
        "catalog_mapping_rate": 1.0,
        "unique_identifier_rate": unique_identifier_rate,
        "finite_rate": float(finite),
        "optimizer_steps": 0,
        "checkpoint_read": False,
        "interaction_targets_read": False,
        "validation_test_predictions_read": False,
        "sports_read": False,
        "lexical_file_sha256": sha256(lexical_path),
        "text_file_sha256": sha256(dataset_dir / "item_plain_text.txt"),
        "set_file_sha256": sha256(set_path),
        "child_file_sha256": sha256(child_path),
    }
    integrity_valid = (
        integrity["catalog_mapping_rate"] == 1.0
        and integrity["unique_identifier_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["optimizer_steps"] == 0
        and not integrity["checkpoint_read"]
        and not integrity["interaction_targets_read"]
        and not integrity["validation_test_predictions_read"]
        and not integrity["sports_read"]
    )
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }


def decision_markdown(summary: dict) -> str:
    lines = [
        "# SCDL-N1 Decision",
        "",
        f"- Fixed decision: **`{summary['decision']}`**",
        f"- Integrity valid: `{str(summary['integrity_valid']).lower()}`",
        "- Data scope: catalog metadata and frozen lexical IDs only",
        "- Checkpoint/interaction targets/validation/test/Sports read: `false`",
        "",
        "## Frozen gate results",
        "",
    ]
    for dataset, result in summary["results"].items():
        lines.extend([f"### {dataset}", ""])
        for name, passed in result["checks"].items():
            lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    actual_sha = sha256(Path(__file__))
    expected_sha = config["integrity"]["code_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(f"code SHA mismatch expected={expected_sha} actual={actual_sha}")
    results = {
        dataset: audit_dataset(dataset, spec, config, args.output_root)
        for dataset, spec in config["datasets"].items()
    }
    integrity_valid = all(row["integrity_valid"] for row in results.values())
    scientific_pass = all(row["scientific_pass"] for row in results.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "SCDL_S0_DESIGN_ALLOWED"
        if scientific_pass
        else "STOP_SCDL_NO_SIBLING_LEXICALIZATION_DEFICIT"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "integrity_valid": integrity_valid,
        "checkpoint_read": False,
        "interaction_targets_read": False,
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
