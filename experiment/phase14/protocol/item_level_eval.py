"""Collision-aware item-level evaluator for Phase-14 Stage 14-0A.

The original GRAM evaluator compares decoded lexical strings.  That is valid only
when every decoded string identifies exactly one catalog item.  This module builds
``decoded_path -> [item_ids]`` from the authoritative item-to-path file and refuses
ambiguous, unknown, or duplicate top-K outputs in formal mode.

``record`` mode exists only to audit frozen legacy raw-v1 predictions: invalid
outputs are recorded and never credited as item hits.  New/formal runs must use the
default ``hard_fail`` policy.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable


KS = (1, 3, 5, 10, 20, 50)
SAVED_METRICS = tuple(
    [f"hit@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
)


class EvaluationIntegrityError(RuntimeError):
    """Raised when a formal evaluation contains an item-identity violation."""


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_tokens(lexical_id: str) -> tuple[str, ...]:
    tokens = tuple(token for token in lexical_id.split("|") if token)
    if not tokens:
        raise ValueError(f"Semantic ID has no tokens: {lexical_id!r}")
    return tokens


def decode_lexical_id(lexical_id: str) -> str:
    """Reproduce GRAM/T5 decoding for delimiter-separated lexical IDs."""
    return "".join(token.replace("▁", " ") for token in semantic_tokens(lexical_id)).strip()


def load_item_paths(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    item_to_lexical: dict[str, str] = {}
    decoded_to_items: dict[str, list[str]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            item, sep, lexical = line.partition(" ")
            if not sep or not lexical:
                raise ValueError(f"{path}:{line_no}: malformed item/path row")
            if item in item_to_lexical:
                raise ValueError(f"{path}:{line_no}: duplicate item {item}")
            item_to_lexical[item] = lexical
            decoded_to_items[decode_lexical_id(lexical)].append(item)
    if not item_to_lexical:
        raise ValueError(f"No item paths loaded from {path}")
    return item_to_lexical, dict(decoded_to_items)


def load_sequences(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no}: sequence too short for validation target")
            if parts[0] in rows:
                raise ValueError(f"{path}:{line_no}: duplicate user {parts[0]}")
            rows[parts[0]] = parts[1:]
    return rows


def load_set(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def parse_prediction_rows(path: Path) -> list[dict]:
    if "test" in path.name.lower():
        raise ValueError(f"Stage 14-0A refuses test predictions: {path}")
    rows: list[dict] = []
    seen_users: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith(("idx\t", "hit@", "ndcg@")):
                continue
            fields = line.split("\t")
            if len(fields) < 16:
                continue
            try:
                saved = [float(value) for value in fields[1:13]]
                scores = [float(value) for value in fields[15].split("||")]
            except ValueError:
                continue
            predictions = fields[14].split("||") if fields[14] else []
            if len(predictions) != len(scores):
                raise ValueError(
                    f"{path}:{line_no}: prediction/score mismatch "
                    f"{len(predictions)} != {len(scores)}"
                )
            user = fields[0]
            if user in seen_users:
                raise ValueError(f"{path}:{line_no}: duplicate prediction user {user}")
            seen_users.add(user)
            rows.append(
                {
                    "line_no": line_no,
                    "user_id": user,
                    "saved_metrics": dict(zip(SAVED_METRICS, saved)),
                    "gold_decoded": fields[13],
                    "predictions": predictions,
                    "scores": scores,
                }
            )
    if not rows:
        raise ValueError(f"No prediction rows parsed from {path}")
    return rows


def metrics_for_rank(rank: int | None) -> dict[str, float]:
    return {
        **{f"hit@{k}": float(rank is not None and rank <= k) for k in KS},
        **{
            f"ndcg@{k}": (1.0 / math.log2(rank + 1)) if rank is not None and rank <= k else 0.0
            for k in KS
        },
    }


def average_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        return {metric: None for metric in SAVED_METRICS}
    return {
        metric: sum(row[metric] for row in values) / len(values)
        for metric in SAVED_METRICS
    }


def evaluate(
    *,
    dataset_dir: Path,
    item_path_file: Path,
    predictions_tsv: Path,
    output_dir: Path,
    invalid_policy: str = "hard_fail",
    split: str = "validation",
) -> dict:
    if split != "validation":
        raise ValueError("Stage 14-0A currently permits validation only")
    if invalid_policy not in {"hard_fail", "record"}:
        raise ValueError(f"Unsupported invalid policy: {invalid_policy}")

    user_sequence = dataset_dir / "user_sequence.txt"
    cold_file = dataset_dir / "cold_split_meta" / "cold_items.txt"
    warm_file = dataset_dir / "cold_split_meta" / "warm_items.txt"
    inputs = [item_path_file, predictions_tsv, user_sequence, cold_file, warm_file]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    item_to_lexical, decoded_to_items = load_item_paths(item_path_file)
    sequences = load_sequences(user_sequence)
    cold_items = load_set(cold_file)
    warm_items = load_set(warm_file)
    if cold_items & warm_items:
        raise EvaluationIntegrityError("cold/warm item sets overlap")
    if cold_items | warm_items != set(item_to_lexical):
        raise EvaluationIntegrityError("cold/warm sets do not exactly partition identifier catalog")

    prediction_rows = parse_prediction_rows(predictions_tsv)
    issue_counts: collections.Counter[str] = collections.Counter()
    issue_examples: dict[str, list[dict]] = collections.defaultdict(list)
    strict_rows: list[dict[str, float]] = []
    saved_rows: list[dict[str, float]] = []
    slices: dict[str, list[dict[str, float]]] = {"all": [], "warm": [], "cold": []}
    prediction_records: list[dict] = []

    def issue(kind: str, detail: dict) -> None:
        issue_counts[kind] += 1
        if len(issue_examples[kind]) < 10:
            issue_examples[kind].append(detail)

    for row in prediction_rows:
        user = row["user_id"]
        if user not in sequences:
            issue("unknown_user", {"user_id": user, "line_no": row["line_no"]})
            continue
        target = sequences[user][-2]
        target_candidates = decoded_to_items.get(row["gold_decoded"], [])
        if target not in target_candidates:
            issue(
                "gold_target_mismatch",
                {"user_id": user, "target": target, "gold_decoded": row["gold_decoded"]},
            )
        if len(target_candidates) > 1:
            issue(
                "ambiguous_gold",
                {"user_id": user, "target": target, "items": target_candidates},
            )

        seen_decoded: set[str] = set()
        seen_items: set[str] = set()
        row_invalid = False
        strict_rank: int | None = None
        for rank, decoded in enumerate(row["predictions"][: max(KS)], 1):
            if decoded in seen_decoded:
                issue("duplicate_decoded_topk", {"user_id": user, "rank": rank, "decoded": decoded})
                row_invalid = True
                continue
            seen_decoded.add(decoded)
            candidates = decoded_to_items.get(decoded, [])
            if not candidates:
                issue("unknown_prediction", {"user_id": user, "rank": rank, "decoded": decoded})
                row_invalid = True
                continue
            if len(candidates) != 1:
                issue(
                    "ambiguous_prediction",
                    {"user_id": user, "rank": rank, "decoded": decoded, "items": candidates},
                )
                row_invalid = True
                continue
            item = candidates[0]
            if item in seen_items:
                issue("duplicate_item_topk", {"user_id": user, "rank": rank, "item": item})
                row_invalid = True
                continue
            seen_items.add(item)
            if item == target and strict_rank is None:
                # Preserve the original beam rank.  Invalid/ambiguous entries ahead
                # of a valid target remain occupied positions and are not collapsed.
                strict_rank = rank

        strict_metrics = metrics_for_rank(strict_rank)
        strict_rows.append(strict_metrics)
        saved_rows.append(row["saved_metrics"])
        slice_name = "cold" if target in cold_items else "warm"
        slices["all"].append(strict_metrics)
        slices[slice_name].append(strict_metrics)
        prediction_records.append(
            {
                "user_id": user,
                "target_item": target,
                "is_cold": target in cold_items,
                "strict_rank": strict_rank,
                "legacy_string_hit50": bool(row["saved_metrics"]["hit@50"]),
                "row_has_invalid_output": row_invalid,
            }
        )

    fatal_issue_count = sum(issue_counts.values())
    if invalid_policy == "hard_fail" and fatal_issue_count:
        raise EvaluationIntegrityError(
            f"Formal item evaluation failed with {fatal_issue_count} integrity issues: "
            f"{dict(issue_counts)}"
        )

    strict = average_metrics(strict_rows)
    saved = average_metrics(saved_rows)
    metric_abs_diff = {metric: abs(strict[metric] - saved[metric]) for metric in SAVED_METRICS}
    alias_string_hits_removed = sum(
        record["legacy_string_hit50"] and record["strict_rank"] is None
        for record in prediction_records
    )
    duplicate_path_groups = sum(len(items) > 1 for items in decoded_to_items.values())
    items_in_duplicate_groups = sum(
        len(items) for items in decoded_to_items.values() if len(items) > 1
    )

    summary = {
        "experiment_id": "GRAM_PHASE14_STAGE14_0A_ITEM_LEVEL_EVAL",
        "status": "completed",
        "split": split,
        "test_predictions_opened": False,
        "invalid_policy": invalid_policy,
        "formal_item_level_valid": fatal_issue_count == 0,
        "n_catalog_items": len(item_to_lexical),
        "n_decoded_paths": len(decoded_to_items),
        "duplicate_path_groups": duplicate_path_groups,
        "items_in_duplicate_path_groups": items_in_duplicate_groups,
        "n_prediction_rows": len(prediction_records),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": dict(issue_examples),
        "alias_string_hits_removed_at_50": alias_string_hits_removed,
        "legacy_saved_metrics": saved,
        "strict_item_metrics": strict,
        "metric_abs_diff": metric_abs_diff,
        "max_metric_abs_diff": max(metric_abs_diff.values()),
        "slice_metrics": {
            name: {"n": len(rows), **average_metrics(rows)}
            for name, rows in slices.items()
        },
        "inputs": {
            "dataset_dir": str(dataset_dir.resolve()),
            "item_path_file": str(item_path_file.resolve()),
            "predictions_tsv": str(predictions_tsv.resolve()),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "scope": "application-level declared opens",
            "test_files_opened": [],
            "files": [
                {"path": str(path.resolve()), "mode": "read", "sha256": hashes[str(path.resolve())]}
                for path in inputs
            ],
        },
    )
    atomic_json(
        output_dir / "item_path_audit.json",
        {
            "duplicate_path_groups": duplicate_path_groups,
            "items_in_duplicate_path_groups": items_in_duplicate_groups,
            "issue_counts": dict(sorted(issue_counts.items())),
            "issue_examples": dict(issue_examples),
        },
    )
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "split": split,
            "test_predictions_opened": False,
            "target_rule": "user_sequence[-2]",
            "reverse_map_source": "item2lexid multimap; lexid2cfid not used",
            "invalid_policy": invalid_policy,
        },
    )
    atomic_json(
        output_dir / "config.json",
        {
            "dataset_dir": str(dataset_dir.resolve()),
            "item_path_file": str(item_path_file.resolve()),
            "predictions_tsv": str(predictions_tsv.resolve()),
            "output_dir": str(output_dir.resolve()),
            "invalid_policy": invalid_policy,
            "split": split,
        },
    )
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_strict_validation.jsonl").open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--item-path-file", required=True, type=Path)
    parser.add_argument("--predictions-tsv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--invalid-policy", choices=("hard_fail", "record"), default="hard_fail")
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args()
    summary = evaluate(**vars(args))
    print(json.dumps({
        "status": summary["status"],
        "formal_item_level_valid": summary["formal_item_level_valid"],
        "duplicate_path_groups": summary["duplicate_path_groups"],
        "alias_string_hits_removed_at_50": summary["alias_string_hits_removed_at_50"],
        "max_metric_abs_diff": summary["max_metric_abs_diff"],
        "summary": str((args.output_dir / "summary.json").resolve()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
