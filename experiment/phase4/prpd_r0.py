#!/usr/bin/env python3
"""PRPD R0: popularity-residual, frozen-output effect gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from experiment.phase4.rpcd_t0 import (
    ROOT,
    deduplicate,
    fuse,
    load_dataset,
    metric,
    popularity_tail,
    resolve_inputs,
    sha256,
    stable_fraction,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_teacher(path: Path) -> Dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            user = row.pop("user")
            if user in rows:
                raise ValueError(f"duplicate teacher user: {user}")
            rows[user] = row
    return rows


def popularity_midrank_percentile(
    sequences: Mapping[str, Sequence[str]], catalog: Sequence[str]
) -> Dict[str, float]:
    counts = Counter(item for sequence in sequences.values() for item in sequence[:-2])
    grouped = Counter(counts.get(item, 0) for item in catalog)
    below = 0
    percentile_by_count = {}
    n = len(catalog)
    for count in sorted(grouped):
        tied = grouped[count]
        percentile_by_count[count] = (below + 0.5 * tied) / n
        below += tied
    return {item: percentile_by_count[counts.get(item, 0)] for item in catalog}


def reciprocal_rank_scores(items: Sequence[str]) -> Dict[str, float]:
    return {
        item: 1.0 / math.log2(rank + 1.0)
        for rank, item in enumerate(deduplicate(items), 1)
    }


def residual_fuse(
    gram: Sequence[str],
    sasrec: Sequence[str],
    popularity: Mapping[str, float],
    gamma: float,
    weight: float,
) -> list:
    gram = deduplicate(gram)
    sasrec = deduplicate(sasrec)
    gram_scores = reciprocal_rank_scores(gram)
    sas_scores = reciprocal_rank_scores(sasrec)
    union = deduplicate(list(gram) + list(sasrec))
    stable_order = {item: index for index, item in enumerate(union)}
    return sorted(
        union,
        key=lambda item: (
            -(
                (1.0 - weight) * gram_scores.get(item, 0.0)
                + weight
                * (
                    sas_scores.get(item, 0.0)
                    - gamma * popularity.get(item, 0.0)
                    if item in sas_scores
                    else 0.0
                )
            ),
            stable_order[item],
        ),
    )


def safe_relative(new: float, old: float) -> float:
    if old <= 0:
        raise ValueError("relative metric baseline must be positive")
    return new / old - 1.0


def evaluate(
    rows: Sequence[dict],
    teachers: Mapping[str, dict],
    users: set,
    popularity: Mapping[str, float],
    tail_items: set,
    gamma: float,
    weight: float,
    keep_arrays: bool = False,
) -> dict:
    gram_recall = hybrid_recall = gram_ndcg = hybrid_ndcg = 0.0
    tail_gram_ndcg = tail_hybrid_ndcg = 0.0
    n = tail_n = 0
    ndcg_pairs = []
    recall_pairs = []
    tail_pairs = []
    head_pairs = []
    for row in rows:
        if row["user"] not in users:
            continue
        gram = deduplicate(row["pred_items"])
        hybrid = residual_fuse(
            gram, teachers[row["user"]]["items"], popularity, gamma, weight
        )
        gold = row["gold"]
        gr, gn = metric(gram, gold, 10)
        hr, hn = metric(hybrid, gold, 10)
        gram_recall += gr
        hybrid_recall += hr
        gram_ndcg += gn
        hybrid_ndcg += hn
        n += 1
        if keep_arrays:
            ndcg_pairs.append((gn, hn))
            recall_pairs.append((gr, hr))
        if gold in tail_items:
            tail_gram_ndcg += gn
            tail_hybrid_ndcg += hn
            tail_n += 1
            if keep_arrays:
                tail_pairs.append((gn, hn))
        elif keep_arrays:
            head_pairs.append((gn, hn))
    result = {
        "n": n,
        "gram_recall@10": gram_recall / n,
        "hybrid_recall@10": hybrid_recall / n,
        "recall10_absolute_gain": (hybrid_recall - gram_recall) / n,
        "gram_ndcg@10": gram_ndcg / n,
        "hybrid_ndcg@10": hybrid_ndcg / n,
        "ndcg10_absolute_gain": (hybrid_ndcg - gram_ndcg) / n,
        "ndcg10_relative_gain": safe_relative(hybrid_ndcg, gram_ndcg),
        "tail_n": tail_n,
        "tail_gram_ndcg@10": tail_gram_ndcg / tail_n,
        "tail_hybrid_ndcg@10": tail_hybrid_ndcg / tail_n,
        "tail_ndcg10_relative_gain": safe_relative(
            tail_hybrid_ndcg, tail_gram_ndcg
        ),
    }
    if keep_arrays:
        result["_arrays"] = {
            "ndcg": np.asarray(ndcg_pairs, dtype=np.float64),
            "recall": np.asarray(recall_pairs, dtype=np.float64),
            "tail_ndcg": np.asarray(tail_pairs, dtype=np.float64),
            "head_ndcg": np.asarray(head_pairs, dtype=np.float64),
        }
        head = result["_arrays"]["head_ndcg"]
        result["head_n"] = len(head)
        result["head_gram_ndcg@10"] = float(head[:, 0].mean())
        result["head_hybrid_ndcg@10"] = float(head[:, 1].mean())
        result["head_ndcg10_relative_gain"] = safe_relative(
            float(head[:, 1].mean()), float(head[:, 0].mean())
        )
    return result


def paired_bootstrap(
    pairs: np.ndarray,
    iterations: int,
    seed: int,
    relative: bool,
) -> list:
    rng = np.random.default_rng(seed)
    n = len(pairs)
    values = np.empty(iterations, dtype=np.float64)
    batch = 100
    for start in range(0, iterations, batch):
        size = min(batch, iterations - start)
        indices = rng.integers(0, n, size=(size, n))
        sampled = pairs[indices]
        old = sampled[:, :, 0].mean(axis=1)
        new = sampled[:, :, 1].mean(axis=1)
        values[start : start + size] = new / old - 1.0 if relative else new - old
    return [float(value) for value in np.percentile(values, (2.5, 97.5))]


def strip_arrays(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_arrays"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config = json.loads(args.config.read_text())
    source_config_path = ROOT / config["source_config"]
    source_config = json.loads(source_config_path.read_text())
    source_summary_path = ROOT / config["source_summary"]
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary["decision"] != "STOP_RPCD_NO_TEACHER_COMPLEMENTARITY":
        raise ValueError("unexpected source RPCD decision")
    resolved = resolve_inputs(source_config)
    data = {}
    preflight = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(args.config),
        "source_config_sha256": sha256(source_config_path),
        "source_summary_sha256": sha256(source_summary_path),
        "test_predictions_read": False,
        "sequence_test_target_indexed": False,
        "datasets": {},
    }
    for dataset, paths in resolved.items():
        loaded = load_dataset(paths)
        teacher_path = ROOT / config["teacher_outputs"][dataset]
        teachers = read_teacher(teacher_path)
        if set(teachers) != set(loaded["sequences"]):
            raise ValueError(f"{dataset}: teacher user set mismatch")
        bad_targets = [
            user
            for user, row in teachers.items()
            if row["target"] != loaded["sequences"][user][-2]
        ]
        if bad_targets:
            raise ValueError(f"{dataset}: teacher target mismatch {bad_targets[:3]}")
        pop = popularity_midrank_percentile(loaded["sequences"], loaded["catalog"])
        tail = popularity_tail(loaded["sequences"])
        calibration = {
            user
            for user in loaded["sequences"]
            if stable_fraction(user, config["selection"]["calibration_salt"])
            < config["selection"]["calibration_fraction"]
        }
        audit = set(loaded["sequences"]) - calibration
        data[dataset] = {
            **loaded,
            "teachers": teachers,
            "popularity": pop,
            "tail": tail,
            "calibration": calibration,
            "audit": audit,
        }
        preflight["datasets"][dataset] = {
            "users": len(loaded["sequences"]),
            "teacher_users": len(teachers),
            "target_match_rate": 1.0,
            "calibration_users": len(calibration),
            "audit_users": len(audit),
            "teacher_sha256": sha256(teacher_path),
        }
    # Mechanical identity: gamma=0 must match the exact RPCD T0 fusion.
    checked = 0
    for dataset, loaded in data.items():
        for row in loaded["rows"][:100]:
            sasrec = loaded["teachers"][row["user"]]["items"]
            for weight in config["residual"]["weights"]:
                expected = fuse(row["pred_items"], sasrec, float(weight))
                observed = residual_fuse(
                    row["pred_items"],
                    sasrec,
                    loaded["popularity"],
                    0.0,
                    float(weight),
                )
                if observed != expected:
                    raise AssertionError("gamma=0 RPCD identity failure")
                checked += 1
    preflight["gamma0_identity_checks"] = checked
    preflight["gamma0_identity_rate"] = 1.0
    write_json(args.output_dir / "preflight.json", preflight)
    grid = []
    for gamma in config["residual"]["gammas"]:
        for weight in config["residual"]["weights"]:
            per_dataset = {}
            eligible = True
            macro = []
            for dataset, loaded in data.items():
                result = evaluate(
                    loaded["rows"],
                    loaded["teachers"],
                    loaded["calibration"],
                    loaded["popularity"],
                    loaded["tail"],
                    float(gamma),
                    float(weight),
                )
                per_dataset[dataset] = result
                macro.append(result["ndcg10_relative_gain"])
                eligible &= (
                    result["recall10_absolute_gain"] >= 0.0
                    and result["tail_ndcg10_relative_gain"] >= 0.0
                )
            grid.append(
                {
                    "gamma": gamma,
                    "weight": weight,
                    "eligible": eligible,
                    "macro_ndcg10_relative_gain": sum(macro) / len(macro),
                    "datasets": per_dataset,
                }
            )
    eligible = [row for row in grid if row["eligible"]]
    if not eligible:
        raise AssertionError("identity configuration should always be eligible")
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_ndcg10_relative_gain"],
            -float(row["gamma"]),
            -float(row["weight"]),
        ),
    )
    audit = {}
    gate_rows = []
    for dataset, loaded in data.items():
        result = evaluate(
            loaded["rows"],
            loaded["teachers"],
            loaded["audit"],
            loaded["popularity"],
            loaded["tail"],
            float(selected["gamma"]),
            float(selected["weight"]),
            keep_arrays=True,
        )
        arrays = result["_arrays"]
        bootstrap = {
            "ndcg10_absolute_gain_ci95": paired_bootstrap(
                arrays["ndcg"],
                int(config["bootstrap"]["iterations"]),
                int(config["seed"]) + 11,
                False,
            ),
            "ndcg10_relative_gain_ci95": paired_bootstrap(
                arrays["ndcg"],
                int(config["bootstrap"]["iterations"]),
                int(config["seed"]) + 12,
                True,
            ),
            "recall10_absolute_gain_ci95": paired_bootstrap(
                arrays["recall"],
                int(config["bootstrap"]["iterations"]),
                int(config["seed"]) + 13,
                False,
            ),
            "tail_ndcg10_relative_gain_ci95": paired_bootstrap(
                arrays["tail_ndcg"],
                int(config["bootstrap"]["iterations"]),
                int(config["seed"]) + 14,
                True,
            ),
            "head_ndcg10_relative_gain_ci95": paired_bootstrap(
                arrays["head_ndcg"],
                int(config["bootstrap"]["iterations"]),
                int(config["seed"]) + 15,
                True,
            ),
        }
        audit[dataset] = {**strip_arrays(result), "bootstrap": bootstrap}
        checks = {
            "ndcg10_relative_gain": result["ndcg10_relative_gain"]
            >= config["gates"]["hybrid_ndcg10_relative_gain_min"],
            "recall10_absolute_gain": result["recall10_absolute_gain"]
            >= config["gates"]["hybrid_recall10_absolute_gain_min"],
            "tail_ndcg10_relative_gain": result["tail_ndcg10_relative_gain"]
            >= config["gates"]["tail_ndcg10_relative_gain_min"],
        }
        gate_rows.append(
            {"dataset": dataset, "checks": checks, "pass": all(checks.values())}
        )
    passed = all(row["pass"] for row in gate_rows)
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": (
            "PRPD_R1_DESIGN_ALLOWED" if passed else "STOP_PRPD_NO_DEBIASED_EFFECT"
        ),
        "selected_shared_config": {
            "gamma": selected["gamma"],
            "weight": selected["weight"],
            "calibration_macro_ndcg10_relative_gain": selected[
                "macro_ndcg10_relative_gain"
            ],
        },
        "grid": grid,
        "audit": audit,
        "gate_rows": gate_rows,
        "integrity": {
            "preflight_passed": True,
            "gamma0_identity_rate": 1.0,
            "shared_config": True,
            "target_match_rate": 1.0,
            "test_predictions_read": False,
            "sequence_test_target_indexed": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "selected_shared_config": summary["selected_shared_config"],
                "audit": summary["audit"],
                "gate_rows": summary["gate_rows"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
