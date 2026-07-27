#!/usr/bin/env python3
"""CCRR R0: candidate-conditional frozen-output reranking effect gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiment.phase4.prpd_r0 import paired_bootstrap, read_teacher
from experiment.phase4.rpcd_t0 import (
    ROOT,
    deduplicate,
    fuse,
    load_dataset,
    metric,
    resolve_inputs,
    sha256,
    stable_fraction,
)


FEATURE_SCHEMA = [
    "gram_present",
    "sasrec_present",
    "gram_reciprocal_rank",
    "sasrec_reciprocal_rank",
    "sasrec_logit_z",
    "normalized_rank_difference",
    "source_agreement",
    "training_popularity_percentile",
    "head_indicator",
    "log_history_length",
    "seen_in_history",
    "sasrec_rr_x_popularity",
    "gram_rr_x_popularity",
    "sasrec_rr_x_history_length",
    "gram_rr_x_history_length",
    "agreement_x_sasrec_logit_z",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def user_set_sha256(users: Sequence[str]) -> str:
    payload = "\n".join(sorted(users)).encode()
    return hashlib.sha256(payload).hexdigest()


def popularity_features(
    sequences: Mapping[str, Sequence[str]], catalog: Sequence[str]
) -> tuple[Dict[str, float], set]:
    counts = Counter(item for sequence in sequences.values() for item in sequence[:-2])
    order = sorted(catalog, key=lambda item: (counts.get(item, 0), item))
    percentile = {
        item: (rank + 0.5) / len(order) for rank, item in enumerate(order)
    }
    head_count = max(1, math.ceil(len(catalog) * 0.2))
    head = set(
        sorted(catalog, key=lambda item: (-counts.get(item, 0), item))[:head_count]
    )
    return percentile, head


def candidate_union(gram: Sequence[str], sasrec: Sequence[str]) -> list[str]:
    return deduplicate(list(gram[:50]) + list(sasrec[:50]))


def source_maps(items: Sequence[str]) -> Dict[str, int]:
    return {item: rank for rank, item in enumerate(deduplicate(items[:50]), 1)}


def standardized_scores(items: Sequence[str], scores: Sequence[float]) -> Dict[str, float]:
    unique_items = []
    unique_scores = []
    seen = set()
    for item, score in zip(items[:50], scores[:50]):
        if item not in seen:
            unique_items.append(item)
            unique_scores.append(float(score))
            seen.add(item)
    values = np.asarray(unique_scores, dtype=np.float64)
    std = float(values.std())
    normalized = np.zeros_like(values) if std == 0.0 else (values - values.mean()) / std
    return {item: float(value) for item, value in zip(unique_items, normalized)}


def feature_matrix(
    gram: Sequence[str],
    sasrec_items: Sequence[str],
    sasrec_scores: Sequence[float],
    history: Sequence[str],
    candidates: Sequence[str],
    popularity: Mapping[str, float],
    head: set,
) -> np.ndarray:
    gram_rank = source_maps(gram)
    sas_rank = source_maps(sasrec_items)
    sas_z = standardized_scores(sasrec_items, sasrec_scores)
    history_set = set(history)
    log_history = math.log1p(len(history))
    rows = []
    for item in candidates:
        gr = gram_rank.get(item, 51)
        sr = sas_rank.get(item, 51)
        gp = float(item in gram_rank)
        sp = float(item in sas_rank)
        grr = 0.0 if gr == 51 else 1.0 / math.log2(gr + 1.0)
        srr = 0.0 if sr == 51 else 1.0 / math.log2(sr + 1.0)
        agreement = gp * sp
        pop = float(popularity[item])
        score_z = float(sas_z.get(item, 0.0))
        rows.append(
            [
                gp,
                sp,
                grr,
                srr,
                score_z,
                (sr - gr) / 50.0,
                agreement,
                pop,
                float(item in head),
                log_history,
                float(item in history_set),
                srr * pop,
                grr * pop,
                srr * log_history,
                grr * log_history,
                agreement * score_z,
            ]
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(candidates), len(FEATURE_SCHEMA)):
        raise AssertionError("feature shape mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite feature")
    return matrix


def prepare_rows(loaded: dict, teachers: Mapping[str, dict]) -> list[dict]:
    rows = []
    for row in loaded["rows"]:
        user = row["user"]
        teacher = teachers[user]
        candidates = candidate_union(row["pred_items"], teacher["items"])
        if len(candidates) != len(set(candidates)):
            raise AssertionError("duplicate candidate")
        rows.append(
            {
                "user": user,
                "gold": row["gold"],
                "gram": deduplicate(row["pred_items"][:50]),
                "sasrec": deduplicate(teacher["items"][:50]),
                "sasrec_scores": teacher["scores"][:50],
                "candidates": candidates,
                "history": loaded["sequences"][user][:-2],
            }
        )
    return rows


def fit_ranker(
    rows: Sequence[dict],
    fit_users: set,
    popularity: Mapping[str, float],
    head: set,
    model_config: Mapping[str, object],
) -> tuple[StandardScaler, LogisticRegression, dict]:
    features = []
    labels = []
    fit_user_order = []
    positive_users = 0
    for row in rows:
        if row["user"] not in fit_users:
            continue
        matrix = feature_matrix(
            row["gram"],
            row["sasrec"],
            row["sasrec_scores"],
            row["history"],
            row["candidates"],
            popularity,
            head,
        )
        target = np.asarray(
            [int(item == row["gold"]) for item in row["candidates"]], dtype=np.int8
        )
        positive_users += int(target.sum() == 1)
        features.append(matrix)
        labels.append(target)
        fit_user_order.append(row["user"])
    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("fit labels must contain both classes")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        solver=str(model_config["solver"]),
        C=float(model_config["C"]),
        class_weight=str(model_config["class_weight"]),
        max_iter=int(model_config["max_iter"]),
        tol=float(model_config["tol"]),
        random_state=int(model_config["random_state"]),
    )
    model.fit(x_scaled, y)
    details = {
        "fit_users": len(fit_users),
        "fit_user_sha256": user_set_sha256(fit_user_order),
        "fit_candidate_rows": int(len(y)),
        "positive_candidate_rows": int(y.sum()),
        "positive_users": positive_users,
        "converged": int(model.n_iter_[0]) < int(model_config["max_iter"]),
        "iterations": int(model.n_iter_[0]),
        "feature_schema": FEATURE_SCHEMA,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    return scaler, model, details


def model_ranking(
    row: Mapping[str, object],
    popularity: Mapping[str, float],
    head: set,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> list[str]:
    candidates = row["candidates"]
    matrix = feature_matrix(
        row["gram"],
        row["sasrec"],
        row["sasrec_scores"],
        row["history"],
        candidates,
        popularity,
        head,
    )
    scores = model.decision_function(scaler.transform(matrix))
    return [
        candidates[index]
        for index in sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    ]


def evaluate(
    rows: Sequence[dict],
    users: set,
    popularity: Mapping[str, float],
    head: set,
    tail: set,
    scaler: StandardScaler,
    model: LogisticRegression,
    keep_arrays: bool = False,
) -> dict:
    sums = Counter()
    pairs = {"ndcg": [], "recall": [], "tail_ndcg": [], "b1_ndcg": []}
    candidate_identity = 0
    n = tail_n = 0
    for row in rows:
        if row["user"] not in users:
            continue
        b0 = row["gram"]
        b1 = fuse(row["gram"], row["sasrec"], 0.2)
        r1 = model_ranking(row, popularity, head, scaler, model)
        candidate_identity += int(set(r1) == set(row["candidates"]))
        gold = row["gold"]
        b0r, b0n = metric(b0, gold, 10)
        b1r, b1n = metric(b1, gold, 10)
        r1r, r1n = metric(r1, gold, 10)
        sums.update(
            b0_recall=b0r,
            b0_ndcg=b0n,
            b1_recall=b1r,
            b1_ndcg=b1n,
            r1_recall=r1r,
            r1_ndcg=r1n,
        )
        pairs["ndcg"].append((b0n, r1n))
        pairs["recall"].append((b0r, r1r))
        pairs["b1_ndcg"].append((b1n, r1n))
        n += 1
        if gold in tail:
            sums.update(tail_b0_ndcg=b0n, tail_r1_ndcg=r1n)
            pairs["tail_ndcg"].append((b0n, r1n))
            tail_n += 1
    result = {
        "n": n,
        "tail_n": tail_n,
        "B0_recall@10": sums["b0_recall"] / n,
        "B0_ndcg@10": sums["b0_ndcg"] / n,
        "B1_recall@10": sums["b1_recall"] / n,
        "B1_ndcg@10": sums["b1_ndcg"] / n,
        "R1_recall@10": sums["r1_recall"] / n,
        "R1_ndcg@10": sums["r1_ndcg"] / n,
        "R1_vs_B0_recall10_absolute_gain": (
            sums["r1_recall"] - sums["b0_recall"]
        )
        / n,
        "R1_vs_B0_ndcg10_relative_gain": sums["r1_ndcg"] / sums["b0_ndcg"] - 1.0,
        "R1_vs_B1_ndcg10_relative_gain": sums["r1_ndcg"] / sums["b1_ndcg"] - 1.0,
        "tail_B0_ndcg@10": sums["tail_b0_ndcg"] / tail_n,
        "tail_R1_ndcg@10": sums["tail_r1_ndcg"] / tail_n,
        "tail_R1_vs_B0_ndcg10_relative_gain": (
            sums["tail_r1_ndcg"] / sums["tail_b0_ndcg"] - 1.0
        ),
        "candidate_set_identity_rate": candidate_identity / n,
    }
    if keep_arrays:
        result["_arrays"] = {
            key: np.asarray(value, dtype=np.float64) for key, value in pairs.items()
        }
    return result


def gate(result: Mapping[str, float], config: Mapping[str, object]) -> dict:
    checks = {
        "ndcg10_relative_gain": result["R1_vs_B0_ndcg10_relative_gain"]
        >= float(config["ndcg10_relative_gain_min"]),
        "recall10_absolute_gain": result["R1_vs_B0_recall10_absolute_gain"]
        >= float(config["recall10_absolute_gain_min"]),
        "tail_ndcg10_relative_gain": result[
            "tail_R1_vs_B0_ndcg10_relative_gain"
        ]
        >= float(config["tail_ndcg10_relative_gain_min"]),
    }
    if config["must_exceed_B1_ndcg"]:
        checks["exceeds_B1_ndcg"] = result["R1_vs_B1_ndcg10_relative_gain"] > 0.0
    return {"checks": checks, "pass": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    config = json.loads(args.config.read_text())
    source_config_path = ROOT / config["source_rpcd_config"]
    source_summary_path = ROOT / config["source_rpcd_summary"]
    source_config = json.loads(source_config_path.read_text())
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary["selected_shared_epoch"] != 8:
        raise ValueError("CCRR requires locked RPCD epoch 8")
    if float(source_summary["selected_shared_weight"]) != 0.2:
        raise ValueError("CCRR B1 requires locked RPCD weight 0.2")
    resolved = resolve_inputs(source_config)
    prepared = {}
    preflight = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(args.config),
        "source_rpcd_config_sha256": sha256(source_config_path),
        "source_rpcd_summary_sha256": sha256(source_summary_path),
        "feature_schema": FEATURE_SCHEMA,
        "test_predictions_read": False,
        "sequence_test_target_indexed": False,
        "datasets": {},
    }
    for dataset, spec in config["datasets"].items():
        loaded = load_dataset(resolved[dataset])
        teachers = read_teacher(ROOT / spec["teacher_top50"])
        users = set(loaded["sequences"])
        if set(teachers) != users:
            raise ValueError(f"{dataset}: teacher user mismatch")
        if any(
            teachers[user]["target"] != loaded["sequences"][user][-2] for user in users
        ):
            raise ValueError(f"{dataset}: validation target mismatch")
        calibration = {
            user
            for user in users
            if stable_fraction(user, config["selection"]["calibration_salt"])
            < float(config["selection"]["calibration_fraction"])
        }
        audit = users - calibration
        if calibration & audit or calibration | audit != users:
            raise AssertionError("split integrity failure")
        popularity, head = popularity_features(
            loaded["sequences"], loaded["catalog"]
        )
        tail = set(loaded["catalog"]) - head
        rows = prepare_rows(loaded, teachers)
        all_candidates = [len(row["candidates"]) for row in rows]
        catalog_set = set(loaded["catalog"])
        unknown = sum(
            item not in catalog_set
            for row in rows
            for item in row["candidates"]
        )
        if unknown:
            raise ValueError(f"{dataset}: unknown candidates={unknown}")
        prepared[dataset] = {
            "rows": rows,
            "popularity": popularity,
            "head": head,
            "tail": tail,
            "calibration": calibration,
            "audit": audit,
        }
        preflight["datasets"][dataset] = {
            "users": len(users),
            "catalog_items": len(loaded["catalog"]),
            "calibration_users": len(calibration),
            "audit_users": len(audit),
            "calibration_user_sha256": user_set_sha256(calibration),
            "audit_user_sha256": user_set_sha256(audit),
            "mean_union_candidates": float(np.mean(all_candidates)),
            "min_union_candidates": min(all_candidates),
            "max_union_candidates": max(all_candidates),
            "unknown_candidates": unknown,
            "duplicate_candidates": 0,
            "target_match_rate": 1.0,
        }
    write_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0

    calibration_results = {}
    fitted = {}
    calibration_gates = []
    for dataset, data in prepared.items():
        scaler, model, fit_details = fit_ranker(
            data["rows"],
            data["calibration"],
            data["popularity"],
            data["head"],
            config["model"],
        )
        if fit_details["fit_user_sha256"] != user_set_sha256(data["calibration"]):
            raise AssertionError(f"{dataset}: fit user leakage")
        if fit_details["fit_user_sha256"] == user_set_sha256(data["audit"]):
            raise AssertionError(f"{dataset}: audit users entered fit")
        result = evaluate(
            data["rows"],
            data["calibration"],
            data["popularity"],
            data["head"],
            data["tail"],
            scaler,
            model,
        )
        gate_row = {"dataset": dataset, **gate(result, config["gates"])}
        calibration_results[dataset] = result
        calibration_gates.append(gate_row)
        fitted[dataset] = (scaler, model, fit_details)

    calibration_qualified = all(row["pass"] for row in calibration_gates)
    audit = {}
    audit_gates = []
    if calibration_qualified:
        for offset, (dataset, data) in enumerate(prepared.items()):
            scaler, model, _ = fitted[dataset]
            result = evaluate(
                data["rows"],
                data["audit"],
                data["popularity"],
                data["head"],
                data["tail"],
                scaler,
                model,
                keep_arrays=True,
            )
            arrays = result.pop("_arrays")
            bootstrap = {
                "R1_vs_B0_ndcg10_relative_gain_ci95": paired_bootstrap(
                    arrays["ndcg"],
                    int(config["bootstrap"]["iterations"]),
                    int(config["seed"]) + 20 + offset,
                    True,
                ),
                "R1_vs_B0_recall10_absolute_gain_ci95": paired_bootstrap(
                    arrays["recall"],
                    int(config["bootstrap"]["iterations"]),
                    int(config["seed"]) + 30 + offset,
                    False,
                ),
                "tail_R1_vs_B0_ndcg10_relative_gain_ci95": paired_bootstrap(
                    arrays["tail_ndcg"],
                    int(config["bootstrap"]["iterations"]),
                    int(config["seed"]) + 40 + offset,
                    True,
                ),
                "R1_vs_B1_ndcg10_relative_gain_ci95": paired_bootstrap(
                    arrays["b1_ndcg"],
                    int(config["bootstrap"]["iterations"]),
                    int(config["seed"]) + 50 + offset,
                    True,
                ),
            }
            audit[dataset] = {**result, "bootstrap": bootstrap}
            audit_gates.append(
                {"dataset": dataset, **gate(result, config["gates"])}
            )
    passed = calibration_qualified and all(row["pass"] for row in audit_gates)
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": (
            "CCRR_R1_DESIGN_ALLOWED"
            if passed
            else "STOP_CCRR_NO_CANDIDATE_CONDITIONAL_EFFECT"
        ),
        "calibration_qualified": calibration_qualified,
        "calibration": calibration_results,
        "calibration_gate_rows": calibration_gates,
        "audit": audit,
        "audit_gate_rows": audit_gates,
        "models": {
            dataset: details for dataset, (_, _, details) in fitted.items()
        },
        "integrity": {
            "preflight_passed": True,
            "feature_schema_shared": True,
            "model_recipe_shared": True,
            "candidate_set_identity_rate": min(
                result["candidate_set_identity_rate"]
                for result in calibration_results.values()
            ),
            "audit_rows_used_for_fit": 0,
            "target_match_rate": 1.0,
            "test_predictions_read": False,
            "sequence_test_target_indexed": False,
            "model_optimizer_steps": 0,
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "calibration_qualified": calibration_qualified,
                "calibration": calibration_results,
                "calibration_gate_rows": calibration_gates,
                "audit": audit,
                "audit_gate_rows": audit_gates,
                "integrity": summary["integrity"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
