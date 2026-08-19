"""Validation-only candidate admission for Phase-13 v1-R² P1.

P1 reuses the frozen P0 GRAM and resolver top-50 lists.  Validation users are
split deterministically into calibration and audit halves.  A linear pairwise
ranker is fitted only on calibration candidates, using catalog-visible item
newness plus rank/route signals; the untouched audit half determines the Gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from route_resolve import (
    atomic_json,
    average_metrics,
    ranking_metrics,
    read_key_value_lines,
    read_set,
    semantic_route,
    sha256_file,
    unique_in_order,
)


FEATURE_NAMES = (
    "gram_rr",
    "resolver_rr",
    "depth1_route_rr",
    "in_both",
    "is_cold_item",
    "cold_x_gram_rr",
    "cold_x_resolver_rr",
    "cold_x_route_rr",
    "cold_x_in_both",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--item-id-file", required=True)
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--route-depth", type=int, default=1)
    return p.parse_args()


def stable_partition(user_id: str) -> str:
    """Stable 50/50 split independent of row order and Python hash salt."""
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    return "calibration" if digest[0] % 2 == 0 else "audit"


def read_prediction_records(path: Path) -> list[dict]:
    if "test" in path.name.lower():
        raise ValueError(f"Refusing a test prediction file in P1: {path}")
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open() as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            uid = str(row["user_id"])
            if uid in seen:
                raise ValueError(f"Duplicate user at {path}:{line_no}: {uid}")
            seen.add(uid)
            for key in ("target", "is_cold", "v0_top50", "resolver_top50"):
                if key not in row:
                    raise KeyError(f"Missing {key} at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No P0 predictions in {path}")
    return rows


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def candidate_features(
    candidate: str,
    gram_items: list[str],
    resolver_items: list[str],
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
) -> list[float]:
    gram_rank = {item: rank for rank, item in enumerate(gram_items, 1)}
    resolver_rank = {item: rank for rank, item in enumerate(resolver_items, 1)}
    route_order = unique_in_order(item_routes[item] for item in gram_items)
    route_rank = {route: rank for rank, route in enumerate(route_order, 1)}
    gram_rr = reciprocal_rank(gram_rank.get(candidate))
    resolver_rr = reciprocal_rank(resolver_rank.get(candidate))
    route_rr = reciprocal_rank(route_rank.get(item_routes[candidate]))
    in_both = float(candidate in gram_rank and candidate in resolver_rank)
    is_cold = float(candidate in cold_items)
    return [
        gram_rr,
        resolver_rr,
        route_rr,
        in_both,
        is_cold,
        is_cold * gram_rr,
        is_cold * resolver_rr,
        is_cold * route_rr,
        is_cold * in_both,
    ]


def build_candidates_and_features(
    row: dict,
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
) -> tuple[list[str], torch.Tensor]:
    gram = unique_in_order(row["v0_top50"])
    resolver = unique_in_order(row["resolver_top50"])
    candidates = unique_in_order([*gram, *resolver])
    features = torch.tensor(
        [candidate_features(x, gram, resolver, item_routes, cold_items) for x in candidates],
        dtype=torch.float32,
    )
    return candidates, features


def fit_pairwise_ranker(
    rows: list[dict],
    item_routes: dict[str, tuple[str, ...]],
    cold_items: set[str],
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[torch.Tensor, dict, list[dict]]:
    differences: list[torch.Tensor] = []
    covered = 0
    covered_cold = 0
    covered_warm = 0
    for row in rows:
        candidates, features = build_candidates_and_features(row, item_routes, cold_items)
        if row["target"] not in candidates:
            continue
        covered += 1
        covered_cold += int(row["target"] in cold_items)
        covered_warm += int(row["target"] not in cold_items)
        positive_idx = candidates.index(row["target"])
        mask = torch.arange(len(candidates)) != positive_idx
        differences.append(features[positive_idx].unsqueeze(0) - features[mask])
    if not differences:
        raise ValueError("No calibration target occurs in the frozen candidate union")
    pair_diffs = torch.cat(differences, dim=0)

    torch.manual_seed(seed)
    weights = torch.nn.Parameter(torch.zeros(len(FEATURE_NAMES)))
    optimizer = torch.optim.Adam([weights], lr=lr)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        margins = pair_diffs @ weights
        pairwise_loss = F.softplus(-margins).mean()
        regularizer = l2 * weights.square().sum()
        loss = pairwise_loss + regularizer
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            record = {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "pairwise_loss": float(pairwise_loss.detach()),
                "l2_penalty": float(regularizer.detach()),
            }
            history.append(record)
            print(
                f"[fit] epoch={epoch}/{epochs} loss={record['loss']:.6f}",
                flush=True,
            )
    report = {
        "n_calibration_users": len(rows),
        "n_covered_users": covered,
        "n_covered_cold": covered_cold,
        "n_covered_warm": covered_warm,
        "n_pairwise_examples": len(pair_diffs),
    }
    return weights.detach(), report, history


def rank_candidates(candidates: list[str], features: torch.Tensor, weights: torch.Tensor) -> list[str]:
    scores = features @ weights
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
    return [candidates[i] for i in order]


def summarize_metrics(metric_rows: dict[str, dict[str, list[dict[str, float]]]]) -> dict:
    return {
        model: {slice_name: average_metrics(rows) for slice_name, rows in slices.items()}
        for model, slices in metric_rows.items()
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    p0_path = Path(args.p0_predictions).resolve()
    item_id_path = Path(args.item_id_file).resolve()
    cold_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log"}
    unexpected = [p.name for p in output_dir.iterdir() if p.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing P1 scientific artifacts: {unexpected}")
    for path in (p0_path, item_id_path, cold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing a test input in P1: {path}")

    records = read_prediction_records(p0_path)
    item_to_lexical = read_key_value_lines(item_id_path)
    catalog = set(item_to_lexical)
    cold_items = read_set(cold_path)
    if not cold_items <= catalog:
        raise ValueError("Cold-item metadata contains IDs outside the catalog")
    item_routes = {
        item: semantic_route(lexical, args.route_depth)
        for item, lexical in item_to_lexical.items()
    }

    calibration: list[dict] = []
    audit: list[dict] = []
    all_uids: set[str] = set()
    for row in records:
        uid = str(row["user_id"])
        target = str(row["target"])
        if target not in catalog:
            raise ValueError(f"Target outside catalog for {uid}: {target}")
        if bool(row["is_cold"]) != (target in cold_items):
            raise ValueError(f"P0 cold-state mismatch for {uid}; refusing leaked metadata")
        for key in ("v0_top50", "resolver_top50"):
            ranking = row[key]
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"Duplicate candidate in {uid}/{key}")
            if not set(ranking) <= catalog:
                raise ValueError(f"Non-catalog candidate in {uid}/{key}")
        all_uids.add(uid)
        (calibration if stable_partition(uid) == "calibration" else audit).append(row)
    calibration_uids = {str(row["user_id"]) for row in calibration}
    audit_uids = {str(row["user_id"]) for row in audit}
    if calibration_uids & audit_uids or calibration_uids | audit_uids != all_uids:
        raise RuntimeError("Calibration/audit partition integrity failure")
    if not calibration or not audit:
        raise RuntimeError("Stable partition produced an empty split")

    config = {
        **vars(args),
        "p0_predictions": str(p0_path),
        "item_id_file": str(item_id_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P1_ADMISSION",
        "split": "validation_calibration_audit",
        "partition_rule": "sha256(user_id)[0] parity: even=calibration, odd=audit",
        "feature_names": FEATURE_NAMES,
        "candidate_pool": "unique(v0_top50 + resolver_top50)",
        "catalog_state_available_at_inference": True,
        "test_predictions_opened": False,
        "input_sha256": {
            str(path): sha256_file(path) for path in (p0_path, item_id_path, cold_path)
        },
        "n_catalog_items": len(catalog),
        "n_records": len(records),
        "n_calibration": len(calibration),
        "n_audit": len(audit),
    }
    atomic_json(output_dir / "config.json", config)

    weights, fit_report, training_history = fit_pairwise_ranker(
        calibration,
        item_routes,
        cold_items,
        args.epochs,
        args.lr,
        args.l2,
        args.seed,
    )
    torch.save(
        {
            "weights": weights,
            "feature_names": FEATURE_NAMES,
            "route_depth": args.route_depth,
            "seed": args.seed,
            "fit_report": fit_report,
        },
        output_dir / "admission.pt",
    )

    model_names = ("v0_gram", "resolver_only", "p1_admission", "label_aware_oracle")
    metric_rows = {
        model: {slice_name: [] for slice_name in ("all", "warm", "cold")}
        for model in model_names
    }
    candidate_hits = {slice_name: 0 for slice_name in ("all", "warm", "cold")}
    output_records: list[dict] = []
    for row in audit:
        uid = str(row["user_id"])
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        gram = unique_in_order(row["v0_top50"])
        resolver = unique_in_order(row["resolver_top50"])
        candidates, features = build_candidates_and_features(row, item_routes, cold_items)
        p1 = rank_candidates(candidates, features, weights)
        oracle = resolver if slice_name == "cold" else gram
        rankings = {
            "v0_gram": gram,
            "resolver_only": resolver,
            "p1_admission": p1,
            "label_aware_oracle": oracle,
        }
        for model, ranking in rankings.items():
            if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
                raise RuntimeError(f"Invalid output for audit user {uid}/{model}")
            metrics = ranking_metrics(ranking, target)
            metric_rows[model]["all"].append(metrics)
            metric_rows[model][slice_name].append(metrics)
        for name in ("all", slice_name):
            candidate_hits[name] += int(target in candidates)
        output_records.append({
            "user_id": uid,
            "target": target,
            "is_cold": slice_name == "cold",
            "candidate_union_hit": target in candidates,
            "v0_top50": gram,
            "resolver_top50": resolver,
            "p1_top50": p1[:50],
        })

    metrics = summarize_metrics(metric_rows)
    audit_counts = {
        slice_name: len(metric_rows["p1_admission"][slice_name])
        for slice_name in ("all", "warm", "cold")
    }
    candidate_ceiling = {
        slice_name: candidate_hits[slice_name] / max(audit_counts[slice_name], 1)
        for slice_name in audit_counts
    }
    p1_cold = metrics["p1_admission"]["cold"]["ndcg@10"]
    resolver_cold = metrics["resolver_only"]["cold"]["ndcg@10"]
    p1_warm = metrics["p1_admission"]["warm"]["ndcg@10"]
    v0_warm = metrics["v0_gram"]["warm"]["ndcg@10"]
    p1_all = metrics["p1_admission"]["all"]["ndcg@10"]
    v0_all = metrics["v0_gram"]["all"]["ndcg@10"]
    gates = {
        "p1_cold_ndcg10_ge_0_90x_resolver": p1_cold >= 0.90 * resolver_cold,
        "p1_warm_ndcg10_ge_0_97x_v0": p1_warm >= 0.97 * v0_warm,
        "p1_all_ndcg10_gt_v0": p1_all > v0_all,
        "catalog_outputs_unique": True,
        "calibration_audit_disjoint": not bool(calibration_uids & audit_uids),
        "audit_not_used_for_fit": True,
        "validation_only": True,
    }
    verdict = "PASS_TO_R2_P2_DISCUSSION" if all(gates.values()) else "FAIL_STOP_R2_P1"
    summary = {
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P1_ADMISSION",
        "status": "completed",
        "verdict": verdict,
        "split": "validation_audit",
        "test_predictions_opened": False,
        "metrics": metrics,
        "audit_candidate_union_hit_rate": candidate_ceiling,
        "gates": gates,
        "fit_report": fit_report,
        "feature_weights": dict(zip(FEATURE_NAMES, [float(x) for x in weights])),
        "training_history": training_history,
        "n_calibration": len(calibration),
        "n_audit": len(audit),
        "runtime_seconds": time.time() - started,
        "label_aware_oracle_is_diagnostic_only": True,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_audit.jsonl").open("w") as f:
        for row in output_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} audit={len(audit)} "
        f"cold_ndcg10={p1_cold:.6f} warm_ndcg10={p1_warm:.6f} "
        f"all_ndcg10={p1_all:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
