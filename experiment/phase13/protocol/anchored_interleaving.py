"""Warm-anchored constrained interleaving for Phase-13 v1-R² P2.

The P0 GRAM list is the safety anchor. A calibration-selected, fixed number of
catalog-known cold resolver candidates is inserted after a protected prefix.
The untouched audit half is evaluated exactly once after configuration choice.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from route_admission import read_prediction_records, stable_partition
from route_resolve import (
    atomic_json,
    average_metrics,
    ranking_metrics,
    read_key_value_lines,
    read_set,
    sha256_file,
    unique_in_order,
)


PREFIX_GRID = (5, 6, 7, 8, 9)
QUOTA_GRID = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--item-id-file", required=True)
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calibration-warm-retention", type=float, default=0.98)
    p.add_argument("--audit-warm-retention", type=float, default=0.97)
    p.add_argument("--audit-cold-improvement", type=float, default=2.0)
    return p.parse_args()


def anchored_interleave(
    gram_items: list[str],
    resolver_items: list[str],
    cold_items: set[str],
    protected_prefix: int,
    cold_quota: int,
) -> tuple[list[str], list[str]]:
    """Protect a GRAM prefix, insert cold items, then preserve source orders."""
    gram = unique_in_order(gram_items)
    resolver = unique_in_order(resolver_items)
    protected = gram[:protected_prefix]
    protected_set = set(protected)
    inserted = [
        item for item in resolver
        if item in cold_items and item not in protected_set
    ][:cold_quota]
    ranking = unique_in_order([*protected, *inserted, *gram[protected_prefix:], *resolver])
    return ranking, inserted


def metric_summary(rows: list[dict], cold_items: set[str], config: tuple[int, int] | None) -> dict:
    metrics = {name: [] for name in ("all", "warm", "cold")}
    inserted_counts: list[int] = []
    for row in rows:
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        gram = unique_in_order(row["v0_top50"])
        resolver = unique_in_order(row["resolver_top50"])
        if config is None:
            ranking = gram
            inserted = []
        else:
            ranking, inserted = anchored_interleave(
                gram, resolver, cold_items, config[0], config[1]
            )
        result = ranking_metrics(ranking, target)
        metrics["all"].append(result)
        metrics[slice_name].append(result)
        inserted_counts.append(len(inserted))
    return {
        "metrics": {name: average_metrics(values) for name, values in metrics.items()},
        "mean_inserted_cold": sum(inserted_counts) / max(len(inserted_counts), 1),
        "full_quota_rate": (
            sum(n == config[1] for n in inserted_counts) / max(len(inserted_counts), 1)
            if config is not None else 0.0
        ),
    }


def select_config(grid_rows: list[dict]) -> dict | None:
    feasible = [row for row in grid_rows if row["feasible"]]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            row["metrics"]["cold"]["ndcg@10"],
            row["metrics"]["all"]["ndcg@10"],
            row["metrics"]["warm"]["ndcg@10"],
            -row["cold_quota"],
            row["protected_prefix"],
        ),
    )


def evaluate_rankings(
    rows: list[dict],
    cold_items: set[str],
    config: tuple[int, int],
    catalog: set[str],
) -> tuple[dict, list[dict]]:
    model_names = ("v0_gram", "resolver_only", "p2_interleaving", "label_aware_oracle")
    metric_rows = {
        model: {name: [] for name in ("all", "warm", "cold")}
        for model in model_names
    }
    output_records: list[dict] = []
    inserted_total = 0
    inserted_full = 0
    for row in rows:
        uid = str(row["user_id"])
        target = str(row["target"])
        slice_name = "cold" if target in cold_items else "warm"
        gram = unique_in_order(row["v0_top50"])
        resolver = unique_in_order(row["resolver_top50"])
        p2, inserted = anchored_interleave(gram, resolver, cold_items, *config)
        oracle = resolver if slice_name == "cold" else gram
        rankings = {
            "v0_gram": gram,
            "resolver_only": resolver,
            "p2_interleaving": p2,
            "label_aware_oracle": oracle,
        }
        for model, ranking in rankings.items():
            if len(ranking) != len(set(ranking)):
                raise RuntimeError(f"Duplicate output for {uid}/{model}")
            if not set(ranking) <= catalog:
                raise RuntimeError(f"Non-catalog output for {uid}/{model}")
            result = ranking_metrics(ranking, target)
            metric_rows[model]["all"].append(result)
            metric_rows[model][slice_name].append(result)
        if p2[:config[0]] != gram[:config[0]]:
            raise RuntimeError(f"Protected prefix changed for {uid}")
        inserted_total += len(inserted)
        inserted_full += int(len(inserted) == config[1])
        output_records.append({
            "user_id": uid,
            "target": target,
            "is_cold": slice_name == "cold",
            "protected_prefix": config[0],
            "cold_quota": config[1],
            "inserted_cold_items": inserted,
            "v0_top50": gram,
            "resolver_top50": resolver,
            "p2_top50": p2[:50],
        })
    metrics = {
        model: {name: average_metrics(values) for name, values in slices.items()}
        for model, slices in metric_rows.items()
    }
    diagnostics = {
        "mean_inserted_cold": inserted_total / max(len(rows), 1),
        "full_quota_rate": inserted_full / max(len(rows), 1),
    }
    return {"metrics": metrics, "diagnostics": diagnostics}, output_records


def main() -> None:
    args = parse_args()
    started = time.time()
    p0_path = Path(args.p0_predictions).resolve()
    item_id_path = Path(args.item_id_file).resolve()
    cold_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log"}
    unexpected = [p.name for p in output_dir.iterdir() if p.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing P2 scientific artifacts: {unexpected}")
    for path in (p0_path, item_id_path, cold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing a test input in P2: {path}")

    records = read_prediction_records(p0_path)
    catalog = set(read_key_value_lines(item_id_path))
    cold_items = read_set(cold_path)
    if not cold_items <= catalog:
        raise ValueError("Cold-item metadata contains IDs outside catalog")
    calibration: list[dict] = []
    audit: list[dict] = []
    for row in records:
        uid = str(row["user_id"])
        target = str(row["target"])
        if target not in catalog:
            raise ValueError(f"Target outside catalog for {uid}")
        if bool(row["is_cold"]) != (target in cold_items):
            raise ValueError(f"Cold-state mismatch for {uid}")
        for key in ("v0_top50", "resolver_top50"):
            if len(row[key]) != len(set(row[key])) or not set(row[key]) <= catalog:
                raise ValueError(f"Invalid source ranking for {uid}/{key}")
        (calibration if stable_partition(uid) == "calibration" else audit).append(row)
    calibration_uids = {str(row["user_id"]) for row in calibration}
    audit_uids = {str(row["user_id"]) for row in audit}
    if calibration_uids & audit_uids or len(calibration_uids | audit_uids) != len(records):
        raise RuntimeError("Calibration/audit split integrity failure")

    config_payload = {
        **vars(args),
        "p0_predictions": str(p0_path),
        "item_id_file": str(item_id_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P2_ANCHORED_INTERLEAVING",
        "split": "validation_calibration_audit",
        "partition_rule": "sha256(user_id)[0] parity: even=calibration, odd=audit",
        "prefix_grid": PREFIX_GRID,
        "quota_grid": QUOTA_GRID,
        "selection_order": "max cold ndcg10, all ndcg10, warm ndcg10; min quota; max prefix",
        "test_predictions_opened": False,
        "input_sha256": {
            str(path): sha256_file(path) for path in (p0_path, item_id_path, cold_path)
        },
        "n_catalog_items": len(catalog),
        "n_calibration": len(calibration),
        "n_audit": len(audit),
    }
    atomic_json(output_dir / "config.json", config_payload)

    baseline = metric_summary(calibration, cold_items, None)["metrics"]
    grid_rows: list[dict] = []
    for prefix in PREFIX_GRID:
        for quota in QUOTA_GRID:
            if prefix + quota > 10:
                continue
            evaluation = metric_summary(calibration, cold_items, (prefix, quota))
            metrics = evaluation["metrics"]
            feasible = (
                metrics["warm"]["ndcg@10"]
                >= args.calibration_warm_retention * baseline["warm"]["ndcg@10"]
                and metrics["all"]["ndcg@10"] > baseline["all"]["ndcg@10"]
            )
            grid_rows.append({
                "protected_prefix": prefix,
                "cold_quota": quota,
                "feasible": feasible,
                "metrics": metrics,
                "mean_inserted_cold": evaluation["mean_inserted_cold"],
                "full_quota_rate": evaluation["full_quota_rate"],
            })
            print(
                f"[calibration] prefix={prefix} quota={quota} feasible={feasible} "
                f"warm={metrics['warm']['ndcg@10']:.6f} "
                f"cold={metrics['cold']['ndcg@10']:.6f} "
                f"all={metrics['all']['ndcg@10']:.6f}",
                flush=True,
            )
    atomic_json(output_dir / "calibration_grid.json", {
        "baseline_v0": baseline,
        "rows": grid_rows,
        "audit_opened_for_selection": False,
    })
    selected = select_config(grid_rows)
    if selected is None:
        summary = {
            "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P2_ANCHORED_INTERLEAVING",
            "status": "completed",
            "verdict": "FAIL_STOP_R2_P2",
            "reason": "NO_CALIBRATION_FEASIBLE_CONFIG",
            "selected_config": None,
            "calibration_baseline": baseline,
            "n_calibration": len(calibration),
            "n_audit": len(audit),
            "audit_evaluated": False,
            "test_predictions_opened": False,
            "runtime_seconds": time.time() - started,
        }
        atomic_json(output_dir / "summary.json", summary)
        print("[result] verdict=FAIL_STOP_R2_P2 no feasible calibration config", flush=True)
        return

    selected_config = (selected["protected_prefix"], selected["cold_quota"])
    audit_result, output_records = evaluate_rankings(
        audit, cold_items, selected_config, catalog
    )
    metrics = audit_result["metrics"]
    p2 = metrics["p2_interleaving"]
    v0 = metrics["v0_gram"]
    gates = {
        "calibration_feasible_config_exists": True,
        "audit_warm_ndcg10_ge_0_97x_v0": (
            p2["warm"]["ndcg@10"] >= args.audit_warm_retention * v0["warm"]["ndcg@10"]
        ),
        "audit_cold_ndcg10_ge_2x_v0": (
            p2["cold"]["ndcg@10"] >= args.audit_cold_improvement * v0["cold"]["ndcg@10"]
        ),
        "audit_cold_hit10_ge_2x_v0": (
            p2["cold"]["hit@10"] >= args.audit_cold_improvement * v0["cold"]["hit@10"]
        ),
        "audit_all_ndcg10_gt_v0": p2["all"]["ndcg@10"] > v0["all"]["ndcg@10"],
        "catalog_outputs_unique": True,
        "calibration_audit_disjoint": not bool(calibration_uids & audit_uids),
        "audit_not_used_for_selection": True,
        "validation_only": True,
    }
    verdict = (
        "PASS_TO_R2_MEDIUM_SMOKE_DISCUSSION"
        if all(gates.values()) else "FAIL_STOP_R2_P2"
    )
    summary = {
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P2_ANCHORED_INTERLEAVING",
        "status": "completed",
        "verdict": verdict,
        "selected_config": {
            "protected_prefix": selected_config[0],
            "cold_quota": selected_config[1],
        },
        "selection_metrics_calibration": selected["metrics"],
        "calibration_baseline": baseline,
        "metrics_audit": metrics,
        "audit_diagnostics": audit_result["diagnostics"],
        "gates": gates,
        "n_calibration": len(calibration),
        "n_audit": len(audit),
        "test_predictions_opened": False,
        "label_aware_oracle_is_diagnostic_only": True,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_audit.jsonl").open("w") as f:
        for row in output_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} prefix={selected_config[0]} quota={selected_config[1]} "
        f"warm={p2['warm']['ndcg@10']:.6f} cold={p2['cold']['ndcg@10']:.6f} "
        f"all={p2['all']['ndcg@10']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
