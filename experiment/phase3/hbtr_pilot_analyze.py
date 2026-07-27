#!/usr/bin/env python3
"""Analyze the locked HBTR 10% pilot and apply its preregistered gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("Toys", "Beauty")
CONTROLS = ("C0", "C1", "C2", "C3", "C4")
METRICS = ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10")
SEED = 2023
BOOTSTRAP_RESAMPLES = 10_000


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def relative_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return math.inf if value > 0 else 0.0
    return (value - baseline) / baseline


def paired_bootstrap(
    lhs: list[dict], rhs: list[dict], metric: str, seed: int
) -> dict[str, float]:
    lhs_by_user = {row["user_id"]: row for row in lhs}
    rhs_by_user = {row["user_id"]: row for row in rhs}
    if set(lhs_by_user) != set(rhs_by_user):
        raise ValueError("paired bootstrap user sets differ")
    users = sorted(lhs_by_user)
    differences = np.asarray(
        [lhs_by_user[user][metric] - rhs_by_user[user][metric] for user in users],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    # Chunked indices keep peak analysis memory bounded.
    chunk = 250
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        size = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(users), size=(size, len(users)))
        bootstrap_means[start : start + size] = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "probability_positive": float((bootstrap_means > 0).mean()),
        "resamples": BOOTSTRAP_RESAMPLES,
        "users": len(users),
    }


def add_gate(gates: list[dict], name: str, passed: bool, observed, threshold) -> None:
    gates.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "threshold": threshold,
        }
    )


def analyze(root: Path) -> dict:
    summaries: dict[str, dict[str, dict]] = {}
    rows: dict[str, dict[str, list[dict]]] = {}
    training: dict[str, dict[str, dict]] = {}
    integrity_errors: list[str] = []
    for dataset in DATASETS:
        summaries[dataset] = {}
        rows[dataset] = {}
        training[dataset] = {}
        reference_users = None
        for control in CONTROLS:
            control_dir = root / dataset / control
            summaries[dataset][control] = load_json(
                control_dir / "validation_summary.json"
            )
            training[dataset][control] = load_json(
                control_dir / "training_summary.json"
            )
            rows[dataset][control] = load_rows(
                control_dir / "validation_per_user.csv"
            )
            users = [row["user_id"] for row in rows[dataset][control]]
            if len(users) != 2048 or len(set(users)) != 2048:
                integrity_errors.append(
                    f"{dataset}/{control}: validation users are not 2048 unique users"
                )
            if reference_users is None:
                reference_users = users
            elif users != reference_users:
                integrity_errors.append(
                    f"{dataset}/{control}: validation user order differs from C0"
                )
            if summaries[dataset][control].get("test_data_read") is not False:
                integrity_errors.append(f"{dataset}/{control}: test_data_read is not false")
            if training[dataset][control].get("test_data_read") is not False:
                integrity_errors.append(f"{dataset}/{control}: training read test data")

    bootstrap = {}
    for dataset_index, dataset in enumerate(DATASETS):
        bootstrap[dataset] = {}
        for control_index, control in enumerate(CONTROLS[:-1]):
            bootstrap[dataset][f"C4_vs_{control}"] = paired_bootstrap(
                rows[dataset]["C4"],
                rows[dataset][control],
                "NDCG@10",
                SEED + dataset_index * 100 + control_index,
            )

    gates: list[dict] = []
    add_gate(gates, "protocol_integrity", not integrity_errors, integrity_errors, [])
    c4_c0_relative = {}
    for dataset in DATASETS:
        c4 = summaries[dataset]["C4"]["groups"]
        c0 = summaries[dataset]["C0"]["groups"]
        ndcg_rel = relative_change(c4["overall"]["NDCG@10"], c0["overall"]["NDCG@10"])
        c4_c0_relative[dataset] = ndcg_rel
        add_gate(
            gates,
            f"{dataset}_C4_vs_C0_ndcg10_positive",
            c4["overall"]["NDCG@10"] > c0["overall"]["NDCG@10"],
            c4["overall"]["NDCG@10"] - c0["overall"]["NDCG@10"],
            "> 0",
        )
        add_gate(
            gates,
            f"{dataset}_C4_vs_C0_recall10_no_decline",
            c4["overall"]["Recall@10"] >= c0["overall"]["Recall@10"],
            c4["overall"]["Recall@10"] - c0["overall"]["Recall@10"],
            ">= 0",
        )
        for metric in ("Recall@10", "NDCG@10"):
            tail_rel = relative_change(c4["tail"][metric], c0["tail"][metric])
            add_gate(
                gates,
                f"{dataset}_tail_{metric}_relative_decline",
                tail_rel >= -0.01,
                tail_rel,
                ">= -0.01",
            )
    add_gate(
        gates,
        "at_least_one_dataset_C4_ndcg10_relative_gain_2pct",
        max(c4_c0_relative.values()) >= 0.02,
        c4_c0_relative,
        ">= 0.02 for at least one dataset",
    )

    macro = {
        control: sum(
            summaries[dataset][control]["groups"]["overall"]["NDCG@10"]
            for dataset in DATASETS
        )
        / len(DATASETS)
        for control in CONTROLS
    }
    for component in ("C1", "C2", "C3"):
        add_gate(
            gates,
            f"C4_macro_ndcg10_exceeds_{component}",
            macro["C4"] > macro[component],
            macro["C4"] - macro[component],
            "> 0",
        )
    for dataset in DATASETS:
        c4_ndcg = summaries[dataset]["C4"]["groups"]["overall"]["NDCG@10"]
        best_component = max(
            summaries[dataset][control]["groups"]["overall"]["NDCG@10"]
            for control in ("C1", "C2", "C3")
        )
        rel = relative_change(c4_ndcg, best_component)
        add_gate(
            gates,
            f"{dataset}_C4_within_0.5pct_best_component",
            rel >= -0.005,
            rel,
            ">= -0.005",
        )

    resource_ratios = {}
    for dataset in DATASETS:
        c0_train = training[dataset]["C0"]
        c4_train = training[dataset]["C4"]
        c0_valid = summaries[dataset]["C0"]
        c4_valid = summaries[dataset]["C4"]
        resource_ratios[dataset] = {
            "peak_reserved_increase": relative_change(
                c4_train["peak_reserved_mib"], c0_train["peak_reserved_mib"]
            ),
            "training_wall_time_increase": relative_change(
                c4_train["wall_time_seconds"], c0_train["wall_time_seconds"]
            ),
            "validation_latency_increase": relative_change(
                c4_valid["per_user_latency_seconds"],
                c0_valid["per_user_latency_seconds"],
            ),
        }
        limits = {
            "peak_reserved_increase": 0.25,
            "training_wall_time_increase": 1.0,
            "validation_latency_increase": 0.05,
        }
        for key, limit in limits.items():
            add_gate(
                gates,
                f"{dataset}_{key}",
                resource_ratios[dataset][key] <= limit,
                resource_ratios[dataset][key],
                f"<= {limit}",
            )

    failed = [gate for gate in gates if not gate["passed"]]
    if integrity_errors:
        decision = "STOP"
        decision_reason = "protocol integrity failure"
    elif not failed:
        decision = "GO"
        decision_reason = "all preregistered gates passed"
    elif len(failed) == 1:
        decision = "MODIFY"
        decision_reason = "one integrity-clean preregistered gate failed"
    else:
        decision = "STOP"
        decision_reason = f"{len(failed)} preregistered gates failed"

    return {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "hbtr_pilot_analysis_v1",
            "design_status": "EXPLORATORY_NO_EFFECT_CLAIM",
        },
        "decision": decision,
        "decision_reason": decision_reason,
        "metrics": summaries,
        "macro_ndcg10": macro,
        "paired_bootstrap_ndcg10": bootstrap,
        "resource_ratios": resource_ratios,
        "gates": gates,
        "failed_gate_count": len(failed),
        "effect_claim_allowed": False,
        "test_data_read": False,
    }


def write_comparison(path: Path, analysis: dict) -> None:
    fields = ["dataset", "control", "group", *METRICS, "n"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for control in CONTROLS:
                groups = analysis["metrics"][dataset][control]["groups"]
                for group, values in groups.items():
                    writer.writerow(
                        {"dataset": dataset, "control": control, "group": group, **values}
                    )


def write_report(path: Path, analysis: dict) -> None:
    lines = [
        "# GRAM 第三阶段 HBTR 10% Pilot 报告",
        "",
        f"- 决策：**{analysis['decision']}**",
        f"- 原因：{analysis['decision_reason']}",
        "- 证据级别：探索性机制筛选；不允许效果声明；全程未读取测试目标。",
        "",
        "## 主结果",
        "",
        "| 数据集 | C0 NDCG@10 | C4 NDCG@10 | 相对变化 | C0 Recall@10 | C4 Recall@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        c0 = analysis["metrics"][dataset]["C0"]["groups"]["overall"]
        c4 = analysis["metrics"][dataset]["C4"]["groups"]["overall"]
        lines.append(
            f"| {dataset} | {c0['NDCG@10']:.6f} | {c4['NDCG@10']:.6f} | "
            f"{relative_change(c4['NDCG@10'], c0['NDCG@10']):+.2%} | "
            f"{c0['Recall@10']:.6f} | {c4['Recall@10']:.6f} |"
        )
    lines += ["", "## 预注册门槛", ""]
    for gate in analysis["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        lines.append(
            f"- [{mark}] `{gate['name']}`：observed={gate['observed']}; "
            f"threshold={gate['threshold']}"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "该 pilot 是从锁定全量基线继续训练的 10% 用户机制筛选，不是独立重复实验，也未使用测试集。"
        "Bootstrap 区间仅描述锁定验证用户上的配对不确定性，不构成确认性显著性结论。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=ROOT / "artifacts/phase3/hbtr_pilot"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_HBTR_10%Pilot报告.md",
    )
    args = parser.parse_args()
    analysis = analyze(args.input_root)
    args.input_root.mkdir(parents=True, exist_ok=True)
    with (args.input_root / "summary.json").open("w") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_comparison(args.input_root / "comparison.csv", analysis)
    write_report(args.report, analysis)
    print(json.dumps({"decision": analysis["decision"], "reason": analysis["decision_reason"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
