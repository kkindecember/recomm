#!/usr/bin/env python3
"""Post-hoc S0b reliability-abstention probe on locked validation predictions."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from s0_offline_diagnostics import (
    decode_item_ids,
    head_items,
    metric_at_k,
    read_neighbors,
    read_predictions,
    read_sequences,
    sha256,
    split_sample,
    training_popularity,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts/phase3/s0b"
REPORT_PATH = ROOT / "report/第三阶段/GRAM_第三阶段_S0b可靠性拒绝探针报告.md"
REGISTRY_PATH = ROOT / "artifacts/phase3/experiment_registry.csv"
PROMOTION_PATH = ROOT / "artifacts/phase3/promotion_decisions.md"

K = 20
RECENCY_DECAY = 0.90
MARGIN = 0.05
BETAS = (0.0, 0.25)
LAMBDAS = (0.05, 0.20)
THRESHOLDS = (0.50, 0.75)
MIN_SUPPORTS = (1, 2)
METRIC_KS = (5, 10, 50)

DATASETS = {
    "Toys": {
        "prediction": ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv",
        "data_dir": ROOT / "GRAM/rec_datasets/Toys",
    },
    "Beauty": {
        "prediction": ROOT / "GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv",
        "data_dir": ROOT / "GRAM/rec_datasets/Beauty",
    },
}


def config_id(beta: float, lam: float, tau: float, min_support: int) -> str:
    return f"b{beta:g}_l{lam:g}_t{tau:g}_s{min_support}"


def confidence_features(
    history: Sequence[str],
    candidates: Sequence[str],
    neighbors: Mapping[str, Sequence[str]],
    beta: float,
) -> Tuple[List[float], List[int], float, float, int]:
    """Compute inference-only relation confidence; no target is accepted."""
    denom = max(1, len(history))
    scores: List[float] = []
    supports: List[int] = []
    for candidate in candidates:
        best = 0.0
        supporters = 0
        if candidate is not None:
            for age, anchor in enumerate(reversed(history)):
                nearest = neighbors.get(anchor, ())[:K]
                try:
                    rank = nearest.index(candidate) + 1
                except ValueError:
                    continue
                supporters += 1
                best = max(
                    best,
                    (RECENCY_DECAY**age) / math.log2(rank + 1.0),
                )
        scores.append(best + beta * supporters / denom)
        supports.append(supporters)
    ordered = sorted(scores, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return scores, supports, top1, top1 - top2, max(supports, default=0)


def abstention_active(top1: float, gap: float, max_support: int, tau: float) -> bool:
    return top1 >= tau and (gap >= MARGIN or max_support >= 2)


def rerank(
    candidates: Sequence[str],
    model_scores: Sequence[float],
    relation_scores: Sequence[float],
    supports: Sequence[int],
    active: bool,
    lam: float,
    min_support: int,
) -> List[str]:
    if not active:
        return list(candidates)
    scored = []
    for rank, (candidate, model_score, relation_score, support) in enumerate(
        zip(candidates, model_scores, relation_scores, supports)
    ):
        boost = lam * relation_score if support >= min_support else 0.0
        scored.append((model_score + boost, -rank, candidate))
    scored.sort(reverse=True)
    return [row[2] for row in scored]


def summarize(pairs: Sequence[Tuple[Sequence[str], str]]) -> dict:
    totals = {f"recall@{k}": 0.0 for k in METRIC_KS}
    totals.update({f"ndcg@{k}": 0.0 for k in METRIC_KS})
    for ranking, gold in pairs:
        for k in METRIC_KS:
            recall, ndcg = metric_at_k(ranking, gold, k)
            totals[f"recall@{k}"] += recall
            totals[f"ndcg@{k}"] += ndcg
    n = len(pairs)
    return {"n": n, **{key: value / n for key, value in totals.items()}}


def prepare_dataset(dataset: str) -> dict:
    spec = DATASETS[dataset]
    data_dir = spec["data_dir"]
    index_paths = sorted(data_dir.glob("item_generative_indexing_hierarchy_*.txt"))
    if len(index_paths) != 1:
        raise ValueError(f"{dataset}: expected one hierarchy index, got {index_paths}")
    input_paths = {
        "prediction": spec["prediction"],
        "sequence": data_dir / "user_sequence.txt",
        "neighbors": data_dir / "similar_item_sasrec.txt",
        "item_index": index_paths[0],
        "s0_summary": ROOT / f"artifacts/phase3/s0/{dataset}/validation/summary.json",
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    _, text_to_item = decode_item_ids(input_paths["item_index"], "t5-small", True)
    prediction_rows, prediction_audit = read_predictions(
        input_paths["prediction"], text_to_item
    )
    sequences = read_sequences(input_paths["sequence"])
    neighbors = read_neighbors(input_paths["neighbors"])
    heads = head_items(training_popularity(sequences))
    samples = []
    mismatches = []
    for row in prediction_rows:
        history, expected_target = split_sample(sequences[row["user"]], "validation")
        if expected_target != row["gold"]:
            mismatches.append((row["user"], row["gold"], expected_target))
        union = set()
        for anchor in history:
            union.update(neighbors.get(anchor, ())[:K])
        samples.append(
            {
                **row,
                "history": history,
                "pop_group": "head" if expected_target in heads else "tail",
                "coverage_group": "covered" if expected_target in union else "uncovered",
            }
        )
    if mismatches:
        raise ValueError(f"{dataset}: target lineage mismatch: {mismatches[:3]}")
    baseline_pairs = [(sample["pred_items"], sample["gold"]) for sample in samples]
    baseline = summarize(baseline_pairs)
    s0_summary = json.loads(input_paths["s0_summary"].read_text())
    for key in ("recall@5", "recall@10", "recall@50", "ndcg@5", "ndcg@10", "ndcg@50"):
        if abs(baseline[key] - s0_summary["baseline"][key]) > 1e-12:
            raise ValueError(f"{dataset}: baseline drift for {key}")
    return {
        "samples": samples,
        "neighbors": neighbors,
        "baseline": baseline,
        "prediction_audit": prediction_audit,
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
    }


def group_metrics(samples: Sequence[dict], rankings: Mapping[str, Sequence[str]]) -> dict:
    result = {}
    groups = {
        "overall": lambda sample: "overall",
        "popularity": lambda sample: sample["pop_group"],
        "coverage": lambda sample: sample["coverage_group"],
    }
    for group_type, group_fn in groups.items():
        buckets = defaultdict(list)
        for sample in samples:
            buckets[group_fn(sample)].append(
                (rankings[sample["user"]], sample["gold"])
            )
        for group, pairs in buckets.items():
            result[(group_type, group)] = summarize(pairs)
    return result


def relative(after: float, before: float) -> float:
    return after / before - 1.0 if before else 0.0


def evaluate_config(dataset_data: dict, beta: float, lam: float, tau: float, min_support: int) -> dict:
    samples = dataset_data["samples"]
    neighbors = dataset_data["neighbors"]
    rankings = {}
    active_count = 0
    for sample in samples:
        relation_scores, supports, top1, gap, max_support = confidence_features(
            sample["history"], sample["pred_items"], neighbors, beta
        )
        active = abstention_active(top1, gap, max_support, tau)
        active_count += int(active)
        rankings[sample["user"]] = rerank(
            sample["pred_items"],
            sample["scores"],
            relation_scores,
            supports,
            active,
            lam,
            min_support,
        )
    candidate_groups = group_metrics(samples, rankings)
    baseline_rankings = {sample["user"]: sample["pred_items"] for sample in samples}
    baseline_groups = group_metrics(samples, baseline_rankings)
    overall = candidate_groups[("overall", "overall")]
    baseline = baseline_groups[("overall", "overall")]
    tail = candidate_groups[("popularity", "tail")]
    baseline_tail = baseline_groups[("popularity", "tail")]
    uncovered = candidate_groups[("coverage", "uncovered")]
    baseline_uncovered = baseline_groups[("coverage", "uncovered")]
    row = {
        "config_id": config_id(beta, lam, tau, min_support),
        "beta": beta,
        "lambda": lam,
        "tau": tau,
        "min_support": min_support,
        "active_rate": active_count / len(samples),
        **overall,
        "ndcg@10_relative_delta": relative(overall["ndcg@10"], baseline["ndcg@10"]),
        "recall@10_absolute_delta": overall["recall@10"] - baseline["recall@10"],
        "tail_ndcg@10_relative_delta": relative(tail["ndcg@10"], baseline_tail["ndcg@10"]),
        "uncovered_recall@10_relative_delta": relative(
            uncovered["recall@10"], baseline_uncovered["recall@10"]
        ),
        "uncovered_ndcg@10_relative_delta": relative(
            uncovered["ndcg@10"], baseline_uncovered["ndcg@10"]
        ),
    }
    row["dataset_pass"] = (
        row["ndcg@10_relative_delta"] >= 0.01
        and row["recall@10_absolute_delta"] >= -0.005
        and row["uncovered_recall@10_relative_delta"] >= -0.01
        and row["uncovered_ndcg@10_relative_delta"] >= -0.01
        and row["tail_ndcg@10_relative_delta"] >= 0.0
        and 0.05 <= row["active_rate"] <= 0.60
    )
    return row


def selection_key(row: dict) -> tuple:
    return (
        row["macro_ndcg@10_relative_delta"],
        -row["lambda"],
        row["tau"],
        row["min_support"],
        row["config_id"],
    )


def write_report(summary: dict, grid_by_dataset: Mapping[str, List[dict]]) -> None:
    selected = summary["selected_or_diagnostic_config"]
    lines = [
        "# GRAM 第三阶段 S0b 可靠性拒绝探针报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-22",
        "- Verification Status: ANALYZED",
        "- Version Label: s0b_posthoc_v1",
        "- Design Status: POST_HOC EXPLORATORY AMENDMENT",
        "",
        "## 1. 设计边界",
        "",
        "本探针在 S0 结果后提出，只使用锁定的 Beauty/Toys validation 预测和推理时可得的关系置信特征。"
        "没有读取 test、没有训练模型、没有使用目标商品构造 abstention。网格在运行前固定为 16 个共同配置。",
        "",
        "## 2. 共同配置结果",
        "",
        f"整体决定：**{summary['decision']}**。通过全部跨数据集门槛的配置数："
        f"{summary['passing_config_count']} / 16。",
        "",
        f"{'选中' if summary['decision'] == 'GO' else '诊断最优'}配置："
        f"`{selected['config_id']}`（beta={selected['beta']}、lambda={selected['lambda']}、"
        f"tau={selected['tau']}、min_support={selected['min_support']}）。",
        "",
        "| 数据集 | Active rate | NDCG@10 相对变化 | Recall@10 绝对变化 | Tail NDCG@10 | Uncovered Recall@10 | Uncovered NDCG@10 | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ("Toys", "Beauty"):
        row = next(
            item for item in grid_by_dataset[dataset]
            if item["config_id"] == selected["config_id"]
        )
        lines.append(
            f"| {dataset} | {row['active_rate']:.3%} | {row['ndcg@10_relative_delta']:+.3%} | "
            f"{row['recall@10_absolute_delta']:+.6f} | {row['tail_ndcg@10_relative_delta']:+.3%} | "
            f"{row['uncovered_recall@10_relative_delta']:+.3%} | "
            f"{row['uncovered_ndcg@10_relative_delta']:+.3%} | {row['dataset_pass']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 晋级解释",
            "",
        ]
    )
    if summary["decision"] == "GO":
        lines.append(
            "同一配置通过两数据集全部门槛，允许进入 S1 实现正确性 smoke；这仍不是论文效果证据。"
        )
    else:
        lines.append(
            "没有共同配置通过全部门槛。按修订计划停止 UCRF-v1 offline path，不得扩大网格或直接启动 S1。"
            "下一步需重新预注册 learned-gate 周期，或转向优先级 2。"
        )
    lines.extend(
        [
            "",
            "## 4. 产物",
            "",
            "- `artifacts/phase3/s0b/grid_metrics.csv`",
            "- `artifacts/phase3/s0b/joint_configs.csv`",
            "- `artifacts/phase3/s0b/summary.json`",
            "- `artifacts/phase3/experiment_registry.csv`",
            "- `artifacts/phase3/promotion_decisions.md`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


def update_promotion(summary: dict) -> None:
    selected = summary["selected_or_diagnostic_config"]
    addition = (
        "\n## S0b → S1\n\n"
        f"整体决定：**{summary['decision']}**。这是 post-hoc exploratory amendment；"
        f"16 个锁定共同配置中有 {summary['passing_config_count']} 个通过。\n\n"
        f"{'选中' if summary['decision'] == 'GO' else '诊断最优'}配置："
        f"`{selected['config_id']}`；macro NDCG@10 relative delta="
        f"{selected['macro_ndcg@10_relative_delta']:+.6%}。\n"
    )
    current = PROMOTION_PATH.read_text() if PROMOTION_PATH.exists() else "# GRAM 第三阶段晋级记录\n"
    if "\n## S0b → S1\n" in current:
        current = current.split("\n## S0b → S1\n", 1)[0].rstrip() + "\n"
    PROMOTION_PATH.write_text(current.rstrip() + "\n" + addition)


def write_registry(grid_by_dataset: Mapping[str, List[dict]], input_hashes: Mapping[str, dict]) -> None:
    rows = []
    code_hash = sha256(Path(__file__))
    for dataset in ("Toys", "Beauty"):
        for row in grid_by_dataset[dataset]:
            rows.append(
                {
                    "hypothesis": "H2_reliability_abstention",
                    "config_id": f"S0b_{row['config_id']}",
                    "code_sha256": code_hash,
                    "dataset": dataset,
                    "split_hash": input_hashes[dataset]["prediction"],
                    "seed": 2023,
                    "gpu_budget_hours": 0,
                    "status": "ANALYZED",
                    "validation_ndcg@10": row["ndcg@10"],
                    "validation_ndcg@10_relative_delta": row["ndcg@10_relative_delta"],
                    "active_rate": row["active_rate"],
                    "dataset_pass": row["dataset_pass"],
                    "promotion_decision": "candidate" if row["dataset_pass"] else "reject",
                }
            )
    write_csv(REGISTRY_PATH, rows, tuple(rows[0].keys()))


def main() -> int:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared = {dataset: prepare_dataset(dataset) for dataset in ("Toys", "Beauty")}
    grid_by_dataset: Dict[str, List[dict]] = {"Toys": [], "Beauty": []}
    for beta in BETAS:
        for lam in LAMBDAS:
            for tau in THRESHOLDS:
                for min_support in MIN_SUPPORTS:
                    for dataset in ("Toys", "Beauty"):
                        row = evaluate_config(
                            prepared[dataset], beta, lam, tau, min_support
                        )
                        row["dataset"] = dataset
                        grid_by_dataset[dataset].append(row)

    flat_rows = [row for dataset in ("Toys", "Beauty") for row in grid_by_dataset[dataset]]
    write_csv(OUTPUT_DIR / "grid_metrics.csv", flat_rows, tuple(flat_rows[0].keys()))
    joint_rows = []
    for cid in sorted({row["config_id"] for row in flat_rows}):
        per_dataset = {
            dataset: next(row for row in grid_by_dataset[dataset] if row["config_id"] == cid)
            for dataset in ("Toys", "Beauty")
        }
        base = per_dataset["Toys"]
        joint_rows.append(
            {
                "config_id": cid,
                "beta": base["beta"],
                "lambda": base["lambda"],
                "tau": base["tau"],
                "min_support": base["min_support"],
                "macro_ndcg@10_relative_delta": sum(
                    row["ndcg@10_relative_delta"] for row in per_dataset.values()
                ) / 2.0,
                "toys_pass": per_dataset["Toys"]["dataset_pass"],
                "beauty_pass": per_dataset["Beauty"]["dataset_pass"],
                "joint_pass": all(row["dataset_pass"] for row in per_dataset.values()),
            }
        )
    write_csv(OUTPUT_DIR / "joint_configs.csv", joint_rows, tuple(joint_rows[0].keys()))
    passing = [row for row in joint_rows if row["joint_pass"]]
    selected = max(passing or joint_rows, key=selection_key)
    decision = "GO" if passing else "STOP"
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "s0b_posthoc_v1",
            "design_status": "POST_HOC_EXPLORATORY_AMENDMENT",
        },
        "decision": decision,
        "passing_config_count": len(passing),
        "selected_or_diagnostic_config": selected,
        "locked_grid_size": len(joint_rows),
        "constraints": {
            "both_ndcg@10_relative_delta_min": 0.01,
            "both_recall@10_absolute_delta_min": -0.005,
            "both_uncovered_relative_delta_min": -0.01,
            "both_tail_ndcg@10_relative_delta_min": 0.0,
            "active_rate_range": [0.05, 0.60],
        },
        "input_sha256": {
            dataset: prepared[dataset]["input_sha256"] for dataset in prepared
        },
        "audit": {
            dataset: {
                "rows": len(prepared[dataset]["samples"]),
                "unknown_gold_count": prepared[dataset]["prediction_audit"]["unknown_gold_count"],
                "unknown_prediction_count": prepared[dataset]["prediction_audit"]["unknown_prediction_count"],
            }
            for dataset in prepared
        },
        "wall_time_seconds": time.time() - started,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    write_registry(
        grid_by_dataset,
        {dataset: prepared[dataset]["input_sha256"] for dataset in prepared},
    )
    update_promotion(summary)
    write_report(summary, grid_by_dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
