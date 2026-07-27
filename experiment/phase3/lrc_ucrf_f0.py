#!/usr/bin/env python3
"""LRC-UCRF F0: CPU-only learnability and calibration feasibility probe."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s0_offline_diagnostics import read_neighbors, read_sequences, sha256


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "artifacts/phase3/lrc_ucrf_f0"
REPORT = ROOT / "report/第三阶段/GRAM_第三阶段_LRC-UCRF_F0可学习性报告.md"
PROMOTION = ROOT / "artifacts/phase3/promotion_decisions.md"
DATA_ROOT = ROOT / "GRAM/rec_datasets"

K = 20
M = 20
MAX_HISTORY = 20
RECENCY_DECAY = 0.90
BETA = 0.25
ACTIVE_TARGETS = (0.10, 0.20, 0.30, 0.40)
SEED = 2023

FEATURE_NAMES = (
    "history_length",
    "union_size",
    "pool_per_history",
    "anchor_overlap_ratio",
    "top1_score",
    "top2_score",
    "top1_top2_gap",
    "top20_score_mean",
    "top20_score_std",
    "top20_score_min",
    "max_support",
    "top20_support_mean",
    "multi_support_fraction",
    "latest_anchor_agreement",
    "score_entropy",
)


def stable_user_is_calibration(user: str) -> bool:
    return int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 5 == 0


def retrieval_features(
    history: Sequence[str], neighbors: Mapping[str, Sequence[str]]
) -> Tuple[np.ndarray, List[str]]:
    """Return inference-only features and ranked items; target is not accepted."""
    history = list(history)[-MAX_HISTORY:]
    support: Dict[str, int] = {}
    best: Dict[str, float] = {}
    for age, anchor in enumerate(reversed(history)):
        for rank, candidate in enumerate(neighbors.get(anchor, ())[:K], start=1):
            support[candidate] = support.get(candidate, 0) + 1
            score = (RECENCY_DECAY**age) / math.log2(rank + 1.0)
            best[candidate] = max(best.get(candidate, 0.0), score)
    denom = max(1, len(history))
    scored = [
        (best[item] + BETA * support[item] / denom, support[item], item)
        for item in best
    ]
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    top = scored[:M]
    top_scores = np.array([row[0] for row in top], dtype=np.float64)
    top_supports = np.array([row[1] for row in top], dtype=np.float64)
    if len(top_scores) == 0:
        top_scores = np.zeros(1, dtype=np.float64)
        top_supports = np.zeros(1, dtype=np.float64)
    top1 = float(top_scores[0])
    top2 = float(top_scores[1]) if len(top_scores) > 1 else 0.0
    shifted = top_scores - top_scores.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    entropy /= math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    latest = set(neighbors.get(history[-1], ())[:K]) if history else set()
    top_items = [row[2] for row in top]
    theoretical_slots = max(1, len(history) * K)
    features = np.array(
        [
            len(history),
            len(scored),
            len(scored) / denom,
            1.0 - len(scored) / theoretical_slots,
            top1,
            top2,
            top1 - top2,
            float(top_scores.mean()),
            float(top_scores.std()),
            float(top_scores.min()),
            float(top_supports.max()),
            float(top_supports.mean()),
            float((top_supports >= 2).mean()),
            sum(item in latest for item in top_items) / max(1, len(top_items)),
            entropy,
        ],
        dtype=np.float64,
    )
    return features, top_items


def build_dataset(dataset: str) -> dict:
    data_dir = DATA_ROOT / dataset
    sequence_path = data_dir / "user_sequence.txt"
    neighbor_path = data_dir / "similar_item_sasrec.txt"
    sequences = read_sequences(sequence_path)
    neighbors = read_neighbors(neighbor_path)
    train_x, train_y, train_users = [], [], []
    cal_x, cal_y, cal_users = [], [], []
    val_x, val_y, val_users = [], [], []
    for user, sequence in sequences.items():
        if len(sequence) < 3:
            continue
        pre_history = sequence[:-3][-MAX_HISTORY:]
        pre_target = sequence[-3]
        features, retrieved = retrieval_features(pre_history, neighbors)
        label = int(pre_target in retrieved)
        target_x, target_y, target_users = (
            (cal_x, cal_y, cal_users)
            if stable_user_is_calibration(user)
            else (train_x, train_y, train_users)
        )
        target_x.append(features)
        target_y.append(label)
        target_users.append(user)

        validation_history = sequence[:-2][-MAX_HISTORY:]
        validation_target = sequence[-2]
        features, retrieved = retrieval_features(validation_history, neighbors)
        val_x.append(features)
        val_y.append(int(validation_target in retrieved))
        val_users.append(user)

    arrays = {
        "train_x": np.vstack(train_x),
        "train_y": np.asarray(train_y, dtype=np.int64),
        "cal_x": np.vstack(cal_x),
        "cal_y": np.asarray(cal_y, dtype=np.int64),
        "val_x": np.vstack(val_x),
        "val_y": np.asarray(val_y, dtype=np.int64),
    }
    dataset_dir = OUTPUT_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train_users.txt").write_text("\n".join(train_users) + "\n")
    (dataset_dir / "calibration_users.txt").write_text("\n".join(cal_users) + "\n")
    return {
        **arrays,
        "train_users": train_users,
        "cal_users": cal_users,
        "val_users": val_users,
        "input_sha256": {
            "sequence": sha256(sequence_path),
            "neighbors": sha256(neighbor_path),
            "train_users": sha256(dataset_dir / "train_users.txt"),
            "calibration_users": sha256(dataset_dir / "calibration_users.txt"),
        },
    }


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(ece)


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prevalence = float(y.mean())
    return {
        "n": len(y),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "auprc_lift": float(average_precision_score(y, probability) / prevalence),
        "brier": float(brier_score_loss(y, probability)),
        "ece": expected_calibration_error(y, probability),
    }


def fixed_models() -> dict:
    return {
        "C1_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED,
            ),
        ),
        "C2_hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=SEED,
        ),
    }


def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    active = probability >= threshold
    true_positive = int(((y == 1) & active).sum())
    positives = int((y == 1).sum())
    precision = true_positive / int(active.sum()) if active.any() else 0.0
    recall = true_positive / positives if positives else 0.0
    prevalence = float(y.mean())
    return {
        "threshold": float(threshold),
        "active_rate": float(active.mean()),
        "precision": precision,
        "precision_lift": precision / prevalence if prevalence else 0.0,
        "positive_recall": recall,
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> Tuple[dict, List[dict]]:
    candidates = []
    for target_active in ACTIVE_TARGETS:
        threshold = float(np.quantile(probability, 1.0 - target_active))
        row = threshold_metrics(y, probability, threshold)
        row["target_active_rate"] = target_active
        candidates.append(row)
    eligible = [row for row in candidates if row["positive_recall"] >= 0.25]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda row: (
            row["precision_lift"],
            -row["active_rate"],
            row["positive_recall"],
        ),
    )
    return selected, candidates


def run_dataset(dataset: str) -> Tuple[dict, List[dict]]:
    data = build_dataset(dataset)
    train_x, train_y = data["train_x"], data["train_y"]
    cal_x, cal_y = data["cal_x"], data["cal_y"]
    val_x, val_y = data["val_x"], data["val_y"]
    metric_rows = []
    fitted = {}
    train_prevalence = float(train_y.mean())
    for model_name, model in fixed_models().items():
        model.fit(train_x, train_y)
        raw_cal = model.predict_proba(cal_x)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal, cal_y)
        cal_probability = calibrator.transform(raw_cal)
        val_probability = calibrator.transform(model.predict_proba(val_x)[:, 1])
        cal_metrics = classification_metrics(cal_y, cal_probability)
        val_metrics = classification_metrics(val_y, val_probability)
        metric_rows.extend(
            {"dataset": dataset, "model": model_name, "split": split, **metrics}
            for split, metrics in (("calibration", cal_metrics), ("validation", val_metrics))
        )
        fitted[model_name] = {
            "cal_probability": cal_probability,
            "val_probability": val_probability,
            "cal_metrics": cal_metrics,
            "val_metrics": val_metrics,
        }

    selected_name = min(
        fitted,
        key=lambda name: (
            fitted[name]["cal_metrics"]["brier"],
            0 if name == "C1_logistic" else 1,
        ),
    )
    selected_model = fitted[selected_name]
    selected_threshold, threshold_candidates = choose_threshold(
        cal_y, selected_model["cal_probability"]
    )
    validation_threshold = threshold_metrics(
        val_y, selected_model["val_probability"], selected_threshold["threshold"]
    )
    val_metrics = selected_model["val_metrics"]
    constant_probability = np.full(len(val_y), train_prevalence)
    constant_brier = float(brier_score_loss(val_y, constant_probability))
    brier_improvement = 1.0 - val_metrics["brier"] / constant_brier
    passed = (
        val_metrics["auroc"] >= 0.60
        and val_metrics["auprc_lift"] >= 1.50
        and brier_improvement >= 0.05
        and val_metrics["ece"] <= 0.05
        and 0.10 <= validation_threshold["active_rate"] <= 0.40
        and validation_threshold["precision_lift"] >= 1.50
        and validation_threshold["positive_recall"] >= 0.25
    )
    summary = {
        "dataset": dataset,
        "selected_model": selected_name,
        "train_n": len(train_y),
        "calibration_n": len(cal_y),
        "validation_n": len(val_y),
        "train_prevalence": train_prevalence,
        "validation_metrics": val_metrics,
        "constant_brier": constant_brier,
        "brier_relative_improvement": brier_improvement,
        "calibration_threshold_selection": selected_threshold,
        "threshold_candidates": threshold_candidates,
        "validation_threshold_metrics": validation_threshold,
        "passed": passed,
        "input_sha256": data["input_sha256"],
    }
    dataset_dir = OUTPUT_ROOT / dataset
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary, metric_rows


def write_outputs(dataset_summaries: Mapping[str, dict], metric_rows: List[dict], started: float) -> dict:
    fieldnames = tuple(metric_rows[0].keys())
    with (OUTPUT_ROOT / "model_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    feature_schema = {
        "feature_names": FEATURE_NAMES,
        "k": K,
        "m": M,
        "max_history": MAX_HISTORY,
        "recency_decay": RECENCY_DECAY,
        "beta": BETA,
        "target_in_features": False,
        "test_item_used": False,
    }
    (OUTPUT_ROOT / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2) + "\n"
    )
    decision = "GO" if all(row["passed"] for row in dataset_summaries.values()) else "STOP"
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "lrc_ucrf_f0_v1",
            "design_status": "NEW_PREREGISTERED_CYCLE",
        },
        "decision": decision,
        "datasets": dataset_summaries,
        "wall_time_seconds": time.time() - started,
        "feature_schema_sha256": sha256(OUTPUT_ROOT / "feature_schema.json"),
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def write_report(summary: dict) -> None:
    lines = [
        "# GRAM 第三阶段 LRC-UCRF F0 可学习性报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-22",
        "- Verification Status: ANALYZED",
        "- Version Label: lrc_ucrf_f0_v1",
        "- Design Status: NEW PREREGISTERED CYCLE",
        "",
        "## 1. 结论",
        "",
        f"LRC-F0 整体决定：**{summary['decision']}**。本实验只检验 coverage reliability 是否可学习，"
        "不构成推荐效果结论。",
        "",
        "| 数据集 | 模型 | Prevalence | AUROC | AUPRC lift | Brier 改善 | ECE | Active rate | Precision lift | Positive recall | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ("Toys", "Beauty"):
        row = summary["datasets"][dataset]
        metrics = row["validation_metrics"]
        threshold = row["validation_threshold_metrics"]
        lines.append(
            f"| {dataset} | {row['selected_model']} | {metrics['prevalence']:.3%} | "
            f"{metrics['auroc']:.4f} | {metrics['auprc_lift']:.3f}× | "
            f"{row['brier_relative_improvement']:+.3%} | {metrics['ece']:.4f} | "
            f"{threshold['active_rate']:.3%} | {threshold['precision_lift']:.3f}× | "
            f"{threshold['positive_recall']:.3%} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 2. 数据与泄漏边界",
            "",
            "训练/校准标签来自倒数第三次交互，validation 标签来自倒数第二次交互；最后一次 test 商品未使用。"
            "特征函数不接收 target，只读取历史与 SASRec top-20 邻居。用户哈希确定性划分 80%/20%。",
            "",
            "## 3. 晋级规则",
            "",
        ]
    )
    if summary["decision"] == "GO":
        lines.append("Beauty/Toys 均通过全部必要条件，允许设计独立的 LRC-S1 smoke。")
    else:
        lines.append("至少一个数据集未通过必要条件；不得实现或启动 LRC-S1，按计划转向方向 B。")
    lines.extend(
        [
            "",
            "## 4. 产物",
            "",
            "- `artifacts/phase3/lrc_ucrf_f0/summary.json`",
            "- `artifacts/phase3/lrc_ucrf_f0/model_metrics.csv`",
            "- `artifacts/phase3/lrc_ucrf_f0/feature_schema.json`",
            "- `artifacts/phase3/lrc_ucrf_f0/{Toys,Beauty}/dataset_summary.json`",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


def update_promotion(summary: dict) -> None:
    addition = (
        "\n## LRC-F0 → LRC-S1\n\n"
        f"整体决定：**{summary['decision']}**。这是 UCRF-v1 STOP 后建立的独立预注册周期。\n\n"
        + "\n".join(
            f"- {dataset}: pass={row['passed']}, model={row['selected_model']}, "
            f"AUROC={row['validation_metrics']['auroc']:.6f}, "
            f"AUPRC lift={row['validation_metrics']['auprc_lift']:.6f}, "
            f"Brier improvement={row['brier_relative_improvement']:.6f}."
            for dataset, row in summary["datasets"].items()
        )
        + "\n"
    )
    current = PROMOTION.read_text()
    if "\n## LRC-F0 → LRC-S1\n" in current:
        current = current.split("\n## LRC-F0 → LRC-S1\n", 1)[0].rstrip() + "\n"
    PROMOTION.write_text(current.rstrip() + "\n" + addition)


def main() -> int:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_summaries = {}
    metric_rows = []
    for dataset in ("Toys", "Beauty"):
        dataset_summary, rows = run_dataset(dataset)
        dataset_summaries[dataset] = dataset_summary
        metric_rows.extend(rows)
    summary = write_outputs(dataset_summaries, metric_rows, started)
    write_report(summary)
    update_promotion(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
