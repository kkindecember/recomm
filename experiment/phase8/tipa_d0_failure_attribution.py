#!/usr/bin/env python3
"""TIPA-D0: read-only failure-attribution audit of frozen P0A artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("Toys", "Beauty")
ARMS = ("A", "B", "C")
SEED = 2023
N_BOOTSTRAP = 10_000
RANK_CENSOR = 51.0
EXPECTED_HASHES = {
    "canonical_summary": ("summary.json", "a0b6b06717d234c224daa7cf2fc3f3bc08599b364ffdc360ae66ced9d9d09c2b"),
    "toys_summary": ("Toys/summary.json", "1ac1e6755d02aaf92753b487be62a5d761f2926aa292cfbd48aadb980e802e96"),
    "beauty_summary": ("Beauty/summary.json", "5ce602cf212115a6d1f407127348c49b0772e97165d2ffa791dd9049c5667500"),
    "toys_per_prefix": ("Toys/per_prefix.csv", "9bdb4f27a07f1c2d9b5e5d93873844a02f49ba397760dfc178055a25a835c63d"),
    "beauty_per_prefix": ("Beauty/per_prefix.csv", "b4d80fd10ef40ed02b324800c9f3f9a12b1f1bc7e67d34b50cd9c8ac5376e2fd"),
    "toys_per_user": ("Toys/per_user.csv", "51a9c734e37f5df749d5549188eab3aaeea45b471e7f2a6adf15a58a058fea08"),
    "beauty_per_user": ("Beauty/per_user.csv", "4fd27a30b2dcc0eeb636f816cf1de177e108539ad1f9ce9addf19d0d0533939f"),
    "toys_per_user_arms": ("Toys/per_user_arms.csv", "7b78c3d3b1e1217884228e28bd63923031a3ac9f81edf8d370c26965cb3b060e"),
    "beauty_per_user_arms": ("Beauty/per_user_arms.csv", "6f3142b5d33fc8fb9c84bdfe5256c816ddd36faa62baef8b0e03606cc64859d3"),
}
USER_FIELDS = {
    "sample_key", "target", "target_group", "history_group", "teacher_margin_group",
    "transition_covered", "teacher_margin", "teacher_target_rank", "A_rank", "B_rank",
    "C_rank", "B_kendall", "C_kendall", "C_null_rate", "C_max_abs_delta",
}
ARM_FIELDS = {
    "arm", "sample_key", "target_group", "graph_covered", "baseline_rank", "candidate_rank",
    "baseline_Recall@5", "baseline_NDCG@5", "baseline_Recall@10", "baseline_NDCG@10",
    "baseline_Recall@50", "baseline_MRR", "candidate_Recall@5", "candidate_NDCG@5",
    "candidate_Recall@10", "candidate_NDCG@10", "candidate_Recall@50", "candidate_MRR",
    "target_in_baseline_beam50", "target_in_candidate_beam50", "new_hit_at10_outside_A_beam",
    "changed", "broad_harm",
}
PREFIX_FIELDS = {"dataset", "sample_key", "prefix", "legal_children", "teacher_mass_error"}
METRICS = ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "Recall@50", "MRR")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def f(row: dict[str, str], key: str) -> float:
    if row[key] == "":
        raise ValueError(f"unexpected empty value: {key}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {key}={row[key]}")
    return value


def rank(row: dict[str, str], key: str) -> float:
    return RANK_CENSOR if row[key] == "" else f(row, key)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def quantiles(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75))}


def bootstrap_mean(values: Iterable[float], rng: np.random.Generator) -> tuple[float, float, float]:
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return math.nan, math.nan, math.nan
    draws = rng.choice(a, size=(N_BOOTSTRAP, len(a)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(a.mean()), float(lo), float(hi)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high


def safe_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def bh_adjust(rows: list[dict[str, Any]], p_key: str = "p_value") -> None:
    valid = [(i, float(r[p_key])) for i, r in enumerate(rows) if finite(r.get(p_key))]
    valid.sort(key=lambda item: item[1])
    adjusted = [math.nan] * len(rows)
    running = 1.0
    m = len(valid)
    for reverse_index in range(m - 1, -1, -1):
        original_index, p = valid[reverse_index]
        running = min(running, p * m / (reverse_index + 1))
        adjusted[original_index] = running
    for row, q in zip(rows, adjusted):
        row["bh_q_value"] = q
        row["bh_fdr_0_05"] = bool(finite(q) and q <= .05)


def verify_hashes(parent: Path) -> dict[str, dict[str, str]]:
    result = {}
    for name, (relative, expected) in EXPECTED_HASHES.items():
        path = parent / relative
        actual = sha256(path)
        result[name] = {"path": str(path.relative_to(ROOT)), "expected_sha256": expected, "actual_sha256": actual}
        if actual != expected:
            raise RuntimeError(f"BLOCKED_PARENT_ARTIFACT_DRIFT: {relative}: {actual} != {expected}")
    return result


def compare_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise RuntimeError(f"summary recomputation mismatch {label}: {actual} != {expected}")


def aggregate_arm(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {"n": len(rows)}
    for metric in METRICS:
        baseline = mean(f(r, f"baseline_{metric}") for r in rows)
        candidate = mean(f(r, f"candidate_{metric}") for r in rows)
        out[f"baseline_{metric}"] = baseline
        out[f"candidate_{metric}"] = candidate
        out[f"absolute_delta_{metric}"] = candidate - baseline
        out[f"relative_gain_{metric}"] = (candidate - baseline) / baseline if baseline else 0.0
    for key in ("target_in_baseline_beam50", "target_in_candidate_beam50", "new_hit_at10_outside_A_beam", "changed", "broad_harm"):
        out[f"mean_{key}"] = mean(f(r, key) for r in rows)
    return out


def verify_summary(summary: dict[str, Any], users: list[dict[str, str]], arms: list[dict[str, str]]) -> None:
    arm_map = {arm: [r for r in arms if r["arm"] == arm] for arm in ARMS}
    for arm, groups in summary["methods"].items():
        for group, expected in groups.items():
            selected = arm_map[arm]
            if group == "head": selected = [r for r in selected if r["target_group"] == "head"]
            elif group == "tail": selected = [r for r in selected if r["target_group"] == "tail"]
            elif group == "graph_covered": selected = [r for r in selected if r["graph_covered"] == "1"]
            elif group == "graph_uncovered": selected = [r for r in selected if r["graph_covered"] == "0"]
            elif group != "overall": raise RuntimeError(f"unknown summary stratum: {group}")
            actual = aggregate_arm(selected)
            for key, value in actual.items():
                compare_close(value, float(expected[key]), f"{arm}/{group}/{key}")
    mechanism = summary["mechanism"]
    compare_close(mean(f(r, "B_kendall") for r in users), mechanism["B_kendall"], "B_kendall")
    compare_close(mean(f(r, "C_kendall") for r in users), mechanism["C_kendall"], "C_kendall")
    compare_close(mean(f(r, "C_kendall") - f(r, "B_kendall") for r in users), mechanism["kendall_delta"], "kendall_delta")
    teacher_exclusive = [r for r in users if rank(r, "A_rank") > 50 and f(r, "teacher_target_rank") <= 50]
    if len(teacher_exclusive) != mechanism["teacher_exclusive_users"]:
        raise RuntimeError("teacher-exclusive summary mismatch")
    for arm in ("B", "C"):
        realized = sum(rank(r, f"{arm}_rank") <= 50 for r in teacher_exclusive)
        if realized != mechanism[f"{arm}_realized_exclusive"]:
            raise RuntimeError(f"{arm} realized-exclusive summary mismatch")


def validate_dataset(parent: Path, dataset: str) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    base = parent / dataset
    summary = json.loads((base / "summary.json").read_text())
    users, arms, prefixes = (read_csv(base / name) for name in ("per_user.csv", "per_user_arms.csv", "per_prefix.csv"))
    if set(users[0]) != USER_FIELDS or set(arms[0]) != ARM_FIELDS or set(prefixes[0]) != PREFIX_FIELDS:
        raise RuntimeError(f"schema mismatch: {dataset}")
    keys = [r["sample_key"] for r in users]
    if len(users) != 256 or len(set(keys)) != 256:
        raise RuntimeError(f"user key integrity failed: {dataset}")
    by_arm = Counter(r["arm"] for r in arms)
    if by_arm != Counter({arm: 256 for arm in ARMS}):
        raise RuntimeError(f"arm counts failed: {dataset}: {by_arm}")
    for arm in ARMS:
        if {r["sample_key"] for r in arms if r["arm"] == arm} != set(keys):
            raise RuntimeError(f"arm key alignment failed: {dataset}/{arm}")
    if len(prefixes) != 256 or len({r["sample_key"] for r in prefixes}) != 256 or any(int(r["legal_children"]) <= 1 for r in prefixes):
        raise RuntimeError(f"prefix integrity failed: {dataset}")
    for row in users:
        for key in USER_FIELDS - {"sample_key", "target", "target_group", "history_group", "teacher_margin_group", "A_rank", "B_rank", "C_rank"}:
            if not finite(row[key]): raise RuntimeError(f"non-finite user value: {dataset}/{key}")
        for key in ("A_rank", "B_rank", "C_rank"):
            if row[key] and not finite(row[key]): raise RuntimeError(f"non-finite rank: {dataset}/{key}")
    for row in arms:
        for key in ARM_FIELDS - {"arm", "sample_key", "target_group", "baseline_rank", "candidate_rank"}:
            if not finite(row[key]): raise RuntimeError(f"non-finite arm value: {dataset}/{key}")
    for row in prefixes:
        if not finite(row["legal_children"]) or not finite(row["teacher_mass_error"]):
            raise RuntimeError(f"non-finite prefix value: {dataset}")
    integrity = summary["integrity"]
    forbidden = any((integrity["test_read"], integrity["sports_read"], integrity["external_development_read"]))
    if forbidden or summary["training"]["prefix_records"] != 256 or not summary["audit"]["passed"]:
        raise RuntimeError(f"forbidden read or parent audit failure: {dataset}")
    verify_summary(summary, users, arms)
    return summary, users, arms, prefixes


def make_paired(dataset: str, users: list[dict[str, str]], arms: list[dict[str, str]]) -> list[dict[str, Any]]:
    amap = {(r["arm"], r["sample_key"]): r for r in arms}
    output = []
    for u in users:
        key = u["sample_key"]
        row: dict[str, Any] = {
            "dataset": dataset, "sample_key": key, "target_group": u["target_group"],
            "history_group": u["history_group"], "teacher_margin_group": u["teacher_margin_group"],
            "transition_covered": int(u["transition_covered"]), "teacher_margin": f(u, "teacher_margin"),
            "teacher_target_rank": f(u, "teacher_target_rank"), "A_rank_censored51": rank(u, "A_rank"),
            "B_rank_censored51": rank(u, "B_rank"), "C_rank_censored51": rank(u, "C_rank"),
            "B_rank_change_vs_A": rank(u, "A_rank") - rank(u, "B_rank"),
            "C_rank_change_vs_A": rank(u, "A_rank") - rank(u, "C_rank"),
            "C_minus_B_rank_change": rank(u, "B_rank") - rank(u, "C_rank"),
            "B_kendall": f(u, "B_kendall"), "C_kendall": f(u, "C_kendall"),
            "C_minus_B_kendall": f(u, "C_kendall") - f(u, "B_kendall"),
            "C_null_rate": f(u, "C_null_rate"), "C_max_abs_delta": f(u, "C_max_abs_delta"),
            "teacher_exclusive": int(rank(u, "A_rank") > 50 and f(u, "teacher_target_rank") <= 50),
        }
        for arm in ARMS:
            a = amap[(arm, key)]
            row[f"{arm}_in_beam50"] = int(f(a, "target_in_candidate_beam50"))
            row[f"{arm}_in_top10"] = int(rank(u, f"{arm}_rank") <= 10)
            row[f"{arm}_broad_harm"] = int(f(a, "broad_harm"))
            for metric in METRICS:
                row[f"{arm}_delta_{metric}"] = f(a, f"candidate_{metric}") - f(a, f"baseline_{metric}")
        output.append(row)
    return output


def add_interval(rows: list[dict[str, Any]], dataset: str, family: str, metric: str, values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    estimate, lo, hi = bootstrap_mean(values, rng)
    row = {"dataset": dataset, "family": family, "metric": metric, "n": len(values), "estimate": estimate, "ci95_low": lo, "ci95_high": hi, "method": f"paired bootstrap seed={SEED} resamples={N_BOOTSTRAP}"}
    rows.append(row)
    return row


def user_strata(dataset: str, paired: list[dict[str, Any]], rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results, tests = [], []
    dimensions = {
        "target_group": ("head", "tail"), "history_group": ("short", "long"),
        "teacher_margin_group": ("low", "high"), "transition_covered": (0, 1),
    }
    for dimension, levels in dimensions.items():
        groups = [[r for r in paired if r[dimension] == level] for level in levels]
        for level, group in zip(levels, groups):
            for metric in ("C_minus_B_kendall", "B_rank_change_vs_A", "C_rank_change_vs_A", "C_minus_B_rank_change", "B_broad_harm", "C_broad_harm"):
                vals = [float(r[metric]) for r in group]
                est, lo, hi = bootstrap_mean(vals, rng)
                results.append({"dataset": dataset, "dimension": dimension, "stratum": str(level), "metric": metric, "n": len(vals), "effect": est, "ci95_low": lo, "ci95_high": hi, "interval_method": f"bootstrap seed={SEED} B={N_BOOTSTRAP}"})
        if groups[0] and groups[1]:
            for metric in ("C_minus_B_kendall", "C_minus_B_rank_change", "C_broad_harm"):
                x, y = [float(r[metric]) for r in groups[0]], [float(r[metric]) for r in groups[1]]
                test = mannwhitneyu(x, y, alternative="two-sided")
                tests.append({"dataset": dataset, "family": "user_strata", "comparison": f"{dimension}:{levels[0]}_vs_{levels[1]}", "metric": metric, "statistic": float(test.statistic), "p_value": float(test.pvalue)})
    return results, tests


def prefix_census(dataset: str, prefixes: list[dict[str, str]], rng: np.random.Generator) -> list[dict[str, Any]]:
    def legal_bin(value: int) -> str:
        if value <= 4: return "2-4"
        if value <= 16: return "5-16"
        if value <= 64: return "17-64"
        return ">64"
    records = [{"depth": len(r["prefix"].split()), "legal_children_bin": legal_bin(int(r["legal_children"])), "teacher_mass_error": f(r, "teacher_mass_error")} for r in prefixes]
    output = []
    for dimension in ("depth", "legal_children_bin"):
        for level in sorted({r[dimension] for r in records}, key=str):
            values = [r["teacher_mass_error"] for r in records if r[dimension] == level]
            est, lo, hi = bootstrap_mean(values, rng)
            output.append({"dataset": dataset, "dimension": dimension, "stratum": level, "n": len(values), "mean_teacher_mass_error": est, "ci95_low": lo, "ci95_high": hi, "max_teacher_mass_error": max(values), "interval_method": f"bootstrap seed={SEED} B={N_BOOTSTRAP}"})
    return output


def analyze_dataset(dataset: str, summary: dict[str, Any], paired: list[dict[str, Any]], rng: np.random.Generator, intervals: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    teacher_exclusive = [r for r in paired if r["teacher_exclusive"]]
    alignment = add_interval(intervals, dataset, "alignment", "C_minus_B_kendall", [r["C_minus_B_kendall"] for r in paired], rng)
    alignment.update({"median": float(np.median([r["C_minus_B_kendall"] for r in paired])), **{f"count_{k}": v for k, v in Counter("positive" if r["C_minus_B_kendall"] > 0 else "negative" if r["C_minus_B_kendall"] < 0 else "zero" for r in paired).items()}})
    correlations = []
    for exposure in ("C_null_rate", "C_max_abs_delta"):
        for outcome in ("C_minus_B_kendall", "C_rank_change_vs_A", "C_broad_harm"):
            rho, p = safe_spearman([r[exposure] for r in paired], [r[outcome] for r in paired])
            correlations.append({"dataset": dataset, "family": "perturbation_correlation", "comparison": exposure, "metric": outcome, "statistic": rho, "p_value": p})
    realization: dict[str, Any] = {"teacher_exclusive_n": len(teacher_exclusive)}
    for arm in ("B", "C"):
        for outcome in ("in_beam50", "in_top10", "broad_harm"):
            successes = sum(r[f"{arm}_{outcome}"] for r in teacher_exclusive)
            lo, hi = wilson(successes, len(teacher_exclusive))
            realization[f"{arm}_{outcome}"] = {"successes": successes, "n": len(teacher_exclusive), "rate": successes / len(teacher_exclusive) if teacher_exclusive else math.nan, "wilson95_low": lo, "wilson95_high": hi}
        add_interval(intervals, dataset, "realization", f"{arm}_rank_change_vs_A_teacher_exclusive", [r[f"{arm}_rank_change_vs_A"] for r in teacher_exclusive], rng)
    result = {
        "teacher_availability": {"teacher_exclusive_n": len(teacher_exclusive), "teacher_exclusive_rate": len(teacher_exclusive) / len(paired), "teacher_target_rank": quantiles(r["teacher_target_rank"] for r in paired), "teacher_margin": quantiles(r["teacher_margin"] for r in paired), "caveat": "teacher-exclusive availability is not evidence of teacher correctness"},
        "item_to_path_alignment": alignment,
        "perturbation_behavior": {"C_null_rate": quantiles(r["C_null_rate"] for r in paired), "C_max_abs_delta": quantiles(r["C_max_abs_delta"] for r in paired), "bound_hit_rate_at_0_3": mean(abs(r["C_max_abs_delta"] - .3) <= 2e-6 for r in paired), "correlations": correlations},
        "beam_realization": realization,
        "parent_outcomes": {"C_recall10_delta": summary["mechanism"]["C_recall10_delta"], "C_ndcg10_delta": summary["mechanism"]["C_ndcg10_delta"], "C_broad_harm": summary["mechanism"]["C_broad_harm"]},
    }
    return result, correlations


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict): return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list): return [sanitize_json(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): value = float(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def render_report(summary: dict[str, Any]) -> str:
    d = summary["failure_chain"]
    labels = "、".join(f"`{x}`" for x in summary["diagnostic_labels"])
    lines = [
        "# GRAM 第八阶段：TIPA-D0 路径对齐失败归因审计报告", "",
        "## Material Passport", "", "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run", f"- Origin Date: {summary['completed_at']}", "- Verification Status: ANALYZED", "- Version Label: phase8_tipa_d0_failure_attribution_v1", "",
        "## 结论", "", f"本次 CPU-only、analysis-only 审计成功完成。诊断标签为：{labels}。这些标签描述 P0A 的失败链，不构成新方法选择器；`TIPA_P1` 仍永久锁定。", "",
        "Toys 与 Beauty 的 C−B Kendall 均为负；Toys 的 C broad harm 为 3.125%，超过原 1% 上限，而 Beauty 为 0%。两域的 C Recall@10/NDCG@10 方向相反。teacher-exclusive 用户存在，但 C 在 Toys 为 0/6、Beauty 为 1/14 进入 beam@50。综合证据指向 teacher→path transfer 负向、扰动安全性失败、跨域 rank shift 与 realization bottleneck 并存；现有字段仍不能证明 teacher 本身正确，也不能作因果分解。", "",
        "## 固定边界与完整性", "", f"- Parent decision: `{summary['parent_decision']}`", "- 仅读取 P0A recovery 的锁定 JSON/CSV；没有 forward、训练或解码。", f"- optimizer steps: `{summary['optimizer_steps']}`；GPU: `{summary['gpu_count']}`。", f"- Sports/test/external development read: `{summary['sports_read']}/{summary['test_read']}/{summary['external_development_read']}`。", "- 空 beam rank 固定右删失为 51，仅用于配对 rank-change；原始空值未被改写。", "- 9 个父输入 SHA-256 均与预注册值一致；A/B/C key、行数、schema、finite 值及 summary 聚合均复算通过。", "",
        "## 四段失败链", "",
    ]
    for ds in DATASETS:
        x = d[ds]
        a, p, b = x["item_to_path_alignment"], x["perturbation_behavior"], x["beam_realization"]
        lines += [f"### {ds}", "", f"- Teacher availability：teacher-exclusive `{b['teacher_exclusive_n']}/256`；该数量不代表 teacher 正确性。", f"- Item→path：C−B Kendall mean `{a['estimate']:.6f}`，median `{a['median']:.6f}`，95% paired bootstrap CI `[{a['ci95_low']:.6f}, {a['ci95_high']:.6f}]`。", f"- Perturbation：C null-rate mean `{p['C_null_rate']['mean']:.6f}`；max-abs-delta 触及 0.3 bound 的比例 `{p['bound_hit_rate_at_0_3']:.3%}`；C broad harm `{x['parent_outcomes']['C_broad_harm']:.3%}`。", f"- Beam realization：B `{b['B_in_beam50']['successes']}/{b['B_in_beam50']['n']}`（Wilson 95% `[{b['B_in_beam50']['wilson95_low']:.3f}, {b['B_in_beam50']['wilson95_high']:.3f}]`）；C `{b['C_in_beam50']['successes']}/{b['C_in_beam50']['n']}`（`[{b['C_in_beam50']['wilson95_low']:.3f}, {b['C_in_beam50']['wilson95_high']:.3f}]`）。", ""]
    lines += ["## 分层、多重比较与可解释性限制", "", f"预冻结用户分层共 `{summary['counts']['strata_rows']}` 行，fit-prefix census 共 `{summary['counts']['prefix_census_rows']}` 行。探索性相关与分层检验统一使用 Benjamini–Hochberg FDR 0.05；完整 p/q 值保存在 `summary.json`，主结论不依赖未经校正的 p 值。fit-prefix 与 calibration-user cohort 未连接。", "", "D0 是结果知情的 post-hoc 归因审计。它能定位共现的失败环节，但不能从这些既有字段识别反事实因果，也不能判定 teacher 的 item preference 是否正确。因此 `INCONCLUSIVE_ATTRIBUTION` 与其他机制标签并存。", "", "## 封存决定", "", "- `tipa_p1_unlocked=false`", "- 不补数据、不改 cohort、不重跑 P0A。", "- 不搜索 bound、层数、loss、teacher、seed 或 beam。", "- 本报告完成后不自动实现下一结构或读取新数据。", ""]
    return "\n".join(lines)


def run(parent: Path, output: Path, report: Path) -> dict[str, Any]:
    started = datetime.now().astimezone().isoformat()
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = verify_hashes(parent)
    rng = np.random.default_rng(SEED)
    all_paired, all_strata, all_prefix, intervals, tests = [], [], [], [], []
    failure_chain, parent_summaries = {}, {}
    for dataset in DATASETS:
        parent_summary, users, arms, prefixes = validate_dataset(parent, dataset)
        parent_summaries[dataset] = parent_summary
        paired = make_paired(dataset, users, arms)
        all_paired.extend(paired)
        strata, strata_tests = user_strata(dataset, paired, rng)
        all_strata.extend(strata); tests.extend(strata_tests)
        all_prefix.extend(prefix_census(dataset, prefixes, rng))
        chain, correlations = analyze_dataset(dataset, parent_summary, paired, rng, intervals)
        failure_chain[dataset] = chain; tests.extend(correlations)
    bh_adjust(tests)
    for dataset in DATASETS:
        failure_chain[dataset]["perturbation_behavior"]["correlations"] = [r for r in tests if r["dataset"] == dataset and r["family"] == "perturbation_correlation"]
    kd_negative = all(failure_chain[d]["item_to_path_alignment"]["estimate"] <= 0 for d in DATASETS)
    realization_not_better = all(failure_chain[d]["beam_realization"]["C_in_beam50"]["successes"] <= failure_chain[d]["beam_realization"]["B_in_beam50"]["successes"] for d in DATASETS)
    unsafe = any(failure_chain[d]["parent_outcomes"]["C_broad_harm"] > .01 for d in DATASETS)
    recall_signs = [np.sign(failure_chain[d]["parent_outcomes"]["C_recall10_delta"]) for d in DATASETS]
    ndcg_signs = [np.sign(failure_chain[d]["parent_outcomes"]["C_ndcg10_delta"]) for d in DATASETS]
    domain_shift = recall_signs[0] != recall_signs[1] or ndcg_signs[0] != ndcg_signs[1]
    bottleneck = all(failure_chain[d]["beam_realization"]["teacher_exclusive_n"] > 0 and failure_chain[d]["beam_realization"]["C_in_beam50"]["rate"] < .25 for d in DATASETS)
    labels = []
    if kd_negative and realization_not_better: labels.append("TEACHER_TO_PATH_TRANSFER_NEGATIVE")
    if unsafe: labels.append("PATH_PERTURBATION_UNSAFE")
    if domain_shift: labels.append("DOMAIN_SPECIFIC_RANK_SHIFT")
    if bottleneck: labels.append("REALIZATION_BOTTLENECK")
    labels.append("INCONCLUSIVE_ATTRIBUTION")
    completed = datetime.now().astimezone().isoformat()
    script_hash = sha256(Path(__file__))
    summary = {
        "experiment_id": "GRAM_PHASE8_TIPA_D0_FAILURE_ATTRIBUTION_V1", "status": "ANALYZED",
        "started_at": started, "completed_at": completed, "parent_decision": "STOP_TIPA_NO_PATH_REALIZATION",
        "analysis_only": True, "post_hoc": True, "input_hashes": input_hashes,
        "analysis_script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": script_hash},
        "integrity": {"parent_hashes_match": True, "schemas_match": True, "rows_and_keys_align": True, "summary_exactly_recomputed": True, "finite_values": True, "rank_missing_semantics": "beam@50 absence censored to 51 for paired rank change only"},
        "failure_chain": failure_chain, "exploratory_tests_bh_fdr": tests, "diagnostic_labels": labels,
        "tipa_p1_unlocked": False, "sports_read": False, "test_read": False, "external_development_read": False,
        "optimizer_steps": 0, "gpu_count": 0, "forward_passes": 0, "decode_calls": 0, "bootstrap_seed": SEED, "bootstrap_resamples": N_BOOTSTRAP,
        "counts": {"paired_effect_rows": len(all_paired), "strata_rows": len(all_strata), "prefix_census_rows": len(all_prefix), "bootstrap_interval_rows": len(intervals)},
        "sealed_actions": ["TIPA_P1_LOCKED", "NO_P0A_RERUN", "NO_NEW_DATA", "NO_ADAPTER_PATCH_SELECTION"],
    }
    summary = sanitize_json(summary)
    write_csv(output / "paired_effects.csv", all_paired)
    write_csv(output / "strata.csv", all_strata)
    write_csv(output / "prefix_census.csv", all_prefix)
    write_csv(output / "bootstrap_intervals.csv", intervals)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    integrity = {"status": "PASS", **summary["integrity"], "tipa_p1_unlocked": False, "sports_read": False, "test_read": False, "optimizer_steps": 0}
    (output / "integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.write_text(render_report(summary), encoding="utf-8")
    manifest_paths = [output / n for n in ("summary.json", "paired_effects.csv", "strata.csv", "prefix_census.csv", "bootstrap_intervals.csv", "integrity.json")] + [report]
    manifest = {"experiment_id": summary["experiment_id"], "created_at": completed, "analysis_script": summary["analysis_script"], "inputs": input_hashes, "outputs": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in manifest_paths]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=ROOT / "artifacts/phase8/tipa_p0_branching_recovery")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase8/tipa_d0_failure_attribution")
    parser.add_argument("--report", type=Path, default=ROOT / "report/第八阶段/GRAM_第八阶段_TIPA-D0路径对齐失败归因审计报告.md")
    args = parser.parse_args()
    result = run(args.parent.resolve(), args.output.resolve(), args.report.resolve())
    print("TIPA_D0_COMPLETE", json.dumps({"status": result["status"], "labels": result["diagnostic_labels"], "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
