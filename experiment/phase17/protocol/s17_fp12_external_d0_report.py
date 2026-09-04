#!/usr/bin/env python3
"""Render the finalized Stage17 FP1/FP2 external-D0 report from frozen analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import utc_now


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_SUFFIX = Path("artifacts/phase17/fullport/external_d0/attempt_001/analysis.json")
REPORT_SUFFIX = Path("report/第十七阶段/Stage17_FP12_ExternalD0评测准备报告.md")
README_SUFFIX = Path("report/第十七阶段/README.md")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def render(root: Path) -> str:
    root = root.resolve()
    analysis_path = root / ANALYSIS_SUFFIX
    analysis = _read(analysis_path)
    if (
        analysis.get("single_materialization_count") != 1
        or analysis.get("integrity", {}).get("d1_read") is not False
        or analysis.get("integrity", {}).get("test_read") is not False
        or analysis.get("controlled_recovery") is not True
    ):
        raise RuntimeError("analysis is not the sealed controlled-recovery result")

    arms = (
        "N0_NATIVE_PSID",
        "N1_NATIVE_LATTE",
        "G0_GRAM_B0_FRESH",
        "G1_GRAM_PSID_FULL",
        "G2_GRAM_LATTE_FULL",
    )
    metric_lines = [
        "| Arm | Primary variant | Hit@10 | NDCG@10 | MRR@10 | Hit@50 | NDCG@50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    execution_lines = [
        "| Arm | Attempt | GPU | Wall time (s) | Prediction SHA256 |",
        "|---|---|---:|---:|---|",
    ]
    for arm_id in arms:
        variant = analysis["primary_variants"][arm_id]
        metrics = analysis["overall"][arm_id][variant]["metrics"]
        metric_lines.append(
            "| "
            + " | ".join(
                (
                    arm_id,
                    variant,
                    _fmt(metrics["hit@10"]),
                    _fmt(metrics["ndcg@10"]),
                    _fmt(metrics["mrr@10"]),
                    _fmt(metrics["hit@50"]),
                    _fmt(metrics["ndcg@50"]),
                )
            )
            + " |"
        )
        source = analysis["arm_artifact_sources"][arm_id]
        summary = _read(root / source["summary_path"])
        execution_lines.append(
            "| "
            + " | ".join(
                (
                    arm_id,
                    str(summary.get("attempt_id", "attempt_001")),
                    str(summary["physical_gpu"]),
                    _fmt(summary["result"]["wall_seconds"]),
                    source["predictions_sha256"],
                )
            )
            + " |"
        )

    comparison_lines = [
        "| Comparison | ΔNDCG@10 | 95% CI | ΔHit@10 | Gain/Loss/Tie | Changed target rank |",
        "|---|---:|---|---:|---|---:|",
    ]
    for label, comparison in analysis["comparisons"].items():
        ndcg = comparison["effects"]["ndcg@10"]
        hit = comparison["effects"]["hit@10"]
        outcomes = comparison["primary_user_outcomes"]
        comparison_lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    _fmt(ndcg["mean_delta"]),
                    f"[{_fmt(ndcg['ci95_low'])}, {_fmt(ndcg['ci95_high'])}]",
                    _fmt(hit["mean_delta"]),
                    f"{outcomes['gain']}/{outcomes['loss']}/{outcomes['tie']}",
                    _fmt(comparison["changed_target_rank_rate"]),
                )
            )
            + " |"
        )

    gate_lines = ["| Gate | Verdict | Passed checks | Failed checks |", "|---|---|---|---|"]
    for gate_name in ("FP1", "FP2"):
        gate = analysis["gates"][gate_name]
        passed = [key for key, value in gate["checks"].items() if value]
        failed = [key for key, value in gate["checks"].items() if not value]
        gate_lines.append(
            f"| {gate_name} | `{gate['verdict']}` | {', '.join(passed) or '—'} | "
            f"{', '.join(failed) or '—'} |"
        )

    subgroup_lines = [
        "| Dimension | Group | Users | ΔNDCG@10 | ΔHit@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for dimension, groups in analysis["subgroups"]["FP2_G2_MINUS_G0"].items():
        for group, record in groups.items():
            subgroup_lines.append(
                f"| {dimension} | {group} | {record['users']} | "
                f"{_fmt(record.get('delta_ndcg@10', 0.0))} | "
                f"{_fmt(record.get('delta_hit@10', 0.0))} |"
            )

    recovery = analysis["recovery_provenance"]
    integrity = analysis["integrity"]
    psid = analysis["psid_collision_diagnostics"]
    next_action = analysis["next_action"]
    lines = [
        "# Stage17 FP1/FP2 External D0 正式结果报告",
        "",
        f"- 生成时间：`{utc_now()}`",
        f"- 正式分析：`{ANALYSIS_SUFFIX}`（SHA256 `{sha256(analysis_path)}`）",
        f"- 外部用户数：`{analysis['external_users']}`",
        "- 数据纪律：D0 仅物化一次；本次恢复只复用 sealed bundle；D1/D2/test/Sports 均未读取。",
        "",
        "## 结论",
        "",
        f"- FP1：`{analysis['gates']['FP1']['verdict']}`。",
        f"- FP2：`{analysis['gates']['FP2']['verdict']}`。",
        f"- 冻结决策：`{next_action}`。",
        "- 只有 Gate 明确通过的分支才可进入计划规定的后续注册；任何情况下当前仍不得读取 D1。",
        "",
        "## 主结果",
        "",
        *metric_lines,
        "",
        "## 配对效应",
        "",
        *comparison_lines,
        "",
        "## Gate 审计",
        "",
        *gate_lines,
        "",
        "## G2 vs G0 子组",
        "",
        *subgroup_lines,
        "",
        "## 完整性与机制证据",
        "",
        f"- 用户严格对齐：`{integrity['exact_user_alignment']}`；五臂 primary ranking 均非空：`{all(integrity['all_primary_rankings_nonempty'].values())}`。",
        f"- G2 constrained path 全合法：`{integrity['all_g2_constrained_paths_legal']}`。",
        f"- PSID collision aliases after：`{psid['collision_aliases_after']}`；reassigned items：`{psid['reassigned_items']}`。",
        f"- N1 机制：`{json.dumps(analysis['overall']['N1_NATIVE_LATTE']['beam500_agg_max']['mechanisms'], ensure_ascii=False, sort_keys=True)}`。",
        f"- G2 机制：`{json.dumps(analysis['overall']['G2_GRAM_LATTE_FULL']['beam500_agg_max']['mechanisms'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 受控恢复与运行审计",
        "",
        "attempt_001 中 N0/N1 原进程保留；G0/G2 在生成任何预测前因 PyTorch 1.11 不接受 `weights_only` 参数而失败；G1 未越过 GPU admission。研究者明确回复“同意受控恢复”后，attempt_002 恢复 G0/G2；随后研究者把 G1 更正为立即并行运行并要求完成后保持资源占用，因此 G1 使用独立 attempt_003。所有恢复均复用同一 sealed bundle。",
        "",
        *execution_lines,
        "",
        f"- 恢复授权：`{recovery['authorization_path']}`（SHA256 `{recovery['authorization_sha256']}`）。",
        f"- 恢复证据：`{recovery['recovery_provenance_path']}`（SHA256 `{recovery['recovery_provenance_sha256']}`）。",
        f"- Bundle SHA256：`{recovery['bundle_sha256']}`；single materialization count：`{recovery['single_materialization_count']}`。",
        "- G1 完成后的 GPU4 资源维护使用独立 v2 守护与 `run-NNNN` 目录；`result_selection_eligible=false`、`repeat_metrics_ignored=true`、`affects_scientific_result=false`，不重开 external D0。",
        "- 恢复与守护回归：相关 `25 passed`；Phase17 全量 `256 passed, 1 skipped, 1 warning`。",
        "- `automatic_retry=false`、`raw_external_projection_reopened=false`；attempt_001 失败证据未覆盖。",
        "",
        "## 下一步",
        "",
        f"严格执行分析冻结动作：`{next_action}`。在新 preregistration、预算与显式授权完成前，不启动 D1、FP4 或任何 D0 调参。",
        "",
    ]
    return "\n".join(lines)


def write_report(root: Path) -> Path:
    root = root.resolve()
    report = root / REPORT_SUFFIX
    _atomic_text(report, render(root))
    readme = root / README_SUFFIX
    content = readme.read_text(encoding="utf-8")
    old_prefix = "| S17-FP12-EXTERNAL-D0 |"
    replacement = (
        "| S17-FP12-EXTERNAL-D0 | `COMPLETED` | "
        "`Stage17_FP12_ExternalD0评测准备报告.md` | 正式 Gate 与下一动作见报告；D1 保持锁定 |"
    )
    rows = [replacement if row.startswith(old_prefix) else row for row in content.splitlines()]
    _atomic_text(readme, "\n".join(rows) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = write_report(args.root)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
