#!/usr/bin/env python3
"""Build the human-readable stage-3 S0 report from machine-readable outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
S0_ROOT = ROOT / "artifacts/phase3/s0"
REPORT = ROOT / "report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md"


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def load_run(dataset: str):
    run_dir = S0_ROOT / dataset / "validation"
    summary_path = run_dir / "summary.json"
    coverage_path = run_dir / "coverage.csv"
    if not summary_path.is_file() or not coverage_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    with coverage_path.open() as handle:
        coverage = [
            row
            for row in csv.DictReader(handle)
            if row["group"] == "overall"
        ]
    return summary, coverage


def main() -> int:
    loaded = {dataset: load_run(dataset) for dataset in ("Toys", "Beauty")}
    complete = all(loaded.values())
    status = "ANALYZED" if complete else "PARTIAL"
    dataset_decisions = {
        dataset: run[0]["promotion_gate"]["decision"]
        for dataset, run in loaded.items()
        if run
    }
    if complete and all(value == "GO" for value in dataset_decisions.values()):
        overall_decision = "GO"
    elif complete:
        overall_decision = "MODIFY"
    else:
        overall_decision = "INCOMPLETE"
    lines = [
        "# GRAM 第三阶段 S0 离线诊断报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-22",
        f"- Verification Status: {status}",
        "- Version Label: s0_offline_v1",
        "- Upstream: `plan/GRAM_第三阶段_创新探索与渐进式实验计划.md`",
        "",
        "## 1. 执行状态",
        "",
        "S0 仅使用 CPU 和既有 best-checkpoint 预测，不训练模型、不占用 GPU。"
        "重排公式和小型网格只允许在 validation 上选择；本报告不会用 test 结果调参。",
        "",
        "| 数据集 | validation 预测 | 状态 |",
        "|---|---|---|",
    ]
    for dataset in ("Toys", "Beauty"):
        if loaded[dataset]:
            summary, _ = loaded[dataset]
            lines.append(
                f"| {dataset} | 有 | {summary['material_passport']['verification_status']} |"
            )
        else:
            reason = "缺失；需从锁定 checkpoint 补推理" if dataset == "Beauty" else "缺失"
            lines.append(f"| {dataset} | 无 | {reason} |")

    for section_number, dataset in enumerate(("Toys", "Beauty"), start=2):
        if not loaded[dataset]:
            continue
        summary, coverage = loaded[dataset]
        baseline = summary["baseline"]
        selected = summary["selected_config"]
        oracle = summary["beam_oracle"]
        audit = summary["audit"]
        gate = summary["promotion_gate"]
        lines.extend(
            [
                "",
                f"## {section_number}. {dataset} validation 结果",
                "",
                f"### {section_number}.1 Lineage 与完整性",
                "",
                f"- 用户数：{audit['rows']:,}",
                f"- 商品数：{audit['item_count']:,}",
                f"- 目标错配：{audit['target_mismatches']}",
                f"- 未映射 gold：{audit['unknown_gold_count']}",
                f"- 未映射 beam prediction：{audit['unknown_prediction_count']}",
                f"- CPU wall time：{audit['wall_time_seconds']:.1f} 秒",
                "",
                f"### {section_number}.2 Relation coverage",
                "",
                "| k | 最近商品覆盖率 | 最近 20 条历史并集覆盖率 | 平均并集候选数 |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in coverage:
            lines.append(
                f"| {row['k']} | {pct(float(row['latest_item_coverage']))} | "
                f"{pct(float(row['relation_coverage']))} | {float(row['mean_union_size']):.2f} |"
            )
        lines.extend(
            [
                "",
                f"### {section_number}.3 Beam 上限与离线重排",
                "",
                "| 指标 | Baseline | 选中重排 | 变化 |",
                "|---|---:|---:|---:|",
                f"| Recall@5 | {baseline['recall@5']:.6f} | {selected['recall@5']:.6f} | "
                f"{selected['recall@5'] - baseline['recall@5']:+.6f} |",
                f"| Recall@10 | {baseline['recall@10']:.6f} | {selected['recall@10']:.6f} | "
                f"{selected['recall@10_absolute_delta']:+.6f} |",
                f"| NDCG@10 | {baseline['ndcg@10']:.6f} | {selected['ndcg@10']:.6f} | "
                f"{pct(selected['ndcg@10_relative_delta'])} relative |",
                f"| Beam-50 oracle Recall@5/10 | — | {oracle['beam50_target_recall']:.6f} | "
                "目标在 beam 内即可达到 |",
                "",
                f"选中配置：`{selected['config_id']}`，即 k={selected['k']}、"
                f"consensus weight={selected['consensus_weight']}、"
                f"fusion weight={selected['fusion_weight']}、recency decay={selected['recency_decay']}。",
                "",
                f"预注册晋级判定：**{gate['decision']}**（primary gate="
                f"{str(gate['primary_gate']).lower()}，subgroup gate="
                f"{str(gate['subgroup_gate']).lower()}）。",
            ]
        )

    lines.extend(
        [
            "",
            "## 4. 整体晋级决策",
            "",
            f"双数据集整体判定：**{overall_decision}**。",
            "",
        ]
    )
    if overall_decision == "GO":
        lines.append(
            "Beauty/Toys 均达到预注册 S0 门槛，可以进入 S1 实现正确性 smoke。"
        )
    elif overall_decision == "MODIFY":
        lines.append(
            "Beauty/Toys 已按同一协议完成，但至少一个数据集未达到门槛。按照预注册规则，"
            "当前不得直接进入 S1；先进行一次有边界的 S0b 可靠性拒绝探针，判断能否在不使用"
            "目标信息的条件下避免 no-CF-covered 用户退化。S0b 属于结果后提出的探索性修正，"
            "必须与原 S0 分开标记，不能冒充预注册验证。"
        )
    else:
        lines.append(
            "当前只能形成单数据集分析，不能据此完成 S0 或晋级 S1。Beauty 没有现成的 "
            "best-checkpoint validation prediction；必须补一次锁定 epoch-25 checkpoint 的 validation "
            "推理，随后用完全相同、已固定的脚本分析。"
        )
    lines.extend(
        [
            "",
            "- 结果是离线相关性证据，不等于训练后的因果增益。",
            "- 用户是配对评测单位，不是独立训练重复；S0 不做跨 seed 显著性主张。",
            "- test 尚未用于公式或超参数选择。只有双数据集 validation 配置锁定后，才允许一次性 test 诊断。",
            "- `ANALYZED` 不等于 `VERIFIED`；独立复跑前不得升级验证状态。",
            "",
            "## 5. 产物",
            "",
            "- 脚本：`experiment/phase3/s0_offline_diagnostics.py`",
            "- 机器可读结果：`artifacts/phase3/s0/<dataset>/validation/`",
            "- 后台状态：`experiment/phase3/phase3_s0_status.json`",
            "- 后台日志：`artifacts/phase3/logs/s0_toys_validation.log`；"
            "`artifacts/phase3/logs/s0_beauty_validation.log`",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
