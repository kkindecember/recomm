# GRAM 第十四阶段 Stage 14-0A：Item 级评测回归报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-20
- Verification Status: VERIFIED
- Version Label: phase14_stage14_0a_item_eval_v1

## 结论

`PASS_STAGE14_0A`。新 evaluator 从 `item2lexid` 构建 `decoded path -> [item_ids]` multimap；正式模式对 ambiguous、unknown、top-K duplicate 一律 hard-fail，不再使用会静默覆盖碰撞的 `lexid2cfid`。

| 检查 | Toys | Beauty |
|---|---:|---:|
| v0 duplicate path groups | 0 | 0 |
| v0 integrity issues | 0 | 0 |
| v0 严格 item 指标 vs 历史保存指标最大差值 | 0 | 0 |
| raw-v1 duplicate path groups | 932 | 719 |
| raw-v1 collision group 内 item 数 | 2,284 | 1,641 |
| raw-v1 被严格口径移除的历史字符串 H@50 命中 | 198 | 234 |

因此 v0 历史核心 validation 数字可继续使用；raw-v1 仍不是合法 item-level 结果，只能保留为 legacy alias audit。运行未读取 test prediction，15 项测试通过。

## 产出

- 代码：`experiment/phase14/protocol/item_level_eval.py`
- 测试：`experiment/phase14/tests/test_item_level_eval.py`
- 状态：`artifacts/phase14/diagnostics/item_level_eval/status.json`
- 汇总：`artifacts/phase14/diagnostics/item_level_eval/summary.json`
- 每个子任务均含 `config.json`、`summary.json`、`item_path_audit.json`、`data_provenance.json`、`input_file_sha256.json`、`open_file_manifest.json` 与严格预测审计记录。

下一 Gate：Stage 14-0B learned NLL/rank/beam diagnosis；本结果不决定 R2PD 路线。
