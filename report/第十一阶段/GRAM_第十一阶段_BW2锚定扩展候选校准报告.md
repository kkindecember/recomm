# GRAM 第十一阶段 BW2：锚定扩展候选校准报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-05
- Verification Status: `INTEGRITY_GATE_FAILED`
- Version Label: phase11_bw2_anchored_expansion_v1
- Experiment ID: `GRAM_PHASE11_BW2_ANCHORED_EXPANSION_VALIDATION_V1`

## 1. Executive conclusion

BW2 正常完成计算，但未通过预注册完整性门，因此科学门按协议标记为 `not_evaluated`。失败原因不是
指标退化，而是独立 beam200 的前 50 候选与独立 beam50 的平均集合 overlap 只有 Toys
`0.911211`、Beauty `0.910859`，低于冻结阈值 `0.98`。

这表明增加 `num_beams` 不只是向原 beam50 末尾追加候选，而会改变约 9% 的 top50 搜索结果。
因此，BW2 的 anchored 结果同时混合“标准化方式变化”和“搜索路径变化”，不能作为对
anchor-normalization 假设的干净因果检验。

## 2. Execution audit

首次执行在科学计算前因复用的 Phase-9 loader 硬编码 50 candidates 而 exit 1；未生成科学结果，
未读取 test。修复仅新增显式 `expected_width` 解析与回归测试，`9 passed`。研究者明确回复“继续”后，
以完全相同的样本、公式和门槛重试，exit 0。

## 3. Integrity gate

| Check | Toys | Beauty |
|---|---|---|
| BW1 integrity passed | PASS | PASS |
| all 512 users | PASS | PASS |
| mean beam50/anchor50 overlap ≥ 0.98 | **FAIL (0.911211)** | **FAIL (0.910859)** |
| anchor top50 relative-order identity | PASS | PASS |
| item-head SHA matches BW1 | PASS | PASS |

Overall integrity gate：`FAILED`；scientific gate：`NOT_EVALUATED`。

## 4. Directional output（非确认性）

以下结果仅用于诊断，不算通过预注册科学门：

| Dataset | beam50 PCRF Hit@10 | anchored beam200 Hit@10 | delta | NDCG@10 delta | expansion users in top10 |
|---|---:|---:|---:|---:|---:|
| Toys | 0.123047 | 0.125000 | +0.001953 | +0.000457 | 47 |
| Beauty | 0.103516 | 0.103516 | 0 | +0.000070 | 31 |

Toys 有 1 个目标从 top10 外进入 top10、0 个退出；Beauty 为 0/0。Toys paired bootstrap
95% CI `[0,+0.005859]`，Beauty `[0,0]`。结果方向安全，但无法排除约 9% top50 搜索集合变化的贡献。

## 5. Next decision

不降低 overlap 门槛、不在 validation 上调 margin 或 PCRF 权重。下一步按照预注册失败分支，使用
train-prefix pseudo-future 构造 expansion admission gate：在更早的序列位置生成训练/校准 beams，
validation `-2` 只用于一次冻结评价。该设计必须把“生成宽候选”和“学习准入”分离，并保留 beam50
fallback，避免宽 beam 的搜索漂移直接破坏已确认机制。

## 6. Artifacts

- preregistration：`plan/第十一阶段/GRAM_第十一阶段_BW2锚定扩展候选校准计划.md`
- summary：`artifacts/phase11/bw2_anchored_expansion/summary.json`
- per-user：`artifacts/phase11/bw2_anchored_expansion/per_user.tsv`
- execution audit：`artifacts/phase11/bw2_anchored_expansion/execution_audit.json`
- evaluator：`experiment/phase11/eval_bw2_anchored_expansion.py`
