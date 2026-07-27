# GRAM 第三阶段 HBTR-v2 Failure Autopsy 报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: hbtr_v2_autopsy_v1
- Design Status: RESULT-INFORMED POST-HOC EXPLORATORY

## 结论

- 诊断决策：**V2_DESIGN_ALLOWED**
- HBTR-v1 保持 STOP；本报告不解锁 GPU、25%、全量、更多 seed 或 test。

## 核心结果

| 数据集 | eligible/all | prefix非平凡/有效行 | tail非平凡/有效行 | joint非平凡/有效行 | C4 vs C0 NDCG@10 | C4 top-10净迁入 |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 21.78% | 44.01% | 8.77% | 5.46% | +0.643% | +3 |
| Beauty | 18.11% | 32.98% | 3.84% | 2.41% | -0.061% | -1 |

## 锁定诊断门槛

- [PASS] 两数据集 eligible/all ≥15%。
- [PASS] 两数据集 prefix 非平凡行 ≥25%。
- [FAIL] 两数据集 tail 非平凡行 ≥20%。
- [FAIL] 两数据集 joint 非平凡行 ≥10%。
- [PASS] 至少一个数据集在同一 C1/C4 对照中同时具有正 NDCG@10 差和正 top-10 净迁入。

## 解释边界

该诊断在看到 HBTR-v1 pilot 后建立，只判断联合机制是否缺乏可辨识激活以及
是否允许设计独立 HBTR-v2。阈值不是效果门槛，不能把 HBTR-v1 STOP 改判为
MODIFY/GO；所有指标均来自既有 training-only cache 与 validation 结果，未读取 test。
