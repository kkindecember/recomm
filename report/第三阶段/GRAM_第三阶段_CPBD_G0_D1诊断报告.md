# GRAM 第三阶段：CPBD G0-D1 static truncation census

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: VERIFIED
- Version Label: `cpbd_g0_d1_v1`

## 固定决策

**`G0_D2_DESIGN_ALLOWED`**

本诊断只审计结构性截断；未加载 checkpoint、未评分、未训练、未使用 GPU，
也未读取 validation/test target 或效果。

## 双数据集主结果

| 数据集 | items | recoverable>=8 | recoverable median | metadata retention median | displaced CF median | gate |
|---|---:|---:|---:|---:|---:|---|
| Toys | 11,924 | 0.7642 | 33.00 | 0.6562 | 29.00 | PASS |
| Beauty | 12,101 | 0.9998 | 83.00 | 0.2742 | 79.00 | PASS |

## 解释边界

通过只表示当前 GRAM serialization 在双数据集中存在广泛、可由固定内容重排
机械恢复的 metadata displacement。它不表示被恢复 metadata 有推荐价值，
也不表示 metadata-first 是最终方法。下一步若获准，只能先预注册固定预算、
固定 CF identity、带位置 control 的 frozen outcome diagnosis。
