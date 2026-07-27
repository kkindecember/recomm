# GRAM 第三阶段 PENS H0-D 诊断报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: ANALYZED
- Version Label: `pens_h0_d_v1`

- 决策：**`STOP_PENS_NO_CAUSAL_BENEFIT`**
- 边界：冻结 validation checkpoint/cohort；未训练、未生成 beam、未读 test。

## 双数据集 gate

| Dataset | Integrity | Structural | Causal benefit | No broad harm |
|---|---:|---:|---:|---:|
| Toys | True | True | False | False |
| Beauty | True | True | False | False |

## 锁定统计

### Toys

- exposure–norm Pearson=-0.973954; `||P20||/||P1||`=5.042281
- tail-miss norm-only gain: mean=0.004552, 95% CI=[-0.067095, 0.079676], P(>0)=0.449219
- tail-hit norm-only gain mean=-0.340117

### Beauty

- exposure–norm Pearson=-0.952188; `||P20||/||P1||`=5.390997
- tail-miss norm-only gain: mean=-0.851986, 95% CI=[-0.998229, -0.700632], P(>0)=0.148438
- tail-hit norm-only gain mean=-0.622155

## 解释边界

zero-position 与 history-length 分层仅为描述性结果。只有双数据集结构复制、
tail-miss 因果收益和 tail-hit 无广泛伤害全部通过才解锁 H1。
