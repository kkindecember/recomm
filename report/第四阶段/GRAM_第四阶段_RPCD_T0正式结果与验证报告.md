## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-27
- Verification Status: ANALYZED
- Version Label: `rpcd_t0_validation_v1`

# RPCD T0 正式结果与验证报告

## 1. 执行结果

- ID：`GRAM_PHASE4_RPCD_T0`
- 类型：training + analysis
- 状态：completed
- 命令：`bash experiment/phase4/run_phase4_rpcd_t0.sh start`
- 正式运行：2026-07-27 12:51:13–12:52:41（约 88 秒）
- 共享 SASRec checkpoint epoch：8
- 共享 hybrid weight：0.2
- 科学决策：`STOP_RPCD_NO_TEACHER_COMPLEMENTARITY`
- 异常：无；全部 loss/metric 有限，GPU3 已恢复

## 2. 完整性

- Toys 19,412、Beauty 22,363 条 prediction/sequence 用户逐一对齐；
- validation target 对齐率 100%；
- test prediction 未读取，`sequence[-1]` 未索引；
- epoch 仅由 training-prefix hash calibration 选择；
- fusion 配置仅由 validation calibration 20% 选择；
- 相同 epoch 8 和 weight 0.2 锁定到双域 audit 80%；
- 配置 SHA-256：
  `d27240ad0f553d7d54b064c50033ba9de8f0069796f6a5527453b42d6964dbbf`；
- summary SHA-256：
  `ea27d60ce7db71b3bdb27041d7ecd7babab1e18e66b07dd34f6579f308bbe793`。

## 3. Audit 指标

| 数据集 | GRAM NDCG@10 | Hybrid NDCG@10 | 相对增益 | 95% bootstrap CI | Recall@10 绝对增益 |
|---|---:|---:|---:|---:|---:|
| Toys | 0.076196 | 0.076454 | +0.339% | [-0.401%, +1.087%] | +0.0836pp |
| Beauty | 0.065952 | 0.066086 | +0.203% | [-0.644%, +1.084%] | +0.0390pp |

| 数据集 | union Recall@50 增益 | 95% CI | miss@10→SAS hit@50 | tail NDCG 相对增益 | 95% CI |
|---|---:|---:|---:|---:|---:|
| Toys | +3.172pp | [+2.902pp, +3.442pp] | 5.699% | -2.484% | [-3.168%, -1.848%] |
| Beauty | +3.032pp | [+2.776pp, +3.283pp] | 6.296% | -3.379% | [-4.518%, -2.407%] |

NDCG/Recall 的小幅正点估计不确定且远低于预注册 +1% 门槛；union coverage 的正增益
稳定，但 tail harm 也稳定。不能只报告前者而忽略后者。

## 4. 门槛复算

| 数据集 | union +3pp | miss recovery 10% | NDCG +1% | Recall nondecrease | tail ≥-0.5% | 总结 |
|---|---|---|---|---|---|---|
| Toys | PASS | FAIL | FAIL | PASS | FAIL | FAIL |
| Beauty | PASS | FAIL | FAIL | PASS | FAIL | FAIL |

按 conjunctive stop rule，原 RPCD 不进入 T1。

## 5. 统计解释

Overall Confidence：**CAUTION**

原因不是结果计算可疑，而是当前只有一个 SASRec seed，且 Beauty/Toys 已参与多轮
development。paired bootstrap 使用 audit users、seed 2023、5,000 次重采样；它描述
用户抽样不确定性，不覆盖训练 seed、模型选择或跨数据集泛化不确定性。Sports 尚未
运行，因此这些数字不能当作论文的独立确认结果。

## 6. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | NOTE | 已分 Toys/Beauty 与 head/tail；整体小正增益与 tail 负增益并存，禁止只报整体 |
| Ecological fallacy | NOTE | 推断单位与评估单位均为用户，不从域均值推断每个用户受益 |
| Berkson's paradox | CAUTION | audit 是 hash 留出，但 Beauty/Toys 属反复使用的开发域 |
| Collider bias | NOTE | 未按结果变量筛选 audit；tail 是 training-only popularity 预定义分组 |
| Base-rate neglect | NOTE | 同时报 miss 总数与条件 hit rate，未只报条件准确率 |
| Regression to mean | NOTE | 未按极端 metric 选择 audit users |
| Survivorship bias | NOTE | prediction/sequence 用户全集对齐，无训练后用户剔除 |
| Look-elsewhere effect | CAUTION | weight grid 在 calibration 选择并锁定 audit，但整个第三/四阶段已探索多个方向 |
| Garden of forking paths | CAUTION | T0 有预注册并按失败停止；跨方向持续搜索仍需在论文中区分探索/确认 |
| Correlation ≠ causation | NOTE | 这里只判断 ranking association/effect gate，不作现实世界因果主张 |
| Reverse causality | NOTE | leave-one-out 时间顺序固定，不从结果反推用户偏好成因 |

## 7. 产物

- `artifacts/phase4/rpcd_t0/summary.json`
- `artifacts/phase4/rpcd_t0/preflight.json`
- `artifacts/phase4/rpcd_t0/sasrec_Toys_epoch8.pt`
- `artifacts/phase4/rpcd_t0/sasrec_Beauty_epoch8.pt`
- `artifacts/phase4/rpcd_t0/teacher_top50_{Toys,Beauty}.jsonl`
- `artifacts/phase4/logs/rpcd_t0.log`

首次 NaN 工程运行仍单独保存在
`artifacts/phase4/rpcd_t0_invalid_nan_20260727_123239/`，未与正式结果混合。
