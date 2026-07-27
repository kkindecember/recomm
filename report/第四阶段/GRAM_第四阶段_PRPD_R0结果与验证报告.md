## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-27
- Verification Status: ANALYZED
- Version Label: `prpd_r0_validation_v1`

# PRPD R0 结果与验证报告

## 1. 结果

- Experiment ID：`GRAM_PHASE4_PRPD_R0`
- 类型：CPU-only deterministic analysis
- 状态：completed
- 用时：118.53 秒
- 扫描：5 个 gamma × 11 个 weight，共 55 个共享配置
- 选择结果：`gamma=0, weight=0`
- 决策：`STOP_PRPD_NO_DEBIASED_EFFECT`

只有五个 `weight=0` identity 配置满足 calibration 的双域 Recall/tail
nondecrease。锁定 identity 后，Toys/Beauty audit 的 NDCG、Recall、head 和 tail
增益均精确为 0，未达到双域 NDCG +1% 主门槛。

## 2. 最接近通过的配置

| 配置 | Toys NDCG | Beauty NDCG | Toys tail | Beauty tail | 结论 |
|---|---:|---:|---:|---:|---|
| gamma=0, w=0.2 | +0.437% | +2.264% | -2.583% | -2.906% | broad 正、tail 明显失败 |
| gamma=0, w=0.1 | +0.581% | +1.583% | -0.336% | -0.903% | Beauty broad 通过、双域 tail 失败 |
| gamma=0.25, w=0.1 | +0.162% | +0.404% | -0.044% | -0.135% | harm 变小但增益消失 |
| gamma=0.5, w=0.1 | -0.358% | -0.781% | +0.026% | -0.059% | broad/Recall 开始下降 |

表中均为 calibration 描述值，不是 audit confirmation。它们说明 popularity
subtraction 存在明确 trade-off，而不是遗漏了一个接近门槛的安全配置。

## 3. 完整性

- Toys/Beauty teacher user set 与 sequence user set 完全一致；
- teacher target 与 `sequence[-2]` 对齐率 100%；
- test prediction 未读取，`sequence[-1]` 未索引；
- 两域使用同一个 hash calibration/audit split；
- 1,100 次 `gamma=0` 与 RPCD fusion 排名逐项 exact match；
- 配置 SHA-256：
  `f12e5e4c06ad9dba6b211cbfca5b22fd6ae0f3423ff5ce889e0c7c1b8edc0d77`；
- preflight SHA-256：
  `88df2c6600b80b8a4d998ca2b19fa80480499adc2aab387c75a9e6c0296f3963`；
- summary SHA-256：
  `cf86b1b90eaa8dea5b640db04bb7fee4b97bafff674ddc847f19af7996b233a6`。

## 4. Statistical Interpretation

Overall Confidence：**CAUTION**

对这个预注册的 55 格网，identity 胜出和主门槛失败是确定的；但这不能外推为“所有
流行度去偏方法都无效”。本实验只检验 user-internal reciprocal-rank score 减
training-popularity midrank percentile 的一种简单残差代理。Beauty/Toys 又是反复
使用的 development domains，不能当独立论文确认。

## 5. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 结果 |
|---|---|---|
| Simpson's paradox | NOTE | 同时报 overall/head/tail；不以 Beauty broad gain 掩盖 tail harm |
| Ecological fallacy | NOTE | 不从域均值推断每个用户受益 |
| Berkson's paradox | CAUTION | hash audit 完整，但域本身是多轮开发选择 |
| Collider bias | NOTE | 配置选择未使用 audit outcome |
| Base-rate neglect | NOTE | 报告 55 个配置和 eligible 配置数，而非只报最好值 |
| Regression to mean | NOTE | 未按 audit 极端值选择 users |
| Survivorship bias | NOTE | 无用户丢失，teacher/sequence 集合完全一致 |
| Look-elsewhere effect | CAUTION | 55 格网只在 calibration 搜索，但跨方向探索很多 |
| Garden of forking paths | CAUTION | R0 已预注册；整体研究仍应区分探索与 Sports 确认 |
| Correlation ≠ causation | NOTE | 仅陈述离线排序差异 |
| Reverse causality | NOTE | popularity 只由 validation/test 前交互构造 |

## 6. 产物

- `artifacts/phase4/configs/prpd_r0_preregistered.json`
- `artifacts/phase4/prpd_r0/preflight.json`
- `artifacts/phase4/prpd_r0/summary.json`
- `experiment/phase4/prpd_r0.py`
- `experiment/phase4/test_prpd_r0.py`
