# GRAM 第四阶段：FCRD F0 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-27
- Verification Status: ANALYZED
- Version Label: `fcrd_f0_validation_v1`

## 1. 执行结果

- Experiment ID：`GRAM_PHASE4_FCRD_F0`
- 类型：冻结 checkpoint 的 full-catalog GPU inference + deterministic analysis
- 状态：completed
- 命令：`bash experiment/phase4/run_phase4_fcrd_f0.sh start`
- 用时：112.49 秒
- 扫描：7 个 gamma × 11 个融合权重，共 77 个共享配置
- calibration 合格配置：0 / 77
- 锁定配置：fail-closed identity，`gamma=0, weight=0`
- 决策：**`STOP_FCRD_NO_FULL_CATALOG_RESIDUAL_EFFECT`**

## 2. 核心结果

FCRD 修正了 PRPD 只重排旧 top-50 的结构限制。结果证明 full-catalog
residualization 确实能够把原 top-50 外的 tail items 带入候选集，但总体覆盖与尾部
覆盖之间存在稳定冲突：

| gamma（weight=0，calibration） | Toys overall union 增益 | Toys tail union 增益 | Beauty overall union 增益 | Beauty tail union 增益 |
|---:|---:|---:|---:|---:|
| 0.0 | +3.359pp | +0.137pp | +3.052pp | +0.045pp |
| 0.1 | +3.230pp | +0.183pp | +2.961pp | +0.135pp |
| 0.3 | +3.256pp | +0.366pp | +2.984pp | +0.226pp |
| 0.5 | +2.972pp | +0.503pp | +2.803pp | +0.542pp |
| 1.0 | +2.222pp | +1.830pp | +1.876pp | +1.625pp |

小 gamma 可基本保住 overall +3pp，却远未达到 tail +1pp；`gamma=1` 达到 tail
门槛，却使两个数据集的 overall coverage 均跌破 +3pp。不存在遗漏在门槛附近、可同时
满足两者的共享配置。

由于 calibration 无合格配置，审计按预注册锁定 identity：

| 数据集 | audit overall union 增益 | 95% CI | audit tail union 增益 | 95% CI | NDCG/Recall/tail NDCG 变化 |
|---|---:|---:|---:|---:|---:|
| Toys | +3.172pp | [+2.895pp, +3.449pp] | +0.235pp | [+0.134pp, +0.347pp] | 0 |
| Beauty | +3.032pp | [+2.787pp, +3.283pp] | +0.088pp | [+0.033pp, +0.154pp] | 0 |

## 3. 解释

本实验排除了“PRPD 失败只是因为先截断 top-50”这一解释。完整目录去流行度能够改变
candidate support，但增加 tail exposure 的代价是丢失更多 head/overall candidates；
统一 gamma 无法同时满足双域 coverage 约束。由于没有 calibration-qualified 配置，
不能进入 residual Trie projection 或 student 蒸馏。

该结论只否定当前的全局 popularity residual 公式，不等价于所有个体化候选选择方法
无效。RPCD/FCRD 共同保留的正证据是：外部序列模型能稳定提供约 3pp 的互补候选覆盖；
尚未解决的是如何按用户、按候选判断何时采用这部分覆盖。

## 4. 完整性与复现

- preflight 全部通过；
- Toys 19,412、Beauty 22,363 个用户及 11,924/12,101 个 catalog item 完整对齐；
- validation target 对齐率 100%；
- `gamma=0` teacher top-50 identity rate = 100%；
- 两域使用同一 `(gamma, weight)`；
- test prediction 未读取，`sequence[-1]` 未索引；
- optimizer steps = 0；
- 配置 SHA-256：
  `e022f4aa9d7e6a24c5c27aa452c448bf9c56e9950a1d06014be8e7a88543d22d`；
- preflight SHA-256：
  `6912e54b4cfe5ed3fdbfc55392a16d5f1555f969356788c6b265a656a32eed92`；
- summary SHA-256：
  `d8b7f2cabd20b56ec84f0740b2368a53ba9538f1840c2b223cc59125aa688d99`。

## 5. Statistical Interpretation

Overall Confidence：**CAUTION**。

77 个预注册配置的失败和 coverage trade-off 在当前双域 calibration 上是明确的；
audit identity 的区间也稳定。但 Beauty/Toys 已被多轮用于方向生成，且 SASRec 只有
一个 seed，因此不能把当前结果外推成独立数据集、跨种子或所有 residualization
方法的确认性结论。

## 6. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | overall 与 tail 随 gamma 呈相反 trade-off，必须同时报告 |
| Ecological fallacy | NOTE | 不从域均值断言每个用户均受益或受损 |
| Berkson's paradox | CAUTION | hash audit 独立于 calibration，但两个域已反复参与开发 |
| Collider bias | NOTE | 配置只由 calibration 选择；audit outcome 未参与选择 |
| Base-rate neglect | NOTE | 同时报 overall/tail coverage 和各子组样本量 |
| Regression to mean | NOTE | 未按极端 audit 用户或结果重新筛样本 |
| Survivorship bias | NOTE | 用户与 catalog 全量对齐，无结果后剔除 |
| Look-elsewhere effect | CAUTION | 77 格网只在 calibration 搜索，但整个项目已探索多方向 |
| Garden of forking paths | CAUTION | F0 已预注册；跨阶段方向搜索仍须标为探索性 |
| Correlation ≠ causation | NOTE | 只陈述离线排序干预结果，不作现实因果主张 |
| Reverse causality | NOTE | popularity 只由 validation/test 之前的训练交互构造 |

## 7. 产物

- `artifacts/phase4/configs/fcrd_f0_preregistered.json`
- `artifacts/phase4/fcrd_f0/preflight.json`
- `artifacts/phase4/fcrd_f0/summary.json`
- `artifacts/phase4/fcrd_f0/{Toys,Beauty}_full_catalog_top50.npz`
- `experiment/phase4/fcrd_f0.py`
- `experiment/phase4/test_fcrd_f0.py`
- `artifacts/phase4/logs/fcrd_f0.log`
