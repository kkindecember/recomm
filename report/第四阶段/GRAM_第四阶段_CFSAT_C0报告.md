# GRAM 第四阶段：CF-SAT C0 报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-27
- Verification Status: ANALYZED（同配置 deterministic 决定复现；非独立环境验证）
- Version Label: `cfsat_c0_v2_lineage_audited`

固定决定：**`STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL`**。

| Dataset | Integrity | Margin mean [95% CI] | Positive users | Helpful nodes | Deficit/helpful | Deficit users | Supported depths |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | True | 0.462004 [0.343082, 0.596947] | 0.8725 | 0.4815 | 0.2429 | 0.4608 | 2 |
| Beauty | True | 0.815420 [0.613018, 1.039991] | 0.9510 | 0.3191 | 0.1602 | 0.3235 | 2 |

## 固定解释

两个数据集的 clean–corrupt margin 均强通过：真实 CF 比格式、K、可见长度和
metadata 位置匹配的错误 CF 更支持 gold prefix，说明 frozen GRAM 已能区分
collaborative evidence 真伪。

但主方向要求的不只是“能区分”，还要求“真实 CF 在足够多节点上优于 no-CF”，否则
继续强化 CF 的训练面过窄。node-level helpful rate 只有 Toys 48.15%、Beauty
31.91%，均低于预注册 60%，因此触发
`STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL`。Beauty 的 deficit/helpful 16.02% 也低于
20%，构成独立的后续门槛失败。

这里不与 MARC 报告的约 85% sample-level CF positive rate矛盾：MARC 先对同一
user 的多个 generation depths 求平均再判断 sample sign；C0 的 60% 门槛针对所有
非 EOS Trie nodes。聚合单位不同，不得相互替代。

## 完整性与数据边界

- Toys/Beauty 均为 fit 307 / calibration 103 / audit 102，跨 split user overlap=0；
- clean serialization、K、donor target exclusion、overlap、可见长度、真实 Collator
  attention mask 和 metadata start identity 均为 1.0；
- Trie membership、finite rate、checkpoint parameter SHA identity 均为 1.0；
- optimizer steps=0；未生成 beam，未读取 validation/test 或 `sequence[-2:]`。

## 执行 lineage

第一次完整运行已经写出同一 STOP 决定，但普通 sandbox 无法观察宿主 tmux/PID，
运行一度被误判为异常并手工恢复资源。随后以完全相同的 config/code/checkpoint
启动 deterministic 重跑；重跑再次得到同一决定并正常恢复 GPU3 资源。当前
`summary.json` 来自第二次运行，wall time 290.23 秒；日志保留第一次 561.01 秒的
同决定记录。该复现不算独立环境验证，也未用于修改 cohort、corruption 或门槛。

## 统计解释与谬误扫描

总体置信等级：**CAUTION**。停止决定由预注册阈值直接失败支持；它不能证明任何
CF-SAT 训练实现必然无效，只证明当前方向未获得进入训练阶段所需的双数据集前提。

11/11 statistical fallacies checked：

| 类型 | 结论 |
|---|---|
| Simpson's paradox | CAUTION：sample-level 与 node-level 正比例明显不同；已明确限定 C0 使用 node-level gate，未混合聚合单位。 |
| Ecological fallacy | user-cluster margin 只用于 user-level CI；node helpful rate 不外推为 user 因果效应。 |
| Berkson's paradox | cohort 仅包含索引、历史和 donor 条件均满足的 training-prefix users，代表性受限。 |
| Collider bias | 未依据 clean/corrupt 分数选择 cohort；未发现由结果共同决定的选择变量。 |
| Base-rate neglect | 同时报 positive-user rate、helpful-node rate、deficit rate 和 user coverage。 |
| Regression to the mean | 无极端组 pre/post 选择，不适用。 |
| Survivorship bias | donor 不合格样本可能被拒绝，但本次两域 rejected 均为 0。 |
| Look-elsewhere effect | 全部预注册 gate 和失败项均报告；未用强 margin 结果覆盖 helpful-rate 失败。 |
| Garden of forking paths | 评分后未修改 0.10 margin、60% helpful、20% deficit 或双数据集规则。 |
| Correlation ≠ causation | clean–corrupt frozen intervention只支持模型敏感性差异，不声明训练会改善 Recall/NDCG。 |
| Reverse causality | corruption 是固定输入干预，gold prefix 只用于离线评分；正式推理 feature 未读取 target。 |

C0 只使用 training-prefix frozen scoring，不训练、不生成 beam、不读取
validation/test。C1 correctness smoke、10% pilot 和 fresh-dataset confirmation
均不解锁。
