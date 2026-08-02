# GRAM 后续结构性方向分阶段实验治理规则

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: PROSPECTIVE_GOVERNANCE_FROZEN
- Version Label: `gram_structural_direction_staged_governance_v1`
- Scope: GACR-v8、candidate drafting + GRAM verification，以及此后所有结构性新方向
- Exclusion: 已于 2026-08-02 19:19+08:00 启动的 GACR-v7

## 1. 目的与适用边界

本规则用于避免在尚未确认机制方向时，直接投入双域、3 seeds、fresh validation 的完整实验；同时
避免因为 guardrail 上一两个用户的离散变化，连续设计多个局部补丁版本。它只约束 v7 之后尚未启动的
实验，不追溯修改 GACR-v7 的冻结配置、输入哈希、运行状态或科学判定。

任何 v8、fallback 或新结构方向的详细子计划，都必须显式引用本文件并声明 P0/P1 所处阶段。未通过
P0 不得启动 P1；P0/P1 均不得读取 Sports/test。

## 2. P0：单 seed、train/calibration-only 机制 pilot

每个结构性新方向首先只允许运行一个轻量机制 pilot：

- 固定 seed=`2023`；不得看到结果后更换 seed；
- 只使用 train 与隔离 calibration，禁止构造或读取新的 fresh validation cohort；
- 优先复用经过 SHA256 锁定的候选 records、特征与 checkpoint，避免重复执行昂贵的冻结 GRAM
  候选构建；
- 只实现回答核心机制问题所需的最小模型与最小 ablation，不在 P0 扫描大规模超参数网格；
- P0 运行前必须冻结：唯一改变因素、机制指标、最低有效信号、伤害非劣界、资源上限和停止规则；
- P0 只回答“机制是否存在、实现是否可训练、是否值得完整验证”，不得把 calibration 收益作为独立
  泛化结果或论文主结果。

P0 至少检查：loss/gradient/checkpoint finite、identity 或预期退化路径、候选/用户对齐、backbone
未更新、Sports/test 未读取，以及该方向预注册的直接机制量。没有达到运行前冻结的最低机制信号时，
立即停止该方向的当前实现，不得以更换 seed 或读取 fresh validation 寻找正结果。

## 3. P1：三 seeds fresh validation

只有 P0 明确通过预注册机制门，才允许另写并冻结 P1 子计划：

- Toys、Beauty × seeds `2023/2024/2025`；
- 使用与全部历史 cohort 隔离的新 fresh development cohort；
- 保留原始 GRAM、当前 incumbent、P0 机制对应臂和必要 nested ablation；
- 报告逐用户配对结果、域/seed 稳定性、绝对差与相对差、head/tail、Recall@10/50、NDCG@10、
  broad harm、coverage、效率和资源；
- P1 不得继续选择 P0 超参数；P0 中冻结的结构、权重和训练预算必须原样进入 P1；
- P1 完成后只保存结果，未经研究者明确要求不得自动分析、重试或启动下一版本。

3 seeds 的作用是确认跨随机初始化稳定性，不再用于方向初筛。方向初筛的计算预算必须集中在 P0。

## 4. Guardrail 的实质性与禁止补丁规则

后续实验必须在运行前为离散指标冻结可解释的非劣界，并将点估计、逐用户配对区间与实际受影响用户数
同时报告。不得把一两个用户造成的微小离散变化自动解释为需要新模型版本，但也不得忽略越过预注册
非劣界的实质伤害。

具体治理规则：

1. guardrail 变化位于预冻结非劣界内：标记为可接受波动或不确定风险，不得因此设计 gate、
   attenuation、multiplier、soft weighting 等局部救援版本；
2. 点估计轻微越界但配对区间跨越非劣界：停止当前 P1 的替换资格，记录不确定性；不得仅为该 cell
   追加补丁版本；
3. 明确越界且方向稳定：否决当前方法，下一步必须修改被证据指向的结构性瓶颈，而不是给原输出追加
   用户 gate 或缩放系数；
4. 每个核心假设最多允许“一次 P0 + 一次 P1”。除实现错误或完整性失败外，不运行同机制的 vX.1、
   vX.2 邻近救援；
5. 实现错误的恢复不得改变科学配置、seed、cohort 或输入 checkpoint，并须在独立 recovery 产物中
   锁定原配置及已完成 checkpoint SHA256。

## 5. 当前路线的执行映射

| 方向 | P0 核心问题 | P1 启动条件 | 邻近补丁限制 |
|---|---|---|---|
| GACR-v8 | 真实 path score 与 candidate interaction 是否产生 train/calibration 机制增量 | 单 seed 的 path-aware 与 listwise nested ablation 达到预冻结机制门 | 失败后不增加层数、hidden size 或搜索 loss |
| Candidate drafting F0/F1 | 新 drafter 是否提供双域独占 coverage | 无训练 coverage 审计与单 seed 资格门通过 | coverage 不足时不训练 verifier |
| GRAM verifier F2 | 统一 lexical-path likelihood 能否兑现新增 coverage | drafter coverage 资格通过，且 verifier pilot 有 realization 信号 | 不以 gate/融合权重网格救援 |

## 6. 资源与可复现要求

- 候选 records、特征、用户集合和输入 checkpoint 能跨 seed/ablation 复用时，必须缓存并记录 SHA256；
- P0 与 P1 使用独立输出目录、状态文件和具名 tmux；长实验结束后恢复 CodeLlama；
- 所有 runner 禁止自动 retry；scientific exit 与资源恢复状态分开记录；
- P0 通过只代表允许投入 P1，不代表方法有效；P1 development 通过也不代表论文确认性结论；
- Sports 继续作为一次性 confirmation domain，test 继续封存到方法、统计和 checkpoint 选择规则完全冻结。

## 7. 强制决策模板

后续子计划必须在启动前填入：

```text
阶段：P0 / P1
唯一结构假设：
固定 seed/cohort：
直接机制指标：
最低有效信号：
guardrail 非劣界：
通过后唯一下一步：
失败后停止项：
禁止的邻近补丁：
候选/特征缓存 SHA256：
Sports/test read：false / false
```

缺少任一项时不得启动。该模板的目的不是增加文档负担，而是确保方向初筛便宜、正式验证充分、失败后
能够真正停止，并将计算预算投入结构性创新而非连续局部修补。
