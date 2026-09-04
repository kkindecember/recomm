# GRAM 第十八阶段：PCPS-GRAM 词法锚定协同前缀生存与低风险验证计划 v0.2

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent (plan mode)`
- Created：2026-09-03
- Revision：`v0.2`；补入 2025–2026 顶会证据与共享服务器运行规则
- Canonical Plan：本文件为 Stage18 当前权威版本；v0.1 已由本版本覆盖
- Status：`PLAN_REVISED_FOR_DISCUSSION / NO_STAGE18_CODE_OR_EXPERIMENT_STARTED`
- Evidence Basis：Phase9 PCRF、Phase10 CF1、Phase11 BW1–BW3、Stage17 S3/S4/FP2/FP3 已完成结果
- Primary Direction：`PCPS-GRAM`（Popularity-Calibrated Collaborative Prefix Survival GRAM）
- Primary Baseline：相同 fold、相同训练预算的 `Fresh GRAM + frozen PCRF`
- Protected Data：Stage17 external D0 target 不得再用于方法选择；D1/D2、official test、Sports 继续封存
- Authorization Boundary：本文件只授权计划与只读证据整理；不授权代码实现、GPU 训练、D1/D2 读取或自动启动后继步骤
- Runtime Policy：预计或实测超过 10 分钟的任务必须后台运行；agent 不实时监看，研究者通过
  `artifacts/phase18/status/*.status.json` 自主观察
- Small-GPU Policy：小实验可直接选择任意不冲突、预计不 OOM 的 GPU；不设利用率阈值或过度显存余量要求
- Large-GPU Policy：大实验启动前报告最少/建议 GPU 数、单卡显存和预计时长并向研究者申请；优先多卡并行独立 arms/folds
- Post-run Occupancy：大实验科学部分结束后，在获批资源中的一张卡启动不参与选结果的重复轮用于占位，并用独立 status 标识
- GPU1 Policy：GPU1 默认不释放、不用于 Stage18；若以后明确授权，必须继承交接与恢复约束

## 0. 执行摘要

### 0.1 对历史问题的确认

PCRF 本身没有在后续实验中失效。相反，它是当前证据最扎实的正常场景增益机制：

- Toys independent test：Hit@10 `+0.007573`，95% CI `[+0.005461,+0.009633]`；
  NDCG@10 `+0.004161`；
- Beauty one-shot independent test：Hit@10 `+0.005277`，95% CI
  `[+0.003488,+0.007155]`；NDCG@10 `+0.003037`；
- Toys/Beauty × seeds 2023/2024/2025 共 6 个 validation 单元：Hit@10、NDCG@10、
  tail Hit@10 全部为正；
- fresh beam 复现与历史 cache 完全一致；
- 所有确认中 Hit@50 都严格不变，因为 PCRF 只能重排已经进入 beam50 的候选。

后来失败的是三类“PCRF 扩展”，不是 PCRF：

1. 外部候选 union 后再做 cross-fitted calibrator；
2. beam200 扩展候选的阈值/准入 gate；
3. 用新的 semantic/continuous identifier 整体替换 GRAM lexical identifier。

因此 Stage18 不否定 PCRF，也不再从另一篇论文重新起步。它只验证一个由既有证据直接导出的新问题：

> 能否在完全保留 GRAM lexical ID、FiD 和合法 Trie 的前提下，用 fold-train-only 的
> PCRF 协同信号选择真正困难的合法前缀竞争者，在训练阶段提高目标路径的 beam 生存率，
> 从而突破 PCRF 的 Hit@50=0 增益上限，并让最终 `PCPS-GRAM + PCRF` 稳健超过
> `Fresh GRAM + PCRF`？

### 0.2 本计划对“不再重复失败”的操作定义

科学实验不能保证正结果。本计划能保证的是：

- 不重复已经被证伪的 identifier replacement、候选 union gate、margin 放宽、宽 beam 直接重排；
- 在完整训练前先验证新机制是否有可作用的真实错误样本；
- 只保留一个主方法和必要的因果对照，不做论文方法组合搜索；
- 所有效果参数在未见 confirmation fold 前冻结；
- 弱正、CI 跨 0、只在 calibration 好看均判为不晋级；
- 任一关键 Gate 失败立即止损，不用换 seed、扫参数或重开同一外部 fold 挽救。

## 1. 历史证据审计

### 1.1 PCRF 是已确认成功，而不是偶然成功

| Evidence | Dataset / scope | Primary result | Interpretation |
|---|---|---:|---|
| Phase9 P9-2E | Toys independent test，19,412 users | Hit@10 `+0.007573`；NDCG@10 `+0.004161` | paired CI 明确为正 |
| Phase9 P9-R | Toys fresh beam，512 users | PCRF top10 overlap `1.0` | 排除 cache 偶然 |
| Phase9 P9-X | Beauty independent test，22,363 users | Hit@10 `+0.005277`；NDCG@10 `+0.003037` | 跨数据集确认 |
| Phase9 P9-S | 2 domains × 3 seeds | 6/6 Hit/NDCG/tail delta 为正 | 排除单 seed 偶然 |
| 所有 Phase9 confirmation | fixed beam50 | Hit@50 delta `0` | 明确的候选覆盖上限 |

冻结 PCRF 公式为：

```text
pop_z      = zscore(log(1 + train_frequency))
cf_pc      = zscore(cf_z - 0.5 * pop_z)
tail_mass  = fraction(original_top10 frequency <= train_only_q1)
reliability = 1 - tail_mass
joint      = seq_z + 1.0 * reliability * cf_pc
```

Stage18 不重新搜索 `lambda/beta/gamma/q1`，不把两个 independent test 再用作开发集。

### 1.2 PCRF 后续扩展为什么失败

| Stage | 做了什么 | 看起来有希望的信号 | 独立/正式结果 | 本阶段禁止重复的原因 |
|---|---|---|---|---|
| Phase10 CF1-A/A2 | GRAM50 与 CF 候选 union | coverage `+0.054760`；budgeted union 保留 96.43% oracle gain | 只证明候选存在 | coverage 不等于 top10 可利用性 |
| Phase10 C1 | cross-fitted monotone listwise calibrator | Hit@50 `+0.014630` | Hit@10 `-0.000309`；1/5 folds 正 | 新校准器没有超过 PCRF anchor |
| Phase10 C2 | PCRF-anchored safe insertion | tail Hit@10 `+0.005814`；366 个 CF-only target 进 top50 | Hit@10 `-0.000567`，CI `[-0.001648,+0.000515]`；0 个 CF-only target 进 top10 | 把损失从 tail 转移到 head，不是 Pareto improvement |
| Phase11 BW1 | beam50→200 | Toys/Beauty candidate recall `+0.117188/+0.126953` | GRAM Hit@10 均为 0 增益；PCRF 无增益/退化 | 搜索空间存在，但旧分数不会兑现它 |
| Phase11 BW3 P1C | train-prefix listwise admission | calibration Hit@10 `+0.066406/+0.054688` | P2 两域 Hit/NDCG/tail delta 全为 `0` | calibration hotspot 没有跨 offset 泛化 |
| Phase11 P2 diagnostic | target selection shift | expansion pool 含 60/65 个 target | 21/15 admissions 中 0 个 target；target logit 跌约 4.8–5.0 | item-head anchor 正类信号跨 offset 崩塌 |

Phase11 的关键教训不是“候选扩展永远无效”，而是：不能在生成结束后用一个依赖 item-head
热点的 gate 决定谁进入 top10。Stage18 将协同信号限制为训练期 hard-negative/置信度信息，
最终候选仍由原 lexical decoder + Trie 端到端生成。

### 1.3 Stage17 提供了什么边界

| Direction | Result | Boundary inherited by Stage18 |
|---|---:|---|
| A0 full-vocabulary BEAR proxy | `ΔNDCG@10=+0.000165`，无 paired CI | 只有弱信号；不能重做同一个 full-vocabulary proxy |
| PAWA-lite | `ΔNDCG@10=+0.000234`，CI 跨 0 | 不能只改深度权重或加轻量 decoder reducer |
| GRAM-LATTE-Full | G2−G0 `-0.036657`；G2−G1 `-0.003872` | lexical ID 不得替换为 PSID/latent forest |
| GRAM-SETRec-Paper-Full | S2−S0 `+0.001145`，CI `[-0.000154,+0.002510]`；full-set recovery `0.000468` | 只算 weak point estimate；不能以连续 set identifier 取代 lexical ID |

Stage17 没有证明“架构增强不可能成功”。它证明当前 GRAM 的 lexical identifier 与生成路径是强资产，
而整套 identifier 替换的损失远大于外来机制收益。Stage18 因而采用 lexical-anchored 约束。

### 1.4 2025–2026 顶会定向证据扫描

本轮只纳入可核验的正式会议论文、官方论文页或作者公开全文。检索范围聚焦 2025–2026 的
SIGIR、KDD、ACL/EMNLP，并按“是否能解释本地 PCRF/beam 现象、是否需要替换 lexical ID、
是否能在训练期而非后处理介入”筛选。

| Paper | Venue | Relevant idea | Stage18 decision |
|---|---|---|---|
| *Constrained Auto-Regressive Decoding Constrains Generative Retrieval* | SIGIR 2025 | constrained decoding 会产生 step-wise marginal mismatch；beam search 仅依赖边际分布存在理论局限 | **采纳约束**：不能只有 token-level loss，必须同时优化并报告完整 path cumulative score |
| *Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model*（UniGRF） | SIGIR 2025 | ranking-driven enhancer 把 ranking 信息反馈进生成模型；gradient-guided weighting 平衡目标 | **采纳原则**：PCRF 反馈进入训练，不再做推理后 gate；loss 权重按梯度口径冻结 |
| *Towards Distribution Matching between Collaborative and Language Spaces for Generative Recommendation*（DMRec） | SIGIR 2025 | collaborative 与 language space 的分布匹配比简单拼接更合理 | **局部采纳**：只在同一合法 sibling/path 集内做分布/排序约束；不新增全局 meta-network |
| *LOHRec: Leveraging Order and Hierarchy in Generative Sequential Recommendation* | EMNLP 2025 Findings | 用有序候选、hierarchy 和完整 identifier 概率做 learning-to-rank，而非只学单个正例 | **局部采纳**：对真实 on-policy paths 做累计概率排序；不迁移 quantized ID 或外部 LLM teacher |
| *BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models* | SIGIR 2026 | 正例每步 token 至少保持在 top-B 是避免 beam 提前剪枝的 relaxed necessary condition | **采纳**：实现 legal-Trie 范围内的 prefix survival；Stage17 的 full-vocabulary proxy 不能替代它 |
| *Uncertainty-aware Generative Recommendation*（UGR） | KDD 2026 | 对模型高置信生成的错误候选施加更强反馈，避免所有错误统一处理 | **局部采纳**：按 frozen parent 的 path probability 加权错误 beam；不引入 GRPO、confidence token 或 SID |

这些论文形成两条相互补充、也相互制约的证据：

1. BEAR 支持显式保护目标前缀；
2. SIGIR 2025 的 constrained-decoding 理论又提醒，只满足逐 token 条件并不足以保证最终 item 排序；
3. UniGRF、LOHRec 与 UGR 共同支持把候选级排序、完整路径概率和错误置信度反馈到训练；
4. 本地 Phase9 进一步提供了一个已经跨域、多 seed 确认的 collaborative signal——PCRF。

因此 v0.2 将 PCPS 从单纯 prefix margin 修订为一个**合法前缀 + 完整路径的双层目标**。这不是把六篇
论文模块堆叠起来：模型架构、identifier、decoder 和 inference 均不变，只改变一个训练 loss 中
“比较哪些负例、以什么证据加权、同时约束 token 还是 path”三个部分。

本轮明确排除 2025–2026 中以新 SID、order-agnostic ID、Term ID、continuous/parallel ID 或
vocabulary expansion 为核心的工作。它们可能在别的基座上有效，但与本地 FP2/FP3 已观察到的
lexical-ID replacement 大损失正面冲突；除非将来出现新的本地反证，不进入 Stage18。

## 2. 不得重复清单（Hard Exclusion List）

下列路径在 Stage18 全部禁止；修改名称不构成新方法：

1. 不替换 native lexical ID，不训练 PSID、semantic ID、continuous set ID 或 latent forest；
2. 不把 CF-only/beam200 候选在推理后通过新 gate、margin 或 top-k admission 插入 top10；
3. 不直接把 beam width 50 扩到 100/200/500 当作 treatment；宽 beam 只可生成训练期困难样本和诊断；
4. 不重跑 Phase10 C1/C2 的 cross-fitted calibrator，不降低 Phase11 margin，不增大 admissions；
5. 不重跑 A0 full-vocabulary proxy，不把 PAWA-lite 换一组深度权重再试；
6. 不同时叠加 LATTE、SETRec、PAWA、BEAR、TED 等多个模块；
7. 不读取已消耗的 Stage17 external D0 target 来选方法、loss weight、epoch 或 checkpoint；
8. 不再次读取 Toys/Beauty official validation/test，不读取 Sports；
9. 不因单 fold 弱正、某个 subgroup 正向或某个 secondary metric 正向而晋级；
10. 不用多 seed 作为挽救失败均值的手段；只有单 seed 主 Gate 已通过后才做稳健性确认；
11. 不允许效果失败后的自动重试。实现错误最多一次具名 correction，且必须证明未读取效果结果；
12. 不允许 `PCPS-GRAM` 自身候选覆盖不改善、仅靠末端 PCRF 偶然涨点后仍宣称机制成功。

## 3. 研究问题、假设与可证伪预测

### RQ1：历史失败是否集中在 lexical beam 的前缀剪枝？

在 D0 train-prefix 内部 rolling folds 上，目标存在于独立 beam200 但不在 beam50 的用户中，是否存在
稳定、可定位的 first-drop depth，并能找到 fold-local PCRF 选出的实际合法竞争前缀？

### RQ2：PCPS 是否改善生成，而不仅是重新排序？

相对 matched continuation，PCPS 必须提高 target-prefix survival 与 beam50 Hit@50；由于末端 PCRF
不会改变 Hit@50，这两个指标可以直接验证训练期机制。

### RQ3：PCPS + PCRF 是否超过已确认 PCRF，而不是只超过裸 GRAM？

主比较固定为：

```text
(PCPS-trained GRAM + frozen PCRF) - (matched continued GRAM + frozen PCRF)
```

若只超过裸 GRAM、不能超过 PCRF baseline，则 Stage18 不晋级。

### RQ4：信号能否跨 offset、跨域、跨 seed？

同一冻结配置必须依次通过 D0-train-only confirmation、Toys D1、Beauty D1，之后才允许 D2 多 seed。

### 可证伪预测

- P1：在 baseline beam200 可覆盖而 beam50 丢失的目标中，first-drop events 不是零，且 PCRF-guided
  hard-negative set 能覆盖足够多的实际 pruner；否则没有训练抓手；
- P2：PCPS 相对 generic legal-prefix survival 在相同预算下有额外 prefix-survival / Hit@50 收益；
- P3：PCPS 的 Hit@50 改善先于或伴随 top10 改善；若 Hit@50 不动而 top10 只微涨，不能声称突破 PCRF 上限；
- P4：主增益在未参与任何选择的 fold 上保持，不能重现 Phase11 `calibration 暴涨 → validation 归零`。

## 4. 方法定义：PCPS-GRAM

### 4.1 保持不变的强基座

- GRAM FiD encoder 与正常场景多 passage 输入；
- frozen native lexical item identifiers；
- frozen catalog mapping、EOS 与 legal Trie；
- beam50、length penalty 与统一 item-level evaluator；
- Phase9 PCRF 公式与 inference 行为；
- train-only item frequency、CF/item-head 与 leakage contract。

`alpha=0` 时，PCPS 实现必须与 matched continuation 在 loss、logits、beam ranking 和指标上退化等价。

### 4.2 Fold-local teacher 与样本边界

对每一个 rolling fold 单独训练/冻结：

- `G_parent`：只见该 fold cutoff 之前数据的 GRAM parent；
- `H_cf`：只见该 fold cutoff 之前 transition 的 item-head；
- `freq_i`、`q1`：只由该 fold cutoff 之前 item occurrence 计算；
- on-policy beam：由 `G_parent` 在 fold-train examples 上生成；
- target item：仅作为标准 supervised next-item label，不进入 teacher fit、frequency fit 或 negative cache fit。

严禁复用在更晚 offset 或完整 D0 train-prefix 上训练的 item-head，避免重现 Phase11 的
target-specific item-anchor shift。

### 4.3 合法前缀竞争集合

对用户 `u` 的真实 lexical path `y=(y_1,...,y_D)`，在每一深度 `d`：

1. 保留真实前缀 `y_{<d}`；
2. 从 locked Trie 取得该节点的合法 next-token 集；
3. 从 parent on-policy beam200 中提取在该节点与 target child 竞争的真实 pruner；
4. 若实际 pruner 少于 `K=8`，从每个合法 sibling 的 descendant items 中，按 fold-local
   popularity-corrected CF score 补足高难负例；
5. 不在全 vocabulary 计算 proxy rank，不把非法 token 纳入 denominator。

宽 beam 和 descendant 检索只构建训练样本；正式 inference 固定回 beam50，不做候选插入。

### 4.4 PCRF-guided prefix-and-path survival loss

只做逐 token 排序会接近 Stage17 A0 proxy，也没有处理 SIGIR 2025 揭示的完整 path/beam 局限。
因此 PCPS 同时包含 legal-prefix 与 full-path 两层，但仍保持为一个训练 hook。

令目标 child 为 `c*`，合法困难 sibling 集为 `H_{u,d}`，student next-token logit 为
`z_{u,d,c}`。前缀项为：

```text
L_prefix = mean_{u,d} a_{u,d} *
           log(1 + sum_{c in H_{u,d}}
               exp(m_{u,d,c} + z_{u,d,c} - z_{u,d,c*}))
```

对真实目标完整 lexical path `y*` 和 frozen parent 实际生成的错误 beam paths
`B_u^-`，用 teacher forcing 计算 student 的长度归一化累计 log-probability：
`S_theta(y)=log P_theta(y)/|y|^eta`，其中 `eta` 与冻结 inference length penalty 一致。
完整路径项为：

```text
w_model(u,j) = softmax_{j in B_u^-}(S_parent(y_j))
w_cf(u,j)    = 1 + reliability_u * sigmoid(cf_pc(y_j) - cf_pc(y*))
w(u,j)       = normalize_j(w_model(u,j) * w_cf(u,j))

L_path = mean_u log(1 + sum_{j in B_u^-}
                    w(u,j) * exp(m_path + S_theta(y_j) - S_theta(y*)))

L_PCPS  = 0.5 * normalized(L_prefix) + 0.5 * normalized(L_path)
L_total = L_CE + alpha * L_PCPS
```

设计含义：

- `L_prefix` 保护 target child 不在合法 Trie 分支处被过早剪掉，对应 BEAR 的 beam-aware 视角；
- `L_path` 直接比较完整 item path，避免把逐步 marginal 改善误写成最终 item 改善；
- lexical paths 若长度不同，必须使用同一冻结 `eta` 归一化；不得照搬 UGR 对等长 SID 的假设；
- `w_model` 让 frozen parent 越自信的错误 path 获得越强反馈，对应 UGR 的 confident-error 思路；
- `w_cf` 只在训练期提高 collaborative-confusing negatives 的难度，使用 Phase9 已冻结的 PCRF
  reliability/popularity correction；
- 两项先各自按用户归一化再等权，避免 identifier 长度、分支数或 candidate 数改变损失尺度；
- 不使用 RL/GRPO、DPO、confidence token、外部 LLM teacher 或 learned depth adapter；
- first-drop depth 只用于报告，不用于效果后调权。

最终精确 `normalized()` 定义、margin、截断范围、`K=8` 与 empty-beam fallback 必须在 S18-1
代码契约中写入机器可读 config；不得依据 confirmation accuracy 修改。

### 4.5 必要对照

| Arm | Training | Inference | Purpose |
|---|---|---|---|
| `C0_CONT` | matched GRAM continuation | native beam50 | 裸生成基线 |
| `C1_CONT_PCRF` | 与 C0 相同 | beam50 + frozen PCRF | **主基线** |
| `A0_LEGAL_GENERIC` | 相同 prefix+path loss，但令 `w_cf=1`、不读 CF | native beam50 | 排除只是 legal beam/path ranking 效应 |
| `M0_PCPS` | PCPS loss | native beam50 | 验证候选生成机制 |
| `M1_PCPS_PCRF` | 与 M0 相同 | beam50 + frozen PCRF | **主 treatment** |
| `S0_SHUFFLED_CF` | 只在 bounded probe 中打乱 user↔CF teacher | native beam50 | 排除额外 loss/计算量伪效应 |

`C1` 和 `M1` 只是 `C0/M0` prediction 的冻结 PCRF 视图，不产生额外训练 run。

## 5. 数据与防泄漏协议

### 5.1 已消耗与未消耗数据

| Data | Stage18 status | Allowed use |
|---|---|---|
| Stage17 D0 external target | **consumed / sealed** | 只能引用既有报告，不得重新评估或选参 |
| D0 shadow train slice | available | 构造内部 rolling folds |
| Toys D1 | unopened for Stage18 | 仅在全部内部 Gate 通过、配置 SHA 冻结后一次性打开 |
| Beauty D1 | unopened for Stage18 | 仅在 Toys D1 通过后一次性打开 |
| D2 | unopened | 仅在双域 D1 通过后做稳定性确认 |
| Toys/Beauty official validation/test | historically consumed/protected | 禁止读取 |
| Sports | protected | 禁止读取 |

唯一允许的 D0 输入是：

- `artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt`
- `artifacts/phase17/s0_audit/shadow_data/Beauty/D0/user_sequence.txt`

Stage18 loader 只能读取每行的 `shadow_items[:-2]`。`shadow_items[-2]` 是已消耗 external D0 target，
`shadow_items[-1]` 是 guard；两者都不得反序列化进 Stage18 internal runner。

### 5.2 D0-train-only rolling folds

令 `H_u = shadow_items[:-2]`：

| Fold | Model/teacher visible prefix | Target | Role |
|---|---|---|---|
| `I-1` | `H_u[:-4]` | `H_u[-4]` | 可作用性与 teacher 跨 offset 稳定性诊断 |
| `I0` | `H_u[:-3]` | `H_u[-3]` | 机制开发；只允许基于训练动力学冻结 alpha |
| `I1` | `H_u[:-2]` | `H_u[-2]` | 第一个 efficacy confirmation |
| `I2` | `H_u[:-1]` | `H_u[-1]` | 第二个 efficacy confirmation |

- 用户历史不足时在该 fold 排除，并完整报告 eligibility；不得以 treatment outcome 过滤；
- 每个 fold 的 parent、item-head、frequency、negative cache、checkpoint 必须独立构建；
- later fold 可按 rolling-origin 规则看到 earlier fold target，但不得复用 earlier treatment checkpoint；
- I-1/I0 看到的任何效果值都不能用于选择 margin、K、depth weight、beam width 或 epoch；
- `alpha` 只允许在 100-user gradient calibration 上从 `{0.1,0.3}` 中选择，使
  `||grad_PCPS|| / ||grad_CE||` 落入 `[0.10,0.30]`；若两者都不满足则停止，不补新值。

### 5.3 外部 D1/D2

- D1/D2 继续使用 Stage17 已冻结的 shadow projection 与 SHA；
- training job 只能看该 fold `shadow_items[:-2]`，evaluation controller 才可在 checkpoint 和 config
  SHA 冻结后一次性读取 `shadow_items[-2]`；
- external target 打开后禁止改方法、阈值、epoch、checkpoint selection 或 PCRF 公式；
- 一个 domain-fold 只允许一个 scientific attempt；基础设施失败按 attempt ledger 记录，恢复必须单独授权。

## 6. 分阶段低风险漏斗

### S18-0：历史证据与执行契约冻结（CPU）

目标：把本计划中的 success/failure matrix、数据封存、baseline identity 和禁止路径转成机器可检验合同。

拟建产物：

- `experiment/phase18/config/s18_evidence_contract.json`
- `experiment/phase18/config/s18_data_contract.json`
- `experiment/phase18/tests/test_s18_no_repeat_contract.py`
- `artifacts/phase18/s0_audit/summary.json`
- `report/第十八阶段/Stage18_S0_历史证据与执行契约报告.md`

Gate：所有历史数字可回溯；D0 external、D1/D2、official test、Sports deny-list 测试通过；
`C0 + PCRF` 在 Phase9 frozen cache 上可重建到 `1e-12` 排序口径。失败则停止，不进入 GPU。

### S18-1：可作用性与 first-drop 诊断（CPU + bounded generation）

固定 Toys/Beauty 各 1,024 名 eligible I-1/I0 用户，运行 parent beam50/200，只做诊断，不训练 treatment。

必须报告：

- beam200−beam50 target headroom；
- first target-prefix dropout depth 分布；
- actual pruner 的 legal-child coverage；
- `K=8` PCRF-guided hard negatives 对 actual pruner 的覆盖率；
- frozen parent 错误 beam 的完整 path probability、PCRF difficulty 与真实 target path 的 paired gap；
- target 与 hard negatives 的 fold-local CF/popularity margin；
- 每个统计的用户分母和 attrition。

预注册 Gate（两域分别满足）：

1. beam200−beam50 target headroom `>= +0.05`；
2. 至少 100 个 beam200-only target events，可避免极小样本决策；
3. 至少 50% first-drop events 能构造非空的合法 actual-pruner set；
4. PCRF-guided `K=8` 对 actual-pruner 的 recall `>= 0.50`；
5. target-prefix 与 hard-negative logits、teacher score 全部 finite，Trie legality=1；
6. fold-local item-head 的 target score 从 I-1 到 I0 diagnostic subset 不得重现 Phase11 量级的
   collapse：标准化均值漂移绝对值 `< 1.0`。I1/I2 target 在此阶段完全不可读。

任一域 Gate 1–5 失败：主假设 `NO_ACTIONABLE_PREFIX_BOTTLENECK`，Stage18 终止。
Gate 6 失败：`CF_TEACHER_UNSTABLE`，终止，不尝试去冗余或阈值修复。

### S18-2：100-user overfit 与 1k mechanism probe（单 GPU、短预算）

顺序固定：

1. tiny synthetic/unit contracts；
2. 100-user overfit：`C0/A0/M0/S0`；
3. I0 1k-user、固定 1 epoch：同四臂；
4. 只基于 gradient ratio 从 `{0.1,0.3}` 冻结 alpha，不按 NDCG 选值。

工程 Gate：

- `alpha=0` 与 C0 loss/logits identity tolerance `<=1e-6`，beam top50 exact match；
- 所有 generated paths legal，finite fraction=1；
- M0 的 `L_prefix`、`L_path` 和 target/negative gradient 均非零且方向正确；
- S0 shuffled teacher 与 M0 使用完全相同 batch、参数量和训练步数；
- 峰值显存、step time 和全链路预计 wall time 已实测。

机制 Gate（I0 1k，仅作淘汰，不作成功声明）：

- M0 相对 C0 的 beam50 target-prefix survival `>= +0.02`；
- M0 相对 C0 的 target-vs-confident-error full-path margin 为正向改善；
- M0 相对 C0 的 Hit@50 `>= +0.005`；
- M0 相对 S0 的 target-prefix survival `>= +0.01`；
- M0 NDCG@10 不低于 C0 超过 `0.001`；
- A0 若与 M0 相同或更好，记为 `CF_GUIDANCE_NOT_ADDITIVE` 并终止当前 PCPS 主线。

失败后只允许检查实现合同。若合同正确，立即止损；不得增加 epoch、beam、K、margin 或新 alpha。

### S18-3：D0-train-only 双 offset、双域确认

使用 S18-2 冻结配置与 fresh fold-local parents：

- I1：Toys + Beauty；
- I2：Toys + Beauty；
- 每个 domain-fold 只训练 C0 与 M0；C1/M1 从相应 prediction 确定性计算；
- checkpoint selection 只按 train-prefix internal loss/既有 GRAM patience，不能看 I1/I2 target 指标。

主比较为 `M1_PCPS_PCRF - C1_CONT_PCRF`。四个 domain-fold 合并 Gate：

1. macro `ΔNDCG@10 >= +0.0015`；
2. pooled paired bootstrap 95% CI lower `> 0`；
3. 至少 3/4 domain-fold `ΔNDCG@10 > 0`，且任何一格不得 `< -0.001`；
4. macro `ΔHit@10 >= 0`；
5. M0−C0 macro `ΔHit@50 >= +0.003`，且至少 3/4 格为正；
6. target-prefix survival macro `>= +0.02`；
7. head/mid/tail 与 short/medium/long 中，样本数 `>=500` 的组不得出现
   `ΔNDCG@10 < -0.003` 或 `ΔHit@10 < -0.005`；
8. M1 必须超过 C1；仅 M0 超过 C0 不足以晋级。

若 `0 < ΔNDCG@10 < 0.0015`、CI 跨 0 或只有 2/4 格为正，统一记为
`WEAK_SIGNAL_CLOSED`。不因“有一点提升”继续在 I1/I2 追调。

### S18-4：Toys D1 一次性独立准入

前置条件：S18-3 全 Gate 通过、代码/config/checkpoint-selection policy SHA 冻结、研究者另行授权。

Gate：

- M1−C1 `ΔNDCG@10 >= +0.0015`；
- paired 95% CI lower `>0`；
- `ΔHit@10 >=0`；
- M0−C0 `ΔHit@50 >= +0.003`；
- 无预注册大 subgroup catastrophe；
- 机制指标方向与 S18-3 一致。

任一主 Gate 失败：Stage18 终止；Beauty D1 与 D2 保持未开，不在 Toys D1 上修参数或重跑。

### S18-5：Beauty D1 跨域准入

只在 Toys D1 通过后解锁。配置、训练预算、alpha、K、margin、beam 与 PCRF 公式完全不变。

Gate 与 Toys D1 相同。若点估计为正但未达到 `+0.0015` 或 CI 跨 0，记为
`CROSS_DOMAIN_WEAK_NOT_PROMOTED`，不进入 D2。

### S18-6：D2 多 seed 稳健性确认

只在双域 D1 通过后解锁。seeds 固定为 2023/2024/2025，不用 seed 选择 checkpoint 或丢弃异常结果。

最终 Gate：

- 两域各至少 2/3 seeds 的 M1−C1 NDCG@10 为正；
- 两域 seed-mean `ΔNDCG@10 >= +0.0015`；
- 6 个单元 pooled paired CI lower `>0`；
- 两域 seed-mean Hit@10 不退化；
- 两域 seed-mean M0−C0 Hit@50 `>= +0.003`；
- 无大 subgroup catastrophe、无 illegal path、无 data-contract violation。

通过后才可称为 `PCPS-GRAM CONFIRMED`。本计划不再读取 official test。

## 7. 统计与判定协议

- paired unit：同一 domain/fold/seed 的同一用户；
- primary metric：NDCG@10；co-primary mechanism metric：M0−C0 Hit@50；
- 每次 confirmatory comparison 使用固定 seed 的 2,000 次 paired user bootstrap；
- 同时报告 point delta、95% CI、gain/loss/tie 与完整分母；
- macro 平均按 domain-fold 等权，pooled 结果按用户合并；两者都报告；
- Hit@50 与 prefix survival 不使用 PCRF 重排后的伪变化；用 M0 vs C0 原生 beam 检验；
- I0、1k probe、subgroup、first-drop depth 都是诊断，不可替代主 Gate；
- 不做 p-hacking 式多阈值、多 epoch、多 checkpoint、多 beam 报告；
- 所有预注册 arms 都进入报告，包括失败、崩溃和零变化。

## 8. 实现与代码边界

### 8.1 计划新增路径

```text
experiment/phase18/
  config/
  core/
    data_contract.py
    fold_local_teacher.py
    legal_prefix_miner.py
    pcps_loss.py
    evaluator.py
  protocol/
  tests/
  run_stage18_s0_audit.sh
  run_stage18_s1_actionability.sh
  run_stage18_s2_mechanism_probe.sh
  run_stage18_s3_internal_confirmation.sh
  run_stage18_s4_toys_d1.sh
  run_stage18_s5_beauty_d1.sh
  run_stage18_s6_d2_multiseed.sh
```

可复用但不得静默修改语义的已有组件：

- `experiment/phase17/core/loss_hooks.py`
- `experiment/phase17/core/generation_hooks.py`
- `GRAM/src/utils/generation_trie.py`
- Phase9 frozen PCRF evaluator/formula；
- Stage17 status、attempt ledger、resource telemetry 与 leakage guard。

不得直接改写 Phase9 canonical artifact；若抽取公共模块，先用回归测试证明原 PCRF 排序 exact identity。

### 8.2 测试最低集合

- alpha-zero degeneration；
- legal-child mask 与 illegal-token exclusion；
- target child/negative gradient direction；
- per-user/per-depth normalization；
- variable-length lexical path score 与 inference length penalty identity；
- shuffled teacher determinism；
- fold cutoff 与 forbidden target denial；
- parent/item-head/frequency fold-local SHA identity；
- beam50 baseline identity；
- PCRF exact reconstruction；
- duplicate item/path aggregation；
- NaN/Inf、empty sibling、short history、single-child prefix；
- external target single-read controller；
- no-auto-retry attempt ledger。

## 9. 资源与运行治理

### 9.1 后台运行与研究者自助观察

- 预计或实测 wall time `>10 分钟` 的任何任务必须进入具名 `tmux` 后台 session；
- agent 启动并完成 startup handshake 后即交还控制，不实时轮询、不在线 babysit，也不要求为实验持续占用会话；
- 研究者通过 `artifacts/phase18/status/<experiment_id>.status.json` 观察状态；需要进一步诊断时再明确要求 agent 检查；
- 后台 runner 自己负责原子更新 status，heartbeat 间隔不超过 60 秒；长 epoch 内也要写存活 heartbeat；
- status 至少包含：`scientific_state`、`execution_state`、`status_code`、`stage`、`progress`、
  `heartbeat_at`、`updated_at`、`tmux_session`、launcher/workload PID、physical GPU IDs、log/result 路径、
  input/config SHA、D0/D1/D2/test/Sports read flags；
- `status` 是观察入口，详细 stdout/stderr 写 `run.log`，不得把大量日志塞入 status；
- 任意成功、失败、OOM、中断或被授权停止的终态都必须落盘；进程消失但 status 仍为 RUNNING 时，
  后续 `status` 命令必须识别为 stale/failed，不得伪装仍在运行。

### 9.2 小实验 GPU 规则

- S18-0 为 CPU only；S18-1 bounded generation、S18-2 的 100-user/1k probe 属于小实验；
- 小实验无需逐次申请 GPU，可在启动前读取一次当前 GPU 状态，直接选择任意没有明显冲突且预计不 OOM 的卡；
- 不设置 GPU utilization 百分比门槛，不要求保留 18/30 GiB 等过度保守 headroom；判断标准仅为
  “已有进程不受影响，当前剩余显存足以完成本次 workload”；
- 若无历史 peak，先跑最小 smoke 得到实测；必要时降低 micro-batch 并用 accumulation 保持有效 batch，
  这属于资源适配，不得改变样本、loss、epoch 或评价口径；
- 不终止、暂停、迁移其他用户进程；出现竞争或 OOM 风险时换卡或等待；
- 当前任何 `RUNNING_OCCUPANCY_REPEAT` 且 heartbeat 新鲜的卡都视为已占位，不作为小实验首选。

### 9.3 大实验申请与并行策略

- S18-3 及之后的 full-fold/full-data training 默认为大实验；启动前必须向研究者报告并申请资源；
- 申请内容必须包含：训练 run 数、每 run GPU 数、实测单卡 peak、最小可运行 GPU 数、建议 GPU 数、
  预计串行/并行 wall time、拟使用的物理卡、以及科学结束后拟保留的占位卡；
- 优先把互相独立的 arms、domains、folds 或 seeds 分配到多张 GPU 并行，以缩短总 wall time；
- 并行 run 必须共享同一冻结 config/SHA 和 cohort contract，不能因不同卡临时改变 batch、epoch 或 checkpoint policy；
- 若模型本身没有经过验证的 DDP 路径，优先采用 arm/fold-level 并行，不为“多卡”临时改训练语义；
- 研究者只授权当前步骤的资源；S18-3 通过不自动授权 S18-4，上一 Gate 失败则不申请下一步。

S18-2 结束后才生成正式资源申请表，避免用未实测数字预占共享服务器。

### 9.4 大实验结束后的重复占位

- 大实验所有科学 arms 达到终态并写完 canonical summary 后，从该步骤获批 GPU 中选一张启动重复轮占位；
- 资源申请时优先写明拟留哪张卡；若运行期间卡况变化，可选另一张获批且足以不 OOM 的卡，并在 status 记录原因；
- 占位重复必须与科学结果严格隔离：`scientific_state=COMPLETED`、
  `execution_state=RUNNING_OCCUPANCY_REPEAT`、`affects_scientific_result=false`、
  `result_selection_eligible=false`、`repeat_metrics_ignored=true`；
- 每轮使用 fresh CUDA child process，重复输出写独立 `run-NNNN/`，不得覆盖 canonical result；
- 占位轮同样在后台运行，由 runner 自更新 heartbeat；agent 不实时监看；
- 不得自动停止当前已有的其他占位轮，也不得把重复轮指标用于挑 checkpoint、seed 或报告效果；
- 若研究者之后要把占位卡交给新大实验，必须通过具名 stop/handoff 流程，先写 STOPPED/HANDED_OFF 状态再复用。

当前已知参考：

- `artifacts/phase17/status/s17_fp12_external_d0_g1_guard_v3.status.json`
- 该文件在 v0.2 最终核验时显示 `scientific_state=COMPLETED`、
  `execution_state=WAITING_FOR_GPU`、`target_gpu_id=4`、`process_alive=true`；这表示科学结果已经完成，
  占位守护进程仍存活并等待继续在 GPU4 运行，不代表核验瞬间已有 workload 占用 GPU4；
- 后续以该动态 status 为准：只要 guard 的 `process_alive=true`、heartbeat 新鲜，且 execution 处于
  `WAITING_FOR_GPU` 或 `RUNNING_OCCUPANCY_REPEAT`，Stage18 都不停止该进程，并把物理 GPU4 视为已预约，
  不选作小实验或新大实验用卡；只有完成显式 stop/handoff 后才可复用。

### 9.5 审计与安全

- 显存报告同时保留整卡 observed used memory 与 workload allocated/reserved，二者不得混写；
- 禁止终止其他用户进程；无资源时等待，不通过降低 scientific gate 或减少必需 arms 规避申请；
- 任意 attempt 都写入 `artifacts/phase18/attempts/*.attempts.jsonl`；科学状态与资源/占位状态分离；
- 不自动重试失败 scientific attempt；基础设施 correction 与外部数据重开均须单独授权；
- 无研究者新授权时，runner 不得从一个 Stage 自动进入下一个 Stage。

## 10. 产物与报告合同

每一步必须有唯一 canonical `summary.json`、逐用户 TSV/Parquet、config SHA、input SHA、attempt ledger、
status 和一份中文报告。建议路径：

```text
artifacts/phase18/s0_audit/
artifacts/phase18/s1_actionability/
artifacts/phase18/s2_mechanism_probe/
artifacts/phase18/s3_internal_confirmation/
artifacts/phase18/s4_toys_d1/
artifacts/phase18/s5_beauty_d1/
artifacts/phase18/s6_d2_multiseed/
artifacts/phase18/status/
artifacts/phase18/attempts/
artifacts/phase18/authorizations/
report/第十八阶段/
```

报告必须明确区分：

- `ENGINEERING_PASS`
- `MECHANISM_PASS`
- `WEAK_SIGNAL_CLOSED`
- `FAILED_SCIENTIFIC_GATE`
- `CONFIRMED`

“训练完成”“loss 下降”“机制激活”都不能写成 accuracy success。

## 11. 决策树

```text
历史/数据合同通过？
  否 -> 关闭 Stage18
  是 -> 两域存在可作用 prefix bottleneck，且 fold-local CF teacher 稳定？
          否 -> 关闭 Stage18，不训练
          是 -> 100-user/1k 中 PCPS 超过 generic/shuffled 并改善 Hit@50？
                  否 -> 关闭 PCPS，不调参
                  是 -> I1/I2 双域强 Gate 通过？
                          否 -> WEAK_SIGNAL_CLOSED 或 FAILED；D1 不开
                          是 -> Toys D1 一次性通过？
                                  否 -> 关闭；Beauty D1/D2 不开
                                  是 -> Beauty D1 一次性通过？
                                          否 -> 关闭；D2 不开
                                          是 -> D2 三 seed 稳健性确认
```

## 12. 成功、弱正与失败的最终定义

### 成功

只有 S18-6 全 Gate 通过，才能声称：PCPS 在不替换 GRAM lexical identifier 的情况下，稳定改善
原生 beam coverage，并在 PCRF 之上提供跨域、跨 seed 的 NDCG@10 增益。

### 弱正

点估计大于 0 但小于 `+0.0015`、CI 跨 0、或只在部分 fold 为正，一律记录为真实的
`weak signal`，但不继续消耗新外部数据。弱正可以进入论文 negative/ablation discussion，不能通过
后验调参升级为成功。

### 失败

Stage18 的某一假设失败不等于“第十七阶段失败”或“整个研究失败”。它只表示：在已经确认有效的
PCRF 上，当前这条训练期前缀扩展没有产生足够强、足够稳定的增量。由于本计划的漏斗在 D1 前设置了
三层止损，大多数错误假设应在 CPU/bounded/内部 folds 阶段被淘汰，而不是再次花费完整 full-port 预算。

## 13. 论文与机制来源边界

- GRAM：semantic-to-lexical translation 与 multi-granular late fusion，ACL 2025：
  <https://aclanthology.org/2025.acl-long.1596/>
- Constrained Auto-Regressive Decoding：constraint / beam marginal limitation，SIGIR 2025：
  <https://arxiv.org/abs/2504.09935>
- UniGRF：ranking-driven generation enhancement 与 gradient-guided weighting，SIGIR 2025：
  <https://arxiv.org/abs/2504.16454>
- DMRec：collaborative-language distribution matching，SIGIR 2025：
  <https://doi.org/10.1145/3726302.3730098>
- LOHRec：ordered candidates、hierarchical generation 与 path-level learning-to-rank，EMNLP 2025 Findings：
  <https://aclanthology.org/2025.findings-emnlp.977/>
- BEAR：beam-search-aware target-prefix survival regularization，SIGIR 2026：
  <https://arxiv.org/abs/2601.22925>
- UGR：uncertainty-weighted confident-error feedback，KDD 2026：
  <https://arxiv.org/abs/2602.11719>
- KDD 2026 Generative Recommendation tutorial：tokenization / architecture / optimization 三轴综述：
  <https://applied-machine-learning-lab.github.io/KDD2026_GenRec_Tutorial/>

PCPS-GRAM 不是上述任一方法的复现。它借鉴 legal-prefix survival 的优化视角，但其研究假设、
full-path ranking、confident-error weighting、fold-local PCRF teacher、collaborative hard-negative mining、
lexical-anchor 与止损协议的组合来自顶会证据和本项目 Phase9–11/Stage17 的共同约束。任何后续论文
表述必须保持这一命名与归属边界，不得把局部借鉴写成 1:1 reproduction。

## 14. 当前停点

本计划已完成历史复核与方案设计，但尚未实施。下一步若研究者明确同意执行，只启动 `S18-0` 的
CPU 合同与证据审计；S18-1 及其后的任何 GPU/数据步骤均不自动解锁。
