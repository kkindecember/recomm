# GRAM 第四阶段：CF-SAT 方法创新与渐进实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: experiment plan
- Origin Date: 2026-07-27
- Verification Status: ANALYZED（既有实验与代码）；PREREGISTERED（C0）
- Version Label: `code_plan_v4_rpcd_t0_proposed`
- Archived Direction: PCSA（未执行；因缺少 span/coalition 前提证据而降为备选）
- Upstream Evidence: `artifacts/phase3/marc_l0/summary.json`

### Amendment Record

- 2026-07-27：CF-SAT C0 已完成。Toys/Beauty 的全部 cohort、serialization、
  donor/target exclusion、K/overlap、真实 Collator mask、metadata start、Trie、
  finite、no-update 与 parameter-SHA integrity gate 均通过。clean 相对等预算错误
  CF 的 user-level margin 在两域均明显为正，但 node-level helpful CF coverage
  仅为 Toys 48.15%、Beauty 31.91%，均低于预注册 60%；Beauty sensitivity-deficit
  rate 16.02% 也低于 20%。固定决定为
  **`STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL`**，C1/C2/C3 不解锁。
- 第一次完整运行已写出同一 STOP 决定，但因 sandbox 看不到宿主 tmux/PID，被误判为
  异常退出并手工恢复资源；随后启动了完全相同配置的 deterministic 重跑。重跑再次
  得到同一决定并正常恢复资源。当前 summary 来自第二次运行；这属于执行 lineage
  复核，不是独立环境验证，也不用于修改门槛或救援方向。

## 1. 本阶段保留的事实与方向选择

第三阶段目前唯一跨 Toys/Beauty 稳定成立的正信号是 collaborative evidence：

| 已有结果 | 本阶段约束 |
|---|---|
| CF sample utility 正比例：Toys 85.94%，Beauty 84.57% | 主方向应增强真实 CF 的利用，而不是主要学习删除 CF |
| CF critic AUROC：Toys 0.8703，Beauty 0.9507；active utility CI 均大于 0 | CF 对 gold Trie decision 的效用具有稳定结构 |
| semantic corruption、Beauty semantic coverage、Beauty dynamic-K transport 失败 | 不再做 semantic/CF 统一 controller，也不做动态 K |
| CPBD、FFNF 暴露 128-token 字段预算冲突 | 正式推理不改变输入 token、顺序、K、长度和 Trie |

此前 PCSA 要求单 neighbor 同时存在大量 helpful/harmful spans，并进一步要求可预测的
非加性 pair interaction；这些尚无直接证据，而且形成多层串行失败风险。因此 PCSA
不作为主方向，只有未来 C0 额外发现强邻居级异质性时才可重新立项。

## 2. 新方向：CF-SAT

**Counterfactual Collaborative Sensitivity Alignment Training**
（反事实协同证据敏感性对齐训练）

核心研究问题：

> GRAM 虽然把 collaborative neighbors 序列化进输入，但普通 target CE 并不显式要求
> 模型区分“真实 CF 关系”和“格式、预算、位置相同但关系错误的 CF 证据”。能否利用
> training-only paired counterfactuals，对真实 CF 在每个合法 lexical-ID prefix 上的
> 生成敏感性进行对齐，并在完全不改变正式推理路径的条件下改善推荐？

对同一个 training-prefix sample 构造：

- \(x^+\)：matched GRAM clean input；
- \(x^-\)：只把每个 history item 的 CF neighbor list 换成确定性的错误 donor list；
- donor 与 clean 保持相同 K、CF 可见 token 数、metadata 起始位置、总 attention-mask
  长度和 passage 顺序，并排除 target、anchor 自身及高重合 donor；
- validation/test、beam hit/miss 和 target identity 均不得作为正式推理 feature。

在 gold Trie prefix \(y_{<t}\) 上定义：

```text
cf_utility(t) = log p(y_t | x+) - log p(y_t | no-CF)
cf_margin(t)  = log p(y_t | x+) - log p(y_t | x-)
```

若 C0 通过，训练目标暂定为：

```text
L = L_clean_trie_ce
  + alpha * sum_t w_t * max(0, margin_t - cf_margin(t))
  + beta  * L_clean_anchor
```

其中 `w_t` 只允许由 training-only frozen counterfactual advantage 构造；正式推理只
运行 clean GRAM，因此不增加模型参数，不需要 critic，也不增加推理分支。

候选创新表述必须收窄为：

> 在固定检索、固定序列化和固定 Trie 的生成式推荐中，以等预算 collaborative
> counterfactuals 对真实 CF 证据沿 lexical-ID prefixes 的生成敏感性进行
> advantage-gated alignment，同时保持推理路径与 GRAM 基线一致。

不声称首次使用 collaborative contrastive learning、counterfactual learning、
prefix supervision 或 corruption augmentation；新颖性在正式论文前仍需全文复核。

## 3. C0：0-training collaborative sensitivity premise audit

### 3.1 目的与固定结论空间

C0 只回答三个问题：

1. frozen GRAM 是否在 gold Trie path 上区分真实 CF 与等预算错误 CF；
2. 已知 CF 有帮助的节点中，是否仍存在足够多“没有正确区分真实/错误 CF”的
   sensitivity deficit；
3. clean–corrupt 差异是否可排除 token 数、字段位移、target 泄漏等机械解释。

C0 不训练、不生成 beam、不读取 validation/test，也不声明 Recall/NDCG 效果。
固定结论只能是：

- `CFSAT_C1_DESIGN_ALLOWED`；
- `STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL`；
- `STOP_CFSAT_NO_TRAINABLE_DEFICIT`；
- `EXECUTION_INVALID`。

### 3.2 数据、模型与样本

- 数据集：Toys、Beauty；
- checkpoint 与 matched K 沿用 MARC L0：Toys K5、Beauty K10；
- 每个数据集固定 512 个 users，继续使用 `sequence[-3]` 作为 training target、
  `sequence[:-3]` 作为 history；
- 使用与 MARC 相同的 hash split：fit 307 / calibration 103 / audit 102；
- 主结论只使用 audit users；fit/calibration 仅保留给未来 C1，不参与 C0 调参；
- batch size=8，GRAM optimizer steps=0，beam generation=0。

### 3.3 等预算 collaborative corruption

对每个 history anchor item：

1. clean passage 必须与当前 `item2input` 逐字符串一致；
2. 计算 clean CF segment 在 GRAM delimiter filtering 后的可见 token 数；
3. donor 必须具有相同 K 和相同 CF 可见 token 数；
4. donor item、donor neighbors 均不得等于当前 target，donor 不得等于 anchor；
5. donor neighbor set 与 clean neighbor set 的 Jaccard overlap 必须 ≤0.20；
6. 候选以 `SHA256(seed,dataset,user,anchor,target,donor)` 排序，取最小值；
7. 若无合格 donor，该 sample 不得静默放宽规则，记为 rejected；
8. clean/corrupt 经真实 Collator 后，每个 passage 的 attention-mask length 必须一致，
   metadata 起始位置必须一致，序列顺序和 passage 数必须一致。

corrupt input 只替换 CF neighbors，保留 anchor lexical ID 和原 metadata。不得使用
随机删除、K 改变、字段重排或 validation-derived hard negative。

### 3.4 输出指标

逐 node 保存：

- `lp_clean`、`lp_corrupt`、`lp_no_cf`；
- `cf_utility`、`cf_margin`；
- dataset、user、split、depth、Trie child count；
- history length、clean/corrupt visible length、donor overlap；
- target exclusion、serialization replay、mask/metadata-position identity 标记。

逐 user 先对非 EOS nodes 求均值，再在 audit users 上做 10,000 次 user-cluster
bootstrap。不得把同一 user 的不同 depths 当成独立样本计算主置信区间。

定义：

```text
helpful node      := cf_utility > 0
sensitivity deficit := helpful node AND cf_margin < 0.10
```

`0.10` 为运行前冻结的 natural-log probability margin，不因结果修改。

### 3.5 完整性门槛

两个数据集均必须满足：

1. exact cohort caps = 307/103/102，跨 split user overlap=0；
2. held-out fields、validation/test、beam hit/miss 均未读取；
3. clean serialization replay=100%；
4. target/donor exclusion=100%，K identity=100%，donor overlap gate=100%；
5. clean/corrupt passage attention-mask length identity=100%；
6. clean/corrupt metadata start identity=100%；
7. Trie gold-child membership=100%，finite rate=100%；
8. optimizer steps=0，checkpoint 参数 SHA256 前后不变。

任一失败即 `EXECUTION_INVALID`，不得解释科学指标。

### 3.6 科学晋级门槛

两个数据集必须同时满足：

1. audit user-level mean `cf_margin > 0` 的 bootstrap 95% CI lower >0；
2. audit users 中 mean `cf_margin > 0` 的比例 ≥55%；
3. helpful node rate ≥60%；
4. helpful nodes 中 sensitivity-deficit rate ≥20%；
5. deficit 覆盖至少 30% audit users；
6. depth 0 之外至少两个 non-trivial depths 各有 ≥50 个 helpful nodes；
7. `cf_margin` 不能由 clean/corrupt token 数差解释，因为该差必须恒为 0。

若 1–3 任一失败：
`STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL`。

若 1–3 通过但 4–6 任一失败：
`STOP_CFSAT_NO_TRAINABLE_DEFICIT`。

全部通过才是 `CFSAT_C1_DESIGN_ALLOWED`。不得降低 0.10 margin、20% deficit、
30% user coverage 或双数据集要求。

## 4. 后续阶段（仅保留轮廓）

### C1：correctness smoke

32–64 个 training samples；验证 paired loss、advantage gate、clean anchor、有限梯度、
checkpoint reload 和 clean inference identity。只有 clean CE 改善而非仅 corrupt CE
恶化才允许进入 pilot。

### C2：10% pilot

固定比较 matched GRAM continuation、等计算量 clean CE、普通 corruption
augmentation、无 advantage gate paired loss、完整 CF-SAT。完整方法必须在
Toys/Beauty 同时改善 NDCG@10、Recall@10 无实质下降，并超过全部机制对照。

### C3：fresh-dataset confirmation

配置在读取 Sports 效果前冻结；Sports 为主要确认集，Beauty/Toys 只作重复验证。
正式报告 Recall/NDCG、head/tail、broad harm、训练开销及完全不变的推理开销。

## 5. 当前状态

当前状态：**`STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL`**。

C0 完整性通过，但双数据集均未达到 helpful-node coverage 必要门槛，Beauty 还未达到
deficit-rate 门槛。按预注册停止规则，不实现 CF-SAT 训练模块，不启动 C1/C2/C3，
不因 clean–corrupt margin 为正而降低 60% helpful coverage 或只选择 Toys 晋级。

## 6. 新候选方向：RPCD

**Rank-to-Path Collaborative Distillation**
（从商品排序到 lexical path 的协同蒸馏）

### 6.1 为什么这次改变问题

现有失败共同说明：

1. 修改、删除、重排 CF 文本会受到 128-token 字段竞争和 prefix 异质性影响；
2. 高 AUROC critic 不等于可兑现的生成收益：用 MARC critic 在 audit nodes 上选择
   full/no-CF 后，Toys/Beauty Trie-local CE 分别恶化约 2.53%/1.55%；
3. `similar_item_sasrec.txt` 只是静态 item–item top-20，不是根据完整用户历史预测
   next item 的 teacher；LRC 的 relation-pool target prevalence 也只有 Toys 10.43%、
   Beauty 8.84%；
4. 因此不再从文本邻居或 sign gate 榨取小收益，而先训练真正的 user-conditioned
   SASRec teacher，并要求它先在最终 ranking 指标上与 GRAM 形成可验证互补。

RPCD 的方法单位不是 CF span，而是 teacher 的完整 catalog item distribution。
对用户历史 \(h\)，SASRec 给出 \(q(i\mid h)\)。对于当前 lexical Trie prefix \(p\)
及合法 child \(c\)，定义 mass-conserving teacher：

```text
q_path(c | p, h)
  = sum_{i in subtree(p+c)} q(i | h)
    / sum_{j in subtree(p)} q(j | h)
```

学生仍使用原 GRAM 输入、identifier、Trie 和 beam；只在训练期增加合法 children
上的 teacher distribution：

```text
L = L_gram_ce + alpha * w(h,t) * KL(q_path || p_gram)
```

`w(h,t)` 只允许由 training-only teacher quality/entropy 和 gold-rank 构造。正式推理
不运行 SASRec，不增加输入 token、参数或延迟。

### 6.2 新颖性边界

- UniGRec 已用 SASRec item embeddings 对 tokenizer/recommender 表示做 collaborative
  distillation；RPCD 不声称首次从 SASRec 蒸馏；
- UNGER/PRORec 已做 semantic/collaborative code integration 与知识蒸馏；
- TrieRec 已向生成推荐 Transformer 注入 Trie topology；
- RPCD 只保留以下 search-bounded 候选差异：把 user-conditioned full-catalog ranker
  分布按既有 GRAM lexical Trie 的 subtree probability mass 精确投影为每个 prefix 的
  合法-child 教师，并在完全不改 identifier/tokenizer/inference 的条件下蒸馏。

正式论文前必须对 UniGRec、UNGER、TrieRec 与 generative-retrieval distillation 做
全文级差异复核，不能把 SASRec teacher、KL 或 Trie supervision 单独声称为创新。

### 6.3 T0：先证明 teacher/hybrid 真有最终指标收益

这是当前唯一建议执行的步骤。T0 不修改或训练 GRAM。

#### A. 训练 user-conditioned teacher

- 在 Beauty/Toys 原始 `user_sequence.txt` 上重新训练同结构 SASRec；
- 输入只到 `sequence[:-2]`，`sequence[-2]` 为 validation target，
  `sequence[-1]` test 不读取；
- 两域使用同一结构和超参数：hidden=64、blocks=2、heads=2、maxlen=50、
  dropout=0.2、seed=2023；
- checkpoint selection 只使用 training-prefix hash calibration，不以 Beauty/Toys
  validation 反复选 epoch；
- 保存每个 validation user 的 top-50 item、logit、rank 和完整 lineage。

现有静态 `similar_item_sasrec.txt` 不能作为 teacher 输出，也不能用其 target coverage
救援 teacher qualification。

#### B. 先做 hybrid effect gate

使用已有 matched GRAM validation beam-50 与 SASRec top-50：

1. 候选固定为二者并集；
2. 两路分数先在各自 top-50 内作固定 rank/robust normalization；
3. 融合权重网格在 hash calibration 20% users 上选择；同一个权重同时用于
   Toys/Beauty audit 80%，不得分别调参；
4. audit 主比较为 matched GRAM，不读取 test；
5. Beauty/Toys 已被多次分析，因此这里只是 development evidence；最终确认必须使用
   冻结配置的 Sports。

双数据集必要门槛：

1. GRAM/SASRec top-50 union Recall 相对 GRAM beam-50 绝对增加 ≥3 个百分点；
2. 在 GRAM miss@10 users 中，SASRec hit@50 rate ≥10%；
3. locked hybrid NDCG@10 相对 GRAM 提升 ≥1%；
4. locked hybrid Recall@10 不下降；
5. tail NDCG@10 不下降超过 0.5% relative；
6. target、history、catalog mapping、分数、候选去重和 test exclusion 全部通过。

任一数据集失败即 `STOP_RPCD_NO_TEACHER_COMPLEMENTARITY`，不得实现蒸馏。
全部通过才是 `RPCD_T1_DESIGN_ALLOWED`。

### 6.4 后续仅保留轮廓

- T1：验证 full-catalog probability 到 Trie child mass 的守恒、gold path、有限梯度和
  zero-weight GRAM identity；
- T2：10% pilot，比 matched continuation、普通 item-embedding distillation、
  top-K pseudo-label 和完整 RPCD；
- T3：配置冻结后在 Sports 做主要确认，再报告 Beauty/Toys repeated-validation。

### 6.5 首次工程运行记录（不构成科学结果）

2026-07-27 用户确认后启动 T0。数据预检、target 对齐与 test exclusion 均通过，
但 SASRec 从第 1 epoch 起出现非有限 loss，两个数据集的训练指标固定在随机水平。
根因定位为 PyTorch 1.11 中“左 padding + causal mask + key-padding mask”使部分
attention query 的全部 key 被遮蔽，NaN 经后续层传播。

因此首次输出中自动生成的 `STOP_RPCD_NO_TEACHER_COMPLEMENTARITY` **作废**，不得引用
为方法证据。原产物完整保留在
`artifacts/phase4/rpcd_t0_invalid_nan_20260727_123239/`。修复只把序列改为右 padding
并按实际长度抽取最后 hidden state，不改变预注册数据、模型容量、训练超参数、融合
网格或科学门槛；同时增加非有限 loss 立即失败和 padding 回归测试。

当前状态：**`RPCD_T0_ENGINEERING_FIX_VERIFIED_AWAITING_RERUN`**。5/5 单元测试与
独立 finite forward/backward 检查通过；尚无有效 T0 科学结论，也未解锁 T1。

### 6.6 修复后正式 T0 结果

2026-07-27 按相同配置完成有效重跑：10 个 epoch 的 loss 均有限并持续下降，共享
training-prefix calibration 选择 epoch 8；validation calibration 选择双域共享融合
权重 0.2。GPU3 在任务结束后恢复成功。

Audit 80% users 的结果：

| 数据集 | union Recall@50 绝对增益 | miss@10→SAS hit@50 | hybrid NDCG@10 相对增益 | Recall@10 绝对增益 | tail NDCG@10 相对增益 |
|---|---:|---:|---:|---:|---:|
| Toys | +3.172pp | 5.699% | +0.339% | +0.084pp | -2.484% |
| Beauty | +3.032pp | 6.296% | +0.203% | +0.039pp | -3.379% |

两域只有 union Recall@50 与 Recall@10 nondecrease 通过；miss recovery、NDCG 主门槛
和 tail harm 门槛均失败。paired bootstrap 95% CI 显示两域 NDCG 增益均跨 0，而
tail 相对下降区间均完全低于 0。结论不是“完全没有协同信息”，而是“协同 teacher
提供了约 3pp 的新候选覆盖，但原始 SASRec 分布的 popularity bias 无法通过统一
rank fusion 转化为安全的 top-10 收益”。

当前状态：**`STOP_RPCD_NO_TEACHER_COMPLEMENTARITY`**。按预注册不实现原 RPCD
full-distribution path distillation，不降低门槛、不分别为两域选权重。

## 7. 下一候选：PRPD

**Popularity-Residual Path Distillation**（流行度残差路径蒸馏）

RPCD-T0 唯一值得保留的新证据是双域 union Recall@50 稳定增加约 3pp；明确失败模式
则是 tail NDCG 显著下降。下一方向不再蒸馏原始 SASRec 分布，而只研究 teacher 相对
training-only item popularity 先验的用户条件残差：

```text
q_res(i | h; gamma) ∝ q_sas(i | h) / p_train(i)^gamma
```

若残差 teacher 通过 effect gate，再把 `q_res` 而不是 `q_sas` 按 lexical Trie
subtree mass 投影。该方向只能把“残差化 + user-conditioned Trie path projection +
inference-free student”作为组合候选差异；流行度校正、SASRec、KL 和 Trie 各自都
不是新贡献。

### 7.1 R0：唯一建议的第一步

R0 是 CPU-only、冻结输出的 effect gate，不训练或修改 GRAM/SASRec：

1. 输入固定为本次有效 T0 的 GRAM beam-50、SASRec top-50/logit 和
   `sequence[:-2]` training popularity；
2. 候选仍是两路 top-50 并集；先将 SASRec logit 转为用户内 rank score，再减去
   training popularity percentile，构造 `q_res` 的排序代理；
3. `gamma ∈ {0, 0.25, 0.5, 0.75, 1.0}`，融合权重仍为
   `{0.0, 0.1, ..., 1.0}`；只在原 hash calibration 20% 上按双域共享配置选择，
   audit 80% 一次性读取；
4. 主门槛为双域 audit NDCG@10 相对 GRAM 均 ≥1%、Recall@10 均不下降、tail
   NDCG@10 均不下降；同时报告 head/tail、union coverage 和 paired bootstrap CI；
5. `gamma=0` 必须精确复现 RPCD-T0 rank fusion，target/test exclusion、候选去重、
   mapping 和共享配置必须通过。

任一域失败即 `STOP_PRPD_NO_DEBIASED_EFFECT`；只有全部通过才允许设计 Trie residual
mass projection。

### 7.2 后续轮廓

- R1：验证 residual distribution、Trie subtree mass 守恒和 zero-weight identity；
- R2：10% pilot，对比原始 RPCD、普通 popularity reweight 与 PRPD；
- R3：冻结配置后在 Sports 做确认。

当前状态：**`PRPD_R0_PROPOSED_AWAITING_CONFIRMATION`**。

### 7.3 R0 正式结果

2026-07-27 完成 55 个共享配置的 CPU-only 正式扫描，输入、target、test exclusion
和 1,100 次 `gamma=0` 排名 identity 检查全部通过。只有五个 `weight=0` 的 identity
配置同时满足 calibration 上双域 Recall 与 tail nondecrease；所有真正启用 residual
teacher 的配置至少违反一项约束。按 tie-break 锁定
`gamma=0, weight=0`，audit 两域所有增益均精确为 0。

Calibration 上最接近可用的是 `gamma=0.25, weight=0.1`：

- Toys/Beauty NDCG@10 相对增益为 +0.162%/+0.404%；
- Recall@10 均小幅上升；
- tail NDCG@10 仍下降 -0.044%/-0.135%；
- broad gain 远低于双域 +1% 门槛。

未经残差化的 `gamma=0, weight=0.1` 虽有 Toys/Beauty +0.581%/+1.583% calibration
NDCG 增益，但 tail 分别下降 -0.336%/-0.903%，且 RPCD audit 已显示该类配置不能
稳定泛化。更强的残差化开始改善 tail 时，Recall 与 broad NDCG 转为明显下降。

当前状态：**`STOP_PRPD_NO_DEBIASED_EFFECT`**。不设计 residual Trie projection，
不放松 tail nondecrease，不用 calibration 的 Beauty 单域增益救援。

## 8. 下一候选：CPGV

**Collaborative Proposal with Generative Verification**
（协同提议与生成验证）

T0 已证明 SASRec 能在 GRAM beam-50 外稳定补回约 3pp gold candidates；R0 则证明
仅靠全局 rank/popularity arithmetic 无法安全排序。下一步不再继续设计标量 fusion，
而直接问一个尚未回答的机制问题：**这些被 SASRec 找回、但被 beam search 漏掉的
gold lexical paths，冻结 GRAM 的 exact teacher-forced path likelihood 能否识别？**

若 exact likelihood 能识别，失败主要来自 constrained beam 的搜索/候选覆盖，而不是
GRAM 表示完全不懂该商品；后续可在训练期用协同 teacher 提议 candidate、用 GRAM
验证并蒸馏 search-error correction，正式推理仍只保留 GRAM。若不能识别，则约 3pp
union coverage 只是外部 ranker 信号，不能沿当前模型兑现。

### 8.1 V0：唯一建议的第一步

V0 是冻结模型的机制诊断，不训练参数、不调融合权重：

1. cohort 固定为 audit 80% 中 `gold ∉ GRAM top50` 且
   `gold ∈ SASRec top50` 的全部 users，并按 training popularity 报 head/tail；
2. 对每个 user 固定候选为 SASRec top-50，使用 matched GRAM checkpoint 和原始
   GRAM input，对每条候选 lexical ID 计算 length-normalized exact teacher-forced
   log-likelihood；禁止把 target 放入 input；
3. 主要指标为 gold 在 SASRec proposal 内经 GRAM exact score 重排后的 Recall@10；
   次指标为 MRR、相对 SASRec 原 rank 的提升、gold-vs-nongold pairwise concordance；
4. Toys/Beauty 必须各有至少 300 个 eligible users，mapping/score finite/target
   exclusion 为 100%；
5. 双域 exact-rescore Recall@10 必须 ≥25%，且相对原 SASRec proposal Recall@10
   绝对提高 ≥5pp；head/tail 分别报告，不以某一组救援整体失败。

失败即 `STOP_CPGV_GRAM_CANNOT_VERIFY_PROPOSALS`；通过才允许 V1 在完整 target-free
union 上做一次 final-metric effect gate。

### 8.2 后续轮廓

- V1：完整 audit 的 target-free proposal + exact verification，要求双域 NDCG +1%；
- V2：只在训练期生成 proposal/verification targets，蒸馏到 GRAM path margin；
- V3：冻结配置后在 Sports 确认，并与 candidate generation、reranking、原 RPCD 对照。

当前状态：**`CPGV_V0_PROPOSED_AWAITING_CONFIRMATION`**。

### 8.3 V0 正式结果

2026-07-27 完成 493 个 Toys 和 544 个 Beauty eligible users 的冻结 exact scoring，
共评估 51,850 条 candidate paths。mapping、finite、Trie membership、target
exclusion 和 no-update 均为 100%，GPU3 已恢复。

| 数据集 | SASRec proposal R@10 | GRAM exact-rescore R@10 | 绝对差 | pairwise concordance |
|---|---:|---:|---:|---:|
| Toys | 25.963% | 22.110% | -3.854pp | 0.580 |
| Beauty | 20.772% | 9.926% | -10.846pp | 0.506 |

Toys exact Recall@10 未到 25%，且没有比 SASRec 提高 5pp；Beauty 更明显恶化，
paired bootstrap 的 Recall 差 95% CI 为 [-14.890pp, -6.618pp]。Beauty
pairwise concordance 95% CI [0.487, 0.526] 覆盖随机 0.5。两域均失败。

eligible targets 几乎全是 head：Toys 472/493，Beauty 536/544。tail 子组只有 21/8，
即使点估计显示 exact rank 偶有改善，也不能用极小、target-selected 子组救援整体
失败或声称 tail 机制成立。

当前状态：**`STOP_CPGV_GRAM_CANNOT_VERIFY_PROPOSALS`**。不进入完整 union exact
rescoring，不实现 proposal-verification distillation。

## 9. 下一候选：FCRD

**Full-Catalog Residual Distillation**
（全目录协同残差蒸馏）

R0 的一个结构性限制现在必须纠正：它只在已经截断的 SASRec top-50 内减 popularity，
因此无论 gamma 多大，都不可能找回原 top-50 外的 tail items。V0 又显示 GRAM 不能
替外部 proposal 做可靠验证。因此下一项不再重排旧 top-50，也不依赖 GRAM verifier；
而是在取 top-K **之前**对 SASRec full-catalog logits 做 residualization：

```text
log q_res(i | h; gamma)
  = logit_sas(i | h) - gamma * log p_train(i)
```

这会真正改变 proposal candidate set。只有它能在增加 tail candidate coverage 的同时
兑现最终 NDCG，才有理由把 full-catalog residual mass 投影到 Trie 并蒸馏。

### 9.1 F0：唯一建议的第一步

1. 冻结 RPCD-T0 的 epoch-8 SASRec checkpoint，对每个 validation history 计算完整
   11,924/12,101 item logits；seen history items 继续屏蔽；
2. `p_train(i)=(count_train(i)+1)/sum_j(count_train(j)+1)`，在 full catalog 上固定
   `gamma ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}`，每个 gamma 重新取 top-50；
3. 与 GRAM beam-50 合并后，融合权重仍为 `{0.0,0.1,...,1.0}`；同一个
   `(gamma,weight)` 在原 calibration 20% 上选择并锁定到双域 audit 80%；
4. `gamma=0` 必须精确复现 RPCD teacher top-50；test 不读取，SASRec/GRAM 均不训练；
5. 双域 audit 必须同时满足：overall union Recall@50 绝对增益 ≥3pp、tail union
   Recall@50 绝对增益 ≥1pp、NDCG@10 相对提升 ≥1%、Recall@10 不下降、tail
   NDCG@10 不下降。

失败即 `STOP_FCRD_NO_FULL_CATALOG_RESIDUAL_EFFECT`；通过才允许设计 full-catalog
residual mass 到 Trie child 的蒸馏。该 F0 才是对 PRPD 假设的充分 effect gate，
R0 只排除了 post-top50 arithmetic。

### 9.2 后续轮廓

- F1：验证 full residual probability 与 Trie subtree mass 守恒；
- F2：10% student pilot，对比 raw RPCD、post-top50 PRPD 与 FCRD；
- F3：冻结配置后在 Sports 做确认。

当前状态：**`FCRD_F0_PROPOSED_AWAITING_CONFIRMATION`**。

### 9.3 F0 正式结果

F0 已按锁定命令完成，preflight、user/catalog/target lineage、`gamma=0` top-50
恒等性和 test exclusion 全部通过。77 个共享配置中 calibration-qualified 配置为
**0**，因此按 fail-closed 规则锁定 `gamma=0, weight=0`。

关键结果不是“full catalog 没起作用”，而是它暴露了明确的 coverage trade-off：

| gamma（weight=0，calibration） | Toys overall / tail union 增益 | Beauty overall / tail union 增益 |
|---:|---:|---:|
| 0.0 | +3.359pp / +0.137pp | +3.052pp / +0.045pp |
| 0.3 | +3.256pp / +0.366pp | +2.984pp / +0.226pp |
| 0.5 | +2.972pp / +0.503pp | +2.803pp / +0.542pp |
| 1.0 | +2.222pp / +1.830pp | +1.876pp / +1.625pp |

小 gamma 保住 overall coverage 时无法恢复足够 tail；大 gamma 达到 tail +1pp 时又使
overall 跌破 +3pp。故 PRPD 失败不能再归因于 top-50 截断，统一 popularity residual
也不能进入 Trie projection 或 student 蒸馏。

当前状态：**`STOP_FCRD_NO_FULL_CATALOG_RESIDUAL_EFFECT`**。

## 10. 下一候选：CCRR

**Candidate-Conditional Residual Reranking**
（候选条件化跨空间残差排序）

RPCD/FCRD 已经稳定证明 SASRec 能在 GRAM beam-50 之外补充约 3pp 候选覆盖；原始
融合失败的共同点，是每个用户、每个候选都使用同一个全局权重。CPGV 又证明 GRAM
exact likelihood 不能直接充当 verifier。因此下一步不再设计第三个手工融合公式，
而只检验一个更窄的问题：

> 仅使用推理时可得的 GRAM、SASRec、训练流行度和用户历史特征，候选级模型能否学习
> 哪一路分数在当前用户—候选对上更可信，并把已有 union coverage 兑现为双域 NDCG？

这与 LRC-UCRF 不同：LRC 预测“历史 CF 邻居集合是否覆盖目标”的用户级 reliability；
CCRR 直接在已经生成的 GRAM∪SASRec 候选上学习候选级相关性和跨空间交互。它也不改
identifier、Trie、GRAM/SASRec checkpoint 或候选集合。

### 10.1 R0：唯一建议的第一步

1. **冻结数据与候选。** 复用 RPCD epoch-8 的 raw SASRec top-50 和 GRAM beam-50，
   候选严格固定为二者并集；不使用 FCRD gamma 网格，不重新训练任何神经模型。
2. **固定 target-free 特征。** 每个 user-candidate 只允许使用：GRAM/SASRec
   in-list 标志、rank、reciprocal rank、SASRec 用户内标准化 logit、两路 rank
   差/一致性、training popularity percentile、head/tail 标志、历史长度，以及候选
   是否出现在历史中。特征函数不得接收 target、audit label 或 test。
3. **固定 split。** 沿用现有 SHA-256 calibration 20% / audit 80%；calibration
   target 只用于拟合候选标签和选择模型，audit target 只在模型与 tie-break
   完全锁定后评测。两个数据集分别拟合参数，但必须共享同一特征 schema、模型类别和
   超参数；不允许 dataset-specific 搜索。
4. **只比较三项。** `B0=GRAM rank`、`B1=RPCD 固定 rank fusion`、
   `R1=class-balanced logistic residual`。为避免把大模型容量当机制，本轮不扫描
   GBDT/MLP，不做超参数网格；logistic 的正则、class weight、标准化和 tie-break
   在运行前写死。
5. **校准门槛。** R1 在两域 calibration 都必须满足：相对 B0
   NDCG@10 ≥+1%、Recall@10 不下降、tail NDCG@10 不下降；否则直接
   `STOP_CCRR_NO_CANDIDATE_CONDITIONAL_EFFECT`，不得查看 audit 后改模型。
6. **audit 晋级门槛。** calibration 通过后，锁定 R1 到 audit；两域仍须同时满足
   NDCG@10 ≥+1%、Recall@10 不下降、tail NDCG@10 不下降，并优于 B1。
   paired bootstrap 只描述用户抽样不确定性；test 始终不读。
7. **完整性门。** 必须验证候选集合逐用户不变、feature/label 分离、audit rows
   未进入 scaler/fit、未知/重复 item 为 0、两域 recipe 完全一致、全指标有限。

R0 是一次零 GPU、冻结模型的可学习性兼 effect gate。若失败，说明约 3pp 互补候选在
现有轻量跨空间特征下仍不可兑现，应停止整个 SASRec 协同修补家族；不再尝试 GBDT、
MLP 或更多融合网格。

### 10.2 后续轮廓

- R1：仅在 R0 双域通过后，设计如何把候选条件化 residual 蒸馏进 Trie child；
- R2：10% student pilot 与组件消融；
- R3：冻结后在未参与方向生成的数据集确认。

当前状态：**`CCRR_R0_PROPOSED_AWAITING_CONFIRMATION`**。

### 10.3 R0 正式结果

CCRR-R0 在 4/4 单元测试和完整 preflight 后完成。两域 logistic 均收敛，候选集合
identity 100%，fit 只包含 calibration users，audit rows used for fit=0，test 未读。

| 数据集 | R1 vs B0 NDCG@10 | Recall@10 绝对增益 | tail NDCG | R1 vs B1 NDCG |
|---|---:|---:|---:|---:|
| Toys | +6.305% | +0.594pp | +9.193% | +5.843% |
| Beauty | +11.147% | +0.949pp | **-3.795%** | +8.687% |

该结果很重要：候选级模型能大幅兑现 overall union signal，说明此前并非“完全没有可
学习效果”；但 Beauty tail 与 overall 方向相反。按 conjunctive calibration gate，
`calibration_qualified=false`，audit 保持未读，也不得追加 tail weight、GBDT/MLP
或事后改门槛。

当前状态：**`STOP_CCRR_NO_CANDIDATE_CONDITIONAL_EFFECT`**。至此停止 SASRec
协同修补家族。下一周期必须允许改变 GRAM 表示、训练目标和推理候选结构。

## 11. 结构性转向：GCDH

**Generative–Catalog Dual Head**
（生成路径—全目录双头推荐）

此前方向确实过度受限：identifier、Trie、decoder 与输入布局长期被冻结，训练大多是
零更新诊断或小比例修补。CCRR 现在提供了足够强的正证据支持扩大改动：总体排序信号
可以学，但尾部信号不能只靠冻结候选上的轻量融合保住。

GCDH 不再使用 SASRec teacher。它在 GRAM 原 lexical-ID decoder 之外增加一个
full-catalog item head：

```text
h_user = masked_mean(encoder_hidden_states of coarse-history passage)
z_item = W_item h_user
L = L_lexical_path + lambda_item * BalancedSoftmax(z_item, target_item)
```

生成头继续保留语义路径和可解释 lexical ID；catalog head 直接对 11,924/12,101 个
item 提供不经过单路径 Trie 的监督与候选。训练时 Balanced Softmax 使用
training-prefix item frequency，推理时用未加频率项的 raw catalog logits。这样改变
的不是一个融合权重，而是表示空间、监督粒度和候选生成路径。

### 11.1 P0：唯一建议的第一步——25% 真实训练 pilot

1. **模型改动。** 在 `GRAM` 中加入 catalog embedding/head；只池化第一个
   coarse-history passage 的 encoder states，严格按 attention mask mean pooling。
   item head 权重由各 item lexical identifier 的 token embedding mean 初始化；
   原 decoder、Trie、item prompt 和输入序列保持不变，便于归因。
2. **真实训练而非离线拟合。** 从原 best checkpoint 继续训练，使用固定 hash
   分层的 25% training users、5 epochs、一个 seed。`C0` 为完全相同用户/step 的
   lexical-CE continuation；`C1` 为 lexical CE + catalog Balanced Softmax。
   两域使用同一 `lambda_item=0.2`，不扫描权重、不做 dataset-specific 调参。
3. **候选与排序。** C1 推理同时取得 GRAM beam-50 和 catalog-head top-50；候选为
   两者并集。主排序固定为 catalog raw logit，GRAM sequence score 只作为同分
   tie-break；同时报告 generator-only、head-only 和 union oracle，不能隐藏任一头。
4. **一次性 pilot cohort。** 在每域 validation 中用新 salt 固定 4,096 users，并按
   head/tail 与历史长度分层；不得使用既有 CCRR calibration/audit 标签挑样本。
   test 不读取。
5. **必要效果门槛。** 相对 matched C0，两个数据集都必须满足：
   NDCG@10 ≥+2%、Recall@10 不下降、tail NDCG@10 ≥+2%；catalog-head union
   Recall@50 相对 GRAM 至少增加 2pp。任一域失败即
   `STOP_GCDH_NO_DUAL_HEAD_EFFECT`。
6. **训练与实现门。** loss/logit/gradient 全部有限；item head 非零梯度；
   coarse-passage mask 与 catalog target 对齐 100%；C0/C1 用户、batch、step、
   optimizer 和 wall-clock 记录完整；peak reserved 增幅不得超过 35%。
7. **解释边界。** P0 是 architecture effect pilot，不是论文确认。Beauty/Toys 已
   参与多轮方向生成；通过后仍须在未参与方向生成的数据集和至少 3 seeds 确认。

这一步明确接受比此前更大的改动和训练成本，但仍用 matched continuation 防止把
“多训练 5 epochs”误当作双头机制收益。

### 11.2 后续轮廓

- P1：P0 通过后做 full-user、3-seed 训练与生成头/catalog 头消融；
- P2：独立数据集确认、效率和 calibration 分析；
- P3：新颖性边界与论文表述收窄。

当前状态：**`GCDH_P0_PROPOSED_AWAITING_CONFIRMATION`**。
