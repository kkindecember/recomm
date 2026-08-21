# GRAM 第十四阶段：R² 突破与 ColdPath 蒸馏探索计划 v0.1

> **建立日期**：2026-08-19
> **当前状态**：`PLAN_ONLY / NO_PHASE14_EXPERIMENT_STARTED`
> **工作名**：R²-to-Path Distillation（R2PD，暂定名，不作为最终论文命名）
> **阶段目标**：保留 R² 已验证的冷物品可恢复性，把它从外部候选 portfolio 迁移成 GRAM 原生、collision-safe 的冷路径生成能力，并显式控制 warm forgetting。

---

## 0. 先回答最重要的问题

### 0.1 R² 能不能作为论文里的一个点？

**能。** 现有证据支持把它写成以下两类贡献之一：

1. **诊断性贡献**：原始 GRAM 对零交互 cold item 的 native top-50 reachability 只有 Toys `1.03%`、Beauty `1.31%`；同一历史经 warm-only content resolver 后可达性为 `11.40% / 11.03%`。这证明问题不是用户历史完全没有 cold preference 信息，而是信息没有进入 GRAM 的可生成路径。
2. **方法性基线/第一模块**：冻结规则 `portfolio@2` 在 Toys/Beauty validation 上分别取得 overall NDCG@10 `+4.96% / +4.15%`、cold H@50 `2.89× / 2.49×`，代价是 warm NDCG@10 `−4.09% / −5.26%`。它是可复现的 Pareto 点，不是“什么都没跑出来”。

但 R² 目前不够单独撑起强方法论文，原因也必须诚实写出：

- 通过点主要来自一个简单 external resolver + fixed portfolio；
- 单 split、单 resolver seed、validation only；
- warm 损失是真实且显著的；
- 当前 resolver top-50 仍漏掉约 `89%` 的 cold target；
- AAAI 2026 的 SpecGR 已发表 draft-then-verify inductive generative recommendation，单纯把 resolver 与 GRAM 组合起来的新颖性空间已很小。

因此第十四阶段的合理定位不是丢掉 R²，而是：

> **R² 是“冷物品在内容空间可恢复”的证据和教师；第十四阶段方法回答“怎样让这种可恢复性进入 GRAM 的生成路径，同时不牺牲 warm”。**

### 0.2 本阶段只研究一个核心问题

> 冷 item 从 GRAM 的真实训练 target 中被完全移除后，能否仅依靠 catalog metadata、warm interaction 与 R² teacher，把冷 item 的 unique lexical path 变成有概率支持的 decoder path，并在推理时由 GRAM 原生 beam 命中？

这比“再提高 resolver 2 个点”更接近生成推荐论文的核心，也比“再做第 9 个 allocator”更可能形成机制贡献。

---

## 1. 已有证据：哪些是真的，哪些已经被否定

### 1.1 可继续使用的正证据

| 证据 | Toys | Beauty | 可支持的结论 |
|---|---:|---:|---|
| v0 GRAM cold H@50 | 1.03% | 1.31% | native cold path 基本坍塌 |
| resolver cold Recall@50 | 11.40% | 11.03% | 内容可归纳专家存在约 9–11 倍 reachability |
| R² `portfolio@2` overall NDCG@10 | +4.96% | +4.15% | 双域同方向 overall gain |
| R² `portfolio@2` cold H@50 | 2.89× | 2.49× | 冷侧收益可复现 |
| R² `portfolio@2` warm NDCG@10 | −4.09% | −5.26% | 收益伴随真实 warm trade-off |
| learned allocation vs matched random | 显著更优 | 显著更优 | 学习型 allocation 不是完全无效，但不是当前主瓶颈 |

### 1.2 已经被否定或必须降级的路线

| 路线 | 结论 | 第十四阶段处理 |
|---|---|---|
| 原 v1 Semantic Bridge raw ID | 增益主要来自 lexical-ID collision/alias | 禁止作为正方法恢复 |
| collision-safe v1 | Toys/Beauty 正式 FAIL | 仅作为“路径无监督”的反证 |
| v2 LLM prior / v3 semantic-collaborative alignment | 双域显著退化；语义空间与 collaborative hierarchy 不匹配 | 禁止继续做 hard semantic target/alignment |
| depth-3 route fusion | depth 3 已近 item identity，压制 resolver 正确项 | 禁止重新调 RRF 权重救援 |
| P1–P7 / CBSA allocator | 未突破 frozen R²；主要受候选池与稀疏正例限制 | 不做第 9 个 allocator |
| extra epochs / static hard negatives | resolver recall 不升反降 | 不把常规训练调参作为主线 |
| pseudo-cold setwise selector | pseudo-cold 有微增益，真实 cold 降到 resolver top-1 的 0.558× | 新伪监督必须先通过 item-disjoint transfer Gate |
| Tier-1 RRF / recall-then-place | post-processing 层饱和；GRAM score加入后 cold 变差 | 不再做冻结双路后处理搜索 |

### 1.3 T1-4 多兴趣 Resolver 的位置

T1-4 代码和冻结协议属于第十三阶段尚未完成的 closure。它若补跑，可回答“简单多兴趣表示能否提高 resolver recall”，但无论 PASS/FAIL：

- 不能成为第十四阶段主创新；
- 不能自动触发更多 resolver 变体；
- 不影响 R2PD 的可行性判断；
- 最多成为 resolver ablation 或 teacher-strength sensitivity。

---

## 2. GRAM 原论文与原始代码：真正可以下手的位置

### 2.1 原论文中的结构约束

原始 GRAM（ACL 2025）由三块组成：

1. hierarchical semantic indexing：item text embedding 经层级 k-means 形成 coarse-to-fine hierarchy，再翻译成现有 T5 vocabulary 的 lexical IDs；
2. collaborative semantics verbalization：用 collaborative neighbor 的 lexical IDs 丰富 item prompt；
3. multi-granular late fusion：coarse user-ID prompt 与多个 fine item-text prompts 分别编码，在 decoder cross-attention 处融合并生成下一个 lexical ID。

论文假定每个 item 有 unique ID；但本项目的 v1 实验已经证明，实际 cold-ID 重映射若产生重复 lexical path，字符串级 evaluator 会把同路径 item 当成命中。因此第十四阶段所有方法必须使用 item-level unique path 与 item-level evaluator，不能只比较生成字符串。

### 2.2 原始训练为何天然看不到 cold target

以 vendored 原始提交 `2062dbb` 为审计基线：

- `GRAM/src/data/multi_task_dataset_rec.py::load_train` 对每个用户只读取 `sequence[:-2]`，并构造 `A→B, AB→C, ...` 的 prefix augmentation；
- cold split 在构建数据集时已经把 cold item 从 train prefix 全部删除；
- 因此 cold item 不仅没有真实 interaction label，也从未作为 decoder target 出现；
- 现有 trie 只保证 beam 的下一 token 合法，不能给未训练路径增加概率；
- 原 evaluator 比较 decoded lexical-ID string，而不是反解后的唯一 catalog item。

这与第十三阶段的观测完全吻合：v0 GRAM 只能偶尔命中共享/已支持路径，不能稳定生成 zero-interaction item 的完整 unique path。

### 2.3 最小实现接口

第十四阶段不先改 hierarchy，不改 prompt，不改 encoder。最小落点为：

| 位置 | 原职责 | phase14 最小扩展 |
|---|---|---|
| `GRAM/src/data/multi_task_dataset_rec.py` | warm next-item CE 样本 | 保持冻结；phase14 在外层 dataset/wrapper 生成 teacher distributions，不原地污染原数据类 |
| `GRAM/src/processor/Collator.py` | `target_ids` tokenization/mask | 新 collator 额外携带 sparse prefix targets、sample confidence 与 provenance |
| `GRAM/src/model/gram.py` | T5 forward/generate | 优先在 phase14 wrapper 计算 prefix distillation；不直接耦合现有 phase9/12 扩展 |
| `GRAM/src/runner/single_runner_gram.py` | `loss=model(...labels=target_ids)[0]` | 新 phase14 runner 组合 CE、cold-path KD 与 warm retention loss |
| `GRAM/src/utils/generation_trie.py` | legal-token mask | 增加只读 subtree index/descendant mass 工具，不修改合法性语义 |
| `GRAM/src/utils/evaluate.py` | lexical string match | 新 item-level evaluator：unique reverse map、ambiguous path hard-fail、stable unique top-K |

实现纪律：新代码放 `experiment/phase14/`；若最终必须改 GRAM 内部，应以最小 patch 和 feature flag 实现，并先保存 original/current behavior parity test。

---

## 3. 顶会文献给出的边界

### 3.1 最接近工作的 WHY / HOW / WHAT

| 工作 | WHY：解决什么 | HOW：核心做法 | WHAT：对本阶段的约束/启发 |
|---|---|---|---|
| GRAM, ACL 2025 | LLM token 与 item 语义/协同关系未充分结合 | lexical hierarchy + collaborative verbalization + late fusion | 原 backbone；缺少 zero-interaction path supervision |
| SpecGR, AAAI 2026 | GR 几乎不能生成 unseen items | inductive drafter 提候选，GR teacher-forcing likelihood 验证，guided redrafting | R² 的最近邻；“drafter+verifier”不能再当新贡献 |
| AGRec, Findings ACL 2025 | autoregressive decoder 难利用图协同信号、beam prefix 同质化 | GNN logits 增强 + rankable FSM | 单纯在 decoding logits 加 resolver prior 已经拥挤，只适合强 baseline |
| SETRec, SIGIR 2025 | 顺序 ID 有 beam local optimum 与 token dependency | order-agnostic semantic+CF set identifiers、并行生成 | 说明 factorized/independent code 有潜力，但需要大改 GRAM 且赛道已拥挤 |
| DIGER, SIGIR 2026 | tokenizer 重构目标与推荐目标错位 | differentiable SID + Gumbel exploration/decay | end-to-end 改 ID 很昂贵，不能作为第一轮低风险方案 |
| Cold-Starts in Generative Recommendation, SIGIR 2026 | 既有 cold 结论受模型规模、ID、训练策略混杂 | 统一 cold protocol 和 factor-wise controls | 要求本阶段一次只改一个因素，并报告 warm/cold/overall |
| Can Generative Recommendation Reach Cold Items?, arXiv 2026 | SID 可组合不等于能到达未来 item | temporal split、coldness taxonomy、oracle-prefix probing | 直接支持“unsupported path”诊断；但目前是预印本，应标注非同行评审 |
| DSI++, EMNLP 2023 | 新文档写入生成索引会遗忘旧文档 | pseudo-query memory + rehearsal / flatter minima | 冷路径注入必须配 warm replay/retention，否则 warm 下跌不是偶然 |
| USIM, NeurIPS 2024 | OOV item 只有 content embedding、没有用户交互优化 | RL 想象用户序列并反向优化 OOV embedding | 普通“给冷 item 合成用户序列”已不是空白；本阶段不走 RL imagination |
| ALDI, SIGIR 2023 | cold/warm item 需在同一榜单竞争，分布错位会相互伤害 | ranking/rating/identity aligning distillation | warm-cold score calibration 必须显式评估 |
| UCC, SIGIR 2023 | 生成冷交互可改善 cold，但可能伤 warm | uncertainty-aware interaction generation + consistency | teacher confidence 可用，但不应直接照搬图模型结构 |
| CCFCRec, WWW 2023 | content embedding 缺少 warm collaborative signal | content/co-occurrence 双分支 contrastive transfer | 可作为备选 item adapter，不宜硬对齐 GRAM lexical hierarchy |
| CGRC, SIGIR 2024 | cold node 没有图边，不能用高阶协同 | mask warm item edges，学习重构伪冷连接 | 支持 item-disjoint masked simulation；同时提醒随机伪冷与真实 cold 的 domain gap |

### 3.2 文献过滤后的结论

以下看似自然的论文故事已经不够新：

- “content retriever 提 cold candidate，GRAM 排序”——SpecGR；
- “把 auxiliary score 加到 decoder token logits”——AGRec；
- “想象冷 item 会被哪些用户消费”——USIM；
- “重新设计 semantic ID 以利于 cold”——SETRec、DIGER、Term-ID GRAM 等大量工作；
- “多兴趣 resolver 提召回”——是检索模块优化，不是 GRAM 冷路径机制。

仍有可讲空间、且与本项目证据最贴合的方向是：

> **不改变 GRAM identifier，不在推理期永久依赖外部 portfolio；把 R² 的 item distribution 精确投影为 trie 上的 prefix-conditional soft targets，让 cold path 得到分层支持，再用 frozen-v0 retention 防止 warm forgetting。**

这与 SpecGR 的 inference-time candidate verification、AGRec 的 inference-time logit fusion、DSI++ 的 query replay均不同，但最终是否足够新颖，必须以实现和实验结果再判断，当前不能预先宣称。

---

## 4. 候选突破方向排序

### 4.1 Priority A：R²-to-Path Distillation（主线，推荐）

#### 核心思想

给定用户历史 `h`，R² resolver 在全 catalog 上产生 item distribution `q(i|h)`。对 collision-safe unique path `z(i)=(z1,...,zL)`，把 item probability 汇总成每个 trie prefix 的下一 token 分布：

```text
Q(a | p, h) = Σ_{i: z(i) starts with p+a} q(i|h)
               ------------------------------------------------
               Σ_{j: z(j) starts with p} q(j|h)
```

然后让 GRAM decoder 在相同 history 和 prefix 下拟合 `Q`，而不是只用一个 noisy top-1 cold item 做 hard CE。

暂定损失：

```text
L = L_warm_CE
  + λ_cp · c(h) · L_subtree_KD
  + μ_keep · L_frozen_v0_retention
```

- `L_warm_CE`：原始 warm next-item 监督，保持推荐任务锚点；
- `L_subtree_KD`：R² item mass 投影后的逐层 soft-target KL/CE；
- `c(h)`：由 teacher entropy/margin 得到的置信权重；
- `L_frozen_v0_retention`：在 warm histories 上匹配冻结 v0 的 token/path distribution，防止注入 cold path 时 catastrophic forgetting。

#### 它针对了哪条本地反证

- 原 v1 只改变 ID，没有给 cold unique path 训练支持；R2PD 直接改变 path probability。
- v2/v3 把 semantic embedding 硬对齐 collaborative hierarchy；R2PD 不规定“某个语义向量必须等于某条协同路径”，只蒸馏当前用户条件下的 item mass。
- P5 pseudo-cold top-1 selector迁移失败；R2PD 用 soft distribution 和 confidence，不依赖单点伪标签。
- R² portfolio 通过但 warm 下跌；retention 项直接把 warm protection放进训练目标，而不是事后挪两个 slot。

#### 最大风险

1. teacher recall 只有 11%，学生上限可能受限；
2. R² distribution 的 calibration 可能很差，soft mass 只是把噪声扩散到更多路径；
3. arbitrary collision-safe suffix 很难由共享规律泛化，可能只能靠 catalog-specific onboarding memorization；
4. full GRAM fine-tuning可能再次伤 warm；
5. 若最终只有推理期 external reranking 有效，方法会退化成 SpecGR/AGRec 邻近工作。

因此必须先做 Section 6 的低成本 oracle/teacher-forcing Gate，不能直接跑 30 epoch 双域。

### 4.2 Priority B：Candidate-Scored R² / SpecGR-style verifier（必须做的强对照，不作为主创新）

对 R² top-M candidate 用 GRAM teacher-forcing 计算 normalized path likelihood；对 cold item 忽略或单列纯 identification suffix 的概率，再与 resolver score 作无需训练或 source-only calibration 的组合。

用途：

- 检验 GRAM 是否至少能在“候选已给出”的条件下区分 cold items；
- 给 R2PD 提供一个 inference-time upper/control baseline；
- 若此对照已解决问题，则没有必要先重训 GRAM。

限制：结构与 SpecGR 非常接近，不能把“GRAM verifier + R² drafter”当原创主方法。

### 4.3 Priority C：Mask-and-Reconstruct Cold Onboarding（备选）

在 warm train 内做 item-disjoint episodic masking：把一组 warm item 的全部 interaction target 暂时遮掉，模拟 catalog-known zero-interaction item；从 metadata 和剩余 warm graph/history 学习恢复其用户条件 path distribution。

只在以下条件下进入：

- R2PD teacher mass不足，但 pseudo-cold item-disjoint Gate 显示可迁移；
- Section 6 证明真实 cold 与模拟 cold 的 metadata/frequency/path 支持分布没有灾难性偏移。

它受 CGRC、USIM 启发，也受到本项目 P5 pseudo-cold→real-cold 失败的直接警告。不得使用 actual cold validation target interaction 生成训练样本。

### 4.4 Priority D：Factorized / Order-agnostic Cold ID（高成本保留方向）

若 oracle-prefix 诊断显示：coarse prefix 已支持，但目标总在深层 sequential pruning 中丢失，可考虑把唯一标识拆为相互弱依赖的 semantic dimensions，或引入独立 identification head。

不优先的原因：

- 会同时改 tokenizer、ID、decoder 与 evaluator，难以归因；
- SETRec、DIGER、SIGIR 2026 reproducibility study 已把 compositional/independent IDs 变成明显赛道；
- 第十三阶段 ID-heavy v1–v3 已消耗大量实验且连续失败。

### 4.5 Priority E：继续提高 resolver recall（只作 teacher sensitivity）

多兴趣、可学习 item adapter、content-to-collaborative contrastive transfer可能抬高 teacher ceiling，但它们只能作为 `q(i|h)` 的替换组件。除非新 resolver 在双域 cold Recall@50 相对 frozen R² 至少提升 30%，并同步改善最终 native generation，否则不单独发展成主故事。

---

## 5. 论文假设与可证伪预测

### H1：瓶颈是 unsupported decoder path，而不只是 item representation

**预测**：对 cold target 做 oracle teacher-forcing 时，v0 的 target path NLL 在某个深度出现显著断崖；该深度的 prefix survival 与 cold hit 强相关。

**证伪**：cold target path NLL 与 warm 接近，但 beam 仍失败，则主要问题是 search/competition，不应重训 path。

### H2：R² 的 item mass 可以变成有效的 prefix supervision

**预测**：在 item-disjoint pseudo-cold audit 中，soft subtree target 比 top-1 hard path CE 提高 exact path rank/survival，且不依赖 target 是否在 teacher top-1。

**证伪**：soft target 只降低 teacher loss，却不提高 held pseudo-cold exact item；停止 full GRAM。

### H3：warm forgetting 可以通过 retention 明显缓解

**预测**：加入 frozen-v0 retention 后，warm NDCG retention 从 `<95%` 回到 `≥97%`，同时保留至少 `95%` 的 cold gain。

**证伪**：在合理 `μ_keep` 下 warm/cold 仍严格跷跷板；不要继续密集搜权重，应转 factorized interface 或承认 Pareto 极限。

### H4：R2PD 能突破 R²，而不只是模仿 R²

“突破”必须至少满足一个匹配代价条件：

- 在 cold H@50 不低于 `0.95×R² portfolio@2` 时，warm NDCG retention 比 R² 高至少 2 percentage points；或
- 在 warm retention 不差于 R² 时，cold H@50 或 overall NDCG@10 的 paired-bootstrap 95% CI 下界高于 R²；或
- native GRAM 在不使用 inference-time external portfolio 的情况下达到 R² 的同等级 cold gain，并保持 overall > v0。

只做到“比 v0 好”不叫突破 R²，只能叫 native recovery。

---

## 6. 分阶段执行计划

### Stage 14-0：口径与可行性审计（CPU/小 GPU，先做）

#### 14-0A：collision-safe item-level evaluation contract

产出：

- item→path 与 path→item reverse map；
- duplicate path、ambiguous decoding、unknown item、top-K duplicate 均 hard-fail；
- v0 历史 prediction 复算 parity；
- raw v1 alias hit 必须在 strict evaluator 下消失，作为回归测试。

Gate：若历史核心数字无法 byte/float tolerance 对齐，停止，先修口径。

#### 14-0B：oracle-prefix / target-path probing

在 Toys/Beauty development validation 上只读计算：

- cold/warm target 的 token-level NLL 与 rank；
- prefix@1...L survival、首次跌出 beam 的深度；
- seen-token / seen-prefix / unseen-suffix coldness taxonomy；
- v0 beam、R² top50、二者 union 的 item recall；
- R² distribution 投影到各层后的 entropy、target mass 与 subtree coverage。

继续 Gate：至少一个域满足以下全部条件：

1. R² top50 对 cold target recall ≥ `5×` v0 top50；
2. cold target 的失败集中在可定位的 prefix/tail 深度，而非完全随机；
3. R² 对 target path 的 projected mass/rank 显著优于 uniform/catalog prior；
4. unique-path/evaluator 工程约束全部通过。

否则 `FAIL_STOP_R2PD_FEASIBILITY`。

#### 14-0C：SpecGR-style candidate verifier control

冻结 v0、冻结 R²，比较：

- resolver score only；
- GRAM normalized candidate likelihood only；
- resolver + GRAM verifier；
- frozen R² portfolio@2。

所有比较必须匹配候选池和 warm cost。该实验只判定“score interface 是否有信息”，不声称新方法。

### Stage 14-1：train-only pseudo-cold transfer screen

在 warm train item 内建立 deterministic、item-disjoint 的 pseudo-cold split；先从所有历史与 CE 样本中删除 audit pseudo-cold item 的真实 interaction，再按实际 cold onboarding 的同一规则允许其 metadata/ID 接收 R² 产生的 synthetic soft mass。换言之，audit ground truth 永远不可见，但 synthetic supervision 可以见到该 catalog path。比较：

| Arm | 唯一改动 | 目的 |
|---|---|---|
| A0 | frozen v0 | reference |
| A1 | top-1 hard cold-path CE | 检验最朴素 synthetic path supervision |
| A2 | soft subtree distillation | 检验 soft mass 是否优于 noisy hard label |
| A3 | A2 + frozen-v0 retention | 检验 warm protection |

Screen Gate：

- A2 的 held pseudo-cold exact-path MRR/Recall@50 ≥ `1.10×A1`；
- A3 保留 A2 至少 95% 的 pseudo-cold gain；
- A3 warm teacher-forced NLL 退化 ≤2%；
- audit item 的真实 interaction/ground-truth transition 从未作为 CE、teacher fitting 或置信模型 target 进入训练；它若收到 synthetic path mass，必须能追溯到仅由可见 history、catalog metadata 与冻结 teacher 产生；
- actual cold interactions与 validation labels没有进入 teacher/student construction。

未通过则停止，不进入 full GRAM。

### Stage 14-2：GRAM matched smoke

固定 256–512 users、固定 checkpoint、固定 steps；smoke 只验证：

- loss 正确下降且各分量有限；
- cold path gradient 非零；
- warm retention gradient 生效；
- generated path 全部合法、唯一、可反解；
- 显存与 runtime 符合预算；
- 同输入 `λ_cp=μ_keep=0` 与原 v0 behavior parity。

smoke 不以 H@10 的一两个事件决定 efficacy。

### Stage 14-3：Toys source-domain full Gate

只允许一组从 Stage 14-1 冻结的配置；不得根据 full validation 回调 `λ_cp/μ_keep/top-M/temperature`。

主比较：v0、R² portfolio@2、Candidate-Scored R²、A1、A2、A3。至少报告 overall/warm/cold H@10、H@50、NDCG@10、事件数、paired bootstrap CI、prefix survival。

进入跨域 Gate：

- native cold H@50 ≥ `2×v0`；
- warm NDCG@10 ≥ `0.97×v0`；
- overall NDCG@10 > v0；
- cold H@50 与 overall NDCG@10 对 v0 的 paired-bootstrap CI 下界 >0；
- A3 至少 Pareto 不劣于 A2，且满足 Section 5 的一个 R² breakthrough 条件；
- ambiguous/duplicate/unknown output 全为 0。

若只满足 native recovery、不满足 R² breakthrough，记 `PASS_NATIVE_RECOVERY_ONLY`，允许分析但不自动宣称主方法成功。

### Stage 14-4：Beauty fixed-config source confirmation

完整迁移 Toys 冻结的算法和超参数；只允许 domain-local catalog/embedding/index 重新构建，不重新搜索 loss weights。

Gate 与 Toys 相同，并要求关键方向一致。Beauty 已被第十三阶段多次查看，因此这里只是 source-domain confirmation，不是 untouched final test。

### Stage 14-5：Sports untouched confirmation（锁定）

只有 Toys + Beauty 均达到 `PASS_R2_BREAKTHROUGH_SOURCE`，且用户明确确认后才可：

- 解锁一次 Sports；
- 先落 manifest/hash/预注册 Gate；
- 禁止根据 Sports 调参或 recovery；
- 完成后无论成败都结束 phase14 方法选择。

### Stage 14-6：publication-level protocol（仅方法成立后）

- ≥3 seeds；
- cold ratios 至少 `{0.2, 0.5}`；
- frequency-stratified split + absolute-time temporal split；
- 与 SpecGR/TIGER/GRAM/R² 以及 discriminative inductive baselines 比较；
- 报告 catalog-known zero-interaction 与 strict newly-added/no-update 两种 setting；
- 训练/推理成本、index update cost、candidate pool size sensitivity；
- ablation：hard vs soft、无 retention、无 confidence、prefix level、teacher strength。

---

## 7. 数据口径与防泄漏规则

### 7.1 当前 cold50 setting 的准确名称

当前 Toys/Beauty `cold50` 是：按全量 item frequency 分桶采样 50% cold item，把它们从每个用户的 train prefix 删除，但保留 catalog metadata、validation/test target。它应被称为：

> **catalog-known, metadata-available, zero-interaction item cold-start simulation**

不能直接写成“训练时完全未知的新物品”，因为 cold catalog、text、ID 仍已知；若 R2PD 给 cold path synthetic target，模型也会在 onboarding 阶段看到该 path，只是看不到真实 user-item interaction。

### 7.2 严禁的泄漏

- 用 validation/test target identity 生成 synthetic query/history；
- 用 actual cold item 的原始被删除 interactions 训练 teacher、student 或置信模型；
- 用 cold/warm target label 作为推理 feature；
- 用 Beauty/Sports 结果回调 Toys 冻结参数后再称跨域确认；
- 在 lexical-ID 层统计命中而不验证唯一 item reverse map。

### 7.3 开发域状态

- Toys validation：高度开发污染，只能作 source development；
- Beauty validation：已用于 v1/R²/CBSA/Tier0/Tier1，不能称 pristine；
- Toys/Beauty test：保持封存；
- Sports：保持 untouched final domain，未经授权不得读取。

---

## 8. 统计与比较原则

1. 所有 gain 同时报 absolute、relative、event count；稀疏 cold hit 不只报百分比。
2. user-level paired bootstrap，至少 10,000 resamples；报告 95% CI。
3. warm/cold/overall 三组都报，不以 overall 隐藏 warm→cold trade-off。
4. R² 比较必须 warm-cost matched 或直接做 Pareto frontier；禁止重犯“高 cold 只是花更多 warm slot”的错误。
5. 全部方法使用同一 catalog、unique ID map、beam size、evaluator。
6. 任何超参搜索只在 train/pseudo-cold development 内完成；source full Gate 后冻结。
7. full run 不因“差一点”改 Gate；失败后的新假设必须开新编号并使用新证据预算。

---

## 9. 资源与工程计划

### 9.1 目录

```text
experiment/phase14/
  protocol/
  configs/
  tests/
  run_phase14_*.sh
artifacts/phase14/
  diagnostics/
  explore/
  formal/
report/第十四阶段/
```

### 9.2 预计成本

| Stage | 资源 | 预计时长 | 是否需 30G lease |
|---|---|---:|---|
| 14-0A/B | CPU / 小 GPU teacher forcing | 分钟–数小时 | 否 |
| 14-0C | 小 GPU batched candidate scoring | <数小时 | 否 |
| 14-1 | 单小 GPU，train-only screen | 数小时 | 否 |
| 14-2 | 单大显存 GPU smoke | <2h | 视 checkpoint 而定 |
| 14-3 Toys full | GRAM full train/inference | 历史约 10–16h | 是 |
| 14-4 Beauty full | GRAM full train/inference | 历史约 26–31h | 是 |

任何 >10 分钟实验继续沿用独立 tmux、hard timeout、status.json、telemetry、no automatic retry。大显存实验启动前先报告预计占用并由用户指定 GPU；不得停止其他进程或调整 holder，除非用户明确授权。

### 9.3 每个正式 artifact 最少包含

```text
config.json
manifest.json
status.json
summary.json
run.log
gpu_telemetry.csv        # 使用 GPU 时
predictions_*.jsonl
item_path_audit.json
data_provenance.json
```

---

## 10. 停止规则

满足任一项立即停止对应路线：

1. collision-safe/item-level evaluator 无法与历史 v0 对齐；
2. actual cold interaction 或 held target 泄漏；
3. 14-0B 显示 R² teacher 对目标 path 没有高于 prior 的支持；
4. item-disjoint pseudo-cold 中 soft distillation 不优于 hard top-1；
5. A3 无法在 warm retention `≥97%` 下保留至少 95% cold gain；
6. full Toys 不满足 native `2×` cold H@50 或 overall >v0；
7. 方法收益只来自 ambiguous lexical path；
8. 需要同时改 ID、decoder、resolver、split 才能得到正结果，无法归因；
9. 结果实质等价于 SpecGR/AGRec，而没有独立机制或明显实证优势；
10. Toys/Beauty source Gate 未通过却想提前读取 Sports。

---

## 11. 可能的论文故事（只在结果成立后采用）

### 暂定主线

1. **Cold-path collapse diagnosis**：collision-safe、item-level 评估显示 lexical semantic ID 的共享前缀并不自动带来 unique cold-item reachability。
2. **R² recoverability probe**：warm-only inductive resolver 证明用户历史中存在可恢复的 cold preference，但外部 portfolio 有 warm trade-off 与 candidate ceiling。
3. **R²-to-Path Distillation**：将 item teacher distribution 投影成 prefix-conditional subtree targets，在不改 identifier 的条件下给冷路径支持。
4. **Warm retention**：用 frozen-v0 path retention 把 dynamic catalog onboarding 与 old-item forgetting统一起来。
5. **Rigorous protocol**：unique item evaluation、cost-matched Pareto、catalog-known/temporal 两种 cold setting。

### 可以声称与不能声称的边界

若 source + untouched domain 成立，可以声称：

- R² 是有效的 recoverability teacher/diagnostic；
- prefix-level soft transfer 能提高 native collision-safe cold reachability；
- warm retention 改善了 R² 的 Pareto trade-off。

在没有相应实验前不能声称：

- 完全 zero-shot 新物品；
- 不需要 catalog onboarding；
- universally beats SpecGR；
- R² 当前双域 validation 等于 publication-level SOTA；
- 多兴趣 resolver 是主要创新。

---

## 12. 下一步唯一动作

**先实现 Stage 14-0A + 14-0B，不启动 GRAM full training。**

首批应创建：

```text
experiment/phase14/protocol/item_level_eval.py
experiment/phase14/protocol/oracle_prefix_probe.py
experiment/phase14/tests/test_item_level_eval.py
experiment/phase14/tests/test_oracle_prefix_probe.py
experiment/phase14/configs/stage14_0_toys_beauty.json
```

完成诊断并形成 `report/第十四阶段/GRAM_第十四阶段_Stage14-0_冷路径可行性诊断报告.md` 后，再由 Gate 决定是否实现 R2PD。不得从本计划直接跳到 full Toys/Beauty。

---

## 13. 核心参考文献（截至 2026-08-19）

1. Lee et al. **GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion.** ACL 2025. https://aclanthology.org/2025.acl-long.1596/
2. Ding et al. **Inductive Generative Recommendation via Retrieval-based Speculation.** AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/38486
3. Wang et al. **AGRec: Adapting Autoregressive Decoders with Graph Reasoning for LLM-based Sequential Recommendation.** Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.369/
4. Lin et al. **Order-agnostic Identifier for Large Language Model-based Generative Recommendation.** SIGIR 2025. https://arxiv.org/abs/2502.10833
5. Fu et al. **Differentiable Semantic ID for Generative Recommendation.** SIGIR 2026. https://arxiv.org/abs/2601.19711
6. Zhang et al. **Cold-Starts in Generative Recommendation: A Reproducibility Study.** SIGIR 2026. https://arxiv.org/abs/2603.29845
7. Peng et al. **Can Generative Recommendation Reach Cold Items? A Temporal Perspective on Semantic-ID Generation.** arXiv preprint, 2026（非同行评审）. https://arxiv.org/abs/2607.21101
8. Mehta et al. **DSI++: Updating Transformer Memory with New Documents.** EMNLP 2023. https://aclanthology.org/2023.emnlp-main.510/
9. Liu et al. **Fine-Tuning Out-of-Vocabulary Item Recommendation with User Sequence Imagination.** NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/10d52f5d2ef0f69ac10da7c962fb6db9-Abstract-Conference.html
10. Huang et al. **Aligning Distillation For Cold-start Item Recommendation.** SIGIR 2023. https://doi.org/10.1145/3539618.3591732
11. Liu et al. **Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation.** SIGIR 2023. https://doi.org/10.1145/3539618.3592078
12. Zhou et al. **Contrastive Collaborative Filtering for Cold-Start Item Recommendation.** WWW 2023. https://doi.org/10.1145/3543507.3583286
13. Kim et al. **Content-based Graph Reconstruction for Cold-start Item Recommendation.** SIGIR 2024. https://doi.org/10.1145/3626772.3657801
14. Rajput et al. **Recommender Systems with Generative Retrieval.** NeurIPS 2023. https://arxiv.org/abs/2305.05065

---

## 14. 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-19 | 新开第十四阶段，保留 R² 为有效点 | R² 双域正向证据未被否定；用户希望将其作为论文组成点 |
| 2026-08-19 | 停止以 allocator/resolver 小调参为主线 | Tier0/Tier1 已定位 candidate recall/path support 才是瓶颈 |
| 2026-08-19 | 主候选改为 R2PD | 与 GRAM 原始训练缺口、本地反证、dynamic generative retrieval 文献最一致 |
| 2026-08-19 | Candidate verifier 降为 baseline | SpecGR AAAI 2026 已覆盖 drafter-verifier 框架 |
| 2026-08-19 | pseudo user-sequence 降为备选 | USIM NeurIPS 2024 已覆盖，且本地 pseudo-cold transfer 曾失败 |
| 2026-08-19 | Sports 继续封存 | Toys/Beauty 均已是开发域，需保留一次真正独立确认 |
