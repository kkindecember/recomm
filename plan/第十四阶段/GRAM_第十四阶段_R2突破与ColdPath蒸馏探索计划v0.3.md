# GRAM 第十四阶段：R²-to-Path Distillation 与冷路径原生可达性 v0.3

> **建立日期**：2026-08-20（v0.1 为 2026-08-19；v0.2 为 2026-08-20）
> **当前状态**：`PLAN_REVISED / STRUCTURAL_DIAGNOSTIC_ONLY / NO_MODEL_TRAINING_STARTED`
> **工作名**：R2PD（R²-to-Path Distillation，暂定名；是否进入训练仍由 Stage 14-0B 决定）
> **前一版**：`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.2.md`（保留不删，便于审计）
> **计划验证状态**：结构统计已复算；R2PD、SpecGR 适配与所有 phase14 训练结果均未验证

---

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-20
- Verification Status: PARTIALLY_VERIFIED（仅 catalog 结构统计与本地代码落点已核对）
- Version Label: phase14_code_plan_v0.3

---

## 0. v0.3 相对 v0.2 改了什么（先读这一节）

v0.3 接受 v0.2 的战略收缩，但修复五个会影响科学有效性或可执行性的缺口：结构统计被过度解释为 learned NLL 上界、R2PD 没有定义 cold-only prefix 如何获得梯度、`lexid2cfid` 会静默覆盖 collision、遗漏 GenRecEdit、Pareto Gate 不可判定。

### 0.1 七项冻结决策

| # | 决策 | 原因 |
|---|---|---|
| 1 | 诊断定位为**迁移 + 受控扩展** | 2607.21101 已做 oracle-prefix/taxonomy；GRAM 的 hierarchical lexical ID 仍是不同机制，但不能预称“相反结论” |
| 2 | **删除 warm ≥ 0.97 硬门槛**，改为预注册的 R² cost-matched 判据 | 绝对 retention 阈值未经论证；但不能用模糊的“Pareto 外移”替代 |
| 3 | **Gate 从 25+ 条压到每 Stage 1 主 Gate + 1 kill** | 当前是探索模式（见 memory `feedback_experiment_mode.md`），v0.1 用的是确认模式的治理强度 |
| 4 | **真 SpecGR 提前到 M1；恢复同 backbone verifier control** | 官方 SpecGR 的 TIGER/UniSRec 结果不能单独隔离训练期迁移与推理期 verifier |
| 5 | **GenRecEdit 纳入最近邻边界** | 它已覆盖 cold SID pattern 注入、warm preservation 与 update cost；“零推理外部依赖”不是未经限定的唯一轴 |
| 6 | **Sports 从核心阶段延期，不视为永久取消** | `Sports_cold50` 不存在；方法未过双域前不投入 30–45h，但最终论文须承认两域限制或补第三域 |
| 7 | **主线只由 14-0B 的真实 NLL/beam 诊断决定** | catalog prefix overlap 只提供结构先验，不能提前否定末层或 search 假设 |

### 0.2 已实测的新证据：结构先验，不是 learned failure 证明

**产出**：`experiment/phase14/protocol/cold_prefix_support.py`（只读 catalog，`test_read: false`）
**锁定**：`experiment/phase14/tests/test_cold_prefix_support.py`（8 tests，OK）
**结果**：`artifacts/phase14/diagnostics/cold_prefix_support_{toys,beauty}.json`

`test_read: false` 只是脚本生成的 provenance 字段，不单独构成无泄漏证明；可信依据是代码依赖审计显示它只读取 identifier 与 `cold_split_meta/{cold,warm}_items.txt`。正式报告仍需记录输入文件 hash 和实际 open-file manifest。

对每个 cold item，问它的 lexical path 最深前缀 `z[:k]` 有多深仍被**至少一个 warm item** 共享：

| | Toys_cold50 | Beauty_cold50 |
|---|---:|---:|
| identifier | `c32_l5`（5 层，少数 6） | `c128_l7`（7 层，少数 8） |
| cold / warm item | 5963 / 5961 | 6052 / 6049 |
| baseline 重复路径 | **0** | **0** |
| 除末位外前缀唯一的 item 占比 | 87.3% | 95.4% |
| **cold 累计断裂于深度 ≤2** | **67.8%** | **82.5%** |
| **cold「除末位外全路径」被 warm 支持** | **8.6%** | **2.66%** |

arXiv 2607.21101 在 TIGER/RQ-VAE 上观察到较晚位置的 fine-grained completion 困难；本项目的结构统计显示，GRAM cold path 往往更早失去**完整 warm-item prefix overlap**。这提示两类 identifier 的结构支持形态可能不同，但还没有证明 learned NLL 的断崖位置不同。

必须区分三层含义：

1. **已证据化**：67.8% / 82.5% 的 cold path 在 depth≤2 已无任何 warm item 共享该完整前缀；
2. **待 14-0B 验证**：模型的 token NLL、target rank 与 beam survival 是否也在相应位置恶化；
3. **需要受控 ID 对照才能声称**：identifier 构造方式“决定” failure depth。若不做同数据/同 backbone 的 ID control，只能写“associated with”。

神经 decoder 可能组合出从未完整出现过的 prefix，也可能在结构上有 warm overlap 的 prefix 处提前失败。因此该量**不是 learned reachability 或 NLL cliff 的严格上界**，不能单独否定末层 head、search failure 或 R2PD。

### 0.3 预期差异化轴（结果成立后才能采用）

诊断、drafter-verifier、模型编辑都已有邻近工作。R2PD 的可检验差异化组合是：

> **把 user-conditioned resolver item distribution 离线蒸馏为 GRAM 的 prefix-conditional native decoder probability；部署时使用未改接口的标准 GRAM beam，不需要 drafter、candidate verifier 或 edit-memory triggering。**

“部署时无外部模块”不等于“免费 onboarding”。必须同时报告 offline update time、GPU-hours、更新参数量、cold batch size sensitivity 与模型存储；若每批新 item 都需 full retraining，只能称 **fixed-catalog offline adaptation**，不能称低成本动态 onboarding。

---

## 1. 已有证据

### 1.1 可继续使用的正证据

| 证据 | Toys | Beauty | 支持的结论 |
|---|---:|---:|---|
| v0 GRAM cold H@50 | 1.03% | 1.31% | native cold path 基本坍塌 |
| resolver cold Recall@50 | 11.40% | 11.03% | 内容可归纳专家存在约 9–11 倍 reachability |
| R² `portfolio@2` overall NDCG@10 | +4.96% | +4.15% | 双域同方向 overall gain |
| R² `portfolio@2` cold H@50 | 2.89× | 2.49× | 冷侧收益可复现 |
| R² `portfolio@2` warm NDCG@10 | −4.09% | −5.26% | 收益伴随真实 warm trade-off |
| learned allocation vs matched random | 显著更优 | 显著更优 | 学习型 allocation 非无效，但不是主瓶颈 |
| **cold path 在 depth≤2 失去 exact warm-prefix overlap** | **67.8%** | **82.5%** | 结构新颖性较早出现；不等同于 learned NLL 断崖 |

### 1.2 已否定 / 必须降级的路线（**禁止恢复**）

> 这张表是 phase-13 最宝贵的沉淀。接手的 AI 必须先读，否则会重复已失败的 8 轮工作。**v0.3 原样保留这些负结果。**

| 路线 | 结论 | 处理 |
|---|---|---|
| 原 v1 Semantic Bridge raw ID | 增益主要来自 lexical-ID collision/alias | 禁止作为正方法恢复 |
| collision-safe v1 | Toys/Beauty 正式 FAIL | 仅作「路径无监督」的反证 |
| v2 LLM prior / v3 semantic-collaborative alignment | 双域显著退化 | 禁止继续做 hard semantic target/alignment |
| depth-3 route fusion | depth 3 已近 item identity，压制 resolver 正确项 | 禁止重调 RRF 权重救援 |
| P1–P7 / CBSA allocator | 未突破 frozen R²；受候选池与稀疏正例限制 | **不做第 9 个 allocator** |
| extra epochs / static hard negatives | resolver recall 不升反降 | 不作主线 |
| pseudo-cold setwise selector | 真实 cold 降到 resolver top-1 的 0.558× | 新伪监督必须先过 item-disjoint Gate |
| Tier-1 RRF / recall-then-place | 后处理层饱和；GRAM score 加入后 cold 变差 | 不再做冻结双路后处理搜索 |

### 1.3 T1-4 多兴趣 Resolver

属 phase-13 未完成的 closure。补跑与否都：不作 phase14 主创新、不触发更多 resolver 变体、不影响主线判断，最多作 teacher-strength sensitivity。

---

## 2. 实现落点（已逐一核对存在）

| 位置 | 原职责 | phase14 扩展 |
|---|---|---|
| `GRAM/src/model/gram.py:80-98` | T5 forward | **`last_loss_components` 钩子已存在**，cf0 已示范如何叠加辅助 loss —— 新 loss 直接复用此模式 |
| `GRAM/src/runner/single_runner_gram.py:253-273` | loss 累加 | 组合 CE + cold-path KD + warm retention |
| `GRAM/src/data/multi_task_dataset_gram.py:93-99` | `item2cfid` / `lexid2cfid` | `lexid2cfid` 遇 collision 会静默覆盖；14-0A 必须从 `item2lexid` 重建 `path -> [item_ids]` multimap 并 hard-fail |
| `GRAM/src/processor/Collator.py` | target tokenization | 新 collator 显式携带 synthetic decoder prefixes、sparse next-token targets、prefix mass、confidence、provenance |
| `GRAM/src/utils/generation_trie.py`(155行) | legal-token mask | 增加只读 subtree index / descendant mass 工具，不改合法性语义 |
| `GRAM/src/utils/evaluate.py`(58行) | lexical string match | 新 item-level evaluator读取 `path -> [item_ids]` multimap；仅 singleton path 可计 item hit |
| `GRAM/src/data/multi_task_dataset_rec.py` | warm CE 样本 | **保持冻结**，在外层 wrapper 生成 teacher 分布 |

纪律：新代码放 `experiment/phase14/`；若必须改 GRAM 内部，用最小 patch + feature flag，并先存 behavior parity test。

---

## 3. 文献边界

### 3.1 直接约束

| 工作 | HOW | 对本阶段的约束 |
|---|---|---|
| **GRAM**, ACL 2025 | lexical hierarchy + collaborative verbalization + late fusion | 原 backbone；缺 zero-interaction path supervision |
| **SpecGR**, AAAI 2026 Oral | inductive drafter + GR teacher-forcing verifier + guided redrafting | **最近邻竞争者，代码开源**。drafter+verifier 不能再当新贡献。→ M1 必跑 |
| **2607.21101**, arXiv 2026 | temporal split、coldness taxonomy、oracle-prefix probing | 通用诊断框架已被做过；本项目只主张迁移到 GRAM 并检验 hierarchical lexical ID 是否呈现不同 failure profile |
| **ColdGenrec**, SIGIR 2026 | 统一 cold protocol、factor-wise controls，代码开源 | 要求一次只改一因素、报 warm/cold。其 temporal split 供 M5 使用（⚠️ Toys 仅 133 cold item） |
| **GenRecEdit**, 2026 | position-wise cold SID model editing + warm preservation + one-one triggering | **直接竞争者**：已覆盖 cold pattern 注入和 update cost。R2PD 必须证明 user-conditioned distribution transfer、标准 beam 与无 edit memory 的差异 |
| **MemGen-GR**, KDD 2026 | instance/token memorization-generalization 分析 + support coverage | 进一步压缩“只做 token/support 诊断”的新颖性；诊断只能服务方法机制 |
| AGRec, Findings ACL 2025 | GNN logits 增强 + rankable FSM | 推理期 logit fusion 已拥挤，只适合作强 baseline |
| SETRec / DIGER, SIGIR 2025/2026 | order-agnostic set ID / 可微 SID | 重做 ID 体系赛道已拥挤且昂贵，不走 |
| DSI++, EMNLP 2023 | pseudo-query rehearsal | 冷路径注入必须配 warm retention |
| ALDI / UCC / CCFCRec / CGRC | cold-warm 蒸馏对齐、不确定性、对比迁移、伪冷重构 | 提供 retention/confidence/pseudo-cold 设计参考 |

### 3.2 已不够新的故事（不要写）

- content retriever 提 cold candidate + GRAM 排序 → SpecGR
- auxiliary score 加到 decoder logits → AGRec
- 想象冷 item 的用户序列 → USIM
- 对 cold SID pattern 做 position-wise model editing → GenRecEdit
- 重设计 semantic ID → SETRec / DIGER
- 多兴趣 resolver 提召回 → 检索模块优化，非 GRAM 机制

---

## 4. 主线：learned diagnosis 驱动分支（决策 7）

**Stage 14-0B 唯一决策证据：在 GRAM 上实测 cold target path 的 NLL/rank profile 与 beam 存活深度。§0.2 的 catalog overlap 只作解释变量，不参与路线 hard Gate。**

分支规则**事先写死，不可事后修改**：

| 实测断崖深度 | 路线 |
|---|---|
| **中层（depth 2–3）**，且 R² projected mass 高于 prior | **走 R2PD**：teacher 对对应 prefix 有可迁移信息 |
| 末层（最后 1–2 层） | 转末层 identification head（保留 identifier 不变） |
| 全程均匀低 / 无断崖 | 问题在 search/competition 而非 path support → 转 beam 侧或终止 |

### 4.1 R2PD（预期主线）

给定仅由可见 train history 构造的 `h`，冻结 R² resolver 输出 catalog score。先在 pseudo-cold train split 上冻结 temperature `τ`、candidate size `M` 与 confidence 规则，把 score 变成 stop-gradient 分布 `qτ(i|h)`；validation/test target 不参与 calibration。

令 `S_M(h)` 为 teacher top-M item，`z(i)` 为 collision-safe unique path，`P_M(h)` 为这些 path 的所有前缀。对任意 `p ∈ P_M(h)` 定义绝对 prefix mass 与条件 next-token target：

```text
m(p | h) = Σ_{i∈S_M(h): z(i) 以 p 开头} qτ(i|h)

Q(a | p, h) = Σ_{i∈S_M(h): z(i) 以 p+a 开头} qτ(i|h) / m(p|h)
```

`p` 只从 `P_M(h)` 取，因此分母为正；低于 `m_min` 的 prefix 跳过。学生分布 `Pθ(a|p,h)` 只在 trie 的 legal children 上归一化。关键实现不是只在真实 warm target prefix 上算 KL，而是把 `P_M(h)` 中的 **synthetic prefix 作为 decoder teacher-forcing input**；否则 cold-only prefix 不会获得梯度。

prefix KD：

```text
L_cp(h) = [1 / Z(h)] · Σ_{p∈P_M(h), m(p|h)≥m_min}
          m(p|h) · c(p,h) · KL[Q(.|p,h) || Pθ(.|p,h)]
```

- `m(p|h)` 保留 teacher 的绝对 prefix mass，避免每个低质量深层 prefix 被等权放大；
- `c(p,h)` 由 history margin/entropy 与 prefix coverage 组成，只能在 train/pseudo-cold 上拟合；
- `Z(h)` 为有效权重和；无有效 prefix 时该样本 `L_cp=0`，不得强行归一化；
- top-M 截断之外的质量记入 `tail_mass` 审计，不得默认为 0 且不报告；
- teacher path、prefix、mass、temperature、来源 checkpoint 全部写入 provenance。

总损失：

```text
L = L_warm_CE + λ_cp · L_cp + μ_keep · L_frozen_v0_retention
```

- `L_warm_CE`：原 warm next-item 监督，保持任务锚点
- `L_cp`：对 teacher candidate path 的 synthetic prefixes 做 mass-weighted soft KL，包含 cold-only prefix
- `L_frozen_v0_retention`：在原 warm CE path prefixes 上匹配冻结 v0 的 legal-child distribution，防遗忘；teacher 与 v0 均 stop-gradient

**必须先通过的 mechanism unit test**：构造一条从未出现在 warm CE target 中的合法 cold path，让 teacher 将全部质量放在该 path；一次受控更新后，该 path 每个受监督 prefix 的目标 next-token log-prob 都应上升，且 `λ_cp=μ_keep=0` 时逐位恢复 v0 behavior。

**它针对哪条本地反证**：v1 只改 ID 未给 path 训练支持 → R2PD 直接改 path probability；v2/v3 硬对齐语义与协同 → R2PD 只蒸馏当前用户条件下的 item mass；P5 单点伪标签迁移失败 → R2PD 用 soft 分布 + confidence；R² warm 下跌 → retention 进训练目标而非事后补救。

**风险**：teacher Recall@50 仅约 11%，且 top-M 外 89% target absence 是已知天花板；R² score 可能不可校准；full fine-tuning 可能再伤 warm；若只有推理期 reranking 有效则退化为 SpecGR 邻近路线；若每次 catalog 更新都需 full retraining，则部署优势可能不成立。

---

## 5. 可证伪预测

| # | 假设 | 预测 | 证伪 |
|---|---|---|---|
| H1 | 瓶颈包含 learned unsupported decoder path | cold target NLL/rank 在可定位位置恶化，且 failure depth 与 exact warm-prefix overlap、beam survival、cold hit 有稳定关联 | NLL 与 warm 接近但 beam 仍失败 ⇒ 问题在 search；结构 overlap 不能替代该判断 |
| H2 | R² item mass 可变成有效 prefix supervision | item-disjoint pseudo-cold 上 soft subtree target 优于 top-1 hard CE | soft 只降 teacher loss 不提 held pseudo-cold exact item ⇒ 停止 |
| H3 | warm forgetting 可被 retention 缓解 | 在 M5 预注册 `μ_keep` sweep 中，相近 `G_c` 下 `C_w` 降低，或相近 `C_w` 下 `G_c` 提高 | 三个冻结 operating points 均未改善任一 cost-matched 比较 ⇒ 承认 Pareto 极限 |
| H4 | 标准 native beam 可保留 R² 的主要增量收益 | 无 portfolio/drafter/edit-memory 时，R2PD 通过 §6 M3 的预注册 R² cost-matched 判据，且 overall 显著优于 v0 | 只比 v0 好但未达到 R²-transfer Gate ⇒ 仅为 native recovery；只有挂外部模块才有效 ⇒ 无独立主故事 |

**H4 是主卖点的直接检验。** 不再使用 `warm≥0.97×v0`，也不使用“同量级”“Pareto 外移”这类事后可移动措辞；M3 以 R² 的增量 cold gain 和实际 warm cost 为参照。

---

## 6. 执行计划（M1–M6）

### M1：诊断 + 竞争者（小 GPU）

**14-0A｜item-level evaluator 回归测试（约 1 天，非故事主线）**

本轮已确认 baseline identifier 重复路径 = 0（Toys/Beauty），但 raw v1 存在 collision，因此 evaluator 仍是**必要的回归测试**，不是贡献。

- 从 `item2lexid` 重建 `path -> [item_ids]` multimap；禁止直接信任会覆盖 collision 的 `lexid2cfid`
- duplicate path / ambiguous decoding / unknown item / top-K duplicate 一律 hard-fail
- v0 历史 prediction 复算 parity；raw v1 的 alias hit 须在 strict evaluator 下消失
- **kill**：历史核心数字无法对齐 ⇒ 停止，先修口径

**14-0B｜断崖诊断（主 Gate）**

Toys/Beauty validation 上只读计算：cold/warm target 的 token 级 NLL 与 rank；prefix@1..L survival 与首次跌出 beam 的深度；v0 beam / R² top50 / 二者并集的 item recall；R² 分布投影到各层后的 entropy、target mass、subtree coverage。

- **主 Gate**：learned NLL/rank 或 beam survival 的 failure profile 可定位，且 R² 对 target path 的 projected mass/rank 显著优于冻结的 uniform、popularity 与 catalog-text prior 中最强者
- **kill**：R² teacher 对目标 path 没有高于 prior 的支持 ⇒ `FAIL_STOP_R2PD`
- 产出 → `report/第十四阶段/Stage14-0_冷路径可行性诊断报告.md`，据 §4 分支表选路线

**14-0C｜竞争者与同-backbone interface control（与上并行）**

先做 SpecGR compatibility audit：官方实现以 TIGER/UniSRec 和自己的数据 pipeline 为主，不能把原仓库数字直接与 GRAM/Toys 比较。只有相同 split、catalog、candidate budget、beam K、evaluator 与 cold definition 全部对齐后，才称“真 SpecGR baseline”。

并恢复冻结的同-backbone control：R² candidate score only、GRAM candidate likelihood only、R²+GRAM verifier、R² portfolio@2。它只回答“训练期 path transfer 是否优于推理期 candidate interface”，不作为创新。

- **这是 go/no-go**：若 SpecGR 已打平或打赢且我们无差异化优势，phase14 需重新定位
- 同时它成为主表的强 baseline（符合「只选开源 baseline」约束）
- GenRecEdit 若存在可审计、可做同协议适配的公开实现，则进入 M1 compatibility audit；若无法公平复现，只做文献边界与 update-cost 对照，不伪造数字

### M2：pseudo-cold screen + smoke（小–中 GPU）

**14-1｜item-disjoint pseudo-cold transfer screen**（保留 v0.1 设计，吸取 P5 教训）

在 warm train item 内建 deterministic、item-disjoint 的 pseudo-cold split；按真实 cold50 的 frequency stratum 匹配，并记录 path length/deepest-overlap/text-length 分布差异。先从所有历史与 CE 样本中删除 audit item 的真实 interaction，再按真实 cold onboarding 的同一规则允许其 metadata/ID 接收 R² 的 synthetic soft mass。**audit ground truth 永不可见，synthetic supervision可见该 catalog path。**

| Arm | 唯一改动 |
|---|---|
| A0 | frozen v0（reference） |
| A1 | top-1 hard cold-path CE |
| A2 | soft subtree distillation |
| A3 | A2 + frozen-v0 retention |

- **主 Gate**：预先指定 exact-path MRR 为 primary；A2−A1 的 item-level paired-bootstrap 95% CI 下界 >0，且 A2 对 A0 的 Recall@50/beam survival 不退化。Recall@50 为 secondary，不与 MRR 组成事后“二选一”
- **kill**：audit item 真实 interaction 以任何形式进入 CE / teacher fitting / 置信模型

**14-2｜matched smoke**：固定 256–512 users、固定 checkpoint/steps。验证 loss 各分量有限且下降、synthetic cold-only prefix 的目标 token log-prob 上升、retention 梯度生效、生成 path 全部合法唯一可反解、显存与 runtime 达标。

- **主 Gate**：`λ_cp = μ_keep = 0` 时与原 v0 behavior **逐位 parity**
- smoke 不以 H@10 的一两个事件判 efficacy

### M3：Toys full（30G lease）

核心 arm：v0 / R² portfolio@2 / 同-backbone R²+GRAM verifier / R2PD 主 arm；通过 compatibility audit 的真 SpecGR 作为强 baseline。GenRecEdit 仅在同协议实现可审计时加入，不为凑表强行复现。

报告 overall/warm/cold 的 H@10、H@50、NDCG@10、事件数、paired bootstrap CI、prefix survival。定义：

```text
G_c(M) = cold_H@50(M) - cold_H@50(v0)          # 越大越好
C_w(M) = warm_NDCG@10(v0) - warm_NDCG@10(M)   # warm cost，越小越好
```

`cold H@50` 是 primary efficacy；`warm NDCG@10` 是 primary cost；`overall NDCG@10` 是 secondary utility。R²-transfer 的非劣 margin 预注册为保留 R² incremental cold gain 的 90%：`G_c(R2PD) ≥ 0.90×G_c(R²)`。该 90% 只用于工程/论文成功分类，并同时报告无 margin 的原始差值与 CI；不得事后改成 80% 或 95%。

成功标签按层级记录：

1. `PASS_NATIVE_RECOVERY`：R2PD 对 v0 的 cold H@50 paired-bootstrap 95% CI 下界 >0，且 overall NDCG@10 point estimate >v0；
2. `PASS_R2_TRANSFER`：满足 1，且满足以下之一：
   - `G_c(R2PD) ≥ 0.90×G_c(R²)`，同时 warm NDCG@10 对 R² 的 paired CI 下界 >0；或
   - `C_w(R2PD) ≤ C_w(R²)`，同时 cold H@50 对 R² 的 paired CI 下界 >0；
3. `PASS_PARETO_DOMINANCE`：cold 不低于 R²、warm 不低于 R²，且至少一个维度对 R² 的 paired CI 下界 >0。

- **主 Gate**：至少 `PASS_NATIVE_RECOVERY` 才允许进入 Beauty；只有 `PASS_R2_TRANSFER` 以上才能采用“R² 知识成功迁入 native path”的主论文故事
- **kill**：ambiguous/duplicate/unknown output 非 0
- **无固定 warm retention 硬门槛**；“Pareto frontier”仅用于 M5 预注册多 operating-point sweep，不拿单个 arm 冒充 frontier

### M4：Beauty full（30G lease）

迁移 Toys 冻结的算法、temperature/top-M 选择规则和 dimensionless loss-weight policy；只允许 domain-local catalog/embedding/index 重建。若因 c32/c128 几何不同必须改变绝对阈值，只能使用 M2 预先写明的归一化规则，不能看 Beauty validation 后手调。

- **主 Gate**：若 Toys 为 `PASS_NATIVE_RECOVERY`，Beauty 也必须达到 `PASS_NATIVE_RECOVERY`；若 Toys 为 `PASS_R2_TRANSFER`，Beauty 至少达到 native recovery 且 warm cost 不超过 R²。只有两域都达到 `PASS_R2_TRANSFER` 才能称“双域 R² transfer”；否则必须写 mixed evidence
- Beauty 已被 phase-13 多次查看，这是 source-domain confirmation，非独立终验

### M5：补充实验 + 主表（30G lease）

- **test freeze**：开封前写入 commit/config hash、3 个 seeds、全部 arm、metric code hash 与 exclusion rules；Toys/Beauty 的 3-seed test jobs 作为一个批次启动，全部结束前不查看中间 test 指标
- **temporal 小节**：用 ColdGenrec 开源脚本，只跑 v0 + 冻结主 arm。⚠️ Toys temporal 仅 133 cold item，只作方向性外部有效性，不单独声称显著
- **ablation / Pareto sweep（validation only）**：hard vs soft、无 retention、无 confidence、prefix level、teacher strength，以及预注册的 3 个 `μ_keep` operating points；test 主表不追着最优点重开
- **结构机制边界**：若要写“identifier construction 决定 failure depth”，必须补同数据/同 backbone 的 ID control；否则全文统一写“associated with”
- **部署成本**：除推理 latency 外，报告 offline update wall time、GPU-hours、peak memory、更新参数量、checkpoint 增量、每 100/500/全部 cold items 的 batch sensitivity，并与 SpecGR/GenRecEdit 可获得数字分开注明“本地复现”或“论文报告”
- **第三域策略**：Sports 不在核心探索预算内。只有 Toys+Beauty 达 `PASS_R2_TRANSFER` 且用户批准资源后，才建立第三域；若不补，投稿限制中明确“两类 Amazon 域、无独立第三域”

### M6：写作 + 返工缓冲

---

## 7. 数据口径与防泄漏

### 7.1 口径的准确名称

当前 `cold50` 是：按全量 item frequency 分桶采样 50% cold item，从每个用户 train prefix 删除，但保留 catalog metadata 与 validation/test target。应称：

> **catalog-known, metadata-available, zero-interaction item cold-start simulation**

**不能**写成"训练时完全未知的新物品"——cold catalog/text/ID 仍已知；R2PD 给 cold path synthetic target 时，模型在 onboarding 阶段会看到该 path，只是看不到真实 user-item interaction。

主口径用它（保住 phase-13 沉淀与统计功效：cold 事件 4367/5287 量级）；M5 用 temporal 作方向性补充。

### 7.2 严禁的泄漏（原样保留）

- 用 validation/test target identity 生成 synthetic query/history
- 用 actual cold item 被删除的原始 interaction 训练 teacher/student/置信模型
- 用 cold/warm target label 作推理 feature
- 用 Beauty 结果回调 Toys 冻结参数后仍称跨域确认
- 在 lexical-ID 层统计命中而不验证唯一 item reverse map

### 7.3 域状态

- Toys validation：高度开发污染，仅作 source development
- Beauty validation：已用于 v1/R²/CBSA/Tier0/Tier1，**不能称 pristine**
- **Sports：核心探索阶段不使用**（决策 6），但不是被证据否定；是否建立第三域由 M5 publication Gate 与资源共同决定
- Toys/Beauty test：仅在配置/代码/seeds 全冻结后一次性批量开封；任何 test 后方法修改都必须开新版本且旧 test 不再称确认集

---

## 8. 统计与比较原则

1. 所有 gain 同时报 absolute、relative、event count；稀疏 cold hit 不只报百分比
2. user-level paired bootstrap，≥10,000 resamples，报 95% CI
3. warm/cold/overall 三组都报，不以 overall 隐藏 trade-off
4. **R² 比较使用 §6 的 `G_c/C_w` 预注册规则**；Pareto frontier 只由多个事先冻结 operating points 构成，禁止把单点称为 frontier
5. 全部方法同一 catalog、unique ID map、beam size、evaluator
6. 探索期只在 train/pseudo-cold/Toys validation 调参；Beauty 只做固定规则迁移；最终 test 配置冻结后一次性跑完
7. full run 不因"差一点"改 Gate
8. primary 指标只检验 cold H@50；warm NDCG@10 是约束成本，overall 与其他 K 为 secondary，防止多指标挑正结果
9. bootstrap 单位为 user；同时报告 cold-event 数与 per-item coverage，避免少数热门 cold item 主导平均值

---

## 9. 资源

```text
experiment/phase14/{protocol,configs,tests}/   artifacts/phase14/{diagnostics,explore,formal}/
report/第十四阶段/
```

| Stage | 资源 | 时长 | 30G lease |
|---|---|---:|---|
| 14-0A/B | CPU / 小 GPU | 分钟–数小时 | 否 |
| 14-0C compatibility + controls | 小 GPU | 数小时–数天（适配失败也须留报告） | 否 |
| 14-1 / 14-2 | 单小 GPU | 数小时 | 否 |
| M3 Toys core 4–5 arms | GRAM full | 每 arm 10–16h | **是** |
| M4 Beauty core 4–5 arms | GRAM full | 每 arm 26–31h | **是** |
| M5 | temporal + ablation + 主表 | 数天 | **是** |

>10 分钟实验沿用：独立 tmux、hard timeout、status.json、telemetry、**不自动重试**。大显存实验启动前报告预计占用并由用户指定 GPU；**不得停止他人进程或调整 holder，除非用户明确授权**。

**已知坑**（memory `feedback_runner_tmux_bug.md`）：`run_phase13_explore.sh start` 用 tmux 拉 launch_cmd 可能 10s 内 exit=1，改用 `setsid nohup` 绕开；`finish()` 不检查 workload_rc（CUDA OOM 会误判成功）；status.json 的 workload_pid 恒为 0；**判活用 workload PID，不要用 tmux session**。

每个正式 artifact 至少含：`config.json` `manifest.json` `status.json` `summary.json` `run.log` `gpu_telemetry.csv` `predictions_*.jsonl` `item_path_audit.json` `data_provenance.json`。

---

## 10. 停止规则（路线级 4 条）

1. **泄漏**：actual cold interaction 或 held target 进入训练/teacher/置信模型
2. **收益来自 alias**：方法增益只来自 ambiguous lexical path
3. **无法归因**：需同时改 ID、decoder、resolver、split 才能得正结果
4. **无差异化或成本倒置**：结果实质等价于 SpecGR/AGRec/GenRecEdit，且无独立机制；或部署省下的推理成本明显小于每批 catalog full retraining 的代价

（各 Stage 另有自己的 1 条 kill，见 §6）

---

## 11. 论文故事（仅结果成立后采用）

1. **GRAM cold-path failure profile**：把 [2607.21101] 的可达性分析迁移到 hierarchical lexical ID；只有 learned NLL/beam 与受控 ID evidence 成立时才讨论与 RQ-VAE 的差异。无 ID control 时只写“结构 overlap 与 failure depth 相关”
2. **R² recoverability probe**：warm-only inductive resolver 证明用户历史中存在可恢复的 cold preference，但外部 portfolio 有 warm trade-off 与候选天花板
3. **R²-to-Path Distillation**：把 item teacher 分布投影为 prefix-conditional subtree targets，不改 identifier 即给冷路径支持
4. **★ 标准 native beam 部署**：与 SpecGR/GenRecEdit 对比，推理时不需要 drafter、candidate verifier 或 edit-memory triggering；同时完整披露 offline update cost
5. **Warm retention**：frozen-v0 path retention 把 catalog onboarding 与 old-item forgetting 统一
6. **Rigorous protocol**：collision-hard-fail item evaluation、预注册 R² cost matching、frequency + temporal 双口径、test single-opening

### 可以 / 不可以声称

**可以**（达到对应 Gate 时）：R² 是有效 recoverability teacher；mass-weighted prefix soft transfer 提高 native collision-safe cold reachability；retention 改善 warm-cost/cold-gain trade-off；标准 GRAM beam 推理不依赖外部 drafter/verifier/edit memory。

**不可以**：完全 zero-shot 新物品；低成本动态 onboarding（除非 update 实验支持）；identifier construction 因果决定 failure depth（除非有 ID control）；universally beats SpecGR/GenRecEdit；当前 validation 等于 SOTA；多兴趣 resolver 是主创新；**有独立未污染第三域终验**。

---

## 12. 下一步唯一动作

**M1 三件事并行，不启动 GRAM full training：**

```text
experiment/phase14/protocol/item_level_eval.py        # 14-0A，从 item2lexid 建 collision multimap
experiment/phase14/protocol/oracle_prefix_probe.py    # 14-0B，主 Gate
experiment/phase14/tests/test_item_level_eval.py
experiment/phase14/tests/test_oracle_prefix_probe.py
experiment/phase14/configs/stage14_0_toys_beauty.json
# 14-0C：SpecGR compatibility audit + 同-backbone verifier control
```

已完成（本次规划期间）：

```text
experiment/phase14/protocol/cold_prefix_support.py         ✅
experiment/phase14/tests/test_cold_prefix_support.py       ✅ 8 tests OK
artifacts/phase14/diagnostics/cold_prefix_support_*.json   ✅
```

**不得从本计划直接跳到 full training。**

---

## 13. 核心参考文献

1. Lee et al. **GRAM.** ACL 2025. https://aclanthology.org/2025.acl-long.1596/
2. Ding et al. **Inductive Generative Recommendation via Retrieval-based Speculation (SpecGR).** AAAI 2026 Oral. https://arxiv.org/abs/2410.02939 ｜ code: https://github.com/Jamesding000/SpecGR
3. Peng et al. **Can Generative Recommendation Reach Cold Items?** arXiv 2026（非同行评审）. https://arxiv.org/abs/2607.21101 ｜ code: https://github.com/Lucas-PJ/GRColdItemReachability
4. Zhang et al. **Cold-Starts in Generative Recommendation: A Reproducibility Study.** SIGIR 2026. https://arxiv.org/abs/2603.29845 ｜ code: https://github.com/zhangzhen-research/ColdGenrec
5. Wang et al. **AGRec.** Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.369/
6. Lin et al. **Order-agnostic Identifier (SETRec).** SIGIR 2025. https://arxiv.org/abs/2502.10833
7. Fu et al. **Differentiable Semantic ID (DIGER).** SIGIR 2026. https://arxiv.org/abs/2601.19711
8. Mehta et al. **DSI++.** EMNLP 2023. https://aclanthology.org/2023.emnlp-main.510/
9. Liu et al. **USIM.** NeurIPS 2024.
10. Huang et al. **ALDI.** SIGIR 2023. https://doi.org/10.1145/3539618.3591732
11. Liu et al. **UCC.** SIGIR 2023. https://doi.org/10.1145/3539618.3592078
12. Zhou et al. **CCFCRec.** WWW 2023. https://doi.org/10.1145/3543507.3583286
13. Kim et al. **CGRC.** SIGIR 2024. https://doi.org/10.1145/3626772.3657801
14. Rajput et al. **TIGER.** NeurIPS 2023. https://arxiv.org/abs/2305.05065
15. Shen et al. **GenRecEdit: Adapting Model Editing for Generative Recommendation with Cold-Start Items.** 2026. https://arxiv.org/abs/2603.14259
16. Ding et al. **On the Memorization and Generalization of Generative Recommendation (MemGen-GR).** KDD 2026. https://github.com/Jamesding000/MemGen-GR

---

## 14. 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-19 | 新开第十四阶段，保留 R² 为有效点 | 双域正向证据未被否定 |
| 2026-08-19 | 停止 allocator/resolver 小调参主线 | Tier0/Tier1 已定位候选召回/path support 才是瓶颈 |
| 2026-08-19 | 主候选定为 R2PD | 与 GRAM 训练缺口、本地反证、文献最一致 |
| 2026-08-19 | Candidate verifier 降为 baseline | SpecGR 已覆盖 drafter-verifier |
| 2026-08-19 | pseudo user-sequence 降为备选 | USIM 已覆盖，且本地 pseudo-cold 曾失败 |
| **2026-08-20 / v0.2** | **诊断贡献降级为「复现+迁移」** | 2607.21101 已覆盖通用诊断；v0.3 进一步限定为 GRAM 迁移 + 受控扩展 |
| **2026-08-20 / v0.2** | **删除 warm≥0.97 硬门槛，改 Pareto frontier（v0.3 再改为 `G_c/C_w` Gate）** | 绝对阈值未经论证；但单点也不能称 frontier |
| **2026-08-20** | **Gate 精简；探索期允许调参** | 当前是探索模式，v0.1 用错治理强度 |
| **2026-08-20** | **真 SpecGR 提前到 M1** | 直接竞争者，代码开源，打平则无故事 |
| **2026-08-20 / v0.2** | **放弃 Sports（已被 v0.3 改为延期）** | Sports_cold50 不存在 + 8 卡全满；temporal Toys 仅 133 cold item |
| **2026-08-20 / v0.2** | **结构统计否定末层假设（已被 v0.3 撤销）** | 当时把 exact prefix overlap 错当成 learned NLL 上界；保留此行用于审计，不再作为路线依据 |
| **2026-08-20 / v0.2** | **“唯一干净轴”表述（已被 v0.3 收窄）** | 遗漏 GenRecEdit 与 offline update cost，不能继续使用未经限定的唯一性措辞 |
| **2026-08-20 / v0.3** | **撤销“结构统计否定末层假设”** | exact warm-prefix overlap 不是 learned NLL/beam 上界；路线只由 14-0B 决定 |
| **2026-08-20 / v0.3** | **补全 R2PD 训练算子** | synthetic cold-only prefixes 必须显式 teacher-forcing，并按 absolute prefix mass 加权 |
| **2026-08-20 / v0.3** | **collision evaluator 改用 multimap** | `lexid2cfid` 会静默覆盖 collision，不能作为 strict reverse map |
| **2026-08-20 / v0.3** | **加入 GenRecEdit 与 update-cost 边界** | cold SID 注入并非空白；部署优势必须同时计算 offline 更新成本 |
| **2026-08-20 / v0.3** | **Pareto 改为预注册 `G_c/C_w` Gate** | 单个 arm 不是 frontier；“同量级/方向一致”不可判定 |
| **2026-08-20 / v0.3** | **Sports 改为延期而非永久放弃** | 方法过双域前不投入；投稿时补第三域或明确两域限制 |

---

# 附录 A：v0.2 作者对 v0.3 的回评（2026-08-20）

> **性质**：这是 v0.2 撰写者收到 v0.3 后的回应，不是新版本。
> **总体结论**：**接受 v0.3 作为当前有效计划**。它指出的三处错误经独立核实全部成立，其中一处推翻了 v0.2 的核心论断。
> **待专家裁决**：A.2 的两项遗留问题 + A.3 的三条待确认项。每项都写了「建议方案」与「若驳回则需补什么」，可直接勾选。

---

## A.1 三处指控的独立核实结果（全部成立，无异议）

不是照单接受，逐条验证过：

| v0.3 的指控 | 核实方式 | 结论 |
|---|---|---|
| `lexid2cfid` 静默覆盖 collision | 复现其 dict 推导：`item2lexid={A:p1,B:p1,C:p2}` → `lexid2cfid` 只剩 2 条，A 被 B 覆盖 | **成立**。用它建 evaluator 等于用出问题的对象检测该问题 |
| 遗漏 GenRecEdit | 查证 arXiv 2603.14259 确实存在（人大，2026-03；training-free，约 9.5% 重训时间） | **成立** |
| 「结构统计是 learned NLL 上界」应撤销 | 见 A.1.1，新增逐层 token 口径实测 | **成立**，v0.2 的论证方式错误 |

### A.1.1 撤销「上界」论断的量化依据

v0.3 的反驳（decoder 可组合出从未完整出现过的 prefix）是对的。已补测「逐层 token 支持」这一宽松口径——只要每层 token 在该层被 warm 见过，trie 上路径即可走通：

| cold 累计断裂 ≤depth 2 | 完整前缀（v0.2 所用，严格） | 逐层 token（可组合空间，宽松） | 差 |
|---|---:|---:|---:|
| Toys | 67.8% | **32.3%** | 35.5 pp |
| Beauty | 82.5% | **47.5%** | 35.0 pp |

**双域均相差约 35 个百分点**，即「可组合但从未被完整观测」的空间极大。v0.2 声称的「真实断崖只会更浅不会更深」不成立。

**但有一点建议 v0.3 补记**：方向性先验并未被推翻。即使按最宽松的 token 口径，也只有 **Toys 26.8% / Beauty 16.3%** 的 cold item 能撑到末层。这不足以支撑任何 hard Gate，但足以作为 14-0B 的**预期与检验点**——若 14-0B 实测断崖恰好落在末层，反而说明结构先验与 learned 行为严重脱节，本身是需要解释的信号。建议在 §4 分支表加一行注记，不改变任何 Gate。

### A.1.2 已同步修正的产物（v0.2 遗留物，现已无误导）

```text
experiment/phase14/protocol/cold_prefix_support.py
  - 删除全部「上界」表述，改为「结构先验，不能单独否定任何路线」
  - 新增 deepest_token_supported()，输出严格/宽松双口径对比
experiment/phase14/tests/test_cold_prefix_support.py
  - 删除 test_break_is_shallow_not_terminal（承载已撤销论断）
  - 新增 TestTokenVsPrefixSupport，锁定 35pp 差距
  - 10 tests OK
artifacts/phase14/diagnostics/cold_prefix_support_{toys,beauty}.json  已重新生成
```

---

## A.2 v0.3 未处理的两个问题（建议在专家评审时一并裁决）

### A.2.1 【高】新增工作量未回算预算，M5 可能不可行

v0.3 新增了：SpecGR compatibility audit、GenRecEdit 适配、受控 ID control、3-seed test 批次、3 个 `μ_keep` operating points、update-cost benchmark。但 §9 资源表与 M1–M6 排期**未作任何调整**。

按 **v0.3 §9 自报的每 arm 时长**推算：

| 项 | 4 arm | 5 arm |
|---|---:|---:|
| M3 Toys（10–16h/arm） | 40–64h | 50–80h |
| M4 Beauty（26–31h/arm） | 104–124h | 130–155h |
| **M3+M4 小计** | **6.0–7.8 天** | **7.5–9.8 天** |
| **M5 若 3 seeds 需重训全部 arm ×2 域** | **18.0–23.5 天** | **22.5–29.4 天** |

**M5 是真正的风险点**，v0.3 §6 M5 只写了「数天」。且这还未计入：
- memory 记录的 **8 卡全满**（GPU3 仅 7MiB 空闲）；
- 血泪教训第 5 条：**共享 GPU 下 test inference 可慢 20×**；
- SpecGR / GenRecEdit 适配失败重试的时间。

**建议方案（三选一，请专家裁定）**：

| 选项 | 内容 | 代价 |
|---|---|---|
| **A（推荐）** | M5 的 3 seeds **只跑 v0 与 R2PD 主 arm 两个配置**；其余 baseline 用 M3/M4 的单 seed 结果并在表中明确标注 seed 数 | 主表 seed 不齐，需在 limitation 说明 |
| B | 保持全 arm 3 seeds，但**砍到 3 arm**（v0 / R² portfolio@2 / R2PD），SpecGR 与 GenRecEdit 降为文献对照 | 失去最强 baseline 的自跑数字 |
| C | 维持 v0.3 现状，但在 §9 写明 M5 需 18–29 天 GPU，并**预先取得用户对该预算的批准** | 与「3–6 个月投出」的硬约束正面冲突 |

无论选哪个，建议 v0.3 增加一条硬规则：**M1 结束时强制复核 arm 数与 M5 预算，未复核不得进入 M3。**

### A.2.2 【高】对 GenRecEdit 的差异化比 v0.3 所写更薄，且成本方向可能相反

v0.3 §11 第 4 条把卖点写成「推理时不需要 drafter、candidate verifier 或 edit-memory triggering」。

问题：**GenRecEdit 本来就没有 drafter 和 verifier**（它是权重编辑 + One-One trigger）。因此对 GenRecEdit 的差异只剩「无 edit-memory triggering」一项，即"推理时少一次 trigger 查表"。

更关键的是成本方向：

| | onboarding 方式 | 成本 |
|---|---|---|
| GenRecEdit | training-free 权重编辑 | 约 9.5% 重训时间 |
| R2PD | full fine-tuning | 100% 重训 |

**在 onboarding 成本这个维度上，R2PD 大概率是输的。** v0.3 §10 停止规则第 4 条已埋此雷（"成本倒置"），但它排在**路线级停止规则**里，意味着可能到 M3/M4 之后才触发。

**建议方案**：把「成本倒置」判定**提前为 M1 的一条 go/no-go**，与 SpecGR compatibility audit 并列。具体做法不需要跑实验，只需在 M1 结束时回答：

> 若 R2PD 相对 GenRecEdit 的全部优势只是「推理时少一次 trigger 查表」，而 onboarding 成本高一个数量级——这个组合是否还足以支撑一篇 CCF-B 的主故事？

若答案为否，应在投入 M3/M4 之前重新定位（例如把卖点从"部署成本"转向"user-conditioned 分布迁移的机制差异"，即 v0.3 §0.3 已写但未作为主轴的那一半）。

**若专家驳回此建议**，则需要补充说明：在 M3/M4 已消耗 6–10 天 GPU 之后才发现成本倒置，如何避免整个阶段沉没。

---

## A.3 三条较小的待确认项

| # | 事项 | 建议 |
|---|---|---|
| 1 | §0.2 保留了 `test_read: false` 的讨论，但该字段确实只是脚本自报 | 已同意 v0.3 的处理（改用依赖审计 + 输入 hash）。建议正式报告模板里直接加 `input_file_sha256` 字段，避免每次口头说明 |
| 2 | Toys 是 `c32_l5`、Beauty 是 `c128_l7`，**层深与分支数都不同** | v0.3 §6 M4 已注意到"c32/c128 几何不同"，但 §0.2 的深度表仍把两域并列呈现。建议表格加脚注：两域 depth 不可直接对齐，Beauty 的 depth 2 与 Toys 的 depth 2 语义不同 |
| 3 | §5 H1 的证伪条件写「NLL 与 warm 接近但 beam 仍失败 ⇒ 问题在 search」 | 建议补一个中间情形：**NLL 正常但 R² teacher 也无信息** ⇒ 此时 R2PD 与末层 head 均不适用，应直接触发 `FAIL_STOP`，而非落入"转 beam 侧"的模糊分支 |

---

## A.4 结论

**v0.3 在科学严谨性上优于 v0.2，建议以 v0.3 为基础继续。** A.1 的三条无异议。

请专家就以下三项裁决：

1. **A.2.1 预算**：选 A / B / C，或提出第四方案；是否加入「M1 结束强制复核」硬规则
2. **A.2.2 成本倒置**：是否把该判定从"路线级停止规则"提前为"M1 go/no-go"
3. **A.3**：三条小项是否采纳

裁决后由 v0.3 作者出 v0.4，或明确标注「以上意见已阅并驳回，v0.3 保持不变」。
