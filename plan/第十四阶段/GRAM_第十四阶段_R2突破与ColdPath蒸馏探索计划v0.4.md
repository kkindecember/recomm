# GRAM 第十四阶段：R²-to-Path Distillation 与冷路径原生可达性 v0.4

> **建立日期**：2026-08-20（v0.4 根据 v0.3 末尾专家回评形成；同日按附录 B 原地补全 ablation 预算）
> **当前状态**：`M1_STAGE14_0A_PASS / STAGE14_0B_PASS_PATH_TRANSFER_GATE / STAGE14_0C_PASS_INTERFACE_CONTROL / STAGE14_0D_PASS_WITH_M2_PENDING / NO_MODEL_TRAINING_STARTED`
> **工作名**：R2PD（R²-to-Path Distillation，暂定名；是否进入训练仍由 Stage 14-0B 决定）
> **前一版**：`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.3.md`（含专家回评，保留不删）
> **计划验证状态**：Stage14 evaluator/probe/verifier 已通过 24 项测试；14-0B 双域 validation 的 frozen parity、score-aware survival 与 tie-aware R² prior Gate 已通过；14-0C 双域同-backbone verifier 已完成并显示 frozen likelihood 会显著压低 cold top-10 placement；SpecGR/GenRecEdit 仅完成兼容性审计、未本地适配，R2PD 与所有模型训练结果仍未验证

---

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-20
- Verification Status: PARTIALLY_VERIFIED（M1 evaluator、path probe、interface control 与竞争审计已核对；R2PD 训练结论未验证）
- Version Label: phase14_code_plan_v0.4

---

## 0. v0.4 对专家回评的裁决（先读这一节）

v0.4 保留 v0.3 的方法与科学口径，接受专家回评 A.1 和 A.3 的全部事实修正；对两个高优先级问题作如下仲裁：

| 回评项 | 裁决 | v0.4 处理 |
|---|---|---|
| A.2.1 预算 | **接受问题，不选 A/B/C，采用 D：分阶段 seed promotion** | 全 baseline 先做双域 seed-0；仅达 promotion Gate 后扩主表到总计 3 seeds；M1 初锁 arm 与预算，M2 smoke 后定版 |
| A.2.2 成本倒置 | **部分接受** | 提前到 M1 做 story/resource go-no-go；但 GenRecEdit 论文报告的 9.5% 不能自动杀死机制不同的方法，R2PD 主轴改为 user-conditioned distribution transfer，成本作为透明 trade-off |
| A.3.1 输入 hash | **接受** | 正式 artifact 强制 `input_file_sha256.json` 与 `open_file_manifest.json` |
| A.3.2 raw depth 不可跨域直比 | **接受并加强** | 同时报 raw depth 与 normalized depth `d/L`；Toys depth 2 与 Beauty depth 2 不作等价解释 |
| A.3.3 NLL 正常且 teacher 无信息 | **接受** | 新增明确 `FAIL_STOP_PATH_TRANSFER` 分支，不再模糊转 beam |

附录 B 随后确认上述裁决成立，并指出 M5 ablation 未进入预算包络。该缺口已在本版主文原地解决：全套 ablation 仅限 Toys validation seed-0、最多 7 个新增训练 run；当前训练侧 all-in 上限为 466–629 GPU-hours（19–27 个顺序 GPU-days）。

### 0.1 十项冻结决策

| # | 决策 | 原因 |
|---|---|---|
| 1 | 诊断定位为**迁移 + 受控扩展** | 2607.21101 已做 oracle-prefix/taxonomy；GRAM 的 hierarchical lexical ID 仍是不同机制，但不能预称“相反结论” |
| 2 | **删除 warm ≥ 0.97 硬门槛**，改为预注册的 R² cost-matched 判据 | 绝对 retention 阈值未经论证；但不能用模糊的“Pareto 外移”替代 |
| 3 | **Gate 从 25+ 条压到每 Stage 1 主 Gate + 1 kill** | 当前是探索模式（见 memory `feedback_experiment_mode.md`），v0.1 用的是确认模式的治理强度 |
| 4 | **真 SpecGR 提前到 M1；恢复同 backbone verifier control** | 官方 SpecGR 的 TIGER/UniSRec 结果不能单独隔离训练期迁移与推理期 verifier |
| 5 | **GenRecEdit 纳入最近邻边界** | 它已覆盖 cold SID pattern 注入、warm preservation 与 update cost；“零推理外部依赖”不是未经限定的唯一轴 |
| 6 | **Sports 从核心阶段延期，不视为永久取消** | `Sports_cold50` 不存在；方法未过双域前不投入 30–45h，但最终论文须承认两域限制或补第三域 |
| 7 | **主线只由 14-0B 的真实 NLL/beam 诊断决定** | catalog prefix overlap 只提供结构先验，不能提前否定末层或 search 假设 |
| 8 | **结构统计使用 strict-prefix + layer-token 双口径** | 两者约 35pp 差距证实可组合空间很大；任何单一结构口径都不能代替 learned probe |
| 9 | **采用 staged evidence budget（方案 D）** | seed-0 全面对照先判方向；只有过 promotion Gate 才扩到 3 seeds，避免在失败路线预付全部 GPU 成本 |
| 10 | **M1 增加 competitor-story 与 arm-budget 双 Gate** | 在 full training 前确认 R2PD 不只剩“少一次 trigger 查表”，并由用户批准条件预算 |

### 0.2 已实测的新证据：结构先验，不是 learned failure 证明

**产出**：`experiment/phase14/protocol/cold_prefix_support.py`（只读 catalog，`test_read: false`）
**锁定**：`experiment/phase14/tests/test_cold_prefix_support.py`（10 tests，2026-08-20 本地复跑 OK）
**结果**：`artifacts/phase14/diagnostics/cold_prefix_support_{toys,beauty}.json`

`test_read: false` 只是脚本生成的 provenance 字段，不单独构成无泄漏证明；可信依据是代码依赖审计显示它只读取 identifier 与 `cold_split_meta/{cold,warm}_items.txt`。正式报告仍需记录输入文件 hash 和实际 open-file manifest。

对每个 cold item，问它的 lexical path 最深前缀 `z[:k]` 有多深仍被**至少一个 warm item** 共享：

| | Toys_cold50 | Beauty_cold50 |
|---|---:|---:|
| identifier | `c32_l5`（5 层，少数 6） | `c128_l7`（7 层，少数 8） |
| cold / warm item | 5963 / 5961 | 6052 / 6049 |
| baseline 重复路径 | **0** | **0** |
| 除末位外前缀唯一的 item 占比 | 87.3% | 95.4% |
| strict exact-prefix：cold 累计失去支持于 depth≤2 | **67.8%** | **82.5%** |
| loose layer-token：cold 累计失去支持于 depth≤2 | **32.3%** | **47.5%** |
| **cold「除末位外全路径」被 warm 支持** | **8.6%** | **2.66%** |

arXiv 2607.21101 在 TIGER/RQ-VAE 上观察到较晚位置的 fine-grained completion 困难；本项目的结构统计显示，GRAM cold path 往往更早失去**完整 warm-item prefix overlap**。这提示两类 identifier 的结构支持形态可能不同，但还没有证明 learned NLL 的断崖位置不同。

strict 与 loose 在两个域均相差约 35 个百分点，说明“每层 token 都见过、但完整 prefix 组合未见过”的空间很大。必须区分三层含义：

1. **已证据化**：67.8% / 82.5% 的 cold path 在 depth≤2 已无任何 warm item 共享该完整前缀；
2. **待 14-0B 验证**：模型的 token NLL、target rank 与 beam survival 是否也在相应位置恶化；
3. **需要受控 ID 对照才能声称**：identifier 构造方式“决定” failure depth。若不做同数据/同 backbone 的 ID control，只能写“associated with”。

神经 decoder 可能组合出从未完整出现过的 prefix，也可能在结构上有 warm overlap 的 prefix 处提前失败。因此该量**不是 learned reachability 或 NLL cliff 的严格上界**，不能单独否定末层 head、search failure 或 R2PD。

⚠️ **跨域深度不可直接对齐**：Toys 为 `c32_l5`（少数长度 6），Beauty 为 `c128_l7`（少数长度 8）；branch factor、总长均不同。所有诊断同时报告 raw depth `d` 与相对位置 `d/L`，跨域只比较 normalized profile。即使 loose 口径仍显示较多路径在 nominal terminal depth 前失去 token support，它也只能作为 14-0B 的预期；若 learned failure 反而集中在末层，应记录为“结构—模型行为脱节”，而不是修改 Gate 迎合结构先验。

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
| **cold path 在 depth≤2 失去 layer-wise token support** | **32.3%** | **47.5%** | strict/loose 相差约 35pp，证明完整 prefix 未见不等于 token 不可组合 |

### 1.2 已否定 / 必须降级的路线（**禁止恢复**）

> 这张表是 phase-13 最宝贵的沉淀。接手的 AI 必须先读，否则会重复已失败的 8 轮工作。**v0.4 继续保留这些负结果。**

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

| learned probe 结果 | teacher 信息 | 路线 |
|---|---|---|
| NLL/rank 在中层明显恶化 | R² projected mass/rank 高于最强 prior | **走 R2PD**：teacher 对失败 prefix 有可迁移信息 |
| NLL/rank 只在最后 1–2 层恶化 | R² 在末层无额外优势或信息集中于末层 | 转末层 identification head，R2PD 仅作 control |
| NLL/rank 接近 warm，但 beam survival/hit 失败 | R² 高于 prior | 问题偏 search/competition；只进入同-backbone candidate/verifier control，不启动 R2PD full training |
| NLL/rank 接近 warm或无可定位 profile | R² 不高于 prior | **`FAIL_STOP_PATH_TRANSFER`**：R2PD 与末层 head 均无依据，本阶段不模糊转 beam 搜索 |

结构 strict/loose profile 仅用于解释：若 learned failure 与两种结构先验明显错位，14-0B 报告必须把该错位作为结果，不得用结构统计重写分支。

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
| H1 | 瓶颈包含 learned unsupported decoder path | cold target NLL/rank 在可定位位置恶化；raw `d`、normalized `d/L` 与 strict/loose structure、beam survival、cold hit 的关系可报告但不要求全部同向 | NLL 接近 warm且 R² 无高于 prior 的信息 ⇒ `FAIL_STOP_PATH_TRANSFER`；若 R² 有信息但 beam 失败，只转同-backbone search control |
| H2 | R² item mass 可变成有效 prefix supervision | item-disjoint pseudo-cold 上 soft subtree target 优于 top-1 hard CE | soft 只降 teacher loss 不提 held pseudo-cold exact item ⇒ 停止 |
| H3 | warm forgetting 可被 retention 缓解 | 在 M5 预注册 `μ_keep` sweep 中，相近 `G_c` 下 `C_w` 降低，或相近 `C_w` 下 `G_c` 提高 | 三个冻结 operating points 均未改善任一 cost-matched 比较 ⇒ 承认 Pareto 极限 |
| H4 | 标准 native beam 可保留 R² 的主要增量收益 | 无 portfolio/drafter/edit-memory 时，R2PD 通过 §6 M3 的预注册 R² cost-matched 判据，且 overall 显著优于 v0 | 只比 v0 好但未达到 R²-transfer Gate ⇒ 仅为 native recovery；只有挂外部模块才有效 ⇒ 无独立主故事 |

**H4 是主卖点的直接检验。** 不再使用 `warm≥0.97×v0`，也不使用“同量级”“Pareto 外移”这类事后可移动措辞；M3 以 R² 的增量 cold gain 和实际 warm cost 为参照。

---

## 6. 执行计划（M1–M6）

### M1：诊断 + 竞争者（小 GPU）

**14-0A｜item-level evaluator 回归测试（约 1 天，非故事主线）**

> **执行更新（2026-08-20）**：✅ `PASS_STAGE14_0A`。Toys/Beauty v0 均为 0 duplicate path、0 integrity issue，严格 item 指标与历史保存指标最大差值均为 0；raw-v1 分别检出 932/719 个碰撞路径组，并移除 198/234 个字符串级 H@50 alias 命中。报告：`report/第十四阶段/GRAM_第十四阶段_Stage14-0A_Item级评测回归报告.md`。

本轮已确认 baseline identifier 重复路径 = 0（Toys/Beauty），但 raw v1 存在 collision，因此 evaluator 仍是**必要的回归测试**，不是贡献。

- 从 `item2lexid` 重建 `path -> [item_ids]` multimap；禁止直接信任会覆盖 collision 的 `lexid2cfid`
- duplicate path / ambiguous decoding / unknown item / top-K duplicate 一律 hard-fail
- 每次读取 identifier、cold/warm item list、prediction、checkpoint config 都写 `input_file_sha256.json` 与 `open_file_manifest.json`
- v0 历史 prediction 复算 parity；raw v1 的 alias hit 须在 strict evaluator 下消失
- **kill**：历史核心数字无法对齐 ⇒ 停止，先修口径

**14-0B｜断崖诊断（主 Gate）**

> **执行更新（2026-08-20）**：21 项测试通过；正式 Toys 8,789 / Beauty 10,655 validation users 均完成，frozen beam parity mismatch=0、未读 test。cold H@50 仅 1.03% / 1.31%，首次跌出 beam 中位均为 raw d=2；learned 断崖分别集中于 Toys d3 与 Beauty d2。tie-aware 配对统计显示 R² 在双域四个 normalized-depth quartile 的 target-prefix mass/rank 均显著优于最强固定 prior，裁决 `PASS_PATH_TRANSFER_GATE`。当前转入 14-0C/14-0D，二者完成前仍禁止模型训练。

Toys/Beauty validation 上只读计算：cold/warm target 的 token 级 NLL 与 rank；prefix@1..L survival 与首次跌出 beam 的 raw depth `d`、path length `L`、normalized depth `d/L`；v0 beam / R² top50 / 二者并集的 item recall；R² 分布投影到各层后的 entropy、target mass、subtree coverage；strict-prefix 与 layer-token support 仅作为协变量。

- **主 Gate**：learned NLL/rank 或 beam survival 的 failure profile 可定位，且 R² 对 target path 的 projected mass/rank 显著优于冻结的 uniform、popularity 与 catalog-text prior 中最强者
- **kill**：R² teacher 对目标 path 没有高于 prior 的支持，且 learned probe 不显示可被 teacher 修复的 path failure ⇒ `FAIL_STOP_PATH_TRANSFER`
- 产出 → `report/第十四阶段/Stage14-0_冷路径可行性诊断报告.md`，据 §4 分支表选路线

**14-0C｜竞争者与同-backbone interface control（与上并行）**

> **执行更新（2026-08-20）**：冻结 v0/R²、candidate budget=50、beam K=50，在 Toys 8,789 / Beauty 10,655 validation users 上完成完整 path mean-likelihood verifier。verifier 保留 R² cold H@50（8.27% / 8.10%），但相对 R² score-only 的 cold NDCG@10 分别下降 `−0.001895`（95% CI `[−0.002953,−0.000877]`）与 `−0.004142`（`[−0.005311,−0.003036]`）。裁决 `PASS_INTERFACE_CONTROL_COMPLETE_PATH_TRANSFER_STILL_NEEDED`：推理期候选接口不能替代训练期 path transfer。SpecGR/GenRecEdit 官方实现均需 TIGER→GRAM、split/SID/evaluator port，M1 不升为本地 arm。

先做 SpecGR compatibility audit：官方实现以 TIGER/UniSRec 和自己的数据 pipeline 为主，不能把原仓库数字直接与 GRAM/Toys 比较。只有相同 split、catalog、candidate budget、beam K、evaluator 与 cold definition 全部对齐后，才称“真 SpecGR baseline”。

并恢复冻结的同-backbone control：R² candidate score only、GRAM candidate likelihood only、R²+GRAM verifier、R² portfolio@2。它只回答“训练期 path transfer 是否优于推理期 candidate interface”，不作为创新。

- **competitor-story go/no-go**：M1 报告必须说明 R2PD 相对 GenRecEdit 的主差异是“visible-history-conditioned soft item distribution → batch prefix acquisition”，而非只剩“少一次 trigger 查表”；并区分 item-centric edit、额外 edit state/custom triggering、标准模型兼容性与预测质量这些轴
- GenRecEdit 的 `9.5% retraining time` 只能标为论文报告值，协议/硬件不同，不能直接作为本地自动 kill；但若审计后 R2PD 的可信优势确实只剩微小 lookup latency，`FAIL_STOP_STORY_DIFFERENTIATION`
- **SpecGR go/no-go**：若它已打平或打赢，且 R2PD 在机制、标准接口或 cost-matched quality 上均无可信优势，phase14 重新定位
- 同时它成为主表的强 baseline（符合「只选开源 baseline」约束）
- GenRecEdit 若存在可审计、可做同协议适配的公开实现，则进入 M1 compatibility audit；若无法公平复现，只做文献边界与 update-cost 对照，不伪造数字

**14-0D｜M1/M2 arm-budget lock（进入 M3 前硬门）**

> **执行更新（2026-08-20 / M1 初锁）**：core4 冻结为 v0 / R² portfolio@2 / same-backbone verifier / R2PD；SpecGR/GenRecEdit 当前不 promotion。当前 active package 为 358–488 GPU-hours；若未来 trainable competitor 通过 port，仍保留 466–629 GPU-hours 的全局 contingency ceiling。30G lease 规划 slowdown 暂取 `1.0–1.25×`，无 lease shared interval `1–20×` 不可作为 M3 执行方案。M2 必须用实测 step time 定版；M3 仍未获用户批准。报告见 `report/第十四阶段/Stage14_M1_竞争边界与资源冻结报告.md`。

M1 结束生成 `Stage14_M1_竞争边界与资源冻结报告.md` 初版；M2 smoke 后用实测 step time 定版。至少包含：可公平运行的 arm、每 arm 是否需训练、seed-0 实测/历史时长、额外 seed 成本、当前可用 GPU、`shared_gpu_slowdown_multiplier`、最小/推广预算、用户批准记录。slowdown 不得只写成定性风险，必须给出实测值或带依据的保守区间。

- seed-0 core 默认：v0 / R² portfolio@2 / 同-backbone verifier / R2PD；SpecGR 仅在 compatibility 通过后升为第 5 arm
- GenRecEdit 不因文献重要而自动占 GPU；只有本地实现、协议和 artifact 可审计才升为自跑 arm
- **硬 Gate**：arm 数、promotion set、最大 GPU-hours、最长 wall-clock envelope 未复核并获用户批准，禁止进入 M3

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

**14-2｜matched smoke**：固定 256–512 users、固定 checkpoint/steps。验证 loss 各分量有限且下降、synthetic cold-only prefix 的目标 token log-prob 上升、retention 梯度生效、生成 path 全部合法唯一可反解、显存与 runtime 达标；同时实测 step time、peak memory，并外推 R2PD full-update 成本区间，回填 14-0D 预算。

- **主 Gate**：`λ_cp = μ_keep = 0` 时与原 v0 behavior **逐位 parity**
- smoke 不以 H@10 的一两个事件判 efficacy

### M3：Toys full（30G lease）

**本阶段只跑 seed-0 screening，不预付另外两个 seeds。** 核心 arm：v0 / R² portfolio@2 / 同-backbone R²+GRAM verifier / R2PD 主 arm；通过 compatibility audit 的真 SpecGR 作为强 baseline。GenRecEdit 仅在同协议实现可审计时加入，不为凑表强行复现。

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

**同样只跑 seed-0 screening。** 迁移 Toys 冻结的算法、temperature/top-M 选择规则和 dimensionless loss-weight policy；只允许 domain-local catalog/embedding/index 重建。若因 c32/c128 几何不同必须改变绝对阈值，只能使用 M2 预先写明的归一化规则，不能看 Beauty validation 后手调。

- **主 Gate**：若 Toys 为 `PASS_NATIVE_RECOVERY`，Beauty 也必须达到 `PASS_NATIVE_RECOVERY`；若 Toys 为 `PASS_R2_TRANSFER`，Beauty 至少达到 native recovery 且 warm cost 不超过 R²。只有两域都达到 `PASS_R2_TRANSFER` 才能称“双域 R² transfer”；否则必须写 mixed evidence
- Beauty 已被 phase-13 多次查看，这是 source-domain confirmation，非独立终验

### M5：条件式 seed promotion + 补充实验 + 主表（30G lease）

**promotion Gate（方案 D）**：只有 Toys 与 Beauty seed-0 均达到 `PASS_NATIVE_RECOVERY`，且至少一个域达到 `PASS_R2_TRANSFER`，才把主表扩到总计 3 seeds。否则停止 seed expansion，保留单 seed 机制结果，不打开 test。

3-seed promotion set：

1. **必须**：v0、R2PD；M3/M4 已完成的 seed-0 计入总计 3 seeds，只新增 seed-1/2；
2. **R² portfolio@2**：在每个 v0 seed checkpoint 上重新评估；若 resolver 本身仍固定单 seed，必须单列披露，不把三次 checkpoint evaluation 写成 resolver 三 seed；
3. **一个最强 protocol-aligned competitor**：在 SpecGR 与同-backbone verifier 中按 M1 compatibility 预先确定。若该 competitor 需要训练，则 promotion 到相同 3 seeds；若不能公平训练，只能留在单-seed/机制表，不能与三 seed 主结果做同等显著性比较；
4. **不 promotion**：A1/A2/A3 之外的 ablation、3 个 `μ_keep` operating points、失败的 compatibility arm，均只在 **Toys validation seed-0** 报告；Beauty 不重复全套 ablation。

M5 报告必须分别记录 `resolver_seed_count` 与 `backbone_checkpoint_seed_count`。R² portfolio@2 在三个 v0 checkpoint 上重评，不等于 resolver 自身训练了三个 seeds；正文、表格和 artifact 均不得混写。

- **test freeze**：所有 promotion checkpoint 训练完成后，写入 commit/config hash、seed list、全部主表 arm、metric code hash 与 exclusion rules；Toys/Beauty test jobs 作为一个批次启动，全部结束前不查看中间 test 指标
- **temporal 小节**：用 ColdGenrec 开源脚本，只跑 v0 + 冻结主 arm。⚠️ Toys temporal 仅 133 cold item，只作方向性外部有效性，不单独声称显著
- **ablation / Pareto sweep（仅 Toys validation seed-0）**：最多新增 7 个 full-training runs——hard teacher 1、无 retention（`μ_keep=0`）1、无 confidence 1、预注册 prefix-level 变体至多 1、teacher strength 两侧各 1、第三个 `μ_keep` operating point 1。主 R2PD 与 `μ_keep=0` 分别复用为 soft/reference 和 3-point sweep 中的两个点，不重复计费。prefix-level 只有在纯评估期截断且不改变训练监督时才可记为 inference-only；否则占用上述训练名额。Beauty 不重复全套 ablation，test 主表不追着最优点重开
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
10. failure depth 同时报 raw `d`、path length `L`、normalized `d/L`；不同 branch factor/长度的域不直接比较 raw depth
11. 3-seed 主表逐 seed 报告并给 mean±std；paired user bootstrap 在每个 matched seed 内完成，禁止把同一用户的三个 seed 当独立样本池化。最终 robustness 只有在 3/3 seed effect 同方向时写“seed-robust”，否则写 mixed-seed evidence

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
| 14-0D budget lock | CPU | M1 初版；M2 smoke 后定版 | 否 |
| 14-1 / 14-2 | 单小 GPU | 数小时 | 否 |
| M3/M4 seed-0 core 4 arms | GRAM full | 144–188 GPU-hours（顺序约 6.0–7.8 天） | **是** |
| M3/M4 seed-0 若加第 5 arm | GRAM full | 合计 180–235 GPU-hours（7.5–9.8 GPU-days） | **是** |
| M5 v0+R2PD 新增 seed-1/2 | GRAM full | 额外 144–188 GPU-hours（6.0–7.8 GPU-days） | **是，过 promotion Gate 才发生** |
| M5 最强 trainable competitor 新增 seed-1/2 | full train | 额外 72–94 GPU-hours（3.0–3.9 GPU-days） | **条件式** |
| M5 Toys seed-0 ablation（最多 7 个新增训练 run） | GRAM full | 额外 70–112 GPU-hours（2.9–4.7 GPU-days） | **是，过 promotion Gate 才发生** |
| M5 inference/temporal | inference / 混合 | 另计；共享 GPU inference 可能慢 20× | 视任务 |

因此 v0.4 的条件预算不是含糊的“数天”：seed-0 五 arm 上界为 180–235 GPU-hours；promotion 后 v0+R2PD 新增 144–188；最强 trainable competitor 若扩至三 seed 再加 72–94；Toys-only ablation 最多再加 70–112。训练侧 all-in 最大包络为 **466–629 GPU-hours，即 19.4–26.2 个顺序 GPU-days（对外取整写 19–27 天）**。该数仍不含 inference/temporal 另计项与共享环境 slowdown；批准记录必须用 `shared_gpu_slowdown_multiplier` 把顺序 GPU-days 转成现实 wall-clock 区间。预算**分阶段批准，不一次性承诺**。

**硬规则**：14-0D 在 M1 形成初版、M2 smoke 后用实测 step time 定版；未完成 arm/seed/GPU-hours/slowdown-adjusted wall-clock 复核并取得用户明确批准，不得启动 M3。M3/M4 未过 promotion Gate，不得启动额外 seeds 或 Toys ablation；Beauty 全套 ablation 不在本阶段授权范围内。

>10 分钟实验沿用：独立 tmux、hard timeout、status.json、telemetry、**不自动重试**。大显存实验启动前报告预计占用并由用户指定 GPU；**不得停止他人进程或调整 holder，除非用户明确授权**。

**已知坑**（memory `feedback_runner_tmux_bug.md`）：`run_phase13_explore.sh start` 用 tmux 拉 launch_cmd 可能 10s 内 exit=1，改用 `setsid nohup` 绕开；`finish()` 不检查 workload_rc（CUDA OOM 会误判成功）；status.json 的 workload_pid 恒为 0；**判活用 workload PID，不要用 tmux session**。

每个正式 artifact 至少含：`config.json` `manifest.json` `status.json` `summary.json` `run.log` `gpu_telemetry.csv` `predictions_*.jsonl` `item_path_audit.json` `data_provenance.json` `input_file_sha256.json` `open_file_manifest.json`。资源冻结报告的 `summary.json` 强制包含 `shared_gpu_slowdown_multiplier`；M5/R² artifact 强制包含 `resolver_seed_count` 与 `backbone_checkpoint_seed_count`。

---

## 10. 停止规则（路线级 5 条）

1. **泄漏**：actual cold interaction 或 held target 进入训练/teacher/置信模型
2. **收益来自 alias**：方法增益只来自 ambiguous lexical path
3. **无法归因**：需同时改 ID、decoder、resolver、split 才能得正结果
4. **M1 无机制差异化**：文献/代码审计后，R2PD 相对 GenRecEdit/SpecGR 只剩微小 lookup latency，无法提出 user-conditioned distribution transfer、标准接口或质量上的独立可检验主张
5. **cost-quality 同时被支配**：M2 成本外推或 seed-0 本地结果显示 R2PD update 明显更贵，同时在质量、warm trade-off、标准模型兼容性上均无优势。GenRecEdit 论文报告的 9.5% 单独不触发此条

（各 Stage 另有自己的 1 条 kill，见 §6）

---

## 11. 论文故事（仅结果成立后采用）

1. **GRAM cold-path failure profile**：把 [2607.21101] 的可达性分析迁移到 hierarchical lexical ID；只有 learned NLL/beam 与受控 ID evidence 成立时才讨论与 RQ-VAE 的差异。无 ID control 时只写“结构 overlap 与 failure depth 相关”
2. **R² recoverability probe**：warm-only inductive resolver 证明用户历史中存在可恢复的 cold preference，但外部 portfolio 有 warm trade-off 与候选天花板
3. **★ History-conditioned probabilistic path acquisition**：R²-to-Path Distillation 把每个可见用户历史下的 soft item distribution 迁入 prefix-conditional decoder probability；区别于 item-centric pseudo-history editing 或推理期 candidate interface
4. **标准 native beam 是部署性质，不是总成本胜利宣言**：推理时不需要 drafter、candidate verifier 或独立 edit-memory triggering，但完整披露 full-update cost；除非实测支持，不声称 onboarding 比 GenRecEdit 更便宜
5. **Warm retention**：frozen-v0 path retention 把 catalog onboarding 与 old-item forgetting 统一
6. **Rigorous protocol**：collision-hard-fail item evaluation、预注册 R² cost matching、frequency + temporal 双口径、test single-opening

### 可以 / 不可以声称

**可以**（达到对应 Gate 时）：R² 是有效 recoverability teacher；history-conditioned、mass-weighted prefix soft transfer 提高 native collision-safe cold reachability；retention 改善 warm-cost/cold-gain trade-off；标准 GRAM beam 推理不依赖外部 drafter/verifier/独立 edit memory。

**不可以**：完全 zero-shot 新物品；低成本动态 onboarding（除非 update 实验支持）；identifier construction 因果决定 failure depth（除非有 ID control）；universally beats SpecGR/GenRecEdit；当前 validation 等于 SOTA；多兴趣 resolver 是主创新；**有独立未污染第三域终验**。

---

## 12. 下一步唯一动作

M1 已完成。下一步进入 M2：先做 14-1 pseudo-cold mechanism unit test / CPU 数据审计，再做 14-2 固定 256–512 users 的 matched smoke；只在 smoke 后用实测 step time 定版 14-0D。M3 full training 在预算复核和用户明确批准前仍禁止启动。

**M1 四件事已完成，未启动 GRAM full training：**

```text
experiment/phase14/protocol/item_level_eval.py        # 14-0A，从 item2lexid 建 collision multimap
experiment/phase14/protocol/oracle_prefix_probe.py    # 14-0B，主 Gate
experiment/phase14/tests/test_item_level_eval.py
experiment/phase14/tests/test_oracle_prefix_probe.py
experiment/phase14/configs/stage14_0_toys_beauty.json
# 14-0C：SpecGR compatibility audit + 同-backbone verifier control
# 14-0D：竞争故事 + arm/seed/GPU-hours 预算初版（M2 smoke 后定版）
```

已完成（本次规划期间）：

```text
experiment/phase14/protocol/cold_prefix_support.py         ✅
experiment/phase14/tests/test_cold_prefix_support.py       ✅ 10 tests OK（strict-prefix + layer-token）
artifacts/phase14/diagnostics/cold_prefix_support_*.json   ✅
experiment/phase14/protocol/item_level_eval.py              ✅
experiment/phase14/tests/test_item_level_eval.py             ✅（与结构测试合计 15 tests OK）
artifacts/phase14/diagnostics/item_level_eval/                ✅ PASS_STAGE14_0A
report/第十四阶段/GRAM_第十四阶段_Stage14-0A_Item级评测回归报告.md ✅
experiment/phase14/protocol/oracle_prefix_probe.py             ✅ formal dual-domain verified
experiment/phase14/protocol/synthesize_oracle_prefix_probe.py  ✅ tie-aware dual-domain synthesis
experiment/phase14/tests/test_oracle_prefix_probe.py            ✅（phase14 合计 21 tests OK）
artifacts/phase14/diagnostics/oracle_prefix_probe_toys_smoke_gpu0_retry4/ ✅ PASS（2 users；score-aware；beam parity mismatch=0）
artifacts/phase14/diagnostics/oracle_prefix_probe_toys_medium_gpu0_score_aware/ ✅ PASS（128 users）
report/第十四阶段/GRAM_第十四阶段_Stage14-0B_Toys_Smoke报告.md ✅
artifacts/phase14/diagnostics/oracle_prefix_probe_formal_dual_domain_score_aware_recovery/ ✅ PASS（双域；parity mismatch=0）
artifacts/phase14/diagnostics/oracle_prefix_probe_tie_aware_teacher_correction/ ✅ PASS（tie-aware teacher correction）
report/第十四阶段/Stage14-0_冷路径可行性诊断报告.md ✅ PASS_PATH_TRANSFER_GATE
experiment/phase14/protocol/same_backbone_verifier.py ✅ dual-domain verified
experiment/phase14/tests/test_same_backbone_verifier.py ✅（phase14 合计 24 tests OK）
artifacts/phase14/controls/same_backbone_verifier_formal_dual_domain_gpu5_recovery/ ✅ PASS_INTERFACE_CONTROL_COMPLETE_PATH_TRANSFER_STILL_NEEDED
artifacts/phase14/m1/resource_lock/ ✅ PASS_WITH_M2_PENDING
report/第十四阶段/Stage14_M1_竞争边界与资源冻结报告.md ✅
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
| **2026-08-20 / v0.4** | **采纳 strict-prefix + layer-token 双结构口径** | 双域约 35pp 差距证实可组合空间；raw depth 不跨域直接对齐 |
| **2026-08-20 / v0.4** | **预算采用方案 D：staged seed promotion** | seed-0 全面对照先筛；双域过 Gate 后只扩主表核心方法到总计 3 seeds |
| **2026-08-20 / v0.4** | **成本倒置提前到 M1，但不以外部 9.5% 自动 kill** | 先验证机制主张不只剩 trigger lookup；最终以本地 cost-quality 是否同时被支配判断 |
| **2026-08-20 / v0.4** | **正式 artifact 增加输入 hash/open manifest** | `test_read:false` 是自报字段，不能替代可审计 provenance |
| **2026-08-20 / v0.4** | **新增 `FAIL_STOP_PATH_TRANSFER` 明确分支** | NLL 无 path failure 且 R² 不高于 prior 时，R2PD/末层 head 均无依据 |
| **2026-08-20 / v0.4 原地预算修订** | **ablation 限定 Toys seed-0、最多 7 个新增训练 run** | 接受附录 B.2；补入 70–112 GPU-hours，训练侧 all-in 包络修正为 466–629 GPU-hours（19–27 顺序 GPU-days） |
| **2026-08-20 / M1 14-0A** | **item-level evaluator 回归通过，允许进入 14-0B** | 双域 v0 严格 parity 精确成立；raw-v1 alias 再确认，正式评测默认 collision hard-fail |
| **2026-08-20 / M1 14-0B smoke** | **Toys 2-user 诊断端到端通过** | frozen beam parity mismatch=0，未读 test；仅验证管线，不据 2-user 样本做路线判断 |
| **2026-08-20 / M1 14-0B formal** | **双域 `PASS_PATH_TRANSFER_GATE`** | cold learned failure 可定位；tie-aware 配对统计下 R² prefix mass/rank 双域全 normalized quartile 显著优于最强固定 prior；转 14-0C/14-0D，不直接训练 |
| **2026-08-20 / M1 14-0C formal** | **`PASS_INTERFACE_CONTROL_COMPLETE_PATH_TRANSFER_STILL_NEEDED`** | verifier 保留 R² cold H@50，但双域 cold NDCG@10 均显著低于 R² score-only；冻结 GRAM likelihood 不能替代 path transfer |
| **2026-08-20 / M1 14-0D 初锁** | **core4 与条件预算冻结，M3 未授权** | SpecGR/GenRecEdit 兼容性未通过；active package 358–488 GPU-h，M2 实测后再定版并申请 M3 批准 |

---

# 附录 B：v0.3 回评者对 v0.4 的确认（2026-08-20）

> **性质**：附录 A 的提出者对 v0.4 裁决的回应，不是新版本。
> **结论**：**A.1 / A.2 / A.3 全部争议已解决，v0.4 可作为执行版本。** 三项裁决均已核实，其中方案 D 优于我原提的 A/B/C。
> **原回评剩余项**：一个数字缺口（B.2）与两条执行期提醒（B.3），均不构成阻塞。
> **主文落实状态**：2026-08-20 已按用户授权原地修正。B.2 采用 Toys seed-0、最多 7 个新增训练 run；B.3 两项已转为强制 artifact 字段。以下保留专家原回评，供审计。

---

## B.1 三项裁决的核实

### B.1.1 A.2.1 预算 —— 方案 D 优于我提的三个选项，接受

我原提 A（砍 seed）/ B（砍 arm）/ C（认预算），三个都是**静态**取舍：在不知道方法是否有效时先决定砍什么。

v0.4 的方案 D（staged seed promotion）把决策**改成条件式**：seed-0 先全面对照，只有双域过 `PASS_NATIVE_RECOVERY` 且至少一域过 `PASS_R2_TRANSFER` 才扩 seed。**这比我的三个选项都好** —— 失败路线上不预付 statistical rigor 的成本，成功路线上不牺牲主表质量。

§9 的预算表已逐行复算，与 v0.3/v0.4 自报的每 arm 时长一致：

| 项 | v0.4 所写 | 复算 |
|---|---|---|
| seed-0 core 4 arm | 144–188 GPU-h（6.0–7.8 d） | ✅ |
| seed-0 5 arm | 180–235 GPU-h（7.5–9.8 d） | ✅ |
| M5 v0+R2PD seed-1/2 | 144–188 GPU-h（6.0–7.8 d） | ✅ |
| M5 competitor seed-1/2 | 72–94 GPU-h（3.0–3.9 d） | ✅ |
| **最大包络** | 15–22 顺序 GPU-days | 复算 **16.5–21.5**，✅ |

14-0D 作为「M1 出初版、M2 smoke 后用实测 step time 定版、未批准不得进入 M3」的硬门，正是我建议的「M1 结束强制复核」，且比我提的更严 —— 加了 M2 实测回填。**接受，无保留。**

### B.1.2 A.2.2 成本倒置 —— 部分接受的理由成立，我原提法确实过强

v0.4 指出：GenRecEdit 论文报告的 9.5% 是**不同协议、不同硬件**下的数字，不能自动 kill 一个机制不同的方法。**这个反驳是对的，我原来的表述把"文献报告值"当成了"本地可比基准"。**

更重要的是 v0.4 把主轴从「部署成本」换成了 **history-conditioned probabilistic path acquisition**（§11 第 3 条）。这解决了我担心的根本问题：

| | GenRecEdit | R2PD |
|---|---|---|
| 注入对象 | item-centric，cold SID pattern | **user-conditioned soft distribution** |
| 触发 | One-One trigger + edit state | 标准 beam，无额外状态 |

差异不再是「少一次查表」这种量级的东西，而是**注入什么**的差异。这是可辩护的机制主张。

同时 §10 第 5 条把 kill 条件收紧为「cost **与** quality **同时**被支配」，而非单看成本 —— 逻辑正确。§11 第 4 条明写「标准 native beam 是部署性质，不是总成本胜利宣言」，也堵住了我担心的过度声称。**接受。**

### B.1.3 A.3 三条小项 —— 全部采纳，其中一条被加强

| # | 我的建议 | v0.4 处理 |
|---|---|---|
| 1 | 报告模板加 `input_file_sha256` | ✅ 加了，且追加 `open_file_manifest.json` |
| 2 | 层深不可跨域对齐加脚注 | ✅ **超出建议**：不止加注，而是要求所有诊断同时报 raw `d` 与 normalized `d/L`（§8 第 10 条），跨域只比 normalized |
| 3 | H1 补「NLL 正常且 teacher 无信息」分支 | ✅ 新增 `FAIL_STOP_PATH_TRANSFER`，§4 分支表从 3 行扩到 4 行，四种组合全覆盖 |

§4 分支表现在是 (NLL profile) × (teacher 信息) 的完整二维划分，没有落空区间。**这比我提的更完整。**

另外注意到 v0.4 §0.2 采纳了我在 A.1.1 建议的「结构—模型行为脱节」提法，且写法更严谨：明确要求「若 learned failure 反而集中在末层，应记录为脱节，而不是修改 Gate 迎合结构先验」。

---

## B.2 一个数字缺口：M5 ablation 未计入预算包络（建议补，不阻塞）

§6 M5 列出的 ablation 是：`hard vs soft`、`无 retention`、`无 confidence`、`prefix level`、`teacher strength`，加 3 个 `μ_keep` operating points。

**这些绝大多数需要重新训练**（改 loss 权重、改监督形式都不是 inference-time 可得），约 7 个训练 run。但 §9 只把它记作「另计」，未进入 15–22 天的包络：

| 范围 | 额外成本 |
|---|---|
| 仅 Toys seed-0 | 70–112 GPU-h（**2.9–4.7 d**）|
| 双域 seed-0 | 252–329 GPU-h（**10.5–13.7 d**）|

若双域做全套 ablation，实际包络会从 15–22 天变成 **26–36 天**，超出一倍。

**建议（三选一，不阻塞进入 M1）**：

1. **§6 M5 明写 ablation 仅在 Toys seed-0 做**，Beauty 不做 —— 成本可控（+3–5 天），且 ablation 本就是机制说明而非跨域主张；
2. 或 §9 增加一行 `M5 ablation ~7 runs`，把包络诚实改为 26–36 天并同样纳入 14-0D 批准范围；
3. 或明确哪几项可用 inference-time 近似（例如 `prefix level` 若只是评估时截断不同深度，可能不需重训），只对真正需要训练的项计费。

**推荐 1**，与 v0.4「ablation 只在 validation seed-0 报告」的定位一致，只是把「seed-0」进一步限定为「Toys seed-0」。

> **v0.4 主文处理记录（2026-08-20）**：已采纳方案 1，并结合方案 3 冻结最多 7 个新增训练 run。§6 将全套 ablation 限定为 Toys validation seed-0；§9 增加 70–112 GPU-hours，训练侧 all-in 包络修正为 466–629 GPU-hours（19.4–26.2 顺序 GPU-days，对外写 19–27 天）。

---

## B.3 两条执行期提醒（已落实为 artifact 字段）

1. **§9 的 GPU-days 是顺序估计**，未计 memory 记录的 8 卡全满与「共享 GPU 下 test inference 慢 20×」。14-0D 定版时应把 slowdown 作为显式乘数写进批准记录，而不是留作口头风险。

2. **§6 M5 promotion 第 2 条**（R² portfolio@2 在每个 v0 seed checkpoint 上重评，但 resolver 本身仍单 seed，必须单列披露）—— 这条很容易在写作时被简化成「R² 也跑了 3 seeds」。建议在 M5 的 report 模板里预留一个强制字段记录 resolver seed 数，靠模板而非记忆来保证。

---

## B.4 结论

**v0.4 的三项裁决全部成立，附录 A 提出的问题已解决，无遗留争议。**

- A.2.1：方案 D 优于我提的 A/B/C，且 14-0D 双阶段定版比我的建议更严
- A.2.2：对「9.5% 不能自动 kill」的反驳正确；主轴改为 history-conditioned distribution transfer 后，差异化不再依赖成本论证
- A.3：三条全采纳，其中 normalized depth 与 `FAIL_STOP_PATH_TRANSFER` 都超出原建议

**建议按 v0.4 执行。** B.2 的 ablation 预算缺口建议在 14-0D 初版时一并处理（推荐限定为 Toys seed-0），不必为此再出 v0.5。

> **执行状态（2026-08-20）**：上述 B.2/B.3 建议均已在 v0.4 主文落实；当前有效预算以 §9 的 466–629 GPU-hours（19–27 顺序 GPU-days）为准，附录中的 15–22 天仅是修订前历史数字。

---
