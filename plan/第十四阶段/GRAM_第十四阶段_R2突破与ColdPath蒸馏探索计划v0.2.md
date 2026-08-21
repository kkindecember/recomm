# GRAM 第十四阶段：冷路径原生可达性 v0.2

> **建立日期**：2026-08-20（v0.1 为 2026-08-19）
> **当前状态**：`PLAN_ONLY / NO_PHASE14_EXPERIMENT_STARTED`
> **工作名**：R2PD（R²-to-Path Distillation，暂定名，最终路线由 Stage 14-0B 诊断决定）
> **前一版**：`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.1.md`（保留不删，便于对照）

---

## 0. v0.2 相对 v0.1 改了什么（先读这一节）

v0.1 的论证质量高，但有四个会致命的问题，且在规划期间实测出一条**推翻 v0.1 技术前提**的新证据。

### 0.1 六项决策

| # | 决策 | 原因 |
|---|---|---|
| 1 | **诊断贡献降级**为「复现 + 迁移」 | arXiv 2607.21101 已完整做过 oracle-prefix 与深度 taxonomy，数据集重叠、代码开源。v0.1 只标注"非同行评审"就放过，低估了 |
| 2 | **删除 warm ≥ 0.97 硬门槛**，只报 Pareto frontier | phase-13 自己认定该阈值"未经论证"并因此误杀 P1–P7，v0.1 却又装了回去 |
| 3 | **Gate 从 25+ 条压到每 Stage 1 主 Gate + 1 kill** | 当前是探索模式（见 memory `feedback_experiment_mode.md`），v0.1 用的是确认模式的治理强度 |
| 4 | **真 SpecGR 提前到 M1** | 它是直接竞争者且代码开源；若它已打平，phase14 没有故事。必须第 1 个月知道 |
| 5 | **放弃 Sports；temporal 降为 M5 补充小节** | `Sports_cold50` 不存在，需 30–45h v0 训练 + 整条 pipeline，8 卡全满排不进 |
| 6 | **主线由诊断分支决定**，不预先押 R2PD | 见 0.2 —— 但新证据已把预期分支指向 R2PD |

### 0.2 规划期间实测的新证据（改变了技术判断）

**产出**：`experiment/phase14/protocol/cold_prefix_support.py`（只读 catalog，`test_read: false`）
**锁定**：`experiment/phase14/tests/test_cold_prefix_support.py`（8 tests，OK）
**结果**：`artifacts/phase14/diagnostics/cold_prefix_support_{toys,beauty}.json`

对每个 cold item，问它的 lexical path 最深前缀 `z[:k]` 有多深仍被**至少一个 warm item** 共享：

| | Toys_cold50 | Beauty_cold50 |
|---|---:|---:|
| identifier | `c32_l5`（5 层，少数 6） | `c128_l7`（7 层，少数 8） |
| cold / warm item | 5963 / 5961 | 6052 / 6049 |
| baseline 重复路径 | **0** | **0** |
| 除末位外前缀唯一的 item 占比 | 87.3% | 95.4% |
| **cold 累计断裂于深度 ≤2** | **67.8%** | **82.5%** |
| **cold「除末位外全路径」被 warm 支持** | **8.6%** | **2.66%** |

**这推翻了 v0.1 的一个隐含前提，也推翻了本轮评审最初的建议。**

arXiv 2607.21101 在 TIGER/RQ-VAE 上报告瓶颈是**末层** fine-grained path completion（粗层码本共享，模型落对区域、死在最后一两位）。据此本应做末层 identification head。

**GRAM 不是这样。** 冷路径在**深度 1–3 就已脱离 warm 支持**，只有 8.6% / 2.66% 的 cold item 是「只差最后一个 token」。根因是 identifier 构造方式不同：GRAM 用 item text embedding 上的层级 k-means，分裂到叶子近乎唯一（除末位外前缀唯一者占 87.3% / 95.4%），没有 RQ-VAE 那种共享粗码本。

三个后果：

1. **末层 identification head 方案大概率无效** —— 断的不是末层；
2. **R2PD 可行性反而提高** —— 断在中层，正是 BGE 文本 teacher 有信息的区间（v0.1 §4.1 风险 3「arbitrary suffix 难泛化」只对那 8.6% 成立，不是主要矛盾）；
3. **「与 RQ-VAE 相反的断裂模式」本身是可发表的小贡献** —— 说明 identifier 构造方式决定冷路径在哪里失败。

⚠️ **该量是 path support 的上界式刻画**：只看 identifier 结构，不看模型是否真把概率放上去。真实 NLL 断崖只会更浅、不会更深。因此它**足以否定末层假设**，但**不能替代 Stage 14-0B**（仍需实测断崖位置与 beam 存活）。

### 0.3 论文主卖点（唯一干净的差异化轴）

诊断被 2607.21101 占了，drafter-verifier 被 SpecGR 占了。剩下的干净轴是：

> **SpecGR 推理时必须挂 drafter；本方法把冷物品接入做成一次性训练期操作，推理时是纯原生 GRAM beam，零额外开销。**

可测（推理延迟对比 + 移除外部检索器后的性能），符合 CCF-B「清晰问题定位 + 一个结构性但可控的改动」。**全部实验围绕这条组织。**

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
| **cold 断裂深度 ≤2 占比**（本轮新增） | **67.8%** | **82.5%** | **断在中层，非末层** |

### 1.2 已否定 / 必须降级的路线（**禁止恢复**）

> 这张表是 phase-13 最宝贵的沉淀。接手的 AI 必须先读，否则会重复已失败的 8 轮工作。**v0.2 原样保留。**

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
| `GRAM/src/data/multi_task_dataset_gram.py:93-99` | `item2cfid` / `lexid2cfid` | **item-level 反解映射已存在，evaluator 直接复用，不需新写** |
| `GRAM/src/processor/Collator.py` | target tokenization | 新 collator 携带 sparse prefix targets、confidence、provenance |
| `GRAM/src/utils/generation_trie.py`(155行) | legal-token mask | 增加只读 subtree index / descendant mass 工具，不改合法性语义 |
| `GRAM/src/utils/evaluate.py`(58行) | lexical string match | 新 item-level evaluator（复用 `lexid2cfid`） |
| `GRAM/src/data/multi_task_dataset_rec.py` | warm CE 样本 | **保持冻结**，在外层 wrapper 生成 teacher 分布 |

纪律：新代码放 `experiment/phase14/`；若必须改 GRAM 内部，用最小 patch + feature flag，并先存 behavior parity test。

---

## 3. 文献边界

### 3.1 直接约束

| 工作 | HOW | 对本阶段的约束 |
|---|---|---|
| **GRAM**, ACL 2025 | lexical hierarchy + collaborative verbalization + late fusion | 原 backbone；缺 zero-interaction path supervision |
| **SpecGR**, AAAI 2026 Oral | inductive drafter + GR teacher-forcing verifier + guided redrafting | **最近邻竞争者，代码开源**。drafter+verifier 不能再当新贡献。→ M1 必跑 |
| **2607.21101**, arXiv 2026 | temporal split、coldness taxonomy、oracle-prefix probing | **诊断已被做完**。但其结论（末层瓶颈）基于 RQ-VAE，**与本项目 §0.2 实测相反** —— 这是我们的对照点 |
| **ColdGenrec**, SIGIR 2026 | 统一 cold protocol、factor-wise controls，代码开源 | 要求一次只改一因素、报 warm/cold。其 temporal split 供 M5 使用（⚠️ Toys 仅 133 cold item） |
| AGRec, Findings ACL 2025 | GNN logits 增强 + rankable FSM | 推理期 logit fusion 已拥挤，只适合作强 baseline |
| SETRec / DIGER, SIGIR 2025/2026 | order-agnostic set ID / 可微 SID | 重做 ID 体系赛道已拥挤且昂贵，不走 |
| DSI++, EMNLP 2023 | pseudo-query rehearsal | 冷路径注入必须配 warm retention |
| ALDI / UCC / CCFCRec / CGRC | cold-warm 蒸馏对齐、不确定性、对比迁移、伪冷重构 | 提供 retention/confidence/pseudo-cold 设计参考 |

### 3.2 已不够新的故事（不要写）

- content retriever 提 cold candidate + GRAM 排序 → SpecGR
- auxiliary score 加到 decoder logits → AGRec
- 想象冷 item 的用户序列 → USIM
- 重设计 semantic ID → SETRec / DIGER
- 多兴趣 resolver 提召回 → 检索模块优化，非 GRAM 机制

---

## 4. 主线：诊断驱动分支（决策 6）

**Stage 14-0B 唯一任务：在 GRAM 上实测 cold target path 的 NLL 断崖深度与 beam 存活深度。**

分支规则**事先写死，不可事后修改**：

| 实测断崖深度 | 路线 |
|---|---|
| **中层（depth 2–3）← §0.2 预期** | **走 R2PD**：文本 teacher 在该区间有信息 |
| 末层（最后 1–2 层） | 转末层 identification head（保留 identifier 不变） |
| 全程均匀低 / 无断崖 | 问题在 search/competition 而非 path support → 转 beam 侧或终止 |

### 4.1 R2PD（预期主线）

给定历史 `h`，R² resolver 在全 catalog 产生 `q(i|h)`。对 collision-safe unique path `z(i)`，把 item 概率汇成每个 trie prefix 的下一 token 分布：

```text
Q(a | p, h) = Σ_{i: z(i) 以 p+a 开头} q(i|h)  /  Σ_{j: z(j) 以 p 开头} q(j|h)
```

损失：

```text
L = L_warm_CE + λ_cp · c(h) · L_subtree_KD + μ_keep · L_frozen_v0_retention
```

- `L_warm_CE`：原 warm next-item 监督，保持任务锚点
- `L_subtree_KD`：R² item mass 投影后的逐层 soft-target KL
- `c(h)`：teacher entropy/margin 导出的置信权重
- `L_frozen_v0_retention`：warm history 上匹配冻结 v0 的分布，防遗忘

**它针对哪条本地反证**：v1 只改 ID 未给 path 训练支持 → R2PD 直接改 path probability；v2/v3 硬对齐语义与协同 → R2PD 只蒸馏当前用户条件下的 item mass；P5 单点伪标签迁移失败 → R2PD 用 soft 分布 + confidence；R² warm 下跌 → retention 进训练目标而非事后补救。

**风险**：teacher recall 仅 11%，学生上限受限；R² 分布 calibration 可能差；full fine-tuning 可能再伤 warm；若只有推理期 reranking 有效则退化为 SpecGR。

---

## 5. 可证伪预测

| # | 假设 | 预测 | 证伪 |
|---|---|---|---|
| H1 | 瓶颈是 unsupported decoder path | cold target NLL 在**中层**出现断崖，且该深度 prefix survival 与 cold hit 强相关 | NLL 与 warm 接近但 beam 仍失败 ⇒ 问题在 search，不重训 path |
| H2 | R² item mass 可变成有效 prefix supervision | item-disjoint pseudo-cold 上 soft subtree target 优于 top-1 hard CE | soft 只降 teacher loss 不提 held pseudo-cold exact item ⇒ 停止 |
| H3 | warm forgetting 可被 retention 缓解 | 加 retention 后 warm-cold Pareto frontier 整体外移 | 在合理 `μ_keep` 下 frontier 不动 ⇒ 承认 Pareto 极限 |
| H4 | 推理期零外部依赖仍能达到 R² 同级冷收益 | native GRAM（无 portfolio、无 drafter）cold H@50 达 R² portfolio@2 同量级，且 overall > v0 | 只有挂外部模块才有效 ⇒ 退化为 SpecGR，无故事 |

**H4 是主卖点的直接检验。** 注意：v0.1 的「突破 R²」三选一条件已删除对 warm 0.97 的依赖，改为 Pareto 比较。

---

## 6. 执行计划（M1–M6）

### M1：诊断 + 竞争者（小 GPU）

**14-0A｜item-level evaluator 回归测试（约 1 天，非故事主线）**

本轮已确认 baseline identifier 重复路径 = 0（Toys/Beauty），故对 identifier 不变的 arm 该 evaluator 近乎 no-op。它是**必要的回归测试**，不是贡献。

- 复用 `lexid2cfid` 建 item↔path 双向映射
- duplicate path / ambiguous decoding / unknown item / top-K duplicate 一律 hard-fail
- v0 历史 prediction 复算 parity；raw v1 的 alias hit 须在 strict evaluator 下消失
- **kill**：历史核心数字无法对齐 ⇒ 停止，先修口径

**14-0B｜断崖诊断（主 Gate）**

Toys/Beauty validation 上只读计算：cold/warm target 的 token 级 NLL 与 rank；prefix@1..L survival 与首次跌出 beam 的深度；v0 beam / R² top50 / 二者并集的 item recall；R² 分布投影到各层后的 entropy、target mass、subtree coverage。

- **主 Gate**：断崖深度可定位（非全程均匀），且 R² 对 target path 的 projected mass 显著优于 uniform/catalog prior
- **kill**：R² teacher 对目标 path 没有高于 prior 的支持 ⇒ `FAIL_STOP_R2PD`
- 产出 → `report/第十四阶段/Stage14-0_冷路径可行性诊断报告.md`，据 §4 分支表选路线

**14-0C｜跑通真 SpecGR（与上并行）**

用 [官方代码](https://github.com/Jamesding000/SpecGR) 在 `Toys_cold50` 上跑通，与 v0 / R² portfolio@2 同口径比较。

- **这是 go/no-go**：若 SpecGR 已打平或打赢且我们无差异化优势，phase14 需重新定位
- 同时它成为主表的强 baseline（符合「只选开源 baseline」约束）

### M2：pseudo-cold screen + smoke（小–中 GPU）

**14-1｜item-disjoint pseudo-cold transfer screen**（保留 v0.1 设计，吸取 P5 教训）

在 warm train item 内建 deterministic、item-disjoint 的 pseudo-cold split：先从所有历史与 CE 样本中删除 audit item 的真实 interaction，再按真实 cold onboarding 的同一规则允许其 metadata/ID 接收 R² 的 synthetic soft mass。**audit ground truth 永不可见，synthetic supervision 可见该 catalog path。**

| Arm | 唯一改动 |
|---|---|
| A0 | frozen v0（reference） |
| A1 | top-1 hard cold-path CE |
| A2 | soft subtree distillation |
| A3 | A2 + frozen-v0 retention |

- **主 Gate**：A2 的 held pseudo-cold exact-path MRR/Recall@50 ≥ 1.10×A1
- **kill**：audit item 真实 interaction 以任何形式进入 CE / teacher fitting / 置信模型

**14-2｜matched smoke**：固定 256–512 users、固定 checkpoint/steps。验证 loss 各分量有限且下降、cold path 梯度非零、retention 梯度生效、生成 path 全部合法唯一可反解、显存与 runtime 达标。

- **主 Gate**：`λ_cp = μ_keep = 0` 时与原 v0 behavior **逐位 parity**
- smoke 不以 H@10 的一两个事件判 efficacy

### M3：Toys full（30G lease）

4 arm：v0 / R² portfolio@2 / SpecGR / 主 arm。约 4×10–16h。

报告 overall/warm/cold 的 H@10、H@50、NDCG@10、事件数、paired bootstrap CI、prefix survival，以及 **warm-cold Pareto frontier**。

- **主 Gate**：native cold H@50 显著高于 v0（paired-bootstrap 95% CI 下界 >0），且 Pareto frontier 相对 R² portfolio 外移
- **kill**：ambiguous/duplicate/unknown output 非 0
- **无 warm 硬门槛**（决策 2）

### M4：Beauty full（30G lease）

同 4 arm，约 4×26–31h ≈ 5 天。迁移 Toys 配置；只允许 domain-local catalog/embedding/index 重建。

- **主 Gate**：关键方向与 Toys 一致
- Beauty 已被 phase-13 多次查看，这是 source-domain confirmation，非独立终验

### M5：补充实验 + 主表（30G lease）

- **temporal 小节**：用 ColdGenrec 开源脚本，只跑 v0 + 最优 arm。⚠️ Toys temporal 仅 133 cold item，**只能声称方向一致，不能声称显著**
- **ablation**：hard vs soft、无 retention、无 confidence、prefix level、teacher strength
- **推理成本对比**（主卖点证据）：本方法 vs SpecGR 的推理延迟与外部依赖
- **test 开封一次**，主表 3 seeds

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
- Toys/Beauty test：封存，M5 开封一次
- **Sports：本阶段不使用**（决策 5）。limitation 必须明写"Toys/Beauty validation 已用于方法开发，无独立未污染终验域"

---

## 8. 统计与比较原则（原样保留）

1. 所有 gain 同时报 absolute、relative、event count；稀疏 cold hit 不只报百分比
2. user-level paired bootstrap，≥10,000 resamples，报 95% CI
3. warm/cold/overall 三组都报，不以 overall 隐藏 trade-off
4. **R² 比较必须 warm-cost matched 或直接做 Pareto frontier**；禁止重犯"高 cold 只是花更多 warm slot"
5. 全部方法同一 catalog、unique ID map、beam size、evaluator
6. 探索期允许调超参；**最终主表配置冻结后重跑一次**（决策 3）
7. full run 不因"差一点"改 Gate

---

## 9. 资源

```text
experiment/phase14/{protocol,configs,tests}/   artifacts/phase14/{diagnostics,explore,formal}/
report/第十四阶段/
```

| Stage | 资源 | 时长 | 30G lease |
|---|---|---:|---|
| 14-0A/B | CPU / 小 GPU | 分钟–数小时 | 否 |
| 14-0C SpecGR | 小 GPU | 数小时–1 天 | 否 |
| 14-1 / 14-2 | 单小 GPU | 数小时 | 否 |
| M3 Toys×4 | GRAM full | 4×10–16h | **是** |
| M4 Beauty×4 | GRAM full | 4×26–31h | **是** |
| M5 | temporal + ablation + 主表 | 数天 | **是** |

>10 分钟实验沿用：独立 tmux、hard timeout、status.json、telemetry、**不自动重试**。大显存实验启动前报告预计占用并由用户指定 GPU；**不得停止他人进程或调整 holder，除非用户明确授权**。

**已知坑**（memory `feedback_runner_tmux_bug.md`）：`run_phase13_explore.sh start` 用 tmux 拉 launch_cmd 可能 10s 内 exit=1，改用 `setsid nohup` 绕开；`finish()` 不检查 workload_rc（CUDA OOM 会误判成功）；status.json 的 workload_pid 恒为 0；**判活用 workload PID，不要用 tmux session**。

每个正式 artifact 至少含：`config.json` `manifest.json` `status.json` `summary.json` `run.log` `gpu_telemetry.csv` `predictions_*.jsonl` `item_path_audit.json` `data_provenance.json`。

---

## 10. 停止规则（从 10 条压到 4 条）

1. **泄漏**：actual cold interaction 或 held target 进入训练/teacher/置信模型
2. **收益来自 alias**：方法增益只来自 ambiguous lexical path
3. **无法归因**：需同时改 ID、decoder、resolver、split 才能得正结果
4. **无差异化**：结果实质等价于 SpecGR/AGRec，且无独立机制或推理成本优势

（各 Stage 另有自己的 1 条 kill，见 §6）

---

## 11. 论文故事（仅结果成立后采用）

1. **Cold-path collapse，且断在中层**：把 [2607.21101] 的可达性分析迁移到 hierarchical k-means identifier，发现与 RQ-VAE **相反**的断裂模式 —— identifier 构造方式决定冷路径在哪里失败（§0.2）
2. **R² recoverability probe**：warm-only inductive resolver 证明用户历史中存在可恢复的 cold preference，但外部 portfolio 有 warm trade-off 与候选天花板
3. **R²-to-Path Distillation**：把 item teacher 分布投影为 prefix-conditional subtree targets，不改 identifier 即给冷路径支持
4. **★ 推理期零外部依赖**：与 SpecGR 对比，冷物品接入是一次性训练期操作，推理是纯原生 beam（**主卖点**）
5. **Warm retention**：frozen-v0 path retention 把 catalog onboarding 与 old-item forgetting 统一
6. **Rigorous protocol**：unique item evaluation、cost-matched Pareto、frequency + temporal 双口径

### 可以 / 不可以声称

**可以**（若双域成立）：R² 是有效 recoverability teacher；prefix-level soft transfer 提高 native collision-safe cold reachability；retention 改善 Pareto；**推理期无需外部检索器**。

**不可以**：完全 zero-shot 新物品；不需要 catalog onboarding；universally beats SpecGR；当前 validation 等于 SOTA；多兴趣 resolver 是主创新；**有独立未污染域的终验**。

---

## 12. 下一步唯一动作

**M1 三件事并行，不启动 GRAM full training：**

```text
experiment/phase14/protocol/item_level_eval.py        # 14-0A，复用 lexid2cfid
experiment/phase14/protocol/oracle_prefix_probe.py    # 14-0B，主 Gate
experiment/phase14/tests/test_item_level_eval.py
experiment/phase14/tests/test_oracle_prefix_probe.py
experiment/phase14/configs/stage14_0_toys_beauty.json
# 14-0C：clone SpecGR 官方代码，接 Toys_cold50
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

---

## 14. 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-19 | 新开第十四阶段，保留 R² 为有效点 | 双域正向证据未被否定 |
| 2026-08-19 | 停止 allocator/resolver 小调参主线 | Tier0/Tier1 已定位候选召回/path support 才是瓶颈 |
| 2026-08-19 | 主候选定为 R2PD | 与 GRAM 训练缺口、本地反证、文献最一致 |
| 2026-08-19 | Candidate verifier 降为 baseline | SpecGR 已覆盖 drafter-verifier |
| 2026-08-19 | pseudo user-sequence 降为备选 | USIM 已覆盖，且本地 pseudo-cold 曾失败 |
| **2026-08-20** | **诊断贡献降级为「复现+迁移」** | 2607.21101 已做完，数据集重叠、代码开源 |
| **2026-08-20** | **删除 warm≥0.97 硬门槛，改 Pareto frontier** | 该阈值未经论证且已误杀 P1–P7，不可重蹈 |
| **2026-08-20** | **Gate 精简；探索期允许调参** | 当前是探索模式，v0.1 用错治理强度 |
| **2026-08-20** | **真 SpecGR 提前到 M1** | 直接竞争者，代码开源，打平则无故事 |
| **2026-08-20** | **放弃 Sports；temporal 降为 M5 小节** | Sports_cold50 不存在 + 8 卡全满；temporal Toys 仅 133 cold item |
| **2026-08-20** | **实测 cold 断裂深度，否定末层假设** | Toys 67.8% / Beauty 82.5% 断于深度≤2；仅 8.6%/2.66% 是"只差末位"。**与 RQ-VAE 相反，末层 head 方案作废，R2PD 可行性提高** |
| **2026-08-20** | **主卖点定为「推理期零外部依赖」** | 诊断与 drafter-verifier 均被占，这是唯一干净的差异化轴 |
