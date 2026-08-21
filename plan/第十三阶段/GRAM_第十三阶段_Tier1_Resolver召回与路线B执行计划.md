# GRAM 第十三阶段：Tier-1 Resolver 召回提升 + 路线 B 生成路径修复

**创建日期**：2026-08-19
**状态**：待执行（交接给其他 AI 运行）
**前置证据**：`report/第十三阶段/GRAM_第十三阶段_Tier0_诊断三连_核心论断更正与瓶颈重定位.md`
**定位**：本计划取代 Section 3.5.9 的原 Tier-1 草案，是当前唯一待执行的实验线

---

## 0. 交接必读：这份计划为什么长这样

如果你是接手运行的 AI，**先读这一节**，否则你会重复过去两周已经失败过的 8 轮工作。

### 0.1 已经被排除的方向（不要再做）

Phase 13 已经完成 **8 轮学习型 slate 分配机制**，全部未通过 Gate：

| 机制 | 结果 |
|---|---|
| P1 linear admission rerank | FAIL（warm −11.07%） |
| P2 anchored interleaving | FAIL（warm 差 0.0001，噪声级） |
| P3 confidence abstention | validation PASS → 独立 test FAIL |
| P4 counterfactual slot router | validation PASS → 独立 tranche FAIL |
| P5 setwise selector | FAIL（pseudo-cold 迁移失败） |
| P6 risk-limited portfolio | FAIL（差 2× 门槛 1.95%） |
| P7 robust slate（ensemble std） | FAIL（退化为全 abstain，coverage=0） |
| R²-v2 CBSA budget-conditioned | FAIL（Beauty cold retention 91.86%<95%） |

**2026-08-19 三层饱和分析证明这不是"机制不够聪明"，而是操作空间本身已耗尽**：

1. **候选池饱和**（Tier-0 B）：cold target 落在 resolver top-50 内仅 `11.40%`(Toys)/`11.03%`(Beauty)，约 **89% 的 cold 用户答案根本不在池内**。插入 N 个候选的理论天花板：@2=2.11%、@3=3.11%、@10=7.17%；
2. **用户选择饱和**（Tier-0 A2）：oracle 用户选择 `0.029998` ≈ 无条件全覆盖 `0.029769`。把"选哪些用户干预"做到完美，收益也就这么多；
3. **候选排序饱和**（Tier-1 B）：RRF/Borda 融合、学习型 setwise selector 全部打不过朴素的 resolver 顺序。`w=0`（纯 resolver）在网格中最优，加入 GRAM 分数后 cold 单调下降。

**结论：在冻结 GRAM + 冻结 resolver 的前提下，`portfolio@2/@3` 就是最优解。后处理层已到顶。**

因此**禁止**：新增第 9 个 allocator、恢复 P1–P7/CBSA、恢复旧 v2–v5、在现有 validation 上继续调 threshold/prefix/quota/融合权重。

### 0.2 一个重要的认知修正

原 plan 主张"简单 portfolio 打败复杂 gating"，**该论断已于 2026-08-19 撤回**。原比较未匹配 warm 代价（portfolio@2 warm 95.91% vs P6 99.56%）。匹配后学习型分配**显著优于随机**（双域双机制，A3 置换检验 p=0.005，0/200 置换追平）。

**教训（写进你的工作习惯）**：比较两个方法前必须确认它们在同一工作点。七轮实验的方向判断曾建立在一个未对齐的比较上。

### 0.3 为什么现在做 Tier-1

上述所有饱和结论都是**以当前这个欠训练的 resolver 为前提**。resolver 现状：

| 项 | 当前值 | 问题 |
|---|---|---|
| epochs | **12** | 几乎确定欠训练 |
| 负样本 | in-batch 随机（batch=256） | 对 12k item 全库检索过弱 |
| hard negative mining | 无 | — |
| temperature | 0.07，从未调过 | — |
| hidden_dim | 512，从未调过 | — |
| 用户表示 | recency-weighted 均值池化（decay=0.85） | 长历史多兴趣被抹平 |
| item 侧 | BGE 完全冻结 | — |

**resolver 召回是唯一未被认真优化、且能直接抬高所有下游天花板的维度。** 召回从 11% 提到 18%，portfolio 收益按比例放大，不需要任何新机制。

---

## 1. 实验协议（硬规则，继承 phase12/13，不可绕过）

### 1.1 GPU 保护

- **实验前**：`nvidia-smi` 确认目标卡实际空闲；GPU0 约 30 GiB、GPU5 约 20 GiB 是**用户占位资源，不得调整/释放/kill**；
- 2026-08-19 实测 8 卡接近满载。**资源不足时先向用户申请，不得挤占他人进程**；
- 占位者让位/恢复：`tools/run_codellama.sh`(GPU6) 或 `tools/gram_ablation_scan.sh`(其他卡)，exit trap 保证恢复；
- **本计划 Tier-1 实验为 resolver 训练（非 GRAM 训练/beam search），按 Section 6.2 不需要 30G lease**；路线 B 涉及 GRAM 重训，**必须走完整 protocol + 30G lease**。

### 1.2 运行方式

- 预计 >10 分钟的实验：**必须**独立 tmux 后台 runner + 持续写 `status.json` + hard timeout + **不自动重试**；
- 预计 ≤10 分钟：可前台，但仍写 status.json；
- **已知 runner bug（务必规避）**：`run_phase13_explore.sh start` 用 tmux 拉 launch_cmd 时可能 10s 内 exit=1；改用 `setsid nohup` 直接 exec worker 绕开。见 memory `feedback_runner_tmux_bug.md`；
- 其他已知坑：finish() 不检查 workload_rc（CUDA OOM 会误判成功）；status.json 的 workload_pid 恒为 0；**判活要用 workload PID，不要用 tmux session**。

### 1.3 数据防火墙（最重要）

| 数据 | 状态 | 允许用途 |
|---|---|---|
| Toys/Beauty **train + validation** | 已污染（P1–P7 调参用过） | 仅方法开发与方向选择 |
| Toys/Beauty **test** | **封存** | 禁止读取 |
| **Sports 全部** | **封存** | 禁止读取，禁止建 cold split |

- Tier-1 的 validation 结果**只用于选择方向**，不得作为论文主结果；
- 任何候选若要声称 efficacy，**必须另行预注册并在未查看的数据上确认**；
- 每个实验的 summary.json 必须写 `test_read: false`。

### 1.4 Report 强制规则

每次实验完成（不论成功/失败）必须：
1. 写 report 到 `report/第十三阶段/`；
2. 更新本计划的进度表（Section 5）；
3. 更新 memory `project_current_run.md`；
4. 确认 GPU 占位已恢复。

**未完成上述任何一步，不允许进入下一项实验。**

---

## 2. Tier-1：Resolver 召回提升（当前主线）

### 2.1 单一研究问题

> 在不重训 GRAM、不改变下游 portfolio 规则的前提下，能否显著提高 resolver 的 cold 召回？

**主指标：cold recall@50**（validation）。当前基线：**Toys `11.40%`（498/4,367）、Beauty `11.03%`（583/5,287）**。

选该指标的理由：(a) 直接测量已被证明的瓶颈；(b) 事件密度 498/583，远高于 cold NDCG@10 的 23/16，**具备统计分辨率**（cold NDCG@10 在本 setting 下已证明不可用作主指标）；(c) 召回提升会按比例放大下游收益，无需新机制。

**辅助指标**（同时报告，不设门槛）：cold recall@10/@20、warm recall@50、下游 portfolio@2 的 cold H@50 / warm N@10 / overall N@10。

### 2.2 现有实现与复用

**核心文件：`experiment/phase13/protocol/route_resolve.py`**

关键位置：
- `ResidualUserProjector`（L242-255）：`x + residual_scale * MLP(x)`，然后 L2 normalize；
- `multi_positive_inbatch_loss`（L257-267）：in-batch contrastive，同 batch 内相同 target 视为多正例；
- `train_projector`（L269-311）：训练循环；
- `recency_weighted_history`（L173-183）：历史池化，`decay^age` 加权后 L2 normalize；
- 参数默认值（L35-62）：`epochs=12, batch_size=256, hidden_dim=512, dropout=0.1, lr=1e-3, weight_decay=1e-4, temperature=0.07, recency_decay=0.85, max_history=20, seed=12345`。

**冻结输入（不要重新生成）**：
- Toys embedding：`artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt`
- Beauty embedding：`artifacts/phase13/embeddings/Beauty_bge_large_en_v1_5_cls_l2.pt`
- Toys v0 预测：`artifacts/phase13/explore/v0_toys/predictions/20260809_085251_Toys_cold50_sequential_pred_validation.tsv`
- Beauty v0 预测：`artifacts/phase13/explore/v0_beauty/predictions/20260811_103607_Beauty_cold50_sequential_pred_validation.tsv`（6 个中的**最后一个**）
- 参考 config：`artifacts/phase13/explore/v1_r2_{toys,beauty}_p0/config.json`

**注意**：`route_resolve.py` 内含 depth-3 route 融合逻辑，该接口已被 P0 否定。Tier-1 **只关心 resolver 本身的召回**，评测时使用纯 `resolver_top50`，不要启用 route prior（或设 `--route-prior-weight 0`）。

### 2.3 实验序列（按顺序执行，每步独立 Gate）

#### T1-1：训练量扫描（最高优先级，最可能白捡）

- **改动**：`--epochs` ∈ {12(基线复现), 30, 60, 100, 150}，其余全部不变；
- **必须先复现基线**：epochs=12 + seed=12345 必须重现 cold recall@50 = 11.40%(Toys)。**不能复现就停下排查，不要继续**；
- **产出**：每个 epochs 的 cold recall@50 曲线 + 训练 loss 曲线；
- **判读**：若曲线在某点后平台或下降 → 过拟合，取最优点进 T1-2；若单调上升到 150 → 继续扩到 300；
- **成本**：单卡，12 epochs 约 118s（Toys P0 实测），150 epochs 约 25 分钟；全扫描约 1 小时；
- **域**：先 Toys（快），确认方向后再 Beauty。

#### T1-2：Hard negative mining（最可能有实质收益）

- **动机**：当前 in-batch 随机负样本（batch=256）对 12k item 全库检索监督过弱。模型没被教会区分"语义相近但不是答案"的 item；
- **实现方案**（在 `multi_positive_inbatch_loss` 基础上扩展，新写函数不要改原函数）：
  - 每个 epoch 开始时用当前模型对训练用户检索 top-K，取**非目标**的高分 item 作为 hard negative（K 建议 50-200）；
  - 或更省事的静态版：用**冻结 BGE embedding** 预计算每个 target item 的最近邻作为 hard negative（不随训练更新，成本低很多，先试这个）；
  - loss 改为 `positives vs (in-batch negatives + hard negatives)`；
- **超参**：hard negative 数量 ∈ {8, 16, 32}/样本；
- **Gate**：cold recall@50 相对 T1-1 最优点提升；
- **成本**：静态版几乎不增加训练时间；动态版每 epoch 多一次全库检索。

#### T1-3：温度与容量

- `temperature` ∈ {0.03, 0.05, 0.07, 0.1, 0.15}；`hidden_dim` ∈ {512, 1024, 2048}；
- **注意**：这是纯调参，收益预期最低。**只在 T1-1/T1-2 已有明确收益后做**，且不要做完整网格（先各自单独扫，不交叉）。

#### T1-4：用户表示升级

- 当前是 recency-weighted 均值池化，长历史的多兴趣被抹平；
- 候选：(a) attention pooling（可学 query）；(b) 多向量表示（取 top-M 兴趣向量，检索时取 max 相似度）；
- **(b) 更可能有效**，因为 cold 场景下用户的小众兴趣正是被均值抹掉的部分；
- 这是本序列中唯一有**结构创新**成分的一项，若成功可作为方法贡献的一部分。

#### T1-5：item 侧轻量微调（可选）

- BGE 当前完全冻结。可试 LoRA 或最后几层解冻；
- **风险**：item embedding 一动，所有下游冻结产物（P0 预测、portfolio 候选）全部失效，需重算；
- **建议放最后**，且只在前面几项收益不足时做。

### 2.4 Tier-1 Gate

**PASS 条件**（进入下游验证）：
- Toys **和** Beauty 的 cold recall@50 均相对各自基线提升 **≥ 30% 相对**（即 Toys ≥14.8%、Beauty ≥14.3%）；
- 该阈值理由：低于此幅度，下游 portfolio 收益的绝对增量会淹没在 cold H@50 的事件噪声里（当前 130/172 个事件）；
- warm recall@50 退化 ≤ 5% 相对。

**PASS 后的动作**：
1. 用新 resolver 重跑 `portfolio@2/@3`，报告新的 Pareto 前沿 + paired bootstrap CI（复用 `experiment/phase13/protocol/b1_portfolio_confirmation.py`）；
2. **此时仍不读 test**。若要声称 efficacy，另行预注册；
3. 写 report，更新进度表。

**FAIL（提升 <30%）**：resolver 召回也接近饱和 → **Tier-1 结束，转路线 B 或路线 C**，不做 T1-6。

### 2.5 实现要求

- 新代码放 `experiment/phase13/protocol/tier1_resolver_*.py`，**不要原地改 `route_resolve.py`**（它是 P0 的冻结产物依赖）；
- 复用现有单测框架 `experiment/phase13/tests/`，至少覆盖：warm-only train target（cold target count 必须为 0）、embedding 覆盖完整、no test read；
- 每次运行写 `artifacts/phase13/explore/tier1_resolver_<变体>/{status.json,summary.json,config.json,run.log}`；
- summary.json 必须含：`cold_recall_at_50`、各 k 值 recall、`test_read: false`、输入 hash、seed、完整超参。

---

## 3. 路线 B：修复生成路径（Tier-1 FAIL 或 PASS 后的主方法线）

### 3.1 动机

Tier-0 B 实测：cold target 在 **GRAM top-50 内仅 1.03%(Toys)/1.31%(Beauty)** —— **GRAM 的生成路径对零交互 item 基本是死的**。

现在所有方案（portfolio、融合、RTP）都是"承认它死了，然后在尾部塞检索结果"。这是**绕过**，不是**修复**。审稿人会问：那你为什么还用生成式推荐？

**路线 B 的问题**：能否让 GRAM 的 decoder 本身学会生成 cold item 的 token 路径？

### 3.2 为什么这次可能和之前不同

Phase 13 已有 **6 次 ID 侧改动全部失败**（MiniLM/E5/BGE encoder、residual MLP、regularized residual、capacity-aware assignment）。**必须先理解它们为什么失败，否则会是第 7 次。**

失败的共同点：**它们都只改 ID 分配，不改训练过程**。cold item 被分配了一个 ID，但 GRAM 训练时**从来没见过这个 ID 作为 target**（cold item 按定义没有交互）。所以 decoder 对这些 token 路径的先验概率极低，beam search 自然不会生成它们。

**路线 B 的关键区别：改训练目标，让 decoder 见过 cold item 的路径。**

### 3.3 候选方案（择一深入，不要并行铺开）

#### B-1：文本导出的伪交互监督（推荐首选）

- 对每个 cold item，用其文本 embedding 找 **top-M 最相似的 warm item**；
- 把这些 warm item 的用户交互序列**复制一份**，target 替换为该 cold item，作为伪训练样本；
- 伪样本加权（权重 < 1，或按相似度加权）混入 GRAM 训练；
- **假设**：decoder 学到"这类历史 → 这类 token 路径"的映射，从而对 cold item 的路径产生非零先验；
- **风险**：伪样本可能污染 warm 性能。必须做权重扫描并报告 warm 退化。

#### B-2：ID 空间的语义正则

- 在 GRAM 训练 loss 上加一项：约束 item embedding 空间中相似的 item 有相似的 token 路径前缀；
- 相比 B-1 更温和，但可能收益也更小。

#### B-3：Cold-aware beam search（推理时，最便宜）

- 训练不变，只改 beam search：对 cold item 的路径给一个基于文本相似度的 prior bonus；
- **成本最低**（不重训），但本质上仍是后处理，可能落入已证明饱和的区间；
- **建议作为 B-1 的对照基线**，而非主方案。

### 3.4 成本与资源（务必先申请）

| 项 | 成本 |
|---|---|
| GRAM v0 训练实测 | Toys **10.3h**、Beauty **26.1h**（单卡 A6000） |
| collision-safe v1 实测 | Toys 15.2h、Beauty 31.0h |
| 峰值显存 | 约 16 GB allocated / 19-24 GB reserved |
| 磁盘 | 约 4.2 GB/次（6 model + 6 optimizer checkpoint） |
| **必须** | 完整 protocol + **30G lease** + 占位者让位/恢复 |

**每个 B 候选一次双域验证 ≈ 40-50 小时 GPU。** 8 卡目前满载，**必须先向用户申请资源并确认排期**。

**强烈建议**：先做 **debug 规模 smoke**（`--debug_train 100 --debug_test 100`，1 epoch）验证 pipeline 通路，再上全量。历史上有多次全量跑到一半发现实现错误的记录。

### 3.5 路线 B Gate

- 主指标：**cold H@50 in GRAM top-50**（即生成路径自身的可达性，当前 1.03%/1.31%）；
- PASS：该指标相对 v0 提升 ≥ 2×，且 warm NDCG@10 退化 ≤ 5%；
- 这是"生成路径被修复"的直接证据，与后处理层完全解耦；
- 若 PASS，方法故事成立：**我们让生成式推荐器本身能够触达零交互 item**，而非在尾部打补丁。

---

## 4. 论文形态（三种，按证据强度）

### 4.1 若路线 B PASS（最强，方法论文）

> **生成式推荐对零交互 item 结构性不可达（GRAM top-50 命中率仅 1.0-1.3%）。我们证明后处理层的补救存在三重饱和上界（候选池 7%、用户选择、候选排序），随后通过 [B-1 伪监督] 修复生成路径本身，使其可达性提升 N×。**

全部负结果成为论证的必要环节。目标：CCF-B full paper。

### 4.2 若仅 Tier-1 PASS（中等）

> **瓶颈定位 + resolver 召回提升 + 诚实 Pareto 前沿。**

方法贡献偏弱（"把检索器训得更好"容易被视为调参），除非 T1-4 的多向量用户表示有实质创新。目标：CCF-B short / workshop。

### 4.3 若均 FAIL（分析论文，保底）

现有证据链已足够支撑：

1. **Semantic-ID 碰撞污染测量**：原 v1 报 +186%/+133%，碰撞审计后 collision-safe 重跑为 −49%/−9.87%。GRAM 评测是纯字符串比较（`GRAM/src/utils/evaluate.py:16`），检测不到 ID 别名；
2. **事件稀疏使标准指标失去分辨率**：cold NDCG@10 在 4,367/5,234 个 cold user 上仅 23/16 个 hit 事件。历史上 P2（差 0.0001）、P6（差 1 个 hit）、Beauty v1（9→8 个 DCG 事件）的 FAIL 均为噪声级误判；
3. **三层饱和分析 + 8 个失败机制的系统性负结果**，配 paired bootstrap CI。

⚠️ **措辞红线**：我已实测四个数据集的**原始** GRAM ID 文件 `duplicate_excess` **全为 0**（Toys/Beauty/Sports/Yelp）。**碰撞是 v1 方法引入的，不是 GRAM/TIGER baseline 的缺陷。**

- ✅ 可写："任何预测 cold item semantic ID 的方法都必须做全局唯一性审计，否则指标会虚高约 5 倍"
- ❌ 不可写："生成式推荐的评测存在缺陷" —— 会被一句话打死

---

## 5. 进度表（每次实验后必须更新）

| 实验 | 域 | 状态 | 主指标 | Gate | Report | 日期 |
|---|---|---|---|---|---|---|
| Tier-0 A 匹配代价随机基线 | Toys | ✅ done | P6 0.018090 vs 随机 0.012411 | 主张 3 撤回 | Tier0 诊断三连 | 2026-08-19 |
| Tier-0 A2 效用排序扫描 | Toys | ✅ done | 9/20(@2)、12/20(@3) 点超 2sd | 学习型有效 | 同上 | 2026-08-19 |
| Tier-0 A3 CBSA 置换检验 | Toys+Beauty | ✅ done | 0/200 置换，p=0.005 | 双域确认 | 同上 | 2026-08-19 |
| Tier-0 B 候选池天花板 | Toys+Beauty | ✅ done | cold 可达 11.40%/11.03% | 瓶颈=召回 | 同上 | 2026-08-19 |
| Tier-1 A 融合扫描 | Toys+Beauty | ✅ done | cold H@50 2.2× 但 cold H@10 不变、overall 低于 v0 | 部分成功不可用 | Tier1 A-B 负结果报告 | 2026-08-19 |
| Tier-1 B RTP 混合 | Toys | ✅ done | w=0（=portfolio@2）最优；w=0.9 的 cold CI 全负 | **FAIL，排序维度饱和** | 同上 | 2026-08-19 |
| **T1-1 epochs 扫描** | Toys | ⬜ 待运行 | cold recall@50（基线 11.40%） | — | — | — |
| T1-2 hard negative | — | ⬜ 待运行 | — | — | — | — |
| T1-3 温度/容量 | — | ⬜ 条件执行 | — | — | — | — |
| T1-4 用户表示 | — | ⬜ 条件执行 | — | — | — | — |
| 路线 B | — | ⬜ 需先申请 GPU | GRAM top-50 cold 可达性 | — | — | — |

---

## 6. 已完成实验的产物索引

**Tier-0 / Tier-1 脚本**（均为 evaluation-only，纯 CPU）：
- `experiment/phase13/protocol/tier0_matched_cost_baseline.py`
- `experiment/phase13/protocol/tier0_prioritisation_sweep.py`
- `experiment/phase13/protocol/tier0_cbsa_permutation.py`
- `experiment/phase13/protocol/tier0_pool_ceiling.py`
- `experiment/phase13/protocol/tier1_fusion_sweep.py`
- `experiment/phase13/protocol/tier1_recall_then_place.py`

**产物**：`artifacts/phase13/explore/tier0_*/summary.json`、`tier1_*/summary.json`（全部 `test_read=false`）

**关键参考实现**：
- resolver 训练：`experiment/phase13/protocol/route_resolve.py`
- portfolio + bootstrap 评测：`experiment/phase13/protocol/b1_portfolio_confirmation.py`
- Pareto 重算：`experiment/phase13/protocol/pareto_recompute.py`
- cold/warm 评测：`experiment/phase13/protocol/eval_cold_warm.py`
- 完整 GRAM runner 模板：`experiment/phase12/run_phase12_hi_gram.sh`

---

## 7. 尚未完成的调研（不阻塞 Tier-1，投稿前必补）

两个调研 agent 于 2026-08-19 因 **API 余额不足（403）** 中断，未产出结果：

1. **文献核对**：
   - arxiv 2607.21101 *Can Generative Recommendation Reach Cold Items?*（TIGER-Scorer）与 arxiv 2603.29845 *Cold-Starts in Generative Recommendation: A Reproducibility Study* 的确切主张与留白；
   - **semantic-ID 碰撞作为评测有效性问题**是否已有人写过；
   - **推荐系统评测的统计功效/事件稀疏**是否已有人系统讨论；
   - **"简单启发式槽位预留 vs 学习型选择"**在稀疏正例下的负结果是否已有前人工作；
   - ⚠️ 特别需要确认：**"匹配代价的学习型 vs 随机对照"这个实验设计**是否已有前人做过（这是 Tier-0 的核心方法论贡献）。

2. **会议截稿期**：ECIR 2027（含 Reproducibility track）、RecSys 2027（Reproducibility track）、CIKM 2027、DASFAA 2027、PAKDD 2027、SIGIR 2027、WSDM 2027，以及 TOIS/TKDE/IPM/TORS 期刊路线。

**已知约束（影响 venue 选择）**：仓库**只有 GRAM 一个 backbone**，无 TIGER/LC-Rec/RQ-VAE tokenizer。若审稿人要求跨 backbone 验证，那是从零实现，不是改配置。

---

## 8. 关联文档

- `plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md` — Phase 13 完整历史记录（Section 3.5 评价口径、Section 9 进度表）
- `report/第十三阶段/GRAM_第十三阶段_Tier0_诊断三连_核心论断更正与瓶颈重定位.md` — **本计划的直接依据**
- `report/第十三阶段/GRAM_第十三阶段_v1_collision-safe_双域重跑验证报告.md` — 碰撞审计与重跑
- `report/第十三阶段/GRAM_第十三阶段_v1_success-mechanism_碰撞审计报告.md` — 碰撞机制分析
- `report/第十三阶段/GRAM_第十三阶段_v1-R2_Beauty-B1_无条件portfolio跨域确认报告.md` — portfolio 跨域 PASS
- Memory：`project_current_run.md`（当前状态）、`feedback_experiment_protocol.md`（协议）、`feedback_runner_tmux_bug.md`（runner 坑）、`user_constraints.md`（资源约束）
