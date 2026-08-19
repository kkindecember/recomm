# Phase 13 Exploratory: CANARD MVP 逐步验证 & 迭代

**创建日期**:2026-08-07
**状态**:R²-v2 方法探索已预注册、尚未实现/训练（2026-08-19 更新：原 collision-safe v1 **双域 FAIL**；v1-R² P0–P7 的 learned gating 均未通过独立确认；冻结的 `resolver + unconditional portfolio@2` 在 Beauty B1 跨域确认中正式 **PASS**。用户已选择方法论文路线，下一步是实现并源域筛查单一可训练的 R²-v2；不直接启动 publication 全矩阵，不把它记作 P8）
**目的**:保留 CANARD/v1-R² 完整证据链，并以最低成本验证新的单一可训练 R²-v2 是否能严格优于冻结的 `portfolio@2`
**预计工期**:6-8 周(每一步顺利 4-5 周,考虑常态 iteration 6-8 周)

---

## ⚠️ 定位说明

**本文档不是"试多个方向"的撒网 plan**,而是**验证 CANARD 一个方向 + 如何 iteratively 改进** 的 MVP-style plan。

CANARD 的核心假设:
> "Text-based signal(sentence-BERT + LLM prior + hierarchical alignment)能显著挽救 GRAM 的 cold-start item hierarchical id 分配问题。"

如果这个核心假设在 **collision-safe v1 的 3 次有效尝试后仍完全崩掉**，才考虑启动本文档末尾的 **Plan Z fallback**（换方向）。当前优先完成 v1 iteration，不以旧 collision-unsafe v2/v3 代替这一证据链。

Exploratory 阶段完成后,进入 `PLAN_PUBLICATION.md` 做全矩阵实验和写论文。

### 2026-08-17 证据口径重置（必须先读）

后续 lexical-ID 碰撞审计确认，原 v1 中不同 item 可共享同一完整 lexical ID，而旧评测只比较解码字符串，无法保证命中的是同一底层 item。因此：

- 原 v1 Toys / Beauty 的 `+186% / +133%` 只能保留为 **collision-unsafe raw 历史记录**，原 PASS 结论作废；
- 依赖旧 v1 与碰撞不安全 ID 的旧 v2/v3，**正式 efficacy Gate 结论一并失效**；其实现、OOV/层级错位诊断仍可复用，但不能作为新主线的增益证据；
- 新主线从 `v1_collision_safe_*` 重新立基。Toys 与 Beauty 均已完成并正式 FAIL，**双域汇总结论已于 2026-08-18 确证为 FAIL**（原文"Beauty 仍在运行 / PENDING"已过期，见下方 Beauty 结果节）；
- v1 的 iteration 额度从第一个有效的 collision-safe 运行开始计算。当前有效尝试为 iter1，尚余 iter2/iter3，不应直接跳到 v2 或 Plan Z。
- E5 candidate 虽达到宽松的 MLP 收敛门槛，但更直接的 cold-ID 对照显示深层 prefix 与完整路径均弱于 MiniLM，已在 pre-GRAM 阶段止损；没有运行 E5 smoke/正式 GRAM，也不计为一次正式 efficacy Gate。

证据报告：`report/第十三阶段/GRAM_第十三阶段_v1_collision-safe_双域重跑验证报告.md`。

---

## 1. 探索原则

### 1.1 快速迭代

- **主数据集 + 快筛域**:Beauty 负责主结论完整性，Toys 用于快速诊断与 iteration Gate；当前 collision-safe 重跑保留双域结果
- **单 cold ratio**:η = 50%(足够暴露 cold 问题,不需要 20% 弱刺激)
- **单 seed**:GRAM training seed=2023，cold-split seed=12345(不追求 seed 方差稳定性,那是 publication 阶段的事)
- **只对比 vanilla GRAM 一个 baseline**:不做 trivial/content/SOTA(那些是 publication 阶段的事)
- **快速 gate**:每步 2-3 天到 1 周出定性结论

### 1.2 GPU 资源约束(硬规则,不可绕过)

**服务器资源紧张,一旦放开 30G lease 大概率排不回来**。因此探索阶段依然严格遵守 phase12 完整 protocol,但**占位者不限于 CodeLlama**——见 README 里"GPU 占位者"表格,GPU6 用 CodeLlama,其他卡用 `tools/gram_ablation_scan.sh` 伪装占位。

- **主实验卡(默认 GPU6,可切换)必须始终有 30G 保留**:
  - **GPU6**:CodeLlama 常驻(平时占位,~30 GiB)
  - **其他卡**(如 GPU0 副线):ablation-scan holder 占位(伪装 hyperparameter scan,~30 GiB)
  - 实验启动前:对应工具 `stop` 释放 GPU
  - 实验期间:`gpu_memory_lease.py` sidecar hold (30G - workload peak) MiB
  - 实验结束(不论成功/失败/中断):**立即**对应工具 `start` 恢复占位
  - runner 脚本必须在 exit trap 里保证恢复,不允许"意外退出让占位掉了"

- **两卡并行探索**(HI-GRAM 后新增的能力):
  - GPU6 走 CodeLlama protocol(主线实验,例如 v0 baseline)
  - GPU0 走 ablation-scan protocol(副线实验,例如 v1 pilot 或 v0 二次 iteration)
  - 两个 runner 各自独立 tmux session,status.json 分别在各自 artifacts 目录下
  - Runner 通过环境变量 `PROTECTOR_TOOL=codellama|ablation_scan` 决定用哪种占位者

- **LLM API 调用**(v2, v4 相关部分):
  - 走 API(DeepSeek V4)不占本地 GPU,不需要 lease
  - 但依然要在**保护 GPU 上运行**(即 GPU6 上 CodeLlama 常驻状态下调用 API,占用行为不冲突)

- **仅 CPU 的部分**(数据 split, cache, metrics summary):
  - 不需要 lease,但也不允许在没保护的卡上跑训练

**违反后果**:如果哪次 CodeLlama 掉了,可能 24-48 小时内排不回 GPU,直接卡死 exploratory 进度。

### 1.3 累加式验证

从最简单的组件开始,一次只加一个,每步独立评估边际贡献。

**优势**:
- 每步失败原因隔离,好 debug
- 通过的 v0-v5 序列自然形成 ablation study
- 早期止损:v1-v2 就挂 → 整个 semantic-based 方向不行,及早换 fallback
- 每步的 negative result 都是论文素材(消融或 discussion)

**2026-08-19 适用性更正**：以上是原 v0–v5 计划的历史原则。碰撞审计与 v1-R² 证据链已经打断原累加关系；当前只执行本文新增的 **R²-v2** 单方法线。旧 LLM-v2、hierarchical-v3、reflection-v4、dual-path-v5 不再作为可顺推的后继版本。

### 1.4 iteration 上限

每一步 vN 如果 gate 未通过,最多 3 次调整:
- 第 1 次:参数/超参调整(learning rate, loss weights, prompt 措辞)
- 第 2 次:实现细节调整(换 encoder,换 LLM,换 layer arch)
- 第 3 次:降级尝试(简化设计,去掉部分复杂度)

3 次都失败 → **该组件砍掉**,报告 negative,继续下一步(下一步跳过依赖该组件的部分)。

**2026-08-19 适用性更正**：此“三次调整”仅解释旧 v0–v5 历史流程，**不适用于 R²-v2**。R²-v2 的止损以其“单一性、止损与允许的修复”小节为准：一次 source Gate，看到结果后不做科学参数 rescue。

### 1.5 Report 强制规则(硬性)

**每一次尝试完成后**(不论成功/失败/边缘),都必须写一份 report 到 `report/第十三阶段/`,命名规范参考 phase9/11:

```
report/第十三阶段/GRAM_第十三阶段_v<N>_iter<M>_<描述>报告.md
```

例如:
- `GRAM_第十三阶段_v0_vanilla-baseline_cold-setting验证报告.md`
- `GRAM_第十三阶段_v1_iter1_MinimumSemanticBridge结果报告.md`
- `GRAM_第十三阶段_v1_iter2_e5-encoder换用结果报告.md`
- `GRAM_第十三阶段_v3_iter3_alignment-loss降级尝试结果报告.md`

**每份 report 必含 section**:
1. **实验目的**:本次尝试想验证/改进什么
2. **配置**:数据集 / η / seed / 模型细节 / 超参 / LLM 版本(如涉及)
3. **命令与产物路径**:实际启动命令 + artifacts 目录
4. **核心数字**:cold NDCG@10 / warm NDCG@10 / cold Recall@10 / warm Recall@10
5. **对比**:vs vanilla GRAM(相对增益)+ vs 上一版(边际增益)
6. **Gate 结论**:pass / edge / fail(附具体门槛数字)
7. **失败原因分析**(如 fail 或 edge):log grep / loss curve / 中间结果观察
8. **下一步动作**:proceed to vN+1 / iterate vN again / abort / switch to Plan Z
9. **资源使用**:GPU 卡号 / 训练时长 / 峰值显存 / API 成本(如涉及)
10. **GPU 保护恢复确认**:实验后 CodeLlama 已重新占位(必查项)

**规则**:
- 不写 report 视为该 iteration **无效**,不能进下一步
- Report 是**决策依据**,续接时先读 report 而不是 log
- 失败的 iteration report 特别重要 —— 是消融素材 + 论文 negative result 来源

---

## 2. CANARD MVP v0-v5 逐步验证

### v0:Cold Protocol + Vanilla GRAM Baseline

**目的**:确认 setting 有效(vanilla GRAM 在 cold protocol 下确实崩)

**做什么**:
- 写 `protocol/cold_split.py`(item-level frequency-stratified 采样,移除 50% items 为 cold)
- 在 Beauty η=50% 上跑 vanilla GRAM 完整 30 epoch(单 seed)
- 对 test set 分 warm/cold subset 评测 NDCG@10, Recall@10

**时间**:1-2 天代码 + 20h 训练 = **2-3 天**

**Gate v0**:
- ✅ **通过**:vanilla GRAM 在 cold subset 上 Recall@10 ≤ 0.5%(相对 warm ≥ 90% 退化)
- ❌ **失败**:GRAM 在 cold 上依然 OK(比如 >2% Recall@10)—— 说明 cold-start 不是真问题,setting 不成立,启动 Plan Z

**iteration**:如果 gate 失败,可能是 split 不够激进,提高到 η=80% 再试

---

### v1:Minimum Semantic Bridge(无 LLM 无 alignment)

**目的**:验证"text signal 能对 cold 有救"这个最基础假设

**做什么**:
- Sentence-BERT 编码 cold item text(title + description + categories)
- **1 层 MLP** 映射 text embedding → hierarchical id(每层独立 softmax)
- Cross-entropy loss 训练 warm items 的 (text → id) 映射
- 推理时:cold item 通过 MLP 拿到 id → 加入 Trie → 参与 beam search
- **没有** LLM prior、hierarchical alignment、uncertainty routing、dual path

**时间**:3-5 天(实现 + pilot + 1-2 次 iteration)

**Gate v1**:
- ✅ **通过**:cold NDCG@10 相对 v0 提升 **≥ 5%**(相对增益,即 v0 cold NDCG=0.005 → v1 cold NDCG ≥ 0.00525)
  - 弱信号也算过,只要证明"text 能救"
- ⚠️ **边缘**:提升 2-5% —— 允许一次 iteration(换 BGE encoder / 加层 MLP)
- ❌ **失败**:提升 <2% 或退化 → **重大信号**,text-based 方向可能有根本问题

**iteration 选项**:
1. 换 encoder:sentence-BERT → E5-large-v2 → BGE-large-en-v1.5（2026-08-17 已冻结 E5 先行）
2. 加深 MLP:1 层 → 2 层 with residual
3. 换预训练目标:考虑用 SASRec pretraining 得到的 item embedding(而不是 pure text)

**如果 3 次都失败**:启动 Plan Z(text 信号对 hierarchical id 家族根本不 work,换 category hard constraint 或 retrieval-only)

#### v1 collision-safe 当前结论与 iter2 冻结决策（2026-08-17）

**Toys 有效结果**（30/30 epoch、6/6 validation、完整 test 8,789 行、missing user map=0）：

| Subset | H@10 | NDCG@10 |
|---|---:|---:|
| overall | 4.733189% | 2.847344% |
| warm | 9.293766% | 5.598919% |
| cold | **0.270149%** | **0.154617%** |

- v0 Toys cold NDCG@10 = 0.305%，v1 PASS 门槛 = 0.32025%；
- collision-safe v1 = 0.154617%，绝对下降 0.150383 percentage point，相对 **−49.31%**，故 Toys 单域 Gate = **FAIL**；
- warm NDCG@10 相对 v0 上升 3.61%，问题集中在 cold item，而不是 GRAM 整体训练崩坏；
- 资源闭环状态为 `succeeded_resource_degraded`：科学运行完成，但实验后占位保护恢复 degraded。该运维问题与科学 Gate 分开记录；
- 在 cold test 内，追加消歧后缀的 target（n=1,290）NDCG@10=0.186286%，未修改 target（n=3,152）=0.141656%。这是事后、非随机分组诊断，不能作因果结论，但说明失败不能简单归因于“后缀本身破坏预测”。

#### v1 collision-safe Beauty iter1 结果：双域 FAIL 确证（2026-08-18 补记）

**本节为事后补记。** Beauty collision-safe iter1 实际于 2026-08-18 06:06:57 完成（`status=succeeded_resource_degraded`，科学流程完整、资源恢复 degraded），但结论未被及时写入本文档。在补记之前，P0–P7 各节的"下一步"多次写有"等待 Beauty 方向证据"——该证据实际早已存在，属于流程失误，记录于此以免重复。

完整 test：10,655 预测行、missing user map=0、cold item 6,052、cold user 5,234。

| Subset | 指标 | v0 | collision-safe v1 | 相对 | 绝对事件数 v1/v0 |
|---|---|---:|---:|---:|---:|
| overall | NDCG@10 | 0.03608269 | 0.03589837 | −0.51% | 382 / 384 |
| warm | NDCG@10 | 0.06918884 | 0.06900067 | −0.27% | 374 / 375 |
| **cold** | **NDCG@10** | **0.00179372** | **0.00161338** | **−10.05%** | **8 / 9** |
| cold | H@10 | 0.00305694 | 0.00305694 | ±0.00% | **16 / 16** |

- Beauty 单域 Gate = **FAIL**（要求 ≥+5%，实得 −10.05%）。结合 Toys 的 −49.31%，**collision-safe v1 双域 FAIL 正式确证**；
- 但必须注意 Beauty 的失败**幅度与性质均与 Toys 不同**：Toys 是−49%的实质性崩塌，Beauty 的 cold H@10 前后**完全相同的 16 个命中**，cold NDCG 的 −10.05% 仅来自 9→8 个 DCG 事件的排序变化。在 5,234 个 cold user 上只有 16 个命中，该域的 cold NDCG@10 **不具备区分 v0 与 v1 的统计分辨率**；
- 因此 Beauty 的正确解读是"**vanilla GRAM 与 semantic bridge 在 cold 上同样接近于零，本指标测不出差异**"，而不是"v1 在 Beauty 上明确有害"。这一点直接触发下方 Section 3.5 的评价口径重定义；
- 资源闭环为 degraded：`degraded_scan_18263mib_on_gpu5`（restore_rc=1），GPU5 的 ablation-scan holder 未能恢复到满额。该运维问题与科学 Gate 分开记录，需单独确认当前占位状态；
- 证据：`artifacts/phase13/explore/v1_collision_safe_beauty/{status.json,metrics_cold_warm.json,run.log,predictions/}`。

#### iter2 encoder 选择（2026-08-17）

**iter2 选择：`intfloat/e5-large-v2`，不先跑 BGE。** 理由是现有 embedding 脚本已经使用 attention-mask mean pooling，与 E5 官方编码协议一致；BGE-large-en-v1.5 的标准协议使用 CLS pooling，先换 BGE 会同时改变 encoder 与 pooling，增加一个混杂变量。

iter2 冻结配置：

- 仅先跑 Toys 做快速 Gate；Beauty 的 collision-safe iter1 继续跑完，不中止；
- encoder=`intfloat/e5-large-v2`，所有 item 文本统一加 `query: ` 前缀（本任务将 embedding 作为下游特征），attention-mask mean pooling，输出做 L2 normalization；
- `max_length=256`，与当前预计算协议保持一致；MLP 仍为 1 层，GRAM training seed=2023、cold-split seed=12345，GRAM 超参、collision-safe 生成与完整性审计全部不变；
- 主比较为 E5 iter2 vs 同域 v0，辅助比较为 E5 iter2 vs Sentence-BERT collision-safe iter1；沿用 v1 Gate：≥5% PASS、2-5% EDGE、<2% 或退化 FAIL；
- BGE 仅保留为 iter3 候选；若启用，冻结为 `BAAI/bge-large-en-v1.5` + CLS pooling + L2 normalization，不使用检索 query instruction；
- 在 E5 Toys 结果前不加深 MLP，避免 encoder 与 architecture 同时变化。若 E5 仍 FAIL，再根据 Beauty 方向决定 iter3 选 BGE、2-layer residual MLP 或终止 v1。

#### E5 iter2 candidate MLP 收敛筛查与止损结论（2026-08-17）

为排除 E5 在 200 epoch 时仍未收敛造成的误杀，先执行不含 GRAM 的预筛查：复用冻结的 E5 embedding，以相同 seed、warm/validation split、1-layer independent heads、LR=1e-3、batch size=512 从头训练 400 epoch。预注册门槛为 MiniLM 最佳 validation average accuracy 的 95%。

| 指标 | MiniLM iter1 | E5 200 epoch | E5 400 epoch screen |
|---|---:|---:|---:|
| best validation average accuracy | 0.406040 | 0.336577 | **0.391275** |
| best epoch | 166 | 200（触边） | **398** |
| raw ID duplicate excess | 1,352 | 1,572 | **1,352** |
| collision-safe 修改 cold item | 1,686（28.274%） | 2,170（36.391%） | **1,791（30.035%）** |
| collision-safe output duplicate excess | 0 | 0 | **0** |

- 冻结的筛查门槛为 `0.95 × 0.406040 = 0.385738`；E5 达到 `0.391275`，高出门槛 1.44%，因此结论为 **`PASS_TO_SMOKE`**；
- 200 epoch 的 `0.3365771800279617` 与前一次运行逐值复现，说明提升来自继续训练，而不是 split/seed 漂移；
- E5 仍比 MiniLM 低 3.64%，raw duplicate excess 只追平 MiniLM，collision-safe 修改量仍多 105 个 cold item。故收敛筛查只排除了“200 epoch 训练不足”，没有证明 E5 更强；
- 随后用未参与训练的 5,963 个 cold item 原始 hierarchy 作为只读诊断参照，直接比较两种 MLP 的 cold-ID 映射。该参照不进入训练，只用于决定是否值得花费正式 GRAM 资源：

| Cold-ID 诊断 | MiniLM | E5-400 | E5 相对变化 |
|---|---:|---:|---:|
| 逐层 macro accuracy | **39.933%** | 39.289% | −1.61% |
| prefix@1 | 85.896% | **88.211%** | +2.69% |
| prefix@2 | **61.496%** | 59.366% | −3.46% |
| prefix@3 | **17.760%** | 16.317% | −8.12% |
| prefix@4 | **6.272%** | 5.719% | −8.82% |
| exact 5-level path | **3.471%** | 3.052% | −12.08% |

- E5 只改善第一层，从第二层开始持续弱于 MiniLM；而 GRAM lexical ID 依赖完整 prefix/path。结合 MiniLM collision-safe 正式 Gate 已相对 v0 下降 49.31%，继续为更弱的 E5 候选运行 smoke 或正式 GRAM 的期望收益过低；
- 因此用户于 2026-08-17 确认将最终决策修正为 **`SCREENED_OUT_BEFORE_GRAM`**：取消 E5 smoke，不跑 E5 正式 Gate，不继续堆 E5 epoch。先前 `PASS_TO_SMOKE` 保留为预注册收敛门槛的机械结果，但被更直接、事先未用于训练的 cold-ID 诊断覆盖为资源决策；
- E5 pre-screen 不消耗正式 collision-safe v1 三次尝试额度。iter2 candidate 改为 `MiniLM + 2-layer residual MLP`，先做同样的 pre-GRAM 筛查；BGE 暂缓；
- 筛查在 GPU6 完成，wall time 约 167 秒（MLP 本体 137 秒）。GPU 总占用遥测在共享卡上从 23,266 MiB 到 25,870 MiB，区间变化 2,604 MiB；该数值可能混入同卡其他进程变化，后续同类筛查按约 3 GiB 增量预算，不再使用“<1 GiB”估计；
- 证据：`artifacts/phase13/explore/v1_collision_safe_e5_toys_mlp400/status.json`、`screen_summary.json`、`mlp/training_history.json`、`id_report.json`。本次仅为 pre-GRAM screen，不消耗一次正式 efficacy Gate 结果。

#### iter2 MiniLM 2-layer residual MLP pre-GRAM 筛查冻结方案（2026-08-17）

- 研究问题：保持 MiniLM text representation、数据、split 和 loss 不变时，非线性 residual mapping 是否能显著改善 cold hierarchical-ID 映射；
- 架构：`x(384) → Linear(384,768) → GELU → Linear(768,384) → add(x) → LayerNorm → 5 independent heads`；不加 alignment、LLM prior 或 dropout；
- 训练：seed=12345、AdamW、LR=1e-3、weight decay=1e-4、batch size=512、300 epoch，沿用 v1 的 `torch.randperm` warm/validation split；
- 只跑 MLP training、cold-ID assignment、cold oracle diagnostic 与 collision-safe audit，不启动 GRAM；
- `PASS_TO_SMOKE` 必须同时满足：validation average accuracy ≥ `1.02 × 0.406040 = 0.414161`；cold macro accuracy、prefix@2、prefix@3、exact path 均严格高于 MiniLM 1-layer；raw duplicate excess ≤1,352；
- 若 validation 提升但 cold 指标混合，标记 `REVIEW`，不自动进入 smoke；其余为 `FAIL_STOP_RESIDUAL`。任何后续 smoke/正式运行仍需用户逐次确认；
- 预估 GPU6 增量显存按 3 GiB 预算，最低空闲准入 4 GiB；后台运行、独立 `status.json`、1 小时硬超时、不自动重试、不调整或释放同卡其他资源。

#### iter2 MiniLM 2-layer residual MLP 筛查结果（2026-08-17）

运行完成且所有产物/碰撞不变量通过，但未满足冻结 Gate：

| 指标 | MiniLM 1-layer | MiniLM residual | 相对变化 |
|---|---:|---:|---:|
| best validation average accuracy | **0.406040** | 0.405705 | −0.08% |
| cold macro position accuracy | **39.933%** | 39.839% | −0.24% |
| cold prefix@2 | **61.496%**（3,667/5,963） | 61.462%（3,665/5,963） | −0.05% |
| cold prefix@3 | 17.760%（1,059/5,963） | **18.313%（1,092/5,963）** | +3.12% |
| cold exact 5-level path | 3.471%（207/5,963） | **3.639%（217/5,963）** | +4.83% |
| raw ID duplicate excess | 1,352 | **1,264** | −6.51% |
| collision-safe 修改 cold item | 1,686 | **1,532** | −9.13% |

- 冻结 validation 门槛为 0.414161，residual 仅为 0.405705，且 cold macro/prefix@2 未严格优于 baseline；机械结论为 **`FAIL_STOP_RESIDUAL`**，不进入 smoke；
- residual 最佳点出现在 epoch 14，之后训练 loss 继续降到 0.0024，但 epoch 300 validation 降到 0.395302，呈现快速拟合后过拟合；继续单纯增加 epoch 无意义；
- 深层 prefix、完整 path 与碰撞确有局部改善，但 exact path 绝对只多 10 个 cold item，且这是 single-seed oracle diagnostic，不足以覆盖预注册 Gate 或证明 downstream NDCG 会改善；
- 因此取消本 residual candidate 的 smoke/正式 GRAM。该 pre-screen 与 E5 一样不消耗正式 efficacy Gate 尝试额度；若未来尝试 dropout/更强 weight decay/更小 residual width，应视为一个新的、重新冻结的 regularized-residual candidate，而不能事后改写本结果；
- GPU6 总流程约 80 秒，MLP 本体 34.1 秒；共享卡总显存区间变化 2,301 MiB，低于 3 GiB 预算；未触碰 GPU0，未运行 GRAM；
- 证据：`artifacts/phase13/explore/v1_collision_safe_minilm_residual_toys_screen/status.json`、`screen_summary.json`、`mlp/training_history.json`、`id_report.json`。

#### iter2 regularized residual 三臂 pre-GRAM 筛查冻结方案（2026-08-17 用户确认）

当前 residual 虽未过整体 Gate，但 cold prefix@3、exact path 与 collision 同向改善，同时 epoch 14 后出现明显过拟合。因而允许一次只针对泛化的低成本筛查，不换 encoder、不增加网络深度：

| Arm | 架构 | Dropout | Weight decay | 其余配置 |
|---|---|---:|---:|---|
| A0 control | 384→768→384 residual | 0 | 1e-4 | seed=12345, LR=1e-3, batch=512, 200 epoch |
| A1 dropout | 同 A0 | 0.2 | 1e-4 | 同 A0 |
| A2 stronger-WD | 同 A0 | 0 | 1e-3 | 同 A0 |

- 每个 epoch 除 per-level average 外，新增 warm-validation prefix@2、prefix@3 与 exact-path；checkpoint 统一按 `HScore = 0.5×prefix@2 + 0.3×prefix@3 + 0.2×exact-path` 选择；
- 三臂阶段**禁止读取 cold oracle 指标**。先只用 warm validation 选唯一 winner；warm Gate 为 winner HScore 相对 A0 提升 ≥2%，同时 validation average 相对 A0 退化不超过 0.5%；
- 仅当 warm Gate 通过，才对 winner 做一次 cold-ID assignment/diagnostic。Cold Gate 为：prefix@3 相对 MiniLM 1-layer 提升 ≥3%，exact path 提升 ≥5%，macro 与 prefix@2 退化均不超过 0.5%，raw duplicate excess ≤1,352，collision-safe output duplicate excess=0；
- warm Gate 未过则直接 `FAIL_WARM_GATE`，不生成 winner cold 结果；warm 过而 cold 未过则 `FAIL_STOP_REGULARIZED_RESIDUAL`；全部通过才是 `PASS_TO_SMOKE`；
- 本轮只做 MLP pre-screen，不运行 GRAM；原计划 GPU6，但启动前仅余 3,189 MiB、低于 4 GiB 准入线，未启动也未释放资源；用户随后指定改用 GPU7。GPU7 增量显存预算 3 GiB、最低空闲 4 GiB、后台独立 status、1 小时硬超时、不自动重试、不释放任何现有资源。任何 smoke/正式实验仍需用户再次确认。

#### iter2 regularized residual 三臂筛查结果（2026-08-17）

GPU7 三臂均正常完成，三臂最佳 checkpoint 都出现在 epoch 14。A1 dropout 是 warm winner，但只改善了 1/596 个 validation item 的 prefix@2，prefix@3 与 exact path 均未变化：

| Arm | best HScore | 相对 A0 | validation avg | prefix@2 | prefix@3 | exact path |
|---|---:|---:|---:|---:|---:|---:|
| A0 control | 0.387752 | — | 0.405705 | 0.647651 | 0.186242 | 0.040268 |
| A1 dropout=0.2 | **0.388591** | **+0.216%** | 0.405705 | **0.649329** | 0.186242 | 0.040268 |
| A2 weight decay=1e-3 | 0.387752 | 0.000% | 0.405705 | 0.647651 | 0.186242 | 0.040268 |

- 冻结的 HScore 门槛为 `1.02 × 0.387752 = 0.395507`；A1 只有 0.388591，`winner_hscore_gain_at_least_2pct=false`。validation 下限为 0.403676，A1 的 0.405705 通过保底检查，但不能覆盖 HScore 主门槛；
- 因此机械结论为 **`FAIL_WARM_GATE`**。严格按 warm-select/cold-once 协议停止：**没有计算任何一臂的 cold oracle 指标，没有生成 winner cold ID / collision-safe ID，也没有运行 GRAM smoke 或正式实验**；
- 这说明 dropout=0.2 仅有一个 validation prefix@2 样本的微小波动，更强 weight decay 在所选最佳点没有产生可测改善；不足以支持继续围绕这两个正则项微调。该 pre-GRAM screen 不消耗 collision-safe v1 的正式 efficacy Gate 次数；
- 总流程为 16:16:00–16:19:01（约 181 秒），三臂 MLP 训练合计约 86.9 秒。GPU7 总显存遥测为 39,001–41,305 MiB，区间变化 2,304 MiB，最低空闲 7,266 MiB，未超过 3 GiB 增量预算；该卡为共享卡，利用率不能单独归因于本筛查；
- 证据：`artifacts/phase13/explore/v1_collision_safe_minilm_regularized_residual_toys_screen/status.json`、`screen_summary.json`、`arms/*/training_history.json`、`gpu_telemetry.csv`。本 candidate 到此止损，不申请 GPU0；是否启用 BGE 应等待 Beauty collision-safe iter1 的方向证据后再单独冻结、确认。

#### BGE encoder candidate pre-GRAM 筛查冻结方案（2026-08-17 用户指定 GPU7）

用户决定利用等待 Beauty 的时间检查最后一个正交 encoder candidate，但仍先做低成本 Toys pre-GRAM 筛查，不直接消耗 GPU0 或运行 GRAM：

- encoder 固定为 `BAAI/bge-large-en-v1.5`；输入使用原始 item text，不加 retrieval query instruction，`max_length=256`，采用 CLS pooling + L2 normalization；
- bridge 固定为与 MiniLM/E5 相同的 1-layer independent heads，seed=12345、LR=1e-3、weight decay=1e-4、batch size=512，从头训练 400 epoch，checkpoint 按 warm validation average accuracy 选择；
- 第一阶段只做 warm validation Gate，BGE 必须达到 `0.995 × MiniLM 0.406040269 = 0.404010068`。未过则直接 **`FAIL_WARM_GATE`**，禁止计算 BGE cold oracle 指标，也不生成 BGE ID；
- 仅 warm Gate 通过才允许对 BGE candidate 做一次 cold assignment/diagnostic。Cold Gate 为：prefix@3 相对 MiniLM 提升 ≥3%，exact path 提升 ≥5%，macro 与 prefix@2 相对退化均不超过 0.5%，raw duplicate excess ≤1,352，collision-safe output duplicate excess=0；
- Cold Gate 未全过为 **`FAIL_STOP_BGE`**，全部通过才为 **`PASS_TO_SMOKE`**；无论哪种结果都不自动启动 smoke/正式实验。该 screen 不计入正式 collision-safe v1 efficacy Gate 次数；
- runner 硬限制在用户指定的 GPU7，BGE 编码为 fp32、batch size=16，预计增量显存上限 4,096 MiB，最低空闲准入 4,608 MiB；后台独立 status、总计 2 小时硬超时、不自动重试、不调整或释放任何现有资源；
- BGE 启动前未缓存，首次运行需要从 Hugging Face 下载模型。脚本、CLS pooling、warm-fail/cold-once 分支已经通过 bash/Python 语法检查、19 个 Phase-13 单测以及无 GPU dry-run；用户确认后已于 2026-08-17 16:42 以 `bash experiment/phase13/run_v1_bge_toys_screen.sh start 7` 启动，初始状态为 GPU7 `bge_embedding`；
- 预定证据：`artifacts/phase13/explore/v1_collision_safe_bge_toys_screen/status.json`、`screen_summary.json`、`gpu_telemetry.csv`；embedding 输出为 `artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt`。

#### BGE encoder candidate 筛查结果（2026-08-17）

BGE 正常完成并通过 warm Gate，随后按协议只对 BGE candidate 读取一次 cold diagnostic。它是目前唯一在 warm 与所有 cold 语义映射指标上均一致优于 MiniLM 的 encoder candidate：

| 指标 | MiniLM 1-layer | BGE 1-layer | BGE 相对变化 |
|---|---:|---:|---:|
| best warm validation avg | 0.406040 | **0.420134** | **+3.47%** |
| cold macro position accuracy | 39.933% | **41.513%** | **+3.96%** |
| cold prefix@1 | 85.896%（5,122/5,963） | **89.921%（5,362/5,963）** | **+4.69%** |
| cold prefix@2 | 61.496%（3,667/5,963） | **65.085%（3,881/5,963）** | **+5.84%** |
| cold prefix@3 | 17.760%（1,059/5,963） | **19.520%（1,164/5,963）** | **+9.92%** |
| cold prefix@4 | 6.272%（374/5,963） | **6.742%（402/5,963）** | **+7.49%** |
| cold exact path | 3.471%（207/5,963） | **3.656%（218/5,963）** | **+5.31%** |
| raw duplicate excess | **1,352** | 1,476 | **+9.17%（更差）** |
| collision-safe 修改 cold item | **1,686** | 1,845 | **+9.43%（更差）** |
| collision-safe output duplicate excess | 0 | 0 | 均全局唯一 |

- warm validation 下限为 0.404010，BGE 的 0.420134 通过；cold prefix@3 ≥+3%、exact ≥+5%、macro/prefix@2 ≤0.5% 退化以及 collision-safe 全局唯一也全部通过；
- 唯一失败项是冻结的 `raw_duplicate_excess_not_worse`：1,476 > 1,352。因此预注册机械结论必须保持为 **`FAIL_STOP_BGE`**，没有运行 GRAM smoke 或正式实验，也不申请 GPU0；
- 该 verdict 不能解释为“BGE 没有更强的 ID 映射能力”。恰恰相反，BGE 是当前语义映射证据最强的 encoder；未解决的问题是它把更多 cold item 压入共享 raw path，collision-safe 后需要额外 suffix，可能抵消 downstream 收益；
- 事后只读碰撞分解（不改变 Gate）显示：MiniLM/BGE 分别修改 1,686/1,845 个 cold item，其中 1,109 个重合，BGE 新增修改 736 个、同时解除 577 个，净增 159 个。在 BGE 新增修改的 736 个 item 上，prefix@3 命中由 105 增至 156、exact 由 13 增至 20；说明碰撞增加并不等同于语义映射全面变差，但这仍不能证明 suffix 负担对 GRAM 无害；
- 因而科学结论记录为 **“semantic PASS / collision-burden FAIL / downstream unresolved”**。若未来要覆盖机械止损结论，必须另行预注册一个专门回答“语义增益能否覆盖 suffix 负担”的下游实验并由用户确认，不能事后删除 raw collision Gate；
- 总流程为 16:42:09–16:52:30（约 621 秒），其中 embedding 约 199 秒、MLP 约 146 秒。GPU7 总显存遥测为 39,147–42,663 MiB，区间变化 3,516 MiB，最低空闲 5,908 MiB，未超过 4,096 MiB 增量预算；共享卡利用率不能单独归因于本筛查；
- 证据：`artifacts/phase13/explore/v1_collision_safe_bge_toys_screen/status.json`、`screen_summary.json`、`mlp/training_history.json`、`id_report.json`、`gpu_telemetry.csv`。本次为 pre-GRAM screen，不消耗正式 collision-safe v1 efficacy Gate 次数。

#### BGE semantic-gain / suffix-burden downstream diagnostic smoke 冻结方案（2026-08-17 用户确认）

该 smoke 是在保留原 `FAIL_STOP_BGE` 的前提下新增的独立诊断，研究问题仅为：BGE 更强的语义 ID 映射在 collision-safe suffix 负担存在时，能否在完全匹配的小规模 GRAM pipeline 上给出不劣或正向信号。

- 复用已经完成的 MiniLM collision-safe Toys smoke 作为 matched reference：seed=2023、T5-small、1 epoch、debug train=100、debug test=100、beam=50；BGE 只替换为 `..._v1_bge_mlpcold_collision_safe` ID，其余训练/推理参数逐项相同；
- reference 的 100 个 test user 中只有 45 个 cold user，cold hit@10 为 1/45、NDCG@10 为 0.011111。因此本 smoke **不是有统计功效的 efficacy 比较**，不允许以百分比提升直接宣称 BGE 下游有效；
- 主判据为 lexicographic `(cold hit@10, cold NDCG@10)`：严格更高且 warm NDCG@10 不低于 reference 的 50%，标记 `DIRECTIONAL_SUPPORT_FOR_FULL_GATE_DISCUSSION`；更低或触发 warm 灾难退化保护，标记 `DIRECTIONAL_HARM_STOP`；完全相同则为 `INCONCLUSIVE_TIE`；
- summary 额外做 matched-user paired hit/NDCG 变化与 45 个 cold target 的 MiniLM/BGE suffix-transition 分解，但这些均为诊断证据；任何 verdict 都不自动启动正式实验，也不改写此前的 raw collision Gate；
- 历史 matched smoke 的 GRAM `peak_reserved_mib=18,832`，GPU 总显存遥测区间变化 22,031 MiB。新 runner 将预计增量上限记为 20,480 MiB，并采用 **22,000 MiB 最低空闲准入**、后台独立 status、2 小时硬超时、不自动重试；相对历史 `peak_reserved_mib` 仍保留 3,168 MiB 余量；
- 2026-08-17 用户指定 GPU0，并冻结精确的 scan-holder 生命周期：启动前必须验证 `gram_ablation_scan_gpu0` 正在 GPU0 且 `reserve_mib=40,239`；只停止这个已验证 holder，改以同一 session/state root 启动 `18,000 MiB` holder，即释放 `22,239 MiB`；实验任意终态后改为 `30,000 MiB`，即从 interim holder 收回 `12,000 MiB`。`30,000` 指 holder tensor 的 `reserve_mib`，NVML 物理占用还会包含 holder CUDA context 及原有非 holder 进程；
- holder 改动前记录 GPU0 的全部既有非 holder PID，runner 与独立 watchdog 均禁止向这些 PID 发信号。若实验期间出现新/未知 GPU 进程，恢复器保留 `18,000 MiB` interim protection 并等待，绝不通过 kill 或强行申请 `30,000 MiB` 冒险 OOM；当资源重新满足 `32,500 MiB` 启动余量时再恢复精确 post-run holder；
- 任一代码、输入、holder PID/reserve 或 watchdog preflight 在资源改动前失败，原 `40,239 MiB` holder 保持不变；一旦 transition marker 已写入，无论实验成功、失败、超时还是用户 stop，终态目标均为 `30,000 MiB`，且不自动重跑 workload；
- runner、独立恢复 watchdog 与 matched summary 已通过 bash/Python 语法检查、19 个 Phase-13 单测、错误 GPU 拒绝、三种 verdict 分支检查及 baseline self-tie dry-run。用户确认资源口径后于 2026-08-17 18:03 以 `bash experiment/phase13/run_v1_bge_toys_downstream_smoke.sh start 0` 后台启动；holder 从 40,239 精确调整至 18,000 MiB，调整后 GPU0 空闲 22,422 MiB，既有非 holder PID 1486846 保持运行；

#### BGE downstream diagnostic smoke 结果（2026-08-17）

- 科学流程完整完成：100/100 prediction、missing user map=0、cold/warm evaluation 与 matched summary 均成功产出。冻结主判据的 45 个 cold user 在 hit@10 与 NDCG@10 上完全相同：MiniLM/BGE 均为 `1/45=0.022222` 与 `0.011111`，paired 45/45 user 全部 tie，故机械结论为 **`INCONCLUSIVE_TIE`**，不是运行失败，也不是 `DIRECTIONAL_HARM_STOP`；
- warm guard 通过：warm hit@10 均为 0.018182，BGE warm NDCG@10 为 0.007034、MiniLM 为 0.006061（+16.06%，但只涉及一个命中样本的排序变化）。overall NDCG@10 为 0.008869 vs 0.008333（+6.42%）；这些不能覆盖 cold@10 主判据；
- 非预注册的次要信号中，BGE cold hit@20 从 1/45 增至 2/45、cold NDCG@20 从 0.011111 增至 0.016440；它只多命中 1 个 cold user，样本极小，只记录为弱探索信号，不据此升级 verdict 或自动进入正式 Gate；
- 资源闭环成功：status=`completed`、resource restore=`protected_exact_30000`，独立 watchdog=`protected_exact`；GPU0 holder 已恢复 `reserve_mib=30,000`（NVML 32,028 MiB），原 PID 1486846 仍运行。实验期间遥测占用 26,149–46,973 MiB，最低空闲 1,598 MiB；相对 holder-only 起点的峰值增量 20,824 MiB，比预估 20,480 MiB 高 344 MiB，但没有 OOM；
- 主日志在 summary、status 与 holder 恢复全部完成后出现一次 shell `unexpected EOF`。根因是运行期间修改 runner 的 status 写入代码导致活动 shell 读取到变化的脚本尾部；当前文件重新 `bash -n` 已通过。该异常发生在 prediction/eval/summary/资源恢复之后，且改动不涉及模型、输入或指标代码，因此不使科学产物失效，但作为运维异常保留记录；
- 结论：本次 smoke 排除了“BGE collision-safe suffix 在小样本上造成明显 cold@10 伤害”，但没有提供支持正式全量 Gate 的正向 cold@10 证据。保持原 BGE pre-GRAM `FAIL_STOP_BGE` 与本次 `INCONCLUSIVE_TIE` 并列，不自动重跑、不自动启动正式实验；下一步等待 Beauty collision-safe iter1，再决定是否需要一个有统计功效的 BGE 设计；〔2026-08-18 补注：Beauty 已完成且为 FAIL，该"等待"已解除；但后续方向按 Section 3.5 改为口径重定义，不再回到 BGE 设计〕
- 证据：`artifacts/phase13/explore/smoke_v1_collision_safe_bge_toys/status.json`、`resource_watchdog_status.json`、`preexisting_gpu_pids.txt`、`metrics_cold_warm.json`、`smoke_summary.json`、`gpu_telemetry.csv`、`run.log`。

#### v1 iter2 candidate：BGE prefix-preserving capacity-aware assignment P0（2026-08-17 用户确认准备）

**研究问题**：BGE 已在 warm validation 与所有 cold semantic-prefix 指标上优于 MiniLM，但 raw collision 更多且 100-user downstream smoke 在 cold@10 打平。iter2 不再换 encoder 或增加普通 MLP 深度，而是检验：保留 BGE 高层语义 prefix，同时用容量约束的固定长度唯一分配替代事后追加第六个数字 suffix，能否消除 collision burden 而不丢失深层语义映射。

- 冻结输入：复用 `BAAI/bge-large-en-v1.5` CLS+L2 embedding、best epoch=331 的一层 bridge checkpoint、对应 vocab 与 raw BGE cold IDs；不重新编码、不重新训练、不读取 downstream recommendation 指标；
- 对每个 cold item 固定 BGE top-1 的前三层 token。只计算第 4/5 层 head logits，各取 top-16，形成 `16×16=256` 个尾部候选；Toys 实际共有 3,355 个 prefix-3 group，最大 group 为 93 个 cold item，因此冻结候选容量大于最大同 prefix 竞争规模；
- warm ID byte-for-byte 保持不变并预占其完整路径。每个 prefix-3 group 内使用确定性的 rectangular `linear_sum_assignment`，最小化相对原 BGE top-1 的总 logit penalty，同时要求所有 cold 完整五层路径全局唯一；禁止 adaptive top-k、禁止回退到追加 suffix、禁止修改前三层；
- P0 必须全部通过才标记 **`PASS_TO_MEDIUM_SMOKE_DISCUSSION`**：output duplicate excess=0、warm/order 不变、全部 cold 恰为 5 层、appended suffix=0、BGE prefix@3 完全保留、prefix@4 与 exact path 均不低于 MiniLM raw、macro 相对 MiniLM 退化不超过 0.5%、至少 95% cold 的第 4/5 层分配均落在各自 top-8、全部落在 top-16、infeasible group=0；否则为 **`FAIL_STOP_CAPACITY_ASSIGNMENT`**；
- cold source hierarchy 只作为 pre-GRAM exploratory diagnostic，已经参与 iteration 选择，不能声称无偏的 recommendation efficacy。即使 P0 PASS，也只允许讨论一个另行冻结的 matched medium smoke（目标约 1,000 test user）；不自动启动 P1，不自动运行 full Toys，不消耗正式 v1 efficacy Gate；
- 现有 checkpoint 兼容性已核对：text dim=1,024，五层 head 完整，level sizes=`[30,670,4568,5533,5777]`，embedding shape=`[11924,1024]`，encoder/vocab 一致。实现已通过 bash/Python syntax、4 个 capacity assignment 专项测试与原 19 个 Phase-13 单测，共 23 tests；
- P0 runner 为后台独立 status、1 小时硬超时、不自动重试、不调整 holder/lease、不 kill 其他进程。预计增量显存上限 3,072 MiB、最低空闲准入 4,096 MiB；用户确认后于 2026-08-17 18:59 以 `bash experiment/phase13/run_v1_bge_capacity_assignment_p0.sh start 6` 在 GPU6 后台启动，初始状态为 **`RUNNING / capacity_assignment`**；启动核验时 GPU6 空闲 4,149 MiB，P0 进程 NVML 占用 734 MiB，其余既有进程未被调整；
- 预定证据：`artifacts/phase13/explore/v1_iter2_bge_capacity_assignment_toys_p0/status.json`、`assignment_report.json`、`screen_summary.json`、`gpu_telemetry.csv`；候选 ID 为 `GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_bge_capacitycold.txt`。

#### v1 iter2 BGE capacity-aware assignment P0 结果（2026-08-17）

- P0 于 19:00 完成，运行约 51 秒，status=`completed`，冻结 verdict 为 **`FAIL_STOP_CAPACITY_ASSIGNMENT`**；这不是运行失败，而是候选未通过预注册效果门槛。未启动 GRAM training、downstream smoke、P1 或任何自动重试，GPU6 上的既有进程与资源均未调整；
- 工程与容量约束全部通过：11,924 条 ID 全局唯一，duplicate excess 从 1,476 降为 0；warm ID 与行序完全不变，5,963 个 cold ID 全部保持固定 5 层且无 appended suffix，BGE 前三层完全保留；3,355 个 prefix-3 group 均可行，最大 group=93；
- 分配大多只需低排名候选：74.27% 保持第 4/5 层各自 top-1，99.35% 落在 top-4，99.85% 落在 top-8，100% 落在 top-16。由此可排除“top-16 容量不够”作为本次失败主因，继续单纯扩大 top-k 没有充分依据；
- 但语义尾部 Gate 失败：capacity candidate 的 cold prefix@4=`0.054503`、exact path=`0.022975`，相对 MiniLM raw 的 `0.062720/0.034714` 分别下降 **13.10%/33.82%**；相对 raw BGE 的 `0.067416/0.036559` 分别下降 **19.15%/37.16%**。macro position accuracy=`0.409928`，虽较 MiniLM 高 2.65%，仍不能覆盖两个冻结的深层失败项；
- 解释：容量匹配解决了“唯一且固定长度”的机械问题，却暴露出第 4/5 层独立 logits 对替代 tail 的语义排序不足；25.73% cold item 被改写后，局部 logit 最优的唯一尾部并不等价于层级路径一致。下一轮若继续，重点应是**显式的父路径条件化/合法路径联合建模或 collision-aware tail learning**，而不是更大 top-k、放宽 Gate 或继续调匹配器；
- 决策：本候选在 pre-GRAM 阶段止损，不进入 medium smoke 或正式 Gate。先等待 Beauty collision-safe iter1 完成，再结合双域证据冻结下一项 v1 结构实验；v2/v3/v4/v5 仍不得沿用旧 v1 Gate 顺推；〔2026-08-18 补注：Beauty 已完成，双域 FAIL 确证；下一项实验按 Section 3.5 执行〕
- 证据：`artifacts/phase13/explore/v1_iter2_bge_capacity_assignment_toys_p0/status.json`、`assignment_report.json`、`screen_summary.json`、`gpu_telemetry.csv`、`run.log`。

---

### v1-R²：Semantic Route-and-Resolve 重立基（2026-08-18）

**目的**：停止要求生成器为零交互 cold item 生成完整且唯一的深层 Semantic ID；把共享 Semantic ID 的前三层仅作为语义路由（route），再由基于 item text embedding 的可归纳 resolver 输出真实 catalog item ID。最终 item ID 天然全局唯一，route 允许共享，不再使用 collision suffix。

**最小结构**：
- route：冻结 v0 原始 hierarchy，route depth=`3`。由冻结 GRAM validation beam 的完整 item 分数按前三层聚合成 route prior；不读取 test；
- resolver：复用 Toys 的 `BAAI/bge-large-en-v1.5` CLS+L2 item embedding，以 warm-only train prefix 构造 history→next-item 样本，训练一个 residual user projector，目标为 in-batch contrastive exact-item retrieval；cold item embedding 不作为训练 target；
- resolve：全 catalog resolver 排名与原 GRAM beam、route rank 做固定 RRF 融合，输出真实 item ID。P0 不训练/修改 GRAM，不重写任何 hierarchical-ID 文件；
- P0 冻结为 Toys validation-only exploratory screen；原 v0、collision-safe v1 test 仅作既有背景证据，不参与选参或 P0 Gate。

**P0 Gate 与止损**：
- 工程硬约束：train target 全为 warm、validation target 无泄漏、预测 100% 属于 catalog、每个用户 top-K 无重复、status/summary/配置与输入 hash 完整；
- 效果门槛：R² cold validation NDCG@10 ≥ `1.10 × v0`，cold H@50 ≥ `1.10 × v0`，且 R² cold NDCG@10 严格高于 resolver-only；warm validation NDCG@10 相对 v0 退化 ≤3%；
- 全部满足记为 `PASS_TO_R2_MEDIUM_SMOKE_DISCUSSION`；否则记为 `FAIL_STOP_R2_P0`。P0 不自动读取 test、不自动启动 GRAM/full 双域实验、不自动重试；失败后只根据预先输出的 candidate coverage、route support 与 paired 分解决定下一次结构修改。

**执行规则**：预计 ≤10 分钟的代码检查、单测与小 smoke 可使用当时空闲最多且不影响既有进程的 GPU；预计 >10 分钟的 embedding、训练或正式实验必须通过独立 tmux 后台 runner 启动，并持续写 `artifacts/phase13/explore/<experiment>/status.json`，无需实时监看。需要大显存/full GRAM 时先报告预计卡数与单卡显存，由用户指定 GPU 后再启动。

**计划证据**：`experiment/phase13/protocol/route_resolve.py`、`experiment/phase13/run_v1_r2_toys_p0.sh`、`experiment/phase13/tests/test_route_resolve.py`、`artifacts/phase13/explore/v1_r2_toys_p0/{status.json,summary.json,config.json,run.log}`。结论在实际完成后追加，不覆盖旧 v1 FAIL 记录。

#### v1-R² Toys P0 结果与根因（2026-08-18）

- P0 科学流程完整完成，status=`completed`、runtime=`118.26s`，不是 crash/OOM/timeout；40,344 个训练 transition 的 target 全为 warm，8,789 个 validation user 全量完成，未读取 test，输出 item 全在 catalog 且 top-K 无重复。冻结 verdict 为 **`FAIL_STOP_R2_P0`**，7 项 Gate 通过 5 项；不得改写为 PASS，也不自动重跑；
- exact resolver 单独给出强 cold 信号：cold NDCG@10=`0.015932`，为 v0 `0.002762` 的 **5.77×（+476.87%）**；cold H@50=`0.114037`，为 v0 `0.010305` 的 **11.07×（+1006.67%）**。这支持“history→item text embedding 的可归纳 exact resolver 能显著扩大 cold reachability”，但 resolver-only warm NDCG@10=`0.021202`，比 v0 `0.063580` 低 66.65%，不能单独作为完整推荐器；
- 当前 route fusion 虽仍优于 v0 cold：R² cold NDCG@10=`0.004652`（+68.45%）、H@50=`0.082665`（约 8.02×），但只保留 resolver cold NDCG@10 的 29.20%，且 warm NDCG@10=`0.057571`，相对 v0 **−9.45%**。因此失败的两项 Gate 分别是“R² cold NDCG@10 严格高于 resolver-only”和“warm 退化≤3%”；
- 根因是 P0 把 depth=3 错当成 coarse route：Toys 的 depth1/2/3 route 数分别为 `30 / 786 / 8,402`，平均每 route item 数为 `397.47 / 15.17 / 1.42`；depth3 已接近 item identity。v0 beam 前 8 个唯一 route 对目标 route 的覆盖率在 depth1/2/3 上分别为 cold `64.60% / 25.07% / 4.63%`。因此 depth3 route restriction/bonus 大量排斥了 resolver 找到的正确 cold item；
- 机制结论：**resolver 方向获得强支持，当前 depth3 route 接口被否定**。下一次结构实验若获准，应冻结为更浅的 depth1/2 soft route、保留 full-catalog resolver safety path，并采用不因 route miss 而压制 resolver 的 late fusion；还应使用 warm/cold 分别可靠的 uncertainty/admission，而不是继续调 depth3 RRF 权重。下一次只允许 validation-only 离线/小 GPU P1 screen，先比较 v0、resolver-only 与 non-suppressive route fusion；不读取 test、不自动进入 Beauty/full GRAM；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p0/status.json`、`summary.json`、`config.json`、`predictions_validation.jsonl`、`resolver.pt`、`gpu_telemetry.csv`、`run.log`。

#### v1-R² Toys P1：Catalog-state-aware admission（预注册，2026-08-18）

- P1 不改写 P0 结果，也不重训 GRAM/resolver；仅复用 P0 已冻结的 validation `v0_top50` 与 `resolver_top50`。按 `sha256(user_id)` 奇偶固定切成 calibration/audit 两个互斥半区，所有参数只在 calibration 拟合，Gate 只看一次 audit；仍禁止读取 test 与 Beauty；
- 不尝试从用户历史猜测“下一个 target 是否 cold”。改为候选级 late admission：候选是否零交互是推荐时可知的 catalog state；特征固定为 GRAM/resolver reciprocal rank、depth1 route reciprocal rank、双路交集以及这些信号与 cold-state 的交互。候选池固定为两路 top-50 union，route 只加 soft feature，不过滤 resolver 候选；
- 使用固定 seed 的无偏置线性 pairwise ranker。仅对 calibration 中 target 已在 union 的用户构造 `positive - all negatives`，以 logistic pairwise loss + L2 拟合；不做超参搜索、不看 audit 调权。另报告 union candidate ceiling 和“按真实 target cold/warm 切换”的 label-aware oracle，但后者只作不可部署上界，不参与候选方法 Gate；
- P1 Gate 冻结为：audit cold NDCG@10 ≥ `0.90 × resolver-only`，audit warm NDCG@10 ≥ `0.97 × v0`，audit all NDCG@10 严格高于 v0，且输出全部为 catalog 内唯一 item、calibration/audit 无重叠、test 未读取。全通过记为 **`PASS_TO_R2_P2_DISCUSSION`**，否则记为 **`FAIL_STOP_R2_P1`**；不因结果放宽门槛，不自动进入 P2/Beauty/full GRAM；
- 本实验为预计数分钟的 CPU 小实验，但仍使用独立后台 status runner，证据写入 `artifacts/phase13/explore/v1_r2_toys_p1_admission/`；结果完成后在本节追加，不覆盖预注册文本。

**P1 结果与结论（2026-08-18）**：

- 实验正常完成，runtime=`40.82s`、status=`completed`，不是 crash/OOM/timeout；calibration/audit=`4,381/4,408` 且互斥，未读取 test，输出均为 catalog 内唯一 item。冻结 verdict 为 **`FAIL_STOP_R2_P1`**：7 项 Gate 通过 6 项，唯一失败项为 warm 保真；
- audit overall NDCG@10 从 v0 `0.031579` 提升到 P1 `0.035146`（**+11.30%**）；cold NDCG@10=`0.015830`，保留 resolver-only `0.017025` 的 **92.98%**，通过预注册的 90% Gate；但 warm NDCG@10 从 v0 `0.061621` 降至 `0.054798`（**−11.07%**），低于要求的 `0.97×v0=0.059773`；
- 失败不是 calibration 过拟合：同一冻结权重在 calibration 上 warm NDCG@10 也从 `0.065493` 降至 `0.057940`（−11.53%）。结构性原因是统一线性排序目标给所有 cold candidate 正的全局偏置，并让宽泛的 depth1 route prior 参与所有候选排序；P1 top-10 的 cold candidate 占比为 `35.98%`，而 union 中仅 `24.77%`、v0 top-10 仅 `1.66%`。这扩大了 cold exposure，却把可靠的 warm GRAM item 挤出前十；audit warm 相对 v0 净损失 27 个 H@10（lost 56、gained 29）；
- label-aware oracle（warm 走 v0、cold 走 resolver）audit overall NDCG@10=`0.039131`，比 v0 高 23.92%，说明双专家互补性存在；问题不是 resolver 无效或 union ceiling 不足，而是当前**单一共享线性 score 无法在不知道目标标签时安全分配 top-K 名额**。下一步若继续，不应调 epochs/L2 或放宽 Gate；应把 warm 主榜设为不可被重排的 safety anchor，再以显式 cold quota/替换收益或双名单 constrained interleaving 做小规模 validation-only P2，并预注册 warm 损失预算；不得自动启动。
- 证据：`artifacts/phase13/explore/v1_r2_toys_p1_admission/{status.json,summary.json,config.json,admission.pt,predictions_audit.jsonl,run.log}`。

#### v1-R² Toys P2：Warm-anchored constrained interleaving（预注册，2026-08-18）

- P2 只读 P0 冻结的 validation `v0_top50/resolver_top50` 与 catalog cold-state，不重训或修改 GRAM/resolver，不继承 P1 线性权重；沿用同一 `sha256(user_id)` calibration/audit 切分，禁止读取 test/Beauty；
- 推理规则固定为：完整保留 v0 前 `protected_prefix` 的 item 与次序；随后从 resolver 排名中筛出 catalog-state=cold 且尚未出现的 item，插入最多 `cold_quota` 个；再按原次序补回其余 v0 和 resolver candidate。该机制只给 cold 有限插槽，不允许重新打分或重排受保护 warm 主榜；
- calibration 搜索空间预先冻结为 `protected_prefix ∈ {5,6,7,8,9}`、`cold_quota ∈ {1,2,3}`，且 `protected_prefix+cold_quota≤10`。可行配置必须同时满足 calibration warm NDCG@10 ≥ `0.98×v0`、overall NDCG@10 > v0；在可行集合中按 cold NDCG@10、overall NDCG@10、warm NDCG@10 依次最大化，再以更小 quota、更长 protected prefix 作确定性 tie-break。Audit 不参与选择；若 calibration 无可行配置，直接 `FAIL_STOP_R2_P2`；
- audit Gate 冻结为：warm NDCG@10 ≥ `0.97×v0`，cold NDCG@10 ≥ `2.0×v0`，cold H@10 ≥ `2.0×v0`，overall NDCG@10 > v0；同时要求 calibration/audit 互斥、输出 catalog 内唯一、test 未读取。全通过记为 **`PASS_TO_R2_MEDIUM_SMOKE_DISCUSSION`**，否则 **`FAIL_STOP_R2_P2`**；不因结果改阈值、不自动启动 Beauty/test/full GRAM；
- 这是 validation-only CPU 小实验，使用独立后台 status，证据目录为 `artifacts/phase13/explore/v1_r2_toys_p2_anchored_interleaving/`。结果完成后追加本节结论。

**P2 结果与结论（2026-08-18）**：

- P2 正常完成，runtime=`4.48s`、status=`completed`，不是 crash/OOM/timeout；calibration 选择 `protected_prefix=6, cold_quota=1`，audit 未参与选择，未读取 test，工程约束全部通过。冻结 verdict 为 **`FAIL_STOP_R2_P2`**，9 项 Gate 通过 8 项，唯一失败项仍为 warm 保真；
- 这次是边界失败而非机制崩溃：audit warm NDCG@10 从 v0 `0.06162127` 变为 `0.05970620`，保留率 **96.8922%**，只比 97% Gate 少 **0.1078 个百分点**；阈值为 `0.05977264`，绝对差仅 `0.00006643`，在 2,185 个 warm user 上的总 DCG 差 `0.1452`，小于一个 rank-10 hit 的 `0.2891`。但预注册 Gate 不能事后放宽，因此仍必须记 FAIL；
- cold 与 overall 均显著改善：cold NDCG@10 从 `0.002050` 提至 `0.005828`（**+184.26% / 2.84×**），cold H@10 从 10/2,223 提至 35/2,223（**+250% / 3.5×**）；overall NDCG@10 **+3.03%**、H@10 **+5.26%**。相较 P1 的 warm −11.07%，P2 已把 warm 损失压到 −3.11%，说明 warm anchor/cold slot 的机制方向获得支持；
- 失败来源是 calibration 边界配置的轻微外推偏移：所选配置在 calibration 的 warm 保留率为 `98.0807%`，只高于 98% 可行线 0.0807 个百分点；到 audit 降为 96.8922%。同时每个 audit user 都实际插入了 1 个 cold candidate（full-quota rate=100%），即使 resolver 对该用户没有足够可靠的 cold 证据也强制插入，最终 warm H@10 从 218 降至 205（−13），虽换来 cold H@10 +25，仍略越过 warm safety line；
- 决策：不得把 prefix 改成 7/8/9 后回看同一 audit，也不得把本次近失误改写为 PASS。若继续，下一项必须是**confidence-conditioned abstention**：保留 P2 anchor，但允许 resolver 在证据不足时不使用 cold slot；需要重新导出 resolver cosine score/margin 或候选置信度，并采用新的冻结评估协议，避免继续用同一 audit 做微调。不得自动启动；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p2_anchored_interleaving/{status.json,summary.json,config.json,calibration_grid.json,predictions_audit.jsonl,run.log}`。

#### v1-R² Toys P3：Confidence-conditioned abstention（预注册，2026-08-18）

- P3 固定 P2 的 `protected_prefix=6, cold_quota=1`，不再搜索 prefix/quota；候选仍为 resolver 排名最高的 catalog-state=cold item，但 gate 可选择 abstain，此时输出完整 v0 ranking。冻结 GRAM/resolver，不读取 test/Beauty；
- 为每个用户从冻结 resolver projector 导出 6 个推理时可见特征：top-cold cosine、top-cold 与 second-cold margin、top-cold 相对 resolver overall-top 的 gap、top-cold 的 resolver reciprocal rank、其 depth1 route 在 v0 beam 中的 reciprocal rank、其 v0 reciprocal rank。禁止使用 target identity、target cold/warm 或任何 held-fold label 作为推理特征；
- 因原 calibration/audit 的聚合结果均已查看，P3 不再把旧 audit 声称为 pristine holdout。改用 `sha256(user_id) mod 5` 的固定 5-fold out-of-fold（OOF）协议：每折只在其余四折标准化特征并拟合 L2 logistic gate，监督标签仅为“proposed cold candidate 是否等于 target”；held fold 不参与该折模型或阈值选择；
- 每折 threshold 搜索仅允许 training-fold admission coverage `{0.2,0.3,...,1.0}`。可行 threshold 必须满足 training-fold warm NDCG@10 ≥ `0.985×v0`、cold NDCG@10 > v0、overall NDCG@10 > v0；从可行集合依次最大化 cold、overall、warm NDCG@10，再偏好更低 coverage。选择后只应用到 held fold；另在全 validation 拟合一个 deployable gate 供未来独立数据使用，但不参与 OOF Gate；
- OOF Gate 冻结为：warm NDCG@10 ≥ `0.97×v0`，cold NDCG@10 与 H@10 均 ≥ `2.0×v0`，overall NDCG@10 > v0，candidate-correctness OOF AUROC ≥`0.55`，实际 admission coverage <`0.95`，五折均有可行 threshold，且输出 catalog 内唯一、test 未读取。全通过记为 **`PASS_TO_R2_FRESH_MEDIUM_SMOKE_DISCUSSION`**，否则 **`FAIL_STOP_R2_P3`**；这是 exploratory OOF 证据，不等同于独立确认，不自动进入 Beauty/test/full GRAM；
- 预计为单卡 ≤2,048 MiB、≤10 分钟的小实验；按空闲 GPU 准入启动独立后台 runner，证据目录为 `artifacts/phase13/explore/v1_r2_toys_p3_confidence_abstention/`。若 P3 失败，停止 v1-R² 的 validation 微迭代；若通过，仅讨论新鲜 medium smoke。

**P3 结果与结论（2026-08-18）**：

- P3 正常完成，runtime=`59.52s`、status=`completed`，不是 crash/OOM/timeout；8,789 个 validation user 均得到 held-by-own-gate 的 OOF prediction，五折都有 training-fold 可行 threshold，未读取 test/Beauty，工程与隔离约束通过。预注册 10 项 Gate **全部通过**，冻结 verdict 为 **`PASS_TO_R2_FRESH_MEDIUM_SMOKE_DISCUSSION`**；
- OOF overall NDCG@10 从 v0 `0.0333613` 提升至 `0.0346028`（**+3.72%**），H@10 **+7.07%**；warm NDCG@10 从 `0.0635801` 变为 `0.0627263`（**−1.34%，保留 98.657%**），通过 97% safety Gate；cold NDCG@10 从 `0.00276188` 提至 `0.00612506`（**+121.77% / 2.218×**），cold H@10 从 `0.00526677` 提至 `0.01534234`（**+191.30% / 2.913×**）；
- abstention 机制有效而非退化为 P2 全量插入：OOF admission coverage=`63.74%`、abstention rate=`36.26%`。proposed cold candidate correctness 的 base rate 仅 `0.5916%`，OOF AUROC=`0.71898`；admitted precision=`0.8033%`，相对 base rate 为 `1.358×`。这说明冻结的 resolver cosine/margin/route/rank 信号能辨别较可靠的 cold slot，而不是仅靠降低覆盖率随机换取 warm 保真；
- 逐折审计显示五折 overall NDCG@10 均为正增益（`+1.92%` 至 `+6.12%`），warm retention=`97.47%–99.36%`，fold AUROC=`0.644–0.771`；但 cold NDCG ratio 范围为 `1.83×–2.82×`，其中两折未单独达到 2×。由于 proposed-correct 事件总数仅 52，当前证据仍有稀疏正例与折间波动，不能把 aggregate OOF PASS 夸大为独立确认；
- 机制结论：**v1-R² 的有效形态应冻结为 exact resolver + warm anchor + confidence-conditioned cold-slot abstention**；P0 depth3 route restriction、P1 shared linear rerank、P2 unconditional slot 均不再恢复。P3 不再进行 validation 微调；下一步只能讨论一个预注册的新鲜 medium smoke，且结果无论成败均不得回到本 validation 调 threshold。未经用户指定资源与确认，不自动读取 test、启动 Beauty 或 full GRAM；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p3_confidence_abstention/{status.json,summary.json,config.json,confidence_gates.pt,confidence_features.jsonl,predictions_oof.jsonl,gpu_telemetry.csv,run.log}`。

#### v1-R² Toys P3 fresh medium smoke（预注册，2026-08-18）

- 本实验是 P3 通过后的**首次且一次性 Toys test medium smoke**。模型、特征、anchor 与 gate 全部冻结：复用 P0 `resolver.pt`、BGE item embedding、P3 `confidence_gates.pt/full_model`，固定 threshold=`0.3266778290271759`、`protected_prefix=6`、`cold_quota=1`；不得在 test 上训练、选择 threshold、修改特征或搜索任何超参数；
- 在打开或解析 test prediction 指标前，只根据 `user_sequence.txt` 中的 user ID，按 `(sha256(user_id), user_id)` 升序选择恰好 1,000 人。选择过程不得读取 target；先落盘 `selection_manifest.json`（含规则、样本 user ID/hash 与 manifest hash），再打开 test prediction。解析器必须在拆解 metrics/gold/prediction 字段前跳过非样本 user，样本外 test 指标不得解析、聚合或报告；
- test 语义固定为每条 sequence 的最后一个 item 为 target、此前最多 20 个 item 为 history；v0 来自唯一冻结的 Toys test prediction 文件，resolver 对同一 history 做 full-catalog top-50，P3 仅用六个冻结的 inference-visible 特征和 full gate 决定插入或 abstain。另报告 resolver-only 与 label-aware oracle 作为诊断，不参与调参；
- one-shot Gate 冻结为：样本数恰为 1,000；sample selection target-free 且 manifest 先于 test parse；样本外行未被解析；P3 输出全部为 catalog 内唯一 item；warm NDCG@10 ≥ `0.97×v0`；cold NDCG@10 与 H@10 均 ≥ `1.5×v0`；overall NDCG@10 > v0；实际 admission coverage 位于 `[0.40, 0.80]`。candidate-correctness AUROC/precision 仅作稀疏正例诊断，不设事后 Gate；
- 全部通过记为 **`PASS_TO_R2_FULL_CONFIRMATION_DISCUSSION`**，任一失败记为 **`FAIL_STOP_R2_FRESH_MEDIUM_SMOKE`**。无论结果如何，不得回到 validation/test 调 gate，不自动扩大到 full test、Beauty 或 GRAM 重训；下一步必须重新讨论并预注册；
- 预计单卡增量 ≤2,048 MiB、≤10 分钟，属于小实验：检查当时空闲 GPU 后使用独立后台 runner，持续写 `artifacts/phase13/explore/v1_r2_toys_p3_fresh_medium_smoke/status.json`，不修改或抢占既有进程，不自动重试。证据目录固定为 `artifacts/phase13/explore/v1_r2_toys_p3_fresh_medium_smoke/`，完成后在本节追加冻结结论。

**Fresh medium smoke 结果与结论（2026-08-18）**：

- 实验正常完成，runtime=`10.51s`、status=`completed`，不是 crash/OOM/timeout；target-free manifest 在 test 打开前落盘，恰好解析 1,000 个预选 user（warm/cold 各 500），样本外 7,789 行在字段解析前跳过。冻结 verdict 为 **`FAIL_STOP_R2_FRESH_MEDIUM_SMOKE`**，10 项 Gate 通过 8 项；
- P3 并非无效：overall NDCG@10 从 `0.0300888` 提至 `0.0304630`（**+1.24%**），overall H@10 +2.00%；cold NDCG@10 从 `0.00574798` 提至 `0.00841465`（**+46.39% / 1.464×**），cold H@10 从 `0.010` 提至 `0.018`（**+80% / 1.8×**）。但 cold NDCG 未达到预注册的 1.5×，且 warm NDCG@10 从 `0.0544297` 降至 `0.0525113`（**−3.52%，仅保留 96.476%**），未达到 97% safety Gate；
- admission coverage=`55.3%`，处于预注册 `[40%,80%]` 范围；候选正确事件仅 4/1,000，AUROC=`0.8682` 只能作为高方差诊断，不能据此覆盖两个正式 Gate 失败。该结果说明 resolver + abstention 在新鲜 test 样本仍有 cold 增益，但 P3 gate 的 warm/cold 权衡没有稳定复现 validation OOF 的幅度；
- 决策：不得在这 1,000 个 test user 上调 threshold、prefix、quota 或特征，也不得通过放宽 1.5×/97% 门槛改写 PASS；当前 P3 不进入 full-test/Beauty confirmation。fresh medium smoke 已完成其“在昂贵正式确认前检验 validation 泛化”的止损职责，不需要重复跑；若继续研发，应作为新的 v1-R² iteration，仅用 train/validation 或新的独立域设计改进，再另行预注册尚未查看的数据 Gate；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p3_fresh_medium_smoke/{status.json,summary.json,config.json,selection_manifest.json,test_access.json,predictions_test_medium.jsonl,gpu_telemetry.csv,run.log}`。两次 test-open 前的工程启动失败分别保存在 `status.launch_failed_20260818T131012.json` 与 `status.preopen_failed_20260818T131140.json`，不影响本次科学结果。

#### v1-R² Toys P4：Counterfactual Expected-Utility Slot Router（预注册，2026-08-18）

- P4 是新的 validation-only 结构迭代，不使用或复测 fresh medium 的 1,000 个 test user，也不修改 P3 verdict。继续冻结 GRAM、P0 resolver、BGE embedding 与 `protected_prefix=6/cold_quota=1`；改变的是单一“是否插入”决策：动作集合固定为 `{abstain, insert@7, insert@10}`，使高收益候选可进入第 7 位、中等收益候选只承担第 10 位的最小 warm 位移风险；
- 对每个 user 构造共享 item-relevance 样本：P3 proposed cold candidate 与 v0 ranks 7–10 的 item union。特征仅含推理可见信息：projected-user cosine、相对 resolver top score gap、resolver reciprocal rank、depth1 route reciprocal rank、GRAM reciprocal rank、catalog cold-state、是否 proposed cold、原 GRAM rank；监督标签仅为该 item 是否等于 validation target。禁止把 target identity、target cold/warm 或 held-fold label作为推理特征；
- 仍按 `sha256(user_id) mod 5` 做 outer OOF。每折只在其余四折标准化并拟合 L2 shared logistic relevance model；对 `insert@7/10`，依据候选及被位移 v0 item 的预测相关概率与 NDCG@10 rank discount，计算相对 abstain 的反事实期望效用。动作位置取期望效用较高者；是否执行由 training-fold coverage grid `{0.2,0.3,...,0.8}` 决定，held fold 不参与模型、coverage 或阈值选择；
- training-fold 可行策略必须满足 warm NDCG@10 ≥ `0.99×v0`、cold NDCG@10 > v0、overall NDCG@10 > v0；可行集合按 cold、overall、warm NDCG@10 依次最大化，再偏好更低 coverage。另在全 validation 拟合 deployable shared model与策略，仅供未来全新独立数据使用，不参与 OOF Gate；
- OOF Gate 冻结为：五折均存在可行策略；warm NDCG@10 ≥ `0.98×v0`；cold NDCG@10 ≥ `1.8×v0`、cold H@10 ≥ `2.0×v0`；overall NDCG@10 > v0；五折 held warm retention 最低值 ≥`0.97`；至少四折 held overall NDCG@10 严格优于 v0；实际 intervention coverage <`0.80`；输出 catalog 内唯一且 test/Beauty 未读取。全通过记为 **`PASS_TO_R2_P4_FRESH_DISJOINT_CONFIRMATION_DISCUSSION`**，否则 **`FAIL_STOP_R2_P4`**；不因结果回调门槛或自动启动下一批 test/Beauty；
- 这是预计单卡增量 ≤2,048 MiB、≤10 分钟的小实验，仍使用空闲单卡与独立后台 status runner，证据目录固定为 `artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/`。若失败，优先依据 action mix、predicted-vs-realized utility 与 warm loss rank 分解判断是 relevance calibration 失败还是 slot action 空间不足，不恢复 P3 threshold 微调。

**P4 结果与结论（2026-08-18）**：

- P4 正常完成，runtime=`60.05s`、status=`completed`，不是 crash/OOM/timeout；8,789 个 validation user 全部得到 outer-fold held prediction，五折均有 training-fold 可行策略，未读取 test/Beauty，预注册 10 项 Gate **全部通过**，冻结 verdict 为 **`PASS_TO_R2_P4_FRESH_DISJOINT_CONFIRMATION_DISCUSSION`**；
- OOF overall NDCG@10 从 v0 `0.0333613` 提至 `0.0343708`（**+3.03%**），H@10 **+5.78%**；warm NDCG@10 从 `0.0635801` 变为 `0.0630235`（**−0.88%，保留 99.125%**）；cold NDCG@10 从 `0.00276188` 提至 `0.00535710`（**+93.97% / 1.940×**），cold H@10 从 `0.00526677` 提至 `0.01305244`（**+147.83% / 2.478×**）。相较 P3 OOF，P4 额外恢复 warm NDCG@10 `+0.0002972`，代价是 cold NDCG@10 `−0.0007680`，形成更保守的 Pareto 点；
- 五折 overall 均严格提升（`+1.48%–+6.00%`），held warm retention 最低=`98.114%`；fold cold NDCG ratio=`1.669×–2.213×`，因此 aggregate Gate PASS 仍不等于逐折 cold 都达到 1.8×。shared relevance OOF AUROC=`0.6757`、candidate correctness AUROC=`0.6732`，正例分别只有 138/52，需保留稀疏正例与 calibration 不确定性；
- intervention coverage=`51.81%`，paired validation 中 `insert@7` 对 cold 产生 34 个 NDCG@10 gain、对 warm 产生 37 个 loss，净 DCG 仍为正；但动作分布为 `abstain=4,235 / insert@7=4,542 / insert@10=12`，12 个 `insert@10` 均未改变真实 NDCG。故本轮支持的是**shared relevance + counterfactual displacement-risk gate**，而不是“双位置自适应”本身；论文方法叙述不得把 rank-10 分支夸大为已获实证支持；
- 决策：P4 validation screen 正式 PASS，冻结 full policy 的 coverage target=`0.5`、实际 coverage=`50.006%`、utility threshold=`0.0685426220`。不得回看已使用的 fresh-medium 1,000 人调参；下一步若获准，只能对此前字段未解析的 disjoint Toys test tranche 或新的独立域做一次性确认。确认前需重新预注册样本、Gate 与 stop rule，不自动启动；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/{status.json,summary.json,config.json,counterfactual_slot_router.pt,predictions_oof.jsonl,gpu_telemetry.csv,run.log}`。

#### v1-R² Toys P4 disjoint confirmation（预注册，2026-08-18）

- 这是 P4 full policy 的首次且一次性 test confirmation，不训练、不选参、不回看 P3 fresh-medium 结果改变规则。冻结输入为 P0 `resolver.pt`、BGE embedding、P4 `counterfactual_slot_router.pt/full_model`，固定 utility threshold=`0.06854262202978134`、action positions=`{7,10}` 与八维 feature schema；
- 样本只按 user ID 的 `(sha256(user_id), user_id)` 全序选择 **rank 1001–2000**（1-based）的恰好 1,000 人；rank 1–1000 已被 P3 fresh medium 使用，必须从其 `selection_manifest.json` 读取并验证 overlap=`0`。在打开 test prediction 前先写新的 `selection_manifest.json`，记录 rank range、前批 manifest hash、样本 user/hash 与零重叠证明；target 不参与选择；
- test parser 必须先读取每行 UID 并跳过非样本行，再解析所选行的 prediction beam；不得解析或聚合 rank 1–1000 及其余 6,789 人的 metrics/gold/prediction 字段。P4 对所选人重建 test history、冻结 resolver top-50、shared relevance probability 与反事实 action utility，严格按冻结 threshold 输出 `{abstain,insert@7,insert@10}`；
- one-shot Gate 冻结为：样本数=`1,000`；与前批 overlap=`0`；target-free manifest 先于 test open；样本外字段零解析；输出 catalog 内唯一；warm NDCG@10 ≥`0.97×v0`；cold NDCG@10 与 H@10 均 ≥`1.5×v0`；overall NDCG@10 >v0；intervention coverage 位于 `[0.35,0.65]`。shared relevance/candidate AUROC 与 rank-10 action count 仅作诊断，不设事后 Gate；
- 全部通过记为 **`PASS_TO_R2_FULL_TEST_OR_BEAUTY_DISCUSSION`**，任一失败记为 **`FAIL_STOP_R2_P4_DISJOINT_CONFIRMATION`**。结果无论成败均不得在两个 test tranche 上调模型、threshold 或 action；不自动扩大到剩余 test、Beauty 或 v2；
- 预计单卡增量 ≤2,048 MiB、≤10 分钟，使用空闲单卡和独立后台 status runner，证据目录固定为 `artifacts/phase13/explore/v1_r2_toys_p4_disjoint_confirmation/`。完成后追加冻结结论。

**P4 disjoint confirmation 结果与结论（2026-08-18）**：

- 实验正常完成，runtime=`11.63s`、status=`completed`，不是 crash/OOM/timeout；SHA-256 rank 1001–2000 的 1,000 人与前批 overlap=`0`，manifest 先于 test open 落盘，样本外 7,789 行在字段解析前跳过。11 项预注册 Gate 通过 10 项，冻结 verdict 为 **`FAIL_STOP_R2_P4_DISJOINT_CONFIRMATION`**；
- P4 在独立 tranche 仍有一致方向的弱正增益：overall NDCG@10 从 `0.0255563` 提至 `0.0258310`（**+1.07%**），H@10 **+2.27%**；warm NDCG@10 从 `0.0492438` 变为 `0.0484237`（**−1.67%，保留 98.334%**），通过 97% safety Gate；cold H@10 从 `0.0057471` 提至 `0.0095785`（**+66.67% / 1.667×**），也通过 1.5× Gate；
- 唯一失败项是 cold NDCG@10：从 `0.00386548` 提至 `0.00514262`（**+33.04% / 1.330×**），低于冻结的 1.5×。因此不得用 overall 为正、H@10 通过或“只差一个离散 hit”覆盖正式失败；P4 validation 的 cold `1.940×` 没有在独立 tranche 稳定复现；
- intervention coverage=`47.2%`，动作分布=`abstain 528 / insert@7 471 / insert@10 1`，再次确认 rank-10 分支没有形成有效机制。shared relevance AUROC=`0.6151`、candidate correctness AUROC=`0.5607`，而 proposed candidate 正例仅 3/1,000；相比 resolver-only cold NDCG@10=`0.0124306`，P4 只实现其 `41.37%`，说明主要瓶颈已从 warm 位移控制转为**如何从 resolver candidate set 识别/承载正确 cold item**，单 top-cold slot + 稀疏二分类 gate 的容量不足；
- 决策：P4 不进入 full test、Beauty confirmation 或 v2，不能在已查看的两批 test user 上调整 utility threshold、action 或 feature。v1-R² 的“exact resolver 有强 cold ceiling、warm anchor 可控”机制仍受支持，但 P3/P4 的单候选 selective slotting 未获得独立确认；后续若继续必须是候选集合级/多候选结构或新的训练信号，而不是 P5 式 threshold 微调，并需把新的独立域留作最终确认；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p4_disjoint_confirmation/{status.json,summary.json,config.json,selection_manifest.json,test_access.json,predictions_test_disjoint.jsonl,gpu_telemetry.csv,run.log}`。

#### v1-R² Toys P5-set：Pseudo-Cold Setwise Candidate Selector（预注册，2026-08-18）

- P5-set 是候选集合级结构升级，不修改 P4 threshold/action，也不读取更多 Toys test。冻结 GRAM、P0 resolver/BGE embedding，以及 P4 outer-fold shared-relevance/displacement-risk models；新增 selector 只负责从每个用户的 filtered resolver top-10 cold candidate set 中选出一个候选，再交由该用户对应的 P4 held-fold gate 决定 `{abstain,insert@7,insert@10}`；
- warm catalog 按 `uint64_be(sha256(item_id)[:8]) mod 5` 固定切分：余数 0 为 pseudo-cold item，其他为 selector-train item。selector 训练样本仅来自 train prefix 中 target 属于 selector-train item 的 transition；候选集合为冻结 resolver 在 selector-train catalog 内的 top-10，且只保留 target 确实位于 top-10 的样本。pseudo-cold audit 的 target item 从未作为 selector 训练 target 或 negative，候选集合只在 pseudo-cold catalog 内检索；允许其出现在历史中，但模型没有 item-ID 参数，只读冻结 text embedding；
- setwise selector 固定为共享 candidate encoder + permutation-invariant mean/max set context + scalar head，候选特征为 resolver cosine、相对 set-top gap、set reciprocal rank、candidate 与最近 20 个 history item embedding 的 max/mean/last cosine、set 内 cosine z-score。使用 listwise cross-entropy，固定 `pool=10, hidden=32, epochs=15, batch=256, lr=1e-3, weight_decay=1e-4, seed=32345`，不搜索超参数；训练与 pseudo-cold audit 均不使用 validation/test labels；
- validation 真 cold 的 filtered resolver candidate recall@10 必须先报告。selector 选定一个候选后，用 P4 对该 UID 的 outer-fold model 与该折原冻结 threshold 计算反事实 action；因此 validation target 只用于最终 held-style audit 指标，不参与 selector 或对应 fold risk gate训练。另报告 resolver top-1、selector top-1、candidate-pool oracle 与 label-aware routing oracle；
- Gate 冻结为：candidate-pool cold recall@10 ≥`5×` resolver filtered cold top-1 correctness；pseudo-cold audit selector top-1 accuracy ≥`1.10×` resolver top-1；real-cold selector candidate correctness ≥`1.50×` resolver filtered top-1；下游 warm NDCG@10 ≥`0.97×v0`、cold NDCG@10 与 H@10 均 ≥`2×v0`、overall NDCG@10 >v0；intervention coverage <`0.80`；输出 catalog 内唯一、selector train target 全为 warm、pseudo/train target item split 无交集、test/Beauty 未读取。全通过记为 **`PASS_TO_R2_P5_SET_NEW_DOMAIN_CONFIRMATION_DISCUSSION`**，否则 **`FAIL_STOP_R2_P5_SET`**；
- 预计单卡增量 ≤4,096 MiB、约 10–15 分钟，属于可能超过 10 分钟的训练实验，必须使用独立 tmux 后台 runner 与 `status.json`，不修改既有 GPU 进程、不自动重试。证据目录固定为 `artifacts/phase13/explore/v1_r2_toys_p5_setwise_selector/`；结果无论成败均不得再读取 Toys test，下一次独立确认只能使用新域。

#### v1-R² Toys P5-set 结果与机制结论（2026-08-18）

- 实验正常完成，runtime=`60.54s`、status=`completed`，不是 crash/OOM/timeout；31,995 个 selector-train transition 的 target 全为 warm，pseudo-cold 与 train target item split 无交集，8,789 个 validation user 全量完成，未读取 test/Beauty。12 项预注册 Gate 通过 8 项，冻结 verdict 为 **`FAIL_STOP_R2_P5_SET`**；
- 候选集合假设获得强支持：真实 cold 的 filtered resolver top-1 correctness 为 `52/4,367=1.1907%`，top-10 candidate-pool recall 为 `313/4,367=7.1674%`，达到 **6.019×**，通过冻结的 5× Gate；若能在集合内正确选取，candidate-pool oracle cold NDCG@10 为 `0.0265843`（v0 的 **9.625×**）、H@10 为 `0.0767117`（v0 的 **14.565×**）。因此当前瓶颈不是 resolver 候选池没有答案，而是无法可靠识别答案；
- 当前 pseudo-cold setwise selector 未通过迁移 Gate：在 pseudo-cold audit 的 conditional target-in-pool 样本上，top-1 accuracy 从 `415/2,057=20.1750%` 增至 `444/2,057=21.5848%`，仅为 **1.0699×（+6.99%）**，低于 1.10×；在真实 cold 上则从 resolver top-1 的 52 个正确降至 29 个，只有 **0.5577×（−44.23%）**，远低于 1.5×。逐样本只读审计显示 selector 保留原 52 个正确中的 8 个、破坏 44 个，同时从非 top-1 候选救回 21 个，净损失 23 个；这是明显的 pseudo-cold→真实 zero-interaction cold 分布迁移失败，不是单纯训练未收敛；
- 下游仍有受控弱增益，但未达到冻结 efficacy Gate：overall NDCG@10 `0.0333613→0.0337244`（**+1.09%**），warm NDCG@10 保留 **99.005%**，均通过；cold NDCG@10 `0.00276188→0.00413308`（**1.496× / +49.65%**）、cold H@10 `0.00526677→0.00938860`（**1.783× / +78.26%**），均未达到 2×。intervention coverage=`52.21%`，工程、隔离和输出完整性 Gate 全部通过；
- 机制结论：**“候选集合级结构有足够 ceiling”被确认，但“用 warm item hash split 构造 pseudo-cold 监督即可训练真实 cold selector”被否定。** 训练 loss 从 `2.2978` 稳定降至 `2.1569`，且 pseudo-cold 有小幅正增益，因此继续增加 epoch、hidden size 或微调同一组特征不针对真实失败机制。后续若继续 v1-R²，应更换监督/域模拟（使训练候选集合与真实 zero-interaction cold 的生成分布一致），或采用不依赖单点选对的风险受限 candidate-portfolio；不得在现有 Toys validation/test 上继续调 selector，也不进入 Beauty/full-test confirmation；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p5_setwise_selector/{status.json,summary.json,config.json,predictions_validation.jsonl,selector.pt,gpu_telemetry.csv,run.log}`。

#### v1-R² Toys P6：Risk-Limited Candidate Portfolio（预注册，2026-08-18）

- P6 回答单一结构问题：既然 P5 已确认 filtered resolver top-10 有显著 ceiling、但 learned selector 会破坏正确 top-1，能否**放弃单点选优**，把多个按原 resolver 顺序排列的 cold candidate 作为尾部 portfolio 承载，同时用冻结的 P4 relevance/risk model 控制 warm displacement。P6 不训练或复用 P5 selector，不改变 resolver，不读取 Toys test/Beauty；
- 每个用户先保护冻结 v0 GRAM 前 7 位，再从 resolver top-50 中按原顺序取“不在 GRAM 前 7 位的 cold item”。冻结两个非退化多候选动作：`portfolio@2` 将前 2 个候选放到 ranks 9–10，`portfolio@3` 将前 3 个候选放到 ranks 8–10；其余动作只有 `abstain`。不允许恢复单候选动作，确保本实验检验的是 portfolio 而非 P4 重跑；输出用 stable unique 补回 GRAM/resolver 尾部；
- 风险估计冻结使用 P4 的五个 outer-fold shared-relevance model。对每个 fold，候选与可能被移位的 GRAM item 只使用 target-free P4 特征估计相关概率，并按 NDCG@10 discount 差计算两个 portfolio 的 expected utility；在该 fold 的另外四折上，从 coverage grid `{0.2,…,0.8}` 选择同时满足 train warm NDCG@10 ≥`0.99×v0`、train cold 与 overall NDCG@10 均高于 v0 的 threshold，按 train cold NDCG@10、overall、warm、低 coverage 的固定字典序选最优，再原样用于 held fold。当前用户 target 不进入 action/utility/threshold；最终合并五折 held prediction 为唯一主结果；
- Gate 冻结为：filtered top-3 cold recall ≥`2.5×` filtered top-1；五折均存在 train-feasible policy；OOF warm NDCG@10 ≥`0.97×v0`；OOF cold NDCG@10、H@10 均 ≥`2×v0`，且分别严格高于同 validation 的 P4 OOF；OOF overall NDCG@10 严格高于 P4 OOF；intervention coverage 位于 `[0.20,0.80)` 且所有 intervention 均为 2/3-candidate portfolio；输出 catalog 内唯一、fold/threshold 来源审计通过、test/Beauty 未读取。全通过记为 **`PASS_TO_R2_P6_NEW_DOMAIN_CONFIRMATION_DISCUSSION`**，否则 **`FAIL_STOP_R2_P6`**；
- P5 已观察到按相同 filtered resolver 口径，真实 cold top-1/top-2/top-3 cumulative hits 为 `52/92/136`，top-3 recall=`3.1143%`、相对 top-1=`2.615×`，这里只作为已知结构前提，不作为 P6 efficacy 结论。预计单卡增量 ≤4,096 MiB、数分钟级；仍提供独立 runner、hard timeout 与 `status.json`，不修改既有 GPU 进程、不自动重试。证据目录固定为 `artifacts/phase13/explore/v1_r2_toys_p6_candidate_portfolio/`。

#### v1-R² Toys P6 结果与机制结论（2026-08-18）

- 实验正常完成，科学 runtime=`47.77s`、status=`completed`，不是 crash/OOM/timeout；8,789 个 validation user 均按 outer-fold held policy 产出 prediction，未读取 test/Beauty，输出 catalog 内唯一。13 项冻结 Gate 通过 10 项，冻结 verdict 为 **`FAIL_STOP_R2_P6`**；
- portfolio 结构相对 P4 获得一致增益：OOF overall NDCG@10=`0.0345382`，比 P4 `0.0343708` 高 **0.49%**；warm NDCG@10=`0.0632980`，比 P4 高 **0.44%**、相对 v0 保留 **99.556%**；cold NDCG@10=`0.00541617`，比 P4 高 **1.10%**；cold H@10=`0.0139684`，比 P4 高 **7.02%**。因此“多候选尾部承载优于单候选 P4”获得方向支持；
- 相对 v0，overall NDCG@10 **+3.53%**、cold H@10 **2.652×**，均通过；但 cold NDCG@10 仅 **1.961×（+96.10%）**，距离冻结 2× Gate 约 1.95%，不得因接近门槛改写为 PASS。未经风险限制的 portfolio@2/@3 虽分别达到 cold NDCG `3.159×/4.315×`，但 warm retention 仅 `95.914%/93.445%`，明确证明 candidate-set ceiling 与 warm displacement 之间仍存在结构性冲突；
- 其余两项失败来自 outer-fold 风险分配：fold 2 在最低预注册 20% train coverage 下 warm retention=`98.887%`，比冻结 99% train safety 少 0.113 个百分点；该折因此按协议无 feasible policy、held 全部 abstain，导致总体 intervention coverage=`15.781%`，低于预注册 20%。其余四折选择的都是 20% train coverage，held coverage=`18.12%–20.48%`；五折中 `portfolio@3=1,383`、`portfolio@2=4`，说明有效动作几乎完全依赖 top-3 portfolio，不是退化成 P4 单候选；
- 机制结论：**P6 是当前 Toys validation 上效果最强且 warm-safe 的 v1-R² 结构，但仍没有达到冻结的完整 efficacy/稳定性标准。** 失败已缩小为跨折风险预算不稳定，而不是候选集合或 portfolio 本身无效；然而继续在同一 Toys validation 上加入 10% coverage、把 train warm guard 从 99% 改成 98.5%，或微调 threshold 都属于看到结果后的门槛调优，不应作为下一次有效证据。下一步若继续，应把冻结 portfolio 机制迁移到未用于 P3–P6 调整的新独立域（优先 Beauty 的 domain-local validation pipeline），或在新域预注册 slotwise/knapsack risk allocation；不再读取 Toys test；
- GPU0 遥测 5 条，显存从 38,165 MiB 至 40,207 MiB，观测峰值增量约 2,042 MiB，低于 4,096 MiB 预算。科学 workload 前有 3 次 pre-work 启动失败：两次为沙箱禁止 tmux socket，一次为沙箱无法访问 NVIDIA driver；均在 GPU admission 前终止、未生成科学 artifact，归档为 `status.launch_failed_*.json`。最终经用户授权在沙箱外运行唯一科学 workload；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p6_candidate_portfolio/{status.json,summary.json,config.json,policy.json,predictions_validation.jsonl,gpu_telemetry.csv,run.log,status.launch_failed_*.json}`。

#### v1-R² Beauty P7：Cross-fitted Robust Slate Optimizer（预注册，2026-08-18）

- P7 使用 Beauty 作为 P3–P6 未参与调参的新独立域，串行建立 domain-local `BGE embedding → warm-only P0 resolver → outer-5fold P4 risk model → P7`；只读取 Beauty validation prediction，禁止读取 Beauty/Toys test。Toys 的 threshold、coverage 结果不迁移为 Beauty 参数；
- 冻结候选与动作仍为 resolver top-3 cold portfolio，保护 v0 GRAM 前 7 位，动作仅为 `abstain / portfolio@2(ranks 9–10) / portfolio@3(ranks 8–10)`。P7 不训练候选 selector；每个 outer fold 只在另外四折按用户 bootstrap 训练 3 个 shared-relevance model，分别计算两个 portfolio 的 NDCG@10 expected-utility，再以 `mean utility − β×ensemble std` 作为保守效用；
- `β∈{0,0.5,1,2}` 只在 outer-train 上选择：要求 train warm NDCG@10 ≥`0.99×v0`、cold 与 overall 均高于 v0，随后按 cold、overall、warm、较大 β 的固定字典序选择；held user 仅在最佳 robust utility >0 时执行，否则 abstain。当前 user target 不进入 feature、utility 或 action；held label 只用于最终 OOF audit；
- Gate 冻结为：filtered top-3 cold recall ≥`2.5×` top-1；五折均存在 train-feasible β；OOF warm NDCG@10 ≥`0.97×v0`；cold NDCG@10、H@10 均 ≥`2×v0`；overall NDCG@10 >v0；cold 与 overall NDCG@10 均严格超过 domain-local P4 OOF；coverage 位于 `[0.10,0.80)`，所有非 abstain 动作为多候选 portfolio；输出 catalog 内唯一、outer-fold/bootstrap 隔离审计通过、test 未读取。全通过为 **`PASS_TO_R2_P7_CONFIRMATION_DISCUSSION`**，否则 **`FAIL_STOP_R2_P7`**；
- 预计单 GPU 串行、增量显存 ≤6,144 MiB、总时长可能超过 10 分钟；必须后台运行、hard timeout、独立 `status.json`，不修改既有 GPU 进程且不自动重试。最终证据目录为 `artifacts/phase13/explore/v1_r2_beauty_p7_robust_slate/`，prerequisite 分别写入 `v1_r2_beauty_p0/` 与 `v1_r2_beauty_p4/`。

**取消说明（2026-08-18，用户纠正后）**：该 Beauty P7 预注册在科学流程开始前取消。P6 的正式 verdict 仍为 `FAIL_STOP_R2_P6`，因此不能把 P7 Beauty 表述为已通过 P6 的独立确认；若继续，它只能是新的跨域开发，会提前消耗 Beauty validation。后台在 `beauty_bge_embedding` 阶段被人工停止，P0/P4/P7 均未运行，Beauty validation/test 均未打开，status=`stopped`；本次不产生 efficacy 结论、不计为一次有效实验。下一步恢复为 Toys train/validation 内的 P7 结构修复，只有 P7 先通过冻结 Gate 后才讨论 Beauty 独立域。

#### v1-R² Toys P7：Cross-fitted Robust Slate Risk Allocation（预注册，2026-08-18）

- P7 是 P6 `FAIL_STOP_R2_P6` 后的新结构修复，不是对 P6 门槛的重算。冻结 Toys P0 resolver/BGE embedding、filtered resolver top-3 candidate、GRAM top-7 safety anchor 与 portfolio@2/@3 排位；不读取 Toys test，不复用/修改 P5 selector，不修改 P0/P4/P6 artifact；
- P6 的失败机制是单点 P4 probability + global threshold 在 fold 2 无法同时满足 20% coverage 与 99% train warm retention。P7 删除 coverage threshold：每个 outer fold 只在其余四折按 user bootstrap 训练 3 个 shared-relevance model，对 portfolio@2/@3 分别计算 expected NDCG@10 utility 的 ensemble mean/std，并以 `robust utility = mean − β×std` 直接决定 action；best robust utility ≤0 时 abstain；
- `β∈{0,0.5,1,2}` 在 outer-train 上冻结选择，feasible 条件仍为 warm NDCG@10 ≥`0.99×v0`、cold/overall 均高于 v0；按 train cold、overall、warm、较大 β 的固定字典序选择。held fold label 不参与 model、β、utility 或 action，最终只合并五折 held prediction；
- Gate 冻结为：filtered top-3 cold recall ≥`2.5×` top-1；五折均有 train-feasible β；OOF warm NDCG@10 ≥`0.97×v0`；cold NDCG@10/H@10 均 ≥`2×v0`；overall NDCG@10 >v0；cold 与 overall NDCG@10 均严格高于同域 P6 OOF；coverage 位于 `[0.10,0.80)`，所有 intervention 为多候选 portfolio；输出 catalog 内唯一、bootstrap/outer-fold 隔离通过、validation-only。全通过为 **`PASS_TO_R2_P7_BEAUTY_DISCUSSION`**，否则 **`FAIL_STOP_R2_P7`**；
- 预计单 GPU 增量 ≤4,096 MiB、几分钟级，但继续提供后台 runner、900 秒 hard timeout、`status.json` 与 no-auto-retry。证据目录固定为 `artifacts/phase13/explore/v1_r2_toys_p7_robust_slate/`。只有该 Gate 通过后才允许讨论 Beauty 独立域。

#### v1-R² Toys P7 结果与机制结论（2026-08-18，补记）

- P7 实际已于 2026-08-18 16:20–16:22 完成（runtime=`135.04s`、status=`completed`、GPU0），不是 crash/OOM/timeout；未读取 test/Beauty。14 项冻结 Gate 仅通过 4 项，冻结 verdict 为 **`FAIL_STOP_R2_P7`**。本节为事后补记；
- **P7 完全退化为全 abstain**：`action_counts={abstain: 8789, portfolio@2: 0, portfolio@3: 0}`，coverage=`0.0`。因此 P7 输出与 v0 逐指标完全相同（cold NDCG@10=`0.002762`=1.000×v0、warm=`0.063580`、overall=`0.033361`），既未超过 v0，也未超过 P4/P6，`all_outer_folds_train_feasible=false`；
- 根因是 robust utility `mean − β×std` 与 outer-train feasibility 的组合过严：五折中没有任何 β∈{0,0.5,1,2} 能同时满足 train warm ≥`0.99×v0` 与 cold/overall 均 >v0，故全部折 abstain。fold 0 的 β=0 已需 coverage `93.77%` 才可能达标，仍 `feasible=false`。这说明**用 ensemble std 惩罚 + 99% warm 硬约束的组合，在该事件密度下无可行解**，不是 portfolio 机制失效；
- 与 P6 对照，P7 是严格的倒退：P6 至少在 15.78% coverage 上取得 cold `1.961×`、overall `+3.53%`。P7 的额外 robustness 机制把可行域压缩到空集。**至此 P1–P7 七轮 gating 均未超过无条件 portfolio 基线**（见下方 Section 3.5）；
- 决策：不再在 Toys validation 上继续 P8 式机制迭代。P7 的失败与 P6 的失败指向同一个更基本的问题——warm 保真门槛与 cold 事件稀疏性不相容——该问题必须通过重定义评价口径解决，而不是再加一层风险控制；
- 证据：`artifacts/phase13/explore/v1_r2_toys_p7_robust_slate/{status.json,summary.json,config.json,robust_slate.pt,predictions_validation.jsonl,gpu_telemetry.csv,run.log}`。

#### v1-R² Beauty B1：无条件 portfolio 跨域确认（预注册，2026-08-18）

**研究问题（单一）**：Toys 上观察到的"**无条件 portfolio 插入优于 P1–P7 全部学习型 gating**"，是 Toys 反复调参下的过拟合产物，还是跨域普遍现象？

Beauty 是唯一未被 P1–P7 用于调参的域（Beauty validation 至今未打开；此前 Beauty P7 预注册已在科学流程开始前取消并确认未读取）。本实验**只用一次**。

**为什么候选是 portfolio@2 而非 P6**：3.5.4 实测显示 P6（cold H@50 1.76×、overall +3.53%）弱于其自身实验中的无条件对照 portfolio@2（2.89×、+4.96%）与 portfolio@3（3.84×、+7.35%），且三者 overall CI 下界均 >0。P6 的 outer-fold 风险模型 + coverage threshold 复杂度全部用于服务未经论证的 `warm ≥0.97×` 门槛，故不作为跨域确认候选。

**冻结输入与流程**：

- 串行建立 **domain-local** 组件，不迁移 Toys 权重：Beauty `BAAI/bge-large-en-v1.5` CLS+L2 item embedding（12,101 items，fp32、batch=16、max_length=256）→ warm-only P0 resolver（residual user projector、in-batch contrastive、warm-only train target，超参与 Toys P0 逐项相同，seed 不变）；
- v0 参照冻结为 `artifacts/phase13/explore/v0_beauty/predictions/20260811_103607_Beauty_cold50_sequential_pred_validation.tsv`（6 个 validation 文件中的**最后一个**，与 Toys P0 选取 `20260809_085251_*` 的约定一致，对应 epoch 30 训练完成态）；
- **不重训 GRAM、不修改任何 hierarchical-ID 文件、不读取 Beauty test、不读取 Toys test**；
- Beauty 层级为 `c128_l7`（7 层），Toys 为 `c32_l5`（5 层）。该差异**不影响本实验**：P3–P6 已否定 depth-route 接口，当前机制仅使用 `v0_top50`、`resolver_top50` 与 catalog cold-state，不依赖 hierarchy 层数。**不得因层数差异重调任何参数**；

**参数迁移纪律（关键）**：

`protected_prefix=7`、插入位置 `portfolio@2 → ranks 9–10`、`portfolio@3 → ranks 8–10` 均为 Toys 冻结值，**在 Beauty 上原样使用，禁止重调**。若 Beauty 表现弱于 Toys，那是诚实的泛化结果，必须如实报告，**不得回到 Beauty 上搜索更优 prefix/quota** —— 否则 Beauty 从独立确认域降级为第二个调参域，本阶段将不再拥有任何干净域。

**评估口径（Section 3.5 新口径）**：

- 主 Gate 指标：overall NDCG@10 的 paired bootstrap 95% CI（10,000 次重采样，配对同批 user，seed=20260818）；
- cold 主指标：**cold H@50**（非 cold NDCG@10）；
- 一次性算出全部五个点：`v0 / P6 / portfolio@2 / portfolio@3 / resolver-only`。建立 resolver 后这五者仅为同一份 `resolver_top50` 的不同切法，纯 CPU、秒级、零额外 GPU 成本，构成 Beauty 的完整 Pareto 前沿；
- **主候选唯一指定为 portfolio@2**，仅它参与 Gate 判定；其余四点作为同批报告的曲线上下文，不得事后改选主候选。

**Gate 冻结**：

- **PASS**：portfolio@2 的 overall NDCG@10 差值 CI 下界 >0，**且** cold H@50 差值 CI 下界 >0；
- **INCONCLUSIVE**：任一主指标 CI 跨 0；
- **FAIL**：任一主指标 CI 上界 <0；
- **事件密度保护条款（事先约定）**：若 Beauty v0 的 cold H@50 绝对事件数 **<30**，则无论点估计方向如何，结论一律记为 **INCONCLUSIVE**，不得记 FAIL。依据是 Beauty cold H@10 仅 16 个事件，collision-safe v1 曾因 9→8 个 DCG 事件被误判为 −10.05% FAIL；本条款防止同类噪声误判重演；
- 工程约束：输出为 catalog 内唯一 item、warm-only train target 审计通过、Beauty/Toys test 均未打开。

**结果解读（事先约定，防止事后叙事漂移）**：

- PASS → "简单 portfolio 优于复杂 gating"获跨域支持，构成论文主张 3 的核心证据；
- INCONCLUSIVE → 该现象在 Beauty 的事件密度下不可测；论文主张 3 降级为"Toys 单域观察 + Beauty 事件密度不足"，仍可诚实报告；
- FAIL → 该现象为 Toys 过拟合产物，主张 3 撤回，Pareto 曲线仅作单域结果报告。

**三种结果均不触发后续自动实验**，均不得回到任一域调参。

**资源与执行**：

- **GPU7**，单卡串行。峰值不叠加（embedding 先落盘再训 resolver），依 Toys 实测外推：embedding 峰值增量 3,516 MiB / 199 s，resolver 2,433 MiB / 118 s；
- 冻结 **增量上限 5,120 MiB、最低空闲准入 6,144 MiB、hard timeout 3,600 s**；
- 不属于 GRAM 训练/beam search，按 Section 6.2 **不需要 30G lease**；**不修改、不释放、不 kill 任何既有 GPU 进程**，不自动重试；
- 后台独立 tmux runner + 持续写 `status.json`；证据目录固定为 `artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation/`，prerequisite 写入 `v1_r2_beauty_p0/`；
- 结论完成后追加于本节，不覆盖预注册文本。

**待用户确认后启动。**（历史预注册状态保留；实验已完成，结果见下。）

**B1 结果与冻结结论（2026-08-18）**：

- 实验于 18:35–18:59 在 GPU7 完成，status=`completed`，冻结 verdict=**`PASS`**；10,655 个 validation user 全部进入评估，cold/warm=`5,287/5,368`，skipped=0，Beauty/Toys test 均未由本实验打开；
- 主候选 `portfolio@2` 使 overall NDCG@10 从 `0.03893624` 提高到 `0.04055018`，相对 **+4.15%**；paired-bootstrap 差值为 `+0.00161394 [0.00085109,0.00239939]`，CI 下界 >0；
- cold H@50 从 `0.01305088` 提高到 `0.03253263`，即 **2.49×（+149.28%）**，绝对事件数 `69→172`；差值 CI 为 `[0.01569888,0.02326934]`，下界 >0，且 v0 的 69 个事件未触发 `<30` 保护条款；
- cold NDCG@10 从 `0.00195327` 提高到 `0.00923071`（**4.73×**）；warm NDCG@10 从 `0.07536116` 降到 `0.07139706`（保留 **94.74%**），该 warm 损失 CI 全为负，必须作为真实 Pareto tradeoff 报告；
- `portfolio@3` 在 Beauty 上 cold H@50 更高（3.00×），但 overall 反而略低于 `portfolio@2`（+3.74% vs +4.15%）且 warm 损失更大（−8.38%）。因此主配置冻结为 `portfolio@2`，`portfolio@3` 仅作激进 Pareto 端点；
- **预注册偏差**：summary 显示 `p6_comparison_included=false`，实际只输出了 v0/resolver/@2/@3 四点，没有 Beauty domain-local P6。这不影响只由 `portfolio@2 vs v0` 构成的主 Gate，但将上方预注册的强表述更正为：**portfolio 相对 v0 的收益已跨域复现；portfolio 相对 P1–P7 复杂 gating 的直接优势仍只由 Toys 支持**，不在 Beauty 上事后补跑 P6；
- transition 冻结为 **`PASS_TO_PUBLICATION_PREPARATION`**：先收尾报告、冻结方法、评估论文创新性并重写 publication plan，不直接启动全矩阵或新的 P8 机制；
- 证据：`artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation/{status.json,summary.json,decision.md,run.log,gpu_telemetry.csv}`；报告：`report/第十三阶段/GRAM_第十三阶段_v1-R2_Beauty-B1_无条件portfolio跨域确认报告.md`。

---

### R²-v2：Cross-Domain Budget-Conditioned Slate Allocator（CBSA；预注册，2026-08-19）

**当前状态：`R2_V2_PREREGISTERED_NOT_STARTED`。** 本节是用户选择“方法论文”路线后的正式新方法线。它从 v1-R² 的已冻结 endpoint 出发，但**不是 P8，也不继承旧 v2–v5 的串行组件**。在代码、测试、冻结配置与运行授权完成前，不得启动训练或读取 Sports。

#### 单一研究问题与假设

固定 `domain-local warm-only exact resolver` 和三种可审计动作后，能否训练**一个跨域共享、预算条件化的用户级 slate allocator**，仅凭推理时可见特征判断每个用户应当执行 `no-op / portfolio@2 / portfolio@3`，从而相对当前通过 B1 的固定 `portfolio@2`：

1. 提高 overall NDCG@10；
2. 恢复一部分 warm NDCG@10；
3. cold H@50 至多承担预注册的 5% 相对非劣损失。

若做不到，结论应是“固定 portfolio 已接近当前候选接口的有效前沿”，而不是继续增加 gate、LLM、hierarchical loss 或 reflection。

#### 方法边界：只训练一个 allocator

R²-v2 的唯一新增可训练组件为 **CBSA**。v0/GRAM、BGE item embedding 与各域 resolver 都保持 v1-R² 冻结接口；不联合微调、不生成 hierarchical ID、不调用 LLM、不增加第二个 selector。

- 动作 `a0=no-op`：完整保留 `v0_top50`；
- 动作 `a2=portfolio@2`：保留 v0 ranks 1–8，在 ranks 9–10 放入去重后的 resolver 前两名，再按 v0 原顺序追加未重复 item 至 top-50；
- 动作 `a3=portfolio@3`：保留 v0 ranks 1–7，在 ranks 8–10 放入去重后的 resolver 前三名，再按 v0 原顺序追加未重复 item 至 top-50；
- 所有动作按 exact catalog item ID 做稳定去重，输出必须为 catalog 内唯一 item；resolver 候选不足时确定性退化到较小动作，不得用目标信息补位；
- allocator 输出三个 action logits，部署时取 argmax；完全相等时按 `a0 > a2 > a3` 的安全顺序决胜。

这一定义把当前方法中的“训练 resolver + 固定两槽”升级为“训练 resolver + **训练 slate 决策**”，但仍保持单一、可解释、可做因果式 action-reward 审计的结构。

#### 输入特征与泄漏禁令

特征只能来自生成推荐前即可获得的状态，具体 schema 必须在首次训练前写入 `frozen_config.json` 并锁 SHA256：

- v0 top-K 的归一化 sequence score、rank gap、top-K score 均值/方差/熵；
- resolver top-1/2/3 cosine、相邻 margin、top-K 分数统计；
- v0 与 resolver 的 item overlap、rank agreement、RRF agreement；
- resolver top-K 中 catalog-cold item 的数量/比例及可用去重候选数；
- 用户历史长度、历史 item 的 cold/warm 计数、history embedding 与 resolver 候选的相似度统计；
- warm 保留预算 `rho`，作为一个标量条件输入。

**严禁作为输入**：target item ID、target 是否 cold、任一 action 是否命中 target、per-user NDCG/reward、validation/test label、由目标反推的 oracle action。目标与 cold/warm 标签只能在源域训练阶段构造 action reward 和 aggregate warm constraint，不能进入特征。Sports 的任何 label 不得用于拟合、归一化、预算选择、阈值或代码分支。

#### 单模型结构与冻结训练配置

- 输入连续特征使用**源训练折**均值/标准差标准化；missing indicator 显式拼接，缺失连续值填 0；Sports 只能复用源域统计量；
- MLP：`input → Linear(64) → GELU → Dropout(0.1) → Linear(32) → GELU → Linear(3)`；除标准化统计量外无域专属参数；
- 预算条件：训练时 `rho ∈ {0.93, 0.95, 0.97, 0.99}` 均匀采样并直接拼入输入；**主 Gate 固定 `rho=0.97`**，其他预算只画 Pareto 曲线，不得事后替换主结果；
- optimizer=`AdamW`，lr=`1e-3`，weight decay=`1e-4`，batch size=`512`，epochs=`50`，gradient clip=`5.0`，policy temperature=`1.0`，allocator seed=`20260819`；固定 epoch，无 early stopping、无超参搜索；
- Toys/Beauty 采用 domain-balanced mini-batch，避免样本量较大的域支配目标；
- allocator 参数量、训练 step、checkpoint SHA256 与随机种子必须写入 summary。探索阶段只使用这一套结构与 seed；多 seed 属于 R²-v2 通过独立确认后的 publication 复核，不得用于挑 seed。

对源用户 `u` 和动作 `a`，三条确定性 slate 均可计算 counterfactual `NDCG@10 q_u(a)`。训练采用 `min_theta max_lambda` 的约束 Lagrangian：

`L(theta, lambda; rho) = -E_domain-balanced[sum_a p_theta(a|x_u,rho) q_u(a)] + lambda * [rho * Q_warm(a0) - E_warm(sum_a p_theta(a|x_u,rho) q_u(a))]_+`

其中 `lambda >= 0` 为同一次训练中按 constraint violation 做 projected gradient ascent 的 dual variable（dual lr=`1e-2`，初值 0），`theta` 对该目标做 gradient descent，`Q_warm(a0)` 为对应源训练折的 v0 warm NDCG@10。这样训练目标直接最大化 overall utility，同时把 warm retention 作为约束，**不人为给 cold 指标设置可调权重**；cold H@50 只进入冻结 Gate。

#### 数据角色重置与证据防火墙

- **Toys、Beauty = source/development domains**：可复用已经产生的 `v0_top50`、`resolver_top50` 和 validation labels 构造三动作 reward。由于方法设计发生在查看两域结果之后，两域对 R²-v2 不再具有独立确认资格；B1 对固定 portfolio 的历史 PASS 仍有效，但不能冒充 R²-v2 的 confirmation；
- 源域报告采用按 `user_id + domain + fixed salt` 划分的 5-fold OOF：每折只用另外四折拟合 allocator/标准化统计量，在 held-out fold 输出动作；两个域都必须覆盖五折。OOF 仅用于一次 source Gate；
- **Sports = 唯一保留的独立确认域**：在 source Gate PASS 之前，禁止读取 Sports validation/test、禁止生成其效果指标。允许先做不含 label/metric 的文件存在性与 schema audit，但该 audit 也不得统计 target 分布；
- source Gate PASS 后，在查看 Sports validation 前，用 Toys+Beauty 全部 source 数据训练唯一 final allocator checkpoint；Sports 只训练其 domain-local warm-only resolver，allocator 权重、source normalization、特征 schema、`rho=0.97` 和动作定义全部冻结；
- Toys/Beauty test 与 Sports test 全程封存。R²-v2 的独立确认只允许一次 Sports validation；若将来进入 publication，test 使用需另写 plan。

#### Stage S：Toys + Beauty source OOF Gate（唯一方法筛查）

主对照不是 v0，而是当前冻结 incumbent **unconditional `portfolio@2`**。paired bootstrap 固定 10,000 次、按 user 配对、seed=`20260819`；跨域汇总先在域内求均值，再对 Toys/Beauty 等权，避免按用户数加权。

必须同时满足：

1. 完整性：5 折无交叉、feature/normalization target-free、三动作同 user 对齐、catalog exact-item 唯一性 100%、Toys/Beauty test 未读、Sports validation/test 未读；
2. domain-balanced overall NDCG@10：`CBSA(rho=.97) - portfolio@2` 的 paired-bootstrap 95% CI **下界 > 0**；
3. domain-balanced warm NDCG@10：同一差值的 95% CI **下界 > 0**；
4. domain-balanced cold H@50：相对 portfolio@2 的非劣界为 **−5% relative**，即差值 CI 下界必须 `> -0.05 * H@50_portfolio2`；
5. 方向一致性：Toys 与 Beauty 各自的 overall 点估计差值都 `>0`，且各自 cold H@50 点估计均 `>=0.95× portfolio@2`；
6. 非退化：`a2/a3` 合计 intervention coverage 位于 `[5%,95%]`，否则视为只学到固定规则或全 abstain。

全部满足记 **`PASS_TO_R2_V2_SPORTS_CONFIRMATION_DISCUSSION`**；任一统计门失败记 **`FAIL_STOP_R2_V2_SOURCE`**；CI 触边，或任一源域 `portfolio@2` 的 cold H@50 绝对命中事件不足 30，记 **`INCONCLUSIVE_STOP_R2_V2_SOURCE`**。无论哪种结果，都必须写 report；只有 PASS 才能讨论 Sports，且不得自动启动。

#### Stage C：Sports 一次性迁移确认（仅在 Stage S PASS 后解锁）

Sports confirmation 沿用 source Gate 的三个主指标、bootstrap 次数/seed、`rho=.97` 与 5% cold 非劣 margin，唯一对照仍为 Sports domain-local `portfolio@2`：

- overall NDCG@10 差值 CI 下界 `>0`；
- warm NDCG@10 差值 CI 下界 `>0`；
- cold H@50 差值 CI 下界 `> -0.05 * H@50_portfolio2`；
- Sports `portfolio@2` 的 cold H@50 绝对命中事件若 `<30`，统一记 `INCONCLUSIVE_EVENT_DENSITY`，不得记 PASS/FAIL；
- frozen source allocator checkpoint/hash 不变，Sports-label adaptation steps=`0`，Sports test 未读，工程完整性全部通过。

全部满足才记 **`PASS_TO_R2_V2_PUBLICATION_PLAN_REWRITE`**。任一效能门失败记 **`FAIL_STOP_R2_V2_CONFIRMATION`**；不允许在 Sports 上改 budget、feature、action、网络或 threshold 后重跑。PASS 也只解锁 publication plan 重写，不自动启动全矩阵。

#### 单一性、止损与允许的修复

- 本次只有一个科学候选：CBSA。`rho` 是同一模型的部署条件，不是四个可挑选模型；主结果预先固定 `.97`；
- 允许在读取任何 source outcome 前修复单测、schema、数值稳定性和 runner；若运行后发现纯工程错误，只能保留原 artifact、写明 failure，再由用户确认是否做等价 recovery；
- 一旦看到 Stage S 指标，不得调整 hidden size、loss、dual lr、budget grid、feature、action 或 Gate。Stage S FAIL/INCONCLUSIVE 即停止本 R²-v2，不接旧 v3/v4/v5，不在相同两域发明 R²-v2.1 rescue；
- 这是“可证伪的方法预注册”，不是为了保证得到 PASS 的调参计划。

#### 预期实现、产物与资源（本次仅规划，不执行）

- 实现：`experiment/phase13/protocol/r2_v2_budgeted_slate_allocator.py`；
- 单测：`experiment/phase13/tests/test_r2_v2_budgeted_slate_allocator.py`，至少覆盖 target leakage、fold isolation、action exact-item 去重、budget condition、Sports/test guard、deterministic tie-break 与指标配对；
- Stage S runner：`experiment/phase13/run_v1_r2_v2_source_screen.sh`；artifact：`artifacts/phase13/explore/v1_r2_v2_source_screen/`；
- Stage C 仅在解锁后另建 runner；artifact 预留：`artifacts/phase13/explore/v1_r2_v2_sports_confirmation/`；
- allocator 本身为小 MLP，预计 CPU 可运行、GPU 增量显存 `<2 GiB`。如需重新生成 domain-local embedding/resolver 或 Sports v0，则按实际 workload 另做资源 preflight，并在用户指定 GPU 后遵守 Section 1.2；本预注册**不构成启动授权**。

---

### 旧 v2（历史归档，不执行）:+ LLM Prior(单次 first-pass,无 reflection)

**目的**:验证 LLM 加入能带来增量提升

**做什么**:
- 在 v1 基础上加一个 LLM stage:
  - DeepSeek V4 API 单次 first-pass(不做 reflection、不做 multi-perspective)
  - Prompt:cold item text + 5-shot warm examples → LLM 输出 predicted hierarchical id + confidence
- 训练时加一个 loss:L_llm_prior = KL(MLP output ∥ LLM prediction distribution)
- 总 loss:L_sup + λ_llm · L_llm_prior(λ_llm 从 0.5 开始调)

**时间**:3-5 天(prompt 设计 + API 集成 + 训练 + iteration)

**Gate v2**:
- ✅ **通过**:cold NDCG@10 相对 v1 提升 **≥ 3%**
- ❌ **失败**:提升 <3% 或退化 → LLM prior 在此任务上无用,砍掉这个组件,v3 直接跳过 LLM 相关

**iteration 选项**:
1. 调 λ_llm(0.3, 0.5, 1.0)
2. 换 LLM(DeepSeek V4 → GPT-4o mini upper bound 验证)
3. 换 prompt 措辞(更明确 few-shot 结构,或者加 chain-of-thought)

**API 成本**:v2 pilot 一次约 $3-5(6000 cold items × 1 call × cache),iteration 3 次约 $15。

---

### v2_iter1 历史结果(2026-08-12；正式 Gate 已失效)

**历史执行结果**:cold ndcg@10 **-48% vs collision-unsafe v1**。由于参照的 v1 与 item ID 口径已被碰撞审计否定，该数字不再构成正式 Gate；以下 OOV 与 loss 机制诊断仅作后续设计参考。

**根因诊断**(见 `report/第十三阶段/GRAM_第十三阶段_v2_toys_失败根因诊断报告.md`):

不是 token 空间不对齐,而是**双重设计缺陷**:

1. **OOV 比例极高**:LLM 每层预测有 50-61% 的 token 不在 GRAM 词表内
   - GRAM L1 vocab=30, LLM 用了 1201 个 token
   - 大量合法英文词(如 `▁princess`, `▁batcave`)不在 GRAM 精选闭集里
2. **OOV 退化策略灾难**:代码遇到 OOV 时用 uniform 分布作为 KL target
   - uniform 是熵最大分布 → 破坏 MLP 判别能力
   - 影响 61% 样本 → cold items 因缺乏 supervised signal 崩得最惨

**关键数据**:
- Warm items 上 LLM L1 匹配 GRAM 真值率 = 26.9%(远高于随机 3.3%)
- 说明 **LLM 本身有语义能力**,是**使用方式错了**

---

### v2_iter2:Vocab-Constrained LLM Prior(修复 OOV 问题)

**目的**:修复 iter1 的两个设计缺陷,验证"正确使用"LLM prior 能否带来增量

**改动**:

1. **Prompt 改造**(方向 D):
   - 给 LLM 提供每层的合法 vocab 列表(小层全给,大层给 top-N 常见)
   - Prompt 明确要求"必须从提供的 vocab 中选择"
   - 目标:OOV rate 从 61% 降到 <10%

2. **代码修复**(方向 A):
   - `load_llm_priors`:OOV 时返回 `None` 而不是 uniform
   - `train_cmd`:KL loss 计算时 mask 掉 OOV 层
   - 公式:`L_kl = Σ(mask_l · KL_l) / Σ(mask_l)`

3. **λ_llm 降低**:从 0.5 → 0.2(让 L_CE 主导,LLM prior 作为温和辅助)

**Gate v2_iter2**:
- ✅ **通过**:cold NDCG@10 相对 v1 提升 **≥ 3%**
- ⚠️ **边缘**:提升 0-3% 但 warm 不退化 → 允许 iter3
- ❌ **失败**:再次退化 → 直接跳到 v3(标记 v2 组件 "abandoned")

**iter3 选项**(如果 iter2 依然失败):
1. 方向 C:LLM 完全不做 loss,只做 few-shot retriever
2. 换 LLM:DeepSeek → GPT-4o mini
3. 换 λ_llm 到 0.1(几乎不影响 L_CE)

**时间**:2-3 天(prompt 改造 + 代码改 + 双域实验)
**成本**:$2-3(重新调用 API,vocab constraint 版本)

---

### v2_iter2 历史结果(2026-08-14；正式 Gate 已失效)

**历史执行结果**:cold ndcg@10 在旧口径下双域一致回退 —— Beauty **−43.6%** vs collision-unsafe v1、Toys **−39.8%** vs collision-unsafe v1（iter1 为 −48%）。这些数值不再用于正式 Gate。

**两项修复均已验证生效**,但不足以挽回:
- OOV mask 已正确实现(`semantic_bridge_v2.py:85-104, 241-249`),iter1 的 uniform 退化路径已消除
- vocab-constrained prompt 有效:剔除 API 失败样本后,真实 OOV 率 Beauty L1=13.6%、Toys L1=4.5%(iter1 为 50-61%),但深层仍有 26-32%

**根因(机制层,非调参问题)**:LLM 语义空间与 GRAM 的 SASRec 协同聚类空间**只在浅层对齐**。warm item 上 LLM 与 GRAM 真值一致率:

| 层 | Beauty | Toys |
|---|---|---|
| L1 | 44.5% | 60.4% |
| L2 | 22.7% | 27.5% |
| L3 | 10.1% | 16.5% |
| L4+ | 3.5-5.8% | 6.4-8.2% |

虽远高于随机(175-867 倍,证明 LLM 确有语义能力),但深层 85-96% 的样本上 KL 项把 MLP 往**错误 cluster** 拉。佐证:MLP val_acc 随 KL 项单调下降(Toys v1=0.4060 → iter1=0.3930 → iter2=0.3846),且 **λ 从 0.5 降到 0.2 时 val_acc 继续下降而非回升** —— 与"调小 λ 就能修好"矛盾。

**执行缺陷**:DeepSeek API 余额耗尽,Beauty 2871 次 / Toys 1942 次调用失败,失败样本被写成 `<unk>` + confidence **1.0**(伪装成正常回答),导致 47.5% / 32.6% 的 warm item 完全无 KL 监督。**已于 2026-08-14 补齐全部失败调用并重训 MLP 复核**:完整覆盖下 Beauty **0.2505** / Toys **0.3889**,均仍低于旧 v1 的 0.2630 / 0.4060,且 Beauty 补齐后反而更差(0.2531 → 0.2505)。这排除了旧实验中的“仅由 API 缺失导致”解释，但不能恢复其 collision-safe 正式 Gate 资格。见 `artifacts/phase13/explore/v2_verify/CONCLUSION.md`。

**历史决策（已暂停适用）**:当时按 gate 条文命中"❌ 失败 → 直接跳到 v3,标记 v2 abandoned"。碰撞审计后不得据此继续推进 v2/v3；必须先完成 collision-safe v1 的 Gate 与 iteration。

**给 v3 的提示**:浅层语义信号可靠(L1 44-60%)、深层不可靠。v3 的 hierarchical alignment 若按层加权(浅层高、深层低或为 0),可能正好避开 v2 的坑。

**Report**:`report/第十三阶段/GRAM_第十三阶段_v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL报告.md`

---

### 旧 v3（历史归档，不执行）:+ Hierarchical Contrastive Alignment Loss

**目的**:验证 hierarchical structure aware 的 alignment 能进一步提升

**做什么**:
- 在 v2 基础上加 hierarchical contrastive loss:
  - 对 hierarchical id 每一层 l,采样 triplet (anchor, positive, negative)
    - Positive:同层 cluster 的 warm item
    - Negative:同 (l-1) 层但不同 l 层的 warm item(hard negative,层级 hard mining)
  - InfoNCE loss,每层一个 temperature τ_l
- 总 loss:L_sup + λ_llm · L_llm_prior + Σ_l λ_l · L_align_l

**时间**:5-7 天(实现较复杂,triplet mining pipeline + loss balance 调优)

**Gate v3**:
- ✅ **通过**:cold NDCG@10 相对 v2 提升 **≥ 3%**,warm NDCG@10 退化 **≤ 3%**
- ❌ **失败**:cold 无提升,或 warm 退化过大 → alignment loss 不 work 或权重难调

**iteration 选项**:
1. 调 λ_l 权重(每层 0.1 vs 0.5 vs 1.0)
2. 换 negative mining 策略(hard negative → in-batch negative → semi-hard)
3. 简化为 flat alignment(不分层)看是否分层过复杂

---

### 旧 v4（历史归档，不执行）:+ Multi-Perspective Reasoning + Self-Reflection

**目的**:验证 LLM 深度使用(3 perspective + reflection)相对单次 first-pass 有价值

**做什么**:
- LLM stage 从"单次 first-pass"升级为:
  - 3 个 perspective(category / usage / attribute)独立 first-pass
  - 每个 perspective 做 cluster-inspection self-reflection(输入 warm cluster 实例让 LLM 复查)
  - Learned weights 融合 3 个 perspective 输出
- 融合后的 refined prediction 作为 L_llm_prior 的 target(替代 v2 的单次输出)

**时间**:3-5 天(prompt engineering 是主要工作)

**Gate v4**:
- ✅ **通过**:cold NDCG@10 相对 v3 提升 **≥ 3%**
- ⚠️ **边缘**:提升 1-3%,warm 不退化 —— 保留(reflection 是 novelty 加分,即使增益小也值得)
- ❌ **失败**:无提升或退化 → reflection 无用,砍掉,v5 直接跳过

**iteration 选项**:
1. 调 3 perspective 融合权重(uniform vs learned)
2. 简化 reflection prompt(去掉复杂的 cluster inspection,只做 "review your answer")
3. 加或减 few-shot examples 数量

**API 成本**:v4 每次 pilot 约 $10-15(6 calls per item × 6000 items × cache),iteration 3 次约 $45。

---

### 旧 v5（历史归档，不执行）:+ Uncertainty-Aware Dual-Path Decoding

**目的**:验证 uncertainty routing 相对纯 generative path 有价值

**做什么**:
- 计算 fused uncertainty:σ(w1 · (1 - MLP entropy) + w2 · LLM confidence + b)
- Routing:
  - high conf → generative path(MLP id 进 Trie 参与 beam search)
  - low conf → retrieval path(text similarity 直接找 warm items)
  - mid conf → gated fusion
- w1/w2/b/τ_high/τ_low 在 warm validation 上学

**时间**:3-5 天

**Gate v5**:
- ✅ **通过**:cold NDCG@10 提升 **≥ 3%** 或 warm NDCG@10 更稳(退化从 3% → 1%)
- ⚠️ **边缘**:提升 1-3%,warm 更稳 —— 保留
- ❌ **失败**:无提升,warm 也没变稳 → dual-path 无价值,砍掉,只用 generative path

**iteration 选项**:
1. 调 τ_high, τ_low 阈值搜索
2. 换 retrieval path 的 similarity metric(cosine vs BM25)
3. 简化 gated fusion 为 hard switch(去掉 mid conf 区间)

---

## 3. 探索阶段结束的判决

### 完整通过 → 进 publication phase

- v0 through v5 全部 gate 通过
- 累积 cold NDCG@10 相对 vanilla GRAM 提升 ≥ 20-30%
- Warm 退化 ≤ 3%
- **决策**:进 `PLAN_PUBLICATION.md`,启动全矩阵实验(3 datasets × 3 cold ratios × ablation)

### 部分通过(某些 vN 被砍)→ 简化版进 publication

- 比如 v4 (reflection) 被砍,但 v0-v3, v5 通过
- 累积提升仍然显著(≥ 15-20%)
- **决策**:进 `PLAN_PUBLICATION.md`,method 章节写"we found reflection did not help; we report negative result in ablation"
- Novelty 从 5 层降到 4 层,但依然可发

### 部分通过但增益弱(累积 5-15%)→ 降级 short paper

- **决策**:走 RecSys LBR / SIGIR short / CIKM short 路线
- 论文卖点转为"first cold-start protocol for hierarchical-id GenRec + diagnostic + partial fix"
- 单人 4-6 周能完成

### v0 通过但 collision-safe v1 三次有效尝试均失败 → 启动 Plan Z fallback

- 只有在 collision-safe v1 iter1/iter2/iter3 均失败后，才认为 semantic bridge 缺乏继续投入价值
- 进入本文档 Section 5 的备选方向

---

## 3.5 评价口径重定义（2026-08-18；口径已确认，Beauty B1 已 PASS）

### 3.5.1 为什么必须重定义

P1–P7 七轮机制迭代的完整结果（Toys validation，n=8,789；cold n=4,367）：

| 方案 | cold NDCG@10 | vs v0 | warm 保留 | overall NDCG@10 | vs v0 | verdict |
|---|---:|---:|---:|---:|---:|---|
| v0 GRAM | 0.002762 | 1.000× | 100% | 0.033361 | — | baseline |
| P4 counterfactual router | 0.005357 | 1.940× | 99.13% | 0.034371 | +3.03% | validation PASS→独立 FAIL |
| P6 candidate portfolio | 0.005416 | 1.961× | 99.56% | 0.034538 | +3.53% | FAIL（差 2× 门槛 1.95%） |
| P7 robust slate | 0.002762 | 1.000× | 100% | 0.033361 | 0.00% | FAIL（全 abstain） |
| **portfolio@2 无条件** | 0.008724 | 3.159× | 95.91% | 0.035016 | **+4.96%** | 未预注册 |
| **portfolio@3 无条件** | **0.011918** | **4.315×** | 93.44% | **0.035814** | **+7.35%** | 未预注册 |

**关键观察：七轮 gating 没有一个在 overall NDCG@10 上超过无条件 portfolio。** 从 P1 到 P7，机制复杂度单调上升（linear rerank → 单候选 abstention → counterfactual utility → setwise selector → portfolio+risk → robust ensemble），而每一轮买回的 warm 收益都小于让出的 cold 收益。P7 更是把可行域压缩到空集。

挡住 portfolio@3 的唯一约束是 `warm ≥ 0.97×v0`。该阈值在本文档中**从未被论证**：它不是审稿人要求，不是 baseline 惯例，是 P1 预注册时的一次性取值，随后被 P2–P7 无条件继承。而本项目的主指标 overall NDCG@10 恰恰在 portfolio@3 上达到最优。

### 3.5.2 事件密度问题（更根本）

冻结 Gate 的精度已超过测量精度。cold hit@10 的**绝对事件数**：

| 域 | cold user 数 | v0 cold hit@10 事件 | 对比方 | 事件 |
|---|---:|---:|---|---:|
| Toys | 4,367 | **23** | P6 | 61 |
| Beauty | 5,234 | **16** | collision-safe v1 | **16（完全相同）** |

由此产生的历史误判：

- **P6** 因 cold `1.961×` 未达 `2.0×` 判 FAIL——差距折合约 **1 个 hit**；
- **P2** 因 warm 差 `0.0001078` 判 FAIL——文档自己算过那是 0.145 DCG，**小于一次 rank-10 命中的 0.289**；
- **Beauty v1** 的 cold NDCG −10.05% 仅来自 **9→8 个 DCG 事件**，而 cold H@10 前后同为 16。

这两次是**噪声级失败，不是机制失败**。预注册纪律本身正确且必须保留（这是本阶段最有价值的方法论资产），但**在 16–60 个事件的密度上对二值 Gate 做 7 次连续决策，等同于对噪声做决策**。

### 3.5.3 建议的新口径（三项；均已确认采用，实测见 3.5.4）

**(A) 主指标改为 cold H@50，保留 cold NDCG@10 为辅助**

cold H@50 的事件密度（Toys）：v0=45、resolver-only=498。相比 cold NDCG@10 的 23 个事件，分辨率提升一个量级。理由是本工作的核心主张是"**扩大 cold reachability**"（resolver 已证明 cold H@50 达 11.07×v0），H@50 直接测量该主张，而 NDCG@10 同时混入了排序位置与 slate 竞争，在稀疏事件下噪声占主导。

**(B) 所有 Gate 附 paired bootstrap 95% CI，禁止裸点估计判定**

对同一批 user 的配对差值做 10,000 次 bootstrap。判定规则改为：CI 下界 >0 记 PASS，CI 跨 0 记 `INCONCLUSIVE`（而非 FAIL），CI 上界 <0 记 FAIL。这直接消除 P2/P6 式的"差 1 个 hit 判死刑"，也会诚实地把 Beauty v1 的 −10.05% 标为 INCONCLUSIVE 而非 FAIL。

**(C) 目标从"单点 PASS"改为"报告 Pareto 前沿"**

已有 4 个点（v0 → P6 → portfolio@2 → portfolio@3）构成连续可调的 warm-cold 权衡曲线。**这条曲线本身即是可发表贡献**，强于任何单点 PASS：它量化了"cold 可达性的代价"，而这正是 cold-start GenRec 领域缺失的实证结果。warm budget 不再是通过/失败的闸门，而是曲线的横轴。

### 3.5.4 新口径实测结果（2026-08-18 已执行）

三项口径已在 Toys validation 上实测。纯 CPU、无新训练、未读取 test/Beauty；复用 P0 `predictions_validation.jsonl`（含 `v0_top50`/`resolver_top50`）与 P6 `predictions_validation.jsonl`（含 `portfolio_candidates`/`p6_top50`），按 P6 定义重建无条件 portfolio。脚本 `experiment/phase13/protocol/pareto_recompute.py`，seed=20260818，逐位可复现。

**Pareto 前沿（n=8,789；cold=4,367；warm=4,422）**：

| 方案 | cold H@50 | 事件 | cold H@10 | 事件 | cold N@10 | warm N@10 | warm 保留 | **overall N@10** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0 GRAM | 0.010305 | 45 | 0.005267 | 23 | 0.002762 | 0.063580 | 100% | 0.033361 |
| P6 | 0.018090 | 79 | 0.013968 | 61 | 0.005416 | 0.063298 | 99.56% | 0.034538 |
| portfolio@2 | 0.029769 | 130 | 0.025418 | 111 | 0.008724 | 0.060982 | 95.91% | 0.035016 |
| **portfolio@3** | **0.039615** | **173** | **0.035493** | **155** | **0.011918** | 0.059413 | 93.44% | **0.035814** |
| *(resolver 上限)* | *0.114037* | *498* | *0.038699* | *169* | *0.015932* | *0.021202* | *33.3%* | *0.018584* |

**paired bootstrap 95% CI（method − v0，10,000 次重采样，配对同批 user）**：

| 方案 | cold H@50 | cold N@10 | warm N@10 | overall N@10 |
|---|---|---|---|---|
| P6 | `+0.00779 [+0.00527,+0.01053]` **PASS** | `+0.00265 [+0.00182,+0.00351]` **PASS** | `−0.00028 [−0.00057,−0.00007]` **FAIL** | `+0.00118 [+0.00075,+0.00163]` **PASS** |
| portfolio@2 | `+0.01946 [+0.01534,+0.02359]` **PASS** | `+0.00596 [+0.00474,+0.00731]` **PASS** | `−0.00260 [−0.00346,−0.00181]` **FAIL** | `+0.00166 [+0.00090,+0.00243]` **PASS** |
| portfolio@3 | `+0.02931 [+0.02427,+0.03435]` **PASS** | `+0.00916 [+0.00757,+0.01075]` **PASS** | `−0.00417 [−0.00522,−0.00314]` **FAIL** | `+0.00245 [+0.00151,+0.00342]` **PASS** |

**三项结论（含对 3.5.3 的自我更正）**：

1. **(A) cold H@50 确认采用**。v0 事件数 45 vs H@10 的 23；更重要的是它让 resolver ceiling 显形为 `0.114037`（**11.07×v0**，498 事件），而 H@10 口径把该 ceiling 压缩到 169 事件、外观仅 7.35×。论文主张 2 依赖此口径。
2. **(B) bootstrap 确认采用，但作用与 3.5.3 的预期相反——须更正**。原文预期 CI 会把 P6/P2 的 warm 近失误洗为 INCONCLUSIVE。**实测否定该预期**：三个方案的 warm 差值 CI 上界**全部为负**（P6 最窄者为 `−0.000068`），warm 退化是**统计显著的真实效应**，不是噪声。因此 bootstrap 不能用来"救回"warm 门槛；它的真实价值是**证明 warm-cold 权衡客观存在**，从而为 (C) 的 Pareto 曲线提供统计基础。
3. **(C) Pareto 前沿确认成立**。cold 单调增、warm 单调减、**overall 三者全部 PASS 且单调增**。

**决定性观察**：**三个方案的 overall NDCG@10 的 CI 下界均 >0，portfolio@3 最高**（`+0.00245 [+0.00151,+0.00342]`）。即：若主 Gate 取 overall，portfolio@3 已统计显著地胜出。P1–P7 七轮全部 FAIL 的唯一原因，是那个未经论证的 `warm ≥0.97×` 单点门槛挡掉了一个 overall 显著为正的方案。

### 3.5.5 warm 保护线的处理（建议方案）

**建议：不再把 warm 设为通过/失败闸门。** 依据是 3.5.4 的实测——warm 损失量级为 0.3–4 个 DCG、CI 很窄且显著，它是一个**可诚实报告的 tradeoff**，不是需要"通过"的判据。具体：

- **主 Gate**：overall NDCG@10 的 paired bootstrap CI 下界 >0；
- **cold 主指标**：cold H@50 的 CI 下界 >0；
- **warm**：作为**报告项**附 CI 呈现，不设阈值；论文以 Pareto 曲线形式给出，由读者/审稿人按场景选点。

**若必须给出单一保护线**（例如审稿人要求一个可部署配置），则该取值是**论文取向选择而非统计问题**，实测已把它化简为二选一：

| 保护线 | 选中方案 | cold H@50 | cold 提升 | warm 代价 | overall |
|---|---|---:|---:|---:|---:|
| warm ≥95% | portfolio@2 | 0.029769 | **2.89×** | −4.09% | +4.96% |
| warm ≥93% | portfolio@3 | 0.039615 | **3.84×** | −6.56% | +7.35% |

（原 0.97 线选中 P6，cold 仅 1.76×、overall +3.53%——三者中最弱，且该阈值从未被论证。）

**最终取向：主推 portfolio@2，但不宣称跨域 warm≥95%。** Toys 上 `portfolio@2` 的 warm retention 为 95.91%；Beauty B1 为 94.74%，比 95% 低 0.26 percentage point。因此 95% 不能继续作为跨域安全保证，`portfolio@2` 的正确定位是“相对保守的 Pareto 主点”；`portfolio@3` 仍作激进端点报告。warm 差值与 CI 必须与 cold/overall 收益并列呈现。

### 3.5.6 后续执行顺序（据实测修订）

**不应直接做 "Beauty + P6"。** 理由：

1. **P6 不是最优方案**。P6 的 overall（+3.53%）低于 portfolio@3（+7.35%），且 P6 的复杂度（outer-fold 风险模型 + coverage threshold）全部用于服务那个未经论证的 0.97 warm 约束。把 P6 迁到 Beauty，等于把一个为错误目标优化的机制拿去做确认；
2. **Beauty 是唯一未被污染的域，只能用一次**。P1–P7 全部在 Toys validation 上调参，Beauty validation 至今未打开（P7 Beauty 预注册已在流程开始前取消，确认未读取）。它是本阶段最稀缺的资产。用它确认一个次优方案是浪费；
3. **Beauty 的 cold 事件密度更低**（16 vs 23）。在旧口径下，Beauty 几乎必然产出 INCONCLUSIVE——这正是 collision-safe v1 已经发生的事。**必须先换口径，再用 Beauty**。

建议顺序：

1. ~~确认 (A)(B)(C) 三项口径~~ ✅ **已确认并实测完成**（3.5.4）；
2. ~~在 Toys validation 上重算已有 4 个点的新口径指标 + CI~~ ✅ **已完成**，Pareto 曲线见 3.5.4，证据 `artifacts/phase13/explore/v1_r2_toys_pareto_recompute/`；
3. ~~确认 warm 保护取向，冻结 `portfolio@2` 为主候选~~ ✅ **已完成**；95% 降级为参考线，不再作跨域保证；
4. ~~预注册并执行唯一一次 Beauty B1 确认~~ ✅ **已完成，主 Gate PASS**；portfolio@2 overall +4.15%、cold H@50 2.49×、warm −5.26%；
5. ~~评估“训练 resolver + 无训练 portfolio”的创新性是否足够~~ ✅ **已完成方向选择**；用户于 2026-08-19 选择方法论文路线；
6. **当前：实现前冻结 R²-v2**。按上文 R²-v2 预注册先完成代码、单测、配置 SHA 与 source-only preflight；不启动 publication 全矩阵，不把新方法记为 P8，不读取 Sports。

### 3.5.7 现有证据叙事（结果层成立，方法创新性待审）

即使后续全部 INCONCLUSIVE，以下三条已被现有证据支持，构成完整的 negative-result-driven 工作。**3.5.4 的 CI 实测已把其中两条从"观察"升级为"统计显著"**：

1. **GRAM 的生成路径对零交互 item 结构性不可达**——双域 v0 cold H@10 仅 23/4,367 与 16/5,234；且 collision-safe 修复、E5/BGE encoder 升级、capacity-aware assignment 均无法改善（本文档已积累 6 次独立 pre-GRAM screen 证据）；
2. **exact resolver 存在强 cold ceiling**——cold H@50 达 `0.114037`（**11.07×v0**、498 事件，新口径实测），但 warm 仅 v0 的 33.3%，故不能单独作推荐器；
3. **简单 portfolio 即可兑现该 ceiling，而 Toys 上的复杂 gating 反而更差**——P1–P7 七轮递增复杂度的选择性插入机制，无一超过无条件 top-3 portfolio 基线；portfolio 相对 v0 的 cold/overall 增益已在 Toys 与 Beauty 均通过 paired bootstrap 95% CI 检验。Beauty 未生成 domain-local P6，因此“复杂 gating 更差”的直接对照仍限于 Toys。

第 3 条是本工作最反直觉的实验发现：**在 cold 事件极度稀疏的条件下，学习型选择机制（linear rerank / abstention / counterfactual utility / setwise selector / robust ensemble）的收益被其自身的估计方差抵消，简单的无条件配额反而占优。** 该发现具有分析价值，但尚不能据此断定方法创新性足以支撑 CCF-B full paper。

### 3.5.8 Publication 创新性检查点（2026-08-18 初步；2026-08-19 已选择方法路线）

**历史检查点：`METHOD_NOVELTY_NOT_YET_CLEARED`；当前状态：`METHOD_ROUTE_SELECTED_R2_V2_NOT_STARTED`。**

先纠正“完全没有模型训练”的误解：

- Toys 独立训练了 residual user projector：12 epochs，40,344 个 warm-only transition，5,810 个唯一 warm target，cold target count=0；
- Beauty 另外从头训练了 domain-local projector：12 epochs，49,450 个 warm-only transition，5,926 个唯一 warm target，cold target count=0；
- 两域均用 in-batch contrastive exact-item retrieval 学习 history 到 BGE item embedding 空间的映射，没有迁移 resolver 权重。

真正的创新性风险在于：**可训练 resolver 本身只是小型 residual projector，而最终通过 Gate 的新组合部件是固定 `portfolio@2` 规则。** 它目前更像“有效的接口/策略发现”，而不是一个已成立的新模型。

与最近直接相关工作的初步对照（只用于诊断，尚非完整文献综述）：

1. Peng et al., 2026, *Can Generative Recommendation Reach Cold Items?* 已在 temporal cold-start 下系统分析 SID token/prefix reachability，并提出保留 TIGER encoder、用 learned candidate scorer 绕开 exact SID decoding 的 `TIGER-Scorer`。因此“生成路径不可达 + 改用 item scoring”本身不能再作为本项目的唯一新颖点：`https://arxiv.org/abs/2607.21101`。
2. Zhang et al., 2026, *Cold-Starts in Generative Recommendation: A Reproducibility Study* 已统一评估模型规模、identifier 设计和训练策略，并报告 textual identifier 对 item cold 有利但会伤害 warm 的显式 tradeoff。因此“cold 改善伴随 warm 退化”也已不是单独足够的论文贡献：`https://arxiv.org/abs/2603.29845`。

据此，原本存在以下两种可行定位：

- **分析/基准论文**：不强行加新模型，但必须把 collision-safe item-level audit、GRAM 与其他 SID backbone、temporal/synthetic cold protocol、多数据集多 seed、统一 Pareto/CI 评估做成明显超过已有工作的系统贡献；
- **方法论文**：保留现有 resolver ceiling 与 portfolio 作为诊断/基线，再定义一个真正针对 zero-interaction domain shift 的可学习目标或 slate allocator；不应为了看起来复杂而继续在已读取的 Toys/Beauty validation 上堆 P8。

**用户已于 2026-08-19 明确选择方法论文路线。** 因而本阶段不再停留在“是否训练过模型”的争论，也不直接把 B1 PASS 升级为 publication matrix GO；改为执行本计划新增的 **R²-v2 CBSA** 预注册。它必须相对已经通过的 `portfolio@2` 做 Pareto 改进，并在未参与设计的 Sports 上一次性确认，才有资格重写 publication plan。旧 v2–v5 不再恢复。

## 4. 探索时间表（原计划归档 + 当前 R²-v2）

| Week | 版本 | 主要工作 | 累积时间 |
|---|---|---|---|
| 1 | v0 | Cold protocol + vanilla GRAM Beauty η=50%,gate v0 | 3 天 |
| 1-2 | v1 | Sentence-BERT + 1 层 MLP,gate v1(含 iteration) | +5 天 |
| 2-3 | 旧 v2 | + LLM prior（历史归档，不执行） | +5 天 |
| 3-4 | 旧 v3 | + Hierarchical alignment loss（历史归档，不执行） | +7 天 |
| 5 | 旧 v4 | + Multi-perspective + reflection（历史归档，不执行） | +5 天 |
| 6 | 旧 v5 | + Uncertainty dual-path（历史归档，不执行） | +5 天 |
| 7-8 | Buffer | 补 iteration + 决策进 publication or Plan Z |  |
| 当前 T0 | R²-v2 freeze | 实现、单测、source schema/preflight；不读 Sports | 预计 1–2 天 |
| 当前 T1 | R²-v2 Stage S | Toys+Beauty 5-fold OOF source Gate | 预计 1 天内（不含 cache 重建） |
| 条件 T2 | R²-v2 Stage C | 仅 Stage S PASS 后讨论并执行 Sports 一次性确认 | 另行资源评估 |

原 v0–v5 工期只作历史记录。R²-v2 不套用旧“三次 iteration”额度；它按一次 source Gate + 条件式一次 confirmation 止损。

---

## 5. Plan Z:CANARD 完全失败后的 fallback 方向

**触发条件**:v0 通过(cold setting 有效)，且 collision-safe v1 的 3 次有效尝试均失败。旧 collision-unsafe v1/v2/v3 不计入这 3 次额度。

**备选清单**(优先度顺序):

### Plan Z-A:Category-Hierarchy Hard Constraint

**核心思路**:抛弃 GRAM 训练时聚出的 hierarchical id,直接用**商品自带 category 5 层**作为 hierarchical id。Cold item 天然有 category,不需要预测。

**为什么可能 work**:GRAM 聚类学到的 id 空间和真实 category 未必对齐;直接用 category 强对齐,cold item 自动定位。

**验证**:1 周内跑通,单 seed Beauty η=50%
- ✅ Gate:cold NDCG@10 提升 ≥ 10%(比 v1-v2 的 gate 更严,因为方案更直接)

### Plan Z-B:Prompt-Only Engineering

**核心思路**:不改 hierarchical id,不加 MLP,只改 GRAM 的 item prompt,加 structured tags(brand / category / attribute / usage),让 T5 encoder 从更好的 text 学到更好的 cold-item 表示。

**为什么可能 work**:GRAM 底层是 T5,prompt 质量直接影响 encoder representation。

**验证**:1 周内跑通(只需重跑训练,不改架构)。

### Plan Z-C:Retrieval-Only Baseline Upgrade

**核心思路**:放弃 GRAM 的生成路径处理 cold items,直接用 sentence-BERT retrieval 作为 cold items 的推荐路径(warm items 依然走 GRAM)。论文卖点变成 "hybrid retrieval-generation for cold-start GenRec"。

**验证**:3-5 天(retrieval pipeline 简单)。

**优先度**:Plan Z-A > Plan Z-B > Plan Z-C(A 的 novelty 最强,C 最保守)。

**如果 Plan Z 全部失败**:那时 8 周已过,应该重新讨论是否换 setting(long-tail user 4/5 匹配度)甚至换 backbone。

---

## 6. 实验协议(继承 phase12 完整 protocol)

### 6.1 探索阶段必须使用完整 protocol

参照 `experiment/phase12/run_phase12_hi_gram.sh`(**不用** light_status_sidecar.sh):

- **占位者前后让位**:runner 启动时根据 `PROTECTOR_TOOL` 环境变量选择:
  - `codellama`(GPU6 默认):`tools/run_codellama.sh stop <gpu>` 释放,结束时 `start`
  - `ablation_scan`(其他卡默认):`tools/gram_ablation_scan.sh stop` 释放,结束时 `start <gpu>`
  - exit trap 强制恢复,不管什么原因退出
- **30G GPU lease**:`experiment/gpu_memory_lease.py` sidecar hold (30720 - workload peak) MiB
- **Runner 全程监督**:进度、退出码、异常都写 status.json
- **preflight**:CPU 单测通过 + 占位者就位后才启动
- **postflight**:grep NaN / Traceback / OOM,写 log summary

**理由**:服务器资源紧张,一旦 lease 掉了 24-48h 排不回。即使探索阶段迭代频繁,也不能省这层保护。**协议开销(每次多 1-2 分钟准备)相比"排不回卡"损失可忽略**。

### 6.2 何时可以不用 lease

**仅**以下情况不需要 lease(但依然要在 CodeLlama 常驻的卡上跑):

- **纯 CPU 任务**:数据 split、metrics 汇总、report 生成
- **API-only 调用**(v2 first-pass, v4 reflection):调用 DeepSeek/GPT API,不占本地 GPU 计算(但依然在 CodeLlama 保护的卡上跑 Python 进程)
- **调试/单测**:单元测试执行(几秒到几分钟)

**任何涉及 GRAM 训练、推理、beam search 的部分**都必须走完整 protocol(lease + runner)。

### 6.3 Tmux session 命名

- `gram_phase13_explore_v{N}_iter{M}`(runner)—— 例:`gram_phase13_explore_v1_iter1`
- 对应 lease sidecar 自动派生:`gram_phase13_explore_v{N}_iter{M}_lease`

**不用 `_light` 后缀**(那是 phase12 简化 protocol 的命名)。

### 6.4 artifacts 目录

```
artifacts/phase13/explore/
├── v0_vanilla_baseline/
│   ├── run.log
│   ├── status.json
│   ├── frozen_config.json
│   ├── cold_warm_split.txt
│   ├── gpu_telemetry.csv
│   ├── gpu_lease.json
│   └── metrics_summary.json
├── v1_minimum_bridge/
│   ├── iter_1/
│   │   ├── hypothesis.md              # 本次要改什么,预期效果
│   │   ├── run.log
│   │   ├── status.json
│   │   ├── gpu_telemetry.csv
│   │   ├── frozen_config.json
│   │   └── metrics_summary.json
│   ├── iter_2/
│   └── final/                          # gate 通过后的最终版软链接或复制
├── v2_llm_prior/
│   ├── iter_1/
│   │   ├── llm_calls.jsonl             # 所有 LLM 调用输入输出(可复现)
│   │   ├── llm_cost_summary.json       # API 成本汇总
│   │   └── ...
├── v3_hierarchical_align/
├── v4_reflection_multiperspective/
├── v5_dual_path/
├── v1_r2_v2_source_screen/               # 当前新方法；Toys+Beauty source OOF
└── v1_r2_v2_sports_confirmation/         # 仅 source Gate PASS 后解锁
```

每一版都要有 `metrics_summary.json` 记录:
```json
{
  "version": "v1",
  "iteration": 1,
  "dataset": "Beauty",
  "eta": 0.5,
  "cold_ndcg_at_10": 0.00X,
  "warm_ndcg_at_10": 0.0X,
  "cold_recall_at_10": 0.0X,
  "warm_recall_at_10": 0.0X,
  "delta_vs_prev_version_cold_ndcg": "+X.X%",
  "gate_passed": true/false,
  "next_action": "proceed_to_v2" | "iterate" | "abort"
}
```

### 6.5 命令模板

```bash
# 启动 vN 探索(完整 protocol,含占位者前后让位 + 30G lease)
# 参照 phase12: run_phase12_hi_gram.sh 模板改造

# 主线卡 GPU6(CodeLlama 保护):
bash experiment/phase13/run_phase13_explore.sh start v0_beauty 6
# 副线卡 GPU0(ablation-scan holder 保护):
PROTECTOR_TOOL=ablation_scan \
  bash experiment/phase13/run_phase13_explore.sh start v1_beauty_pilot 0

# 内部会:
#   1. 检查占位者就位($PROTECTOR_TOOL,GPU6 默认 codellama、其他默认 ablation_scan)
#   2. run preflight 单元测试
#   3. tools/{run_codellama,gram_ablation_scan}.sh stop <gpu>  释放 GPU
#   4. 启动 gpu_memory_lease.py sidecar hold (30720 - peak) MiB
#   5. tmux new-session gram_phase13_explore_<sub> 跑训练
#   6. runner 全程监督,写 status.json(记录 resource_reservation)
#   7. 训练结束 tools/{run_codellama,gram_ablation_scan}.sh start <gpu> 立即恢复占位
#   8. exit trap 保证占位者一定恢复(即使 kill/crash)

# 查状态
bash experiment/phase13/run_phase13_explore.sh status v0_beauty
cat artifacts/phase13/explore/v0_vanilla_baseline/status.json | python3 -m json.tool
tail artifacts/phase13/explore/v0_vanilla_baseline/run.log

# 验证 GPU 保护未失效(GPU6 上应看到 CodeLlama ~30 GiB;GPU0 上应看到 ablation-scan holder ~30 GiB)
nvidia-smi -i 6
nvidia-smi -i 0

# gate 通过后归档最终版(硬链接节省空间)
ln -s $(pwd)/artifacts/phase13/explore/v1_minimum_bridge/iter_N \
      artifacts/phase13/explore/v1_minimum_bridge/final
```

### 6.6 每个 vN iteration 结束后必做流程

**顺序不可变,每一步都要做**:

1. **确认 GPU 保护恢复**:`nvidia-smi -i <gpu>` 看占位者是否重新占 ~30 GiB(GPU6 看 CodeLlama,其他卡看 ablation-scan holder)
2. **写 `metrics_summary.json`**:自动化脚本从 log 抽取指标
3. **写 report**:`report/第十三阶段/GRAM_第十三阶段_v<N>_iter<M>_<描述>报告.md`(参考 Section 1.5 的必含 sections)
4. **写 `<vN>/iter_<M>/decision.md`**(简版,给下次续接快速看):
   - Gate 结论(pass/edge/fail)
   - 下一步动作
   - 关联的 report 路径
5. **更新本文档的进度表**(下方 Section 7)
6. **更新 memory `project_current_run.md`**:同步当前进行到哪一版哪一次 iteration

**未完成上述任何一步**,不允许进入下一次 iteration 或下一版。

---

## 7. 探索阶段的资源约束

- **GPU**:2 张(HI-GRAM 后新增能力)
  - GPU6:主线(CodeLlama 保护),跑主要 vN
  - GPU0:副线(ablation-scan holder 保护),跑并行 iteration 或次要实验
  - **单卡跑时**(v1 代码未 ready 等):副线卡由占位者持续吃住不放
- **API**:全阶段 API 成本 upper bound **~$50-80(¥350-560)**
  - v2 pilot ~$5
  - v4 pilot ~$15
  - Iteration 3-5 次 ~$50
- **时间**:6-8 周

**R²-v2 更正（2026-08-19）**：新方法不调用 LLM API；allocator 小模型预计 CPU 可运行或 GPU 增量 `<2 GiB`。旧 v2/v4 的 API 预算不再是当前待执行成本。Sports 的 v0/resolver/cache 若缺失，须在 Stage S PASS 后单独测峰值和申请资源。

**旧路线历史预算（不执行）**：原计划曾考虑用本地 Qwen 2.5 32B 替代 v2/v4 API；R²-v2 已明确不使用该组件。

---

## 8. 关键 TODO(HI-GRAM 收尾前可零成本准备)

**High priority(不占 GPU 可先做)**:
- [ ] 实现 `r2_v2_budgeted_slate_allocator.py` 与 target-leakage / fold-isolation / Sports guard 单测
- [ ] 冻结 R²-v2 feature schema、5-fold salt、三动作构造与 config/code SHA256
- [ ] 完成 Toys/Beauty cache alignment preflight（只核完整性，不输出 efficacy）

**以下为旧路线历史 TODO（已 superseded，不执行）**:

- [ ] 写 `experiment/phase13/protocol/cold_split.py`(纯 CPU)
- [ ] 设计 v1 的 MLP decoder 代码骨架
- [ ] 设计 v2 的 first-pass LLM prompt template(基础版)
- [ ] 搭 LLM API cache infrastructure(SQLite)
- [ ] 准备 warm items 的 (text, id) pool 作为 few-shot 素材
- [ ] 写 `experiment/phase13/tests/` 单元测试骨架(preflight 必过项)

**历史 Medium priority（不执行）**:
- [ ] 写 `experiment/phase13/run_phase13_explore.sh`(**完整 protocol runner**,参照 `run_phase12_hi_gram.sh`,包含 CodeLlama 前后占位 + 30G lease + status.json + exit trap)
- [ ] 写 `metrics_summary.json` 自动生成脚本
- [ ] 写 report 模板(`report/第十三阶段/GRAM_第十三阶段_report_template.md`)

**历史 Low priority（不执行）**:
- [ ] Hierarchical alignment loss 实现
- [ ] Multi-perspective + reflection prompt engineering
- [ ] Dual-path decoder + gate learning

---

## 9. 进度追踪(每次 iteration 后更新)

| 版本 | Iteration | 状态 | Cold NDCG@10 | Δ vs prev | Gate | Report 路径 | 日期 |
|---|---|---|---|---|---|---|---|
| v0_toys | — | ✅ done | 0.00305 | (baseline) | pass | v0_toys_vanilla-baseline | 2026-08-09 |
| v0_beauty | — | ✅ done | 0.00179 | (baseline) | pass | v0_beauty_vanilla-baseline | 2026-08-10 |
| v1_toys（旧） | collision-unsafe | ⚫ invalidated | 0.00872 raw | +186% raw vs v0 | **原 PASS 作废** | v1_toys_MLP-semantic-bridge | 2026-08-11；2026-08-17 作废 |
| v1_beauty（旧） | collision-unsafe | ⚫ invalidated | 0.00418 raw | +133% raw vs v0 | **原 PASS 作废** | v1_beauty_MLP-semantic-bridge | 2026-08-12；2026-08-17 作废 |
| v2_toys（旧） | iter_1 | ⚫ diagnostic-only | 0.00453 | −48% vs 旧 v1 | **正式 Gate 失效** | v2_toys_LLM-prior_gate-FAIL + v2_toys_失败根因诊断 | 2026-08-12；2026-08-17 重标 |
| v2_toys（旧） | iter_2 | ⚫ diagnostic-only | 0.00525 | −39.8% vs 旧 v1 | **正式 Gate 失效** | v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL | 2026-08-14；2026-08-17 重标 |
| v2_beauty（旧） | iter_2 | ⚫ diagnostic-only | 0.00236 | −43.6% vs 旧 v1 | **正式 Gate 失效** | 同上（双域合并报告） | 2026-08-14；2026-08-17 重标 |
| v3（旧） | iter_1 快筛 | ⚫ diagnostic-only | 未跑 GRAM | 依赖旧口径 | **正式 Gate 失效** | v3_iter1_hierarchical-alignment_中期报告与交接 | 2026-08-14；2026-08-17 重标 |
| v1_collision_safe_toys | iter_1 | ✅ scientific done / resource degraded | **0.00154617** | **−49.31% vs v0** | **FAIL** | v1_collision-safe_双域重跑验证报告 | 2026-08-17 |
| v1_collision_safe_beauty | iter_1 | ✅ scientific done / resource degraded | **0.00161338** | **−10.05% vs v0**（cold H@10 16→16 未变） | **FAIL** | 本文档 v1 Beauty 结果节；双域报告待补全 | 2026-08-18 06:06 完成，08-18 补记 |
| v1_collision_safe_e5_toys | iter_2 candidate pre-GRAM | ⛔ screened out | —（未跑 GRAM） | MLP val −3.64%；cold exact path −12.08% vs MiniLM | **SCREENED_OUT_BEFORE_GRAM** | artifacts/.../v1_collision_safe_e5_toys_mlp400/screen_summary.json | 2026-08-17 |
| v1_collision_safe_minilm_residual_toys | iter_2 candidate pre-GRAM | ⛔ screened out | —（未跑 GRAM） | val −0.08%；cold exact +4.83%，但 macro/prefix@2 未过 | **FAIL_STOP_RESIDUAL** | artifacts/.../v1_collision_safe_minilm_residual_toys_screen/screen_summary.json | 2026-08-17 |
| v1_collision_safe_minilm_regularized_residual_toys | iter_2 candidate pre-GRAM | ⛔ screened out | —（未跑 GRAM） | HScore +0.216%；仅多 1 个 warm prefix@2 样本 | **FAIL_WARM_GATE** | artifacts/.../v1_collision_safe_minilm_regularized_residual_toys_screen/screen_summary.json | 2026-08-17 |
| v1_collision_safe_bge_toys | encoder candidate pre-GRAM | ⛔ stopped by frozen collision Gate | —（未跑 GRAM） | semantic 全 PASS；raw duplicates +9.17% | **FAIL_STOP_BGE** | artifacts/.../v1_collision_safe_bge_toys_screen/screen_summary.json | 2026-08-17 |
| smoke_v1_collision_safe_bge_toys | BGE suffix-burden downstream diagnostic | ✅ scientific completed / post-run shell anomaly | **cold NDCG@10 0.011111** | cold@10 与 MiniLM 完全相同；cold@20 多 1 hit（次要） | **INCONCLUSIVE_TIE** | artifacts/.../smoke_v1_collision_safe_bge_toys/smoke_summary.json | 2026-08-17 18:05 完成；holder exact 30000 |
| v1_iter2_bge_capacity_assignment_toys_p0 | iter2 assignment candidate pre-GRAM | ⛔ screened out | —（未跑 GRAM） | collision 归零；prefix@4 −13.10%、exact −33.82% vs MiniLM | **FAIL_STOP_CAPACITY_ASSIGNMENT** | artifacts/.../v1_iter2_bge_capacity_assignment_toys_p0/screen_summary.json | 2026-08-17 19:00 完成 |
| v1-R²_toys_p3 | validation 5-fold OOF | ✅ exploratory pass | 0.006125 | +121.77% vs v0 | **PASS_TO_R2_FRESH_MEDIUM_SMOKE_DISCUSSION** | artifacts/.../v1_r2_toys_p3_confidence_abstention/summary.json | 2026-08-18 |
| v1-R²_toys_p3_fresh_medium | test hash sample 1,000 one-shot | ⛔ confirmation stopped | 0.008415 | +46.39% vs v0；warm −3.52% | **FAIL_STOP_R2_FRESH_MEDIUM_SMOKE** | artifacts/.../v1_r2_toys_p3_fresh_medium_smoke/summary.json | 2026-08-18 |
| v1-R²_toys_p4 | validation outer-5fold OOF | ✅ exploratory pass | 0.005357 | +93.97% vs v0；warm −0.88% | **PASS_TO_R2_P4_FRESH_DISJOINT_CONFIRMATION_DISCUSSION** | artifacts/.../v1_r2_toys_p4_counterfactual_slot_router/summary.json | 2026-08-18 |
| v1-R²_toys_p4_disjoint | test hash ranks 1001–2000 one-shot | ⛔ confirmation stopped | 0.005143 | +33.04% vs v0；warm −1.67% | **FAIL_STOP_R2_P4_DISJOINT_CONFIRMATION** | artifacts/.../v1_r2_toys_p4_disjoint_confirmation/summary.json | 2026-08-18 |
| v1-R²_toys_p5_set | validation pseudo-cold setwise selector | ⛔ selector transfer stopped | 0.004133 | +49.65% vs v0；warm −0.99%；pool recall 6.019× | **FAIL_STOP_R2_P5_SET** | artifacts/.../v1_r2_toys_p5_setwise_selector/summary.json | 2026-08-18 |
| v1-R²_toys_p6_portfolio | validation outer-5fold OOF | ⛔ near-edge / Gate stopped | 0.005416 | +96.10% vs v0；warm −0.44%；P4 cold +1.10% | **FAIL_STOP_R2_P6** | artifacts/.../v1_r2_toys_p6_candidate_portfolio/summary.json | 2026-08-18 |
| v1-R²_toys_p7_robust_slate | validation outer-5fold OOF | ⛔ 退化为全 abstain | 0.002762（=v0） | 0.00%；coverage=0；14 项 Gate 仅过 4 项 | **FAIL_STOP_R2_P7** | artifacts/.../v1_r2_toys_p7_robust_slate/summary.json | 2026-08-18 16:22，08-18 补记 |
| （参照）unconditional_portfolio@3 | validation，无 gating | 📌 新口径下最强激进端 | **0.011918** | cold H@50 **3.84×**；overall **+7.35%**（CI 下界 +0.0015）；warm 保留 93.44% | 新口径 **overall PASS** | artifacts/.../v1_r2_toys_pareto_recompute/ | 2026-08-18 重算 |
| （参照）unconditional_portfolio@2 | validation，无 gating | 📌 新口径下建议主推 | 0.008724 | cold H@50 **2.89×**；overall +4.96%（CI 下界 +0.0009）；warm 保留 95.91% | 新口径 **overall PASS** | 同上 | 2026-08-18 重算 |
| v1_r2_toys_pareto_recompute | 新口径重算 + paired bootstrap | ✅ done（纯 CPU，未读 test） | 见 Section 3.5.4 | 四点 Pareto 前沿；三方案 overall CI 下界均 >0 | 口径 (A)(B)(C) 确认 | experiment/phase13/protocol/pareto_recompute.py | 2026-08-18 |
| v1_r2_beauty_b1_portfolio | Beauty validation 跨域确认 | ✅ completed | **0.009231** | overall **+4.15%**；cold H@50 **2.49×**；warm **−5.26%** | **PASS_TO_PUBLICATION_PREPARATION** | `GRAM_第十三阶段_v1-R2_Beauty-B1_无条件portfolio跨域确认报告.md` | 2026-08-18 18:59 |
| 旧 v4 | iter_1 | ⚫ superseded / not executing | — | — | — | 历史方案见本文旧 v4 节 | 2026-08-19 重标 |
| 旧 v5 | iter_1 | ⚫ superseded / not executing | — | — | — | 历史方案见本文旧 v5 节 | 2026-08-19 重标 |
| R²-v2 CBSA | preregistration | 📝 frozen / not started | — | 主对照=`portfolio@2` | **R2_V2_PREREGISTERED_NOT_STARTED** | 本文 R²-v2 节 | 2026-08-19 |

**每次 iteration 完成后必须回来更新此表**(新增行 or 更新对应行)。

**当前执行顺序（2026-08-19 修订）**：原 collision-safe v1 双域 FAIL；P1–P7 learned gating 未获独立确认；冻结的 `domain-local resolver + unconditional portfolio@2` 已在 Beauty B1 通过跨域主 Gate（overall +4.15%、cold H@50 2.49×、warm −5.26%）。用户已选择方法论文路线，当前 transition 为 **`R2_V2_PREREGISTERED_NOT_STARTED`**：先实现并审计单一可训练 R²-v2 CBSA，再做 Toys+Beauty source OOF Gate；只有相对 `portfolio@2` 的 overall/warm 改善和 cold 非劣同时成立，才讨论 Sports 一次性确认。**不启动旧 v2–v5、P8、Toys/Beauty test 或 publication 全矩阵，不在 Sports 上开发**。GPU5 holder 的旧 degraded 状态与 B1 科学结论分开处理。

---

## 10. 关联文档

- `GRAM_第十三阶段_CANARD主线设计v0.1.md` — 探索通过后进入的完稿 plan
- `../../experiment/phase13/README.md` — 目录结构说明
- `../第十二阶段/` — HI-GRAM plan(参考格式)
- `../../experiment/phase12/run_phase12_hi_gram.sh` — **完整 protocol runner 模板**(必须参照)
- `../../report/第九阶段/` `../../report/第十一阶段/` — report 命名和格式参考
- Memory:
  - `feedback_experiment_protocol.md` — CodeLlama + 30G lease + no auto retry 规则来源
  - `project_current_run.md` — 当前进度
  - `feedback_experiment_mode.md` — 探索模式提醒
  - `user_constraints.md` — 服务器资源紧张、必须保护 GPU 的原因

---

## 11. Notes for Future Sessions

**续接时必做(顺序)**:
0. **先读 R²-v2 新节、Beauty B1 报告与 Section 3.5**——B1 已 PASS；用户已选择方法论文路线，当前任务是实现/审计 R²-v2 的 source Gate，不是恢复 P8 或旧 v2–v5
1. 读本文档(尤其 Section 9 进度追踪表)
2. 优先读 `report/第十三阶段/GRAM_第十三阶段_v1_collision-safe_双域重跑验证报告.md`；Beauty 结果已于 2026-08-18 补记在本文档 v1 Beauty 结果节，该报告仍待补全
3. `nvidia-smi -i <保护卡>` 确认占位者还在（**注意：GPU5 的 ablation-scan holder 在 Beauty 运行后处于 degraded 状态 `degraded_scan_18263mib_on_gpu5`，需先确认/恢复**）
4. 看当前 iteration 状态 `cat artifacts/phase13/explore/<current_v>/iter_<M>/status.json`
5. 参考 memory `project_current_run.md`

**证据使用禁令**：不得再把旧 v1 raw PASS、旧 v2 FAIL 或旧 v3 快筛当作 collision-safe 主线的正式 Gate；它们只可作为历史/诊断材料。

**R²-v2 防火墙**：Toys/Beauty 已降级为 source/development；Sports 是唯一独立确认域。在 Stage S PASS 且用户再次确认前，`sports_read` 必须保持 false。本计划的写入不等于实验启动授权。

**旧 v0–v5 遇到 gate 失败时（历史规则；R²-v2 不适用）**:
- 不立刻放弃 —— 3 次 iteration 上限
- 每次 iteration 前写 `iter_N/hypothesis.md`(这次要改什么、预期效果、如何验证)
- iteration 后写 report(Section 1.5 规则)
- 3 次后确定放弃时,写 `<vN>/failed.md` + 对应 report,进 fallback 或 Plan Z

**GPU 保护失效应急**:
- 如果发现占位者从保护卡消失(被其他人挤掉)
- 立即用对应工具重启:GPU6 → `tools/run_codellama.sh start 6`;副线卡 → `tools/gram_ablation_scan.sh start <gpu>`
- 如果 30G 已经排不到,先跟服务器管理员/其他用户协调
- **不要**在无保护的卡上启动 GRAM 训练(可能中途被 OOM kill,损失整次 iteration)
