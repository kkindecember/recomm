# Phase 13 Exploratory: CANARD MVP 逐步验证 & 迭代

**创建日期**:2026-08-07
**状态**:pending(HI-GRAM 收尾后启动)
**目的**:用最快速、最低成本的方式验证 CANARD 的核心假设是否成立,失败就迭代改进,不做方向漂移
**预计工期**:6-8 周(每一步顺利 4-5 周,考虑常态 iteration 6-8 周)

---

## ⚠️ 定位说明

**本文档不是"试多个方向"的撒网 plan**,而是**验证 CANARD 一个方向 + 如何 iteratively 改进** 的 MVP-style plan。

CANARD 的核心假设:
> "Text-based signal(sentence-BERT + LLM prior + hierarchical alignment)能显著挽救 GRAM 的 cold-start item hierarchical id 分配问题。"

如果这个核心假设**在 v1-v2 就完全崩掉**,才考虑启动本文档末尾的 **Plan Z fallback**(换方向)。**优先假设 CANARD 会 work,只是需要迭代**。

Exploratory 阶段完成后,进入 `PLAN_PUBLICATION.md` 做全矩阵实验和写论文。

---

## 1. 探索原则

### 1.1 快速迭代

- **单数据集**:Beauty(遗产最全,pilot 最快)
- **单 cold ratio**:η = 50%(足够暴露 cold 问题,不需要 20% 弱刺激)
- **单 seed**:12345(不追求 seed 方差稳定性,那是 publication 阶段的事)
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

### 1.4 iteration 上限

每一步 vN 如果 gate 未通过,最多 3 次调整:
- 第 1 次:参数/超参调整(learning rate, loss weights, prompt 措辞)
- 第 2 次:实现细节调整(换 encoder,换 LLM,换 layer arch)
- 第 3 次:降级尝试(简化设计,去掉部分复杂度)

3 次都失败 → **该组件砍掉**,报告 negative,继续下一步(下一步跳过依赖该组件的部分)。

### 1.5 Report 强制规则(硬性)

**每一次尝试完成后**(不论成功/失败/边缘),都必须写一份 report 到 `report/第十三阶段/`,命名规范参考 phase9/11:

```
report/第十三阶段/GRAM_第十三阶段_v<N>_iter<M>_<描述>报告.md
```

例如:
- `GRAM_第十三阶段_v0_vanilla-baseline_cold-setting验证报告.md`
- `GRAM_第十三阶段_v1_iter1_MinimumSemanticBridge结果报告.md`
- `GRAM_第十三阶段_v1_iter2_bge-encoder换用结果报告.md`
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
1. 换 encoder:sentence-BERT → BGE-large → E5
2. 加深 MLP:1 层 → 2 层 with residual
3. 换预训练目标:考虑用 SASRec pretraining 得到的 item embedding(而不是 pure text)

**如果 3 次都失败**:启动 Plan Z(text 信号对 hierarchical id 家族根本不 work,换 category hard constraint 或 retrieval-only)

---

### v2:+ LLM Prior(单次 first-pass,无 reflection)

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

### v2_iter1 结果(2026-08-12)❌ FAIL

**执行结果**:cold ndcg@10 **-48% vs v1**(gate 严重不达标)

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

### v2_iter2 结果(2026-08-14)❌ FAIL — **v2 组件 abandoned**

**执行结果**:cold ndcg@10 双域一致回退 —— Beauty **-43.6%** vs v1、Toys **-39.8%** vs v1(iter1 为 -48%)

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

**执行缺陷**:DeepSeek API 余额耗尽,Beauty 2871 次 / Toys 1942 次调用失败,失败样本被写成 `<unk>` + confidence **1.0**(伪装成正常回答),导致 47.5% / 32.6% 的 warm item 完全无 KL 监督。**已于 2026-08-14 补齐全部失败调用并重训 MLP 复核**:完整覆盖下 Beauty **0.2505** / Toys **0.3889**,均仍低于 v1 的 0.2630 / 0.4060,且 **Beauty 补齐后反而更差**(0.2531 → 0.2505)—— 误判假设排除,FAIL 结论成立。见 `artifacts/phase13/explore/v2_verify/CONCLUSION.md`。

**决策**:按 gate 条文命中"❌ 失败 → 直接跳到 v3,标记 v2 abandoned"。**不做 iter3** —— iter3 的三个候选(λ→0.1 / 换 GPT-4o mini / 方向 C)都绕不开上述机制层错位。

**给 v3 的提示**:浅层语义信号可靠(L1 44-60%)、深层不可靠。v3 的 hierarchical alignment 若按层加权(浅层高、深层低或为 0),可能正好避开 v2 的坑。

**Report**:`report/第十三阶段/GRAM_第十三阶段_v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL报告.md`

---

### v3:+ Hierarchical Contrastive Alignment Loss

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

### v4:+ Multi-Perspective Reasoning + Self-Reflection

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

### v5:+ Uncertainty-Aware Dual-Path Decoding

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

### v0 通过但 v1-v2 就挂 → 启动 Plan Z fallback

- 说明 semantic bridge 从根本上不 work
- 进入本文档 Section 5 的备选方向

---

## 4. 探索时间表(6-8 周)

| Week | 版本 | 主要工作 | 累积时间 |
|---|---|---|---|
| 1 | v0 | Cold protocol + vanilla GRAM Beauty η=50%,gate v0 | 3 天 |
| 1-2 | v1 | Sentence-BERT + 1 层 MLP,gate v1(含 iteration) | +5 天 |
| 2-3 | v2 | + LLM prior (single first-pass),gate v2 | +5 天 |
| 3-4 | v3 | + Hierarchical alignment loss,gate v3 | +7 天 |
| 5 | v4 | + Multi-perspective + reflection,gate v4 | +5 天 |
| 6 | v5 | + Uncertainty dual-path,gate v5 | +5 天 |
| 7-8 | Buffer | 补 iteration + 决策进 publication or Plan Z |  |

**顺利总计**:5-6 周
**含常态 iteration**:6-8 周

---

## 5. Plan Z:CANARD 完全失败后的 fallback 方向

**触发条件**:v0 通过(cold setting 有效)但 v1 或 v1-v2 iteration 3 次都失败(text-based 方向根本不 work)。

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
└── v5_dual_path/
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

**极端省钱预算**:如果 API 也想省,v2/v4 可以先用本地部署 Qwen 2.5 32B(quantized),API 上再验证。但探索阶段推荐直接用 API 以加速。

---

## 8. 关键 TODO(HI-GRAM 收尾前可零成本准备)

**High priority(不占 GPU 可先做)**:
- [ ] 写 `experiment/phase13/protocol/cold_split.py`(纯 CPU)
- [ ] 设计 v1 的 MLP decoder 代码骨架
- [ ] 设计 v2 的 first-pass LLM prompt template(基础版)
- [ ] 搭 LLM API cache infrastructure(SQLite)
- [ ] 准备 warm items 的 (text, id) pool 作为 few-shot 素材
- [ ] 写 `experiment/phase13/tests/` 单元测试骨架(preflight 必过项)

**Medium priority(v0 启动时才要)**:
- [ ] 写 `experiment/phase13/run_phase13_explore.sh`(**完整 protocol runner**,参照 `run_phase12_hi_gram.sh`,包含 CodeLlama 前后占位 + 30G lease + status.json + exit trap)
- [ ] 写 `metrics_summary.json` 自动生成脚本
- [ ] 写 report 模板(`report/第十三阶段/GRAM_第十三阶段_report_template.md`)

**Low priority(v3+ 才要)**:
- [ ] Hierarchical alignment loss 实现
- [ ] Multi-perspective + reflection prompt engineering
- [ ] Dual-path decoder + gate learning

---

## 9. 进度追踪(每次 iteration 后更新)

| 版本 | Iteration | 状态 | Cold NDCG@10 | Δ vs prev | Gate | Report 路径 | 日期 |
|---|---|---|---|---|---|---|---|
| v0_toys | — | ✅ done | 0.00305 | (baseline) | pass | v0_toys_vanilla-baseline | 2026-08-09 |
| v0_beauty | — | ✅ done | 0.00179 | (baseline) | pass | v0_beauty_vanilla-baseline | 2026-08-10 |
| v1_toys | iter_1 | ✅ done | 0.00872 | **+186%** vs v0 | **PASS** | v1_toys_MLP-semantic-bridge | 2026-08-11 |
| v1_beauty | iter_1 | ✅ done | 0.00418 | **+133%** vs v0 | **PASS** | v1_beauty_MLP-semantic-bridge | 2026-08-12 |
| v2_toys | iter_1 | ❌ FAIL | 0.00453 | **-48%** vs v1 | **FAIL** | v2_toys_LLM-prior_gate-FAIL + v2_toys_失败根因诊断 | 2026-08-12 |
| v2_toys | iter_2 | ❌ FAIL | 0.00525 | **-39.8%** vs v1 | **FAIL** | v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL | 2026-08-14 |
| v2_beauty | iter_2 | ❌ FAIL | 0.00236 | **-43.6%** vs v1 | **FAIL** | 同上(双域合并报告) | 2026-08-14 |
| v3 | iter_1 | ❌ 快筛未通过 | (未跑 GRAM) | 均未过 v1/对照组 | 快筛未通过 | v3_iter1_hierarchical-alignment_中期报告与交接 | 2026-08-14 |
| v4 | iter_1 | 未启动 | — | — | — | — | — |
| v5 | iter_1 | 未启动 | — | — | — | — | — |

**每次 iteration 完成后必须回来更新此表**(新增行 or 更新对应行)。

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
1. 读本文档(尤其 Section 9 进度追踪表)
2. 读**最新一份 report**(`ls -lt report/第十三阶段/*.md | head -1`)
3. `nvidia-smi -i <保护卡>` 确认 CodeLlama 还在(如果掉了,先恢复再动)
4. 看当前 iteration 状态 `cat artifacts/phase13/explore/<current_v>/iter_<M>/status.json`
5. 参考 memory `project_current_run.md`

**遇到 gate 失败时**:
- 不立刻放弃 —— 3 次 iteration 上限
- 每次 iteration 前写 `iter_N/hypothesis.md`(这次要改什么、预期效果、如何验证)
- iteration 后写 report(Section 1.5 规则)
- 3 次后确定放弃时,写 `<vN>/failed.md` + 对应 report,进 fallback 或 Plan Z

**GPU 保护失效应急**:
- 如果发现占位者从保护卡消失(被其他人挤掉)
- 立即用对应工具重启:GPU6 → `tools/run_codellama.sh start 6`;副线卡 → `tools/gram_ablation_scan.sh start <gpu>`
- 如果 30G 已经排不到,先跟服务器管理员/其他用户协调
- **不要**在无保护的卡上启动 GRAM 训练(可能中途被 OOM kill,损失整次 iteration)
