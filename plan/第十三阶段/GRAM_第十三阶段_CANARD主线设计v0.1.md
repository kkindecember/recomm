# Phase 13: Cold-Start Item Recommendation for Hierarchical-ID Generative Recommenders

**创建日期**:2026-08-07
**状态**:planning(等待 `PLAN_EXPLORATORY.md` 里 CANARD v0-v5 验证通过后启动)
**主目标会议**:RecSys 2026(CCF-B),备选 CIKM 2026 / WSDM 2027
**预计工期**:8-10 周完稿实验 + 论文(**不含 exploratory phase 的 6-8 周**;两阶段合计 ~4 月)

---

## ⚠️ 使用说明(必读)

**本文档是完稿级 plan**,前提是 exploratory phase(见 `PLAN_EXPLORATORY.md`)已经验证 CANARD 各层组件有效。

本文档默认走 CANARD 完整方案(v5 = LLM stage + Semantic Bridge + Dual-Path Decoding)。**如果 exploratory phase 发现某些层不 work,写作时需要:**
- 更新 Section 0.2 的贡献列表(去掉被砍层)
- 更新 Section 3 的 method design(只写 work 的层)
- 主体骨架(motivation / related work / cold-start protocol / baseline / experiment protocol / risks)保持不变

**如果 CANARD 完全失败**(v1-v2 就挂),启动 `PLAN_EXPLORATORY.md` 里的 Plan Z fallback,那时需要重写本文档。

---

## 0. Executive Summary

### 论文一句话骨架

> "Hierarchical-ID-based generative recommenders (GRAM 系) 在 cold-start items 上完全失败(tail Recall@10 ≈ 0)。我们提出 category-anchored framework,通过 LLM 结构化推理 + 层级语义对齐 + 不确定性引导的双路解码,把 cold Recall@10 从 0 拉到 X,不牺牲 warm 性能。"

### 5 个贡献

| # | 贡献 | 层级 |
|---|---|---|
| 1 | Cold-start protocol for hierarchical-id GenRec | Setting |
| 2 | Diagnostic on GRAM cold-collapse mechanism | 诊断 |
| 3 | LLM stage: multi-perspective reasoning + cluster-inspection reflection + attribute enrichment | Method |
| 4 | Semantic Bridge with hierarchical contrastive alignment + LLM prior + category prior | Method |
| 5 | Uncertainty-aware dual-path decoding (LLM + MLP fused uncertainty) | Method |

### 主线叙事

**Category structure is a first-class citizen throughout the method.** GRAM 训练时聚出的 hierarchical id 与商品自带 category 层级未必对齐,cold items 更暴露这一差距。我们让 category 结构渗透 method 三层:LLM 用 category 组织 multi-perspective 推理;Semantic Bridge 用 category 作 alignment anchor;Decoding 用 LLM 对 category 的 confidence 做 uncertainty routing。

### Method 名称

临时代号:`CANARD` (Category-ANchored Reasoning and Decoding for cold-start items),写作阶段可改。

---

## 1. Motivation

### 1.1 生成式推荐的 cold-start 现状

- Hierarchical-ID / Semantic-ID 生成式推荐器(GRAM/TIGER/CoFiRec/HLLM)成为 24-26 年的主流范式
- 但它们的 item id 在**训练时冻结**(聚类分配 or RQ-VAE 量化)
- Cold items(训练集未出现)只能:随机 id / 最近邻 fallback,导致 beam search Trie 不覆盖或分配错位
- 商业系统每日上架新品,cold-item 是**真实工业痛点**

### 1.2 GRAM 的 cold-start 崩塌(from P9-P10)

**证据(复用现有诊断数据)**:
- P9-1 diagnostics:head item(top 5017)Recall@10 = 3.63%,tail item(bottom 5160)Recall@10 = **0.00%**
- P11 beam coverage:beam=50 时 target coverage 仅 47-55%(Beauty/Toys),beam=200 也只 64-73%
- P12 HI-GRAM 教训:post-hoc 修改 hierarchical id 会破坏 encoder 位置信息 + Trie 约束 → 架构改动不可行,必须从 protocol/calibration 层入手

**因此** cold-start 是一个既有真实价值,又能被现有诊断素材充分支撑的 setting。

---

## 2. Related Work (2024-2026)

### 2.1 生成式推荐的 cold-start 工作

| 论文 | 会议 | 方法 | 与本文差异 |
|---|---|---|---|
| CoFiRec (2025) | arxiv 2511 | Coarse-to-fine tokenizer,cold 定义为 bottom 2% frequency | 改 tokenizer 结构;我们不动 tokenizer,加 alignment head |
| GenRecEdit (2025) | arxiv 2603 | Model editing 处理 cold items | 需修改 GRAM 参数;我们只加外挂 MLP + LLM stage |
| TIGER-SID/Scorer/Edge (2025) | arxiv 2607 | 三个 architecture variants(改 SID space / scorer / cross-attn) | 不同 id 家族(RQ-VAE semantic id vs hierarchical clustering id) |
| SpecGR (2024) | — | Draft-verify inductive generation | 二值 verify;我们 uncertainty-aware gated fusion |
| Item-centric Exploration (RecSys 2025) | RecSys 2025 | Exploration-based cold-start | 不用 exploration 假设 |

### 2.2 长尾/LLM 增强

| 论文 | 会议 | 相关性 |
|---|---|---|
| LLM-ESR (Liu et al., NeurIPS 2024) | NeurIPS 2024 | LLM 增强 long-tail seq rec,focus user 而非 item |
| MI4Rec (2025) | CIKM 2025 | Cold-start with meta-item embeddings |
| Meta-Adaptive Network (2025) | CIKM 2025 | Warm-aware representation |
| Awakening Dormant Users (2025) | arxiv 2602 | Counterfactual for dormant users |

### 2.3 本文独特位置

- **首次**针对 hierarchical clustering id 家族(vs semantic id family)做 cold-start systematic study
- **首次**把 category structure 作为 method 主线贯穿三层
- **首次**在 GenRec 里用 cluster-inspection self-reflection
- **不改 backbone**(HI-GRAM 教训之后),用 plug-and-play 外挂
- 复用 P9/P10 现成的 error decomposition 素材,起点比同赛道对手快 4 周

---

## 3. Method Design

### 3.1 总览

```
[cold item text + categories]
        │
        ▼
┌─────────────────────────────────────────────┐
│  Layer 1: LLM Stage (DeepSeek V4 API)       │
│  ├─ Multi-perspective reasoning             │
│  │   (category / usage / attribute)         │
│  ├─ Cluster-inspection self-reflection      │
│  └─ Attribute enrichment                    │
│  → refined hierarchical id + confidence     │
│    + enriched text                          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Layer 2: Semantic Bridge (learned MLP)     │
│  Loss = L_sup                               │
│       + Σ_l λ_l · L_hierarchical_align_l    │
│       + λ_llm · L_llm_prior                 │
│       + λ_cat · L_category_prior            │
│  → refined id embedding + confidence        │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Layer 3: Uncertainty-Aware Dual-Path       │
│  fused_conf = f(LLM_conf, MLP_entropy)      │
│  ├─ high conf → Generative path (Trie)      │
│  ├─ low conf  → Retrieval path (text sim)   │
│  └─ mid conf  → Gated fusion                │
│  → final ranking                            │
└─────────────────────────────────────────────┘
```

### 3.2 Layer 1: LLM Stage(3-in-1)

**A. Multi-perspective reasoning**

每个 cold item 从 **3 个 category-anchored perspectives** 独立推理:
- **Category-driven**: "从 product category 层级看,这个 item 应该在哪个 hierarchical id path"
- **Usage-driven**: "从使用场景看,应该和哪些 warm items 同 cluster"
- **Attribute-driven**: "从材质/品牌/价格属性看,应该在哪"

每个 perspective 独立 first-pass,输出 predicted id + reasoning chain + confidence。

**B. Cluster-inspection self-reflection**

对每个 perspective 的 first-pass 输出,做 reflection:
- 输入 LLM:cold item + first-pass prediction + **实际属于该 cluster 的 warm items 样例(top-5)+ 邻近 cluster 的样例(top-5)**
- Prompt LLM:"这个 cluster 的实际内容和你的推理一致吗?邻近 cluster 是否更好?"
- 输出:refined prediction + revision reasoning + updated confidence

**C. Attribute enrichment**

LLM 同时生成 cold item 的 enriched attributes:
- Implicit attributes(usage scenario, target user, functional tags)
- Hypothetical queries(简短)
- 这些 enriched text 与原文本 concat,输入 Layer 2 的 embedding

**Fusion**:3 个 perspective 的 refined predictions + confidence 用 learned weights 加权得到 LLM 最终输出(weights 在 warm validation 上学)。

### 3.3 Layer 2: Semantic Bridge

**架构**:Layer-wise MLP decoder(hierarchical id 每层一个 MLP)
- 输入:sentence-BERT / BGE 编码的 semantic vector v(base text + LLM-enriched text)
- 输出:每层 cluster 分布 p(c_l | v)

**四 loss 训练**:

1. **L_sup**:cross-entropy against warm items 的 ground truth id
2. **L_hierarchical_align_l**(每层一个):InfoNCE contrastive
   - positive = 同层 cluster 的 item(细相似)
   - negative = 同 (l-1) 层但不同 l 层的 item(hard negative,层级 hard mining)
3. **L_llm_prior**:KL(MLP output ∥ LLM refined prediction distribution),让 MLP 靠近 LLM 推理结果
4. **L_category_prior**:让 text embedding 距离与 human category 层级距离一致(用商品自带 category 5 层作 anchor)

**总 loss**:L = L_sup + Σ_l λ_l · L_align_l + λ_llm · L_llm + λ_cat · L_cat

**关键 novelty**:category 从 raw metadata 升级为 explicit alignment anchor,补充 GRAM 训练时聚类的 label bias。

### 3.4 Layer 3: Uncertainty-Aware Dual-Path Decoding

**Uncertainty 融合**:
- MLP softmax entropy → H_mlp
- LLM 自认 confidence → C_llm(reflection 后的 confidence)
- fused_conf = σ(w1 · (1 - H_mlp) + w2 · C_llm + b),w1/w2/b 学出

**Routing**:
- fused_conf > τ_high(~0.75):**Generative path** —— MLP 生成的 id 加入 Trie,参与 GRAM 原有 beam search
- fused_conf < τ_low(~0.35):**Retrieval path** —— 用 enriched text embedding 做 cosine similarity,retrieve top-K warm items 直接推荐
- τ_low ≤ fused_conf ≤ τ_high:**Gated fusion** —— 两条 path 各出 ranking,用 fused_conf 加权融合

**τ_high / τ_low / w1 / w2** 在 warm validation set 上学(不用 cold set,避免泄漏)。

---

## 4. Cold-Start Protocol

### 4.1 Data split

- **Item-level split**:训练集移除 X% items(按 frequency 分层采样以保持分布)
- **η ∈ {0%, 20%, 50%, 80%}**:η=0 是标准 warm-only setting,作为对比基线
- Test set 保持不变,但每个 test target 打 warm/cold 标签
- 三个数据集独立做 split:Beauty / Toys / Sports

### 4.2 评测

- **Overall**:Recall@10, NDCG@10, MRR@10(全 test set)
- **Warm subset**:仅 warm target 上的指标
- **Cold subset**:仅 cold target 上的指标(**核心指标**)
- **Cold coverage**:beam search 输出里 cold items 占比
- **Warm regression**:cold protocol 相对 η=0 的 warm 性能变化(必须 ≤ 5% 退化)

### 4.3 复用 P0-P12 素材

| 复用产物 | 位置 | 用途 |
|---|---|---|
| per_user_oof.tsv | artifacts/phase10/cf1_c2_toys_pcrf_anchored/ | User-level 分析,history length 分层 |
| per_user_test.tsv | artifacts/phase9/cf0_b5_toys_pcrf_test_p2e/ | Warm test 结果对比 |
| hit10_transitions.tsv | artifacts/phase10/cf1_c2/ | 追踪 cold vs warm 命中状态转移 |
| admission_gate.json | artifacts/phase11/bw3_p1_admission_recovery/ | Beam admission threshold(可作 baseline)|
| fold_models.json (PCRF) | artifacts/phase10/cf1_c2/ | PCRF cold recalibration 消融(**negative baseline**)|

---

## 5. Baseline Suite

### 5.1 Trivial(自己写,1 天)

- **Random**:从 all items 随机取 top-k
- **Popularity**:训练集频次 top-k 全推
- **Recency**:近期热度(最近 N 天)

### 5.2 Content(自己写,2 天)

- **BM25**:text matching(user history text vs cold item text)
- **Sentence-BERT similarity**:user history embedding avg vs cold item embedding

### 5.3 Backbone-native GRAM variants(改 GRAM 数据加载,3 天)

- **GRAM-vanilla**:cold items 用 fallback cluster id(training-time 随机)
- **GRAM-random**:cold items 每次推理随机分配 cluster id
- **GRAM-nearest-text**:cold items 用 text embedding 最近 warm cluster
- **GRAM-mean-text**:cold items 用 text embedding 均值找 cluster

### 5.4 External SOTA(自己重写,~1 天/个)

- **DropoutNet (KDD 2017)**:训练时 drop feature 学 cold-robust rep(~50 行 PyTorch)
- **SpecGR-adapted**:draft-verify 适配到 GRAM

### 5.5 Negative baseline(证明 PCRF 在 cold 上不 work)

- **GRAM + PCRF cold recalibration**:复用 P10 的 fold_models.json,在 cold subset 重新 fit λ/β/γ
- **预期结果**:marginal 或 negative gain(证明 CF-based calibration 在 cold 上原理不合)
- **在论文里**:作为 "why we need semantic bridge instead of PCRF" 的论证

---

## 6. Sub-phase Roadmap

### 13a: Cold Protocol + Vanilla GRAM Diagnostic(Week 1-2)

**目标**:定义 cold-start protocol,跑 vanilla GRAM 4×η × 2 datasets

**Deliverables**:
- `experiment/phase13/protocol/cold_split.py`
- `artifacts/phase13/13a_diagnostic/{beauty,toys}_eta{0,20,50,80}/`
- Diagnostic report:cold vs warm NDCG@10 曲线

**决策 gate 13a**:
- 通过:vanilla GRAM 在 η=50% 时 NDCG@10 相对 η=0 下降 ≥ 20%(setting 有意义)
- 失败:换 setting(long-tail user 或 cross-domain)

### 13b: Baseline Suite(Week 3-4)

**目标**:实现所有 baselines

**Deliverables**:
- `experiment/phase13/baselines/{trivial,content,gram_variants,external_sota}.py`
- `artifacts/phase13/13b_baselines/` 各 baseline 在 4×η × 2 datasets 的结果
- 完整 baseline table 草稿

### 13c: Method Implementation + Pilot(Week 5-8)

**Week 5**:Layer 1 LLM stage
- DeepSeek V4 API 集成
- Prompt template 设计(3 perspective + reflection + enrichment)
- Prompt cache infrastructure(降低 API 成本)
- Pilot on Beauty η=50%

**Week 6**:Layer 2 Semantic Bridge
- Layer-wise MLP decoder
- 四 loss 训练 pipeline
- Pilot on Beauty η=50%

**Week 7**:Layer 3 Uncertainty routing
- Fused uncertainty 学习
- Gated fusion decoder
- 完整 pipeline 联调

**Week 8**:Pilot iteration
- Beauty η=50% 完整 pipeline vs vanilla GRAM
- 决策 gate 13c

**决策 gate 13c**:
- 通过:cold NDCG@10 相对 vanilla GRAM 提升 ≥ 30%,warm 退化 ≤ 3%
- 通过一半:讨论 v2 或降级
- 失败:降级为 short paper(setting + diagnostic + negative result),投 RecSys LBR

### 13d: Full Experiment Matrix + Ablation(Week 9-12)

**Week 9-10**:主表实验
- 3 datasets × 3 cold ratios × (all baselines + 完整方法)
- 单 seed(setting 论文常规)

**Week 11**:Ablation
- Layer 1 消融(w/o LLM / w/o reflection / w/o multi-perspective / w/o enrichment)
- Layer 2 消融(w/o category prior / w/o LLM prior / w/o alignment / flat alignment)
- Layer 3 消融(w/o dual-path / hard-gate vs uncertainty-gate)
- LLM 消融:DeepSeek V4 vs GPT-4o(small subsample)

**Week 12**:补 ablation 缺口 + PCRF negative baseline

**决策 gate 13d**:
- 通过:至少 2/3 数据集显示 cold 一致提升,主 ablation 有意义
- 失败:讨论是否延期投稿

### 13e: Writing(Week 13-15)

**Week 13-14**:论文初稿
- Section 1 Intro / Section 2 Related / Section 3 Protocol / Section 4 Diagnostic / Section 5 Method / Section 6 Experiments / Section 7 Discussion

**Week 15**:图表打磨 + 反复修改 + 投稿

---

## 7. Success Criteria & Decision Gates(总览)

| Gate | 时机 | 条件 | 失败预案 |
|---|---|---|---|
| 13a | Week 2 结束 | vanilla GRAM 在 η=50% 下 NDCG@10 下降 ≥ 20% | 换 long-tail user setting |
| 13c | Week 8 结束 | Method 相对 vanilla 在 cold 提升 ≥ 30%,warm 退化 ≤ 3% | 降级为 short paper / LBR |
| 13d | Week 12 结束 | ≥ 2/3 数据集一致提升,消融证据完整 | 补实验 or 延期 |

---

## 8. Timeline(14-15 周)

| Week | Sub-phase | 主要工作 |
|---|---|---|
| 1 | 13a | Cold protocol + data split |
| 2 | 13a | Vanilla GRAM 8 次训练,**gate 13a** |
| 3 | 13b | Trivial + content baselines |
| 4 | 13b | Backbone variants + 1 external SOTA |
| 5 | 13c | Layer 1 LLM stage + prompt design |
| 6 | 13c | Layer 2 Semantic Bridge |
| 7 | 13c | Layer 3 Uncertainty routing |
| 8 | 13c | 完整 pipeline pilot,**gate 13c** |
| 9 | 13d | 主表实验(3 datasets × 3 ratios) |
| 10 | 13d | 主表补齐 + LLM ablation(GPT-4o subsample) |
| 11 | 13d | Layer 1/2/3 消融 |
| 12 | 13d | PCRF negative baseline + 补缺口,**gate 13d** |
| 13 | 13e | 论文骨架 + Section 3-5 初稿 |
| 14 | 13e | 全文初稿 + 图表 |
| 15 | 13e | 修改 + 投稿 |

**Buffer**:第 16 周作为 slack(临时补实验、内审修改、被拒补做)

---

## 9. LLM Configuration

### 9.1 主用:DeepSeek V4 API

- **模型**:DeepSeek V4 (最新版本,2026 中期)
- **调用方式**:官方 API + prompt caching(few-shot examples cache hit 60-80%)
- **成本估算**:
  - 每 cold item:2 次调用(3 perspective × 1 first-pass + 3 × 1 reflection = 6 calls,但共享 few-shot 部分 cache)
  - Beauty η=50%:~6000 cold items × avg 2500 tokens per call ≈ 30M input + 8M output
  - 单次 Beauty full run:$3-5(cache 后)
  - 全实验:3 datasets × 3 ratios × 主 pipeline = **~$30-50(¥200-350)**

### 9.2 消融:GPT-4o(upper bound)

- **用途**:证明 LLM-agnostic,展示 stronger LLM 下的性能上界
- **样本**:仅 1 dataset × 1 cold ratio × 500 cold items(subsample)
- **成本**:~$10-30(¥70-200)

### 9.3 可选:本地 Llama 3.1 8B(reproducibility ablation)

- **用途**:证明 method 在开源 8B 模型下依然 work,加强 reproducibility 论证
- **部署**:vLLM on 一张 A6000(FP16 ~16GB)
- **成本**:¥0,占一张 A6000 半天
- **优先级**:低(时间允许再做)

### 9.4 Prompt cache infrastructure(必做)

- 所有 LLM 调用统一走 wrapper,cache 到本地(SQLite / JSON)
- 相同 prompt 直接命中缓存,不重复调用
- 支持 --no-cache flag 强制重跑

**总 API 成本上限:¥550**(主 + 消融 + 少量重试)

---

## 10. Resource Constraints & Experiment Protocol

### 10.1 继承 Phase 12 experiment protocol

**必须遵守**:
1. **CodeLlama 前后占位**(生产 protocol):启动前 `run_codellama.sh stop <gpu>`,结束后 `start <gpu>`
2. **30G GPU lease**:sidecar hold (total - peak) MiB,避免其他人挤占
3. **Tmux session 命名**:`gram_phase13_<sub>` (生产) 或 `gram_phase13_<sub>_light` (简化)
4. **Status.json 联络机制**:字段沿用 phase12 schema
5. **禁止自动重试**:任何非零退出 → 写 failed 状态 → 人工诊断
6. **永不读 test set**(除最终评测):status.json `test_read=false`
7. **CPU 单元测试**:训练启动前 preflight() 必过
8. **训练后 log grep**:检查 NaN / Traceback / OOM

### 10.2 完整 vs 简化 protocol

**完整 protocol**(生产训练,vanilla GRAM baseline / 主 method 训练):
- 参照 `run_phase12_hi_gram.sh` 模板
- 有 CodeLlama、有 30G lease、有 runner 监督
- 用于:所有正式 GRAM base 训练、所有 fold cross-validation

**简化 protocol**(pilot / diagnostic / LLM inference):
- 参照 `light_status_sidecar.sh`
- 无 CodeLlama、无 lease,只有 30s status sidecar
- 用于:LLM API 调用(无 GPU 训练)、baseline 快速迭代、diagnostic 分析

### 10.3 目录结构

```
experiment/phase13/
├── PLAN.md                              # 本文档
├── README.md                            # 快速上手
├── run_phase13_cold.sh                  # 完整 protocol runner(参考 phase12)
├── light_status_sidecar.sh              # symlink to phase12 版本
├── protocol/
│   ├── cold_split.py                    # 数据 split 脚本
│   ├── cold_eval.py                     # cold/warm 分组评测
│   └── frozen_config_template.json      # 每次实验冻结 config
├── baselines/
│   ├── trivial.py                       # random / popularity / recency
│   ├── content.py                       # BM25 / sentence-BERT
│   ├── gram_variants.py                 # 4 个 backbone-native variants
│   └── external_sota/
│       ├── dropoutnet.py                # 自己重写的 DropoutNet
│       └── specgr_adapted.py            # SpecGR 适配
├── method/
│   ├── layer1_llm_stage/
│   │   ├── prompts/
│   │   │   ├── first_pass_category.txt
│   │   │   ├── first_pass_usage.txt
│   │   │   ├── first_pass_attribute.txt
│   │   │   └── reflection.txt
│   │   ├── llm_api.py                   # DeepSeek + GPT-4o wrapper + cache
│   │   └── llm_stage.py                 # multi-perspective + reflection + enrichment
│   ├── layer2_semantic_bridge/
│   │   ├── mlp_decoder.py               # layer-wise MLP
│   │   ├── losses.py                    # 4 loss(sup + hier_align + llm_prior + cat_prior)
│   │   └── train.py                     # Semantic Bridge 训练脚本
│   └── layer3_dual_path/
│       ├── uncertainty.py               # fused uncertainty
│       ├── dual_path_decoder.py         # generative + retrieval + gate
│       └── inference.py                 # 完整 inference pipeline
├── ablation/
│   └── <sub>.py                         # 每个消融独立脚本
└── tests/
    └── test_<component>.py              # unit tests(preflight 必过)

artifacts/phase13/
├── 13a_diagnostic/
│   └── {beauty,toys,sports}_eta{0,20,50,80}/
├── 13b_baselines/
│   └── <baseline_name>/
├── 13c_method_pilot/
├── 13d_full_matrix/
│   ├── main_table/
│   └── ablation/
└── llm_cache/                           # LLM 调用缓存(sqlite)
```

### 10.4 命令模板

```bash
# 启动 13a diagnostic(完整 protocol)
bash experiment/phase13/run_phase13_cold.sh start beauty_eta50 6

# 启动 13c pilot(简化 protocol,LLM API 无 GPU 训练)
tmux new-session -d -s gram_phase13_cold_pilot_light \
  "cd /repo && python experiment/phase13/method/layer1_llm_stage/llm_stage.py \
   --dataset Beauty --eta 0.5 --llm deepseek-v4 \
   >> artifacts/phase13/13c_method_pilot/beauty_eta50/run.log 2>&1"
bash experiment/phase12/light_status_sidecar.sh 13c_pilot_beauty 0 \
  gram_phase13_cold_pilot_light artifacts/phase13/13c_method_pilot/beauty_eta50

# 查状态
bash experiment/phase13/run_phase13_cold.sh status beauty_eta50
cat artifacts/phase13/<sub>/status.json | python3 -m json.tool
```

---

## 11. Risks & Contingencies

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 13a gate 不过(vanilla GRAM cold 下降 < 20%) | 低 | 高 | 换 setting:long-tail user(遗产 4/5 匹配) |
| 13c fix 只 marginal 提升 | 中 | 高 | 降级为 short paper(setting + diagnostic + negative result),投 RecSys LBR / SIGIR short |
| Sports 数据集跑不通 baseline | 中 | 低 | 只用 Beauty + Toys,论文 discussion 说明 leaves Sports as future |
| LLM API rate limit / 服务不稳 | 中 | 中 | Prompt cache + retry with exponential backoff;备用 Kimi/Qwen API |
| Semantic Bridge MLP 训不好 | 中 | 高 | Fallback 到 dual-path retrieval-only 方法,论文重点转 "text alone is insufficient, we need X" |
| Trie 动态扩展破坏 beam search | 中 | 中 | 离线预扩展 Trie(把 cold id 全部预注册),不做真正 online |
| GPU 资源冲突(其他人挤占) | 高 | 中 | 优先跑 CodeLlama-protected GPU,简化 protocol 只在真正空闲卡跑 |
| 论文 12 周赶不完 | 中 | 中 | 用第 16 周 buffer,或者放弃 Sports / 减少 ablation |

---

## 12. Immediate TODOs(HI-GRAM 收尾后立即启动)

**High priority**(Phase 12 结束前可以先做,零 GPU 成本):
- [ ] 写 `protocol/cold_split.py`(纯 CPU)
- [ ] 设计 3 个 first-pass prompt template + 1 个 reflection prompt
- [ ] 搭 LLM API cache infrastructure(SQLite)
- [ ] 收集 warm items 的 (text, hierarchical id) pool 作为 few-shot 素材
- [ ] 写 baseline 里的 trivial + content(纯 CPU)

**Medium priority**(Week 1-4):
- [ ] 写 `run_phase13_cold.sh` runner(参考 phase12)
- [ ] 跑 13a diagnostic
- [ ] 实现 baseline 阵仗

**Low priority**(Week 5+):
- [ ] 实现 Layer 1/2/3
- [ ] 部署本地 Llama 3.1 8B(可选)

---

## 13. Notes for Future Sessions

**续接时先做**:
1. 读本文档
2. 检查 `artifacts/phase13/` 是否有进行中实验,`cat status.json` 查状态
3. 检查 `experiment/phase13/tests/` 是否有失败测试
4. 参考 memory 里 `feedback_experiment_protocol.md` 沿用规则

**重要**:本 plan 是**探索模式**下的规划,不是投稿模式。允许中途调整(v2/v3 方法方向、换 setting),但每次调整需要更新本文档 + 记录决策理由。

**关联 memory**:
- `project_current_run.md` — HI-GRAM 收尾状态
- `feedback_experiment_protocol.md` — 实验协议规则
- `paper_target.md` — CCF-B 目标定位(需从 "GRAM 不换 backbone" 修订为 "GRAM setting 论文")
- `feedback_governance_shift.md` — 架构大改直接完整训练(cold-start method 属此类)
